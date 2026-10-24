"""SurrogateRunner: runs surrogate targets (e.g. ``lcbench``) in-process and paces them.

A surrogate target answers in milliseconds, but the run it stands for took minutes. If the
race saw the instant answer, LLM thinking time would be the only time in the race and every
"GPUs waiting" number would be meaningless. So this runner:

1. evaluates the config at launch (in-process, through the same ``execute_job`` the local
   and Modal workers use, so validation, errors and logs behave the same);
2. holds the slot busy for ``charged_s x runtime_multiplier`` real seconds, where
   ``charged_s`` is the run's own predicted training time (``metrics["train_time_s"]``);
   a run that failed with ``TimeoutError`` is charged its full ``budget_s``; any other
   failure is charged 0 s (it failed at config time);
3. replays the run's streamed metrics at their scaled times (a metric emitted with
   ``train_time_s = T`` at step k is delivered T x multiplier seconds after launch);
4. prices slots like real ones (default: the L4 list price, ``DEFAULT_PRICE_PER_SECOND``),
   so GPU $ in the race is what that training would have cost on an L4. **No compute is
   rented: these GPU $ are virtual (accounting only).**

``runtime_multiplier`` (default 1.0: one predicted second = one real second): set it on the
constructor or with ``BUSY_SURROGATE_RUNTIME_MULT``. It is printed into every ``run_ref``
(``surrogate:<id>:x<mult>``) so an export shows which pacing a race used. Below 1 the
runs are shorter than the real training they stand for, so LLM time weighs more; above 1
less. Both arms of a race get the same runner settings (the race engine checks slot
hardware and prices; the multiplier is a runner constant of the race).

Same Runner protocol as the others: N slots, launch() raises RunnerCapacityError when
full, wait() never raises for training failures, cancel() is idempotent. Pure Python: no
compute SDK import.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import time
from typing import Any

from backend.core.types import ExperimentConfig, ExperimentStatus, MetricPoint, RunHandle, RunResult, Slot
from workers.runners.base import (
    DEFAULT_PRICE_PER_SECOND,
    MetricCallback,
    RunnerCapacityError,
    RunnerError,
    emit_metric,
)
from workers.runners.local import LOG_TAIL_LINES, execute_job, import_modules, result_from_payload

ENV_MULTIPLIER = "BUSY_SURROGATE_RUNTIME_MULT"
RUNTIME_KEY = "train_time_s"
TIME_METRIC = "train_time_s"  # streamed metric that carries the predicted time of its step


def default_multiplier() -> float:
    raw = os.environ.get(ENV_MULTIPLIER)
    return 1.0 if not raw else float(raw)


class _SurrogateRun:
    def __init__(self, handle: RunHandle, payload: dict[str, Any], points: list[tuple[float, MetricPoint]],
                 due_at: float, slot_index: int) -> None:
        self.handle = handle
        self.payload = payload
        self.points = points
        self.due_at = due_at
        self.slot_index = slot_index
        self.task: asyncio.Task[None] | None = None
        self.cancelled = False
        self.result: RunResult | None = None


class SurrogateRunner:
    """In-process runner for surrogate targets, paced by predicted runtime x multiplier."""

    name = "surrogate"

    def __init__(
        self,
        max_slots: int = 16,
        *,
        hardware: str = "L4",
        price_per_second_usd: float | None = None,
        runtime_multiplier: float | None = None,
        preload: list[str] | None = None,
        extra_imports: list[str] | None = None,
    ) -> None:
        if max_slots < 1:
            raise ValueError("max_slots must be >= 1")
        mult = default_multiplier() if runtime_multiplier is None else float(runtime_multiplier)
        if not (mult >= 0.0):
            raise ValueError("runtime_multiplier must be >= 0")
        self.max_slots = max_slots
        self.hardware = hardware
        self._price = DEFAULT_PRICE_PER_SECOND[hardware] if price_per_second_usd is None else float(price_per_second_usd)
        self.runtime_multiplier = mult
        self.extra_imports = list(extra_imports or [])
        self._runs: dict[str, _SurrogateRun] = {}
        self._busy: dict[int, str] = {}
        import_modules(self.extra_imports)
        for t in preload or []:
            self.prewarm(t)

    # -- Runner protocol -------------------------------------------------------

    def slots(self) -> list[Slot]:
        return [Slot(slot_id=f"surrogate-{self.hardware.lower()}-{i}", index=i, hardware=self.hardware,
                     runner=self.name) for i in range(self.max_slots)]

    def busy_slots(self) -> int:
        return len(self._busy)

    def price_per_second(self, hardware: str) -> float:
        return self._price

    def prewarm(self, target: str) -> None:
        """Load the target's surrogate before the race clock starts (both arms alike)."""
        from workers.targets.base import get_target

        fn = getattr(get_target(target), "prewarm", None)
        if fn is not None:
            fn()

    async def launch(self, target: str, config: ExperimentConfig, seed: int, budget_s: int,
                     on_metric: MetricCallback) -> RunHandle:
        free = [i for i in range(self.max_slots) if i not in self._busy]
        if not free:
            raise RunnerCapacityError(f"all {self.max_slots} surrogate slots busy")
        slot = free[0]
        handle = RunHandle(runner=self.name, target=target, config=dict(config), seed=int(seed),
                           budget_s=int(budget_s), hardware=self.hardware)
        handle.run_ref = f"surrogate:{handle.run_id[:12]}:x{self.runtime_multiplier:g}"
        raw: list[tuple[int, str, float]] = []
        job = {"target": target, "config": dict(config), "seed": int(seed), "budget_s": int(budget_s),
               "started_at": handle.started_at, "extra_imports": self.extra_imports}
        # In-process: keep the target's prints out of the host console (they still reach logs_tail).
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            payload = execute_job(job, lambda step, _t, name, value: raw.append((step, name, value)))
        charged = self._charged_s(payload, int(budget_s))
        m = self.runtime_multiplier
        points: list[tuple[float, MetricPoint]] = []
        step_t: dict[int, float] = {s: v for s, n, v in raw if n == TIME_METRIC}
        for step, name, value in raw:
            t_rel = min(step_t.get(step, charged), charged) * m
            points.append((t_rel, MetricPoint(experiment_id=handle.run_id, step=step, t=t_rel, name=name,
                                              value=value)))
        points.sort(key=lambda p: p[0])
        run = _SurrogateRun(handle, payload, points, handle.started_at + charged * m, slot)
        self._runs[handle.run_id] = run
        self._busy[slot] = handle.run_id
        run.task = asyncio.create_task(self._pace(run, on_metric), name=f"surrogate-{handle.run_id[:8]}")
        return handle

    async def wait(self, handle: RunHandle) -> RunResult:
        run = self._runs.get(handle.run_id)
        if run is None:
            raise RunnerError(f"unknown run {handle.run_id}")
        if run.result is not None:
            return run.result
        assert run.task is not None
        try:
            await asyncio.shield(run.task)
        except asyncio.CancelledError:
            if not run.cancelled:
                raise  # the caller's wait() was cancelled, not the run
        if run.cancelled:
            result = result_from_payload(handle, run.payload, status=ExperimentStatus.cancelled, error="cancelled",
                                         ended_at=time.time())
        else:
            result = result_from_payload(handle, run.payload, ended_at=time.time())
        run.result = result
        self._release(run)
        return result

    async def cancel(self, handle: RunHandle) -> None:
        run = self._runs.get(handle.run_id)
        if run is None or run.result is not None:
            return
        run.cancelled = True
        if run.task is not None and not run.task.done():
            run.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await run.task

    # -- internals ---------------------------------------------------------------

    @staticmethod
    def _charged_s(payload: dict[str, Any], budget_s: int) -> float:
        if payload.get("ok"):
            v = (payload.get("result") or {}).get("metrics", {}).get(RUNTIME_KEY)
            if v is None:
                raise RunnerError(f"surrogate target returned no {RUNTIME_KEY!r} metric; cannot pace the run")
            return max(0.0, float(v))
        if str(payload.get("error", "")).startswith("TimeoutError"):
            return float(budget_s)
        return 0.0

    async def _pace(self, run: _SurrogateRun, on_metric: MetricCallback) -> None:
        start = run.handle.started_at
        for t_rel, point in run.points:
            delay = start + t_rel - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                await emit_metric(on_metric, point)
            except Exception as exc:  # noqa: BLE001 (metric loss must not kill the run)
                tail = run.payload.setdefault("logs_tail", [])
                tail.append(f"[runner] on_metric failed: {exc!r}")
                del tail[:-LOG_TAIL_LINES]
        delay = run.due_at - time.time()
        if delay > 0:
            await asyncio.sleep(delay)

    def _release(self, run: _SurrogateRun) -> None:
        if self._busy.get(run.slot_index) == run.handle.run_id:
            del self._busy[run.slot_index]


__all__ = ["ENV_MULTIPLIER", "SurrogateRunner", "default_multiplier"]
