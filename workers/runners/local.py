"""LocalRunner: one subprocess per run on this machine. $0, small slot count, CPU only.

For dev and tests (mock LLM, tiny CPU runs). There are no local GPUs (CLAUDE.md); every
GPU run goes through ModalRunner.

The child is ``python -m workers.runners.local`` reading one JSON job on stdin. It writes
protocol lines prefixed with ``@@BM@@ `` to stdout (metrics as they happen, then one
result); everything else it prints (target logs, tracebacks) is ordinary log output and
feeds ``RunResult.logs_tail``.

This module also holds the compute-agnostic job helpers that ModalRunner reuses:
``execute_job`` (runs inside the worker, never raises) and ``result_from_payload``
(builds the RunResult on the host). No compute SDK is imported here.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import importlib
import io
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from backend.core.types import (
    ExperimentConfig,
    ExperimentStatus,
    MetricPoint,
    RunHandle,
    RunResult,
    Slot,
)
from workers.runners.base import (
    DEFAULT_GRACE_S,
    MetricCallback,
    RunnerCapacityError,
    RunnerError,
    emit_metric,
)

PROTO_PREFIX = "@@BM@@ "
LOG_TAIL_LINES = 50
REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- shared job helpers


class _TailBuffer(io.TextIOBase):
    """A text stream that tees writes to ``inner`` and keeps the last N complete lines."""

    def __init__(self, inner: Any, tail: deque[str]) -> None:
        self._inner = inner
        self._tail = tail
        self._partial = ""

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        if self._inner is not None:
            try:
                self._inner.write(s)
            except Exception:  # noqa: BLE001 (a broken console must not kill the run)
                pass
        buf = self._partial + s
        *lines, self._partial = buf.split("\n")
        for ln in lines:
            if not ln.startswith(PROTO_PREFIX):
                self._tail.append(ln)
        return len(s)

    def flush(self) -> None:
        if self._inner is not None:
            with contextlib.suppress(Exception):
                self._inner.flush()

    def close_partial(self) -> None:
        if self._partial and not self._partial.startswith(PROTO_PREFIX):
            self._tail.append(self._partial)
        self._partial = ""


@contextlib.contextmanager
def capture_logs(tail: deque[str]) -> Iterator[None]:
    """Tee sys.stdout / sys.stderr into ``tail`` (last LOG_TAIL_LINES lines)."""
    out, err = sys.stdout, sys.stderr
    t_out, t_err = _TailBuffer(out, tail), _TailBuffer(err, tail)
    sys.stdout, sys.stderr = t_out, t_err  # type: ignore[assignment]
    try:
        yield
    finally:
        t_out.close_partial()
        t_err.close_partial()
        sys.stdout, sys.stderr = out, err


def import_modules(modules: list[str] | None) -> None:
    for m in modules or []:
        importlib.import_module(m)


def execute_job(
    job: dict[str, Any],
    emit: Callable[[int, float, str, float], None],
) -> dict[str, Any]:
    """Run one job in this process. Never raises.

    ``job``: {target, config, seed, budget_s, started_at, extra_imports?}.
    ``emit(step, t, name, value)`` receives each streamed metric (t = s since started_at).
    Returns a JSON-able payload: {ok, result | error, logs_tail, worker_started_at, ended_at}.
    """
    from workers.targets.base import get_target

    tail: deque[str] = deque(maxlen=LOG_TAIL_LINES)
    worker_started_at = time.time()
    started_at = float(job.get("started_at") or worker_started_at)
    payload: dict[str, Any] = {"worker_started_at": worker_started_at}

    def _emit(step: int, name: str, value: float) -> None:
        try:
            emit(int(step), time.time() - started_at, str(name), float(value))
        except Exception as exc:  # noqa: BLE001 (metric loss must not kill training)
            print(f"[runner] metric emit failed: {exc!r}", file=sys.stderr)

    with capture_logs(tail):
        try:
            import_modules(job.get("extra_imports"))
            target = get_target(job["target"])
            config = target.spec.validate_config(dict(job["config"]))
            out = target.run(config, int(job["seed"]), int(job["budget_s"]), _emit)
            if not isinstance(out, dict) or "score" not in out:
                raise TypeError("target.run must return a dict with a 'score' key")
            score = float(out["score"])
            if not math.isfinite(score):
                raise FloatingPointError(f"non-finite score: {score}")
            metrics: dict[str, float] = {}
            for k, v in out.items():
                if k == "score":
                    continue
                if isinstance(v, bool):
                    metrics[k] = 1.0 if v else 0.0
                elif isinstance(v, (int, float)):
                    metrics[k] = float(v)
            payload.update(ok=True, result={"score": score, "metrics": metrics})
        except Exception as exc:  # noqa: BLE001 (turned into status=failed)
            traceback.print_exc()
            payload.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    payload["logs_tail"] = list(tail)
    payload["ended_at"] = time.time()
    return payload


def result_from_payload(
    handle: RunHandle,
    payload: dict[str, Any] | None,
    *,
    status: ExperimentStatus | None = None,
    error: str | None = None,
    logs_tail: list[str] | None = None,
    ended_at: float | None = None,
    is_trap: bool | None = None,
) -> RunResult:
    """Build the host-side RunResult. ``status``/``error`` override the payload (used for
    timeouts and cancellations, where there may be no payload)."""
    p = payload or {}
    tail = list(logs_tail if logs_tail is not None else p.get("logs_tail") or [])[-LOG_TAIL_LINES:]
    if is_trap is None:
        is_trap = _is_trap(handle.target, handle.config)
    end = ended_at if ended_at is not None else time.time()
    if status is None and p.get("ok"):
        res = p["result"]
        return RunResult(
            score=float(res["score"]),
            status=ExperimentStatus.succeeded,
            metrics={k: float(v) for k, v in (res.get("metrics") or {}).items()},
            logs_tail=tail,
            is_trap=is_trap,
            run_ref=handle.run_ref,
            started_at=handle.started_at,
            ended_at=end,
            hardware=handle.hardware,
        )
    st = status or ExperimentStatus.failed
    return RunResult(
        score=None,
        status=st,
        metrics={},
        logs_tail=tail,
        is_trap=is_trap,
        run_ref=handle.run_ref,
        started_at=handle.started_at,
        ended_at=end,
        hardware=handle.hardware,
        error=error or p.get("error") or st.value,
    )


def _is_trap(target: str, config: ExperimentConfig) -> bool:
    try:
        from workers.targets.base import get_target

        return bool(get_target(target).spec.is_trap(config))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- runner


class _LocalRun:
    def __init__(self, handle: RunHandle, proc: subprocess.Popen[bytes], on_metric: MetricCallback,
                 loop: asyncio.AbstractEventLoop) -> None:
        self.handle = handle
        self.proc = proc
        self.on_metric = on_metric
        self.loop = loop
        self.tail: deque[str] = deque(maxlen=LOG_TAIL_LINES)
        self.payload: dict[str, Any] | None = None
        self.cancelled = False
        self.done = asyncio.Event()  # set on the loop when the child's stdout hits EOF
        self.pending: list[concurrent.futures.Future[None]] = []  # metric callbacks in flight
        self.result: RunResult | None = None
        self.slot_index: int = 0
        self.reader: threading.Thread | None = None


class LocalRunner:
    """Subprocess-per-run runner on this machine (CPU, $0).

    Works on any asyncio loop (Proactor or Selector): the child is a ``subprocess.Popen``
    and a reader thread forwards its output onto the loop.

    ``max_slots`` caps concurrency; launch() raises RunnerCapacityError beyond it. A slot is
    freed by wait() (after cancel(), too).
    ``extra_imports`` / ``extra_sys_path`` let a child import extra target modules (tests).
    ``env`` adds environment variables to children (e.g. BUSY_CIFAR_SYNTHETIC=1).
    """

    name = "local"

    def __init__(
        self,
        max_slots: int = 2,
        *,
        hardware: str = "CPU",
        price_per_second_usd: float = 0.0,
        grace_s: int = DEFAULT_GRACE_S,
        env: dict[str, str] | None = None,
        extra_imports: list[str] | None = None,
        extra_sys_path: list[str] | None = None,
        python: str | None = None,
    ) -> None:
        if max_slots < 1:
            raise ValueError("max_slots must be >= 1")
        self.max_slots = max_slots
        self.hardware = hardware
        self._price = price_per_second_usd
        self.grace_s = grace_s
        self.env = dict(env or {})
        self.extra_imports = list(extra_imports or [])
        self.extra_sys_path = [str(p) for p in (extra_sys_path or [])]
        self.python = python or sys.executable
        self._runs: dict[str, _LocalRun] = {}
        self._busy: dict[int, str] = {}  # slot index -> run_id

    # -- Runner protocol -------------------------------------------------------

    def slots(self) -> list[Slot]:
        return [
            Slot(slot_id=f"local-{self.hardware.lower()}-{i}", index=i, hardware=self.hardware, runner=self.name)
            for i in range(self.max_slots)
        ]

    def busy_slots(self) -> int:
        return len(self._busy)

    def price_per_second(self, hardware: str) -> float:
        return self._price

    async def launch(
        self,
        target: str,
        config: ExperimentConfig,
        seed: int,
        budget_s: int,
        on_metric: MetricCallback,
    ) -> RunHandle:
        free = [i for i in range(self.max_slots) if i not in self._busy]
        if not free:
            raise RunnerCapacityError(f"all {self.max_slots} local slots busy")
        slot = free[0]
        handle = RunHandle(
            runner=self.name, target=target, config=dict(config), seed=int(seed),
            budget_s=int(budget_s), hardware=self.hardware,
        )
        job = {
            "target": target, "config": dict(config), "seed": int(seed), "budget_s": int(budget_s),
            "started_at": handle.started_at, "extra_imports": self.extra_imports,
        }
        env = os.environ.copy()
        env.update(self.env)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        paths = [str(REPO_ROOT), *self.extra_sys_path]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        try:
            proc = subprocess.Popen(  # noqa: S603 (fixed argv: this interpreter + our module)
                [self.python, "-m", "workers.runners.local"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT), env=env,
            )
        except OSError as exc:
            raise RunnerError(f"could not start local worker: {exc}") from exc
        handle.run_ref = f"pid:{proc.pid}"
        run = _LocalRun(handle, proc, on_metric, asyncio.get_running_loop())
        run.slot_index = slot
        self._runs[handle.run_id] = run
        self._busy[slot] = handle.run_id
        assert proc.stdin is not None
        try:
            proc.stdin.write(json.dumps(job).encode() + b"\n")
            proc.stdin.close()
        except OSError:
            pass  # the child died at once; wait() reports it
        run.reader = threading.Thread(target=self._read, args=(run,), daemon=True)
        run.reader.start()
        return handle

    async def wait(self, handle: RunHandle) -> RunResult:
        run = self._runs.get(handle.run_id)
        if run is None:
            raise RunnerError(f"unknown run {handle.run_id}")
        if run.result is not None:
            return run.result
        timeout = handle.budget_s + self.grace_s
        remaining = max(0.0, handle.started_at + timeout - time.time())
        status: ExperimentStatus | None = None
        error: str | None = None
        try:
            await asyncio.wait_for(run.done.wait(), timeout=remaining)
        except TimeoutError:
            if not run.cancelled:
                status = ExperimentStatus.timed_out
                error = f"exceeded {timeout}s (budget {handle.budget_s}s + grace {self.grace_s}s)"
            await self._kill(run)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(run.done.wait(), timeout=10)
        returncode = await asyncio.to_thread(self._reap, run)
        if run.pending:
            await asyncio.gather(*(asyncio.wrap_future(f) for f in run.pending), return_exceptions=True)
        if run.cancelled:
            status, error = ExperimentStatus.cancelled, "cancelled"
        elif status is None and run.payload is None:
            status = ExperimentStatus.failed
            error = f"worker exited with code {returncode} without a result"
        result = result_from_payload(handle, run.payload, status=status, error=error, logs_tail=list(run.tail))
        run.result = result
        self._release(run)
        return result

    async def cancel(self, handle: RunHandle) -> None:
        run = self._runs.get(handle.run_id)
        if run is None or run.result is not None:
            return
        run.cancelled = True
        await self._kill(run)

    # -- internals ---------------------------------------------------------------

    async def _kill(self, run: _LocalRun) -> None:
        if run.proc.poll() is None:
            with contextlib.suppress(OSError):
                run.proc.kill()
            await asyncio.to_thread(self._reap, run)

    @staticmethod
    def _reap(run: _LocalRun) -> int | None:
        try:
            return run.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            return None

    def _release(self, run: _LocalRun) -> None:
        if self._busy.get(run.slot_index) == run.handle.run_id:
            del self._busy[run.slot_index]

    def _read(self, run: _LocalRun) -> None:
        """Reader thread: parse the child's stdout and hand metrics to the loop."""
        assert run.proc.stdout is not None
        try:
            for raw in iter(run.proc.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line.startswith(PROTO_PREFIX):
                    run.tail.append(line)
                    continue
                try:
                    msg = json.loads(line[len(PROTO_PREFIX):])
                except json.JSONDecodeError:
                    run.tail.append(line)
                    continue
                if msg.get("type") == "metric":
                    point = MetricPoint(
                        experiment_id=run.handle.run_id, step=int(msg["step"]), t=float(msg["t"]),
                        name=str(msg["name"]), value=float(msg["value"]),
                    )
                    with contextlib.suppress(RuntimeError):  # loop closed
                        run.pending.append(asyncio.run_coroutine_threadsafe(self._deliver(run, point), run.loop))
                elif msg.get("type") == "result":
                    run.payload = msg["payload"]
        finally:
            with contextlib.suppress(RuntimeError):
                run.loop.call_soon_threadsafe(run.done.set)

    @staticmethod
    async def _deliver(run: _LocalRun, point: MetricPoint) -> None:
        try:
            await emit_metric(run.on_metric, point)
        except Exception as exc:  # noqa: BLE001
            run.tail.append(f"[runner] on_metric failed: {exc!r}")


# --------------------------------------------------------------------------- child entry point


def _child_main() -> int:
    job = json.loads(sys.stdin.readline())
    proto = sys.stdout

    def send(obj: dict[str, Any]) -> None:
        proto.write(PROTO_PREFIX + json.dumps(obj) + "\n")
        proto.flush()

    def emit(step: int, t: float, name: str, value: float) -> None:
        send({"type": "metric", "step": step, "t": t, "name": name, "value": value})

    payload = execute_job(job, emit)
    # logs_tail is rebuilt by the parent from the raw stream; drop it to keep the line short.
    payload["logs_tail"] = []
    send({"type": "result", "payload": payload})
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_child_main())
