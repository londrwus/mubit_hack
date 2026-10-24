"""A TPE study in its own process: GIL isolation for the CPU-bound proposers.

Why (review finding, R67): TPE's ``ask`` is CPU-bound Python + numpy and grows with the study
(about 0.15 s per config at 600 trials on the rl-wide space). In one interpreter the backfill
sampler of a Busy Machines arm, the Optuna arm's own study and the shared event loop (run ends,
slot refills, the timing of every arm's LLM replies) take turns on one GIL, so one arm's asks
slow the others down: the Optuna arm's ``ask_batch(8)`` measured 1.6x slower with a sampler
thread asking next to it. With ``isolation="process"`` each study lives in a child process with
its own interpreter and the event loop only sends operations and receives configs.

Same behaviour, only the process differs: the child builds the same study (``TPESampler`` with
the same seed, ``n_startup_trials`` and ``constant_liar=True``, same direction) and the parent
sends the queued operations (tells, added trials) with each ask, in exactly the order the
in-process classes apply them (``BackfillSampler._drain`` before its asks; the Optuna arm's
``observe`` calls between its asks). Same seed + same sequence of operations -> the same
configs (tests/test_review_proposer_isolation.py).

Protocol (pickled tuples over a ``multiprocessing`` pipe, one request at a time):
    ("run", ops, k) -> ("ok", {"configs": [...], "numbers": [...], "added": n, "add_skipped": n,
                               "open": n})  |  ("err", "Type: message")
    ("close",)      -> ("ok", None) and the child exits
ops: ("tell", number, value) | ("tell_state", number, "fail" | "pruned")
     | ("add", params, value, guarded)   guarded: a ValueError from Optuna is counted, not raised
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import threading
import time
import weakref
from typing import Any, Literal

Isolation = Literal["thread", "process"]
ISOLATION_ENV = "BUSY_PROPOSER_ISOLATION"
READY_TIMEOUT_S = 120.0   # child start: spawn + import optuna (a few seconds on the laptop)
CLOSE_TIMEOUT_S = 5.0


def proposer_isolation(mode: str | None) -> Isolation:
    """Where a race's CPU-bound proposers (the backfill sampler, the Optuna arm's study) run.
    Benchmark races: each in its own process. Dev / mock / dry-run races: a worker thread, as
    before. ``BUSY_PROPOSER_ISOLATION=thread|process`` overrides both (recorded in the manifest)."""
    v = os.environ.get(ISOLATION_ENV, "").strip().lower()
    if v in ("thread", "process"):
        return v  # type: ignore[return-value]
    return "process" if mode == "benchmark" else "thread"


# --------------------------------------------------------------------------- child


def _serve(conn: Any, spec: Any, seed: int, n_startup_trials: int, direction: str) -> None:
    import optuna
    from optuna.trial import TrialState

    from backend.agent.optuna_arm import distributions

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials, constant_liar=True)
    study = optuna.create_study(direction=direction, sampler=sampler)
    dists = distributions(spec)
    trials: dict[int, Any] = {}
    conn.send(("ready", os.getpid()))
    states = {"fail": TrialState.FAIL, "pruned": TrialState.PRUNED}
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg[0] == "close":
            conn.send(("ok", None))
            return
        try:
            _, ops, k = msg
            added = add_skipped = 0
            for op in ops:
                if op[0] == "tell":
                    study.tell(trials.pop(op[1]), op[2])
                elif op[0] == "tell_state":
                    study.tell(trials.pop(op[1]), state=states[op[2]])
                elif op[0] == "add":
                    _, params, value, guarded = op
                    keep = {n: params[n] for n in dists if n in params}
                    try:
                        study.add_trial(optuna.trial.create_trial(
                            params=keep, distributions={n: dists[n] for n in keep}, value=value))
                    except ValueError:
                        if not guarded:
                            raise
                        add_skipped += 1
                        continue
                    added += 1
                else:
                    raise ValueError(f"unknown op {op[0]!r}")
            configs, numbers = [], []
            for _ in range(max(0, int(k))):
                trial = study.ask()
                configs.append(spec.validate_config(spec.suggest_optuna(trial)))
                trials[trial.number] = trial
                numbers.append(trial.number)
            conn.send(("ok", {"configs": configs, "numbers": numbers, "added": added,
                              "add_skipped": add_skipped, "open": len(trials)}))
        except Exception as e:  # noqa: BLE001 - reported to the parent, which raises it there
            conn.send(("err", f"{type(e).__name__}: {e}"))


# --------------------------------------------------------------------------- parent


def _shutdown(proc: Any, conn: Any) -> None:
    try:
        if proc.is_alive():
            try:
                conn.send(("close",))
                if conn.poll(CLOSE_TIMEOUT_S):
                    conn.recv()
            except (OSError, EOFError, BrokenPipeError):
                pass
            proc.join(CLOSE_TIMEOUT_S)
        if proc.is_alive():
            proc.terminate()
            proc.join(CLOSE_TIMEOUT_S)
    finally:
        try:
            conn.close()
        except OSError:
            pass


class StudyWorker:
    """Parent-side handle of one child-process TPE study. ``run`` is blocking (call it from a
    worker thread, e.g. ``asyncio.to_thread``: waiting on the pipe releases the GIL); one request
    at a time (a lock). The child starts on ``start()`` or the first ``run``."""

    def __init__(self, spec: Any, seed: int, n_startup_trials: int, direction: str) -> None:
        self._args = (spec, int(seed), int(n_startup_trials), direction)
        self._lock = threading.Lock()
        self._proc: Any = None
        self._conn: Any = None
        self._finalizer: Any = None
        self.pid: int | None = None
        self.start_s: float | None = None
        self.closed = False

    def start(self) -> None:
        with self._lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._proc is not None:
            return
        if self.closed:
            raise RuntimeError("study worker closed")
        t = time.perf_counter()
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_serve, args=(child, *self._args), daemon=True, name="tpe-study")
        proc.start()
        child.close()
        if not parent.poll(READY_TIMEOUT_S):
            _shutdown(proc, parent)
            raise RuntimeError(f"study worker did not start within {READY_TIMEOUT_S:g}s")
        tag, pid = parent.recv()
        if tag != "ready":
            _shutdown(proc, parent)
            raise RuntimeError(f"study worker failed to start: {pid}")
        self._proc, self._conn, self.pid = proc, parent, int(pid)
        self.start_s = time.perf_counter() - t
        self._finalizer = weakref.finalize(self, _shutdown, proc, parent)

    def run(self, ops: list[tuple[Any, ...]], k: int) -> dict[str, Any]:
        with self._lock:
            self._start_locked()
            try:
                self._conn.send(("run", list(ops), int(k)))
                tag, payload = self._conn.recv()
            except (EOFError, OSError, BrokenPipeError) as e:
                raise RuntimeError(f"study worker died: {type(e).__name__}: {e}") from e
        if tag != "ok":
            raise RuntimeError(f"study worker: {payload}")
        return payload

    def close(self) -> None:
        with self._lock:
            self.closed = True
            if self._finalizer is not None:
                self._finalizer()   # runs _shutdown once
            self._proc = self._conn = None


def finite_or_none(value: float | None) -> float | None:
    return float(value) if value is not None and math.isfinite(float(value)) else None
