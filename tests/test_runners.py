"""Tests for workers/runners/local.py and workers/runners/modal_runner.py.

LocalRunner tests spawn real subprocesses running tiny dummy targets. ModalRunner tests use
in-process fakes for the Modal Function / FunctionCall / Queue objects: no network, no $.
"""

from __future__ import annotations

import asyncio
import importlib
import queue as stdlib_queue
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from backend.core.types import ExperimentStatus, MetricPoint
from workers.runners import LocalRunner, RunnerCapacityError, get_runner
from workers.runners.base import Runner
from workers.runners.local import execute_job

DUMMY_MODULE = "bm_dummy_targets"
DUMMY_SRC = textwrap.dedent(
    '''
    import time
    from workers.targets.base import BaseTarget, ParamSpec, TargetSpec, register_target

    def _spec(name, hardware="CPU"):
        return TargetSpec(
            name=name, description="dummy", metric="acc", direction="max", target_value=0.9,
            hardware=hardware, budget_s=5,
            config_space={
                "x": ParamSpec(type="float", low=0.0, high=1.0, default=0.5),
                "sleep": ParamSpec(type="float", low=0.0, high=120.0, default=0.0),
                "leak": ParamSpec(type="bool", default=False),
            },
            trap_option="leak",
        )

    class Ok(BaseTarget):
        spec = _spec("dummy_ok")
        def run(self, config, seed, budget_s, emit):
            print("dummy starting", flush=True)
            for i in range(3):
                emit(i, "loss", 1.0 / (i + 1))
            time.sleep(config["sleep"])
            print("dummy done", flush=True)
            return {"score": config["x"] + seed * 0.0, "steps": 3, "flag": True, "note": "ignored"}

    class Fail(BaseTarget):
        spec = _spec("dummy_fail")
        def run(self, config, seed, budget_s, emit):
            emit(0, "loss", float("nan"))
            raise FloatingPointError("loss diverged")

    class GpuOk(Ok):
        spec = _spec("dummy_gpu", hardware="L4")

    register_target(Ok())
    register_target(Fail())
    register_target(GpuOk())
    '''
)


@pytest.fixture(scope="module")
def dummy_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("dummy_targets")
    (d / f"{DUMMY_MODULE}.py").write_text(DUMMY_SRC, encoding="utf-8")
    sys.path.insert(0, str(d))
    importlib.import_module(DUMMY_MODULE)
    yield d
    sys.path.remove(str(d))


def local_runner(dummy_dir: Path, **kw: Any) -> LocalRunner:
    return LocalRunner(extra_imports=[DUMMY_MODULE], extra_sys_path=[str(dummy_dir)], **kw)


# --------------------------------------------------------------------------- execute_job


def test_execute_job_success_and_failure(dummy_dir: Path) -> None:
    got: list[tuple] = []
    p = execute_job({"target": "dummy_ok", "config": {"x": 0.7}, "seed": 1, "budget_s": 5},
                    lambda *a: got.append(a))
    assert p["ok"] and p["result"]["score"] == pytest.approx(0.7)
    assert p["result"]["metrics"] == {"steps": 3.0, "flag": 1.0}
    assert len(got) == 3 and got[0][2] == "loss"
    assert any("dummy done" in ln for ln in p["logs_tail"])

    p = execute_job({"target": "dummy_fail", "config": {}, "seed": 0, "budget_s": 5}, lambda *a: None)
    assert not p["ok"] and "FloatingPointError" in p["error"]
    assert any("Traceback" in ln for ln in p["logs_tail"])

    p = execute_job({"target": "dummy_ok", "config": {"x": 9.0}, "seed": 0, "budget_s": 5}, lambda *a: None)
    assert not p["ok"] and "outside" in p["error"]


# --------------------------------------------------------------------------- LocalRunner


async def test_local_success_streams_metrics(dummy_dir: Path) -> None:
    r = local_runner(dummy_dir)
    assert isinstance(r, Runner)
    assert [s.slot_id for s in r.slots()] == ["local-cpu-0", "local-cpu-1"]
    assert r.price_per_second("CPU") == 0.0
    pts: list[MetricPoint] = []

    async def on_metric(p: MetricPoint) -> None:
        pts.append(p)

    h = await r.launch("dummy_ok", {"x": 0.25}, 0, 5, on_metric)
    assert h.run_ref and h.run_ref.startswith("pid:")
    res = await r.wait(h)
    assert res.status == ExperimentStatus.succeeded, res.logs_tail
    assert res.score == pytest.approx(0.25)
    assert res.metrics["steps"] == 3.0
    assert [p.step for p in pts] == [0, 1, 2]
    assert all(p.experiment_id == h.run_id and p.t >= 0 for p in pts)
    assert any("dummy done" in ln for ln in res.logs_tail)
    assert res.is_trap is False and res.duration_s > 0
    assert await r.wait(h) is res  # idempotent
    assert r.busy_slots() == 0


async def test_local_failure_and_trap_flag(dummy_dir: Path) -> None:
    r = local_runner(dummy_dir)
    h = await r.launch("dummy_fail", {"leak": True}, 0, 5, lambda p: None)
    res = await r.wait(h)
    assert res.status == ExperimentStatus.failed and res.score is None
    assert "loss diverged" in (res.error or "")
    assert any("Traceback" in ln for ln in res.logs_tail)
    assert res.is_trap is True
    assert len(res.logs_tail) <= 50


async def test_local_capacity_timeout_cancel(dummy_dir: Path) -> None:
    r = local_runner(dummy_dir, max_slots=2, grace_s=1)
    h1 = await r.launch("dummy_ok", {"sleep": 60.0}, 0, 1, lambda p: None)
    h2 = await r.launch("dummy_ok", {"sleep": 60.0}, 0, 30, lambda p: None)
    with pytest.raises(RunnerCapacityError):
        await r.launch("dummy_ok", {}, 0, 5, lambda p: None)
    t0 = time.time()
    res1 = await r.wait(h1)
    assert res1.status == ExperimentStatus.timed_out
    assert time.time() - t0 < 20
    await r.cancel(h2)
    await r.cancel(h2)  # idempotent
    res2 = await r.wait(h2)
    assert res2.status == ExperimentStatus.cancelled
    assert r.busy_slots() == 0
    h3 = await r.launch("dummy_ok", {}, 0, 5, lambda p: None)  # slots freed
    assert (await r.wait(h3)).status == ExperimentStatus.succeeded


async def test_local_cifar_synthetic_end_to_end() -> None:
    pytest.importorskip("torch")
    from workers.targets.base import get_target

    r = LocalRunner(env={"BUSY_CIFAR_SYNTHETIC": "1", "BUSY_CIFAR_SUBSET": "128", "BUSY_CIFAR_DEVICE": "cpu"},
                    grace_s=180)
    cfg = get_target("cifar").spec.default_config()
    cfg.update(epochs=1.0, batch_size=256, width=0.5, tta=0)
    pts: list[MetricPoint] = []
    h = await r.launch("cifar", cfg, 0, 60, pts.append)
    res = await r.wait(h)
    assert res.status == ExperimentStatus.succeeded, res.logs_tail
    assert 0.0 <= (res.score or 0.0) <= 1.0
    assert "heldout_score" in res.metrics and "reported_score" in res.metrics
    assert {"train_loss", "val_acc", "test_acc"} <= {p.name for p in pts}


def test_get_runner() -> None:
    assert isinstance(get_runner("local", max_slots=1), LocalRunner)
    with pytest.raises(ValueError):
        get_runner("slurm")


# --------------------------------------------------------------------------- ModalRunner (fakes)

modal_runner = pytest.importorskip("workers.runners.modal_runner")
ModalRunner = modal_runner.ModalRunner


class _Aio:
    def __init__(self, fn: Any) -> None:
        self.aio = fn


class FakeQueue:
    def __init__(self) -> None:
        self.parts: dict[str, list[Any]] = {}
        self.lock = threading.Lock()
        self.get_many = _Aio(self._get_many)
        self.clear = _Aio(self._clear)

    def put_many(self, items: list[Any], partition: str | None = None, **_: Any) -> None:
        with self.lock:
            self.parts.setdefault(partition or "", []).extend(items)

    async def _get_many(self, n: int, block: bool = True, timeout: float | None = None,
                        partition: str | None = None) -> list[Any]:
        deadline = time.time() + (timeout or 0)
        while True:
            with self.lock:
                buf = self.parts.get(partition or "", [])
                if buf:
                    out, self.parts[partition or ""] = buf[:n], buf[n:]
                    return out
            if not block:
                return []
            if time.time() >= deadline:
                raise stdlib_queue.Empty
            await asyncio.sleep(0.02)

    async def _clear(self, partition: str | None = None, **_: Any) -> None:
        with self.lock:
            self.parts.pop(partition or "", None)


class FakeCall:
    """Simulates a container: runs execute_job in a thread, pushing to the fake queue."""

    _n = 0

    def __init__(self, job: dict[str, Any], q: FakeQueue, startup_s: float = 0.05) -> None:
        FakeCall._n += 1
        self.object_id = f"fc-fake{FakeCall._n}"
        self.job = job
        self.done = threading.Event()
        self.payload: Any = None
        self.cancelled = False
        self.get = _Aio(self._get)
        self.cancel = _Aio(self._cancel)
        self.cancel_kwargs: dict[str, Any] = {}

        def body() -> None:
            time.sleep(startup_s)
            started = time.time()
            q.put_many([{"type": "started", "ts": started}], partition=job["run_id"])

            def emit(step: int, t: float, name: str, value: float) -> None:
                q.put_many([{"type": "metric", "step": step, "t": t, "name": name, "value": value}],
                           partition=job["run_id"])

            p = execute_job(job, emit)
            p["container_started_at"] = started
            self.payload = p
            self.done.set()

        threading.Thread(target=body, daemon=True).start()

    async def _get(self, timeout: float | None = None) -> Any:
        deadline = time.time() + (timeout or 0)
        while not self.done.is_set():
            if self.cancelled:
                raise RuntimeError("InputCancellation")
            if time.time() >= deadline:
                raise TimeoutError()
            await asyncio.sleep(0.02)
        return self.payload

    async def _cancel(self, terminate_containers: bool = False) -> None:
        self.cancelled = True
        self.cancel_kwargs = {"terminate_containers": terminate_containers}


class FakeFunction:
    def __init__(self, name: str, q: FakeQueue) -> None:
        self.name = name
        self.q = q
        self.calls: list[FakeCall] = []
        self.autoscaler: list[dict[str, Any]] = []
        self.spawn = _Aio(self._spawn)
        self.update_autoscaler = _Aio(self._update)

    async def _spawn(self, job: dict[str, Any]) -> FakeCall:
        c = FakeCall(job, self.q)
        self.calls.append(c)
        return c

    async def _update(self, **kw: Any) -> None:
        self.autoscaler.append(kw)


@pytest.fixture
def fakes(dummy_dir: Path):
    ModalRunner._global_busy.clear()
    q = FakeQueue()
    fns: dict[str, FakeFunction] = {}

    def resolver(name: str) -> FakeFunction:
        return fns.setdefault(name, FakeFunction(name, q))

    yield q, fns, resolver
    ModalRunner._global_busy.clear()


def make_modal(resolver: Any, q: FakeQueue, **kw: Any) -> Any:
    kw.setdefault("slots", {"L4": 2, "CPU": 2})
    return ModalRunner(function_resolver=resolver, queue=q, poll_interval_s=0.05, **kw)


async def test_modal_success_metrics_run_ref(fakes) -> None:
    q, fns, resolver = fakes
    r = make_modal(resolver, q)
    assert isinstance(r, Runner)
    assert len(r.slots()) == 4 and {s.hardware for s in r.slots()} == {"L4", "CPU"}
    pts: list[MetricPoint] = []
    h = await r.launch("dummy_gpu", {"x": 0.4}, 0, 5, pts.append)
    assert h.hardware == "L4" and h.run_ref and h.run_ref.startswith("fc-")
    assert fns["run_l4"].calls[0].job["run_id"] == h.run_id
    res = await r.wait(h)
    assert res.status == ExperimentStatus.succeeded, res
    assert res.score == pytest.approx(0.4) and res.run_ref == h.run_ref
    assert [p.step for p in pts] == [0, 1, 2] and all(p.experiment_id == h.run_id for p in pts)
    assert res.metrics["modal_startup_s"] >= 0.0
    assert any("dummy done" in ln for ln in res.logs_tail)
    await r.drain_background()
    assert h.run_id not in q.parts  # partition cleared (in the background, after wait)
    assert r.busy_slots() == 0

    h2 = await r.launch("dummy_ok", {}, 0, 5, lambda p: None)  # CPU target -> run_cpu
    assert h2.hardware == "CPU" and fns["run_cpu"].calls
    assert (await r.wait(h2)).status == ExperimentStatus.succeeded


async def test_modal_failure_and_invalid_config(fakes) -> None:
    q, fns, resolver = fakes
    r = make_modal(resolver, q)
    h = await r.launch("dummy_fail", {}, 0, 5, lambda p: None)
    res = await r.wait(h)
    assert res.status == ExperimentStatus.failed and "loss diverged" in (res.error or "")
    n_calls = len(fns.get("run_cpu", FakeFunction("x", q)).calls)
    h = await r.launch("dummy_ok", {"x": 5.0}, 0, 5, lambda p: None)
    res = await r.wait(h)
    assert res.status == ExperimentStatus.failed and (res.error or "").startswith("invalid_config")
    assert len(fns["run_cpu"].calls) == n_calls  # never spawned
    assert r.busy_slots() == 0


async def test_modal_capacity_caps(fakes) -> None:
    q, fns, resolver = fakes
    a = make_modal(resolver, q, slots={"L4": 1}, max_gpus=2)
    b = make_modal(resolver, q, slots={"L4": 2}, max_gpus=2)
    ha = await a.launch("dummy_gpu", {"sleep": 0.3}, 0, 5, lambda p: None)
    with pytest.raises(RunnerCapacityError):  # per-runner cap
        await a.launch("dummy_gpu", {}, 0, 5, lambda p: None)
    hb = await b.launch("dummy_gpu", {"sleep": 0.3}, 0, 5, lambda p: None)
    with pytest.raises(RunnerCapacityError):  # workspace-wide GPU cap across runners
        await b.launch("dummy_gpu", {}, 0, 5, lambda p: None)
    await a.wait(ha)
    await b.wait(hb)
    assert ModalRunner._global_busy.get("L4") == 0


def test_modal_config_validation() -> None:
    with pytest.raises(ValueError):
        ModalRunner({"H100": 1})
    with pytest.raises(ValueError):
        ModalRunner({"L4": 11})
    with pytest.raises(ValueError):
        ModalRunner({"L4": 4, "CPU": 97})
    r = ModalRunner({"L4": 1, "CPU": 4}, cpu_cores=2.0)
    assert r.price_per_second("L4") == pytest.approx(0.000222)
    assert r.price_per_second("CPU") == pytest.approx(2 * 0.0000131)
    assert ModalRunner({"L4": 1}, price_table={"L4": 0.001}).price_per_second("L4") == 0.001


async def test_modal_cancel_and_timeout(fakes) -> None:
    q, fns, resolver = fakes
    r = make_modal(resolver, q, grace_s=0)
    h = await r.launch("dummy_gpu", {"sleep": 30.0}, 0, 30, lambda p: None)
    await asyncio.sleep(0.2)
    await r.cancel(h)
    res = await r.wait(h)
    assert res.status == ExperimentStatus.cancelled
    assert fns["run_l4"].calls[-1].cancel_kwargs == {"terminate_containers": False}

    h = await r.launch("dummy_gpu", {"sleep": 30.0}, 0, 1, lambda p: None)
    t0 = time.time()
    res = await r.wait(h)
    assert res.status == ExperimentStatus.timed_out and time.time() - t0 < 5
    assert fns["run_l4"].calls[-1].cancel_kwargs == {"terminate_containers": True}
    assert r.busy_slots() == 0


async def test_modal_prewarm(fakes) -> None:
    q, fns, resolver = fakes
    r = make_modal(resolver, q)
    await r.prewarm("L4")
    await r.cooldown("L4")
    assert fns["run_l4"].autoscaler == [{"min_containers": 2}, {"min_containers": 0}]
    with pytest.raises(ValueError):
        await r.prewarm("L4", 50)
    with pytest.raises(ValueError):
        await r.prewarm("CPU", 101)
    await r.prewarm("CPU", 66)
    await r.cooldown("CPU", reset_scaledown=True)
    assert fns["run_cpu"].autoscaler == [{"min_containers": 66},
                                         {"min_containers": 0, "scaledown_window": modal_runner.SCALEDOWN_WINDOW_S}]


class _WarmCall:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.object_id = "fc-warm"
        self.get = _Aio(self._get)
        self.cancel = _Aio(self._cancel)

    async def _get(self, timeout: float | None = None) -> Any:
        await asyncio.sleep(0.01)
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    async def _cancel(self, terminate_containers: bool = False) -> None:
        return None


class _WarmFunction:
    """Answers warm-up jobs from a pool that grows by ``grow`` containers per round."""

    def __init__(self, start: int, grow: int) -> None:
        self.pool, self.grow, self.jobs, self.i = start, grow, [], 0
        self.spawn = _Aio(self._spawn)

    async def _spawn(self, job: dict[str, Any]) -> _WarmCall:
        self.jobs.append(job)
        self.i += 1
        return _WarmCall({"warmup": True, "container_id": f"ta-{self.i % self.pool}", "import_s": 0.5,
                          "cpu_count": 1})

    def next_round(self) -> None:
        self.pool += self.grow


async def test_modal_warm_up_waits_for_distinct_containers() -> None:
    fn = _WarmFunction(start=3, grow=3)
    rounds: list[str] = []

    def log(msg: str) -> None:
        rounds.append(msg)
        fn.next_round()

    r = ModalRunner({"CPU": 4}, function_resolver=lambda name: fn, queue=FakeQueue())
    res = await r.warm_up("CPU", 8, target="dummy_ok", hold_s=0.0, log=log)
    assert res["ok"] and res["warm"] >= 8 and len(res["rounds"]) == 3  # 3 -> 6 -> 9 containers
    assert [x["distinct_containers"] for x in res["rounds"]] == [3, 6, 8]
    assert all(j["warmup"] and j["target"] == "dummy_ok" and j["hardware"] == "CPU" for j in fn.jobs)
    assert len(fn.jobs) == 8 * 3 and len(rounds) == 3
    assert r.busy_slots() == 0  # warm-up jobs never take a slot

    stuck = _WarmFunction(start=2, grow=0)
    r2 = ModalRunner({"CPU": 4}, function_resolver=lambda name: stuck, queue=FakeQueue())
    res2 = await r2.warm_up("CPU", 4, hold_s=0.0, max_rounds=2)
    assert not res2["ok"] and res2["warm"] == 2 and len(res2["rounds"]) == 2


def test_warmup_payload_reports_container_identity() -> None:
    out = modal_runner.warmup_payload({"warmup": True, "hardware": "CPU", "imports": ["json"], "target": "dummy_ok",
                                       "hold_s": 0.0})
    assert out["warmup"] and out["container_id"] and out["imported"] == ["json"]
    assert out["cpu_count"] and out["cpu_count"] >= 1 and out["import_s"] >= 0
    assert modal_runner._needs_volume("dummy_ok") is False and modal_runner._needs_volume("no-such-target") is True


def test_queue_pusher_batches() -> None:
    q = FakeQueue()
    p = modal_runner._QueuePusher(q, "run1", flush_every_s=0.01)
    for i in range(5):
        p.emit(i, 0.1 * i, "loss", 1.0)
    p.close()
    assert [it["step"] for it in q.parts["run1"]] == [0, 1, 2, 3, 4]


def test_modal_module_definitions() -> None:
    assert set(modal_runner.HARDWARE_FUNCTIONS) == {"L4", "CPU"}
    assert "H100" not in modal_runner.HARDWARE_FUNCTIONS
    for name in ("run_l4", "run_cpu", "prepare_data"):
        assert hasattr(modal_runner, name)


class _HungCall:
    """A FunctionCall whose get/cancel never return and ignore cancellation (the 64-run hang)."""

    object_id = "fc-hung"

    def __init__(self, result_after: int | None = None) -> None:
        self.n_get = 0
        self.result_after = result_after
        self.get = _Aio(self._get)
        self.cancel = _Aio(self._hang)

    async def _hang(self, **_: Any) -> None:
        ignored = False  # swallows the first cancel (the caller's), honours the next (loop teardown)
        while True:
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                if ignored:
                    raise
                ignored = True

    async def _get(self, timeout: float | None = None) -> Any:
        self.n_get += 1
        if self.result_after is not None and self.n_get > self.result_after:
            return {"ok": True, "result": {"score": 1.0, "metrics": {}}, "logs_tail": [], "ended_at": time.time()}
        await self._hang()


class _HungFunction:
    def __init__(self, call: _HungCall) -> None:
        self.call = call
        self.spawn = _Aio(self._spawn)

    async def _spawn(self, job: dict[str, Any]) -> _HungCall:
        return self.call


async def test_bounded_abandons_a_call_that_ignores_cancellation() -> None:
    keep: set[asyncio.Task[Any]] = set()
    t = time.time()
    with pytest.raises(TimeoutError):
        await modal_runner._bounded(_HungCall()._hang(), 0.1, keep)
    assert time.time() - t < 1.0 and len(keep) == 1
    assert await modal_runner._bounded(asyncio.sleep(0, result=7), 1.0) == 7


async def test_modal_wait_never_hangs_on_a_stuck_call(dummy_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modal_runner, "RPC_TIMEOUT_S", 0.2)
    # get recovers on a later attempt: the result is still collected
    call = _HungCall(result_after=2)
    r = ModalRunner({"CPU": 1}, function_resolver=lambda n: _HungFunction(call), queue=FakeQueue(),
                    poll_interval_s=0.05, grace_s=1, startup_timeout_s=1)
    h = await r.launch("dummy_ok", {}, 0, 1, lambda p: None)
    res = await r.wait(h)
    assert res.status == ExperimentStatus.succeeded and call.n_get == 3 and r.busy_slots() == 0
    # get never recovers: the run times out at its deadline and the slot is released
    stuck = _HungCall()
    r2 = ModalRunner({"CPU": 1}, function_resolver=lambda n: _HungFunction(stuck), queue=FakeQueue(),
                     poll_interval_s=0.05, grace_s=0, startup_timeout_s=0.5)
    h2 = await r2.launch("dummy_ok", {}, 0, 1, lambda p: None)
    t = time.time()
    res2 = await r2.wait(h2)
    assert res2.status == ExperimentStatus.timed_out and time.time() - t < 6 and r2.busy_slots() == 0
    t = time.time()
    await r2.cancel(h2)  # cancel of a stuck call returns too
    assert time.time() - t < 2


# --------------------------------------------------------------------------- Modal cancel lag (fairness fix 1)


class _LagCall(FakeCall):
    """A container that keeps training for ``lag_s`` after cancel() (Modal measured ~2 s on rl,
    24 Oct): get() raises the cancellation only once the input has really stopped. With
    ``confirm=False`` Modal never confirms (get keeps timing out)."""

    def __init__(self, job: dict[str, Any], q: FakeQueue, lag_s: float, confirm: bool = True) -> None:
        super().__init__(job, q)
        self.lag_s, self.confirm = lag_s, confirm
        self.cancel_at: float | None = None
        self.cancel_calls: list[dict[str, Any]] = []

    async def _get(self, timeout: float | None = None) -> Any:
        deadline = time.time() + (timeout or 0)
        while not self.done.is_set():
            if self.cancel_at is not None and self.confirm and time.time() >= self.cancel_at + self.lag_s:
                raise RuntimeError("Function call was cancelled by user or a failure.")
            if time.time() >= deadline:
                raise TimeoutError()
            await asyncio.sleep(0.01)
        return self.payload

    async def _cancel(self, terminate_containers: bool = False) -> None:
        self.cancel_calls.append({"terminate_containers": terminate_containers})
        if self.cancel_at is None:
            self.cancel_at = time.time()
        self.cancel_kwargs = {"terminate_containers": terminate_containers}


class _LagFunction(FakeFunction):
    def __init__(self, name: str, q: FakeQueue, lag_s: float, confirm: bool = True) -> None:
        super().__init__(name, q)
        self.lag_s, self.confirm = lag_s, confirm

    async def _spawn(self, job: dict[str, Any]) -> FakeCall:
        c = _LagCall(job, self.q, self.lag_s, self.confirm)
        self.calls.append(c)
        return c


async def test_modal_cancel_keeps_the_slot_until_modal_confirms(dummy_dir: Path) -> None:
    ModalRunner._global_busy.clear()
    q = FakeQueue()
    fn = _LagFunction("run_l4", q, lag_s=0.6)
    r = ModalRunner({"L4": 1}, function_resolver=lambda name: fn, queue=q, poll_interval_s=0.05)
    h = await r.launch("dummy_gpu", {"sleep": 30.0}, 0, 30, lambda p: None)
    waiter = asyncio.create_task(r.wait(h))
    await asyncio.sleep(0.2)
    t_cancel = time.time()
    await r.cancel(h)
    assert time.time() - t_cancel < 0.3          # cancel() itself returns at once
    await asyncio.sleep(0.3)
    # the remote input is still running: the slot is still reserved, wait() has not returned
    assert not waiter.done() and r.busy_slots("L4") == 1 and ModalRunner._global_busy["L4"] == 1
    res = await waiter
    lag = time.time() - t_cancel
    assert res.status == ExperimentStatus.cancelled and res.error == "cancelled"
    assert lag >= 0.6 and res.metrics["cancel_lag_s"] >= 0.6 and res.metrics["cancel_confirmed"] == 1.0
    assert res.ended_at >= t_cancel + 0.6         # duration (and so the arm's $) covers the lag
    assert fn.calls[0].cancel_calls == [{"terminate_containers": False}]   # warm container kept
    assert r.busy_slots() == 0 and ModalRunner._global_busy["L4"] == 0
    ModalRunner._global_busy.clear()


async def test_modal_cancel_never_confirmed_terminates_after_cancel_grace(dummy_dir: Path) -> None:
    ModalRunner._global_busy.clear()
    q = FakeQueue()
    fn = _LagFunction("run_l4", q, lag_s=0.0, confirm=False)
    r = ModalRunner({"L4": 1}, function_resolver=lambda name: fn, queue=q, poll_interval_s=0.05,
                    cancel_grace_s=0.5)
    h = await r.launch("dummy_gpu", {"sleep": 30.0}, 0, 30, lambda p: None)
    await asyncio.sleep(0.1)
    t_cancel = time.time()
    await r.cancel(h)
    res = await r.wait(h)
    assert 0.5 <= time.time() - t_cancel < 3.0
    assert res.status == ExperimentStatus.cancelled and "did not confirm" in (res.error or "")
    assert res.metrics["cancel_confirmed"] == 0.0
    assert fn.calls[0].cancel_calls[-1] == {"terminate_containers": True}   # escalated
    assert r.busy_slots() == 0
    ModalRunner._global_busy.clear()


async def test_modal_cancel_after_the_run_finished_reports_the_real_outcome(dummy_dir: Path) -> None:
    """The run ends before the cancel lands: wait() returns the real payload (the scheduler
    keeps it as early_stop 'too_late'), not a made-up cancellation."""
    ModalRunner._global_busy.clear()
    q = FakeQueue()
    fn = _LagFunction("run_l4", q, lag_s=5.0)
    r = ModalRunner({"L4": 1}, function_resolver=lambda name: fn, queue=q, poll_interval_s=0.05)
    h = await r.launch("dummy_gpu", {"x": 0.3, "sleep": 0.4}, 0, 5, lambda p: None)
    await asyncio.sleep(0.15)
    await r.cancel(h)
    res = await r.wait(h)
    assert res.status == ExperimentStatus.succeeded and res.score == pytest.approx(0.3)
    assert res.metrics["cancel_confirmed"] == 1.0 and res.metrics["cancel_lag_s"] < 5.0
    assert r.busy_slots() == 0
    ModalRunner._global_busy.clear()


async def test_modal_wait_cancelled_from_outside_releases_the_slot(dummy_dir: Path) -> None:
    ModalRunner._global_busy.clear()
    q = FakeQueue()
    fn = _LagFunction("run_l4", q, lag_s=10.0)
    r = ModalRunner({"L4": 1}, function_resolver=lambda name: fn, queue=q, poll_interval_s=0.05)
    h = await r.launch("dummy_gpu", {"sleep": 30.0}, 0, 30, lambda p: None)
    waiter = asyncio.create_task(r.wait(h))
    await asyncio.sleep(0.2)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert r.busy_slots() == 0 and ModalRunner._global_busy["L4"] == 0
    await r.drain_background()
    assert fn.calls[0].cancel_calls == [{"terminate_containers": True}]   # nothing left running unseen
    ModalRunner._global_busy.clear()


def test_modal_cancel_options() -> None:
    r = ModalRunner({"L4": 1}, grace_s=7)
    assert r.cancel_grace_s == 7.0 and r.cancel_terminates is False
    r = ModalRunner({"L4": 1}, grace_s=7, cancel_grace_s=2.5, cancel_terminates=True)
    assert r.cancel_grace_s == 2.5 and r.cancel_terminates is True
