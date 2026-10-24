"""workers/runners/modal_stats.py, offline: fakes for Modal's stats call and for nvidia-smi.
No Modal call, no GPU (CLAUDE.md: the laptop runs CPU-only tests)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from workers.runners import modal_stats as ms

# --------------------------------------------------------------------------- nvidia-smi / sampler


SMI_OUT = "0, 87, 10240, 23034\n"


def test_parse_nvidia_smi() -> None:
    r = ms.parse_nvidia_smi(SMI_OUT)
    assert r == [ms.GpuReading(index=0, util_pct=87.0, mem_used_mb=10240.0, mem_total_mb=23034.0)]
    two = ms.parse_nvidia_smi("0, 10, 100, 1000\n1, [N/A], 200, 1000\nbad line\n")
    assert len(two) == 2 and two[1].index == 1 and two[1].mem_used_mb == 200.0
    assert two[1].util_pct != two[1].util_pct  # NaN for "[N/A]"
    assert ms.parse_nvidia_smi("") == []


def test_nvidia_smi_reader_uses_the_query_and_checks_exit_code() -> None:
    seen: list[Any] = []

    def run(cmd: list[str], **kw: Any) -> Any:
        seen.append((cmd, kw))
        return SimpleNamespace(returncode=0, stdout=SMI_OUT, stderr="")

    assert ms.nvidia_smi_reader(run)()[0].util_pct == 87.0
    assert seen[0][0] == ms.NVIDIA_SMI_QUERY and seen[0][1]["timeout"] > 0

    def bad(cmd: list[str], **kw: Any) -> Any:
        return SimpleNamespace(returncode=9, stdout="", stderr="NVIDIA-SMI has failed")

    with pytest.raises(RuntimeError, match="exit 9"):
        ms.nvidia_smi_reader(bad)()


class _Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_sampler_emits_util_mem_and_total_once_and_reports_zero_windows() -> None:
    utils = iter([0.0, 95.0, 90.0, 0.0])
    clock = _Clock(1000.0)
    pts: list[tuple[int, float, str, float]] = []

    def reader() -> list[ms.GpuReading]:
        clock.t += 1.0
        return [ms.GpuReading(0, next(utils), 5000.0, 23034.0)]

    s = ms.GpuSampler(lambda *a: pts.append(a), started_at=1000.0, reader=reader, clock=clock)
    for _ in range(4):
        assert s.sample()
    names = [p[2] for p in pts]
    assert names.count("gpu_util_pct") == 4 and names.count("gpu_mem_mb") == 4
    assert names.count("gpu_mem_total_mb") == 1
    util = [(p[0], p[1], p[3]) for p in pts if p[2] == "gpu_util_pct"]
    assert util == [(0, 1.0, 0.0), (1, 2.0, 95.0), (2, 3.0, 90.0), (3, 4.0, 0.0)]  # step = sample index
    summ = s.summary()
    assert summ["samples"] == 4 and summ["zero_util_samples"] == 2
    assert summ["mean_util_pct"] == pytest.approx(46.25)


def test_sampler_gives_up_after_consecutive_failures_and_never_raises() -> None:
    calls = {"n": 0}

    def reader() -> list[ms.GpuReading]:
        calls["n"] += 1
        raise FileNotFoundError("nvidia-smi")

    pts: list[Any] = []
    s = ms.GpuSampler(lambda *a: pts.append(a), reader=reader, max_failures=3)
    for _ in range(10):
        assert s.sample() is False
    assert calls["n"] == 3 and pts == [] and s.errors and "FileNotFoundError" in s.errors[0]


def test_sampler_emit_errors_never_reach_the_run() -> None:
    def emit(*a: Any) -> None:
        raise RuntimeError("queue down")

    s = ms.GpuSampler(emit, reader=lambda: [ms.GpuReading(0, 50.0, 1.0, 2.0)])
    assert s.sample() is True


def test_sampler_thread_samples_while_running_and_once_at_stop() -> None:
    lock = threading.Lock()
    pts: list[Any] = []

    def emit(*a: Any) -> None:
        with lock:
            pts.append(a)

    s = ms.GpuSampler(emit, interval_s=0.05, reader=lambda: [ms.GpuReading(0, 70.0, 100.0, 200.0)]).start()
    time.sleep(0.3)
    summ = s.stop()
    n_util = sum(1 for p in pts if p[2] == "gpu_util_pct")
    assert n_util == summ["samples"] >= 4          # start + loop ticks + stop
    assert not s._thread.is_alive()                # type: ignore[union-attr]


def test_sampler_never_blocks_the_run_on_a_slow_or_hung_read() -> None:
    """start() returns at once and stop(final_sample=False) waits at most ~0.05 s, even when a GPU
    read hangs (e.g. nvidia-smi stuck): the training run is never held back by telemetry."""
    release = threading.Event()

    def hung() -> list[ms.GpuReading]:
        release.wait(5.0)
        return [ms.GpuReading(0, 50.0, 1.0, 2.0)]

    t0 = time.monotonic()
    s = ms.GpuSampler(lambda *a: None, interval_s=0.05, reader=hung).start()
    assert time.monotonic() - t0 < 0.2
    time.sleep(0.1)
    t1 = time.monotonic()
    s.stop(final_sample=False)
    assert time.monotonic() - t1 < 0.4
    release.set()


def test_sampler_for_job_only_with_the_flag() -> None:
    reader = lambda: [ms.GpuReading(0, 10.0, 1.0, 2.0)]  # noqa: E731
    pts: list[Any] = []
    assert ms.sampler_for_job({"run_id": "r", "target": "cifar"}, pts.append, reader=reader) is None
    assert ms.sampler_for_job({"gpu_telemetry": None}, pts.append, reader=reader) is None
    assert pts == []
    s = ms.sampler_for_job({"gpu_telemetry": {"interval_s": 0.5}, "started_at": time.time()},
                           lambda *a: pts.append(a), reader=reader)
    assert s is not None and s.interval_s == 0.5
    s.stop()
    assert any(p[2] == "gpu_util_pct" for p in pts)


# --------------------------------------------------------------------------- Modal stats poller


class FakeStats:
    """Stands in for modal.Function.get_current_stats (records (app, function) per call)."""

    def __init__(self, by_app: dict[str, Any]) -> None:
        self.by_app = by_app
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, app: str, function: str) -> Any:
        self.calls.append((app, function))
        v = self.by_app[app]
        if isinstance(v, BaseException):
            raise v
        if callable(v):
            return await v()
        return v


def _stats(backlog: int, total: int, running: int, headroom: int = 0) -> Any:
    # same attribute names as modal.types.FunctionCurrentStats (modal 1.6.2)
    return SimpleNamespace(backlog=backlog, num_total_runners=total, num_running_inputs=running,
                           input_headroom=headroom)


APPS = {"opus": "busy-machines-opus", "opus_bm": "busy-machines-bm"}


def test_snapshot_maps_modal_fields_and_derives_idle() -> None:
    s = ms.snapshot_from_stats("opus", "a", "run_l4", _stats(1, 3, 1, 2), ts=5.0, latency_s=0.1)
    assert (s.containers, s.calls_in_flight, s.backlog, s.input_headroom, s.idle_containers) == (3, 1, 1, 2, 2)
    assert s.ok and s.source == "modal.Function.get_current_stats"
    d = ms.snapshot_from_stats("opus", "a", "run_l4", {"num_total_runners": 2, "num_running_inputs": 5}, ts=1.0)
    assert d.idle_containers == 0 and d.backlog is None   # never negative; missing fields stay None


def test_real_stats_fields_match_the_installed_sdk() -> None:
    """The names we read exist on the installed SDK's FunctionCurrentStats (no guessing)."""
    types = pytest.importorskip("modal.types")
    import dataclasses

    fields = {f.name for f in dataclasses.fields(types.FunctionCurrentStats)}
    assert {"backlog", "num_total_runners", "num_running_inputs", "input_headroom"} <= fields
    modal = pytest.importorskip("modal")
    assert hasattr(modal.Function, "get_current_stats")


def test_hardware_functions_mirror_modal_runner() -> None:
    mr = pytest.importorskip("workers.runners.modal_runner")
    assert ms.HARDWARE_FUNCTIONS == mr.HARDWARE_FUNCTIONS
    assert ms.function_for_hardware("L4") == "run_l4" and ms.function_for_hardware("CPU") == "run_cpu"
    with pytest.raises(ValueError):
        ms.function_for_hardware("H100")


async def test_poll_once_per_arm_on_its_own_app(tmp_path: Path) -> None:
    fake = FakeStats({"busy-machines-opus": _stats(0, 3, 1), "busy-machines-bm": _stats(0, 3, 3)})
    got: list[Any] = []
    log = tmp_path / "stats.jsonl"
    p = ms.ModalStatsPoller(APPS, "run_l4", stats_fn=fake, on_snapshot=got.append, log_path=log)
    snaps = await p.poll_once()
    assert sorted(fake.calls) == [("busy-machines-bm", "run_l4"), ("busy-machines-opus", "run_l4")]
    by = {s.arm: s for s in snaps}
    assert by["opus"].idle_containers == 2 and by["opus_bm"].idle_containers == 0
    assert got == [snaps] and p.latest["opus"].calls_in_flight == 1
    line = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert line["tick"] == 1 and {s["arm"] for s in line["snapshots"]} == {"opus", "opus_bm"}
    v = p.view(history=5)
    assert v["latest"]["opus_bm"]["calls_in_flight"] == 3 and len(v["history"]) == 1


async def test_poll_errors_and_timeouts_are_snapshots_not_crashes() -> None:
    async def hang() -> Any:
        await asyncio.sleep(30)

    fake = FakeStats({"busy-machines-opus": RuntimeError("app not found"), "busy-machines-bm": hang})
    p = ms.ModalStatsPoller(APPS, stats_fn=fake, timeout_s=0.1)
    t0 = time.time()
    snaps = {s.arm: s for s in await p.poll_once()}
    assert time.time() - t0 < 2.0
    assert not snaps["opus"].ok and "app not found" in (snaps["opus"].error or "")
    assert not snaps["opus_bm"].ok and "TimeoutError" in (snaps["opus_bm"].error or "")
    assert snaps["opus"].containers is None and p.errors == 2


async def test_run_polls_on_interval_until_keep_going_is_false() -> None:
    fake = FakeStats({"busy-machines-opus": _stats(0, 1, 1), "busy-machines-bm": _stats(0, 1, 0)})
    ticks: list[int] = []
    p = ms.ModalStatsPoller(APPS, stats_fn=fake, interval_s=0.1, keep_going=lambda: len(ticks) < 3,
                            on_snapshot=lambda s: ticks.append(1))
    await asyncio.wait_for(p.run(), timeout=5)
    assert p.ticks == 3 and len(fake.calls) == 6 and p.stopped_at is not None


async def test_start_stop_and_max_runtime() -> None:
    fake = FakeStats({"busy-machines-opus": _stats(0, 1, 1), "busy-machines-bm": _stats(0, 1, 0)})
    p = ms.ModalStatsPoller(APPS, stats_fn=fake, interval_s=0.05)
    t = p.start()
    assert p.start() is t and p.running
    await asyncio.sleep(0.2)
    await p.stop()
    assert not p.running and p.ticks >= 2
    q = ms.ModalStatsPoller(APPS, stats_fn=fake, interval_s=0.05, max_runtime_s=0.12)
    await asyncio.wait_for(q.run(), timeout=5)
    assert 1 <= q.ticks <= 4


def test_poller_needs_apps_and_never_imports_modal_without_a_poll() -> None:
    with pytest.raises(ValueError):
        ms.ModalStatsPoller({})
    p = ms.ModalStatsPoller(APPS)   # default stats_fn is created lazily, on the first poll only
    assert p._stats_fn is None


# --------------------------------------------------------------------------- per-arm GPU join


def test_tracker_counts_idle_warm_containers_as_zero() -> None:
    tr = ms.ArmGpuTracker(stale_s=5.0)
    tr.observe("opus", "e1", "gpu_util_pct", 90.0, ts=100.0)
    tr.observe("opus", "e1", "gpu_mem_mb", 6000.0, ts=100.0)
    tr.observe("opus", "e1", "loss", 1.0, ts=100.0)       # ignored: not a GPU metric
    s = tr.summary("opus", 3, 1, now=101.0)       # 1 busy at 90 %, 2 warm with no call in flight
    assert s == {"gpu_util_pct": pytest.approx(30.0), "gpu_util_active_pct": 90.0, "gpu_mem_mb": 6000.0,
                 "runs_sampled": 1, "runs_unsampled": 0}
    # 3 containers, 2 calls in flight but only 1 sampled: the unsampled run is left out, not 0 %
    s = tr.summary("opus", 3, 2, now=101.0)
    assert s["gpu_util_pct"] == pytest.approx(45.0) and s["runs_unsampled"] == 1
    tr.observe("opus_bm", "e2", "gpu_util_pct", 80.0, ts=100.0)
    tr.observe("opus_bm", "e3", "gpu_util_pct", 60.0, ts=100.5)
    assert tr.summary("opus_bm", 2, 2, now=101.0)["gpu_util_pct"] == pytest.approx(70.0)
    # Modal's stats lag our samples (1 container / 0 running reported): never above the active mean
    assert tr.summary("opus_bm", 1, 0, now=101.0)["gpu_util_pct"] == pytest.approx(70.0)
    tr.finish("opus_bm", "e2")
    assert tr.summary("opus_bm", 2, 1, now=101.0)["runs_sampled"] == 1
    # stale samples drop out; calls in flight with no sample -> unknown, never a made-up 0
    assert tr.summary("opus", 3, 2, now=200.0) == {"gpu_util_pct": None, "gpu_util_active_pct": None,
                                                   "gpu_mem_mb": None, "runs_sampled": 0, "runs_unsampled": 2}
    assert tr.summary("opus", None, None, now=200.0)["gpu_util_pct"] is None
    # every container warm with no call in flight: 0 % is a fact (nothing of ours runs there)
    assert tr.summary("opus", 3, 0, now=200.0)["gpu_util_pct"] == 0.0
    assert tr.summary("opus", 0, 0, now=200.0)["gpu_util_pct"] is None   # no container, no GPU


def test_calls_in_flight_by_arm() -> None:
    rows = [
        {"id": "e1", "arm_id": "A", "run_ref": "fc-1", "status": "succeeded", "started_at": 1.0, "ended_at": 2.0},
        {"id": "e2", "arm_id": "A", "run_ref": "fc-2", "status": "running", "started_at": 3.0},
        {"id": "e3", "arm_id": "B", "run_ref": None, "status": "pending"},
        {"id": "e4", "arm_id": "B", "run_ref": "fc-4", "status": "running"},
    ]
    out = ms.calls_in_flight_by_arm(rows, {"A": "opus", "B": "opus_bm"})
    assert [c["modal_call_id"] for c in out["opus"]] == ["fc-1", "fc-2"]
    assert [c["in_flight"] for c in out["opus"]] == [False, True]
    assert [c["experiment_id"] for c in out["opus_bm"]] == ["e4"]
    assert ms.latest_call_id(out["opus"]) == "fc-2"
    assert ms.latest_call_id(out["opus_bm"]) is None   # no start time recorded
    assert ms.latest_call_id([]) is None
