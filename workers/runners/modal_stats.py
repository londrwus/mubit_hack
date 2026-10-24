"""Live Modal metrics for our own dashboard (DEMO RACES ONLY: never a benchmark or ablation race).

Two independent pieces, both fake-friendly (tests never touch Modal or a GPU):

1. Host side, ``ModalStatsPoller``: every ``interval_s`` (default 2 s) it asks Modal for each
   arm's function stats in that arm's own deployed app (per-arm apps, ``RaceConfig.modal_apps``):
   ``modal.Function.from_name(app, fn).get_current_stats.aio()``. Checked against the installed
   SDK (modal 1.6.2, ``modal/_functions.py`` and ``modal/types.py``): it returns
   ``FunctionCurrentStats(backlog, num_total_runners, num_running_inputs, input_headroom)``, one
   gRPC call with Modal's own 10 s retry budget. We map it to:

   ===================  ======================  ==================================================
   our field            Modal field             meaning
   ===================  ======================  ==================================================
   ``containers``       ``num_total_runners``   containers alive for the function (warm or busy)
   ``calls_in_flight``  ``num_running_inputs``  inputs (our runs) executing right now
   ``backlog``          ``backlog``             inputs waiting for a container
   ``input_headroom``   ``input_headroom``      inputs Modal could accept now without scaling
   ``idle_containers``  derived                 ``max(0, containers - calls_in_flight)``: warm
                                                containers with no call in flight (run_l4/run_cpu
                                                take one input per container, so these do nothing)
   ===================  ======================  ==================================================

   ``Function.stats()`` (historical counts) and the SDK's metrics API (buckets of >= 60 s) are too
   coarse for a live view, so they are not used.

2. Container side, ``GpuSampler``: while a run trains, a daemon thread reads the GPU every
   ``interval_s`` (default 1 s; pynvml if importable, else ``nvidia-smi``) and emits metric points
   ``gpu_util_pct`` (0-100, mean over the container's GPUs) and ``gpu_mem_mb`` (MiB used, summed),
   plus ``gpu_mem_total_mb`` once. It only runs when the job carries ``gpu_telemetry`` (set by
   ``ModalRunner(gpu_telemetry_s=...)``, i.e. demo presets with ``RaceConfig.gpu_telemetry_s``),
   so benchmark jobs are byte-for-byte unchanged. The points ride the existing per-run metric path
   (Modal Queue -> ``ModalRunner`` -> ``on_metric`` -> DB ``metrics`` + ``metric`` WS events). A
   container only sees its own runs: windows *between* runs (warm, no call in flight) are not
   sampled; ``ArmGpuTracker`` counts them as 0 % (``idle_containers`` above).

``ArmGpuTracker`` joins the two per arm: the latest in-container sample of each run in flight and
Modal's container count, giving ``gpu_util_pct`` over all warm containers (idle ones at 0 %).

This module never imports ``modal`` at import time (the real stats function imports it lazily),
so the container and the host can import it freely; CLAUDE.md allows ``modal`` here
(``workers/runners/modal_*.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import subprocess
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Metric point names the container emits (demo races only).
GPU_UTIL_METRIC = "gpu_util_pct"
GPU_MEM_METRIC = "gpu_mem_mb"
GPU_MEM_TOTAL_METRIC = "gpu_mem_total_mb"
GPU_METRICS: frozenset[str] = frozenset({GPU_UTIL_METRIC, GPU_MEM_METRIC, GPU_MEM_TOTAL_METRIC})

DEFAULT_POLL_INTERVAL_S = 2.0
DEFAULT_SAMPLE_INTERVAL_S = 1.0
STATS_TIMEOUT_S = 8.0            # cap per get_current_stats call (Modal itself retries for 10 s)
SAMPLE_STALE_S = 5.0             # an in-container sample older than this no longer counts as live
STATS_SOURCE = "modal.Function.get_current_stats"

# hardware -> function name; mirrors workers.runners.modal_runner.HARDWARE_FUNCTIONS (a test checks
# it; kept here so importing this module never builds the Modal images).
HARDWARE_FUNCTIONS: dict[str, str] = {"L4": "run_l4", "CPU": "run_cpu"}


# =========================================================================== container side


@dataclass(frozen=True)
class GpuReading:
    index: int
    util_pct: float
    mem_used_mb: float
    mem_total_mb: float


GpuReader = Callable[[], list[GpuReading]]
NVIDIA_SMI_QUERY = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits"]


def _num(s: str) -> float:
    s = s.strip()
    try:
        v = float(s)
    except ValueError:      # "[N/A]", "[Not Supported]"
        return math.nan
    return v


def parse_nvidia_smi(text: str) -> list[GpuReading]:
    """Parse ``nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total
    --format=csv,noheader,nounits`` output (one line per GPU, e.g. ``0, 87, 10240, 23034``)."""
    out: list[GpuReading] = []
    for line in text.strip().splitlines():
        parts = [p for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx = _num(parts[0])
        out.append(GpuReading(index=int(idx) if math.isfinite(idx) else len(out), util_pct=_num(parts[1]),
                              mem_used_mb=_num(parts[2]), mem_total_mb=_num(parts[3])))
    return out


def nvidia_smi_reader(run: Callable[..., Any] = subprocess.run, timeout_s: float = 2.0) -> GpuReader:
    def read() -> list[GpuReading]:
        r = run(NVIDIA_SMI_QUERY, capture_output=True, text=True, timeout=timeout_s, check=False)
        if getattr(r, "returncode", 1) != 0:
            raise RuntimeError(f"nvidia-smi exit {getattr(r, 'returncode', '?')}: {str(getattr(r, 'stderr', ''))[:200]}")
        return parse_nvidia_smi(str(r.stdout))
    return read


def pynvml_reader() -> GpuReader | None:
    """A reader over NVML (cheaper than spawning nvidia-smi), or None if pynvml is missing."""
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    except Exception:  # noqa: BLE001 (not installed, no driver)
        return None

    def read() -> list[GpuReading]:
        out = []
        for i, h in enumerate(handles):
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append(GpuReading(index=i, util_pct=float(u.gpu), mem_used_mb=m.used / 2**20,
                                  mem_total_mb=m.total / 2**20))
        return out
    return read


def default_gpu_reader() -> GpuReader:
    return pynvml_reader() or nvidia_smi_reader()


class GpuSampler:
    """Background GPU sampler for one run inside a container.

    ``emit(step, t, name, value)`` is the run's metric emitter (``_QueuePusher.emit``: thread-safe).
    ``t`` is seconds since ``started_at`` (the job's ``started_at``, like the target's own metrics);
    ``step`` is the sample index. Samples as soon as the thread starts (never on the caller's
    thread), every ``interval_s`` after, and once more at ``stop()`` (unless
    ``final_sample=False``), so the whole run is covered, including its 0 % stretches (data loading,
    eval, Python overhead). After ``max_failures`` consecutive read errors it stops sampling (e.g.
    no GPU on a CPU container); a failure never reaches the training loop."""

    def __init__(
        self,
        emit: Callable[[int, float, str, float], None],
        *,
        started_at: float | None = None,
        interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
        reader: GpuReader | None = None,
        clock: Callable[[], float] = time.time,
        max_failures: int = 3,
    ) -> None:
        self._emit = emit
        self._clock = clock
        self.started_at = float(started_at if started_at is not None else clock())
        self.interval_s = max(0.05, float(interval_s))
        self._reader = reader
        self.max_failures = max_failures
        self.samples = 0
        self.failures = 0
        self.errors: list[str] = []
        self.util: list[float] = []
        self._total_sent = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> bool:
        """Take one sample and emit it. Returns False if the read failed."""
        with self._lock:
            if self.failures >= self.max_failures:
                return False
            try:
                if self._reader is None:
                    self._reader = default_gpu_reader()
                gpus = self._reader()
                if not gpus:
                    raise RuntimeError("no GPU visible")
            except Exception as exc:  # noqa: BLE001 (telemetry must never hurt the run)
                self.failures += 1
                if len(self.errors) < 5:
                    self.errors.append(f"{type(exc).__name__}: {exc}"[:200])
                return False
            self.failures = 0
            t = self._clock() - self.started_at
            utils = [g.util_pct for g in gpus if math.isfinite(g.util_pct)]
            mems = [g.mem_used_mb for g in gpus if math.isfinite(g.mem_used_mb)]
            step = self.samples
            if utils:
                u = sum(utils) / len(utils)
                self.util.append(u)
                self._safe_emit(step, t, GPU_UTIL_METRIC, u)
            if mems:
                self._safe_emit(step, t, GPU_MEM_METRIC, sum(mems))
            if not self._total_sent:
                tot = [g.mem_total_mb for g in gpus if math.isfinite(g.mem_total_mb)]
                if tot:
                    self._safe_emit(step, t, GPU_MEM_TOTAL_METRIC, sum(tot))
                    self._total_sent = True
            self.samples += 1
            return True

    def _safe_emit(self, step: int, t: float, name: str, value: float) -> None:
        with contextlib.suppress(Exception):
            self._emit(step, t, name, float(value))

    def _loop(self) -> None:
        # The first read (pynvml init or an nvidia-smi spawn) also happens here, never on the
        # caller's thread, so starting the sampler never delays the training run.
        if not self._stop.is_set():
            self.sample()
        while not self._stop.wait(self.interval_s):
            self.sample()

    def start(self) -> GpuSampler:
        self._thread = threading.Thread(target=self._loop, name="gpu-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self, *, final_sample: bool = True, join_timeout_s: float | None = None) -> dict[str, Any]:
        """Stop the thread; ``final_sample=False`` skips the closing read (the container path,
        so the run's result is never held back by a GPU read). With ``final_sample=False`` the
        join waits at most 0.5 s by default (a read stuck in nvidia-smi is left to the daemon
        thread, which emits nothing useful after the run's queue is closed)."""
        self._stop.set()
        if join_timeout_s is None:
            join_timeout_s = max(2.0, 2 * self.interval_s) if final_sample else 0.5
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)
        if final_sample:
            self.sample()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        u = self.util
        return {"samples": self.samples, "interval_s": self.interval_s,
                "mean_util_pct": (sum(u) / len(u)) if u else None,
                "zero_util_samples": sum(1 for x in u if x <= 0.0), "errors": list(self.errors)}


def sampler_for_job(job: Mapping[str, Any], emit: Callable[[int, float, str, float], None],
                    *, reader: GpuReader | None = None) -> GpuSampler | None:
    """Start a ``GpuSampler`` if (and only if) the job carries ``gpu_telemetry`` (demo races).
    Returns None for every other job, so benchmark runs never start a thread or read the GPU."""
    tele = job.get("gpu_telemetry")
    if not tele:
        return None
    interval = DEFAULT_SAMPLE_INTERVAL_S
    if isinstance(tele, Mapping):
        with contextlib.suppress(TypeError, ValueError):
            interval = float(tele.get("interval_s") or interval)
    try:
        return GpuSampler(emit, started_at=job.get("started_at"), interval_s=interval, reader=reader).start()
    except Exception:  # noqa: BLE001 (never block the run)
        return None


# =========================================================================== host side: Modal stats


@dataclass
class FunctionStatsSnapshot:
    """One poll of one arm's function. ``ok=False`` keeps the last error and no counts."""

    arm: str
    app: str
    function: str
    ts: float
    ok: bool
    containers: int | None = None
    calls_in_flight: int | None = None
    backlog: int | None = None
    input_headroom: int | None = None
    idle_containers: int | None = None
    poll_latency_s: float | None = None
    error: str | None = None
    source: str = STATS_SOURCE

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


StatsFn = Callable[[str, str], Awaitable[Any]]


def _field(stats: Any, name: str) -> int | None:
    v = stats.get(name) if isinstance(stats, Mapping) else getattr(stats, name, None)
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def snapshot_from_stats(arm: str, app: str, function: str, stats: Any, *, ts: float,
                        latency_s: float | None = None) -> FunctionStatsSnapshot:
    """Map a ``FunctionCurrentStats`` (or any object / dict with its four fields) to a snapshot."""
    containers = _field(stats, "num_total_runners")
    running = _field(stats, "num_running_inputs")
    idle = max(0, containers - running) if containers is not None and running is not None else None
    return FunctionStatsSnapshot(arm=arm, app=app, function=function, ts=ts, ok=True, containers=containers,
                                 calls_in_flight=running, backlog=_field(stats, "backlog"),
                                 input_headroom=_field(stats, "input_headroom"), idle_containers=idle,
                                 poll_latency_s=latency_s)


def modal_stats_fn() -> StatsFn:
    """The real stats function: ``modal.Function.from_name(app, fn).get_current_stats.aio()``.
    Function handles are cached (``from_name`` is lazy; the first call hydrates it)."""
    cache: dict[tuple[str, str], Any] = {}

    async def get(app: str, function: str) -> Any:
        import modal  # lazy: only a live demo poll ever imports it

        fn = cache.get((app, function))
        if fn is None:
            fn = cache[(app, function)] = modal.Function.from_name(app, function)
        return await fn.get_current_stats.aio()

    return get


async def _bounded(aw: Awaitable[Any], timeout: float, keep: set[asyncio.Task[Any]]) -> Any:
    """Await for at most ``timeout`` s; an overrunning call is cancelled and abandoned, never
    awaited (same reason as modal_runner._bounded: a hung aio call may ignore cancellation)."""
    task = asyncio.ensure_future(aw)
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if not done:
        task.cancel()
        keep.add(task)
        task.add_done_callback(keep.discard)
        raise TimeoutError(f"no answer within {timeout:.1f}s")
    return task.result()


class ModalStatsPoller:
    """Polls each arm's function stats every ``interval_s`` until stopped.

    apps: arm -> deployed app name (``RaceConfig.modal_apps``).
    function: the function to poll in every app (``run_l4`` for L4 targets, ``run_cpu`` for CPU;
        see ``function_for_hardware``).
    stats_fn: ``async (app, function) -> FunctionCurrentStats``; default ``modal_stats_fn()``.
    on_snapshot: called with the list of this tick's snapshots (one per arm), in the event loop.
    keep_going: checked before each tick; returning False ends ``run()`` (e.g. the race ended).
    max_runtime_s: hard stop for ``run()`` (a race whose status never leaves 'running' cannot keep
        us polling Modal forever).
    log_path: append every tick as one JSON line (evidence; the WS events themselves are not stored).
    """

    def __init__(
        self,
        apps: Mapping[str, str],
        function: str = "run_l4",
        *,
        stats_fn: StatsFn | None = None,
        interval_s: float = DEFAULT_POLL_INTERVAL_S,
        timeout_s: float = STATS_TIMEOUT_S,
        on_snapshot: Callable[[list[FunctionStatsSnapshot]], Any] | None = None,
        keep_going: Callable[[], bool] | None = None,
        max_runtime_s: float | None = None,
        history: int = 600,
        log_path: str | Path | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not apps:
            raise ValueError("no apps to poll")
        self.apps = {str(getattr(k, "value", k)): str(v) for k, v in apps.items()}
        self.function = function
        self._stats_fn = stats_fn
        self.interval_s = max(0.1, float(interval_s))
        self.timeout_s = float(timeout_s)
        self.on_snapshot = on_snapshot
        self.keep_going = keep_going
        self.max_runtime_s = max_runtime_s
        self.log_path = Path(log_path) if log_path is not None else None
        self._clock = clock
        self.latest: dict[str, FunctionStatsSnapshot] = {}
        self.history: deque[list[FunctionStatsSnapshot]] = deque(maxlen=history)
        self.ticks = 0
        self.errors = 0
        self.started_at: float | None = None
        self.stopped_at: float | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop: asyncio.Event | None = None
        self._abandoned: set[asyncio.Task[Any]] = set()

    # -- polling ------------------------------------------------------------------

    def _fn(self) -> StatsFn:
        if self._stats_fn is None:
            self._stats_fn = modal_stats_fn()
        return self._stats_fn

    async def _one(self, arm: str, app: str) -> FunctionStatsSnapshot:
        t0 = self._clock()
        try:
            stats = await _bounded(self._fn()(app, self.function), self.timeout_s, self._abandoned)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 (network, app not deployed, timeout)
            self.errors += 1
            return FunctionStatsSnapshot(arm=arm, app=app, function=self.function, ts=self._clock(), ok=False,
                                         poll_latency_s=self._clock() - t0, error=f"{type(exc).__name__}: {exc}"[:300])
        t1 = self._clock()
        return snapshot_from_stats(arm, app, self.function, stats, ts=t1, latency_s=t1 - t0)

    async def poll_once(self) -> list[FunctionStatsSnapshot]:
        snaps = list(await asyncio.gather(*(self._one(a, app) for a, app in self.apps.items())))
        self.ticks += 1
        for s in snaps:
            self.latest[s.arm] = s
        self.history.append(snaps)
        if self.log_path is not None:
            with contextlib.suppress(OSError):
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"tick": self.ticks, "snapshots": [s.as_dict() for s in snaps]}) + "\n")
        if self.on_snapshot is not None:
            try:
                r = self.on_snapshot(snaps)
                if asyncio.iscoroutine(r):
                    await r
            except Exception:  # noqa: BLE001 (a dashboard bug must not stop polling)
                pass
        return snaps

    async def run(self) -> None:
        if self._stop is None:
            self._stop = asyncio.Event()
        self.started_at = self._clock()
        try:
            while not self._stop.is_set():
                if self.keep_going is not None and not self.keep_going():
                    break
                if self.max_runtime_s is not None and self._clock() - self.started_at > self.max_runtime_s:
                    break
                t_tick = self._clock()
                await self.poll_once()
                wait = max(0.0, self.interval_s - (self._clock() - t_tick))
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=wait)
        finally:
            self.stopped_at = self._clock()

    def start(self) -> asyncio.Task[None]:
        """Run in the background (call from the event loop). Idempotent while running."""
        if self._task is None or self._task.done():
            self._stop = asyncio.Event()
            self._task = asyncio.create_task(self.run(), name="modal-stats-poller")
        return self._task

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._task is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(self._task, timeout=self.timeout_s + 1.0)

    def view(self, history: int = 0) -> dict[str, Any]:
        """JSON-able state for the API: latest snapshot per arm (+ the last ``history`` ticks)."""
        out: dict[str, Any] = {
            "function": self.function, "interval_s": self.interval_s, "source": STATS_SOURCE,
            "running": self.running, "ticks": self.ticks, "errors": self.errors,
            "started_at": self.started_at, "stopped_at": self.stopped_at,
            "latest": {a: s.as_dict() for a, s in self.latest.items()},
        }
        if history > 0:
            out["history"] = [[s.as_dict() for s in tick] for tick in list(self.history)[-history:]]
        return out


def function_for_hardware(hardware: str) -> str:
    try:
        return HARDWARE_FUNCTIONS[hardware]
    except KeyError:
        raise ValueError(f"no Modal function for hardware {hardware!r}; known: {sorted(HARDWARE_FUNCTIONS)}") from None


# =========================================================================== joining both, per arm


@dataclass
class _Sample:
    ts: float
    util: float | None = None
    mem: float | None = None


@dataclass
class ArmGpuTracker:
    """Latest in-container GPU sample per run in flight, per arm (fed from ``metric`` events
    named ``gpu_util_pct`` / ``gpu_mem_mb``; a run is dropped at ``experiment_finished``).

    ``summary(arm, containers, calls_in_flight)`` (the last two from Modal):

    - ``gpu_util_pct``: mean over the containers whose state we know: every run with a live sample
      at its measured %, every warm container with no call in flight at 0 % (nothing of ours runs
      there). Runs in flight without a sample yet (just started, or telemetry not reaching this
      process) are left out and counted in ``runs_unsampled``; if calls are in flight but none is
      sampled the number is unknown (None), never a made-up 0. This is the 'how busy is this
      arm's paid GPU fleet' number.
    - ``gpu_util_active_pct``: mean over sampled runs only (how well a running experiment uses its GPU).
    - ``idle_containers`` here is ``containers - max(calls_in_flight, runs_sampled)``, so Modal's
      stats lagging behind our samples never counts a busy container as idle."""

    stale_s: float = SAMPLE_STALE_S
    _runs: dict[str, dict[str, _Sample]] = field(default_factory=dict)   # arm -> experiment -> sample

    def observe(self, arm: str, experiment_id: str, name: str, value: float, ts: float) -> None:
        if name not in (GPU_UTIL_METRIC, GPU_MEM_METRIC) or not math.isfinite(float(value)):
            return
        s = self._runs.setdefault(arm, {}).setdefault(experiment_id, _Sample(ts=ts))
        s.ts = max(s.ts, ts)
        if name == GPU_UTIL_METRIC:
            s.util = float(value)
        else:
            s.mem = float(value)

    def finish(self, arm: str, experiment_id: str) -> None:
        self._runs.get(arm, {}).pop(experiment_id, None)

    def live(self, arm: str, now: float) -> dict[str, _Sample]:
        runs = self._runs.get(arm, {})
        for eid in [e for e, s in runs.items() if now - s.ts > self.stale_s]:
            runs.pop(eid, None)
        return dict(runs)

    def summary(self, arm: str, containers: int | None, calls_in_flight: int | None,
                now: float) -> dict[str, Any]:
        live = self.live(arm, now)
        utils = [s.util for s in live.values() if s.util is not None]
        mems = [s.mem for s in live.values() if s.mem is not None]
        n = len(utils)
        fleet: float | None = None
        unsampled = 0
        if containers is not None and calls_in_flight is not None:
            running = max(int(calls_in_flight), n)
            idle = max(0, int(containers) - running)
            unsampled = running - n
            denom = n + idle
            if denom > 0 and not (n == 0 and unsampled > 0):
                fleet = sum(utils) / denom
        return {"gpu_util_pct": fleet, "gpu_util_active_pct": (sum(utils) / n) if n else None,
                "gpu_mem_mb": sum(mems) if mems else None, "runs_sampled": n, "runs_unsampled": unsampled}


def calls_in_flight_by_arm(experiments: Iterable[Mapping[str, Any]],
                           arm_kind_by_id: Mapping[str, str]) -> dict[str, list[dict[str, Any]]]:
    """Modal call id (``run_ref``) of every run per arm, from stored experiment rows (newest
    last). ``in_flight`` marks status 'running'. Rows without a run_ref (never launched) are skipped."""
    out: dict[str, list[dict[str, Any]]] = {}
    for e in experiments:
        ref = e.get("run_ref")
        if not ref:
            continue
        arm = arm_kind_by_id.get(str(e.get("arm_id")), str(e.get("arm_id")))
        out.setdefault(arm, []).append({
            "experiment_id": e.get("id"), "modal_call_id": ref, "status": e.get("status"),
            "in_flight": e.get("status") == "running", "slot_id": e.get("slot_id"),
            "started_at": e.get("started_at"), "ended_at": e.get("ended_at"), "hardware": e.get("hardware"),
        })
    return out


def latest_call_id(calls: Iterable[Mapping[str, Any]]) -> str | None:
    """The Modal call id of the most recently started run (dashboard 'Latest Modal call')."""
    started = [c for c in calls if c.get("modal_call_id") and c.get("started_at") is not None]
    if not started:
        return None
    return str(max(started, key=lambda c: float(c["started_at"]))["modal_call_id"])
