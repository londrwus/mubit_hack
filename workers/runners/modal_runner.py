"""ModalRunner: runs targets as Modal Functions (not Sandboxes). Test/demo compute only.

This is the ONLY module that imports ``modal`` (CLAUDE.md). It is generic over the target
registry: a job names a target, the container imports it via ``get_target`` and runs it.

Layout
- One Modal Function per hardware type: ``run_l4`` (gpu="L4": lm, cifar) and ``run_cpu``
  (CPU containers: rl). A target runs on ``TargetSpec.hardware`` unless ``hardware_for``
  overrides it. Never H100 here (the Sapling verifier lives elsewhere).
- Two images, both with the local ``workers`` and ``backend`` sources (runners import
  ``backend.core.types``): ``gpu_image`` (torch + torchvision + numpy + pydantic) for L4 and
  ``cpu_image`` (jax + optax + numpy + pydantic, the RL target's pins) for CPU. A target
  whose ML libs are not in its hardware's image fails at import inside the container.
- Datasets are cached in the Modal Volume ``busy-machines-data`` mounted at ``/data``
  (``BUSY_DATA_DIR``; LM corpus under ``LM_DATA_DIR=/data/lm``). ``prepare_data(target)``
  downloads a target's data into it once (cifar and lm have a ``prepare`` hook).
- Metrics stream back through the named ``modal.Queue`` ``busy-machines-metrics``, one
  partition per run (``run_id``). The host polls it and forwards each point to ``on_metric``.
- ``RunHandle.run_ref`` is the Modal ``FunctionCall.object_id``.
- Slot caps are enforced on the host (per runner, and globally across runner instances:
  Starter allows 10 GPUs / 100 containers), so launch() raises RunnerCapacityError instead
  of letting Modal queue silently.
- Pre-warm (optimisation_analysis.md section 1a, D32): ``prewarm(hw, n)`` pins ``n`` warm
  containers (``min_containers``) and ``warm_up(hw, n)`` blocks until ``n`` *distinct*
  containers have answered a warm-up job (ML libs imported, JAX/torch backend initialised),
  so a race can start its clock only once every slot is warm. ``cooldown(hw)`` unpins them.
  Warm-up jobs are not runs: they never touch slot reservations, metrics or race $.
- Per-run overhead cuts (same for every arm): the Volume is reloaded only for targets
  that read data from it (a ``prepare`` hook), the metrics Queue handle is cached per
  container, and the host clears a run's queue partition in the background after ``wait``.
- No Modal await can hold a slot forever: every host-side Modal call (result get, cancel,
  queue get/clear) goes through ``_bounded``, which gives up after a timeout and abandons
  the call instead of awaiting its cancellation. At 64 concurrent runs (rl-wide-15m, 24 Oct
  ~02:45 BST) some ``wait()`` calls never returned although the remote call had finished
  (checked with ``FunctionCall.get`` from another process), and cancelling them hung the
  race shutdown. The metrics poller also uses short non-blocking reads instead of
  long-polls, so it does not hold a gRPC stream per run.
- Cancel accounting (fairness review, D41): ``cancel()`` returns at once, but the slot stays
  reserved and ``wait()`` returns only when Modal confirms the input stopped (``get`` raises
  the cancellation, or returns if the run finished first), at most ``cancel_grace_s`` later
  (then the container is terminated). So an early-stopping arm pays for the cancel lag and
  never runs more containers than its slots. ``RunResult.metrics`` carries ``cancel_lag_s`` /
  ``cancel_confirmed`` (hidden from researchers).

Usage
- Deployed (backend): ``modal deploy workers/runners/modal_runner.py`` once, then
  ``ModalRunner(deployed=True)`` looks the functions up by name.
- Ephemeral (scripts/smoke tests): ``async with app.run(): ModalRunner(deployed=False)``.
- Smoke test: ``python -m workers.runners.modal_runner smoke [cifar|lm|rl] ['{json overrides}'] [budget_s]``
  (one short run of the default config; cifar/lm on one L4 ~ $0.01, rl on CPU ~ $0.001).

Per-arm apps (DEMO ONLY, ``RaceConfig.modal_apps``; e.g. the ``demo-cifar-modal-5m`` preset)
- The same three functions (``run_l4``, ``run_cpu``, ``prepare_data``) deployed under one app per
  arm (``PER_ARM_APP_NAMES``: opus -> ``busy-machines-opus``, opus_bm -> ``busy-machines-bm``) by
  setting ``BUSY_MODAL_APP`` at deploy time. Each arm's runner is ``ModalRunner(app_name=...)`` and
  looks its functions up with ``modal.Function.from_name(app_name, fn)``, so Modal's own dashboard
  shows each arm's containers and GPU use on its own app page. The Volume and the metrics Queue
  are workspace objects shared by both apps (runs use distinct queue partitions). Benchmark and
  ablation races never use this (configs refuses it): they keep the single default app.
- Deploy both: ``python -m workers.runners.modal_runner arm-apps deploy`` (runs, per app,
  ``BUSY_MODAL_APP=<name> modal deploy --name <name> workers/runners/modal_runner.py``).
- Tear down both: ``python -m workers.runners.modal_runner arm-apps stop`` (``modal app stop -y <name>``).
- ``arm-apps print`` only prints the commands and the dashboard pages (no Modal call).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import queue as stdlib_queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import modal

from backend.core.types import ExperimentConfig, ExperimentStatus, MetricPoint, RunHandle, RunResult, Slot
from workers.runners.base import (
    DEFAULT_GRACE_S,
    DEFAULT_PRICE_PER_SECOND,
    MetricCallback,
    RunnerCapacityError,
    RunnerError,
    emit_metric,
    price_from_table,
)
from workers.runners.local import execute_job, result_from_payload

RPC_TIMEOUT_S = 10.0       # hard cap on any single host-side Modal call (get/cancel/queue)


async def _bounded(aw: Any, timeout: float, keep: set[asyncio.Task[Any]] | None = None) -> Any:
    """Await ``aw`` for at most ``timeout`` s, else raise TimeoutError. A call that overruns is
    cancelled best-effort and abandoned, never awaited: a hung Modal aio call may not honour
    cancellation (seen at 64 concurrent runs), and awaiting it would hang the caller too."""
    task = asyncio.ensure_future(aw)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except asyncio.CancelledError:
        task.cancel()
        raise
    if not done:
        task.cancel()
        if keep is not None:
            keep.add(task)
            task.add_done_callback(keep.discard)
        raise TimeoutError(f"Modal call did not return within {timeout:.1f}s")
    return task.result()

DEFAULT_APP_NAME = "busy-machines-runner"   # every benchmark / ablation race (single app, unchanged)
APP_NAME = os.environ.get("BUSY_MODAL_APP", DEFAULT_APP_NAME)
# Per-arm apps for demo races only (backend.benchmark.configs.PER_ARM_MODAL_APPS mirrors this; a
# test checks it): the same functions deployed once per arm so Modal's dashboard shows each arm's
# containers / GPU use on its own app page. ``arm-apps`` in the CLI below deploys and stops them.
PER_ARM_APP_NAMES: dict[str, str] = {"opus": "busy-machines-opus", "opus_bm": "busy-machines-bm"}
VOLUME_NAME = "busy-machines-data"
QUEUE_NAME = "busy-machines-metrics"
DATA_MOUNT = "/data"
SCALEDOWN_WINDOW_S = int(os.environ.get("BUSY_MODAL_SCALEDOWN_S", "60"))  # idle warm time (billed)
# Cores reserved per CPU container. 1 keeps the RL race inside budget.md's ~$3 (32 slots x 3 arms
# x 30 min = $2.26 per seed; 2 cores doubles it). rl.py's one Modal check used cpu=2.0, so the RL
# target_value must be calibrated at whatever value is frozen here. Both arms get the same value:
# it is a fairness constant, never change it mid-benchmark.
CPU_CORES = float(os.environ.get("BUSY_MODAL_CPU_CORES", "1"))
CPU_MEMORY_MB = int(os.environ.get("BUSY_MODAL_CPU_MEMORY_MB", "2048"))
FUNCTION_TIMEOUT_S = 900  # hard per-call cap on Modal; the host times out much earlier

STARTER_MAX_GPUS = 10
STARTER_MAX_CONTAINERS = 100

# hardware -> Modal function name. Add an entry (and a function below) for new hardware.
HARDWARE_FUNCTIONS: dict[str, str] = {"L4": "run_l4", "CPU": "run_cpu"}

# L4 image: torch targets (lm, cifar). CPU image: the JAX RL target (versions from rl.py's
# Modal check). gymnax is dev/test-only and stays out of both.
IMAGE_PIP_PACKAGES: list[str] = ["torch==2.14.1", "torchvision", "numpy", "pydantic>=2"]
CPU_IMAGE_PIP_PACKAGES: list[str] = [
    "jax==0.12.0.1", "jaxlib==0.12.0.1", "optax==0.2.8", "numpy==2.5.4", "pydantic>=2",
]
IMAGE_ENV: dict[str, str] = {
    "BUSY_DATA_DIR": DATA_MOUNT,
    "LM_DATA_DIR": f"{DATA_MOUNT}/lm",
    "PYTHONUNBUFFERED": "1",
}


def _image(packages: list[str]) -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(*packages)
        .env(IMAGE_ENV)
        .add_local_python_source("workers", "backend")
    )


gpu_image = _image(IMAGE_PIP_PACKAGES)
cpu_image = _image(CPU_IMAGE_PIP_PACKAGES)
image = gpu_image  # back-compat name

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# --------------------------------------------------------------------------- container side


class _QueuePusher:
    """Batches metric points into the run's queue partition from a background thread, so
    a slow put never stalls the training loop."""

    def __init__(self, q: Any, partition: str, flush_every_s: float = 0.25) -> None:
        self._q = q
        self._partition = partition
        self._buf: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._every = flush_every_s
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def put(self, item: dict[str, Any]) -> None:
        with self._lock:
            self._buf.append(item)

    def emit(self, step: int, t: float, name: str, value: float) -> None:
        self.put({"type": "metric", "step": step, "t": t, "name": name, "value": value})

    def _flush(self) -> None:
        with self._lock:
            items, self._buf = self._buf, []
        if items:
            try:
                self._q.put_many(items, partition=self._partition, partition_ttl=6 * 3600)
            except Exception as exc:  # noqa: BLE001 (losing a metric must not kill training)
                print(f"[modal_runner] metric push failed: {exc!r}", file=sys.stderr)

    def _loop(self) -> None:
        while not self._stop.wait(self._every):
            self._flush()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._flush()


_CONTAINER_QUEUES: dict[str, Any] = {}  # queue name -> handle, reused by every run in this container
_CONTAINER_TOKEN = f"{os.getpid()}-{time.time_ns()}"  # identity of this container process

# Modules a warm-up job imports per hardware (the image's ML libs), so a run never pays them.
WARMUP_IMPORTS: dict[str, list[str]] = {
    "CPU": ["numpy", "jax", "jax.numpy", "optax"],
    "L4": ["numpy", "torch", "torchvision"],
}


def _container_queue(name: str) -> Any:
    q = _CONTAINER_QUEUES.get(name)
    if q is None:
        q = _CONTAINER_QUEUES[name] = modal.Queue.from_name(name, create_if_missing=True)
    return q


def _needs_volume(target: str, extra_imports: list[str] | None = None) -> bool:
    """Only targets with a ``prepare`` hook read the Volume (cifar, lm); rl does not."""
    try:
        from workers.runners.local import import_modules
        from workers.targets.base import get_target

        import_modules(extra_imports)
        return getattr(get_target(target), "prepare", None) is not None
    except Exception:  # noqa: BLE001 (unknown target: reload, as before; execute_job reports the error)
        return True


def warmup_payload(job: dict[str, Any]) -> dict[str, Any]:
    """Body of a warm-up job (no run, no metrics): import the hardware's ML libs and the
    target, initialise the array backend, then hold the container for ``hold_s`` so that
    concurrent warm-up jobs land on distinct containers. Returns the container identity."""
    import importlib

    t0 = time.time()
    hw = str(job.get("hardware") or "CPU")
    imported: list[str] = []
    for m in list(job.get("imports") or WARMUP_IMPORTS.get(hw, [])):
        with contextlib.suppress(Exception):
            importlib.import_module(m)
            imported.append(m)
    target = job.get("target")
    if target:
        with contextlib.suppress(Exception):
            from workers.runners.local import import_modules
            from workers.targets.base import get_target

            import_modules(job.get("extra_imports"))
            get_target(str(target))
    backend = None
    with contextlib.suppress(Exception):
        if "jax" in imported:
            import jax.numpy as jnp

            jnp.zeros(1).block_until_ready()  # JAX backend init (outside any run's budget)
            backend = "jax"
        elif "torch" in imported:
            import torch

            if torch.cuda.is_available():
                torch.zeros(1, device="cuda").sum().item()
            backend = "torch"
    if target and _needs_volume(str(target), job.get("extra_imports")):
        with contextlib.suppress(Exception):
            volume.reload()
    import_s = time.time() - t0
    hold = max(0.0, float(job.get("hold_s") or 0.0) - import_s)
    if hold:
        time.sleep(hold)
    return {
        "warmup": True, "container_id": os.environ.get("MODAL_TASK_ID") or _CONTAINER_TOKEN,
        "container_token": _CONTAINER_TOKEN, "imported": imported, "backend": backend,
        "import_s": import_s, "cpu_count": _usable_cpus(), "started_at": t0, "ended_at": time.time(),
    }


def _usable_cpus() -> int | None:
    """CPUs this process may run on (what ``nproc`` prints), not the host's core count."""
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return os.cpu_count()


def _remote_execute(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("warmup"):
        return warmup_payload(job)
    started = time.time()
    if _needs_volume(job["target"], job.get("extra_imports")):
        with contextlib.suppress(Exception):
            volume.reload()  # see data written by prepare_data after this container started
    q = _container_queue(job.get("queue_name") or QUEUE_NAME)
    pusher = _QueuePusher(q, job["run_id"])
    pusher.put({"type": "started", "ts": started})
    try:
        payload = execute_job(job, pusher.emit)
    finally:
        pusher.close()
    payload["container_started_at"] = started
    return payload


_COMMON_FN_KW: dict[str, Any] = dict(
    volumes={DATA_MOUNT: volume},
    timeout=FUNCTION_TIMEOUT_S,
    scaledown_window=SCALEDOWN_WINDOW_S,
)


@app.function(image=gpu_image, gpu="L4", max_containers=STARTER_MAX_GPUS, **_COMMON_FN_KW)
def run_l4(job: dict[str, Any]) -> dict[str, Any]:
    return _remote_execute(job)


@app.function(image=cpu_image, cpu=CPU_CORES, memory=CPU_MEMORY_MB, max_containers=STARTER_MAX_CONTAINERS,
              **_COMMON_FN_KW)
def run_cpu(job: dict[str, Any]) -> dict[str, Any]:
    return _remote_execute(job)


@app.function(image=gpu_image, volumes={DATA_MOUNT: volume}, cpu=2, timeout=1800)
def prepare_data(target: str, extra_imports: list[str] | None = None) -> dict[str, Any]:
    """Download a target's dataset into the Volume (targets with a ``prepare`` hook)."""
    import importlib

    from workers.targets.base import get_target

    for m in extra_imports or []:
        importlib.import_module(m)
    t0 = time.time()
    t = get_target(target)
    prep = getattr(t, "prepare", None)
    if prep is None:
        return {"target": target, "prepared": False, "seconds": 0.0}
    path = prep(None)
    volume.commit()
    size = os.path.getsize(path) if isinstance(path, str) and os.path.exists(path) else 0
    return {"target": target, "prepared": True, "path": path, "bytes": size, "seconds": time.time() - t0}


# --------------------------------------------------------------------------- host side


@dataclass
class _ModalRun:
    handle: RunHandle
    call: Any | None
    on_metric: MetricCallback
    started_ts: float | None = None       # container start (epoch s), from the queue
    cancelled: bool = False
    cancelled_at: float | None = None     # host time cancel() was called
    cancel_confirmed_at: float | None = None  # host time Modal reported the input over
    poller: asyncio.Task[None] | None = None
    result: RunResult | None = None
    precomputed: RunResult | None = None  # e.g. invalid config: never spawned
    metrics_seen: int = 0
    stop_polling: bool = False
    notes: list[str] = field(default_factory=list)


class ModalRunner:
    """Runner backed by Modal Functions.

    slots: per-hardware concurrency for this runner, e.g. {"L4": 4, "CPU": 32}.
    app_name: the deployed app to look functions up in (default ``APP_NAME``, the single app;
        demo races with per-arm apps pass the arm's app, see ``PER_ARM_APP_NAMES``).
    deployed: True looks functions up in the deployed app ``app_name``; False uses this
        module's functions (call inside ``async with app.run():``).
    price_table: USD/s per hardware (CPU is per core and is multiplied by ``cpu_cores``).
    grace_s: allowed wall-clock beyond budget_s, counted from container start.
    cancel_grace_s: after ``cancel()``, how long the slot stays reserved waiting for Modal to
        confirm the input stopped before the container is terminated (default ``grace_s``).
    cancel_terminates: ``cancel()`` also terminates the container (default False: the input
        is cancelled and the warm container is reused; see ``_wait_call`` for the measurement).
    startup_timeout_s: max wait for a container to start (cold start / scheduling).
    function_resolver / queue: injection points for tests (no network).
    """

    name = "modal"

    # Busy slots per hardware across all ModalRunner instances in this process (both arms
    # share one Modal workspace, so the Starter caps are global).
    _global_busy: ClassVar[dict[str, int]] = {}

    def __init__(
        self,
        slots: dict[str, int] | None = None,
        *,
        app_name: str = APP_NAME,
        deployed: bool = True,
        max_gpus: int = STARTER_MAX_GPUS,
        max_containers: int = STARTER_MAX_CONTAINERS,
        price_table: dict[str, float] | None = None,
        cpu_cores: float = CPU_CORES,
        grace_s: int = DEFAULT_GRACE_S,
        cancel_grace_s: float | None = None,
        cancel_terminates: bool = False,
        startup_timeout_s: float = 300.0,
        poll_interval_s: float = 1.0,
        hardware_for: dict[str, str] | None = None,
        extra_imports: list[str] | None = None,
        queue_name: str = QUEUE_NAME,
        function_resolver: Callable[[str], Any] | None = None,
        queue: Any | None = None,
    ) -> None:
        self._slots = dict(slots if slots is not None else {"L4": 4})
        for hw, n in self._slots.items():
            if hw not in HARDWARE_FUNCTIONS:
                raise ValueError(f"no Modal function for hardware {hw!r}; known: {sorted(HARDWARE_FUNCTIONS)}")
            if n < 0:
                raise ValueError("slot counts must be >= 0")
        gpus = sum(n for hw, n in self._slots.items() if hw != "CPU")
        if gpus > max_gpus:
            raise ValueError(f"{gpus} GPU slots exceed the cap of {max_gpus}")
        if sum(self._slots.values()) > max_containers:
            raise ValueError(f"{sum(self._slots.values())} slots exceed the cap of {max_containers} containers")
        self.app_name = app_name
        self.deployed = deployed
        self.max_gpus = max_gpus
        self.max_containers = max_containers
        self._price = dict(DEFAULT_PRICE_PER_SECOND if price_table is None else price_table)
        self.cpu_cores = cpu_cores
        self.grace_s = grace_s
        self.cancel_grace_s = float(grace_s if cancel_grace_s is None else cancel_grace_s)
        self.cancel_terminates = bool(cancel_terminates)
        self.startup_timeout_s = startup_timeout_s
        self.poll_interval_s = poll_interval_s
        self.hardware_for = dict(hardware_for or {})
        self.extra_imports = list(extra_imports or [])
        self.queue_name = queue_name
        self._resolver = function_resolver
        self._queue = queue
        self._fn_cache: dict[str, Any] = {}
        self._runs: dict[str, _ModalRun] = {}
        self._busy: dict[str, set[str]] = {hw: set() for hw in self._slots}
        self._bg: set[asyncio.Task[None]] = set()
        self._abandoned: set[asyncio.Task[Any]] = set()  # overran _bounded; never awaited

    # -- Runner protocol -------------------------------------------------------

    def slots(self) -> list[Slot]:
        out: list[Slot] = []
        for hw, n in self._slots.items():
            for i in range(n):
                out.append(Slot(slot_id=f"modal-{hw}-{i}", index=i, hardware=hw, runner=self.name))
        return out

    def busy_slots(self, hardware: str | None = None) -> int:
        if hardware is not None:
            return len(self._busy.get(hardware, ()))
        return sum(len(s) for s in self._busy.values())

    def price_per_second(self, hardware: str) -> float:
        p = price_from_table(hardware, self._price)
        return p * self.cpu_cores if hardware == "CPU" else p

    async def launch(
        self,
        target: str,
        config: ExperimentConfig,
        seed: int,
        budget_s: int,
        on_metric: MetricCallback,
    ) -> RunHandle:
        from workers.targets.base import get_target
        from workers.runners.local import import_modules

        import_modules(self.extra_imports)
        spec = get_target(target).spec
        hw = self.hardware_for.get(target, spec.hardware)
        if hw not in self._slots:
            raise RunnerError(f"target {target!r} needs {hw!r} but this runner has slots for {sorted(self._slots)}")
        self._check_capacity(hw)
        handle = RunHandle(
            runner=self.name, target=target, config=dict(config), seed=int(seed),
            budget_s=int(budget_s), hardware=hw,
        )
        # Reject invalid configs on the host: never pay a container start for them.
        try:
            cfg = spec.validate_config(dict(config))
        except ValueError as exc:
            run = _ModalRun(handle=handle, call=None, on_metric=on_metric)
            run.precomputed = result_from_payload(
                handle, None, status=ExperimentStatus.failed, error=f"invalid_config: {exc}",
                logs_tail=[f"[modal_runner] invalid config rejected before launch: {exc}"], ended_at=time.time(),
            )
            self._runs[handle.run_id] = run
            return handle
        job = {
            "run_id": handle.run_id, "target": target, "config": cfg, "seed": int(seed),
            "budget_s": int(budget_s), "started_at": handle.started_at,
            "extra_imports": self.extra_imports, "queue_name": self.queue_name,
        }
        self._reserve(hw, handle.run_id)
        try:
            fn = self._function(hw)
            call = await fn.spawn.aio(job)
        except Exception as exc:
            self._release(hw, handle.run_id)
            raise RunnerError(f"Modal spawn failed: {exc!r}") from exc
        handle.run_ref = str(call.object_id)
        run = _ModalRun(handle=handle, call=call, on_metric=on_metric)
        self._runs[handle.run_id] = run
        run.poller = asyncio.create_task(self._poll(run))
        return handle

    async def wait(self, handle: RunHandle) -> RunResult:
        run = self._runs.get(handle.run_id)
        if run is None:
            raise RunnerError(f"unknown run {handle.run_id}")
        if run.result is not None:
            return run.result
        if run.precomputed is not None:
            run.result = run.precomputed
            return run.result
        assert run.call is not None
        try:
            return await self._wait_call(handle, run)
        finally:
            if run.result is None:
                # wait() itself was cancelled (race shutdown): make sure nothing keeps running
                # unseen, and never leak the slot reservation.
                with contextlib.suppress(Exception):
                    self._bg_task(run.call.cancel.aio(terminate_containers=True))
                self._release(handle.hardware, handle.run_id)

    async def _wait_call(self, handle: RunHandle, run: _ModalRun) -> RunResult:
        """Wait for the remote call. After ``cancel()`` the slot stays reserved (and wait()
        does not return) until Modal reports the input over (``get`` raises or returns), at
        most ``cancel_grace_s`` after the cancel; then the container is terminated. Measured
        on the deployed app (24 Oct, 10 rl runs cancelled 5 s after start): the remote input
        kept training for ~2 s after the cancel RPC, while the old code freed the slot after
        ~1 s, so stopped runs used ~1 s of unaccounted container time each."""
        assert run.call is not None
        payload: dict[str, Any] | None = None
        status: ExperimentStatus | None = None
        error: str | None = None
        polled_after_cancel = False   # always give Modal one get() after the cancel to confirm it
        while True:
            now = time.time()
            if run.started_ts is not None:
                deadline = run.started_ts + handle.budget_s + self.grace_s
            else:
                deadline = handle.started_at + self.startup_timeout_s + handle.budget_s + self.grace_s
            if run.cancelled and run.cancelled_at is not None:
                if not polled_after_cancel:
                    deadline = max(now + 0.1, min(deadline, run.cancelled_at + self.cancel_grace_s))
                else:
                    deadline = min(deadline, run.cancelled_at + self.cancel_grace_s)
            if now >= deadline:
                with contextlib.suppress(Exception):
                    await _bounded(run.call.cancel.aio(terminate_containers=True), RPC_TIMEOUT_S, self._abandoned)
                if run.cancelled:
                    status = ExperimentStatus.cancelled
                    error = (f"cancelled; Modal did not confirm the input stopped within "
                             f"{self.cancel_grace_s:g}s, container terminated")
                    run.notes.append(f"[modal_runner] {error}")
                else:
                    status = ExperimentStatus.timed_out
                    error = (f"exceeded budget {handle.budget_s}s + grace {self.grace_s}s"
                             if run.started_ts is not None
                             else f"container did not start within {self.startup_timeout_s}s")
                break
            get_t = min(1.0, max(0.1, deadline - now))
            polled_after_cancel = polled_after_cancel or run.cancelled
            try:
                payload = await _bounded(run.call.get.aio(timeout=get_t), get_t + RPC_TIMEOUT_S, self._abandoned)
                if run.cancelled:
                    # The run finished before the cancel reached it: report the real outcome.
                    run.cancel_confirmed_at = time.time()
                    run.notes.append("[modal_runner] cancel arrived after the run had finished")
                break
            except TimeoutError:
                continue
            except Exception as exc:  # noqa: BLE001 (remote crash, container OOM-kill, cancellation)
                if run.cancelled:
                    run.cancel_confirmed_at = time.time()
                    status, error = ExperimentStatus.cancelled, "cancelled"
                else:
                    status, error = ExperimentStatus.failed, f"modal: {type(exc).__name__}: {exc}"
                    # Make sure no container is still training before the slot is released:
                    # an orphan would push the workspace past the slot/GPU cap and bill unseen $.
                    with contextlib.suppress(Exception):
                        await _bounded(run.call.cancel.aio(terminate_containers=True), RPC_TIMEOUT_S,
                                       self._abandoned)
                break
        ended = time.time()
        await self._finish_polling(run)
        if isinstance(payload, dict) and payload.get("container_started_at") and run.started_ts is None:
            run.started_ts = float(payload["container_started_at"])
        result = result_from_payload(
            handle, payload if status is None else None, status=status, error=error,
            logs_tail=None if (status is None and payload) else run.notes, ended_at=ended,
        )
        if run.started_ts is not None:
            result.metrics["modal_startup_s"] = max(0.0, run.started_ts - handle.started_at)
        if run.cancelled and run.cancelled_at is not None:
            # Runner accounting (hidden from researchers, scheduler.RUNNER_ACCOUNTING_METRIC_KEYS):
            # cancel() -> Modal confirmed the input over; the slot was busy all that time.
            result.metrics["cancel_lag_s"] = max(0.0, ended - run.cancelled_at)
            result.metrics["cancel_confirmed"] = 1.0 if run.cancel_confirmed_at is not None else 0.0
        run.result = result
        self._release(handle.hardware, handle.run_id)
        return result

    async def cancel(self, handle: RunHandle) -> None:
        """Ask Modal to stop the input. Returns at once; the slot stays reserved until
        ``wait()`` sees Modal confirm the stop (see ``_wait_call``)."""
        run = self._runs.get(handle.run_id)
        if run is None or run.result is not None or run.cancelled:
            return
        run.cancelled = True
        run.cancelled_at = time.time()
        if run.call is not None:
            with contextlib.suppress(Exception):
                await _bounded(run.call.cancel.aio(terminate_containers=self.cancel_terminates), RPC_TIMEOUT_S,
                               self._abandoned)

    def _bg_task(self, aw: Any) -> None:
        task = asyncio.ensure_future(_bounded(aw, RPC_TIMEOUT_S, self._abandoned))
        self._bg.add(task)
        task.add_done_callback(lambda t: (self._bg.discard(t), t.cancelled() or t.exception()))

    # -- extras ----------------------------------------------------------------

    async def prewarm(self, hardware: str = "L4", n: int | None = None, *, scaledown_window: int | None = None) -> None:
        """Keep ``n`` containers of ``hardware`` warm (billed while warm!). Call
        ``cooldown(hardware)`` when done. Needs a deployed or running app."""
        n = self._slots.get(hardware, 0) if n is None else n
        if hardware != "CPU" and n > self.max_gpus:
            raise ValueError(f"cannot prewarm {n} GPUs (cap {self.max_gpus})")
        if n > self.max_containers:
            raise ValueError(f"cannot prewarm {n} containers (cap {self.max_containers})")
        fn = self._function(hardware)
        kw: dict[str, Any] = {"min_containers": n}
        if scaledown_window is not None:
            kw["scaledown_window"] = scaledown_window
        await fn.update_autoscaler.aio(**kw)

    async def cooldown(self, hardware: str = "L4", *, reset_scaledown: bool = False) -> None:
        """Unpin warm containers (they scale down after the scaledown window). With
        ``reset_scaledown`` also restore the deployed ``SCALEDOWN_WINDOW_S``."""
        fn = self._function(hardware)
        kw: dict[str, Any] = {"min_containers": 0}
        if reset_scaledown:
            kw["scaledown_window"] = SCALEDOWN_WINDOW_S
        await fn.update_autoscaler.aio(**kw)

    async def warm_up(
        self,
        hardware: str,
        n: int,
        *,
        target: str | None = None,
        hold_s: float = 4.0,
        timeout_s: float = 300.0,
        max_rounds: int = 8,
        log: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Block until ``n`` distinct containers of ``hardware`` are warm.

        Each round spawns ``n`` concurrent warm-up jobs that import the ML libs (and
        ``target``), initialise the backend and hold their container for ``hold_s``, so
        concurrent jobs need distinct containers. Rounds repeat until one round is answered
        by >= ``n`` distinct containers (or ``max_rounds`` / ``timeout_s`` run out). Pair
        with ``prewarm(hardware, n)`` first so the containers stay up afterwards.
        Warm-up jobs never use slots, so they never count as busy/idle time or race $."""
        if n <= 0:
            return {"hardware": hardware, "wanted": 0, "warm": 0, "rounds": [], "ok": True, "seconds": 0.0}
        fn = self._function(hardware)
        t_start = time.time()
        rounds: list[dict[str, Any]] = []
        seen: set[str] = set()
        warm = 0
        for i in range(max_rounds):
            if time.time() - t_start > timeout_s:
                break
            job = {"warmup": True, "hardware": hardware, "target": target, "hold_s": hold_s,
                   "extra_imports": self.extra_imports}
            t_r = time.time()
            calls = await asyncio.gather(*(fn.spawn.aio(job) for _ in range(n)), return_exceptions=True)
            left = max(5.0, timeout_s - (time.time() - t_start))

            async def _get(c: Any) -> Any:
                if isinstance(c, BaseException):
                    return c
                try:
                    return await asyncio.wait_for(c.get.aio(), timeout=left)
                except BaseException as exc:  # noqa: BLE001
                    with contextlib.suppress(Exception):
                        await c.cancel.aio(terminate_containers=False)
                    return exc

            outs = await asyncio.gather(*(_get(c) for c in calls))
            ids = [o.get("container_id") for o in outs if isinstance(o, dict) and o.get("warmup")]
            errors = [repr(o)[:200] for o in outs if not (isinstance(o, dict) and o.get("warmup"))]
            distinct = set(i for i in ids if i)
            seen |= distinct
            imp = sorted(float(o.get("import_s") or 0.0) for o in outs if isinstance(o, dict) and o.get("warmup"))
            rounds.append({"round": i, "seconds": time.time() - t_r, "answered": len(ids),
                           "distinct_containers": len(distinct), "errors": errors[:3], "n_errors": len(errors),
                           "import_s_max": imp[-1] if imp else None,
                           "cpu_count": sorted({o.get("cpu_count") for o in outs if isinstance(o, dict)} - {None})})
            if log is not None:
                log(f"[warm_up] {hardware} round {i}: {len(distinct)}/{n} distinct containers "
                    f"in {time.time() - t_r:.1f}s ({len(errors)} errors)")
            warm = len(distinct)
            if warm >= n:
                break
        return {"hardware": hardware, "wanted": n, "warm": warm, "ok": warm >= n, "rounds": rounds,
                "containers_seen": len(seen), "seconds": time.time() - t_start, "hold_s": hold_s}

    async def prepare_data(self, target: str) -> dict[str, Any]:
        """Download ``target``'s dataset into the Volume (CPU container, cheap)."""
        fn = self._resolve("prepare_data")
        return await fn.remote.aio(target, self.extra_imports)

    # -- internals ---------------------------------------------------------------

    def _resolve(self, fn_name: str) -> Any:
        if fn_name in self._fn_cache:
            return self._fn_cache[fn_name]
        if self._resolver is not None:
            fn = self._resolver(fn_name)
        elif self.deployed:
            fn = modal.Function.from_name(self.app_name, fn_name)
        else:
            fn = globals()[fn_name]
        self._fn_cache[fn_name] = fn
        return fn

    def _function(self, hardware: str) -> Any:
        try:
            return self._resolve(HARDWARE_FUNCTIONS[hardware])
        except KeyError:
            raise RunnerError(f"no Modal function for hardware {hardware!r}") from None

    def _q(self) -> Any:
        if self._queue is None:
            self._queue = modal.Queue.from_name(self.queue_name, create_if_missing=True)
        return self._queue

    def _check_capacity(self, hw: str) -> None:
        if len(self._busy[hw]) >= self._slots[hw]:
            raise RunnerCapacityError(f"all {self._slots[hw]} {hw} slots busy")
        g = ModalRunner._global_busy
        gpus_busy = sum(n for h, n in g.items() if h != "CPU")
        if hw != "CPU" and gpus_busy >= self.max_gpus:
            raise RunnerCapacityError(f"workspace GPU cap reached ({gpus_busy}/{self.max_gpus})")
        if sum(g.values()) >= self.max_containers:
            raise RunnerCapacityError(f"workspace container cap reached ({self.max_containers})")

    def _reserve(self, hw: str, run_id: str) -> None:
        self._busy[hw].add(run_id)
        ModalRunner._global_busy[hw] = ModalRunner._global_busy.get(hw, 0) + 1

    def _release(self, hw: str, run_id: str) -> None:
        if run_id in self._busy.get(hw, set()):
            self._busy[hw].discard(run_id)
            ModalRunner._global_busy[hw] = max(0, ModalRunner._global_busy.get(hw, 0) - 1)

    async def _handle_items(self, run: _ModalRun, items: list[Any]) -> None:
        for it in items:
            if not isinstance(it, dict):
                continue
            kind = it.get("type")
            if kind == "started":
                run.started_ts = float(it["ts"])
            elif kind == "metric":
                point = MetricPoint(
                    experiment_id=run.handle.run_id, step=int(it["step"]), t=float(it["t"]),
                    name=str(it["name"]), value=float(it["value"]),
                )
                run.metrics_seen += 1
                try:
                    await emit_metric(run.on_metric, point)
                except Exception as exc:  # noqa: BLE001
                    run.notes.append(f"[modal_runner] on_metric failed: {exc!r}")

    async def _poll(self, run: _ModalRun) -> None:
        """Short non-blocking reads every ``poll_interval_s`` (a long-poll per run would hold
        one gRPC stream per run for the whole run)."""
        q = self._q()
        while not run.stop_polling:
            try:
                items = await _bounded(q.get_many.aio(100, block=False, partition=run.handle.run_id),
                                       RPC_TIMEOUT_S, self._abandoned)
            except asyncio.CancelledError:
                raise
            except stdlib_queue.Empty:
                items = []
            except Exception as exc:  # noqa: BLE001 (transient network error / overrun: back off)
                run.notes.append(f"[modal_runner] queue poll error: {exc!r}")
                items = []
            if items:
                await self._handle_items(run, items)
            if not run.stop_polling:
                await asyncio.sleep(self.poll_interval_s)

    async def _finish_polling(self, run: _ModalRun) -> None:
        if run.poller is not None:
            # Stop it with a flag, not cancel(): a read cancelled mid-flight can drop items it
            # already took off the queue. Bounded wait; a poller stuck in a Modal call that
            # ignores cancellation is abandoned.
            run.stop_polling = True
            p = run.poller
            done, _ = await asyncio.wait({p}, timeout=self.poll_interval_s + RPC_TIMEOUT_S + 1.0)
            if not done:
                p.cancel()
                self._abandoned.add(p)
                p.add_done_callback(self._abandoned.discard)
        q = self._q()
        # Drain what the container flushed just before returning.
        for _ in range(20):
            try:
                items = await _bounded(q.get_many.aio(1000, block=False, partition=run.handle.run_id),
                                       RPC_TIMEOUT_S, self._abandoned)
            except Exception:  # noqa: BLE001
                break
            if not items:
                break
            await self._handle_items(run, items)
        # Clearing the partition is housekeeping (it also expires after 6 h): do it in the
        # background so the slot is released one RPC sooner.
        task = asyncio.create_task(self._clear_partition(run.handle.run_id))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def _clear_partition(self, run_id: str) -> None:
        with contextlib.suppress(Exception):
            await _bounded(self._q().clear.aio(partition=run_id), RPC_TIMEOUT_S, self._abandoned)

    async def drain_background(self) -> None:
        """Wait for background housekeeping (queue partition clears)."""
        if self._bg:
            await asyncio.gather(*list(self._bg), return_exceptions=True)


# --------------------------------------------------------------------------- smoke test


SMOKE_TARGETS: tuple[str, ...] = ("cifar", "lm", "rl")


async def smoke(
    budget_s: int | None = None,
    config_overrides: dict[str, Any] | None = None,
    seed: int = 0,
    target: str = "cifar",
) -> dict[str, Any]:
    """Ephemeral app: prepare ``target``'s data in the Volume (if it has a hook), then one
    short run of its default config on its own hardware (one L4 for cifar/lm, CPU for rl)."""
    from workers.targets.base import get_target

    spec = get_target(target).spec
    hw = spec.hardware
    budget = int(budget_s if budget_s is not None else spec.budget_s)
    out: dict[str, Any] = {"target": target, "hardware": hw, "budget_s": budget}
    with modal.enable_output():
        async with app.run():
            runner = ModalRunner({hw: 1}, deployed=False)
            t0 = time.time()
            out["prepare"] = await runner.prepare_data(target)
            out["prepare_wall_s"] = time.time() - t0
            cfg = spec.default_config()
            cfg.update(config_overrides or {})
            points: list[MetricPoint] = []
            h = await runner.launch(target, cfg, seed, budget, points.append)
            r = await runner.wait(h)
            out.update(
                run_ref=h.run_ref, status=r.status.value, score=r.score, error=r.error,
                duration_s=r.duration_s, metrics=r.metrics, metric_points=len(points),
                first_metrics=[p.model_dump() for p in points[:4]], logs_tail=r.logs_tail[-12:],
                est_compute_usd=r.duration_s * runner.price_per_second(hw), config=cfg,
            )
    return out


def _smoke_cli(argv: list[str]) -> dict[str, Any]:
    """argv after "smoke": [target] [json overrides] [budget_s]; target defaults to cifar."""
    import json

    args = list(argv)
    target = args.pop(0) if args and args[0] in SMOKE_TARGETS else "cifar"
    overrides = json.loads(args.pop(0)) if args and args[0].lstrip().startswith("{") else {}
    budget = int(args.pop(0)) if args else None
    return asyncio.run(smoke(budget, overrides, target=target))


# --------------------------------------------------------------------------- per-arm apps (demo)

ARM_APP_ACTIONS: tuple[str, ...] = ("print", "deploy", "stop")
DASHBOARD_URL = "https://modal.com/apps/{workspace}/main/deployed/{app}"


def arm_app_commands(action: str, apps: dict[str, str] | None = None,
                     module_path: str = "workers/runners/modal_runner.py") -> list[tuple[dict[str, str], list[str]]]:
    """(extra env, argv after ``python -m modal``) per per-arm app, for ``deploy`` or ``stop``.
    Deploy sets ``BUSY_MODAL_APP`` (so ``APP_NAME`` is the arm's app) and passes ``--name`` too."""
    apps = dict(PER_ARM_APP_NAMES if apps is None else apps)
    if action == "deploy":
        return [({"BUSY_MODAL_APP": name}, ["deploy", "--name", name, module_path]) for name in apps.values()]
    if action == "stop":
        return [({}, ["app", "stop", "-y", name]) for name in apps.values()]
    raise ValueError(f"unknown action {action!r}; known: deploy, stop")


def arm_apps_text(apps: dict[str, str] | None = None) -> str:
    """The documented commands (PowerShell and bash) and dashboard pages; no Modal call."""
    apps = dict(PER_ARM_APP_NAMES if apps is None else apps)
    lines = ["# Deploy one app per arm (same functions; run from the repo root):"]
    for env, argv in arm_app_commands("deploy", apps):
        name = env["BUSY_MODAL_APP"]
        lines.append(f"$env:BUSY_MODAL_APP='{name}'; python -m uv run modal {' '.join(argv)}; "
                     "Remove-Item Env:BUSY_MODAL_APP   # PowerShell")
        lines.append(f"BUSY_MODAL_APP={name} uv run modal {' '.join(argv)}   # bash")
    lines.append("# Tear down both (stops the apps and terminates their containers):")
    for _, argv in arm_app_commands("stop", apps):
        lines.append(f"python -m uv run modal {' '.join(argv)}")
    lines.append("# Dashboard pages (the deploy prints the exact 'View Deployment' URL):")
    for arm, name in apps.items():
        lines.append(f"{arm}: {DASHBOARD_URL.format(workspace='<workspace>', app=name)}")
    return "\n".join(lines)


def _arm_apps_cli(argv: list[str]) -> int:
    """``arm-apps print|deploy|stop``: deploy / stop run the Modal CLI once per app."""
    import subprocess

    action = argv[0] if argv else "print"
    if action not in ARM_APP_ACTIONS:
        print(f"usage: python -m workers.runners.modal_runner arm-apps [{'|'.join(ARM_APP_ACTIONS)}]")
        return 2
    if action == "print":
        print(arm_apps_text())
        return 0
    rc = 0
    for env, args in arm_app_commands(action):
        cmd = [sys.executable, "-m", "modal", *args]
        print("$ " + " ".join([*(f"{k}={v}" for k, v in env.items()), "modal", *args]), flush=True)
        r = subprocess.run(cmd, env={**os.environ, **env}, check=False)
        rc = rc or r.returncode
    return rc


if __name__ == "__main__":
    import json

    if len(sys.argv) >= 2 and sys.argv[1] == "smoke":
        print(json.dumps(_smoke_cli(sys.argv[2:]), indent=2, default=str))
    elif len(sys.argv) >= 2 and sys.argv[1] == "arm-apps":
        raise SystemExit(_arm_apps_cli(sys.argv[2:]))
    else:
        print("usage: python -m workers.runners.modal_runner smoke [cifar|lm|rl] ['{\"epochs\": 8}'] [budget_s]\n"
              "       python -m workers.runners.modal_runner arm-apps [print|deploy|stop]   (demo: one app per arm)")
