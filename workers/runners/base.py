"""Runner protocol: the compute-agnostic interface (architecture.md section 4).

Rules:
- Only runners import compute SDKs; ``modal`` only in workers/runners/modal_runner.py.
- A runner enforces its own slot cap: launch() raises RunnerCapacityError instead of
  queueing silently (queued time would show up as fake GPU idle time).
- wait() never raises for training failures; it returns RunResult(status=failed, error=...).
  It raises RunnerError only for infrastructure problems it cannot represent.
- on_metric receives MetricPoint(experiment_id=handle.run_id, ...). It may be sync or
  async; call it through ``emit_metric`` so both work.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Protocol, TypeAlias, runtime_checkable

from backend.core.types import ExperimentConfig, MetricPoint, RunHandle, RunResult, Slot

MetricCallback: TypeAlias = Callable[[MetricPoint], Awaitable[None] | None]

# Modal Starter list prices, USD per second (budget.md / architecture.md "Cost accounting").
# A customer would configure their own $/GPU-hour.
DEFAULT_PRICE_PER_SECOND: dict[str, float] = {
    "L4": 0.000222,
    "T4": 0.59 / 3600,
    "A10": 1.10 / 3600,
    "L40S": 1.95 / 3600,
    "A100-80GB": 2.50 / 3600,
    "H100": 0.001097,
    "CPU": 0.0000131,  # per core
}

# Extra wall-clock a runner allows beyond budget_s (startup, eval) before timing out.
DEFAULT_GRACE_S: int = 60


class RunnerError(RuntimeError):
    """Infrastructure failure (not a training failure)."""


class RunnerCapacityError(RunnerError):
    """launch() called while every slot is busy."""


@runtime_checkable
class Runner(Protocol):
    name: str  # "local" | "modal" | later "slurm" | "k8s" | "ray"

    def slots(self) -> list[Slot]:
        """Available GPU/CPU slots (the concurrency cap)."""
        ...

    async def launch(
        self,
        target: str,
        config: ExperimentConfig,
        seed: int,
        budget_s: int,
        on_metric: MetricCallback,
    ) -> RunHandle:
        """Start a run and return immediately (do not wait for training)."""
        ...

    async def wait(self, handle: RunHandle) -> RunResult:
        """Block until the run ends; return its result."""
        ...

    async def cancel(self, handle: RunHandle) -> None:
        """Stop a run. Idempotent; a later wait() returns status=cancelled."""
        ...

    def price_per_second(self, hardware: str) -> float:
        """USD per second for one slot of ``hardware`` (for cost accounting)."""
        ...


async def emit_metric(cb: MetricCallback, point: MetricPoint) -> None:
    """Invoke a sync or async metric callback."""
    res = cb(point)
    if inspect.isawaitable(res):
        await res


def price_from_table(hardware: str, table: dict[str, float] | None = None) -> float:
    """Look up a per-second price; raises KeyError for unknown hardware."""
    t = DEFAULT_PRICE_PER_SECOND if table is None else table
    return t[hardware]
