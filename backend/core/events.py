"""WebSocket events streamed on /ws/races/{id}.

Every event is a JSON object with a ``type`` discriminator plus ``race_id`` and ``ts``.
Serialize with ``event.model_dump_json()``; parse with ``parse_event(raw)``.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, TypeAdapter

from backend.core.types import (
    ArmKind,
    ArmSummary,
    CorrectionLabel,
    CorrectionSource,
    ExperimentConfig,
    ExperimentStatus,
    LLMCallRecord,
    SlotState,
    now,
)


class _EventBase(BaseModel):
    race_id: str
    ts: float = Field(default_factory=now)


class GpuStateEvent(_EventBase):
    """A slot changed state. The scheduler emits one per transition (mirrors gpu_intervals)."""

    type: Literal["gpu_state"] = "gpu_state"
    arm: ArmKind
    slot_id: str
    state: SlotState
    experiment_id: str | None = None  # set when state == busy


class ExperimentStartedEvent(_EventBase):
    type: Literal["experiment_started"] = "experiment_started"
    arm: ArmKind
    experiment_id: str
    slot_id: str
    config: ExperimentConfig
    hypothesis: str = ""
    seed: int
    hardware: str


class MetricEvent(_EventBase):
    """One streamed metric point (same fields as MetricPoint, experiment_id already mapped)."""

    type: Literal["metric"] = "metric"
    arm: ArmKind
    experiment_id: str
    step: int
    t: float
    name: str
    value: float


class ExperimentFinishedEvent(_EventBase):
    type: Literal["experiment_finished"] = "experiment_finished"
    arm: ArmKind
    experiment_id: str
    slot_id: str
    status: ExperimentStatus
    score: float | None = None
    is_trap: bool = False
    duration_s: float = 0.0
    gpu_cost_usd: float = 0.0
    error: str | None = None


class NewBestEvent(_EventBase):
    type: Literal["new_best"] = "new_best"
    arm: ArmKind
    experiment_id: str
    score: float
    previous_best: float | None = None
    config: ExperimentConfig
    hypothesis: str = ""
    elapsed_s: float                  # since the race started
    reached_target: bool = False      # score crosses the target's target_value


class LLMCallEvent(_EventBase):
    type: Literal["llm_call"] = "llm_call"
    call: LLMCallRecord


class CorrectionEvent(_EventBase):
    type: Literal["correction"] = "correction"
    correction_id: str
    experiment_id: str
    arm: ArmKind | None = None
    source: CorrectionSource
    label: CorrectionLabel
    reason: str = ""


class RaceFinishedEvent(_EventBase):
    type: Literal["race_finished"] = "race_finished"
    summaries: list[ArmSummary]
    reason: str = "budget"            # "budget" | "cost_cap" | "cancelled" | "error"


Event: TypeAlias = Annotated[
    GpuStateEvent
    | ExperimentStartedEvent
    | MetricEvent
    | ExperimentFinishedEvent
    | NewBestEvent
    | LLMCallEvent
    | CorrectionEvent
    | RaceFinishedEvent,
    Field(discriminator="type"),
]

EventAdapter: TypeAdapter[Event] = TypeAdapter(Event)

EVENT_TYPES: tuple[str, ...] = (
    "gpu_state",
    "experiment_started",
    "metric",
    "experiment_finished",
    "new_best",
    "llm_call",
    "correction",
    "race_finished",
)


def parse_event(raw: str | bytes | dict) -> Event:
    """Parse a JSON string/bytes or a dict into the matching event model."""
    if isinstance(raw, dict):
        return EventAdapter.validate_python(raw)
    return EventAdapter.validate_json(raw)
