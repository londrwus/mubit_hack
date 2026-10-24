"""WebSocket events streamed on /ws/races/{id}.

Every event is a JSON object with a ``type`` discriminator plus ``race_id`` and ``ts``.
Serialize with ``event.model_dump_json()``; parse with ``parse_event(raw)``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias, Union, get_args

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
    Verdict,
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
    # Who proposed the run: "baseline" | "llm" (the AI planner) | "optuna" | "sampler" (the
    # backfill no-AI sampler). None on events from before the field existed.
    source: str | None = None


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
    # Ground truth for the dashboard ("Model quality"), never shown to a researcher:
    # ``metrics.quality`` of the run (held-out score if reported, else the score unless a trap)
    # for a succeeded run; None otherwise.
    quality: float | None = None


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


# =========================================================================== dashboard additions
# Appended after the core union (not inlined in it) so patches that edit the core union above
# still apply. ``Event`` / ``EventAdapter`` are rebuilt below to include these types; the core
# tuple ``EVENT_TYPES`` keeps its meaning (the original stream types). docs/dashboard_api.md.


class JudgeVerdictEvent(_EventBase):
    """The checker's decision on a claimed new best (LLM arms only), sent when the judge step
    returns, before any ``new_best`` for the same run. ``verdict`` is the decision that counts
    (the judge policy's final on Busy Machines arms, else the LLM judge's); ``llm_verdict`` is
    the LLM judge's own. ``is_trap`` / ``quality`` are ground truth for the dashboard;
    ``fake_caught`` = a known fake (trap run) that the checker rejected (verdict invalid)."""

    type: Literal["judge_verdict"] = "judge_verdict"
    arm: ArmKind
    experiment_id: str
    verdict: Verdict | None = None        # None: no usable judge reply
    llm_verdict: Verdict | None = None
    llm_reason: str | None = None         # judge_policy.llm_reason, else the stored judge reason
    reason: str | None = None             # experiments.judge_reason
    claimed_score: float | None = None    # the run's reported score
    previous_best: float | None = None    # the arm's best reported score when the claim was made
    is_trap: bool = False
    quality: float | None = None
    source: str | None = None
    fake_caught: bool = False


class PlannerStartedEvent(_EventBase):
    """An AI planner (propose) call started on an LLM arm; ``k`` = experiments asked for."""

    type: Literal["planner_started"] = "planner_started"
    arm: ArmKind
    k: int


class PlannerFinishedEvent(_EventBase):
    """The planner call that started at ``started_at`` returned; ``n_added`` proposals were
    accepted (0 = an empty or invalid batch). Not sent for a call cut off by the end of the race."""

    type: Literal["planner_finished"] = "planner_finished"
    arm: ArmKind
    started_at: float
    k: int = 0
    n_added: int = 0


SlotDisplayState = Literal["busy", "waiting", "cancelled", "off"]


class SlotEvent(_EventBase):
    """Derived (dashboard streams only, never emitted by the engine): a slot's display state
    changed. busy = running an experiment; waiting = paid, no experiment; cancelled = waiting
    right after its run was stopped (cancelled / failed / timed out), until the next run;
    off = closed (released early or the race ended)."""

    type: Literal["slot"] = "slot"
    arm: ArmKind
    slot_id: str
    state: SlotDisplayState
    experiment_id: str | None = None      # the running run (busy) or the stopped one (cancelled)
    source: str | None = None             # who proposed that run
    race_elapsed_s: float = 0.0


class TickEvent(_EventBase):
    """Derived (dashboard streams only): the full dashboard snapshot at ``ts`` (race time),
    sent every ``tick_s`` seconds of wall time, once at connect and once after race_finished."""

    type: Literal["tick"] = "tick"
    race_elapsed_s: float = 0.0
    snapshot: dict[str, Any] = Field(default_factory=dict)


class ModalStatsEvent(_EventBase):
    """Live Modal stats for one arm (DEMO RACES on per-arm Modal apps only; never sent for a
    benchmark or ablation race, never stored in the race DB, the bus history or a replay). One per
    arm per poll (~2 s), published by the app (``backend.app.ModalStatsHub``), never by the engine:
    ``modal.Function.get_current_stats`` on the arm's own app plus the latest in-container GPU
    samples of its runs in flight (docs/dashboard_modal.md has units and the 'idle' definition)."""

    type: Literal["modal_stats"] = "modal_stats"
    arm: ArmKind
    app: str                                  # the arm's deployed Modal app
    function: str                             # run_l4 / run_cpu
    ok: bool = True                           # False: the stats call failed (see error)
    error: str | None = None
    containers: int | None = None             # Modal num_total_runners: containers alive (warm or busy)
    calls_in_flight: int | None = None        # Modal num_running_inputs: runs executing now
    backlog: int | None = None                # Modal backlog: inputs waiting for a container
    input_headroom: int | None = None         # Modal input_headroom
    idle_containers: int | None = None        # containers - calls_in_flight: warm, no call in flight
    gpu_util_pct: float | None = None         # mean over known containers, idle ones at 0 (0-100)
    gpu_util_active_pct: float | None = None  # mean over sampled runs in flight (0-100)
    gpu_mem_mb: float | None = None           # GPU memory used, summed over sampled runs (MiB)
    runs_sampled: int = 0                     # runs in flight with a live in-container sample
    runs_unsampled: int = 0                   # runs in flight with no sample yet (left out of gpu_util_pct)
    call_ids: list[str] = Field(default_factory=list)  # Modal call ids of this arm's runs in flight
    latest_call_id: str | None = None         # Modal call id of the arm's most recently started run
    poll_latency_s: float | None = None


ENGINE_EXTRA_EVENT_TYPES: tuple[str, ...] = ("judge_verdict", "planner_started", "planner_finished")
DASHBOARD_EVENT_TYPES: tuple[str, ...] = ("slot", "tick")
# Published by the app for demo races on per-arm Modal apps only (docs/dashboard_modal.md).
MODAL_EVENT_TYPES: tuple[str, ...] = ("modal_stats",)
ALL_EVENT_TYPES: tuple[str, ...] = EVENT_TYPES + ENGINE_EXTRA_EVENT_TYPES + DASHBOARD_EVENT_TYPES + MODAL_EVENT_TYPES

_EXTRA_MODELS = (JudgeVerdictEvent, PlannerStartedEvent, PlannerFinishedEvent, SlotEvent, TickEvent, ModalStatsEvent)
Event = Annotated[  # type: ignore[misc]  # the core union above plus the dashboard additions
    Union[(*get_args(get_args(Event)[0]), *_EXTRA_MODELS)],  # noqa: UP007
    Field(discriminator="type"),
]
EventAdapter = TypeAdapter(Event)  # type: ignore[misc]
