"""Race engine: one asyncio loop per arm, identical for every arm (architecture.md section 3).

How an arm works (``ArmRunner``):
- It owns N slots. At t0 every slot is **idle** and the idle clock runs.
- With ``run_baseline_first`` the scheduler itself launches the starting-code config on
  slot 0 at t0 (no LLM), in every arm.
- The **researcher** (LLM agent or Optuna) is one sequential coroutine. It wakes when a run
  ends or a proposal batch lands, then:
  1. processes finished runs in order: ``summarize_result`` (deterministic code, delta vs
     best, keep/drop), ``diagnose`` **only on failures**, ``judge`` **only on a claimed new
     best** (never on the starting-code baseline). A judged-invalid best is not counted.
  2. if K = demand > 0, makes **one** batched ``propose`` call for K experiments.
- Demand K = N + lookahead - running - buffered, where lookahead = N with pipelining
  (ask for the next batch while runs are in flight) and 0 without. Same flag for all arms.
- A slot is busy from the moment the scheduler launches a run until the runner's
  ``wait()`` returns; it is idle from then until its next launch. LLM wait time is idle.
  Each slot's intervals partition [t0, t_end] exactly, so busy + idle = paid time.
- The arm stops at the shared wall-clock deadline (t_end = deadline exactly) or when its
  running GPU $ + LLM $ reaches the cap. Runs still in flight are cancelled and do not
  count; slots are closed at t_end.
- Deadline rule (``RaceConfig.launch_cutoff_s``, same for every arm): once less than
  ``launch_cutoff_s`` of the race remains, no run is launched (it could not finish), no
  proposal is asked for, buffered proposals are dropped (logged, status cancelled) and each
  slot is **released** when it goes idle: an idle slot at the cutoff is released at the
  cutoff, a busy one when its run ends. A released slot stops paying, so its intervals
  partition [t0, t_release] and it is neither busy nor waiting afterwards.

Busy Machines engine features (D36, D40, D43, D44; ``backend.benchmark.engine_rules``),
switched on per arm by ``RaceConfig.features_for``. Defaults: the Opus arm has none (the
standard agent), "Busy Machines with Opus" (``opus_bm``) and "Mubit + Busy Machines"
(``mubit``) get the judge policy only, Optuna gets none. Median stopping is OFF by default for
every arm (D43) and only runs when a dev race opts in. Both are engine code (no LLM call, no
prompt change):
- ``judge_policy`` (verification, not optimisation, D44): after the LLM judge, an ``invalid``
  stays invalid; any other claim is decided by the calibrated seed-spread rule (real -> valid,
  noise -> unsure). The rule's decision only picks the **shipped** model (``is_best``, the row's
  ``judge_verdict``, the shipped-model metrics); the planner digest and the agent's own best
  (leaderboard, previous best in later judge claims, claimed-new-best checks) see exactly what
  the Opus arm would see: the LLM judge's verdict. The LLM's own verdict and the rule's
  decision are stored per claim (``summary["judge_policy"]``), never shown to the planner.
- ``median_stop`` (off by default): at 25% / 50% of a run's budget (``RaceConfig.median_stop_fractions``), a
  run whose curve value is worse than the median of this arm's completed runs at the same
  point (>= ``median_stop_min_completed`` of them) is cancelled through the runner; its slot
  frees as soon as the runner reports the run over (on Modal: once Modal confirms the input
  stopped, so the stopping arm pays for the cancel lag), and the experiment is stored as
  cancelled with ``summary["early_stop"]`` (checkpoint, value, median, stop latency, estimated
  slot time freed). A run that ends on its own first keeps its real outcome (``too_late``, or
  failed and diagnosed).
Feature flags are in the fairness manifest (``engine``); a race whose BM arms differ, whose
Optuna arm does not share the BM arms' median rule, or (benchmark / ablation) whose arms do not
run their defaults, or whose judge policy has no calibration for the target, is refused.

Modes (``RaceConfig.mode``): ``benchmark`` and ``ablation`` get the same strict integrity checks
(``RaceConfig.strict``); an ablation race (e.g. the ``fast`` arm: a smaller faster model in the
same loop, not a Mubit adapter) is never a headline race.

``RaceEngine`` builds one ``ArmRunner`` per arm from the same ``RaceConfig``, checks that
every arm got identical slots / hardware / prices / budgets / seeds, starts them at the
same t0, and computes the headline metrics with ``backend.benchmark.metrics``.

Recording mode (``x-middleware-record``) is per arm (``RaceConfig.record_mode``): the
benchmark presets send ``off`` on the Opus arm (no task fingerprint, so Mubit cannot route
it to an adapter) and ``metadata`` on the Mubit arm (fingerprint kept, content dropped).
It is the one deliberate difference besides the client, so it sits in the manifest's
``differs_only_in`` and not in the LLM policy that must match. Pending Mubit's confirmation
(docs/mubit_office_hours.md item 2).
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import statistics
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from backend.benchmark.configs import (
    ABLATION_ONLY_ARMS,
    BUSY_MACHINES_ARMS,
    DEFAULT_ENGINE_FEATURES,
    FAST_ARM_LABEL,
    RaceConfig,
    estimate_gpu_cost_usd,
)
from backend.benchmark.engine_rules import (
    EngineState,
    MedianStopRule,
    SeedSpreadRule,
    combine,
    curve_metric,
    load_sigma_tables,
    sigma_key,
    value_at,
)
from backend.benchmark.metrics import is_counted, quality, summarize_arm
from backend.core.events import (
    CorrectionEvent,
    Event,
    ExperimentFinishedEvent,
    ExperimentStartedEvent,
    GpuStateEvent,
    LLMCallEvent,
    MetricEvent,
    NewBestEvent,
    RaceFinishedEvent,
)
from backend.core.types import (
    ArmKind,
    ArmSummary,
    CorrectionLabel,
    CorrectionSource,
    DiagnoseOutput,
    ExperimentConfig,
    ExperimentStatus,
    JudgeOutput,
    LLMCallRecord,
    MetricPoint,
    RunHandle,
    RunResult,
    Slot,
    SlotState,
    Verdict,
    new_id,
    now,
)
from backend.db import BenchmarkIntegrityError, Database
from workers.runners.base import Runner
from workers.targets.base import TargetSpec, get_target

log = logging.getLogger(__name__)

BASELINE_HYPOTHESIS = "Starting code: the default config, run by the scheduler as the baseline."
EMPTY_BATCH_BACKOFF_S = 0.5   # after a reply with no usable config, before asking again
SHUTDOWN_GRACE_S = 5.0        # wait this long for cancelled runs to report back

# Ground-truth run metrics the researcher must never see (targets report them so the
# benchmark can grade honestly; see workers/targets/*.py). Any key containing "heldout" is
# hidden too.
GROUND_TRUTH_METRIC_KEYS = frozenset({
    "heldout_score", "is_trap", "test_bpb", "generalization_gap_bpb", "train_levels_score",
    "final_heldout_acc",
})


# Runner accounting (ModalRunner: cancel() -> Modal confirmed the input stopped). Not about the
# experiment, so researchers never see it (it would also make Modal digests differ from others).
RUNNER_ACCOUNTING_METRIC_KEYS = frozenset({"cancel_lag_s", "cancel_confirmed"})


def agent_visible_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in metrics.items()
            if k not in GROUND_TRUTH_METRIC_KEYS and k not in RUNNER_ACCOUNTING_METRIC_KEYS and "heldout" not in k}


# Ground truth also leaks through log lines (cifar prints "heldout_acc=... trap=True", lm prints
# "trap=True" at start-up and "test_bpb=..." at the end). Redact those key=value pairs before a
# log tail can reach a researcher (diagnose_run sends logs_tail to the LLM).
_GROUND_TRUTH_LOG_RE = re.compile(
    r"\b(?:is_trap|trap|\w*heldout\w*|test_bpb|train_levels\w*|generalization_gap\w*)\s*[=:]\s*[^\s,;]+",
    re.IGNORECASE,
)


def agent_visible_logs(lines: list[str]) -> list[str]:
    return [_GROUND_TRUTH_LOG_RE.sub("[hidden]", ln) for ln in lines]


class AsymmetricRaceError(ValueError):
    """The arms of a race would not get identical resources."""


class CorrectionRefusedError(BenchmarkIntegrityError):
    """A mid-race correction on a benchmark race that does not allow (and record) them."""


# =========================================================================== event bus


class EventBus:
    """In-process pub/sub per race. Keeps a bounded history so a late WebSocket subscriber
    can replay the race so far. Call from the event-loop thread."""

    def __init__(self, history_limit: int = 100_000) -> None:
        self._subs: dict[str, set[asyncio.Queue[Event]]] = defaultdict(set)
        self._history: dict[str, deque[Event]] = defaultdict(lambda: deque(maxlen=history_limit))
        self._global: list[Callable[[Event], None]] = []

    def publish(self, event: Event) -> None:
        self._history[event.race_id].append(event)
        for q in list(self._subs.get(event.race_id, ())):
            q.put_nowait(event)
        for fn in self._global:
            try:
                fn(event)
            except Exception:  # pragma: no cover - listener bugs must not stop a race
                log.exception("event listener failed")

    def subscribe(self, race_id: str, replay: bool = True) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue()
        if replay:
            for ev in self._history.get(race_id, ()):
                q.put_nowait(ev)
        self._subs[race_id].add(q)
        return q

    def unsubscribe(self, race_id: str, q: asyncio.Queue[Event]) -> None:
        self._subs.get(race_id, set()).discard(q)

    def history(self, race_id: str) -> list[Event]:
        return list(self._history.get(race_id, ()))

    def add_listener(self, fn: Callable[[Event], None]) -> None:
        self._global.append(fn)


# =========================================================================== researcher API


class ExperimentSnapshot(BaseModel):
    """What a researcher may see about an experiment. Never includes ground truth
    (``is_trap``, held-out scores: see ``GROUND_TRUTH_METRIC_KEYS``)."""

    id: str
    label: str = ""                       # race-local name, "#1", "#2", ... (proposal order)
    config: ExperimentConfig
    hypothesis: str = ""
    status: ExperimentStatus
    score: float | None = None
    error: str | None = None
    logs_tail: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    is_baseline: bool = False
    duration_s: float | None = None
    verdict: Verdict | None = None
    diagnosis: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None


class ResearchDigest(BaseModel):
    """Compact state for propose_experiments: top-5 leaderboard, last 8 results, current
    best and K (budget.md lever 4). Built by deterministic code."""

    k: int
    n_slots: int
    elapsed_s: float
    remaining_s: float
    best: ExperimentSnapshot | None = None
    leaderboard: list[ExperimentSnapshot] = Field(default_factory=list)  # top 5, best first
    recent: list[ExperimentSnapshot] = Field(default_factory=list)       # last 8, newest last
    running: list[ExperimentConfig] = Field(default_factory=list)        # in flight / queued
    history: list[ExperimentSnapshot] = Field(default_factory=list)      # all processed, oldest first
    n_finished: int = 0
    n_failed: int = 0


class JudgeClaim(BaseModel):
    """Evidence for judge_improvement: the claimed new best vs the previous best."""

    candidate: ExperimentSnapshot
    previous_best: ExperimentSnapshot | None = None
    delta: float | None = None             # candidate - previous (raw metric units)
    improvement: float | None = None       # direction-aware, > 0 means better
    config_diff: dict[str, list[Any]] = Field(default_factory=dict)  # name -> [previous, candidate]


@dataclass
class Proposal:
    config: ExperimentConfig
    hypothesis: str
    experiment_id: str = field(default_factory=new_id)
    llm_call_id: str | None = None
    mubit_request_id: str | None = None


def _noop_sink(_: LLMCallRecord) -> None:
    return None


@runtime_checkable
class Researcher(Protocol):
    """The only thing that differs between arms. ``on_call`` is set by the scheduler and
    must be called with every LLMCallRecord the researcher produces (including retries)."""

    kind: ArmKind
    on_call: Callable[[LLMCallRecord], None]

    async def propose(self, digest: ResearchDigest, k: int) -> list[Proposal]: ...

    async def diagnose(self, exp: ExperimentSnapshot) -> DiagnoseOutput | None: ...

    async def judge(self, claim: JudgeClaim) -> JudgeOutput | None: ...

    def observe(self, exp: ExperimentSnapshot) -> None: ...


@dataclass
class ArmDeps:
    """What a race needs per arm. Only ``researcher`` (and the model it uses) may differ."""

    kind: ArmKind
    researcher: Researcher
    runner: Runner
    model: str | None = None
    llm_mode: str = "none"   # mubit | mock | replay | none
    record_mode: str | None = None  # off | metadata | full; None for a non-LLM arm


def normalize_record(record: Any) -> str:
    """A researcher's ``record`` value (bool or header value) -> off | metadata | full."""
    if isinstance(record, bool) or record is None:
        return "full" if record else "off"
    return str(record)


def race_spec(cfg: RaceConfig, spec: TargetSpec | None = None) -> TargetSpec:
    """The target spec with the race's goal (``cfg.target_value``) and per-run budget
    (``cfg.run_budget_s``) applied, so researchers see the budget runs really get."""
    spec = spec or get_target(cfg.target).spec
    upd: dict[str, Any] = {}
    if cfg.target_value is not None and spec.target_value != cfg.target_value:
        upd["target_value"] = float(cfg.target_value)
    if cfg.run_budget_s is not None and spec.budget_s != cfg.run_budget_s:
        upd["budget_s"] = int(cfg.run_budget_s)
    return spec.model_copy(update=upd) if upd else spec


# =========================================================================== engine features


_SIGMA_TABLES: dict[str, Any] | None = None


def sigma_tables() -> dict[str, Any]:
    """The calibrated seed-spread tables (loaded once per process)."""
    global _SIGMA_TABLES
    if _SIGMA_TABLES is None:
        _SIGMA_TABLES = load_sigma_tables()
    return _SIGMA_TABLES


def target_default_budget_s(cfg: RaceConfig, spec: TargetSpec | None = None) -> float:
    """The REGISTERED target's default per-run budget (what ``bench.goal_key`` compares
    ``run_budget_s`` against). Never the race spec's budget: ``race_spec`` already applied
    ``run_budget_s`` to it, so comparing against it would always give the bare target key
    (R41: ``rl`` instead of ``rl@15s``). ``spec`` is the fallback for an unregistered target."""
    try:
        return float(get_target(cfg.target).spec.budget_s)
    except KeyError:
        if spec is None:
            raise
        return float(spec.budget_s)


def resolve_engine_state(cfg: RaceConfig, kind: ArmKind, spec: TargetSpec, *,
                         default_budget_s: float | None = None,
                         tables: dict[str, Any] | None = None) -> EngineState:
    """One arm's Busy Machines engine settings, from the config (``features_for``).
    ``default_budget_s``: the registered target's default run budget (default:
    ``target_default_budget_s``); it decides the sigma key (``rl`` vs ``rl@15s``)."""
    feats = cfg.features_for(kind)
    st = EngineState(features=feats)
    if "median_stop" in feats:
        st.median_rule = MedianStopRule(tuple(cfg.median_stop_fractions), cfg.median_stop_min_completed)
        st.curve_metric = curve_metric(spec.name, spec.metric)
    if "judge_policy" in feats and kind != ArmKind.optuna:
        if default_budget_s is None:
            default_budget_s = target_default_budget_s(cfg, spec)
        key = sigma_key(spec.name, cfg.run_budget_s, default_budget_s)
        table = (sigma_tables() if tables is None else tables).get(key)
        st.judge_rule = SeedSpreadRule(table)
        if table is None:
            st.notes.append(f"judge_policy: no calibrated seed spread for {key!r}; the LLM verdict stands")
    return st


def check_engine_features(cfg: RaceConfig, states: dict[ArmKind, EngineState], target: str) -> list[str]:
    """Refuse (AsymmetricRaceError) a race whose arms should share engine features but don't:
    the Busy Machines arms (opus_bm, mubit) must be identical; Optuna must run the median rule
    iff they do; in a benchmark the Opus arm has none (the standard agent) and a judge policy
    needs a calibrated seed spread for the target. Returns the checks that passed."""
    feats = {k: tuple(st.features) for k, st in states.items()}
    done: list[str] = []
    bm = {k: f for k, f in feats.items() if k in BUSY_MACHINES_ARMS}
    if len(set(bm.values())) > 1:
        raise AsymmetricRaceError(f"Busy Machines arms must get identical engine features: "
                                  f"{ {k.value: list(f) for k, f in bm.items()} }")
    if len(bm) > 1:
        done.append(f"BM arms {sorted(k.value for k in bm)} share features {list(next(iter(bm.values())))}")
    if ArmKind.optuna in feats and bm:
        bm_median = "median_stop" in next(iter(bm.values()))
        if ("median_stop" in feats[ArmKind.optuna]) != bm_median:
            raise AsymmetricRaceError(
                f"optuna must {'' if bm_median else 'not '}run the median rule like the Busy Machines arms "
                f"(optuna {list(feats[ArmKind.optuna])}, BM {list(next(iter(bm.values())))})")
        done.append(f"optuna median rule matches the BM arms ({bm_median})")
    if cfg.strict:
        # Benchmark / ablation: every arm runs exactly its kind's default features (no per-race overrides),
        # so a headline column always means the same engine.
        off = {k.value: [list(f), list(DEFAULT_ENGINE_FEATURES.get(k, ()))] for k, f in feats.items()
               if tuple(sorted(f)) != tuple(sorted(DEFAULT_ENGINE_FEATURES.get(k, ())))}
        if off:
            raise AsymmetricRaceError(f"{cfg.mode} races run each arm's default engine features; "
                                      f"overridden (got, default): {off}")
        done.append("every arm runs its default engine features")
        if feats.get(ArmKind.opus):
            raise AsymmetricRaceError(f"the opus arm is the standard agent; it has engine features "
                                      f"{list(feats[ArmKind.opus])}")
        if ArmKind.opus in feats:
            done.append("opus arm: no engine features (standard agent)")
        for kind, st in states.items():
            if st.judge_rule is not None and not st.judge_rule.available:
                raise AsymmetricRaceError(f"arm {kind.value}: judge_policy has no calibrated seed spread for "
                                          f"target {target!r} at run budget {cfg.run_budget_s}")
        if any(st.judge_rule is not None for st in states.values()):
            done.append("judge policy: calibrated seed spread available")
    return done


# =========================================================================== helpers


def config_diff(a: ExperimentConfig | None, b: ExperimentConfig) -> dict[str, list[Any]]:
    if a is None:
        return {}
    return {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def summarize_result(spec: TargetSpec, status: ExperimentStatus, score: float | None,
                     best_score: float | None) -> dict[str, Any]:
    """Deterministic ``summarize_result`` (not an LLM call): delta vs best, keep/drop."""
    out: dict[str, Any] = {"status": status.value, "score": score, "best_before": best_score,
                           "delta": None, "improvement": None, "claimed_new_best": False,
                           "decision": "drop"}
    if status != ExperimentStatus.succeeded or score is None:
        return out
    if best_score is not None:
        out["delta"] = score - best_score
        out["improvement"] = (best_score - score) if spec.direction == "min" else (score - best_score)
    if spec.is_better(score, best_score):
        out["claimed_new_best"] = True
        out["decision"] = "keep"
    return out


# =========================================================================== arm state


# Engine records in an experiment summary that the planner must not see (D44: the judge policy
# is verification of the shipped model, not an input to the search).
PLANNER_HIDDEN_SUMMARY_KEYS = frozenset({"judge_policy"})


def planner_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {k: v for k, v in summary.items() if k not in PLANNER_HIDDEN_SUMMARY_KEYS}


@dataclass
class _Exp:
    id: str
    arm_id: str
    config: ExperimentConfig
    hypothesis: str
    seed: int
    proposed_at: float
    label: str = ""
    is_baseline: bool = False
    status: ExperimentStatus = ExperimentStatus.pending
    slot_id: str | None = None
    hardware: str | None = None
    started_at: float | None = None
    ended_at: float | None = None
    score: float | None = None
    is_trap: bool = False
    is_best: bool = False                 # in the SHIPPED chain (the model this arm would ship)
    verdict: Verdict | None = None        # the decision that counts (judge policy's final, else the LLM's)
    verdict_reason: str | None = None
    llm_verdict: Verdict | None = None    # the LLM judge's own verdict: all the planner ever sees (D44)
    diagnosis: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    logs_tail: list[str] = field(default_factory=list)
    run_ref: str | None = None
    llm_call_id: str | None = None
    mubit_request_id: str | None = None
    gpu_cost_usd: float = 0.0
    infra_error: bool = False
    processed: bool = False
    flagged_fake: bool = False
    override_genuine: bool = False
    handle: RunHandle | None = None
    # Busy Machines engine (median_stop / judge_policy); see engine_rules.
    curve: list[tuple[float, float]] = field(default_factory=list)   # (metric t, value), oldest first
    ckpt_values: dict[float, float] = field(default_factory=dict)    # completed run: value per fraction
    ckpt_checked: set[float] = field(default_factory=set)
    early_stop: dict[str, Any] | None = None
    cancel_sent: bool = False
    judge_policy: dict[str, Any] | None = None

    def full_summary(self, base: dict[str, Any] | None) -> dict[str, Any] | None:
        """``summarize_result`` output plus the engine's per-run records."""
        if base is None:
            return None
        out = dict(base)
        if self.early_stop is not None:
            out["early_stop"] = dict(self.early_stop)
        if self.judge_policy is not None:
            out["judge_policy"] = dict(self.judge_policy)
        return out

    @property
    def planner_verdict(self) -> Verdict | None:
        """The verdict the planner sees (D44): the LLM judge's, never the judge policy's
        decision, so a Busy Machines arm's digest is the one the Opus arm would build."""
        return self.llm_verdict if self.judge_policy is not None else self.verdict

    def snapshot(self) -> ExperimentSnapshot:
        """What a researcher sees: ground truth hidden, and (D44) the LLM judge's verdict and a
        summary without the judge policy's record."""
        return ExperimentSnapshot(
            id=self.id, label=self.label, config=dict(self.config), hypothesis=self.hypothesis,
            status=self.status, score=self.score,
            error=_GROUND_TRUTH_LOG_RE.sub("[hidden]", self.error) if self.error else self.error,
            logs_tail=agent_visible_logs(list(self.logs_tail[-50:])),
            metrics=agent_visible_metrics(self.metrics), is_baseline=self.is_baseline,
            duration_s=(self.ended_at - self.started_at) if (self.ended_at and self.started_at) else None,
            verdict=self.planner_verdict, diagnosis=self.diagnosis, summary=planner_summary(self.summary),
        )

    def row(self, race_id: str, runner: str) -> dict[str, Any]:
        return {
            "id": self.id, "arm_id": self.arm_id, "race_id": race_id, "slot_id": self.slot_id,
            "config": self.config, "hypothesis": self.hypothesis, "status": self.status.value,
            "runner": runner, "run_ref": self.run_ref, "seed": self.seed, "hardware": self.hardware,
            "proposed_at": self.proposed_at, "started_at": self.started_at, "ended_at": self.ended_at,
            "score": self.score, "is_best": self.is_best, "is_trap": self.is_trap,
            "is_baseline": self.is_baseline,
            "judge_verdict": self.verdict.value if self.verdict else None,
            "judge_reason": self.verdict_reason, "diagnosis": self.diagnosis, "summary": self.summary,
            "error": self.error, "metrics": self.metrics, "logs_tail": self.logs_tail,
            "llm_call_id": self.llm_call_id, "mubit_request_id": self.mubit_request_id,
            "gpu_cost_usd": self.gpu_cost_usd,
        }


@dataclass
class _SlotState:
    slot: Slot
    price_per_s: float
    state: SlotState = SlotState.idle
    since: float = 0.0
    experiment_id: str | None = None
    closed: bool = False
    closed_at: float | None = None   # released early by the deadline rule (else t_end)


# =========================================================================== arm runner


class ArmRunner:
    """One arm of a race. Every arm in a race is an instance of this same class with the
    same RaceConfig; only ``researcher`` differs."""

    def __init__(
        self,
        *,
        race_id: str,
        arm_id: str,
        deps: ArmDeps,
        cfg: RaceConfig,
        spec: TargetSpec,
        slots: list[Slot],
        t0: float,
        deadline: float,
        db: Database | None,
        bus: EventBus,
        tick_s: float = 0.05,
        engine: EngineState | None = None,
    ) -> None:
        self.race_id = race_id
        self.arm_id = arm_id
        self.kind = deps.kind
        self.deps = deps
        self.cfg = cfg
        self.spec = spec
        self.researcher = deps.researcher
        self.runner = deps.runner
        self.db = db
        self.bus = bus
        self.tick_s = tick_s
        self.t0 = t0
        self.deadline = deadline
        self.run_budget_s = int(cfg.run_budget_s or spec.budget_s)
        self.slots = [_SlotState(slot=s, price_per_s=float(self.runner.price_per_second(s.hardware)), since=t0)
                      for s in slots]
        self.n = len(self.slots)
        self.lookahead = self.n if cfg.pipelining else 0
        self.cutoff_at: float | None = (deadline - float(cfg.launch_cutoff_s)
                                        if cfg.launch_cutoff_s is not None else None)
        self.cutoff_applied = False
        self.launches_skipped_deadline = 0
        self.slots_released_early = 0

        self.exps: dict[str, _Exp] = {}
        self.order: list[str] = []                # proposal order
        self.buffer: deque[str] = deque()         # proposed, not launched
        self.finished: deque[str] = deque()       # finished, not yet processed by the researcher
        self.processed: list[str] = []            # processed, in order
        self.run_tasks: dict[str, asyncio.Task[None]] = {}
        self.intervals: list[dict[str, Any]] = []
        self.llm_calls: list[dict[str, Any]] = []
        self.llm_cost = 0.0
        # Two bests (D44). ``best_id``: the agent's own best from the LLM judge's verdicts; it
        # drives the planner digest, later judge claims and "claimed new best", exactly as in the
        # Opus arm. ``shipped_id``: the model this arm ships, from the decisions that count (the
        # judge policy's, if the arm has one). Without a judge policy they are the same run.
        self.best_id: str | None = None
        self.shipped_id: str | None = None
        self.configs_rejected = 0
        self.empty_batches = 0

        self.stopped = False
        self.stop_reason: str | None = None
        self.t_end: float | None = None
        self.error: str | None = None
        self._stop_event = asyncio.Event()
        self._wake = asyncio.Event()
        self._agent_task: asyncio.Task[None] | None = None
        self._monitor_task: asyncio.Task[None] | None = None

        # Busy Machines engine features (RaceEngine resolves them; standalone: from the config).
        self.engine = engine or resolve_engine_state(cfg, deps.kind, spec)
        self._time_scale = float(getattr(self.runner, "runtime_multiplier", None) or 1.0)
        self._checkpoints: list[tuple[float, float]] = (
            self.engine.median_rule.checkpoints(self.run_budget_s, self._time_scale)
            if self.engine.median_rule is not None else [])
        self._completed: list[str] = []            # succeeded runs (median-rule reference), in end order
        self._cancel_tasks: set[asyncio.Task[None]] = set()
        self.median_checks = 0
        self.median_skipped_few_completed = 0
        self.early_stops_requested = 0

        self.researcher.on_call = self._on_llm_call

    # ------------------------------------------------------------------ public

    @property
    def price_per_s_total(self) -> float:
        return sum(s.price_per_s for s in self.slots)

    def fairness(self) -> dict[str, Any]:
        """Everything that must be identical across arms."""
        return {
            "scheduler": type(self).__name__,
            "n_slots": self.n,
            "hardware": [s.slot.hardware for s in self.slots],
            "price_per_s": [s.price_per_s for s in self.slots],
            "t0": self.t0,
            "deadline": self.deadline,
            "budget_s": self.cfg.budget_s,
            "cost_cap_usd": self.cfg.cost_cap_usd,
            "seed": self.cfg.seed,
            "run_budget_s": self.run_budget_s,
            "pipelining": self.cfg.pipelining,
            "lookahead": self.lookahead,
            "max_proposals_per_call": self.cfg.max_proposals_per_call,
            "run_baseline_first": self.cfg.run_baseline_first,
            "accept_unsure": self.cfg.accept_unsure,
            "launch_cutoff_s": self.cfg.launch_cutoff_s,
            "record_modes": self.cfg.record_modes(),   # the whole per-arm map: same on every arm
            # Engine features are per arm by design (manifest "engine"); the map and the median
            # rule's parameters are race-wide, so they are the same on every arm.
            "engine_features": self.cfg.engine_features_map(),
            "median_stop_rule": {"fractions": list(self.cfg.median_stop_fractions),
                                 "min_completed": self.cfg.median_stop_min_completed},
            "target": self.spec.name,
            "target_value": self.spec.target_value,
            "runner": self.runner.name,
            # Surrogate pacing (workers/runners/surrogate.py): predicted-seconds -> real-seconds.
            # Must match across arms or "GPUs waiting" is not comparable. None for real runners.
            "runtime_multiplier": getattr(self.runner, "runtime_multiplier", None),
        }

    def llm_policy(self) -> dict[str, Any] | None:
        """What must match across LLM arms besides the LLM itself: same researcher class,
        prompts (Mubit groups calls by identical instructions) and retry/fallback policy.
        The record mode is per arm by design (see ``differs_only_in``). None for a non-LLM
        arm (Optuna)."""
        r = self.researcher
        prompts = getattr(r, "prompts", None)
        if prompts is None:
            return None
        return {
            "researcher": type(r).__name__,
            "prompts": getattr(prompts, "version", type(prompts).__name__),
            "max_retries": getattr(r, "max_retries", None),
            "has_fallback": getattr(r, "fallback_llm", None) is not None or getattr(r, "fallback_model", None) is not None,
            "fallback_model": getattr(r, "fallback_model", None),
        }

    async def run(self) -> None:
        for s in self.slots:
            self._emit(GpuStateEvent(race_id=self.race_id, arm=self.kind, slot_id=s.slot.slot_id,
                                     state=SlotState.idle, ts=self.t0))
        if self.cfg.run_baseline_first:
            p = Proposal(config=self.spec.default_config(), hypothesis=BASELINE_HYPOTHESIS)
            self._add_proposal(p, is_baseline=True)
            self._fill_slots()
        self._agent_task = asyncio.create_task(self._agent_loop(), name=f"agent-{self.kind}")
        self._monitor_task = asyncio.create_task(self._monitor(), name=f"monitor-{self.kind}")
        self._wake.set()
        try:
            await self._stop_event.wait()
        finally:
            await self._shutdown()

    def request_stop(self, reason: str = "cancelled") -> None:
        self._stop(reason, min(now(), self.deadline))

    def live_summary(self) -> tuple[ArmSummary, dict[str, Any]]:
        t = self.t_end if self.t_end is not None else min(now(), self.deadline)
        intervals = list(self.intervals)
        if self.t_end is None:
            for s in self.slots:
                if not s.closed and t > s.since:
                    intervals.append({"state": s.state.value, "start": s.since, "end": t,
                                      "price_per_s": s.price_per_s})
        summary, extras = summarize_arm(
            self.kind, direction=self.spec.direction, target_value=self.spec.target_value,
            race_start=self.t0, experiments=[self.exps[i].row(self.race_id, self.runner.name) for i in self.order],
            intervals=intervals, llm_calls=self.llm_calls,
        )
        extras.update({
            "stop_reason": self.stop_reason, "error": self.error, "configs_rejected": self.configs_rejected,
            "empty_batches": self.empty_batches, "agent_best_id": self.best_id,
            "shipped_best_id": self.shipped_id,
            "shipped_best_score": self.exps[self.shipped_id].score if self.shipped_id else None,
            "launch_cutoff_s": self.cfg.launch_cutoff_s,
            "launches_skipped_deadline": self.launches_skipped_deadline,
            "slots_released_early": self.slots_released_early,
            "released_slot_s": sum(max(0.0, t - s.closed_at) for s in self.slots
                                   if s.closed_at is not None and s.closed_at < t),
            "agent_best_score": self.exps[self.best_id].score if self.best_id else None,
            "record_mode": self.deps.record_mode, "target_value": self.spec.target_value,
            "engine_features": list(self.engine.features),
            "median_stop_checks": self.median_checks,
            "median_stop_skipped_few_completed": self.median_skipped_few_completed,
            "early_stops_requested": self.early_stops_requested,
        })
        return summary, extras

    def apply_correction(self, experiment_id: str, label: CorrectionLabel, *, recorded: bool = False) -> bool:
        """``recorded``: the caller listed this correction in the race's fairness manifest
        (``RaceEngine.apply_correction``); a benchmark race refuses an unrecorded one."""
        if self.cfg.strict and not recorded:
            raise CorrectionRefusedError(f"{self.cfg.mode} race: a correction must be recorded in the fairness manifest")
        exp = self.exps.get(experiment_id)
        if exp is None:
            return False
        if label == CorrectionLabel.fake_improvement:
            exp.flagged_fake, exp.override_genuine = True, False
            exp.is_best = False  # no longer an accepted best
        else:
            exp.flagged_fake, exp.override_genuine = False, True
        self._recompute_best()
        self._store(exp)
        return True

    # ------------------------------------------------------------------ stop / cost

    def _cost_at(self, t: float) -> float:
        gpu = sum(s.price_per_s * max(0.0, (min(t, s.closed_at) if s.closed_at is not None else t) - self.t0)
                  for s in self.slots)
        return gpu + self.llm_cost

    def _can_launch(self, t: float | None = None) -> bool:
        """Deadline rule: False once a new run could no longer finish before the deadline."""
        return self.cutoff_at is None or (now() if t is None else t) < self.cutoff_at

    def _release_slot(self, s: _SlotState, t: float) -> None:
        """Deadline rule: close an idle slot early; it stops paying at ``t``."""
        if s.closed or s.state != SlotState.idle:
            return
        self._transition(s, SlotState.idle, t, None)
        s.closed, s.closed_at = True, t
        self.slots_released_early += 1

    def _apply_cutoff(self, t: float) -> None:
        """Deadline rule kicks in (once): drop buffered proposals, release idle slots."""
        if self.cutoff_applied or self.cutoff_at is None:
            return
        self.cutoff_applied = True
        dropped = 0
        while self.buffer:
            exp = self.exps[self.buffer.popleft()]
            exp.status = ExperimentStatus.cancelled
            exp.error = (f"deadline rule: not launched, {max(0.0, self.deadline - t):.1f}s left < "
                         f"launch_cutoff_s {self.cfg.launch_cutoff_s:g}s")
            self._store(exp)
            dropped += 1
        self.launches_skipped_deadline += dropped
        for s in self.slots:
            if s.state == SlotState.idle and not s.closed:
                self._release_slot(s, max(s.since, self.cutoff_at))
        log.info("%s: deadline rule at %.1fs: %d buffered proposal(s) not launched, %d idle slot(s) released, "
                 "%d run(s) still in flight", self.kind, self.cutoff_at - self.t0, dropped,
                 self.slots_released_early, self._running_count())

    def _live(self, t: float | None = None) -> bool:
        return not self.stopped and (now() if t is None else t) < self.deadline

    def _stop(self, reason: str, t_end: float) -> None:
        if self.stopped:
            return
        self.stopped = True
        self.stop_reason = reason
        self.t_end = min(max(t_end, self.t0), self.deadline)
        self._stop_event.set()
        self._wake.set()

    async def _monitor(self) -> None:
        while not self.stopped:
            t = now()
            if t >= self.deadline:
                self._stop("budget", self.deadline)
                return
            if self.cfg.cost_cap_usd is not None and self._cost_at(t) >= self.cfg.cost_cap_usd:
                self._stop("cost_cap", t)
                return
            if not self.cutoff_applied and not self._can_launch(t):
                self._apply_cutoff(t)
            if self.db is not None:
                self.db.flush_metrics()
            await asyncio.sleep(max(0.001, min(self.tick_s, self.deadline - t)))

    # ------------------------------------------------------------------ events / storage

    def _emit(self, ev: Event) -> None:
        self.bus.publish(ev)

    def _store(self, exp: _Exp) -> None:
        if self.db is not None:
            self.db.upsert_experiment(exp.row(self.race_id, self.runner.name))

    def _on_llm_call(self, record: LLMCallRecord) -> None:
        if record.is_replay and self.cfg.strict:
            self.error = f"record/replay call in a {self.cfg.mode} race"
            self._stop("error", now())
            raise BenchmarkIntegrityError(self.error)
        rec = record.model_copy(update={"arm_id": self.arm_id})
        t = now()
        row = rec.model_dump(mode="json")
        row["completed_at"] = t
        self.llm_calls.append(row)
        self.llm_cost += float(rec.cost_usd or 0.0)
        if self.db is not None:
            self.db.insert_llm_call(rec, race_id=self.race_id, completed_at=t)
        self._emit(LLMCallEvent(race_id=self.race_id, call=rec, ts=t))
        if self.cfg.cost_cap_usd is not None and self._cost_at(t) >= self.cfg.cost_cap_usd:
            self._stop("cost_cap", t)

    # ------------------------------------------------------------------ slots

    def _transition(self, s: _SlotState, state: SlotState, t: float, experiment_id: str | None) -> None:
        if s.closed:
            return
        if t > s.since:
            iv = {"arm_id": self.arm_id, "race_id": self.race_id, "slot": s.slot.slot_id,
                  "hardware": s.slot.hardware, "state": s.state.value, "start": s.since, "end": t,
                  "experiment_id": s.experiment_id, "price_per_s": s.price_per_s}
            self.intervals.append(iv)
            if self.db is not None:
                self.db.insert_gpu_interval(**iv)
        s.state, s.since, s.experiment_id = state, max(t, s.since), experiment_id
        self._emit(GpuStateEvent(race_id=self.race_id, arm=self.kind, slot_id=s.slot.slot_id,
                                 state=state, experiment_id=experiment_id, ts=t))

    def _free_slot(self) -> _SlotState | None:
        for s in self.slots:
            if s.state == SlotState.idle and not s.closed:
                return s
        return None

    def _running_count(self) -> int:
        return sum(1 for s in self.slots if s.state == SlotState.busy)

    def _demand(self) -> int:
        if not self._can_launch():
            return 0
        k = self.n + self.lookahead - self._running_count() - len(self.buffer)
        if self.cfg.max_proposals_per_call is not None:
            k = min(k, self.cfg.max_proposals_per_call)
        return max(0, k)

    # ------------------------------------------------------------------ proposals and runs

    def _add_proposal(self, p: Proposal, *, is_baseline: bool = False) -> bool:
        try:
            config = self.spec.validate_config(dict(p.config))
        except ValueError as e:
            self.configs_rejected += 1
            log.info("%s: rejected config %s: %s", self.kind, p.config, e)
            return False
        exp_id = p.experiment_id if p.experiment_id not in self.exps else new_id()
        exp = _Exp(id=exp_id, arm_id=self.arm_id, config=config, hypothesis=p.hypothesis,
                   seed=self.cfg.seed, proposed_at=now(), is_baseline=is_baseline,
                   llm_call_id=p.llm_call_id, mubit_request_id=p.mubit_request_id,
                   label=f"#{len(self.order) + 1}")
        self.exps[exp.id] = exp
        self.order.append(exp.id)
        self.buffer.append(exp.id)
        self._store(exp)
        return True

    def _fill_slots(self) -> None:
        while self.buffer and self._live():
            t = now()
            if not self._can_launch(t):
                self._apply_cutoff(t)
                return
            s = self._free_slot()
            if s is None:
                return
            exp = self.exps[self.buffer.popleft()]
            exp.status, exp.started_at = ExperimentStatus.running, t
            exp.slot_id, exp.hardware = s.slot.slot_id, s.slot.hardware
            self._transition(s, SlotState.busy, t, exp.id)
            self._store(exp)
            self._emit(ExperimentStartedEvent(race_id=self.race_id, arm=self.kind, experiment_id=exp.id,
                                              slot_id=s.slot.slot_id, config=exp.config,
                                              hypothesis=exp.hypothesis, seed=exp.seed,
                                              hardware=s.slot.hardware, ts=t))
            self.run_tasks[exp.id] = asyncio.create_task(self._run_experiment(exp, s), name=f"run-{exp.id[:8]}")

    async def _run_experiment(self, exp: _Exp, s: _SlotState) -> None:
        started = exp.started_at or now()

        def on_metric(point: MetricPoint) -> None:
            if self.stopped:
                return
            if self.engine.median_rule is not None and point.name == self.engine.curve_metric:
                self._on_curve_point(exp, float(point.t), float(point.value))
            if self.db is not None:
                self.db.add_metric(exp.id, point.step, point.t, point.name, point.value)
            self._emit(MetricEvent(race_id=self.race_id, arm=self.kind, experiment_id=exp.id,
                                   step=point.step, t=point.t, name=point.name, value=point.value))

        try:
            handle = await self.runner.launch(self.spec.name, exp.config, exp.seed, self.run_budget_s, on_metric)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # infra: capacity, network, ...
            result = RunResult(score=None, status=ExperimentStatus.failed, error=f"infra: launch failed: {e!r}",
                               started_at=started, ended_at=now(), hardware=s.slot.hardware)
            exp.infra_error = True
        else:
            exp.handle, exp.run_ref = handle, handle.run_ref
            self._store(exp)
            if exp.early_stop is not None and not exp.cancel_sent:
                self._send_cancel(exp)   # the median rule fired before launch() returned
            try:
                result = await self.runner.wait(handle)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                result = RunResult(score=None, status=ExperimentStatus.failed, error=f"infra: wait failed: {e!r}",
                                   started_at=started, ended_at=now(), hardware=s.slot.hardware,
                                   run_ref=handle.run_ref)
                exp.infra_error = True
        self._on_run_end(exp, s, result, now())

    def _on_run_end(self, exp: _Exp, s: _SlotState, result: RunResult, t: float) -> None:
        self.run_tasks.pop(exp.id, None)
        exp.run_ref = result.run_ref or exp.run_ref
        exp.metrics = dict(result.metrics)
        exp.logs_tail = list(result.logs_tail[-50:])
        exp.is_trap = bool(result.is_trap) or self.spec.is_trap(exp.config)
        if not self._live(t):
            # Ended after the budget (or the arm was stopped): it does not count.
            if exp.early_stop is not None:
                exp.early_stop.setdefault("outcome", "race_ended")
            exp.status = ExperimentStatus.cancelled
            exp.ended_at = self.t_end if self.t_end is not None else min(t, self.deadline)
            exp.error = exp.error or "stopped at end of budget"
            exp.gpu_cost_usd = max(0.0, exp.ended_at - (exp.started_at or exp.ended_at)) * s.price_per_s
            self._store(exp)
            return
        status, score, error = result.status, result.score, result.error
        if status == ExperimentStatus.succeeded and (score is None or not math.isfinite(float(score))):
            status, score, error = ExperimentStatus.failed, None, error or f"non-finite score {score!r}"
        if exp.early_stop is not None:
            status, error = self._early_stop_outcome(exp, s, status, error, t, result.metrics)
        if status != ExperimentStatus.succeeded:
            score = None
        elif self.engine.median_rule is not None:
            exp.ckpt_values = self.engine.median_rule.completed_values(exp.curve, self.run_budget_s, self._time_scale)
            self._completed.append(exp.id)
        exp.status, exp.score, exp.error, exp.ended_at = status, (float(score) if score is not None else None), error, t
        exp.gpu_cost_usd = max(0.0, t - (exp.started_at or t)) * s.price_per_s
        self._transition(s, SlotState.idle, t, None)
        if not self._can_launch(t):
            self._apply_cutoff(t)
            self._release_slot(s, t)   # no run can start on it any more: stop paying now
        self._store(exp)
        self._emit(ExperimentFinishedEvent(race_id=self.race_id, arm=self.kind, experiment_id=exp.id,
                                           slot_id=s.slot.slot_id, status=exp.status, score=exp.score,
                                           is_trap=exp.is_trap, duration_s=t - (exp.started_at or t),
                                           gpu_cost_usd=exp.gpu_cost_usd, error=exp.error, ts=t))
        self.finished.append(exp.id)
        self._fill_slots()      # pipelined proposals start immediately
        self._wake.set()

    # ------------------------------------------------------------------ median stopping (engine rule)

    def _reference_ok(self, e: _Exp) -> bool:
        """A completed run the median rule may compare against: not judged invalid / flagged."""
        return not e.flagged_fake and e.verdict != Verdict.invalid

    def _on_curve_point(self, exp: _Exp, t: float, value: float) -> None:
        if not math.isfinite(value) or not math.isfinite(t):
            return
        exp.curve.append((t, value))
        if len(exp.curve) > 1 and exp.curve[-2][0] > t:
            exp.curve.sort(key=lambda p: p[0])
        rule = self.engine.median_rule
        if (rule is None or exp.early_stop is not None or exp.is_baseline or exp.status != ExperimentStatus.running
                or not self._live()):
            return
        for f, c in self._checkpoints:
            if f in exp.ckpt_checked:
                continue
            if t + 1e-9 < c:
                break
            exp.ckpt_checked.add(f)
            v = value_at(exp.curve, c)
            if v is None:            # the first point already lies past the checkpoint
                v = exp.curve[0][1]
            ref = [self.exps[i].ckpt_values[f] for i in self._completed
                   if f in self.exps[i].ckpt_values and self._reference_ok(self.exps[i])]
            stop, med, n = rule.should_stop(v, ref, self.spec.direction)
            self.median_checks += 1
            if n < rule.min_completed:
                self.median_skipped_few_completed += 1
                continue
            if stop:
                now_t = now()
                exp.early_stop = {
                    "rule": "median", "fraction": f, "checkpoint_t": c, "metric": self.engine.curve_metric,
                    "value": v, "median": med, "n_completed": n, "decided_at": now_t,
                    "decided_after_s": now_t - (exp.started_at or now_t),
                }
                self.early_stops_requested += 1
                if exp.handle is not None:
                    self._send_cancel(exp)
                return

    def _send_cancel(self, exp: _Exp) -> None:
        if exp.cancel_sent or exp.handle is None:
            return
        exp.cancel_sent = True
        task = asyncio.create_task(self._cancel_run(exp.handle), name=f"early-stop-{exp.id[:8]}")
        self._cancel_tasks.add(task)
        task.add_done_callback(self._cancel_tasks.discard)

    async def _cancel_run(self, handle: RunHandle) -> None:
        try:
            await self.runner.cancel(handle)
        except Exception:  # noqa: BLE001 - a failed cancel leaves the run to finish normally
            log.exception("%s: early-stop cancel failed", self.kind)

    def _early_stop_outcome(self, exp: _Exp, s: _SlotState, status: ExperimentStatus, error: str | None,
                            t: float, run_metrics: dict[str, float] | None = None) -> tuple[ExperimentStatus, str | None]:
        """Relabel a run as early-stopped only when the runner reports it cancelled. A run that
        finished first is kept ("too_late"); a genuine failure stays failed (and is diagnosed)."""
        es = exp.early_stop
        assert es is not None
        # Decision -> the runner reported the run over. On Modal this includes the wait for
        # Modal to confirm the input stopped (the slot is busy, and paid by this arm, until then).
        es["stop_latency_s"] = max(0.0, t - float(es.get("decided_at") or t))
        rm = run_metrics or {}
        if "cancel_lag_s" in rm:
            es["runner_cancel_lag_s"] = float(rm["cancel_lag_s"])
            es["cancel_confirmed"] = bool(rm.get("cancel_confirmed"))
        if status == ExperimentStatus.succeeded:
            es["outcome"] = "too_late"     # the run finished before the cancel landed: keep it
            return status, error
        if status != ExperimentStatus.cancelled:
            es["outcome"] = f"{status.value}_before_stop"   # e.g. diverged before the cancel landed
            return status, error
        dur = max(0.0, t - (exp.started_at or t))
        # Estimated full length = median wall time of the completed runs that reached the same
        # checkpoint (the rule's own reference set): on rl every run is ~15 s; on lcbench run
        # lengths vary 5-300 s by config, and short runs say nothing about a long one.
        f = es.get("fraction")
        durs = [(self.exps[i].ended_at or 0.0) - (self.exps[i].started_at or 0.0) for i in self._completed
                if f in self.exps[i].ckpt_values]
        ref = float(statistics.median(durs)) if durs else None
        freed = max(0.0, ref - dur) if ref is not None else None
        # Slot time freed for other runs (an estimate). Not money saved: the slot is still paid
        # for if nothing runs on it, so the $ figure is an equivalent, never a $ framing.
        es.update(outcome="stopped", stopped_after_s=dur, ref_duration_s=ref, slot_s_freed=freed,
                  freed_slot_usd_equiv=(freed * s.price_per_s) if freed is not None else None)
        msg = (f"early stop (median rule) at {es['fraction'] * 100:.0f}% of the run budget: "
               f"{es['metric']} {es['value']:.4g} worse than the median {es['median']:.4g} of "
               f"{es['n_completed']} completed runs")
        return ExperimentStatus.cancelled, msg

    # ------------------------------------------------------------------ researcher loop

    def _eligible(self, e: _Exp, *, shipped: bool = False) -> bool:
        """May ``e`` be a best? The planner's view (default: the LLM judge's verdict, D44) or,
        with ``shipped``, the decision that counts (the judge policy's, if the arm has one)."""
        if not e.processed or e.status != ExperimentStatus.succeeded or e.score is None or e.flagged_fake:
            return False
        if e.override_genuine:
            return True
        v = e.verdict if shipped else e.planner_verdict
        if v == Verdict.invalid:
            return False
        if v == Verdict.unsure and not self.cfg.accept_unsure:
            return False
        return True

    def _ships(self, e: _Exp) -> bool:
        return self._eligible(e, shipped=True)

    def _best_of(self, ok: Callable[[_Exp], bool]) -> str | None:
        best: _Exp | None = None
        for i in self.processed:
            e = self.exps[i]
            if ok(e) and (best is None or self.spec.is_better(e.score, best.score)):  # type: ignore[arg-type]
                best = e
        return best.id if best else None

    def _recompute_best(self) -> None:
        self.best_id = self._best_of(self._eligible)
        self.shipped_id = self._best_of(self._ships)

    def _digest(self, k: int) -> ResearchDigest:
        t = now()
        done = [self.exps[i] for i in self.processed]
        good = sorted((e for e in done if self._eligible(e)), key=lambda e: e.score,  # type: ignore[arg-type,return-value]
                      reverse=self.spec.direction == "max")
        running = [self.exps[i].config for i in self.order
                   if self.exps[i].status in (ExperimentStatus.running, ExperimentStatus.pending)]
        return ResearchDigest(
            k=k, n_slots=self.n, elapsed_s=t - self.t0, remaining_s=max(0.0, self.deadline - t),
            best=self.exps[self.best_id].snapshot() if self.best_id else None,
            leaderboard=[e.snapshot() for e in good[:5]],
            recent=[e.snapshot() for e in done[-8:]],
            running=running,
            history=[e.snapshot() for e in done],
            n_finished=sum(1 for e in done if e.status == ExperimentStatus.succeeded),
            n_failed=sum(1 for e in done if e.status in (ExperimentStatus.failed, ExperimentStatus.timed_out)),
        )

    async def _agent_loop(self) -> None:
        try:
            while not self.stopped:
                await self._wake.wait()
                self._wake.clear()
                while self.finished and not self.stopped:
                    await self._process(self.exps[self.finished.popleft()])
                if self.stopped:
                    return
                k = self._demand()
                if k <= 0:
                    continue
                proposals = await self.researcher.propose(self._digest(k), k)
                if self.stopped:
                    return
                added = sum(1 for p in proposals[:k] if self._add_proposal(p))
                self._fill_slots()
                if added == 0:
                    self.empty_batches += 1
                    await asyncio.sleep(EMPTY_BATCH_BACKOFF_S)
                self._wake.set()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("arm %s: researcher crashed", self.kind)
            self.error = f"{type(e).__name__}: {e}"
            self._stop("error", now())

    async def _process(self, exp: _Exp) -> None:
        best = self.exps[self.best_id] if self.best_id else None
        exp.summary = exp.full_summary(summarize_result(self.spec, exp.status, exp.score, best.score if best else None))
        exp.processed = True
        self.processed.append(exp.id)
        snap = exp.snapshot()
        self.researcher.observe(snap)
        if exp.status in (ExperimentStatus.failed, ExperimentStatus.timed_out) and not exp.infra_error:
            diag = await self.researcher.diagnose(snap)
            if diag is not None:
                exp.diagnosis = diag.model_dump(mode="json")
        elif exp.status == ExperimentStatus.succeeded and exp.summary["claimed_new_best"]:
            if not exp.is_baseline:
                claim = JudgeClaim(
                    candidate=snap, previous_best=best.snapshot() if best else None,
                    delta=exp.summary["delta"], improvement=exp.summary["improvement"],
                    config_diff=config_diff(best.config if best else None, exp.config),
                )
                out = await self.researcher.judge(claim)
                if self.engine.judge_rule is not None:
                    self._apply_judge_policy(exp, out, best)
                elif out is not None:
                    exp.verdict, exp.verdict_reason = out.verdict, out.reason
                    exp.llm_verdict = out.verdict
            # The agent's own best (planner view: the LLM judge's verdict, D44).
            if self._eligible(exp) and (self.best_id is None or
                                        self.spec.is_better(exp.score, self.exps[self.best_id].score)):  # type: ignore[arg-type]
                self.best_id = exp.id
            # The shipped model: a judged claim the decision that counts accepts (the judge
            # policy's, if any). Without a judge policy this is exactly the agent's best.
            if self._ships(exp) and (self.shipped_id is None or
                                     self.spec.is_better(exp.score, self.exps[self.shipped_id].score)):  # type: ignore[arg-type]
                prev = self.exps[self.shipped_id].score if self.shipped_id else None
                self.shipped_id = exp.id
                exp.is_best = True
                # ``score`` is the agent's (reported) claim; "reached the goal" is ground truth:
                # the held-out quality of a run the headline metrics count (never a trap run).
                row = exp.row(self.race_id, self.runner.name)
                q = quality(row)
                self._emit(NewBestEvent(
                    race_id=self.race_id, arm=self.kind, experiment_id=exp.id, score=exp.score,  # type: ignore[arg-type]
                    previous_best=prev, config=exp.config, hypothesis=exp.hypothesis,
                    elapsed_s=(exp.ended_at or now()) - self.t0,
                    reached_target=bool(is_counted(row) and q is not None and self.spec.reaches_target(q)),
                ))
        self._store(exp)

    def _apply_judge_policy(self, exp: _Exp, out: Any, best: _Exp | None) -> None:
        """D40: the LLM decides leakage, the seed-spread rule decides real vs noise. D44: the
        result only decides the shipped model (``exp.verdict``); the planner keeps seeing the
        LLM's verdict (``exp.llm_verdict``)."""
        rule = self.engine.judge_rule
        assert rule is not None and exp.summary is not None
        llm_v = getattr(out, "verdict", None) if out is not None else None
        try:
            exp.llm_verdict = Verdict(getattr(llm_v, "value", llm_v)) if llm_v is not None else None
        except ValueError:
            exp.llm_verdict = None
        llm_reason = None
        if out is not None:
            llm_reason = getattr(out, "reason", None) or getattr(out, "reason_code", None)
            llm_reason = getattr(llm_reason, "value", llm_reason)
        d = rule.decide(exp.summary.get("improvement"), best.score if best else None)
        final = combine(llm_v, d, llm_replied=out is not None)
        tags: list[str] = []
        if out is None:
            tags.append("no_leakage_check")   # nothing checked leakage: never confirmed as valid
        if d.degenerate:
            tags.append("degenerate_rule")    # sigma 0 (deterministic surrogate): not a real-gain test
        exp.judge_policy = {
            "llm_verdict": getattr(llm_v, "value", llm_v),
            "llm_reason": llm_reason,
            "llm_replied": out is not None,
            "rule": d.decision, "rule_reason": d.reason,
            "gain": d.gain if d.gain is not None and math.isfinite(d.gain) else None,
            "no_previous_best": best is None,
            "sigma": d.sigma, "k_sigma": rule.k_sigma,
            "degenerate": d.degenerate,
            "tags": tags,
            "final": final.value if final is not None else None,
        }
        exp.verdict = final
        if final == Verdict.invalid:
            exp.verdict_reason = f"LLM judge: invalid ({llm_reason})"
        elif out is None:
            exp.verdict_reason = "no usable LLM judge reply: no leakage check, so unsure"
        elif d.decision in ("real", "noise"):
            exp.verdict_reason = (f"seed-spread rule: {d.reason}"
                                  + (f"; LLM judge: {getattr(llm_v, 'value', llm_v)}" if llm_v is not None else ""))
        else:
            exp.verdict_reason = llm_reason
        exp.summary = exp.full_summary(exp.summary)

    # ------------------------------------------------------------------ shutdown

    async def _shutdown(self) -> None:
        if self.t_end is None:
            # The arm task was cancelled from outside (e.g. Ctrl-C / task.cancel()) without
            # request_stop(): still close the slots and cancel the runs so nothing keeps paying.
            self._stop("cancelled", now())
        assert self.t_end is not None
        for task in (self._agent_task, self._monitor_task):
            if task is not None and not task.done():
                task.cancel()
        if self._cancel_tasks:
            await asyncio.wait(list(self._cancel_tasks), timeout=SHUTDOWN_GRACE_S)
        for task in (self._agent_task, self._monitor_task):
            if task is not None:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        # Close every slot at t_end: the arm stops paying exactly there.
        for s in self.slots:
            self._transition(s, SlotState.idle, self.t_end, None)
            s.closed = True
        # Cancel runs still in flight; they report back as cancelled.
        handles = [e.handle for e in self.exps.values() if e.status == ExperimentStatus.running and e.handle]
        if handles:
            await asyncio.gather(*(self.runner.cancel(h) for h in handles), return_exceptions=True)
        pending = [t for t in self.run_tasks.values() if not t.done()]
        if pending:
            _, still = await asyncio.wait(pending, timeout=SHUTDOWN_GRACE_S)
            for t in still:
                t.cancel()
            if still:
                # Bounded: a run task stuck in a runner call that ignores cancellation must not
                # hang the race (seen with Modal at 64 concurrent runs). Its experiment is
                # marked cancelled below either way.
                _, stuck = await asyncio.wait(still, timeout=SHUTDOWN_GRACE_S)
                if stuck:
                    log.warning("arm %s: %d run task(s) did not finish after cancel; abandoned", self.kind, len(stuck))
        for e in self.exps.values():
            if e.status in (ExperimentStatus.running, ExperimentStatus.pending):
                if e.status == ExperimentStatus.running:
                    e.ended_at = self.t_end
                    price = next((s.price_per_s for s in self.slots if s.slot.slot_id == e.slot_id), 0.0)
                    e.gpu_cost_usd = max(0.0, self.t_end - (e.started_at or self.t_end)) * price
                e.status = ExperimentStatus.cancelled
                e.error = e.error or "stopped at end of budget"
                if e.early_stop is not None:
                    e.early_stop.setdefault("outcome", "race_ended")
                    e.summary = e.full_summary(e.summary or {})
                self._store(e)
        # Results that finished in budget but were not processed yet: code-only summary.
        while self.finished:
            e = self.exps[self.finished.popleft()]
            best = self.exps[self.best_id] if self.best_id else None
            e.summary = e.full_summary(summarize_result(self.spec, e.status, e.score, best.score if best else None))
            self._store(e)
        if self.db is not None:
            self.db.flush_metrics()


# =========================================================================== race engine


class RaceEngine:
    """Runs all arms of one race simultaneously with identical resources."""

    def __init__(
        self,
        cfg: RaceConfig,
        arms: list[ArmDeps],
        *,
        spec: TargetSpec | None = None,
        race_id: str | None = None,
        db: Database | None = None,
        bus: EventBus | None = None,
        tick_s: float = 0.05,
    ) -> None:
        self.cfg = cfg
        self.spec = race_spec(cfg, spec)
        self.race_id = race_id or new_id()
        self.db = db
        self.bus = bus or EventBus()
        self.tick_s = tick_s
        self.arm_deps = arms
        self.arms: dict[ArmKind, ArmRunner] = {}
        self.arm_ids: dict[ArmKind, str] = {d.kind: new_id() for d in arms}
        self.summaries: list[ArmSummary] = []
        self.reports: dict[str, dict[str, Any]] = {}
        self.status = "created"
        self.t0: float | None = None
        self.finished_event = asyncio.Event()
        # Set by the caller before run() when the container pool was warmed before t0
        # (bench.py prewarm_pool); stored in the manifest, never in busy/idle/$.
        self.prewarm_info: dict[str, Any] | None = None
        self._slots = self._assign_slots()
        self._check_symmetry()
        self._check_record_modes()
        self._check_project_routing()
        # Corrections applied mid-race (benchmark races refuse them unless the config says
        # "record"; each one is listed in the fairness manifest so the headline blocks the race).
        self.corrections: list[dict[str, Any]] = []
        # The sigma key compares run_budget_s with the REGISTERED target's default budget (as
        # bench.goal_key does), never with the passed spec's: bench/app pass race_spec(cfg), whose
        # budget_s already is run_budget_s (R41 bug: rl@15s races used the 30 s "rl" table).
        default_budget = target_default_budget_s(cfg, spec)
        self.engine_states: dict[ArmKind, EngineState] = {
            d.kind: resolve_engine_state(cfg, d.kind, self.spec, default_budget_s=default_budget)
            for d in arms}
        self.engine_checks = self._check_engine_features()

    # ------------------------------------------------------------------ setup checks

    def _assign_slots(self) -> dict[ArmKind, list[Slot]]:
        kinds = [d.kind for d in self.arm_deps]
        if sorted(kinds) != sorted(self.cfg.arms):
            raise AsymmetricRaceError(f"arms {kinds} do not match the config {list(self.cfg.arms)}")
        if self.cfg.strict:
            for d in self.arm_deps:
                if d.llm_mode in ("mock", "replay") or d.kind == ArmKind.mock:
                    raise BenchmarkIntegrityError(f"benchmark race: arm {d.kind} uses {d.llm_mode}")
                for attr in ("llm", "fallback_llm"):
                    client = getattr(d.researcher, attr, None)
                    if client is not None and getattr(client, "benchmark_safe", True) is False:
                        raise BenchmarkIntegrityError(f"benchmark race: arm {d.kind} {attr} is not benchmark-safe")
        n = self.cfg.gpus_per_arm
        out: dict[ArmKind, list[Slot]] = {}
        shared: dict[int, int] = defaultdict(int)  # id(runner) -> slots already handed out
        for d in self.arm_deps:
            all_slots = d.runner.slots()
            start = shared[id(d.runner)]
            mine = all_slots[start : start + n]
            if len(mine) < n:
                raise AsymmetricRaceError(f"runner {d.runner.name} offers {len(all_slots) - start} free slots for "
                                          f"arm {d.kind}, need {n}")
            shared[id(d.runner)] += n
            out[d.kind] = list(mine)
        return out

    def _check_symmetry(self) -> None:
        sigs = {}
        for d in self.arm_deps:
            slots = self._slots[d.kind]
            hw = tuple(s.hardware for s in slots)
            if "H100" in hw and not self.cfg.allow_h100:
                raise AsymmetricRaceError("H100 slots need allow_h100=True (budget guard)")
            prices = tuple(float(d.runner.price_per_second(h)) for h in hw)
            sigs[d.kind] = (hw, prices)
        if len(set(sigs.values())) > 1:
            raise AsymmetricRaceError(f"arms get different hardware/prices: {sigs}")
        if self.cfg.max_estimated_gpu_usd is not None and sigs:
            hw, prices = next(iter(sigs.values()))
            est = estimate_gpu_cost_usd(self.cfg, sum(prices) / max(1, len(prices)))
            if est > self.cfg.max_estimated_gpu_usd:
                raise AsymmetricRaceError(
                    f"estimated GPU ${est:.2f} exceeds max_estimated_gpu_usd ${self.cfg.max_estimated_gpu_usd:.2f}")

    def _check_record_modes(self) -> None:
        """Every LLM arm's researcher must send the mode the config gives that arm (a
        benchmark race refuses to start otherwise). Fills ``ArmDeps.record_mode``."""
        for d in self.arm_deps:
            if d.kind == ArmKind.optuna or not hasattr(d.researcher, "record"):
                continue
            want = self.cfg.record_mode_for(d.kind)
            got = normalize_record(getattr(d.researcher, "record", None))
            if d.record_mode is None:
                d.record_mode = got
            if self.cfg.strict and (got != want or d.record_mode != want):
                raise BenchmarkIntegrityError(
                    f"arm {d.kind.value}: researcher sends record={got!r} (deps say {d.record_mode!r}), "
                    f"config says {want!r}")

    def _check_project_routing(self) -> None:
        """Benchmark races: only the mubit arm may route steps to a non-default Mubit project
        (D38's project B). Every other LLM arm sends every step to the default project."""
        if not self.cfg.strict:
            return
        for d in self.arm_deps:
            if d.kind == ArmKind.mubit:
                continue
            for attr in ("llm", "fallback_llm"):
                routing = getattr(getattr(d.researcher, attr, None), "projects", None)
                if routing is None or not hasattr(routing, "arm_map"):
                    continue
                off = {s: p for s, p in routing.arm_map(d.kind.value).items() if p != routing.default}
                if off:
                    raise BenchmarkIntegrityError(
                        f"benchmark race: arm {d.kind.value} {attr} routes steps away from the default Mubit "
                        f"project {routing.default!r}: {off} (only the mubit arm may route)")

    def mubit_projects(self) -> dict[str, Any] | None:
        """Which Mubit project each arm's steps go to (``mubit_client.mubit_projects_manifest``)."""
        try:
            from backend.agent.mubit_client import mubit_projects_manifest
        except Exception:  # noqa: BLE001 - LLM SDK not installed (pure engine tests)
            return None
        return mubit_projects_manifest({d.kind: getattr(d.researcher, "llm", None) for d in self.arm_deps
                                        if d.kind != ArmKind.optuna})

    def _check_engine_features(self) -> list[str]:
        """Arms that should share engine features must (refused otherwise). Returns the checks."""
        return check_engine_features(self.cfg, self.engine_states, self.spec.name)

    def engine_manifest(self) -> dict[str, Any]:
        out: dict[str, Any] = {"features": {k.value: list(st.features) for k, st in self.engine_states.items()},
                               "checks": list(self.engine_checks), "per_arm": {}}
        for k, st in self.engine_states.items():
            a = self.arms.get(k)
            row: dict[str, Any] = {"features": list(st.features), "notes": list(st.notes)}
            if st.median_rule is not None:
                row["median_stop"] = {**st.median_rule.manifest(), "curve_metric": st.curve_metric,
                                      "time_scale": a._time_scale if a else None,
                                      "checkpoints_t": [c for _, c in a._checkpoints] if a else None}
            if st.judge_rule is not None:
                row["judge_policy"] = st.judge_rule.manifest()
            out["per_arm"][k.value] = row
        return out

    def ablation_manifest(self) -> dict[str, Any] | None:
        """An ablation race (or one with an ablation-only arm): never a headline race. Names
        what each ablation arm is, so no reader mistakes it for Mubit's models."""
        kinds = [d.kind for d in self.arm_deps]
        if not self.cfg.is_ablation and not any(k in ABLATION_ONLY_ARMS for k in kinds):
            return None
        return {"headline": False,
                "note": "ablation race: never a headline result",
                "arms": {d.kind.value: {"model": d.model, "label": FAST_ARM_LABEL}
                         for d in self.arm_deps if d.kind in ABLATION_ONLY_ARMS}}

    def fairness_manifest(self) -> dict[str, Any]:
        """Per-arm settings plus proof they are equal (everything except the researcher)."""
        per_arm = {k.value: a.fairness() for k, a in self.arms.items()}
        values = list(per_arm.values())
        identical = all(v == values[0] for v in values)
        policies = {k.value: p for k, a in self.arms.items() if (p := a.llm_policy()) is not None}
        pvals = list(policies.values())
        return {"identical": identical, "per_arm": per_arm,
                "llm_policy": policies, "llm_policy_identical": all(p == pvals[0] for p in pvals),
                "differs_only_in": {k.value: {"researcher": type(a.researcher).__name__,
                                              "model": a.deps.model, "llm_mode": a.deps.llm_mode,
                                              "record_mode": a.deps.record_mode,
                                              "engine_features": list(a.engine.features)}
                                    for k, a in self.arms.items()},
                "engine": self.engine_manifest(),
                "mode": self.cfg.mode,
                "ablation": self.ablation_manifest(),
                "record_mode_note": "per-arm x-middleware-record; off/metadata routing semantics pending "
                                    "Mubit confirmation (docs/mubit_office_hours.md item 2)",
                "goal": {"target_value": self.spec.target_value, "provisional": self.cfg.provisional,
                         "from_config": self.cfg.target_value is not None},
                "prewarm": self.prewarm_info,
                "mubit_projects": self.mubit_projects(),
                "corrections_policy": "allowed (dev)" if not self.cfg.strict else self.cfg.benchmark_corrections,
                "corrections": [dict(c) for c in self.corrections]}

    # ------------------------------------------------------------------ run

    async def run(self) -> list[ArmSummary]:
        if self.status != "created":
            raise RuntimeError(f"race {self.race_id} already {self.status}")
        self.status = "running"
        if self.db is not None and self.db.get_race(self.race_id) is None:
            self.db.insert_race(race_id=self.race_id, target=self.cfg.target, budget_s=self.cfg.budget_s,
                                gpus_per_arm=self.cfg.gpus_per_arm, seed=self.cfg.seed,
                                config=self.cfg.model_dump(mode="json"), name=self.cfg.name, mode=self.cfg.mode,
                                llm_mode=self.cfg.llm, runner=self.cfg.runner)
        t0 = now()
        self.t0 = t0
        deadline = t0 + self.cfg.budget_s
        for d in self.arm_deps:
            arm_id = self.arm_ids[d.kind]
            slots = self._slots[d.kind]
            runner_arm = ArmRunner(race_id=self.race_id, arm_id=arm_id, deps=d, cfg=self.cfg, spec=self.spec,
                                   slots=slots, t0=t0, deadline=deadline, db=self.db, bus=self.bus, tick_s=self.tick_s,
                                   engine=self.engine_states[d.kind])
            self.arms[d.kind] = runner_arm
            if self.db is not None:
                self.db.insert_arm(arm_id=arm_id, race_id=self.race_id, kind=d.kind.value, model=d.model,
                                   llm_mode=d.llm_mode, n_slots=len(slots), hardware=slots[0].hardware,
                                   price_per_s=runner_arm.slots[0].price_per_s)
                self.db.update_arm(arm_id, status="running", started_at=t0)
        manifest = self.fairness_manifest()
        if not manifest["identical"]:
            raise AsymmetricRaceError(f"arms are not identical: {manifest}")
        if self.cfg.strict and not manifest["llm_policy_identical"]:
            raise AsymmetricRaceError(f"LLM arms differ in prompts / retry / fallback / record: {manifest['llm_policy']}")
        if self.db is not None:
            self.db.update_race(self.race_id, status="running", started_at=t0, fairness=manifest)
        reason = "budget"
        try:
            await asyncio.gather(*(a.run() for a in self.arms.values()))
        except asyncio.CancelledError:
            for a in self.arms.values():
                a.request_stop("cancelled")
            reason = "cancelled"
            raise
        finally:
            reasons = {a.stop_reason for a in self.arms.values()}
            if reason != "cancelled":
                if "error" in reasons:
                    reason = "error"
                elif "cancelled" in reasons:
                    reason = "cancelled"
                elif "cost_cap" in reasons:
                    reason = "cost_cap"
            self._finish(reason)
        return self.summaries

    def _finish(self, reason: str) -> None:
        self.summaries = []
        for kind, a in self.arms.items():
            summary, extras = a.live_summary()
            self.summaries.append(summary)
            self.reports[kind.value] = extras
            if self.db is not None:
                self.db.update_arm(a.arm_id, status="finished", stop_reason=a.stop_reason, ended_at=a.t_end,
                                   summary=summary.model_dump(mode="json"), report=extras)
        t_end = max((a.t_end or now()) for a in self.arms.values()) if self.arms else now()
        self.status = "finished" if reason in ("budget", "cost_cap") else reason
        if self.db is not None:
            self.db.update_race(self.race_id, status=self.status, finish_reason=reason, ended_at=t_end)
            self.db.flush_metrics()
        self.bus.publish(RaceFinishedEvent(race_id=self.race_id, summaries=self.summaries, reason=reason))
        self.finished_event.set()

    # ------------------------------------------------------------------ live control

    def live_summaries(self) -> list[tuple[ArmSummary, dict[str, Any]]]:
        return [a.live_summary() for a in self.arms.values()]

    def stop(self, reason: str = "cancelled") -> None:
        for a in self.arms.values():
            a.request_stop(reason)

    def find_arm(self, experiment_id: str) -> ArmRunner | None:
        for a in self.arms.values():
            if experiment_id in a.exps:
                return a
        return None

    def corrections_allowed(self) -> bool:
        return self.cfg.corrections_allowed

    def apply_correction(self, experiment_id: str, label: CorrectionLabel, *, correction_id: str,
                         source: CorrectionSource = CorrectionSource.human, reason: str = "") -> ArmKind | None:
        """Apply a human/verifier correction to the live race. Benchmark races refuse
        (``CorrectionRefusedError``) unless the config says ``benchmark_corrections="record"``;
        then the correction is listed in the fairness manifest (``corrections``), which blocks
        the race from headline numbers."""
        if not self.corrections_allowed():
            raise CorrectionRefusedError(
                f"race {self.race_id} is a benchmark race (benchmark_corrections='refuse'): a correction "
                f"would change the outcome by hand")
        arm = self.find_arm(experiment_id)
        if arm is None:
            return None
        t = now()
        self.corrections.append({
            "correction_id": correction_id, "experiment_id": experiment_id, "arm": arm.kind.value,
            "label": label.value, "source": source.value, "reason": reason, "at": t,
            "elapsed_s": (t - self.t0) if self.t0 is not None else None, "race_status": self.status,
        })
        if self.db is not None and self.arms:
            self.db.update_race(self.race_id, fairness=self.fairness_manifest())
        arm.apply_correction(experiment_id, label, recorded=True)
        self.bus.publish(CorrectionEvent(race_id=self.race_id, correction_id=correction_id,
                                         experiment_id=experiment_id, arm=arm.kind, source=source,
                                         label=label, reason=reason))
        return arm.kind
