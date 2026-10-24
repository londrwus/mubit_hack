"""Race configuration and the frozen benchmark presets.

``RaceConfig`` is the single object that decides everything an arm gets. The engine hands
every arm the *same* values (slots, hardware, wall-clock, $ cap, seed, per-run budget,
batching cap, pipelining, baseline-first), so fairness is a property of the type, not of
care at call sites.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.core.types import ArmKind

OPUS_MODEL = "claude-opus-5-5"  # mirrors backend.agent.llm_base.OPUS_MODEL (kept import-light)
# The ablation arm's model (mirrors backend.agent.llm_base.DEV_MODEL): a smaller, faster provider
# model in the SAME loop, through Mubit with record off. Never a Mubit adapter.
FAST_MODEL = "claude-haiku-5-5"
FAST_ARM_LABEL = "smaller faster model in the same loop, not a Mubit adapter"
# Arms whose provider model is fixed by the arm kind (RaceConfig.model_for); every other LLM
# arm sends ``RaceConfig.model``.
ARM_FIXED_MODELS: dict[ArmKind, str] = {ArmKind.fast: FAST_MODEL}
# Arms that may only race in an ablation (or dev) race, never in a benchmark / headline race.
ABLATION_ONLY_ARMS: frozenset[ArmKind] = frozenset({ArmKind.fast})
# Prompt-ablation arms (dev mode only, never a benchmark): the SAME model as the Opus arm with a
# different propose prompt version and effort, so the strict "identical LLM policy" check would
# (rightly) refuse them in an ablation/benchmark race. Applied by scripts/bench.py (run_race).
LEAN_ARM_LABEL = "Opus 5.5 with the lean planner prompt"
# (propose prompt version, propose effort). STANDARD_PROPOSE mirrors the frozen default
# (backend.agent.prompts.CURRENT_VERSION + mubit_client.DEFAULT_STEP_CONFIGS; kept import-light,
# a test checks the mirror). LEAN_PLANNER_PROPOSE is the lean planner (R62 probe, R63 quality
# check): draft prompt v2 "changes only" (fingerprint cfd2e3dd9710) + effort low.
STANDARD_PROPOSE: tuple[str, str] = ("v1", "medium")
LEAN_PLANNER_PROPOSE: tuple[str, str] = ("v2", "low")
ARM_PROPOSE_VARIANTS: dict[ArmKind, tuple[str, str]] = {ArmKind.lean: LEAN_PLANNER_PROPOSE}  # (prompt version, effort)
PROMPT_ABLATION_ARMS: frozenset[ArmKind] = frozenset(ARM_PROPOSE_VARIANTS)

# x-middleware-record values (mirrors backend.agent.mubit_client.RecordMode, kept import-light).
#   full     = Mubit's default: the call is recorded and can become a kept example.
#   metadata = keeps the call's task fingerprint (so Mubit can match it to a task and route it
#              to an adapter) but drops the request/reply content (never a kept example).
#   off      = drops the content AND the task fingerprint (docs.mubit.ai/reference/api
#              #request-headers-mubit-reads), so the call cannot be matched to a task.
# PENDING Mubit's confirmation (docs/mubit_office_hours.md item 2): that "off" guarantees a
# provider (Opus) answer and that "metadata" is still routable is our reading of the docs,
# not something Mubit has confirmed.
RecordMode = Literal["full", "metadata", "off"]

# Benchmark defaults: the Opus arm must stay pure Opus, so it sends "off" (no fingerprint, so
# no routing to an adapter); the Mubit arm sends "metadata" (fingerprint kept, so routing
# can happen, content not stored). Fallback-to-Opus calls inherit their arm's mode.
# "Busy Machines with Opus" (opus_bm) is Opus on every step like the Opus arm, so it sends "off"
# too; presets that include it name it explicitly.
BENCH_RECORD_MODES: dict[ArmKind, RecordMode] = {ArmKind.opus: "off", ArmKind.mubit: "metadata"}

# Busy Machines engine features (D36, D40). Engine code, never an LLM step, so they change no
# prompt and no Mubit task:
#   judge_policy  the LLM judge decides leakage ("invalid" stays invalid); the calibrated
#                 seed-spread rule (backend.benchmark.engine_rules.SeedSpreadRule, the R37
#                 'rule_calib') decides real-vs-noise for every other claim.
#   median_stop   at fixed fractions of a run's budget, stop the run if its curve value is worse
#                 than the median of completed runs at the same point (needs >= N completed).
#                 OFF by default for every arm and target (D43: no rule passes the D42 offline
#                 gate, R43); kept behind this flag (RaceConfig.engine_features, dev races only:
#                 benchmark races run each arm's defaults).
#   backfill      keeps GPUs busy with a no-AI sampler while the AI thinks (backfill gate,
#                 data/backfill/): every slot the LLM's own proposals do not fill gets an Optuna
#                 TPE config at once. The LLM's proposals always launch first (its demand counts
#                 only its own runs), the sampler is told every finished run of the arm (LLM's
#                 and its own) and, for a claim the LLM judge rejected, the WORST score seen so
#                 far ("checker-gated feedback"; told the raw inflated score TPE finds the trap).
#                 The sampler is never told which option is the trap: fakes are caught only by
#                 the arm's checker (LLM judge + judge policy). Sampler runs render in the
#                 planner digest exactly like LLM runs. Not for the optuna arm.
#   lean_planner  shorter planner replies (D46, R62/R63): the arm's propose call uses the lean
#                 prompt v2 ("changes only": each experiment lists only the settings that differ
#                 from the current best config, expanded to a full config in code) at effort low
#                 (LEAN_PLANNER_PROPOSE). The ONE prompt difference a strict race allows between
#                 LLM arms (scheduler.check_llm_policies): only the propose prompt version and
#                 its effort, only on arms with this feature; the judge / diagnose prompts and
#                 the user message (what the planner sees) stay identical. Unlike the others it
#                 changes a prompt (a new Mubit task fingerprint for propose), which is why it
#                 is declared per arm and disclosed in the fairness manifest. Not for optuna.
#   planner_cadence  (D47) the arm's AI planner is called at most every N seconds; the no-AI
#                 sampler (backfill) keeps GPUs busy in between. Parameter
#                 ``min_propose_interval_s`` (RaceConfig.feature_params), measured between the
#                 STARTS of the arm's propose calls (a call's retries are part of that call). When a
#                 call is due, K = min(max_proposals_per_call, LLM demand) at that moment, where LLM
#                 demand counts only the LLM's own runs and buffered proposals (scheduler._demand
#                 with backfill). Judge and diagnose calls are never throttled. Needs backfill on the
#                 same arm (without it the slots would starve between planner calls). OFF by default
#                 for every arm; a preset turns it on per arm (PRESET_DECLARED_FEATURES), and the
#                 parameter is recorded in the fairness manifest (per_arm.feature_params, engine).
EngineFeature = Literal["judge_policy", "median_stop", "backfill", "lean_planner", "planner_cadence"]
ENGINE_FEATURES: tuple[str, ...] = ("judge_policy", "median_stop", "backfill", "lean_planner", "planner_cadence")
# Plain-language labels (UI copy, docs/demo.md): what each feature does, no jargon.
ENGINE_FEATURE_LABELS: dict[str, str] = {
    "judge_policy": "checks every claimed improvement (AI for leaks, seed-spread rule for noise)",
    "median_stop": "stops runs that are clearly worse than the median (off by default)",
    "backfill": "keeps GPUs busy with a no-AI sampler while the AI thinks",
    "lean_planner": "shorter planner replies: only the settings it changes",
    "planner_cadence": "calls the AI planner at most every N seconds; the no-AI sampler keeps GPUs busy in between",
}
# Features that take parameters (RaceConfig.feature_params[arm][feature]): name -> required params.
FEATURE_PARAMS: dict[str, tuple[str, ...]] = {"planner_cadence": ("min_propose_interval_s",)}
# Features a strict (benchmark / ablation) race may switch on per arm ON TOP of the arm's
# defaults, because a preset declares them with their parameters (manifest + fingerprint carry
# them, so headline.py never pools such a race with one that ran without them). Every other
# feature in a strict race must be exactly the arm kind's default.
PRESET_DECLARED_FEATURES: frozenset[str] = frozenset({"planner_cadence"})
# Features that only make sense together with another feature on the same arm.
FEATURE_REQUIRES: dict[str, str] = {"planner_cadence": "backfill"}


def cadence_label(min_propose_interval_s: float | None) -> str:
    """Plain-language label of the planner cadence with its parameter (UI copy, reports)."""
    if min_propose_interval_s is None:
        return ENGINE_FEATURE_LABELS["planner_cadence"]
    return (f"calls the AI planner at most every {min_propose_interval_s:g} s; the no-AI sampler keeps GPUs "
            "busy in between")
# Backfill sampler settings (the same on every arm that has the feature; manifest "engine").
# n_startup_trials matches the Optuna arm's TPE (OptunaResearcher default), as gated.
BACKFILL_SAMPLER = "optuna-tpe"
BACKFILL_N_STARTUP_TRIALS = 10
BACKFILL_FEEDBACK = "worst_on_rejected"   # checker-gated: a rejected claim is told the worst score so far
# Per arm kind, unless RaceConfig.engine_features overrides it. The Opus arm is the standard
# agent (no features: the plain v1 loop at effort medium, D46). "Busy Machines with Opus" and
# "Mubit + Busy Machines" get the SAME features (only the LLM behind the calls differs): the
# judge policy (D43, D44: it decides the SHIPPED model, never what the planner sees), backfill
# (the backfill gate) and the lean planner (D46). Optuna gets none (D43: it dropped the median
# rule to stay matched with the BM arms). The ablation arms "fast" and "lean" are the standard
# agent too ("lean" gets its prompt from ARM_PROPOSE_VARIANTS, not from a feature).
DEFAULT_ENGINE_FEATURES: dict[ArmKind, tuple[str, ...]] = {
    ArmKind.opus: (),
    ArmKind.opus_bm: ("backfill", "judge_policy", "lean_planner"),
    ArmKind.mubit: ("backfill", "judge_policy", "lean_planner"),
    ArmKind.optuna: (),
    ArmKind.mock: (),
    ArmKind.fast: (),
    ArmKind.lean: (),
}
# Arms that must get identical engine features when they meet in one race.
BUSY_MACHINES_ARMS: frozenset[ArmKind] = frozenset({ArmKind.opus_bm, ArmKind.mubit})


class RaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = ""
    target: str
    arms: list[ArmKind] = Field(default_factory=lambda: [ArmKind.opus, ArmKind.mubit, ArmKind.optuna])
    budget_s: float = Field(gt=0, description="Wall-clock budget per arm (all arms start together).")
    gpus_per_arm: int = Field(ge=1, le=100)
    seed: int = 0
    cost_cap_usd: float | None = Field(default=None, gt=0, description="Per-arm GPU $ + LLM $ cap.")
    run_budget_s: int | None = Field(default=None, ge=1, description="Per-run training budget; default spec.budget_s.")
    pipelining: bool = False          # ask for the next batch while runs are in flight (same for all arms)
    run_baseline_first: bool = True   # slot 0 starts the default (starting-code) config at t0, every arm
    max_proposals_per_call: int | None = Field(default=None, ge=1)  # cap on K
    accept_unsure: bool = True        # judge "unsure" still counts as a new best
    # Deadline rule (optimisation_analysis.md section 1e, D32), same for every arm: a run is
    # launched only while at least this many seconds of the race remain (one run's wall time:
    # run budget + launch/eval/return overhead, measured at calibration). From then on the arm
    # asks for no more proposals and releases each slot as soon as it is idle (it stops paying
    # for it: a released slot is neither busy nor waiting). None = off (runs launched late are
    # cancelled at the deadline, as before).
    launch_cutoff_s: float | None = Field(default=None, gt=0)
    # Pre-warm (optimisation_analysis.md section 1a): before t0, bench.py pins and warms one
    # container per slot of every arm (plus this many spare) and starts the race clock only
    # when all are warm. Pre-warm time is outside [t0, t_end], so never busy/idle/race $.
    prewarm: bool = False
    prewarm_spare: int = Field(default=0, ge=0, le=16)
    # dev: mock/replay allowed, relaxed checks. benchmark: the headline races (strict checks).
    # ablation: the same strict checks as a benchmark (real LLM through Mubit, record modes,
    # identical LLM policy, default engine features, corrections refused) but NEVER a headline
    # race (stored as is_dev=1; scripts/headline.py refuses it). Only mode that may race the
    # ABLATION_ONLY_ARMS.
    mode: Literal["dev", "benchmark", "ablation"] = "dev"
    llm: Literal["mock", "mubit"] = "mock"
    runner: Literal["local", "modal", "surrogate"] = "local"  # surrogate: workers/runners/surrogate.py
    record: bool = False              # legacy, all arms: False => "off", True => "full"
    # Per-arm recording mode; overrides ``record`` for the arms it names (see RecordMode).
    record_mode: dict[ArmKind, RecordMode] = Field(default_factory=dict)
    # Goal (``target_value``) for time-to-target / $-to-target. None => the target spec's
    # value. bench.py sets it from data/calibration/target_values.json (frozen) or from
    # --target-value, in which case ``provisional`` is True (not a frozen goal).
    target_value: float | None = None
    provisional: bool = False
    model: str = OPUS_MODEL           # every LLM arm's model except ARM_FIXED_MODELS (see model_for)
    allow_h100: bool = False
    max_estimated_gpu_usd: float | None = Field(default=None, gt=0, description="Refuse to start above this estimate.")
    # Dev only (llm="mock"): simulated seconds per LLM call per arm, e.g. {"opus": 8, "mubit": 0.3},
    # to rehearse the "agent waits less" story without spending on LLM calls. Mock numbers only.
    mock_latency_s: dict[ArmKind, float] = Field(default_factory=dict)
    mock_trap_rate: float | None = Field(default=None, ge=0, le=1)
    # Engine features per arm (see DEFAULT_ENGINE_FEATURES); an entry overrides that arm's default.
    engine_features: dict[ArmKind, list[EngineFeature]] = Field(default_factory=dict)
    # Per-arm feature parameters, e.g. {opus_bm: {"planner_cadence": {"min_propose_interval_s": 90}}}
    # (FEATURE_PARAMS lists the required names). Only for features the arm runs; recorded in the
    # fairness manifest (per_arm.feature_params, differs_only_in, engine.per_arm).
    feature_params: dict[ArmKind, dict[EngineFeature, dict[str, float]]] = Field(default_factory=dict)
    # Surrogate pacing for the WHOLE process (BUSY_SURROGATE_RUNTIME_MULT, workers/runners/surrogate.py):
    # None = leave the environment alone (multiplier 1.0 unless the env var is set). A value is
    # applied process-wide before any runner is built (apply_surrogate_runtime_mult), so every arm
    # gets it, and every run_ref shows it ("surrogate:<id>:x0.1"). Surrogate runner only.
    surrogate_runtime_mult: float | None = Field(default=None, gt=0)
    # Median stopping (engine rule, D40), the same for every arm that has it: checkpoints as
    # fractions of the per-run budget, and how many completed runs must exist before a run can
    # be stopped.
    median_stop_fractions: list[float] = Field(default_factory=lambda: [0.25, 0.5])
    median_stop_min_completed: int = Field(default=5, ge=1)
    # Mid-race corrections (POST /api/experiments/{id}/flag) in a benchmark race: "refuse"
    # (default: a human must not change a benchmark outcome) or "record" (applied and listed in
    # the fairness manifest's "corrections", which blocks the race from headline numbers; the
    # live demo uses this to show a correction turning into a model update). Dev races always
    # apply them.
    benchmark_corrections: Literal["refuse", "record"] = "refuse"

    @model_validator(mode="after")
    def _check(self) -> RaceConfig:
        if not self.arms:
            raise ValueError("a race needs at least one arm")
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("duplicate arm kinds")
        if self.strict:
            if self.llm != "mubit":
                raise ValueError(f"{self.mode} races must use the real LLM through Mubit (llm='mubit')")
            if ArmKind.mock in self.arms:
                raise ValueError(f"{self.mode} races cannot include the mock arm")
        if self.mode == "benchmark":
            bad = sorted(a.value for a in self.arms if a in ABLATION_ONLY_ARMS)
            if bad:
                raise ValueError(f"arm(s) {bad} are ablation-only ({FAST_ARM_LABEL}); never in a benchmark race")
        prompt_arms = [a for a in self.arms if a in PROMPT_ABLATION_ARMS]
        if prompt_arms and self.mode != "dev":
            raise ValueError(f"arm(s) {sorted(a.value for a in prompt_arms)} ({LEAN_ARM_LABEL}) use a draft propose "
                             "prompt: dev-mode prompt ablations only")
        for a in prompt_arms:
            if a in self.arms and self.record_mode_for(a) != "off":
                raise ValueError(f"arm {a.value!r} uses a draft propose prompt: its record mode must be 'off'")
        if (self.mock_latency_s or self.mock_trap_rate is not None) and self.llm != "mock":
            raise ValueError("mock_latency_s / mock_trap_rate only apply to llm='mock'")
        if any(v < 0 for v in self.mock_latency_s.values()):
            raise ValueError("mock latencies must be >= 0")
        for kind in self.record_mode:
            if kind not in self.arms:
                raise ValueError(f"record_mode names arm {kind.value!r}, which is not in the race")
            if kind == ArmKind.optuna:
                raise ValueError("the optuna arm makes no LLM calls; it has no record_mode")
        if self.target_value is not None and not math.isfinite(self.target_value):
            raise ValueError("target_value must be finite")
        if self.launch_cutoff_s is not None and self.launch_cutoff_s >= self.budget_s:
            raise ValueError("launch_cutoff_s must be shorter than the race budget")
        for kind, feats in self.engine_features.items():
            if kind not in self.arms:
                raise ValueError(f"engine_features names arm {kind.value!r}, which is not in the race")
            if len(set(feats)) != len(feats):
                raise ValueError(f"engine_features for {kind.value!r} has duplicates")
            if kind == ArmKind.optuna and "judge_policy" in feats:
                raise ValueError("the optuna arm has no judge; judge_policy does not apply")
            if kind == ArmKind.optuna and "backfill" in feats:
                raise ValueError("the optuna arm is the sampler; backfill does not apply")
            if kind == ArmKind.optuna and "lean_planner" in feats:
                raise ValueError("the optuna arm has no planner prompt; lean_planner does not apply")
            if kind in PROMPT_ABLATION_ARMS and "lean_planner" in feats:
                raise ValueError(f"arm {kind.value!r} already has its own propose prompt variant; "
                                 "lean_planner does not apply")
            if kind == ArmKind.optuna and "planner_cadence" in feats:
                raise ValueError("the optuna arm has no AI planner; planner_cadence does not apply")
        for kind in self.arms:
            feats = set(self.features_for(kind))
            for f, needs in FEATURE_REQUIRES.items():
                if f in feats and needs not in feats:
                    raise ValueError(f"arm {kind.value!r}: {f} needs {needs} on the same arm (without the no-AI "
                                     "sampler the slots would starve between planner calls)")
            for f, names in FEATURE_PARAMS.items():
                if f in feats:
                    got = (self.feature_params.get(kind) or {}).get(f)  # type: ignore[call-overload]
                    missing = [n for n in names if got is None or n not in got]
                    if missing:
                        raise ValueError(f"arm {kind.value!r}: {f} needs feature_params {missing}")
        for kind, per in self.feature_params.items():
            if kind not in self.arms:
                raise ValueError(f"feature_params names arm {kind.value!r}, which is not in the race")
            feats = set(self.features_for(kind))
            for f, params in per.items():
                if f not in feats:
                    raise ValueError(f"feature_params for arm {kind.value!r} sets {f!r}, which that arm does not run")
                allowed = FEATURE_PARAMS.get(f, ())
                bad = sorted(set(params) - set(allowed))
                if bad:
                    raise ValueError(f"feature_params[{kind.value!r}][{f!r}]: unknown parameter(s) {bad} "
                                     f"(allowed: {list(allowed)})")
                for n, v in params.items():
                    if not math.isfinite(float(v)) or float(v) <= 0:
                        raise ValueError(f"feature_params[{kind.value!r}][{f!r}][{n!r}] must be a finite number > 0")
        if self.surrogate_runtime_mult is not None and self.runner != "surrogate":
            raise ValueError("surrogate_runtime_mult only applies to runner='surrogate'")
        if not self.median_stop_fractions or any(not (0.0 < f < 1.0) for f in self.median_stop_fractions):
            raise ValueError("median_stop_fractions must be non-empty, each in (0, 1)")
        if sorted(self.median_stop_fractions) != list(self.median_stop_fractions):
            raise ValueError("median_stop_fractions must be increasing")
        return self

    def features_for(self, kind: ArmKind | str) -> tuple[str, ...]:
        """The engine features this arm runs with (sorted, for comparison)."""
        kind = ArmKind(kind)
        feats = self.engine_features.get(kind, DEFAULT_ENGINE_FEATURES.get(kind, ()))
        return tuple(sorted(feats))

    def engine_features_map(self) -> dict[str, list[str]]:
        return {k.value: list(self.features_for(k)) for k in self.arms}

    def params_for(self, kind: ArmKind | str) -> dict[str, dict[str, float]]:
        """This arm's declared feature parameters ({feature: {name: value}}), sorted."""
        per = self.feature_params.get(ArmKind(kind)) or {}
        return {str(f): {n: float(v) for n, v in sorted(p.items())} for f, p in sorted(per.items())}

    def feature_params_map(self) -> dict[str, dict[str, dict[str, float]]]:
        """arm -> declared feature parameters, for every arm (empty when none): race-wide."""
        return {k.value: self.params_for(k) for k in self.arms}

    def min_propose_interval_for(self, kind: ArmKind | str) -> float | None:
        """planner_cadence: seconds between the starts of this arm's propose calls; None = off."""
        if "planner_cadence" not in self.features_for(kind):
            return None
        return float(self.params_for(kind)["planner_cadence"]["min_propose_interval_s"])

    def propose_variant_for(self, kind: ArmKind | str) -> tuple[str, str] | None:
        """(propose prompt version, effort) this LLM arm DECLARES instead of the standard
        (``STANDARD_PROPOSE``), or None for the standard planner. Sources: the prompt-ablation
        arms (``ARM_PROPOSE_VARIANTS``, dev only) and the ``lean_planner`` engine feature (D46).
        The only prompt difference a strict race accepts between LLM arms."""
        kind = ArmKind(kind)
        if kind == ArmKind.optuna:
            return None
        if kind in ARM_PROPOSE_VARIANTS:
            return ARM_PROPOSE_VARIANTS[kind]
        if "lean_planner" in self.features_for(kind):
            return LEAN_PLANNER_PROPOSE
        return None

    def propose_setup_for(self, kind: ArmKind | str) -> dict[str, str]:
        """The planner setup this LLM arm must run: ``{"prompt_version", "effort", "source"}``
        (source: standard | lean_planner | ablation arm)."""
        kind = ArmKind(kind)
        v = self.propose_variant_for(kind)
        if v is None:
            return {"prompt_version": STANDARD_PROPOSE[0], "effort": STANDARD_PROPOSE[1], "source": "standard"}
        src = "ablation arm" if kind in ARM_PROPOSE_VARIANTS else "lean_planner"
        return {"prompt_version": v[0], "effort": v[1], "source": src}

    def propose_setups(self) -> dict[str, dict[str, str]]:
        """arm -> declared planner setup, for every LLM arm (Optuna has no planner prompt)."""
        return {k.value: self.propose_setup_for(k) for k in self.llm_arms}

    def record_mode_for(self, kind: ArmKind | str) -> RecordMode:
        """The x-middleware-record mode this arm's calls send (fallback calls included)."""
        kind = ArmKind(kind)
        if kind in self.record_mode:
            return self.record_mode[kind]
        return "full" if self.record else "off"

    def record_modes(self) -> dict[str, RecordMode]:
        """arm -> mode for every LLM arm (Optuna makes no calls)."""
        return {k.value: self.record_mode_for(k) for k in self.llm_arms}

    def model_for(self, kind: ArmKind | str) -> str:
        """The provider model ID this LLM arm sends (``ARM_FIXED_MODELS`` or ``model``)."""
        return ARM_FIXED_MODELS.get(ArmKind(kind), self.model)

    @property
    def is_dev(self) -> bool:
        """Not a headline race (dev or ablation): stored as is_dev=1, never in headline numbers."""
        return self.mode != "benchmark"

    @property
    def strict(self) -> bool:
        """Benchmark-grade integrity checks (benchmark and ablation races)."""
        return self.mode in ("benchmark", "ablation")

    @property
    def is_ablation(self) -> bool:
        return self.mode == "ablation"

    @property
    def corrections_allowed(self) -> bool:
        return not self.strict or self.benchmark_corrections == "record"

    @property
    def llm_arms(self) -> list[ArmKind]:
        return [a for a in self.arms if a != ArmKind.optuna]


def apply_surrogate_runtime_mult(cfg: RaceConfig) -> float | None:
    """Set ``BUSY_SURROGATE_RUNTIME_MULT`` for the whole process from the race config (so every
    arm's SurrogateRunner, and every run_ref, gets it). Refuses (ValueError) if the environment
    already holds a different value: one process, one pacing. Returns the multiplier applied,
    or None when the config declares none."""
    import os

    if cfg.surrogate_runtime_mult is None:
        return None
    from workers.runners.surrogate import ENV_MULTIPLIER

    want = float(cfg.surrogate_runtime_mult)
    raw = os.environ.get(ENV_MULTIPLIER)
    if raw and not math.isclose(float(raw), want, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{ENV_MULTIPLIER}={raw} is set in the environment, but the race declares x{want:g}")
    os.environ[ENV_MULTIPLIER] = f"{want:g}"
    return want


def estimate_gpu_cost_usd(cfg: RaceConfig, price_per_s: float) -> float:
    """Upper bound on GPU $ for a race: every arm pays every slot for the full budget."""
    return len(cfg.arms) * cfg.gpus_per_arm * cfg.budget_s * price_per_s


# --------------------------------------------------------------------------- presets
# Thresholds (target_value) live on the targets themselves and are frozen before races.
# Budgets follow benchmark.md "Seed and time plan" and budget.md "The $30 split".

BENCH_SEEDS: tuple[int, ...] = (0, 1, 2)

PRESETS: dict[str, RaceConfig] = {
    # Laptop: mock LLM + local CPU runner, $0 (RL is the CPU-native target).
    "dev-local": RaceConfig(
        name="dev local race", target="rl", arms=[ArmKind.opus, ArmKind.mubit, ArmKind.optuna],
        budget_s=120, gpus_per_arm=2, mode="dev", llm="mock", runner="local",
        record_mode=dict(BENCH_RECORD_MODES),
    ),
    # LM speedrun: 4 x L4 per arm, 30 min (cut from 1 h if Sapling needs the H100).
    "lm-30m": RaceConfig(
        name="LM speedrun 30 min", target="lm", arms=[ArmKind.opus, ArmKind.mubit],
        budget_s=1800, gpus_per_arm=4, mode="benchmark", llm="mubit", runner="modal",
        cost_cap_usd=15.0, max_estimated_gpu_usd=4.0, record_mode=dict(BENCH_RECORD_MODES),
    ),
    # CIFAR speedrun: 4 x L4 per arm, 30 min.
    "cifar-30m": RaceConfig(
        name="CIFAR speedrun 30 min", target="cifar", arms=[ArmKind.opus, ArmKind.mubit],
        budget_s=1800, gpus_per_arm=4, mode="benchmark", llm="mubit", runner="modal",
        cost_cap_usd=15.0, max_estimated_gpu_usd=4.0, record_mode=dict(BENCH_RECORD_MODES),
    ),
    # RL on CPU: 32 workers per arm, 30 min, all three arms at once.
    "rl-cpu-30m": RaceConfig(
        name="RL on CPU 30 min", target="rl", arms=[ArmKind.opus, ArmKind.mubit, ArmKind.optuna],
        budget_s=1800, gpus_per_arm=32, mode="benchmark", llm="mubit", runner="modal",
        cost_cap_usd=15.0, max_estimated_gpu_usd=3.0, record_mode=dict(BENCH_RECORD_MODES),
    ),
    # REGIME #2 (D32), the headline race: RL on CPU, 30 slots per arm, 15 s runs, 15 min.
    # Same for every arm: K <= 8 per propose call (the measured range of R27; keeps Opus far
    # from its max_tokens), the deadline rule (a 15 s run's measured p90 wall time, see
    # data/calibration/calibration_rl15_v1.json), and a pre-warmed pool before t0. Goal key
    # "rl@15s" (bench.goal_key). Arms (D43, D44): the standard Opus agent vs Optuna, no early
    # stopping on either. opus_bm left (it would differ from opus only by the judge policy,
    # which cannot move race headline numbers under accept_unsure, D41); mubit joins once an
    # adapter serves. Containers: 2 arms x 30 + 2 spare = 62, plus Sapling's H100 = 63 of 100.
    "rl-wide-15m": RaceConfig(
        name="RL wide 15 min (15 s runs)", target="rl", arms=[ArmKind.opus, ArmKind.optuna],
        budget_s=900, gpus_per_arm=30, run_budget_s=15, max_proposals_per_call=8,
        launch_cutoff_s=17.0, prewarm=True, prewarm_spare=2,
        mode="benchmark", llm="mubit", runner="modal",
        cost_cap_usd=8.0, max_estimated_gpu_usd=1.25,
        record_mode={ArmKind.opus: "off"},
    ),
    # CIFAR speedrun, 15 min, the second headline target (D44: Opus vs Optuna, n=3 per target):
    # 3 x L4 per arm (2 arms x 3 = 6 of Starter's 10 GPUs), the target's default 45 s runs, so
    # goal key "cifar" (0.94, frozen in data/calibration/target_values.json). K <= 8 per call
    # (as rl-wide-15m; with 3 slots K is <= 3 anyway). Pre-warmed pool before t0 (one L4 per slot,
    # no spare; pre-warm is outside the race clock and its $). No deadline rule: no measured
    # p90 wall time for cifar at 3 slots yet, so late runs are cancelled at the deadline (same
    # for both arms). GPU $ <= 2 x 3 x 900 s x $0.000222 = $1.20 per seed; per-arm cap = its
    # $0.60 GPU + ~$4.4 LLM.
    "cifar-15m": RaceConfig(
        name="CIFAR speedrun 15 min", target="cifar", arms=[ArmKind.opus, ArmKind.optuna],
        budget_s=900, gpus_per_arm=3, max_proposals_per_call=8, prewarm=True, prewarm_spare=0,
        mode="benchmark", llm="mubit", runner="modal",
        cost_cap_usd=5.0, max_estimated_gpu_usd=1.5,
        record_mode={ArmKind.opus: "off"},
    ),
    # ABLATION, NEVER A HEADLINE RACE (mode "ablation": stored as is_dev=1, scripts/headline.py
    # refuses it). Tests the planner-speed mechanism behind R43's projection with real data:
    # the same researcher, prompts, retry/fallback policy, slots and budget as rl-wide-15m, only
    # the model differs: "opus" (claude-opus-5-5) vs "fast" (claude-haiku-5-5), both through
    # Mubit project A with record off. "fast" is a smaller faster model in the same loop, NOT a
    # Mubit adapter: it measures what a faster planner does to GPUs waiting, not Mubit's models.
    # Containers: 2 arms x 30 + 2 spare = 62.
    "rl-planner-ablation": RaceConfig(
        name=f"RL planner-speed ablation 15 min (15 s runs; fast = {FAST_ARM_LABEL})", target="rl",
        arms=[ArmKind.opus, ArmKind.fast],
        budget_s=900, gpus_per_arm=30, run_budget_s=15, max_proposals_per_call=8,
        launch_cutoff_s=17.0, prewarm=True, prewarm_spare=2,
        mode="ablation", llm="mubit", runner="modal",
        cost_cap_usd=8.0, max_estimated_gpu_usd=1.25,
        record_mode={ArmKind.opus: "off", ArmKind.fast: "off"},
    ),
    # PROMPT ABLATION, NEVER A HEADLINE RACE (dev mode: is_dev=1; bench.py stores it in
    # data/bench/ablation.db; scripts/headline.py refuses it). Same as lcbench-15m except the arms:
    # "opus" (frozen v1 planner prompt, effort medium) vs "lean" (LEAN_ARM_LABEL: the same
    # claude-opus-5-5 with the draft propose_experiments v2 prompt, diff-form reply, effort low).
    # Both through Mubit project A with record off. Dev mode because the prompts differ on purpose
    # (the strict identical-LLM-policy check would refuse it); everything else is identical.
    "lcbench-lean-15m": RaceConfig(
        name=f"LCBench lean-planner ablation 15 min (lean = {LEAN_ARM_LABEL})", target="lcbench",
        arms=[ArmKind.opus, ArmKind.lean],
        budget_s=900, gpus_per_arm=16, max_proposals_per_call=8, mode="dev", llm="mubit", runner="surrogate",
        cost_cap_usd=7.2, max_estimated_gpu_usd=10.0,
        record_mode={ArmKind.opus: "off", ArmKind.lean: "off"},
    ),
    # Surrogate search race (YAHPO LCBench, task 167168): 16 virtual L4 slots per arm, 15 min.
    # Runs are paced by the surrogate's predicted training time (SurrogateRunner, multiplier 1.0
    # unless BUSY_SURROGATE_RUNTIME_MULT is set) and priced at the L4 rate: GPU $ is VIRTUAL
    # (16 x 900 s x $0.000222 = $3.20 per arm), real spend is the LLM only. The per-arm cap is
    # that virtual GPU $ plus $4 of LLM. K <= 8 per propose call (as rl-wide-15m). No deadline
    # rule: run lengths vary 5-300 s by config, so late runs are cancelled at the deadline instead.
    # Arms (D43): Opus vs Busy Machines with Opus vs Optuna; opus_bm stays here as a run-to-run
    # noise control (judge policy only, no early stopping on any arm). The seed-spread rule has
    # sigma 0 here (the surrogate is deterministic per config), so the judge policy is degenerate.
    # NOTE (backfill gate): strict races run each arm's DEFAULT features, and opus_bm's defaults
    # now include backfill, so a new lcbench-15m race's opus_bm is no longer a noise control
    # (its fairness fingerprint differs from the old races, so headline.py never pools them).
    # lcbench-bm-15m is the same race under its own name.
    "lcbench-15m": RaceConfig(
        name="LCBench surrogate 15 min", target="lcbench", arms=[ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna],
        budget_s=900, gpus_per_arm=16, max_proposals_per_call=8, mode="benchmark", llm="mubit", runner="surrogate",
        cost_cap_usd=7.2, max_estimated_gpu_usd=10.0,
        record_mode={ArmKind.opus: "off", ArmKind.opus_bm: "off"},
    ),
    # ---- Busy Machines with Opus (lean planner + backfill + judge policy, D46) vs the standard
    # Opus agent (plain v1 loop, effort medium) vs Optuna. Same settings as the matching headline
    # preset; only opus_bm is added (record off, Opus on every step, project A). Per-arm caps
    # unchanged; the GPU $ guard covers the third arm. opus_bm sends propose v2 at effort low
    # (lean_planner, declared per arm in the fairness manifest); judge v1 is the same on both.
    # RL wide: 30 slots per arm. Containers at once: 3 arms x 30 + 2 spare = 92 (+ Sapling's
    # H100 if it is up = 93) of Modal Starter's 100. GPU $ <= 3 x 30 x 900 s x $0.0000131 = $1.06.
    "rl-wide-bm-15m": RaceConfig(
        name="RL wide 15 min (15 s runs), Busy Machines with Opus", target="rl",
        arms=[ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna],
        budget_s=900, gpus_per_arm=30, run_budget_s=15, max_proposals_per_call=8,
        launch_cutoff_s=17.0, prewarm=True, prewarm_spare=2,
        mode="benchmark", llm="mubit", runner="modal",
        cost_cap_usd=8.0, max_estimated_gpu_usd=1.25,
        record_mode={ArmKind.opus: "off", ArmKind.opus_bm: "off"},
    ),
    # RL wide + PLANNER CADENCE 60 s on Busy Machines (offline gate scripts/sim_cadence_rl.py,
    # data/backfill/cadence_rl.json: 60 s is the shortest interval that projects BM's LLM $ at
    # <= 0.3x the standard Opus agent's in every main projection; a PROJECTION until raced).
    # Identical to rl-wide-bm-15m (30 CPU slots/arm, 15 s runs, 15 min, goal rl@15s = -170 frozen,
    # arms opus / opus_bm / optuna, same caps, pre-warm, deadline rule, record off) except that
    # opus_bm also runs planner_cadence: a propose call at most every 60 s between call starts,
    # K = min(8, LLM demand), judge calls not throttled (D47's feature, preset-declared, disclosed
    # in the fairness manifest). Its fingerprint differs from R67's (cadence): never pooled.
    "rl-wide-bm-cadence-15m": RaceConfig(
        name="RL wide 15 min (15 s runs), Busy Machines with Opus + planner cadence 60 s", target="rl",
        arms=[ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna],
        budget_s=900, gpus_per_arm=30, run_budget_s=15, max_proposals_per_call=8,
        launch_cutoff_s=17.0, prewarm=True, prewarm_spare=2,
        mode="benchmark", llm="mubit", runner="modal",
        cost_cap_usd=8.0, max_estimated_gpu_usd=1.25,
        record_mode={ArmKind.opus: "off", ArmKind.opus_bm: "off"},
        engine_features={ArmKind.opus_bm: ["backfill", "judge_policy", "lean_planner", "planner_cadence"]},
        feature_params={ArmKind.opus_bm: {"planner_cadence": {"min_propose_interval_s": 60.0}}},
    ),
    # CIFAR: 3 L4 per arm x 3 arms = 9 of Starter's 10 GPUs (no spare). GPU $ <= 3 x 3 x 900 s x
    # $0.000222 = $1.80 per seed (guard $2.00); per-arm cap unchanged ($5).
    "cifar-bm-15m": RaceConfig(
        name="CIFAR speedrun 15 min, Busy Machines with Opus", target="cifar",
        arms=[ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna],
        budget_s=900, gpus_per_arm=3, max_proposals_per_call=8, prewarm=True, prewarm_spare=0,
        mode="benchmark", llm="mubit", runner="modal",
        cost_cap_usd=5.0, max_estimated_gpu_usd=2.0,
        record_mode={ArmKind.opus: "off", ArmKind.opus_bm: "off"},
    ),
    # LCBench surrogate: identical to lcbench-15m (opus, opus_bm, optuna; 16 virtual L4 slots).
    "lcbench-bm-15m": RaceConfig(
        name="LCBench surrogate 15 min, Busy Machines with Opus", target="lcbench",
        arms=[ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna],
        budget_s=900, gpus_per_arm=16, max_proposals_per_call=8, mode="benchmark", llm="mubit", runner="surrogate",
        cost_cap_usd=7.2, max_estimated_gpu_usd=10.0,
        record_mode={ArmKind.opus: "off", ArmKind.opus_bm: "off"},
    ),
    # ---- PRE-REGISTERED 'LCBench wide setup' (D47; docs/preregistration_lcbench_wide.md, frozen
    # before any race, implemented exactly as written; gate R66). The standard Opus agent (plain v1
    # loop, effort medium, no engine features) vs Busy Machines with Opus (lean planner v2 / effort
    # low + backfill (Optuna TPE seeded by the race seed, n_startup 10, constant_liar, LLM proposals
    # first, checker-gated 'worst_on_rejected') + judge v1 with the judge policy + PLANNER CADENCE: a
    # propose call at most every 90 s, K = min(8, LLM demand), judge calls not throttled) vs Optuna
    # (the required no-AI baseline, unchanged). 16 virtual L4 per arm, 15 min, all arms start
    # together; runs hold their slot for 0.1 x the surrogate's predicted training time
    # (BUSY_SURROGATE_RUNTIME_MULT=0.1 for the whole process, shown in every run_ref as ":x0.1").
    # K <= 8 per call, baseline first, no deadline rule (late runs are cancelled at the deadline),
    # pipelining off, accept_unsure on. Per-arm cap $9.20 = virtual GPU 16 x 900 x $0.000222 = $3.20
    # + $6 LLM headroom. GPU $ guard 10.0 >= 3 x 3.20 = 9.6 (virtual). Benchmark mode, LLM through
    # Mubit project A, record off on every LLM arm. Seeds 0, 1, 2. Goal 0.78 (frozen, lcbench).
    # Its fingerprint differs from R38/R44/R65 (x0.1 pacing, planner cadence): never pooled.
    "lcbench-wide-bm-15m": RaceConfig(
        name="LCBench wide 15 min (16 slots, runs x0.1), Busy Machines with Opus + planner cadence 90 s",
        target="lcbench", arms=[ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna],
        budget_s=900, gpus_per_arm=16, max_proposals_per_call=8, run_baseline_first=True,
        launch_cutoff_s=None, pipelining=False, accept_unsure=True,
        mode="benchmark", llm="mubit", runner="surrogate", surrogate_runtime_mult=0.1,
        cost_cap_usd=9.2, max_estimated_gpu_usd=10.0,
        record_mode={ArmKind.opus: "off", ArmKind.opus_bm: "off"},
        engine_features={ArmKind.opus_bm: ["backfill", "judge_policy", "lean_planner", "planner_cadence"]},
        feature_params={ArmKind.opus_bm: {"planner_cadence": {"min_propose_interval_s": 90.0}}},
    ),
    # ---- PRE-REGISTERED follow-up 'LCBench wide, planner every >= 30 s' (D49;
    # docs/preregistration_lcbench_wide_c30.md, frozen ~16:45 BST before any race, chosen AFTER R75
    # failed). Identical to lcbench-wide-bm-15m except ONE parameter: opus_bm's planner_cadence
    # min_propose_interval_s = 30 (was 90). Seeds 0, 1, 2. Its fingerprint differs from R75's
    # (cadence 30 vs 90): never pooled. The last cadence setting tried (D49).
    "lcbench-wide-bm-c30-15m": RaceConfig(
        name="LCBench wide 15 min (16 slots, runs x0.1), Busy Machines with Opus + planner cadence 30 s",
        target="lcbench", arms=[ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna],
        budget_s=900, gpus_per_arm=16, max_proposals_per_call=8, run_baseline_first=True,
        launch_cutoff_s=None, pipelining=False, accept_unsure=True,
        mode="benchmark", llm="mubit", runner="surrogate", surrogate_runtime_mult=0.1,
        cost_cap_usd=9.2, max_estimated_gpu_usd=10.0,
        record_mode={ArmKind.opus: "off", ArmKind.opus_bm: "off"},
        engine_features={ArmKind.opus_bm: ["backfill", "judge_policy", "lean_planner", "planner_cadence"]},
        feature_params={ArmKind.opus_bm: {"planner_cadence": {"min_propose_interval_s": 30.0}}},
    ),
    # Live demo: 2 L4s per arm x 3 min (~$0.20).
    # Corrections are recorded (the demo shows one), so this race never feeds headline numbers.
    "demo-3m": RaceConfig(
        name="live demo", target="cifar", arms=[ArmKind.opus, ArmKind.mubit],
        budget_s=180, gpus_per_arm=2, mode="benchmark", llm="mubit", runner="modal",
        cost_cap_usd=3.0, max_estimated_gpu_usd=0.5, record_mode=dict(BENCH_RECORD_MODES),
        benchmark_corrections="record",
    ),
}


def races_for_preset(name: str, seeds: tuple[int, ...] = BENCH_SEEDS) -> list[RaceConfig]:
    """One RaceConfig per seed (GPU seeds run one after another; see benchmark.md)."""
    base = PRESETS[name]
    return [base.model_copy(update={"seed": s, "name": f"{base.name} seed {s}"}) for s in seeds]
