"""Overnight Opus example collection (plan.md step 7, budget.md "Example collection").

Runs real Opus-arm races through the existing race engine (``backend.scheduler``) and LLM
researcher (``backend.agent.loop``) so that Mubit records examples for the three step tasks
(propose_experiments, diagnose_run, judge_improvement). Goal: 75-100 kept examples per step,
covering the full mix: healthy runs, failures (-> diagnose_run), overfit runs and
contamination traps (-> judge_improvement).

    # free: mock LLM + local CPU runner, exercises the whole path
    python -m uv run --extra ml python scripts/collect_examples.py --dry-run --target rl --minutes 2 --seeds 0

    # real (paid): Opus through Mubit, record on; refuses unless the prompts are FROZEN
    python -m uv run python scripts/collect_examples.py --target cifar --runner modal --slots 4 \
        --minutes 480 --usd-cap 20 --seeds 0,1,2 --yes

Rules this script enforces:
- ``record=True`` (the default for a real run) is refused unless every step's prompt is
  frozen (``backend.agent.prompts.FROZEN[step] == CURRENT_VERSION[step]``). A prompt edit
  after collection starts splits the Mubit task and throws the examples away.
- Every LLM call goes through ``MubitClient`` (the Mubit gateway, ``MUBIT_API_KEY``); the
  dry run uses ``MockLLM`` and never touches the network. Never ``ANTHROPIC_API_KEY``.
- A failure storm (``--storm-n`` consecutive runs failing with the same error, e.g. a broken
  worker image) stops the collection instead of buying one diagnose call per crash.
- A paid run needs ``--yes`` after the plan (estimated GPU $, cap) is printed.
- The $ cap covers LLM + GPU across all races; each race gets the remaining cap as its
  ``cost_cap_usd`` and a wall-clock budget that fits the remaining GPU $.
- Collection races are ``mode="dev"`` (``races.is_dev = 1``), so ``assert_benchmark_clean`` /
  ``check_benchmark_rows`` reject them: they are never benchmark results. They live in their
  own DB by default (``data/collection/collect.db``).

Coverage: diagnose_run happens only on failures and judge_improvement only on a claimed new
best, so they are naturally rare. When either lags (fewer valid calls than
``--coverage-ratio`` x propose calls, and below ``--goal``), the researcher wrapper swaps
one slot of a proposal batch for a deliberate **collection probe**, at most ``--probe-rate``
of all launched experiments:

- ``diverge``  (for diagnose_run): aggressive lr / huge batch configs known to fail;
- ``trap``     (for judge_improvement): the current best config with the target's
  contamination-trap option enabled (inflated reported score -> a claimed new best);
- ``overfit``  (for judge_improvement): high-capacity, unregularised config on a reduced
  per-run budget (``--overfit-budget-frac`` of the run budget, at least 10 s).

Probe hypotheses are neutral text (they never say "probe" or "trap"), so the examples the
LLM sees carry no label hints. Ground truth is stored only in the side table
``collection_probes`` (and ``collection_races``) of the collection DB.

Progress (per-step calls, valid rate, recorded, est. $, run mix) is printed and appended
to ``data/collection/collect-{date}-{target}-{HHMMSS}.jsonl`` every ``--log-every-min``.
Mubit session ids are ``collect-{date}/{target}/seed-{n}`` (``metadata.user_id``).

"Kept" examples are decided by Mubit's review; this script can only count valid (and
recorded) calls, which is an upper bound. Check the Mubit console for the kept count.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import logging
import math
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agent import prompts as prompts_pkg  # noqa: E402
from backend.agent.llm_base import OPUS_MODEL  # noqa: E402
from backend.agent.loop import LLMResearcher, load_prompts  # noqa: E402
from backend.benchmark.configs import RaceConfig  # noqa: E402
from backend.core.types import (  # noqa: E402
    ArmKind,
    DiagnoseOutput,
    ExperimentStatus,
    JudgeOutput,
    LLMCallRecord,
    StepType,
    new_id,
    now,
)
from backend.db import Database  # noqa: E402
from backend.scheduler import (  # noqa: E402
    ArmDeps,
    EventBus,
    ExperimentSnapshot,
    JudgeClaim,
    Proposal,
    RaceEngine,
    ResearchDigest,
)
from workers.targets.base import TargetSpec, get_target, list_targets  # noqa: E402

log = logging.getLogger("collect_examples")

COLLECTION_DIR = REPO_ROOT / "data" / "collection"
DEFAULT_DB = COLLECTION_DIR / "collect.db"
DEFAULT_GOAL = 90                 # valid calls per step (plan.md: aim for 75-100 kept)
DRY_RUN_RUN_BUDGET_S = 15         # per-run training budget in the dry run (RL JIT takes ~7 s on a laptop)
MIN_OVERFIT_BUDGET_S = 10         # overfit probes never get less (or the run's budget, if smaller)
# The three race steps this script collects. early_stop (D34) has its own collector
# (scripts/collect_early_stop.py, from completed curves) and its own D39 gate.
STEPS: tuple[StepType, ...] = (StepType.propose_experiments, StepType.diagnose_run, StepType.judge_improvement)

PROBE_KINDS: tuple[str, ...] = ("diverge", "trap", "overfit")
PROBE_STEP: dict[str, StepType] = {
    "diverge": StepType.diagnose_run,
    "trap": StepType.judge_improvement,
    "overfit": StepType.judge_improvement,
}
# Judge probes: two trap probes for every overfit probe (traps reliably claim a new best).
JUDGE_KIND_WEIGHT: dict[str, float] = {"trap": 2.0, "overfit": 1.0}
# Back-off: a probe kind that triggered its step in < 1/3 of its last (>= 3) finished probes is
# switched off for the rest of the collection (e.g. PPO with Adam + grad clipping rarely diverges).
BACKOFF_MIN_TRIES = 3
BACKOFF_WINDOW = 6
BACKOFF_MIN_HIT_RATE = 1 / 3
# Failure-storm breaker: this many consecutive failed runs with the same error (digits
# ignored) stop the whole collection. A broken image / import error would otherwise make
# every run crash instantly and buy one Opus diagnose_run call (and one junk example) each.
DEFAULT_STORM_N = 8

# Per-target probe recipes: overrides applied on top of the current best (or default)
# config, then validated against the spec. Unknown targets use ``_generic_overrides``.
PROBE_RECIPES: dict[str, dict[str, dict[str, Any]]] = {
    "rl": {
        # Max lr, giant rollout x 10 epochs x 32 minibatches: slow and unstable on one core.
        "diverge": {"lr": 1e-2, "anneal_lr": False, "clip_eps": 0.5, "update_epochs": 10,
                    "num_minibatches": 32, "num_envs": 128, "num_steps": 512, "hidden_size": 256,
                    "reward_scale": 1.0, "ent_coef": 0.0},
        # Few envs, many epochs over each batch, big net, no entropy bonus: fits the training levels.
        "overfit": {"num_envs": 4, "num_steps": 512, "update_epochs": 10, "num_minibatches": 32,
                    "hidden_size": 256, "ent_coef": 0.0},
    },
    "cifar": {
        "diverge": {"lr": 40.0, "momentum": 0.95, "warmup_frac": 0.0, "label_smoothing": 0.0,
                    "batch_size": 256, "weight_decay": 0.001},
        "overfit": {"width": 2.0, "epochs": 30.0, "label_smoothing": 0.0, "flip": False, "translate": 0,
                    "weight_decay": 0.001, "tta": 0},
    },
    "lm": {
        "diverge": {"lr": 1e-2, "warmup_frac": 0.0, "weight_decay": 0.0, "n_layer": 8, "n_embd": 512,
                    "n_head": 8, "block_size": 512, "batch_size": 128},
        "overfit": {"n_layer": 8, "n_embd": 512, "n_head": 8, "dropout": 0.0, "weight_decay": 0.0},
    },
}

# Neutral hypotheses: what a researcher might plausibly write. Never mention probe/trap.
PROBE_HYPOTHESES: dict[str, str] = {
    "diverge": "Pushing the learning rate and batch to their limits to see how fast training can go.",
    "trap": "Testing whether turning on {opt} improves the reported score.",
    "overfit": "Testing a larger model with less regularisation for more capacity.",
}


class CollectionRefused(RuntimeError):
    """A pre-flight check failed; nothing was started."""


# =========================================================================== pre-flight


def prompt_freeze_status() -> dict[str, dict[str, Any]]:
    """Per step: the version in use, the frozen version and whether they match."""
    out: dict[str, dict[str, Any]] = {}
    for step in STEPS:
        current = prompts_pkg.CURRENT_VERSION[step]
        frozen = prompts_pkg.FROZEN.get(step)
        try:
            fp = prompts_pkg.prompt_fingerprint(step)
        except OSError:
            fp = None
        out[step.value] = {"current": current, "frozen": frozen, "fingerprint": fp,
                           "ok": frozen is not None and frozen == current}
    return out


def record_problems() -> list[str]:
    """Why ``record=True`` is not allowed right now (empty list = allowed)."""
    problems = []
    for step, st in prompt_freeze_status().items():
        if st["frozen"] is None:
            problems.append(f"{step}: prompt {st['current']} is a DRAFT (FROZEN[{step}] is None)")
        elif st["frozen"] != st["current"]:
            problems.append(f"{step}: FROZEN={st['frozen']} but CURRENT_VERSION={st['current']}")
    return problems


def session_id(date: str, target: str, seed: int) -> str:
    """Mubit session per collection researcher run: ``collect-{date}/{target}/seed-{n}``."""
    return f"collect-{date}/{target}/seed-{seed}"


def utc_date() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d")


# =========================================================================== call tracking


@dataclass
class StepStats:
    calls: int = 0
    valid: int = 0
    retries: int = 0
    fallbacks: int = 0
    recorded: int = 0
    recorded_valid: int = 0
    cost_usd: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)

    def add(self, rec: LLMCallRecord) -> None:
        self.calls += 1
        self.valid += int(rec.valid)
        self.retries += int(rec.is_retry)
        self.fallbacks += int(rec.is_fallback)
        self.recorded += int(rec.recorded)
        self.recorded_valid += int(rec.recorded and rec.valid)
        self.cost_usd += float(rec.cost_usd or 0.0)
        self.latencies_ms.append(float(rec.latency_ms or 0.0))

    def view(self) -> dict[str, Any]:
        lat = sorted(self.latencies_ms)
        return {"calls": self.calls, "valid": self.valid,
                "valid_rate": round(self.valid / self.calls, 4) if self.calls else None,
                "retries": self.retries, "fallbacks": self.fallbacks, "recorded": self.recorded,
                "recorded_valid": self.recorded_valid, "cost_usd": round(self.cost_usd, 6),
                "p50_latency_ms": round(lat[len(lat) // 2], 1) if lat else None}


class CallTracker:
    """Every LLMCallRecord of the collection (all races), per step.

    ``examples(step)`` is the count that can become a Mubit example: valid replies, and in
    record mode only the recorded ones (an upper bound on Mubit's kept examples)."""

    def __init__(self, record: bool) -> None:
        self.record = record
        self.steps: dict[StepType, StepStats] = {s: StepStats() for s in STEPS}
        self.models: Counter[str] = Counter()
        self.sessions: Counter[str] = Counter()

    def add(self, rec: LLMCallRecord) -> None:
        self.steps[StepType(rec.step_type)].add(rec)
        self.models[rec.model] += 1
        self.sessions[rec.session_id] += 1

    def examples(self, step: StepType) -> int:
        s = self.steps[step]
        return s.recorded_valid if self.record else s.valid

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps.values())

    def view(self) -> dict[str, Any]:
        return {s.value: {**st.view(), "examples": self.examples(s)} for s, st in self.steps.items()}


# =========================================================================== probes


def canonical(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, default=str)


def _generic_overrides(spec: TargetSpec, kind: str) -> dict[str, Any]:
    """Fallback recipe for targets without one: max learning rate, no warmup (diverge)."""
    if kind != "diverge":
        return {}
    out: dict[str, Any] = {}
    for name, p in spec.config_space.items():
        if name == spec.trap_option or p.type not in ("float", "int"):
            continue
        low = name.lower()
        if low in ("lr", "learning_rate") or low.endswith("_lr"):
            out[name] = p.high if p.type == "float" else int(p.high or 0)
        elif "warmup" in low:
            out[name] = p.low if p.type == "float" else int(p.low or 0)
    return out


def probe_overrides(spec: TargetSpec, kind: str) -> dict[str, Any]:
    if kind == "trap":
        return {spec.trap_option: spec.trap_value}
    recipe = PROBE_RECIPES.get(spec.name, {}).get(kind)
    return dict(recipe) if recipe is not None else _generic_overrides(spec, kind)


def probe_config(spec: TargetSpec, kind: str, base: dict[str, Any], rng: random.Random,
                 seen: set[str], attempts: int = 12) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """(config, overrides) for a probe of ``kind`` built on ``base``, not in ``seen``; None
    if the target has no recipe for that kind. Diverge/overfit probes keep the trap off."""
    overrides = probe_overrides(spec, kind)
    if not overrides:
        return None
    cfg = dict(base)
    if kind != "trap":
        cfg[spec.trap_option] = spec.config_space[spec.trap_option].default
    cfg.update(overrides)
    try:
        cfg = spec.validate_config(cfg)
    except ValueError as e:
        log.warning("probe %s for %s is invalid: %s", kind, spec.name, e)
        return None
    free = [n for n, p in spec.config_space.items() if n not in overrides and n != spec.trap_option]
    for _ in range(attempts):
        if canonical(cfg) not in seen:
            return cfg, overrides
        if not free:
            break
        name = rng.choice(free)  # perturb one free parameter to avoid an exact repeat
        cfg = spec.validate_config({**cfg, name: spec.random_config(rng)[name]})
    return (cfg, overrides) if canonical(cfg) not in seen else None


@dataclass
class ProbeInfo:
    experiment_id: str
    kind: str
    config: dict[str, Any]
    overrides: dict[str, Any]
    budget_s: int | None
    race_id: str
    processed: bool = False
    status: str | None = None
    score: float | None = None
    diagnosed: bool = False
    verdict: str | None = None


class CoveragePolicy:
    """Decides whether the next proposal batch gets a probe, and which kind.

    A step "lags" when its valid calls (plus probes in flight that should trigger it) are
    below ``goal`` and below ``ratio`` x valid propose calls. At most one probe per batch and
    at most ``rate`` x launched experiments in total. A kind that keeps missing its step is
    switched off (``record_outcome``)."""

    def __init__(self, *, goal: int, ratio: float, rate: float, kinds: tuple[str, ...]) -> None:
        self.goal = goal
        self.ratio = ratio
        self.rate = rate
        self.kinds = tuple(k for k in PROBE_KINDS if k in kinds)
        self.kind_counts: Counter[str] = Counter()
        self.outcomes: dict[str, list[bool]] = {k: [] for k in PROBE_KINDS}
        self.disabled: set[str] = set()

    def record_outcome(self, kind: str, triggered: bool) -> bool:
        """Record whether a finished probe reached its step. True if this switched the kind off."""
        hist = self.outcomes.setdefault(kind, [])
        hist.append(bool(triggered))
        recent = hist[-BACKOFF_WINDOW:]
        if kind not in self.disabled and len(recent) >= BACKOFF_MIN_TRIES                 and sum(recent) / len(recent) < BACKOFF_MIN_HIT_RATE:
            self.disabled.add(kind)
            return True
        return False

    def lagging(self, examples: dict[StepType, int], inflight: dict[StepType, int]) -> list[StepType]:
        prop = examples.get(StepType.propose_experiments, 0)
        out: list[tuple[float, StepType]] = []
        for step in (StepType.diagnose_run, StepType.judge_improvement):
            have = examples.get(step, 0) + inflight.get(step, 0)
            if have >= self.goal:
                continue
            if have < self.ratio * prop:
                out.append((have / max(1, prop), step))
        return [s for _, s in sorted(out, key=lambda x: (x[0], x[1].value))]

    def choose(self, *, examples: dict[StepType, int], inflight: dict[StepType, int], launched: int,
               probes: int, k: int) -> list[str]:
        """Probe kinds to try, best first (empty = no probe in this batch)."""
        if k < 1 or self.rate <= 0 or probes + 1 > self.rate * (launched + k):
            return []
        out: list[str] = []
        for step in self.lagging(examples, inflight):
            kinds = [kd for kd in self.kinds if PROBE_STEP[kd] == step and kd not in self.disabled]
            w = JUDGE_KIND_WEIGHT
            kinds.sort(key=lambda kd: (self.kind_counts[kd] / w.get(kd, 1.0), -w.get(kd, 1.0), kd))
            out += kinds
        return out


class BudgetOverrideRunner:
    """Runner wrapper: launches a config registered with ``override(config, budget_s)`` on
    that per-run budget (overfit probes); everything else passes through unchanged."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name = inner.name
        self._overrides: dict[str, list[int]] = {}
        self.launched_budgets: list[int] = []

    def override(self, config: dict[str, Any], budget_s: int) -> None:
        self._overrides.setdefault(canonical(config), []).append(int(budget_s))

    def slots(self) -> list[Any]:
        return self.inner.slots()

    def price_per_second(self, hardware: str) -> float:
        return self.inner.price_per_second(hardware)

    async def launch(self, target: str, config: dict[str, Any], seed: int, budget_s: int, on_metric: Any) -> Any:
        queue = self._overrides.get(canonical(config))
        if queue:
            budget_s = queue.pop(0)
        self.launched_budgets.append(int(budget_s))
        return await self.inner.launch(target, config, seed, budget_s, on_metric)

    async def wait(self, handle: Any) -> Any:
        return await self.inner.wait(handle)

    async def cancel(self, handle: Any) -> None:
        await self.inner.cancel(handle)


# =========================================================================== DB side tables


_SIDE_DDL = (
    """CREATE TABLE IF NOT EXISTS collection_races (
        race_id TEXT PRIMARY KEY,
        run_id TEXT,
        target TEXT,
        seed INTEGER,
        session_id TEXT,
        model TEXT,
        llm_mode TEXT,
        runner TEXT,
        record INTEGER,
        prompt_version TEXT,
        started_at REAL,
        ended_at REAL,
        stats_json TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS collection_probes (
        experiment_id TEXT PRIMARY KEY,
        race_id TEXT,
        arm_id TEXT,
        run_id TEXT,
        target TEXT,
        kind TEXT NOT NULL,                -- diverge | trap | overfit
        for_step TEXT,                     -- the step it is meant to trigger
        config_json TEXT,
        overrides_json TEXT,
        budget_s INTEGER,                  -- per-run budget it ran with (NULL = race default)
        created_at REAL,
        status TEXT,
        score REAL,
        is_trap INTEGER,
        diagnosed INTEGER NOT NULL DEFAULT 0,
        verdict TEXT,
        verdict_reason TEXT
    )""",
)


class ProbeStore:
    """Collection-only tables next to the scheduler's (thin adapter: ``backend.db`` has no
    probe flag on experiments and no public execute, so this uses its locked ``_exec`` /
    ``_all``). Experiments listed here are collection probes, never benchmark results."""

    def __init__(self, db: Database) -> None:
        self.db = db
        for stmt in _SIDE_DDL:
            db._exec(stmt)  # noqa: SLF001

    def add_race(self, **row: Any) -> None:
        row["stats_json"] = json.dumps(row.pop("stats", None), default=str)
        cols = ", ".join(row)
        self.db._exec(f"INSERT OR REPLACE INTO collection_races ({cols}) VALUES ({', '.join('?' for _ in row)})",  # noqa: SLF001
                      tuple(row.values()))

    def finish_race(self, race_id: str, ended_at: float, stats: dict[str, Any]) -> None:
        self.db._exec("UPDATE collection_races SET ended_at = ?, stats_json = ? WHERE race_id = ?",  # noqa: SLF001
                      (ended_at, json.dumps(stats, default=str), race_id))

    def add_probe(self, p: ProbeInfo, *, arm_id: str | None, run_id: str, target: str) -> None:
        self.db._exec(  # noqa: SLF001
            "INSERT OR REPLACE INTO collection_probes (experiment_id, race_id, arm_id, run_id, target, kind, "
            "for_step, config_json, overrides_json, budget_s, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (p.experiment_id, p.race_id, arm_id, run_id, target, p.kind, PROBE_STEP[p.kind].value,
             canonical(p.config), canonical(p.overrides), p.budget_s, now()))

    def update_probe(self, experiment_id: str, **fields: Any) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.db._exec(f"UPDATE collection_probes SET {sets} WHERE experiment_id = ?",  # noqa: SLF001
                      (*fields.values(), experiment_id))

    def probes(self, race_id: str | None = None) -> list[dict[str, Any]]:
        if race_id is None:
            return self.db._all("SELECT * FROM collection_probes ORDER BY created_at")  # noqa: SLF001
        return self.db._all("SELECT * FROM collection_probes WHERE race_id = ? ORDER BY created_at",  # noqa: SLF001
                            (race_id,))

    def races(self) -> list[dict[str, Any]]:
        return self.db._all("SELECT * FROM collection_races ORDER BY started_at")  # noqa: SLF001


# =========================================================================== researcher wrapper


class CollectionResearcher:
    """Wraps ``LLMResearcher`` (Researcher protocol): tees every LLMCallRecord into the
    tracker and swaps one slot of a proposal batch for a probe when the policy asks."""

    def __init__(self, inner: LLMResearcher, *, spec: TargetSpec, tracker: CallTracker, policy: CoveragePolicy,
                 store: ProbeStore | None, runner: BudgetOverrideRunner, race_id: str, run_id: str, seed: int,
                 run_budget_s: int, overfit_budget_s: int | None, shared: dict[str, Any]) -> None:
        self.inner = inner
        self.kind = inner.kind
        self.spec = spec
        self.tracker = tracker
        self.policy = policy
        self.store = store
        self.runner = runner
        self.race_id = race_id
        self.run_id = run_id
        self.arm_id: str | None = None
        self.run_budget_s = run_budget_s
        self.overfit_budget_s = overfit_budget_s
        self.rng = random.Random(f"collect-{seed}-{race_id}")
        self.shared = shared                       # cross-race counters: launched, probes
        self.probes: dict[str, ProbeInfo] = {}
        self.stop_cb: Callable[[str], None] | None = None   # set by the collector: stops the race
        self.storm_n = DEFAULT_STORM_N
        self._fail_streak: tuple[str, int] = ("", 0)
        self._sink: Callable[[LLMCallRecord], None] = lambda _r: None
        self.on_call = self._sink

    # -- Researcher protocol ----------------------------------------------------

    @property
    def on_call(self) -> Callable[[LLMCallRecord], None]:
        return self._sink

    @on_call.setter
    def on_call(self, fn: Callable[[LLMCallRecord], None]) -> None:
        self._sink = fn

        def tee(rec: LLMCallRecord) -> None:
            self.tracker.add(rec)
            fn(rec)

        self.inner.on_call = tee

    @property
    def llm(self) -> Any:
        return self.inner.llm

    @property
    def fallback_llm(self) -> Any:
        return self.inner.fallback_llm

    def _inflight(self) -> dict[StepType, int]:
        out: Counter[StepType] = Counter()
        for p in self.shared["open_probes"].values():
            out[PROBE_STEP[p.kind]] += 1
        return dict(out)

    async def propose(self, digest: ResearchDigest, k: int) -> list[Proposal]:
        probe = self._maybe_probe(digest, k)
        n_llm = k - (1 if probe else 0)
        out = await self.inner.propose(digest, n_llm) if n_llm > 0 else []
        if probe is not None:
            out = list(out[:n_llm]) + [probe]
        self.shared["launched"] += len(out)
        return out

    def _maybe_probe(self, digest: ResearchDigest, k: int) -> Proposal | None:
        examples = {s: self.tracker.examples(s) for s in STEPS}
        kinds = self.policy.choose(examples=examples, inflight=self._inflight(), launched=self.shared["launched"],
                                   probes=self.shared["probes"], k=k)
        if not kinds:
            return None
        base = dict(digest.best.config) if digest.best else self.spec.default_config()
        seen = {canonical(e.config) for e in digest.history} | {canonical(c) for c in digest.running}
        for kind in kinds:
            built = probe_config(self.spec, kind, base, self.rng, seen)
            if built is None:
                continue
            cfg, overrides = built
            budget = self.overfit_budget_s if kind == "overfit" else None
            hyp = PROBE_HYPOTHESES[kind].format(opt=self.spec.trap_option)
            prop = Proposal(config=cfg, hypothesis=hyp)
            info = ProbeInfo(experiment_id=prop.experiment_id, kind=kind, config=cfg, overrides=overrides,
                             budget_s=budget, race_id=self.race_id)
            if budget is not None and budget != self.run_budget_s:
                self.runner.override(cfg, budget)
            self.probes[info.experiment_id] = info
            self.shared["open_probes"][info.experiment_id] = info
            self.shared["probes"] += 1
            self.shared["probe_kinds"][kind] += 1
            self.policy.kind_counts[kind] += 1
            if self.store is not None:
                self.store.add_probe(info, arm_id=self.arm_id, run_id=self.run_id, target=self.spec.name)
            log.info("probe %s scheduled (%s)", kind, overrides)
            return prop
        return None

    async def diagnose(self, exp: ExperimentSnapshot) -> DiagnoseOutput | None:
        out = await self.inner.diagnose(exp)
        p = self.probes.get(exp.id)
        if p is not None:
            p.diagnosed = True
            if self.store is not None:
                self.store.update_probe(exp.id, diagnosed=1)
        return out

    async def judge(self, claim: JudgeClaim) -> JudgeOutput | None:
        out = await self.inner.judge(claim)
        p = self.probes.get(claim.candidate.id)
        if p is not None and out is not None:
            p.verdict = out.verdict.value
            if self.store is not None:
                self.store.update_probe(claim.candidate.id, verdict=out.verdict.value, verdict_reason=out.reason)
        return out

    def _check_storm(self, exp: ExperimentSnapshot) -> None:
        if exp.status not in (ExperimentStatus.failed, ExperimentStatus.timed_out):
            self._fail_streak = ("", 0)
            return
        key = re.sub(r"[0-9.]+", "#", (exp.error or exp.status.value).strip())[:200]
        last, n = self._fail_streak
        self._fail_streak = (key, n + 1 if key == last else 1)
        if self._fail_streak[1] >= self.storm_n and self.shared.get("storm") is None:
            msg = (f"failure storm: {self._fail_streak[1]} consecutive runs failed with the same error "
                   f"({key[:120]!r}); stopping the collection - check the runner / worker image")
            log.error(msg)
            self.shared["storm"] = msg
            self.shared["notes"].append(msg)
            if self.stop_cb is not None:
                self.stop_cb("failure_storm")

    def observe(self, exp: ExperimentSnapshot) -> None:
        self.inner.observe(exp)
        self._check_storm(exp)
        p = self.probes.get(exp.id)
        if p is not None:
            p.processed, p.status, p.score = True, exp.status.value, exp.score
            self.shared["open_probes"].pop(exp.id, None)
            if self.store is not None:
                self.store.update_probe(exp.id, status=exp.status.value, score=exp.score,
                                        is_trap=int(self.spec.is_trap(exp.config)))
            if PROBE_STEP[p.kind] is StepType.diagnose_run:
                triggered = exp.status in (ExperimentStatus.failed, ExperimentStatus.timed_out)
            else:
                triggered = bool((exp.summary or {}).get("claimed_new_best"))
            if self.policy.record_outcome(p.kind, triggered):
                msg = (f"{p.kind} probes on {self.spec.name} rarely trigger {PROBE_STEP[p.kind].value} "
                       f"({self.policy.outcomes[p.kind][-BACKOFF_WINDOW:]}); switched off for this collection")
                log.warning(msg)
                self.shared["notes"].append(msg)

    def close_open_probes(self) -> None:
        """Race over: probes that never reached the researcher stop counting as in flight."""
        for pid in list(self.probes):
            self.shared["open_probes"].pop(pid, None)


# =========================================================================== orchestrator


@dataclass
class CollectArgs:
    target: str = "rl"
    minutes: float = 60.0
    race_minutes: float | None = None
    slots: int = 4
    usd_cap: float = 20.0
    runner: str = "modal"
    seeds: tuple[int, ...] = (0, 1, 2)
    model: str = OPUS_MODEL
    dry_run: bool = False
    record: bool = True
    run_budget_s: int | None = None
    log_every_min: float = 5.0
    goal: int = DEFAULT_GOAL
    coverage_ratio: float = 0.5
    probe_rate: float = 0.25
    probe_kinds: tuple[str, ...] = PROBE_KINDS
    overfit_budget_frac: float = 0.5
    pipelining: bool = False
    db_path: str = str(DEFAULT_DB)
    out_dir: str = str(COLLECTION_DIR)
    stop_at_goal: bool = True
    mock_trap_rate: float = 0.1
    storm_n: int = DEFAULT_STORM_N


LLMFactory = Callable[[CollectArgs, TargetSpec, int], Any]
RunnerFactory = Callable[[CollectArgs, TargetSpec], Any]


def default_llm_factory(args: CollectArgs, spec: TargetSpec, seed: int) -> Any:
    if args.dry_run:
        from backend.agent.mock_llm import MockLLM

        return MockLLM(spec, seed=seed, trap_rate=args.mock_trap_rate)
    from backend.agent.mubit_client import MubitClient  # the ONLY LLM client (Mubit gateway)

    return MubitClient(arm_models={ArmKind.opus: args.model}, fallback_model=OPUS_MODEL)


def default_runner_factory(args: CollectArgs, spec: TargetSpec) -> Any:
    if args.runner == "local":
        from workers.runners.local import LocalRunner

        env = {"BUSY_CIFAR_SYNTHETIC": "1"} if args.dry_run and spec.name == "cifar" else None
        return LocalRunner(max_slots=args.slots, hardware="CPU", price_per_second_usd=0.0, env=env)
    from workers.runners.modal_runner import ModalRunner  # the only module that imports modal

    return ModalRunner(slots={spec.hardware: args.slots})


class Collector:
    def __init__(self, args: CollectArgs, *, spec: TargetSpec | None = None, llm_factory: LLMFactory | None = None,
                 runner_factory: RunnerFactory | None = None, db: Database | None = None,
                 stdout: TextIO | None = None, tick_s: float = 0.05) -> None:
        self.args = args
        self.spec = spec or get_target(args.target).spec
        self.llm_factory = llm_factory or default_llm_factory
        self.runner_factory = runner_factory or default_runner_factory
        self.record = bool(args.record and not args.dry_run)
        self.db = db or Database(args.db_path)
        self.store = ProbeStore(self.db)
        self.tracker = CallTracker(self.record)
        self.policy = CoveragePolicy(goal=args.goal, ratio=args.coverage_ratio, rate=args.probe_rate,
                                     kinds=args.probe_kinds)
        self.stdout = stdout or sys.stdout
        self.tick_s = tick_s
        self.date = utc_date()
        self.run_id = f"{self.date}-{args.target}-{_dt.datetime.now(_dt.UTC).strftime('%H%M%S')}"
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = out_dir / f"collect-{self.run_id}.jsonl"
        self.export_dir = out_dir / "races"
        self.run_budget_s = int(args.run_budget_s or (min(self.spec.budget_s, DRY_RUN_RUN_BUDGET_S)
                                                       if args.dry_run else self.spec.budget_s))
        self.overfit_budget_s = max(min(self.run_budget_s, MIN_OVERFIT_BUDGET_S),
                                    int(round(self.run_budget_s * args.overfit_budget_frac)))
        self.shared: dict[str, Any] = {"launched": 0, "probes": 0, "probe_kinds": Counter(), "open_probes": {},
                                       "notes": [], "storm": None}
        self.gpu_done = 0.0
        self.mix_done: Counter[str] = Counter()
        self.races: list[dict[str, Any]] = []
        self.t_start = 0.0
        self._clients: list[Any] = []

    # ------------------------------------------------------------------ logging

    def _emit(self, event: str, **data: Any) -> dict[str, Any]:
        row = {"ts": time.time(), "event": event, "run_id": self.run_id, "target": self.spec.name, **data}
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return row

    def _print(self, line: str) -> None:
        print(line, file=self.stdout, flush=True)

    @staticmethod
    def _mix(engine: RaceEngine | None, probe_ids: set[str]) -> Counter[str]:
        c: Counter[str] = Counter()
        arm = engine.arms.get(ArmKind.opus) if engine is not None else None
        if arm is None:
            return c
        for e in arm.exps.values():
            st = e.status
            if st in (ExperimentStatus.pending, ExperimentStatus.running, ExperimentStatus.cancelled):
                continue
            c["finished"] += 1
            c["probe_runs" if e.id in probe_ids else "agent_runs"] += 1
            if st == ExperimentStatus.succeeded:
                c["trap_runs" if e.is_trap else "healthy_runs"] += 1
            else:
                c["failed_runs"] += 1
            if e.diagnosis is not None:
                c["diagnosed"] += 1
            if e.verdict is not None:
                c[f"verdict_{e.verdict.value}"] += 1
                if e.is_trap:
                    c[f"trap_verdict_{e.verdict.value}"] += 1
        return c

    def snapshot(self, engine: RaceEngine | None = None, probe_ids: set[str] | None = None) -> dict[str, Any]:
        live_gpu = 0.0
        if engine is not None and engine.arms:
            live_gpu = sum(s.gpu_cost_usd for s, _ in engine.live_summaries())
        mix = self.mix_done + self._mix(engine, probe_ids or set())
        llm = self.tracker.cost_usd
        gpu = self.gpu_done + live_gpu
        return {
            "elapsed_s": round(time.time() - self.t_start, 1) if self.t_start else 0.0,
            "steps": self.tracker.view(),
            "goal_per_step": self.args.goal,
            "probes": {"total": self.shared["probes"], **dict(self.shared["probe_kinds"]),
                       "disabled": sorted(self.policy.disabled)},
            "notes": list(self.shared["notes"]),
            "launched": self.shared["launched"],
            "mix": dict(mix),
            "est_usd": {"llm": round(llm, 4), "gpu": round(gpu, 4), "total": round(llm + gpu, 4),
                        "cap": self.args.usd_cap},
            "record": self.record,
            "races": len(self.races),
        }

    def _progress_line(self, snap: dict[str, Any]) -> str:
        parts = []
        for step in STEPS:
            s = snap["steps"][step.value]
            vr = f"{s['valid_rate'] * 100:.0f}%" if s["valid_rate"] is not None else "-"
            parts.append(f"{step.value.split('_')[0]} {s['examples']}/{self.args.goal} ({s['calls']} calls, {vr} valid)")
        p, m, u = snap["probes"], snap["mix"], snap["est_usd"]
        kinds = ", ".join(f"{k} {p.get(k, 0)}" + (" off" if k in p["disabled"] else "")
                          for k in PROBE_KINDS if p.get(k))
        mins = snap["elapsed_s"] / 60
        return (f"[collect {mins:6.1f} min] " + " | ".join(parts)
                + f" | probes {p['total']}" + (f" ({kinds})" if kinds else "")
                + f" | runs {m.get('healthy_runs', 0)} ok / {m.get('failed_runs', 0)} failed / {m.get('trap_runs', 0)} trap"
                + f" | est $ LLM {u['llm']:.2f} + GPU {u['gpu']:.2f} = {u['total']:.2f} / cap {u['cap']:.2f}")

    def report(self, engine: RaceEngine | None = None, probe_ids: set[str] | None = None,
               event: str = "progress", **extra: Any) -> dict[str, Any]:
        snap = self.snapshot(engine, probe_ids)
        self._print(self._progress_line(snap))
        return self._emit(event, **snap, **extra)

    # ------------------------------------------------------------------ control

    def goals_met(self) -> bool:
        return all(self.tracker.examples(s) >= self.args.goal for s in STEPS)

    def spent(self) -> float:
        return self.tracker.cost_usd + self.gpu_done

    def plan(self) -> dict[str, Any]:
        a = self.args
        return {"target": self.spec.name, "hardware": self.spec.hardware, "runner": a.runner, "slots": a.slots,
                "minutes": a.minutes, "race_minutes": self._race_s() / 60, "seeds": list(a.seeds),
                "model": "mock-llm" if a.dry_run else a.model, "record": self.record, "dry_run": a.dry_run,
                "run_budget_s": self.run_budget_s, "overfit_budget_s": self.overfit_budget_s,
                "usd_cap": a.usd_cap, "goal_per_step": a.goal, "coverage_ratio": a.coverage_ratio,
                "probe_rate": a.probe_rate, "probe_kinds": list(self.policy.kinds), "pipelining": a.pipelining,
                "db": self.db.path, "jsonl": str(self.jsonl_path),
                "session_ids": [session_id(self.date, self.spec.name, s) for s in a.seeds],
                "prompts": prompt_freeze_status()}

    def _race_s(self) -> float:
        a = self.args
        return (a.race_minutes * 60) if a.race_minutes else (a.minutes * 60 / max(1, len(a.seeds)))

    async def run(self) -> dict[str, Any]:
        a = self.args
        self.t_start = time.time()
        deadline = self.t_start + a.minutes * 60
        min_race_s = max(2.0 * self.run_budget_s, 2.0)
        self._emit("start", plan=self.plan())
        i = 0
        stop = "minutes"
        try:
            while True:
                if a.stop_at_goal and self.goals_met():
                    stop = "goals_met"
                    break
                remaining_s = deadline - time.time()
                remaining_usd = a.usd_cap - self.spent()
                if remaining_usd <= 0.01:
                    stop = "usd_cap"
                    break
                if remaining_s < min_race_s:
                    stop = "minutes"
                    break
                seed = a.seeds[i % len(a.seeds)]
                result = await self._run_race(seed, min(self._race_s(), remaining_s), remaining_usd, min_race_s)
                i += 1
                if result is None:
                    stop = "usd_cap"
                    break
                if self.shared["storm"] is not None:
                    stop = "failure_storm"
                    self._print(f"[collect] {self.shared['storm']}")
                    break
                if result.get("stop_reason") == "error":
                    stop = "error"
                    break
        finally:
            for c in self._clients:
                close = getattr(c, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except Exception:  # noqa: BLE001
                        pass
            self._clients.clear()
            self.db.flush_metrics()
        final = self.report(event="done", stop=stop, probes_in_db=len(self.store.probes()))
        self._print(f"[collect] done ({stop}). progress log: {self.jsonl_path}  db: {self.db.path}")
        return final

    async def _run_race(self, seed: int, race_s: float, remaining_usd: float, min_race_s: float) -> dict[str, Any] | None:
        a, spec = self.args, self.spec
        inner_runner = self.runner_factory(a, spec)
        runner = BudgetOverrideRunner(inner_runner)
        slots = runner.slots()[: a.slots]
        price = sum(float(runner.price_per_second(s.hardware)) for s in slots)
        if any(s.hardware == "H100" for s in slots):
            raise CollectionRefused("H100 slots are never used for collection (budget guard)")
        budget_s = race_s
        if price > 0:
            budget_s = min(budget_s, remaining_usd / price)
        if budget_s < min_race_s:
            self._print(f"[collect] remaining ${remaining_usd:.2f} buys {budget_s:.0f}s of {len(slots)} slots; stopping")
            return None
        race_id = new_id()
        sid = session_id(self.date, spec.name, seed)
        cfg = RaceConfig(
            name=f"collect/{spec.name}/seed-{seed}", target=spec.name, arms=[ArmKind.opus], budget_s=budget_s,
            gpus_per_arm=a.slots, seed=seed, cost_cap_usd=max(0.01, remaining_usd), run_budget_s=self.run_budget_s,
            pipelining=a.pipelining, run_baseline_first=True, mode="dev",
            llm="mock" if a.dry_run else "mubit", runner=a.runner, record=self.record,
            model=a.model, max_estimated_gpu_usd=max(0.01, remaining_usd) if price > 0 else None,
        )
        llm = self.llm_factory(a, spec, seed)
        if not a.dry_run and getattr(llm, "benchmark_safe", True) is False:
            raise CollectionRefused("a real collection run got a mock / replay LLM client")
        if hasattr(llm, "aclose") and llm not in self._clients:
            self._clients.append(llm)
        prompts = load_prompts()
        inner = LLMResearcher(kind=ArmKind.opus, llm=llm, spec=spec, session_id=sid, prompts=prompts,
                              fallback_model=OPUS_MODEL, record=self.record)
        researcher = CollectionResearcher(
            inner, spec=spec, tracker=self.tracker, policy=self.policy, store=self.store, runner=runner,
            race_id=race_id, run_id=self.run_id, seed=seed, run_budget_s=self.run_budget_s,
            overfit_budget_s=self.overfit_budget_s, shared=self.shared)
        deps = ArmDeps(kind=ArmKind.opus, researcher=researcher, runner=runner,  # type: ignore[arg-type]
                       model="mock-llm" if a.dry_run else a.model, llm_mode="mock" if a.dry_run else "mubit")
        engine = RaceEngine(cfg, [deps], spec=spec, race_id=race_id, db=self.db, bus=EventBus(), tick_s=self.tick_s)
        researcher.arm_id = engine.arm_ids[ArmKind.opus]
        researcher.storm_n = a.storm_n
        researcher.stop_cb = engine.stop
        self.store.add_race(race_id=race_id, run_id=self.run_id, target=spec.name, seed=seed, session_id=sid,
                            model=deps.model, llm_mode=deps.llm_mode, runner=a.runner, record=int(self.record),
                            prompt_version=getattr(prompts, "version", None), started_at=time.time(), ended_at=None,
                            stats=None)
        self._print(f"[collect] race {race_id[:8]} seed {seed}: {budget_s / 60:.1f} min on {len(slots)} x "
                    f"{slots[0].hardware if slots else '?'} ({a.runner}), session {sid}, record={self.record}")
        self._emit("race_start", race_id=race_id, seed=seed, session_id=sid, budget_s=budget_s,
                   cost_cap_usd=cfg.cost_cap_usd, prompt_version=getattr(prompts, "version", None))

        reporter = asyncio.create_task(self._reporter(engine, researcher))
        try:
            await engine.run()
        finally:
            reporter.cancel()
            try:
                await reporter
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            researcher.close_open_probes()
        summaries = engine.summaries
        gpu = sum(s.gpu_cost_usd for s in summaries)
        probe_ids = set(researcher.probes)
        mix = self._mix(engine, probe_ids)
        self.gpu_done += gpu
        self.mix_done += mix
        arm = engine.arms.get(ArmKind.opus)
        stop_reason = arm.stop_reason if arm else None
        stats = {"gpu_cost_usd": gpu, "llm_cost_usd": sum(s.llm_cost_usd for s in summaries),
                 "stop_reason": stop_reason, "error": arm.error if arm else None, "mix": dict(mix),
                 "probes": len(probe_ids), "best_score": summaries[0].best_score if summaries else None}
        self.store.finish_race(race_id, time.time(), stats)
        self.races.append({"race_id": race_id, "seed": seed, **stats})
        self._export_race(race_id)
        self.report(event="race_end", race_id=race_id, seed=seed, race=stats)
        if arm is not None and arm.error:
            self._print(f"[collect] race {race_id[:8]} stopped with an error: {arm.error}")
        return stats

    async def _reporter(self, engine: RaceEngine, researcher: CollectionResearcher) -> None:
        every = max(1.0, self.args.log_every_min * 60)
        while True:
            await asyncio.sleep(every)
            self.report(engine, set(researcher.probes))
            if self.args.stop_at_goal and self.goals_met():
                self._print("[collect] every step reached its goal; stopping the race")
                engine.stop("goals_met")
                return

    def _export_race(self, race_id: str) -> None:
        try:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            with (self.export_dir / f"{race_id}.jsonl").open("w", encoding="utf-8") as f:
                for line in self.db.export_race_jsonl(race_id):
                    f.write(line)
                for p in self.store.probes(race_id):
                    f.write(json.dumps({"table": "collection_probes", "row": p}, sort_keys=True, default=str) + "\n")
        except Exception as e:  # noqa: BLE001 - an export failure must not stop the night
            log.warning("export of race %s failed: %s", race_id, e)


# =========================================================================== CLI


def _seeds(s: str) -> tuple[int, ...]:
    out = tuple(int(x) for x in s.replace(" ", "").split(",") if x != "")
    if not out:
        raise argparse.ArgumentTypeError("at least one seed")
    return out


def _kinds(s: str) -> tuple[str, ...]:
    out = tuple(x for x in s.replace(" ", "").split(",") if x)
    bad = [x for x in out if x not in PROBE_KINDS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown probe kinds {bad}; choose from {list(PROBE_KINDS)}")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Overnight Opus example collection through Mubit (plan.md step 7).")
    ap.add_argument("--target", default="rl", help=f"target name ({', '.join(list_targets())})")
    ap.add_argument("--minutes", type=float, default=None, help="total wall-clock (default 60; dry run 2)")
    ap.add_argument("--race-minutes", type=float, default=None, help="per race (default minutes / #seeds)")
    ap.add_argument("--slots", type=int, default=4, help="concurrent runs per race (Modal Starter: max 10 GPUs)")
    ap.add_argument("--usd-cap", type=float, default=20.0, help="total LLM + GPU $ cap across all races")
    ap.add_argument("--runner", choices=["local", "modal"], default=None, help="default modal; dry run: local")
    ap.add_argument("--seeds", type=_seeds, default=(0, 1, 2), help="comma list, cycled across races")
    ap.add_argument("--model", default=OPUS_MODEL, help="provider model ID as sent to Mubit")
    ap.add_argument("--dry-run", action="store_true", help="MockLLM + LocalRunner, no network, $0")
    ap.add_argument("--no-record", action="store_true", help="real calls with x-middleware-record: off (smoke)")
    ap.add_argument("--yes", action="store_true", help="confirm a paid run (real LLM and/or Modal)")
    ap.add_argument("--run-budget-s", type=int, default=None, help="per-run training budget (default spec)")
    ap.add_argument("--log-every-min", type=float, default=None, help="progress interval (default 5; dry run 0.25)")
    ap.add_argument("--goal", type=int, default=DEFAULT_GOAL, help="valid calls wanted per step")
    ap.add_argument("--coverage-ratio", type=float, default=0.5,
                    help="probe when diagnose/judge valid calls < ratio x propose valid calls")
    ap.add_argument("--probe-rate", type=float, default=0.25, help="max share of launched experiments that are probes")
    ap.add_argument("--probe-kinds", type=_kinds, default=PROBE_KINDS, help="comma list of diverge,trap,overfit")
    ap.add_argument("--overfit-budget-frac", type=float, default=0.5, help="per-run budget share for overfit probes")
    ap.add_argument("--storm-n", type=int, default=DEFAULT_STORM_N,
                    help="stop after this many consecutive runs fail with the same error")
    ap.add_argument("--pipelining", action="store_true", help="ask for the next batch while runs are in flight")
    ap.add_argument("--no-stop-at-goal", action="store_true", help="keep going after every step hits --goal")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="collection SQLite DB (not the benchmark DB)")
    ap.add_argument("--out-dir", default=str(COLLECTION_DIR), help="progress JSONL + race exports")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def args_from_namespace(ns: argparse.Namespace) -> CollectArgs:
    dry = bool(ns.dry_run)
    return CollectArgs(
        target=ns.target, minutes=ns.minutes if ns.minutes is not None else (2.0 if dry else 60.0),
        race_minutes=ns.race_minutes, slots=ns.slots, usd_cap=ns.usd_cap,
        runner=ns.runner or ("local" if dry else "modal"), seeds=tuple(ns.seeds), model=ns.model, dry_run=dry,
        record=not (ns.no_record or dry), run_budget_s=ns.run_budget_s,
        log_every_min=ns.log_every_min if ns.log_every_min is not None else (0.25 if dry else 5.0),
        goal=ns.goal, coverage_ratio=ns.coverage_ratio, probe_rate=ns.probe_rate, probe_kinds=tuple(ns.probe_kinds),
        overfit_budget_frac=ns.overfit_budget_frac, pipelining=ns.pipelining, db_path=ns.db, out_dir=ns.out_dir,
        stop_at_goal=not ns.no_stop_at_goal, storm_n=ns.storm_n,
    )


def preflight(ns: argparse.Namespace, args: CollectArgs) -> list[str]:
    """Problems that refuse the run before anything is constructed (no client, no runner)."""
    errs: list[str] = []
    try:
        spec = get_target(args.target).spec
    except KeyError as e:
        return [str(e)]
    if args.dry_run and ns.runner == "modal":
        errs.append("--dry-run never uses Modal: drop --runner modal")
    if args.slots < 1 or args.slots > 100:
        errs.append("--slots must be in 1..100")
    if args.runner == "modal" and spec.hardware != "CPU" and args.slots > 10:
        errs.append("Modal Starter allows at most 10 concurrent GPUs: lower --slots")
    if spec.hardware == "H100":
        errs.append("H100 targets are not used for collection (budget guard)")
    if args.storm_n < 2:
        errs.append("--storm-n must be >= 2")
    if args.minutes <= 0 or args.usd_cap <= 0:
        errs.append("--minutes and --usd-cap must be > 0")
    if not 0 <= args.probe_rate <= 1 or args.coverage_ratio < 0 or not 0 < args.overfit_budget_frac <= 1:
        errs.append("--probe-rate in [0,1], --coverage-ratio >= 0, --overfit-budget-frac in (0,1]")
    if args.record:
        probs = record_problems()
        if probs:
            errs.append("refusing to record examples: the prompts are not FROZEN.\n  "
                        + "\n  ".join(probs)
                        + "\n  A human must freeze each prompt (backend/agent/prompts/__init__.py FROZEN[step] = "
                          "CURRENT_VERSION[step]) before any record=True call; editing a prompt after collection "
                          "starts creates a new Mubit task and discards the examples. Use --dry-run (free) or "
                          "--no-record (paid smoke, nothing recorded) meanwhile.")
        if args.model != OPUS_MODEL:
            errs.append(f"recording with --model {args.model}: examples must come from {OPUS_MODEL} "
                        "(a different model ID is a different Mubit task)")
    paid = not args.dry_run
    if paid and not ns.yes:
        errs.append("this run makes paid calls (Opus through Mubit"
                    + (", Modal GPUs" if args.runner == "modal" else "")
                    + "): re-run with --yes after checking the plan above")
    return errs


def _print_plan(args: CollectArgs) -> None:
    spec = get_target(args.target).spec
    from workers.runners.base import DEFAULT_PRICE_PER_SECOND

    price = 0.0 if args.runner == "local" else DEFAULT_PRICE_PER_SECOND.get(spec.hardware, 0.0)
    est = price * args.slots * args.minutes * 60
    print(f"[collect] plan: target={spec.name} ({spec.hardware}) runner={args.runner} slots={args.slots} "
          f"minutes={args.minutes:g} seeds={list(args.seeds)} model={'mock-llm' if args.dry_run else args.model} "
          f"record={args.record}")
    print(f"[collect] budget: est GPU $ <= {est:.2f} (all slots paid the whole time), total cap ${args.usd_cap:.2f} "
          f"(LLM + GPU); goal {args.goal} valid calls per step")


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if ns.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = args_from_namespace(ns)
    try:
        _print_plan(args)
    except KeyError:
        pass
    errs = preflight(ns, args)
    if errs:
        for e in errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if args.dry_run:
        st = prompt_freeze_status()
        drafts = [s for s, v in st.items() if not v["ok"]]
        if drafts:
            print(f"[collect] note: prompts not frozen ({', '.join(drafts)}); fine for a dry run, "
                  "a recorded run would be refused")
    collector = Collector(args)
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        print(f"[collect] interrupted; progress so far in {collector.jsonl_path}", file=sys.stderr)
        return 130
    except CollectionRefused as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    finally:
        collector.db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
