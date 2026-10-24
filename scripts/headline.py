"""night/bench headline table: the D31/D36 comparison and the benchmark.md section 6b framings.

Reads races from the benchmark SQLite DB (opened READ-ONLY, ``mode=ro``) and/or JSONL
exports, picks a set of races (``--race`` ids, a ``--preset`` name, or every race), and
prints, per race setup (target, budget, slots, run budget, goal):

1. **The headline table** (D31 / D36 layout): one column per arm, in this order when present:
   Standard AI assistant (Opus 5.5) | Busy Machines with Opus | Mubit + Busy Machines |
   Automatic search (no AI, Optuna). Every cell is mean ± std over seeds with its n.
2. **The section 6b framings**, each computed by its fixed formula and naming its baseline in
   the same sentence. Pairs: Mubit + BM vs Opus (headline), Busy Machines with Opus vs Opus
   (our engine alone), Mubit + BM vs Busy Machines with Opus (Mubit's models alone), Mubit +
   BM vs Optuna. A worse result is printed as worse ("12% more expensive"), never dropped.
3. **Evidence**: per number and arm, how many Mubit request IDs / LLM calls and Modal call IDs
   / runs sit behind it (the windows each number actually uses: up to the goal, up to the
   Opus spend, or the whole race).
4. **Gates**: a framing is printed only when every race of the setup is finished, in benchmark
   mode, not mock/replay, with a frozen (not provisional) goal, the arms got identical slots,
   and both arms have >= 3 distinct seeds with a value (for "reach" metrics: reached in every
   seed). Otherwise the line says ``not measured`` and why. The table itself always prints,
   stamped NOT A HEADLINE RESULT when any gate fails. Also (fairness review, D41):
   - races are pooled only with the same **fairness fingerprint** (``races.fairness_json``:
     arm set, per-arm engine features + rule settings, prompt versions actually called,
     record modes, race settings, Mubit project routing); a race without a manifest, or with a
     correction applied mid-race, blocks its setup;
   - every framing that involves the Mubit arm needs **evidence that an adapter served** (a
     per-call ``served_by`` naming a Mubit model, or request IDs listed under
     ``adapter_served.request_ids`` in the manifest); none exists today, so those lines say
     ``not measured: no evidence an adapter served``;
   - an arm whose early stops ran on a remote runner without a confirmed stop (D41 #1) is
     kept out of framings;
   - **ablation races are refused** (``mode="ablation"``, or any ablation-only arm such as
     ``fast`` = a smaller faster model in the same loop, not a Mubit adapter): ``--preset`` of an
     ablation preset exits with an error, and ablation races selected any other way are dropped
     with a note (and, defensively, blocked if they ever reach a setup).
   Each column's engine features and LLM setup are printed; early stopping is shown as slot
   time freed (never $, never in a $ framing); a judge-policy rule with sigma = 0 (LCBench's
   deterministic surrogate) is marked DEGENERATE and its counts are not real-gain stats.
5. **JSON** for the slides (``--json``): every number, its inputs, n, and the gate verdicts.

Same-quality and same-money views (D30), computed per race (same seed) and then averaged:
- *Same quality*: $ and time each arm needed to reach the Opus arm's final held-out quality
  in that race (the Opus arm's own value is the $/time at which it found its best).
- *Same money*: each arm's best held-out quality among runs finished while its spend
  (GPU $ accrued + LLM $ of calls completed) was <= the Opus arm's total spend in that race.
  An arm that spent less than Opus over the whole race keeps its whole race.

Quality framings by target (the metric each target reports, benchmark.md 6b):
- ``lm`` bits per byte (lower is better): "N× lower" = start ÷ end, "X% lower" = 1 − end/start
- ``rl`` mean return, framed as a penalty (= −return): same two formulas on the penalty
- ``cifar`` / ``lcbench`` accuracy: "X% fewer errors" = (err_start − err_end) ÷ err_start
"Start" is the starting code (pooled held-out score of every arm's starting-config runs) for
"the research agent improves the model", or the baseline arm's value for a head-to-head line.
"X% of the gap closed" = (end − start) ÷ (best_known − start) only with ``--best-known``.

Usage:
    python -m uv run --extra ml python scripts/headline.py                       # data/bench/bench.db, all races
    python -m uv run --extra ml python scripts/headline.py --preset rl-wide-15m --json data/bench/headline.json
    python -m uv run --extra ml python scripts/headline.py --race ID1 --race ID2 --race ID3 --out headline.md
    options: --db PATH  --jsonl FILE...  --best-known rl=-120  --min-seeds 3  --pair mubit:opus

No LLM calls, no compute, no writes to the DB.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.benchmark.metrics import gpu_cost_until, is_counted, quality, summarize_arm  # noqa: E402
from backend.core.types import ArmKind  # noqa: E402


def _load_analyze_races() -> Any:
    """scripts/ is not a package: load analyze_races.py once (shared with its tests)."""
    if "analyze_races" in sys.modules:
        return sys.modules["analyze_races"]
    spec = importlib.util.spec_from_file_location("analyze_races", Path(__file__).with_name("analyze_races.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_races"] = mod
    spec.loader.exec_module(mod)
    return mod


ar = _load_analyze_races()
Rows = dict[str, list[dict[str, Any]]]
Stat = ar.Stat

DEFAULT_DB = REPO_ROOT / "data" / "bench" / "bench.db"
MIN_SEEDS = 3

# --------------------------------------------------------------------------- arms

# Canonical arm keys, in column order (D36: Opus | BM with Opus | Mubit + BM; D31 stock column
# if a race ever has one; Optuna last).
ARM_ORDER = ("opus", "opus_bm", "mubit_stock", "mubit", "optuna", "fast", "mock")
ARM_ALIASES = {
    "opus_bm": "opus_bm", "bm_opus": "opus_bm", "opus_busy": "opus_bm", "busy_opus": "opus_bm",
    "opusbm": "opus_bm", "opus_engine": "opus_bm", "busy_machines_opus": "opus_bm",
    "mubit_stock": "mubit_stock", "mubit_baseline": "mubit_stock", "stock": "mubit_stock",
}
ARM_COLUMN = {
    "opus": "Standard AI assistant (Opus 5.5)",
    "opus_bm": "Busy Machines with Opus",
    "mubit_stock": "Mubit baseline (stock)",
    "mubit": "Mubit + Busy Machines",
    "optuna": "Automatic search (no AI, Optuna)",
    "fast": "Smaller faster model in the same loop (Haiku 5.5, not a Mubit adapter; ablation only)",
    "mock": "Mock LLM (dev only)",
}
# How an arm is named inside a sentence.
ARM_NOUN = {
    "opus": "the Opus 5.5 agent",
    "opus_bm": "Busy Machines with Opus",
    "mubit_stock": "the stock Mubit baseline",
    "mubit": "Mubit + Busy Machines",
    "optuna": "Optuna (no AI)",
    "fast": "the smaller faster model in the same loop (not a Mubit adapter)",
    "mock": "the mock LLM",
}
# (arm, baseline): the headline line first, then the decomposition (D36), then no-AI search
# (Opus vs Optuna too: "the research agent improves the model" needs its no-AI baseline, R22).
DEFAULT_PAIRS = (("mubit", "opus"), ("opus_bm", "opus"), ("mubit", "opus_bm"), ("mubit", "optuna"),
                 ("opus", "optuna"))


def canon_arm(kind: str) -> str:
    k = str(kind).strip().lower().replace("-", "_").replace("+", "_").replace(" ", "_")
    return ARM_ALIASES.get(k, k)


def noun(kind: str) -> str:
    return ARM_NOUN.get(kind, kind)


def column(kind: str) -> str:
    return ARM_COLUMN.get(kind, kind)


def subject(kind: str) -> str:
    """The arm as the subject of a sentence ("The Opus 5.5 agent", "Mubit + Busy Machines")."""
    n = noun(kind)
    return n[:1].upper() + n[1:]


def possessive(name: str) -> str:
    return name + ("'" if name.endswith("s") else "'s")


def budget_text(budget_s: float) -> str:
    return f"{budget_s / 60:g}-min" if budget_s % 60 == 0 else f"{budget_s:g}-s"


_KNOWN_KINDS = {k.value for k in ArmKind}


def _safe_kind(kind: str) -> str:
    """``summarize_arm`` builds an ``ArmSummary`` (ArmKind enum). A kind the enum does not know
    yet (e.g. ``opus-bm``) only labels that summary, so it is computed under a stand-in label
    and the real kind is restored by the caller. The numbers do not depend on the label."""
    return kind if kind in _KNOWN_KINDS else ArmKind.mock.value


# --------------------------------------------------------------------------- quality framings

# target -> (kind, metric noun, transform to the framed quantity)
#   kind "cost": framed quantity is a cost (lower is better): N× = start / end, % = 1 - end/start
#   kind "accuracy": framed quantity is the error rate 1 - acc: % = (err_start - err_end) / err_start
QUALITY_FRAMING: dict[str, tuple[str, str]] = {
    "lm": ("cost", "bits per byte"),
    "rl": ("cost", "penalty"),
    "cifar": ("accuracy", "errors"),
    "lcbench": ("accuracy", "errors"),
}


def framed_quantity(target: str, q: float | None) -> float | None:
    """The quantity the 6b formula is applied to: bpb as is, RL penalty = -return, error = 1 - acc."""
    if q is None or target not in QUALITY_FRAMING:
        return None
    if target == "rl":
        return -q
    if QUALITY_FRAMING[target][0] == "accuracy":
        return 1.0 - q
    return q


# --------------------------------------------------------------------------- per race, per arm


@dataclass
class Evidence:
    request_ids: int = 0   # distinct Mubit request IDs among the LLM calls behind the number
    llm_calls: int = 0
    modal_ids: int = 0     # distinct Modal call IDs (run_ref "fc-...") among the runs behind it
    other_run_ids: int = 0  # run refs that are not Modal call IDs (surrogate, local PID)
    runs: int = 0
    races: int = 0

    def add(self, other: Evidence) -> None:
        for k in ("request_ids", "llm_calls", "modal_ids", "other_run_ids", "runs", "races"):
            setattr(self, k, getattr(self, k) + getattr(other, k))


def _evidence(calls: Sequence[Mapping[str, Any]], exps: Sequence[Mapping[str, Any]]) -> Evidence:
    req = {c["mubit_request_id"] for c in calls if c.get("mubit_request_id")}
    refs = {str(e["run_ref"]) for e in exps if e.get("run_ref")}
    modal = {r for r in refs if r.startswith("fc-")}
    return Evidence(request_ids=len(req), llm_calls=len(calls), modal_ids=len(modal),
                    other_run_ids=len(refs - modal), runs=len(exps), races=1)


def _completed(c: Mapping[str, Any]) -> float:
    v = c.get("completed_at")
    if v is None:
        v = c.get("created_at")
    return float(v or 0.0)


def _window(calls: Sequence[Mapping[str, Any]], exps: Sequence[Mapping[str, Any]], t: float | None) -> Evidence:
    """Evidence behind a number measured up to time ``t`` (None = the whole race): LLM calls
    completed by then (their $ is counted) and runs started by then (their GPU time is)."""
    if t is None:
        return _evidence(calls, exps)
    return _evidence([c for c in calls if _completed(c) <= t],
                     [e for e in exps if e.get("started_at") is not None and float(e["started_at"]) <= t])


def spend_at(t: float, intervals: Sequence[Mapping[str, Any]], calls: Sequence[Mapping[str, Any]]) -> float:
    """GPU $ accrued by ``t`` + LLM $ of calls completed by ``t`` (the $-to-target rule)."""
    return gpu_cost_until(intervals, t) + sum(float(c.get("cost_usd") or 0.0) for c in calls if _completed(c) <= t)


def best_within_spend(direction: str, budget_usd: float, experiments: Sequence[Mapping[str, Any]],
                      intervals: Sequence[Mapping[str, Any]], calls: Sequence[Mapping[str, Any]]
                      ) -> tuple[float | None, float | None]:
    """(best counted held-out quality among runs that finished while spend <= budget, the
    time of the last such run). Spend only grows, so the scan stops at the first run past it."""
    best: float | None = None
    t_last: float | None = None
    for e in sorted((e for e in experiments if is_counted(e)), key=lambda e: float(e.get("ended_at") or 0.0)):
        t = float(e["ended_at"])
        if spend_at(t, intervals, calls) > budget_usd + 1e-12:
            break
        q = quality(e)
        t_last = t
        if q is not None and (best is None or (q < best if direction == "min" else q > best)):
            best = q
    return best, t_last


@dataclass
class RaceArm:
    race_id: str
    seed: int
    arm: str                       # canonical kind
    raw_kind: str
    # same budget (the whole race)
    cost_to_goal_usd: float | None
    time_to_goal_s: float | None
    gpu_idle_pct: float
    best_quality: float | None
    shipped_quality: float | None
    start_quality: float | None
    experiments_finished: int
    total_cost_usd: float
    gpu_cost_usd: float
    llm_cost_usd: float
    experiments_per_usd: float | None
    trap_runs: int
    fake_best_accepted: int
    # same quality / same money (relative to this race's Opus arm)
    cost_to_opus_quality_usd: float | None = None
    time_to_opus_quality_s: float | None = None
    quality_at_opus_spend: float | None = None
    evidence: dict[str, Evidence] = field(default_factory=dict)


def race_arms(rows: Rows, *, direction: str, goal: float, is_dev: bool) -> list[RaceArm]:
    race = rows["races"][0]
    t0 = float(race["started_at"])
    safe = dict(rows)
    safe["arms"] = [{**a, "kind": _safe_kind(str(a["kind"]))} for a in rows["arms"]]
    runs = ar.arm_runs_for_race(safe, direction=direction, target_value=goal, is_dev=is_dev)
    out: list[RaceArm] = []
    sliced: dict[str, tuple[list, list, list]] = {}
    for arm, run in zip(rows["arms"], runs, strict=True):
        aid = arm["id"]
        exps = [e for e in rows.get("experiments", []) if e["arm_id"] == aid]
        ivs = [i for i in rows.get("gpu_intervals", []) if i["arm_id"] == aid]
        calls = [c for c in rows.get("llm_calls", []) if c["arm_id"] == aid]
        kind = canon_arm(arm["kind"])
        sliced[kind] = (exps, ivs, calls)
        traps = [e for e in exps if ar._truthy(e.get("is_trap")) and e.get("status") == "succeeded"]
        t_goal = t0 + run.time_to_target_s if run.time_to_target_s is not None else None
        whole = _window(calls, exps, None)
        ra = RaceArm(
            race_id=race["id"], seed=int(race.get("seed") or 0), arm=kind, raw_kind=str(arm["kind"]),
            cost_to_goal_usd=run.cost_to_target_usd, time_to_goal_s=run.time_to_target_s,
            gpu_idle_pct=run.gpu_idle_pct, best_quality=run.best_quality, shipped_quality=run.shipped_quality,
            start_quality=run.start_quality, experiments_finished=run.experiments_finished,
            total_cost_usd=run.total_cost_usd, gpu_cost_usd=run.gpu_cost_usd, llm_cost_usd=run.llm_cost_usd,
            experiments_per_usd=(run.experiments_finished / run.total_cost_usd) if run.total_cost_usd > 0 else None,
            trap_runs=len(traps), fake_best_accepted=sum(1 for e in traps if ar._truthy(e.get("is_best"))),
        )
        for m in ("gpu_idle_pct", "best_quality", "shipped_quality", "experiments_finished",
                  "experiments_per_usd", "total_cost_usd", "fake_best_accepted"):
            ra.evidence[m] = whole
        goal_ev = _window(calls, exps, t_goal) if t_goal is not None else Evidence(races=1)
        ra.evidence["cost_to_goal_usd"] = ra.evidence["time_to_goal_s"] = goal_ev
        out.append(ra)

    kinds = [r.arm for r in out]
    if len(set(kinds)) != len(kinds):
        raise ValueError(f"race {race['id']}: two arms map to the same column {kinds}")
    opus = next((r for r in out if r.arm == "opus"), None)
    if opus is None:
        return out
    for ra in out:
        exps, ivs, calls = sliced[ra.arm]
        if opus.best_quality is not None:
            s, _ = summarize_arm(_safe_kind(ra.raw_kind), direction=direction, target_value=opus.best_quality,
                                 race_start=t0, experiments=exps, intervals=ivs, llm_calls=calls)
            ra.cost_to_opus_quality_usd = s.cost_to_target_usd
            ra.time_to_opus_quality_s = s.time_to_target_s
            t_hit = t0 + s.time_to_target_s if s.time_to_target_s is not None else None
            ev = _window(calls, exps, t_hit) if t_hit is not None else Evidence(races=1)
        else:
            ev = Evidence(races=1)
        ra.evidence["cost_to_opus_quality_usd"] = ra.evidence["time_to_opus_quality_s"] = ev
        q, t_last = best_within_spend(direction, opus.total_cost_usd, exps, ivs, calls)
        ra.quality_at_opus_spend = q
        ra.evidence["quality_at_opus_spend"] = _window(calls, exps, t_last) if t_last is not None else Evidence(races=1)
    return out


# --------------------------------------------------------------------------- groups

# Per-arm numbers aggregated over seeds (field, row label, kind of number).
ROWS: tuple[tuple[str, str, str], ...] = (
    ("cost_to_goal_usd", "Cost to reach the goal", "usd_reach"),
    ("time_to_goal_s", "Time to reach the goal", "min_reach"),
    ("gpu_idle_pct", "GPUs waiting", "pct"),
    ("best_quality", "Model quality @ same budget (best held-out score)", "q"),
    ("cost_to_opus_quality_usd", "Same quality: cost to reach the Opus agent's final quality", "usd_reach"),
    ("time_to_opus_quality_s", "Same quality: time to reach the Opus agent's final quality", "min_reach"),
    ("quality_at_opus_spend", "Same money: best held-out score at the Opus agent's total spend", "q"),
    ("experiments_finished", "Experiments finished", "n"),
    ("experiments_per_usd", "Experiments per $", "per_usd"),
    ("total_cost_usd", "Total spend (GPU $ + LLM $)", "usd"),
    ("shipped_quality", "Shipped model, held-out score", "q"),
    ("fake_best_accepted", "Fake improvements accepted as a new best (in-race traps)", "n"),
)
REACH_FIELDS = {"cost_to_goal_usd", "time_to_goal_s", "cost_to_opus_quality_usd", "time_to_opus_quality_s"}


@dataclass
class ArmGroup:
    arm: str
    seeds: list[int]
    race_ids: list[str]
    stats: dict[str, Any]           # field -> Stat
    evidence: dict[str, Evidence]   # field -> summed over races
    trap_runs: int

    @property
    def n_seeds(self) -> int:
        return len(set(self.seeds))


@dataclass
class Framing:
    id: str
    title: str
    arm: str
    baseline: str
    formula: str
    measured: bool
    value: float | None
    sentence: str
    inputs: dict[str, Any]
    reasons: list[str]
    per_seed: list[float] = field(default_factory=list)


@dataclass
class Group:
    target: str
    direction: str
    goal: float
    goal_source: str
    budget_s: float
    gpus_per_arm: int
    run_budget_s: Any
    runner: str
    names: list[str]
    race_ids: list[str]
    seeds: list[int]
    arms: dict[str, ArmGroup]
    start_quality: float | None
    best_known: float | None
    blockers: list[str]              # setup-level reasons no framing can be printed
    warnings: list[str]
    framings: list[Framing]
    runs: list[RaceArm]
    fingerprint_id: str | None = None
    fingerprint: dict[str, Any] | None = None
    unused_prompts: dict[str, list[str]] = field(default_factory=dict)
    engine: dict[str, EngineStats] = field(default_factory=dict)
    # Mubit arm: (calls with adapter evidence, calls) per race; adapter_ok = >= 1 in every race.
    adapter: list[tuple[int, int]] = field(default_factory=list)

    @property
    def headline_ok(self) -> bool:
        return not self.blockers

    @property
    def adapter_ok(self) -> bool:
        return bool(self.adapter) and all(s > 0 for s, _ in self.adapter)

    def gate(self, kind: str) -> list[str]:
        """Arm-level reasons a framing that involves ``kind`` is not printed."""
        why: list[str] = []
        if kind == "mubit" and not self.adapter_ok:
            why.append(ADAPTER_REASON)
        e = self.engine.get(kind)
        if e is not None and e.stops_unconfirmed:
            why.append(f"{noun(kind)}: {e.stops_unconfirmed} early-stop cancel(s) on a remote runner with no "
                       "confirmation the remote run stopped (D41 #1: slot time after the cancel may be uncounted)")
        return why


def _race_config(rows: Rows) -> dict[str, Any]:
    c = rows["races"][0].get("config")
    return c if isinstance(c, dict) else {}


def _spec(target: str) -> Any:
    try:
        from workers.targets.base import get_target

        return get_target(target).spec
    except Exception:  # unknown target: no direction / placeholder goal
        return None


def setup_key(rows: Rows, is_dev: bool) -> tuple:
    """Races are pooled into one mean only when every element matches, including the fairness
    fingerprint (arm set, per-arm engine features, prompt versions called, record modes, race
    settings, Mubit routing; ``analyze_races.fairness_fingerprint``). None = no manifest."""
    race = rows["races"][0]
    cfg = _race_config(rows)
    goal = cfg.get("target_value")
    if goal is None:
        spec = _spec(race["target"])
        goal = spec.target_value if spec is not None else None
    return (race["target"], float(race["budget_s"]), int(race["gpus_per_arm"]), cfg.get("run_budget_s"),
            None if goal is None else float(goal), bool(is_dev), ar.fingerprint_id(ar.fairness_fingerprint(rows)))


def race_corrections(rows: Rows) -> int:
    """Corrections applied to this race: rows in ``corrections`` plus the manifest's list."""
    f = rows["races"][0].get("fairness")
    listed = (f.get("corrections") if isinstance(f, Mapping) else None) or []
    return len(rows.get("corrections", []) or []) + (len(listed) if isinstance(listed, (list, tuple)) else 1)


# Arms that only ever race in an ablation (backend.benchmark.configs.ABLATION_ONLY_ARMS).
ABLATION_ARMS = frozenset({"fast"})
ABLATION_REASON = "ablation race (smaller faster model in the same loop, not a Mubit adapter): never a headline result"


def is_ablation_race(rows: Rows) -> bool:
    """``mode="ablation"`` in the race config / fairness manifest, or an ablation-only arm."""
    race = rows["races"][0]
    cfg = _race_config(rows)
    f = race.get("fairness")
    if cfg.get("mode") == "ablation" or race.get("mode") == "ablation":
        return True
    if isinstance(f, Mapping) and (f.get("mode") == "ablation" or f.get("ablation")):
        return True
    kinds = {canon_arm(a.get("kind", "")) for a in rows.get("arms", []) or []}
    kinds |= {canon_arm(str(a)) for a in (cfg.get("arms") or [])}
    return bool(kinds & ABLATION_ARMS)


def race_blockers(rows: Rows, is_dev: bool) -> list[str]:
    race = rows["races"][0]
    rid = str(race["id"])[:8]
    cfg = _race_config(rows)
    why: list[str] = []
    if is_ablation_race(rows):
        why.append(f"race {rid} is an {ABLATION_REASON}")
    if race.get("status") != "finished":
        why.append(f"race {rid} not finished (status={race.get('status')})")
    if is_dev:
        why.append(f"race {rid} is dev/mock/replay ({'; '.join(dict.fromkeys(ar.dev_reasons(rows)))})")
    if cfg.get("provisional") or "[provisional goal]" in str(race.get("name") or ""):
        why.append(f"race {rid} ran with a provisional goal")
    elif cfg.get("target_value") is None:
        why.append(f"race {rid} has no goal in its config (spec placeholder, not frozen)")
    if ar.fairness_fingerprint(rows) is None:
        why.append(f"race {rid} has no fairness manifest (arm set / features / prompts unproven)")
    n_corr = race_corrections(rows)
    if n_corr:
        why.append(f"race {rid} had {n_corr} correction(s) applied mid-race (a human changed the outcome)")
    return why


# --------------------------------------------------------------------------- engine + adapter evidence

ADAPTER_REASON = "no evidence an adapter served"
_ADAPTER_MARKS = ("adapter", "mubit", "dedicated")
# Keys a stopped run's ``summary["early_stop"]`` may carry once the runner confirms that the
# remote input actually stopped (D41 #1). Without one, slot time after the cancel on a remote
# runner may be uncounted, so framings involving that arm are not printed. The scheduler writes
# ``runner_cancel_lag_s`` (and ``cancel_confirmed``) when the runner reports the lag: the slot was
# then held, and charged to the arm, until Modal confirmed the stop or the container was terminated.
CANCEL_CONFIRM_KEYS = ("runner_cancel_lag_s", "cancel_confirmed_at", "cancel_lag_s", "remote_stopped_at")
LOCAL_RUNNERS = ("surrogate", "local", "mock")
JUDGE_POLICY_NOTE = ("the judge policy cannot move race headline numbers while unsure claims are accepted "
                     "(accept_unsure); quote it only through its own ground-truth stats (R37/R40, D41 #5)")


def _served_by_adapter(c: Mapping[str, Any]) -> bool:
    v = str(c.get("served_by") or "").strip().lower()
    return bool(v) and any(m in v for m in _ADAPTER_MARKS)


def adapter_evidence(rows: Rows) -> tuple[int, int] | None:
    """(Mubit-arm calls with evidence that a Mubit adapter answered, Mubit-arm calls); None when
    the race has no Mubit arm. Evidence is either a per-call ``served_by`` that names a Mubit
    model (no such field is recorded today: the API response does not say who answered, Q1)
    or the call's Mubit request ID listed in the race's fairness manifest under
    ``adapter_served.request_ids`` (checked by hand against the console's "Served by")."""
    mubit = {a["id"] for a in rows.get("arms", []) if canon_arm(a["kind"]) == "mubit"}
    if not mubit:
        return None
    f = rows["races"][0].get("fairness")
    listed = ((f.get("adapter_served") or {}) if isinstance(f, Mapping) else {}).get("request_ids") or []
    confirmed = {str(x) for x in listed}
    calls = [c for c in rows.get("llm_calls", []) if c.get("arm_id") in mubit]
    served = sum(1 for c in calls if _served_by_adapter(c) or str(c.get("mubit_request_id") or "") in confirmed)
    return served, len(calls)


@dataclass
class EngineStats:
    """Busy Machines engine records for one column, summed over its races. ``slot_s_freed`` is
    slot time an early stop freed (engine estimate); it is never converted to $ here."""
    features: list[str] = field(default_factory=list)
    runs: int = 0
    early_stopped: int = 0
    early_stop_too_late: int = 0
    slot_s_freed: float = 0.0
    stops_unconfirmed: int = 0
    judge_claims: int = 0
    judge_llm_invalid: int = 0
    judge_no_reply: int = 0
    judge_rule_real: int = 0
    judge_rule_noise: int = 0
    judge_final: dict[str, int] = field(default_factory=dict)
    degenerate: bool = False
    degenerate_why: str = ""
    cost_sources: dict[str, int] = field(default_factory=dict)

    def add(self, o: EngineStats) -> None:
        for k in ("runs", "early_stopped", "early_stop_too_late", "slot_s_freed", "stops_unconfirmed",
                  "judge_claims", "judge_llm_invalid", "judge_no_reply", "judge_rule_real", "judge_rule_noise"):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        for src, dst in ((o.judge_final, self.judge_final), (o.cost_sources, self.cost_sources)):
            for k, v in src.items():
                dst[k] = dst.get(k, 0) + v
        self.degenerate = self.degenerate or o.degenerate
        self.degenerate_why = self.degenerate_why or o.degenerate_why
        self.features = self.features or list(o.features)


def _fp_arm(fp: Mapping[str, Any] | None, kind: str) -> Mapping[str, Any] | None:
    """The fingerprint entry of the column ``kind`` (fingerprint keys are stored kinds)."""
    return next((v for k, v in ((fp or {}).get("arms") or {}).items() if canon_arm(k) == kind), None)


def engine_stats(rows: Rows, fp: Mapping[str, Any] | None) -> dict[str, EngineStats]:
    race = rows["races"][0]
    remote = str(race.get("runner") or "") not in LOCAL_RUNNERS
    out: dict[str, EngineStats] = {}
    for a in rows.get("arms", []):
        kind = canon_arm(a["kind"])
        fpa = _fp_arm(fp, kind) or {}
        st = EngineStats(features=list(fpa.get("engine_features") or []))
        exps = [e for e in rows.get("experiments", []) if e.get("arm_id") == a["id"]]
        st.runs = sum(1 for e in exps if e.get("started_at") is not None)
        sigmas: list[float] = []
        for e in exps:
            summ = e.get("summary") if isinstance(e.get("summary"), Mapping) else {}
            es = summ.get("early_stop") or None
            if es:
                if es.get("outcome") == "stopped":
                    st.early_stopped += 1
                    st.slot_s_freed += float(es.get("slot_s_freed", es.get("saved_s")) or 0.0)
                    if remote and not any(es.get(k) is not None for k in CANCEL_CONFIRM_KEYS):
                        st.stops_unconfirmed += 1
                elif es.get("outcome") == "too_late":
                    st.early_stop_too_late += 1
            jp = summ.get("judge_policy") or None
            if jp:
                st.judge_claims += 1
                st.judge_llm_invalid += int(jp.get("llm_verdict") == "invalid")
                st.judge_no_reply += int(jp.get("llm_replied") is False)
                st.judge_rule_real += int(jp.get("rule") == "real")
                st.judge_rule_noise += int(jp.get("rule") == "noise")
                fin = str(jp.get("final"))
                st.judge_final[fin] = st.judge_final.get(fin, 0) + 1
                if jp.get("sigma") is not None:
                    sigmas.append(float(jp["sigma"]))
        table = ((fpa.get("judge_policy") or {}).get("table") or {})
        pts = [p for p in (table.get("points") or []) if isinstance(p, (list, tuple)) and len(p) >= 2]
        if "judge_policy" in st.features and ((sigmas and all(s == 0.0 for s in sigmas))
                                              or (pts and all(float(p[1]) == 0.0 for p in pts))):
            st.degenerate = True
            st.degenerate_why = ("calibrated seed spread σ = 0 (" + str(table.get("source") or "deterministic target")
                                 + "): the rule calls every claimed gain real")
        for c in rows.get("llm_calls", []):
            if c.get("arm_id") == a["id"]:
                src = str(c.get("cost_source") or "provider bill (not tagged)")
                st.cost_sources[src] = st.cost_sources.get(src, 0) + 1
        out[kind] = st
    return out


def _default_features() -> dict[str, list[str]]:
    try:
        from backend.benchmark.configs import DEFAULT_ENGINE_FEATURES

        return {canon_arm(k.value): sorted(v) for k, v in DEFAULT_ENGINE_FEATURES.items()}
    except Exception:  # noqa: BLE001 - only used for a warning
        return {}


def _stat(vals: Sequence[float | None]) -> Any:
    return ar.stat(list(vals))


def build_groups(kept: list[tuple[Rows, bool]], *, best_known: Mapping[str, float] | None = None,
                 min_seeds: int = MIN_SEEDS, pairs: Sequence[tuple[str, str]] = DEFAULT_PAIRS) -> tuple[list[Group], list[str]]:
    best_known = best_known or {}
    notes: list[str] = []
    by_key: dict[tuple, list[tuple[Rows, bool]]] = {}
    for rows, dev in kept:
        by_key.setdefault(setup_key(rows, dev), []).append((rows, dev))
    # Same target / budget / slots / goal but a different fairness fingerprint: kept apart,
    # and the note says what differs.
    by_base: dict[tuple, dict[str | None, list[Rows]]] = {}
    for key, members in by_key.items():
        by_base.setdefault(key[:-1], {}).setdefault(key[-1], []).extend(r for r, _ in members)
    for base, fps in sorted(by_base.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        if len(fps) < 2:
            continue
        items = sorted(fps.items(), key=lambda kv: str(kv[0]))
        ref_id, ref_rows = items[0]
        ref_fp = ar.fairness_fingerprint(ref_rows[0])
        parts = []
        for fid, rs in items:
            ids = ", ".join(f"`{str(r['races'][0]['id'])[:8]}`" for r in rs)
            diff = "" if fid == ref_id else (" — differs from `" + str(ref_id) + "` in: "
                                              + "; ".join(ar.fingerprint_diff(ref_fp, ar.fairness_fingerprint(rs[0]))))
            parts.append(f"fingerprint `{fid}` ({ids}){diff}")
        notes.append(f"{base[0]} races with the same budget/slots/goal were NOT pooled because their fairness "
                     f"fingerprints differ: " + " | ".join(parts))
    groups: list[Group] = []
    for key in sorted(by_key, key=lambda k: tuple(str(x) for x in k)):
        target, budget_s, gpus, run_budget, goal, _dev, fp_id = key
        members = sorted(by_key[key], key=lambda g: (g[0]["races"][0].get("seed") or 0, g[0]["races"][0]["id"]))
        spec = _spec(target)
        if spec is None or goal is None:
            notes.append(f"setup {key}: target {target!r} not in the registry or no goal: skipped")
            continue
        direction = spec.direction
        blockers: list[str] = []
        runs: list[RaceArm] = []
        for rows, dev in members:
            blockers.extend(race_blockers(rows, dev))
            runs.extend(race_arms(rows, direction=direction, goal=goal, is_dev=dev))
        warnings = ar.fairness_warnings([g for g, _ in members])
        # Unequal slots / hardware / price between arms breaks the fairness rule: no framing.
        blockers.extend(w for w in warnings if "(unfair)" in w)
        cfgs = [_race_config(r) for r, _ in members]
        goal_src = "race config" if all(c.get("target_value") is not None for c in cfgs) else "target spec placeholder"
        if any(c.get("provisional") for c in cfgs):
            goal_src += ", PROVISIONAL"
        elif goal_src == "race config":
            goal_src += " (frozen)"
        start_vals = [r.start_quality for r in runs if r.start_quality is not None]
        arms: dict[str, ArmGroup] = {}
        for kind in sorted({r.arm for r in runs}, key=lambda k: ARM_ORDER.index(k) if k in ARM_ORDER else 50):
            rs = [r for r in runs if r.arm == kind]
            ev: dict[str, Evidence] = {}
            for fld, _, _ in ROWS:
                e = Evidence()
                for r in rs:
                    if fld in r.evidence:
                        e.add(r.evidence[fld])
                ev[fld] = e
            arms[kind] = ArmGroup(arm=kind, seeds=[r.seed for r in rs], race_ids=[r.race_id for r in rs],
                                  stats={fld: _stat([getattr(r, fld) for r in rs]) for fld, _, _ in ROWS},
                                  evidence=ev, trap_runs=sum(r.trap_runs for r in rs))
        fp = ar.fairness_fingerprint(members[0][0])
        unused: dict[str, list[str]] = {}
        engine: dict[str, EngineStats] = {}
        adapter: list[tuple[int, int]] = []
        for rows, _ in members:
            rfp = ar.fairness_fingerprint(rows)
            for k, v in ((rfp or {}).get("unused_prompts") or {}).items():
                unused.setdefault(canon_arm(k), [])
                unused[canon_arm(k)] = list(dict.fromkeys(unused[canon_arm(k)] + v))
            for k, st in engine_stats(rows, rfp).items():
                engine.setdefault(k, EngineStats()).add(st)
            ev = adapter_evidence(rows)
            if ev is not None:
                adapter.append(ev)
        defaults = _default_features()
        f0 = members[0][0]["races"][0].get("fairness")
        has_engine = isinstance(f0, Mapping) and "engine" in f0  # races from before the engine: no defaults
        for k, st in engine.items():
            if has_engine and k in defaults and sorted(st.features) != defaults[k]:
                warnings.append(f"column '{column(k)}' ran engine features {st.features or 'none'}, not the "
                                f"default {defaults[k] or 'none'}: the column label describes the default engine")
            est = {s: n for s, n in st.cost_sources.items() if s.startswith("estimate")}
            if est:
                warnings.append(f"{column(k)}: LLM $ of {sum(est.values())} call(s) is an estimate "
                                f"({', '.join(f'cost_source={s}' for s in est)}), not a provider bill")
        g = Group(
            target=target, direction=direction, goal=goal, goal_source=goal_src, budget_s=budget_s,
            gpus_per_arm=gpus, run_budget_s=run_budget, runner=str(members[0][0]["races"][0].get("runner") or ""),
            names=sorted({str(r["races"][0].get("name") or "") for r, _ in members}),
            race_ids=[r["races"][0]["id"] for r, _ in members],
            seeds=[int(r["races"][0].get("seed") or 0) for r, _ in members],
            arms=arms, start_quality=statistics.fmean(start_vals) if start_vals else None,
            best_known=best_known.get(target), blockers=list(dict.fromkeys(blockers)), warnings=warnings,
            framings=[], runs=runs, fingerprint_id=fp_id, fingerprint=fp, unused_prompts=unused,
            engine=engine, adapter=adapter,
        )
        g.framings = framings_for(g, min_seeds=min_seeds, pairs=pairs)
        groups.append(g)
    return groups, notes


# --------------------------------------------------------------------------- framings


def _pct(x: float) -> str:
    return f"{abs(x) * 100:.0f}%"


def _times(x: float) -> str:
    return f"{x:.1f}×"


def _usd(x: float | None) -> str:
    return "n/a" if x is None else (f"${x:.3f}" if abs(x) < 1 else f"${x:.2f}")


def _mins(x: float | None) -> str:
    return "n/a" if x is None else f"{x / 60:.1f} min"


def _q(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4g}"


def _n_ok(s: Any, who: str, min_seeds: int, reach: bool, what: str) -> list[str]:
    if reach and s.n < s.n_total:
        return [f"{who} {what} in {s.n}/{s.n_total} seeds (needs every seed)"]
    if s.n < min_seeds:
        return [f"{who}: n={s.n} seed(s) with a value (needs >= {min_seeds})"]
    return []


def _paired(g: Group, arm: str, base: str, fld: str, fn: Any) -> list[float]:
    """Per-seed values of the framing formula on races where both arms ran (shown in the JSON)."""
    by_race: dict[str, dict[str, RaceArm]] = {}
    for r in g.runs:
        by_race.setdefault(r.race_id, {})[r.arm] = r
    out: list[float] = []
    for rs in by_race.values():
        if arm in rs and base in rs:
            v = fn(getattr(rs[arm], fld), getattr(rs[base], fld))
            if v is not None and math.isfinite(v):
                out.append(v)
    return out


def _cheaper(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None or b <= 0 else 1.0 - a / b


def _ratio(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None or b <= 0 else a / b


def _framed_ratio(target: str, end: float | None, start: float | None) -> float | None:
    """'cost' targets: start ÷ end (N× lower). 'accuracy': (err_start - err_end) ÷ err_start."""
    fe, fs = framed_quantity(target, end), framed_quantity(target, start)
    if fe is None or fs is None:
        return None
    if QUALITY_FRAMING[target][0] == "cost":
        return None if fe <= 0 or fs <= 0 else fs / fe
    return None if fs <= 0 else (fs - fe) / fs


def _quality_phrase(target: str, value: float) -> str:
    kind, unit = QUALITY_FRAMING[target]
    if kind == "cost":
        lower = 1.0 - 1.0 / value  # 1 - end/start, from N = start/end
        if value >= 1.0:
            return f"{_times(value)} lower {unit} ({_pct(lower)} lower)"
        return f"{_times(value)}: {_pct(lower)} HIGHER {unit} (worse)"
    if value >= 0:
        return f"{_pct(value)} fewer errors"
    return f"{_pct(value)} MORE errors (worse)"


def framings_for(g: Group, *, min_seeds: int = MIN_SEEDS,
                 pairs: Sequence[tuple[str, str]] = DEFAULT_PAIRS) -> list[Framing]:
    out: list[Framing] = []
    same = f"on the same {g.gpus_per_arm} slots, {budget_text(g.budget_s)} budget and seeds"
    virt = " GPU $ is virtual (surrogate runner)." if g.runner == "surrogate" else ""
    goal_txt = f"held-out score {'≤' if g.direction == 'min' else '≥'} {g.goal:g}"

    def make(fid: str, title: str, arm: str, base: str, formula: str, value: float | None, sentence: str,
             inputs: dict[str, Any], reasons: list[str], per_seed: list[float] | None = None) -> None:
        gated = g.gate(arm) + (g.gate(base) if base != arm else [])
        reasons = list(dict.fromkeys(gated + (["setup not headline-eligible (see Status)"] if g.blockers else [])
                                     + reasons))
        if value is None and not reasons:
            reasons.append("formula undefined for these values (zero or negative denominator)")
        ok = not reasons
        if ok:
            text = sentence
        elif ADAPTER_REASON in reasons:  # D41 #3: every Mubit framing needs evidence an adapter served
            rest = [r for r in reasons if r != ADAPTER_REASON]
            text = (f"{title}, {noun(arm)} vs {noun(base)}: not measured: {ADAPTER_REASON}"
                    + (f" (also: {'; '.join(rest)})" if rest else ""))
        else:
            text = f"{title}, {noun(arm)} vs {noun(base)}: not measured ({'; '.join(reasons)})"
        out.append(Framing(id=fid, title=title, arm=arm, baseline=base, formula=formula, measured=ok,
                           value=value if ok else None, sentence=text, inputs=inputs, reasons=reasons,
                           per_seed=per_seed or []))

    def seeds_reasons(*kinds: str) -> list[str]:
        r: list[str] = []
        for k in kinds:
            if k not in g.arms:
                r.append(f"no {noun(k)} arm in these races")
            elif g.arms[k].n_seeds < min_seeds:
                r.append(f"{noun(k)}: {g.arms[k].n_seeds} distinct seed(s) (needs >= {min_seeds})")
        return r

    qf = QUALITY_FRAMING.get(g.target)
    # ---- the research agent improves the model: each arm vs the starting code
    for kind, a in g.arms.items():
        if kind == "mock":
            continue
        end = a.stats["best_quality"]
        title = "Model quality vs the starting code"
        reasons = seeds_reasons(kind) or _n_ok(end, noun(kind), min_seeds, False, "")
        if qf is None:
            reasons.append(f"no 6b quality framing defined for target {g.target!r}")
            val = None
        elif g.start_quality is None:
            reasons.append("no starting-config run with a held-out score")
            val = None
        else:
            val = _framed_ratio(g.target, end.mean, g.start_quality)
        formula = ("start ÷ end (N× lower), 1 − end/start (% lower)" if qf and qf[0] == "cost"
                   else "(err_start − err_end) ÷ err_start")
        sentence = (f"{subject(kind)} improved the model to {_quality_phrase(g.target, val)} than the starting code "
                    f"(held-out {_q(end.mean)} vs {_q(g.start_quality)}, mean of {end.n} seeds, {same})."
                    if val is not None and qf else "")
        make(f"quality_vs_start:{kind}", title, kind, "start", formula, val, sentence,
             {"end_mean": end.mean, "end_std": end.std, "n": end.n, "start": g.start_quality}, reasons)
        if g.best_known is not None:
            gap = None
            if end.mean is not None and g.start_quality is not None and g.best_known != g.start_quality:
                gap = (end.mean - g.start_quality) / (g.best_known - g.start_quality)
            make(f"gap_closed:{kind}", "Progress toward the best known result", kind, "start",
                 "(end − start) ÷ (best_known − start)", gap,
                 f"{subject(kind)} closed {gap * 100:.0f}% of the gap from the starting code ({_q(g.start_quality)}) "
                 f"to the best known result ({_q(g.best_known)}), mean of {end.n} seeds, {same}." if gap is not None else "",
                 {"end_mean": end.mean, "start": g.start_quality, "best_known": g.best_known, "n": end.n},
                 seeds_reasons(kind) or _n_ok(end, noun(kind), min_seeds, False, ""))

    # ---- head-to-head pairs
    for arm, base in pairs:
        if arm not in g.arms or base not in g.arms:
            missing = [k for k in (arm, base) if k not in g.arms]
            if "opus_bm" in missing:  # optional column (D36): only listed when the race has it
                continue
            make(f"pair:{arm}:{base}", "Head-to-head", arm, base, "-", None, "", {},
                 [f"no {noun(k)} arm in these races" for k in missing])
            continue
        A, B = g.arms[arm], g.arms[base]
        sr = seeds_reasons(arm, base)
        vs = f"than {noun(base)} {same}"
        vs_pos = f"than {possessive(noun(base))}, {same}"

        # model quality at the same time budget, and at the Opus arm's spend (same money)
        for fld, fid, title, lead in (
            ("best_quality", "quality_same_budget", "Model quality @ same budget", ""),
            ("quality_at_opus_spend", "quality_same_money", "Model quality @ the Opus agent's spend", "same_money"),
        ):
            if lead == "same_money" and "opus" not in g.arms:
                continue
            sa, sb = A.stats[fld], B.stats[fld]
            reasons = sr or (_n_ok(sa, noun(arm), min_seeds, False, "") + _n_ok(sb, noun(base), min_seeds, False, ""))
            if qf is None:
                reasons.append(f"no 6b quality framing defined for target {g.target!r}")
            val = _framed_ratio(g.target, sa.mean, sb.mean) if qf else None
            spend = g.arms["opus"].stats["total_cost_usd"].mean if "opus" in g.arms else None
            pre = f"At the Opus 5.5 agent's total spend ({_usd(spend)} mean per race), " if lead else ""
            who = possessive(subject(arm) if not lead else noun(arm))
            sentence = (f"{pre}{who} best model has {_quality_phrase(g.target, val)} {vs_pos} "
                        f"(held-out {_q(sa.mean)} vs {_q(sb.mean)}, mean of {min(sa.n, sb.n)} seeds).{virt}"
                        if val is not None and qf else "")
            make(f"{fid}:{arm}:{base}", title, arm, base,
                 "baseline ÷ arm on the framed quantity (N× lower), 1 − arm/baseline" if qf and qf[0] == "cost"
                 else "(err_baseline − err_arm) ÷ err_baseline",
                 val, sentence, {"arm_mean": sa.mean, "arm_std": sa.std, "arm_n": sa.n, "base_mean": sb.mean,
                                 "base_std": sb.std, "base_n": sb.n, "opus_spend_usd": spend if lead else None},
                 reasons, _paired(g, arm, base, fld, lambda x, y: _framed_ratio(g.target, x, y)) if qf else [])

        # cost and time to reach the goal, and to reach the Opus arm's final quality (same quality)
        for cfld, tfld, fid, what, reached in (
            ("cost_to_goal_usd", "time_to_goal_s", "goal", f"the goal ({goal_txt})", "reached the goal"),
            ("cost_to_opus_quality_usd", "time_to_opus_quality_s", "opus_quality",
             "the Opus 5.5 agent's final model quality", "reached the Opus agent's final quality"),
        ):
            if fid == "opus_quality" and "opus" not in g.arms:
                continue
            if fid == "opus_quality" and arm == "opus":
                continue
            for fld, kind_ in ((cfld, "cost"), (tfld, "time")):
                sa, sb = A.stats[fld], B.stats[fld]
                reasons = sr or (_n_ok(sa, noun(arm), min_seeds, True, reached)
                                 + _n_ok(sb, noun(base), min_seeds, True, reached))
                val = _cheaper(sa.mean, sb.mean)
                if kind_ == "cost":
                    word = "cheaper" if (val or 0) >= 0 else "MORE EXPENSIVE"
                    nums = f"{_usd(sa.mean)} vs {_usd(sb.mean)}"
                    title, formula = f"Cost to reach {what}", "1 − arm$ ÷ baseline$"
                else:
                    word = "faster" if (val or 0) >= 0 else "SLOWER"
                    nums = f"{_mins(sa.mean)} vs {_mins(sb.mean)}"
                    title, formula = f"Time to reach {what}", "1 − arm_t ÷ baseline_t"
                base_txt = "did" if fid == "opus_quality" and base == "opus" else ""
                sentence = (f"{subject(arm)} reached {what} {_pct(val)} {word} than {noun(base)}{' ' + base_txt if base_txt else ''} "
                            f"{same} ({nums}, mean of {sa.n} seeds).{virt if kind_ == 'cost' else ''}"
                            if val is not None else "")
                make(f"{kind_}_to_{fid}:{arm}:{base}", title, arm, base, formula, val, sentence,
                     {"arm_mean": sa.mean, "arm_std": sa.std, "arm_n": sa.n, "arm_n_total": sa.n_total,
                      "base_mean": sb.mean, "base_std": sb.std, "base_n": sb.n, "base_n_total": sb.n_total},
                     reasons, _paired(g, arm, base, fld, _cheaper))

        # throughput
        sa, sb = A.stats["experiments_per_usd"], B.stats["experiments_per_usd"]
        reasons = sr or (_n_ok(sa, noun(arm), min_seeds, False, "") + _n_ok(sb, noun(base), min_seeds, False, ""))
        val = _ratio(sa.mean, sb.mean)
        sentence = ""
        if val is not None:
            how = (f"{_times(val)} more experiments per dollar {vs}" if val >= 1 else
                   f"only {_times(val)} as many experiments per dollar (FEWER) as {noun(base)} {same}")
            sentence = (f"{subject(arm)} ran {how} "
                        f"({sa.mean:.1f} vs {sb.mean:.1f} per $, mean of {min(sa.n, sb.n)} seeds).{virt}")
        make(f"throughput:{arm}:{base}", "Experiments per dollar", arm, base,
             "(arm experiments/$) ÷ (baseline experiments/$)", val, sentence,
             {"arm_mean": sa.mean, "arm_std": sa.std, "arm_n": sa.n, "base_mean": sb.mean, "base_std": sb.std,
              "base_n": sb.n}, reasons, _paired(g, arm, base, "experiments_per_usd", _ratio))

    out.append(Framing(
        id="fake_improvements", title="Fake improvements let through", arm="mubit", baseline="opus",
        formula="judge v2 vs Opus (and v1) on the 50-case trap set, test split", measured=False, value=None,
        sentence="Fake improvements let through: not measured here (trap-set result, scripts/eval_trap_set.py; "
                 "the in-race trap counts are in the table)."
                 + (f" The Mubit adapter's version is not measured: {ADAPTER_REASON}." if not g.adapter_ok else ""),
        inputs={}, reasons=["measured on the trap set, not in races"] + g.gate("mubit")))
    return out


# --------------------------------------------------------------------------- markdown

TARGET_LABEL = {**ar.TARGET_LABEL, "lcbench": "Tabular classifier (LCBench)"}


def _cell(s: Any, kind: str) -> str:
    if s.n_total == 0:
        return "–"
    if s.n == 0:
        return f"didn't reach it (n=0/{s.n_total})" if kind.endswith("reach") else f"n/a (n=0/{s.n_total})"
    sd = s.std or 0.0
    if kind.startswith("usd"):
        body = f"{_usd(s.mean)} ± {_usd(sd).lstrip('$')}"
    elif kind.startswith("min"):
        body = f"{s.mean / 60:.1f} ± {sd / 60:.1f} min"
    elif kind == "pct":
        body = f"{s.mean:.1f} ± {sd:.1f}%"
    elif kind == "q":
        body = f"{s.mean:.4g} ± {sd:.2g}"
    elif kind == "per_usd":
        body = f"{s.mean:.1f} ± {sd:.1f}"
    else:
        body = f"{s.mean:.1f} ± {sd:.1f}"
    if s.n == 1:  # one value: no spread to show
        body = body.split(" ± ")[0] + (" min" if kind.startswith("min") else "%" if kind == "pct" else "")
    n = f"n={s.n}" if s.n == s.n_total else f"n={s.n}/{s.n_total}" + (" reached" if kind.endswith("reach") else "")
    return f"{body} ({n})"


def _ev_cell(e: Evidence, is_llm: bool) -> str:
    llm = f"{e.request_ids}/{e.llm_calls} Mubit req IDs" if is_llm else "no LLM"
    runs = f"{e.modal_ids}/{e.runs} Modal IDs"
    if e.other_run_ids:
        runs += f" (+{e.other_run_ids} non-Modal run IDs)"
    return f"{llm} · {runs} · {e.races} race(s)"


def _is_llm_arm(g: Group, kind: str) -> bool:
    return kind != "optuna" or g.arms[kind].evidence["total_cost_usd"].llm_calls > 0


def _features_cell(g: Group, kind: str) -> str:
    fpa = _fp_arm(g.fingerprint, kind)
    if fpa is None:
        return "unknown (no manifest)"
    feats = fpa.get("engine_features") or []
    return " + ".join(f.replace("_", " ") for f in feats) if feats else "none"


def _llm_cell(g: Group, kind: str) -> str:
    fpa = _fp_arm(g.fingerprint, kind)
    if fpa is None:
        return "unknown (no manifest)"
    pol = fpa.get("llm_policy")
    if not pol:
        return "no LLM"
    prompts = ", ".join(f"{s.split('_')[0]} {v.split(':')[0]}" for s, v in (pol.get("prompts") or {}).items())
    return f"{fpa.get('model')} · record {fpa.get('record_mode')} · {prompts or 'no calls'}"


def _engine_lines(g: Group, kinds: Sequence[str]) -> list[str]:
    """Per column: early stopping as slot time freed (never $, D41 #5) and the judge policy's
    own counts, a degenerate rule (sigma = 0) marked and kept out of real-gain stats."""
    rows: list[tuple[str, list[str]]] = []
    stop_cells: list[str] = []
    judge_cells: list[str] = []
    any_stop = any_judge = False
    for k in kinds:
        e = g.engine.get(k) or EngineStats()
        if "median_stop" in e.features:
            any_stop = True
            c = (f"{e.early_stopped} of {e.runs} runs stopped · {e.slot_s_freed / 60:.1f} slot-min freed "
                 "(engine estimate)")
            if e.early_stop_too_late:
                c += f" · {e.early_stop_too_late} finished before the cancel landed"
            if e.stops_unconfirmed:
                c += f" · ⚠️ {e.stops_unconfirmed} cancels not confirmed by the runner (D41 #1)"
            stop_cells.append(c)
        else:
            stop_cells.append("off")
        if "judge_policy" in e.features:
            any_judge = True
            base = f"{e.judge_claims} claims · LLM said invalid {e.judge_llm_invalid}"
            if e.judge_no_reply:
                base += f" · no LLM reply {e.judge_no_reply}"
            if e.degenerate:
                judge_cells.append(f"**DEGENERATE** ({e.degenerate_why}) · {base} · rule counts excluded "
                                   "from real-gain stats")
            else:
                judge_cells.append(f"{base} · rule: real {e.judge_rule_real}, noise {e.judge_rule_noise} · final "
                                   + ", ".join(f"{v} {n}" for v, n in sorted(e.judge_final.items())))
        else:
            judge_cells.append("off")
    if not (any_stop or any_judge):
        return []
    if any_stop:
        rows.append(("Early stopping (median rule)", stop_cells))
    if any_judge:
        rows.append(("Judge policy (LLM for leakage + seed-spread rule)", judge_cells))
    out = ["**Busy Machines engine** (slot time freed is not converted to $ and is in no $ framing):\n",
           "| Engine | " + " | ".join(column(k) for k in kinds) + " |", "|---|" + "---|" * len(kinds)]
    out += [f"| {label} | " + " | ".join(cells) + " |" for label, cells in rows]
    out.append("")
    if any_judge:
        out.append(f"- Judge policy: {JUDGE_POLICY_NOTE}.")
        out.append("")
    return out


def render_markdown(groups: list[Group], ex: Any, notes: Sequence[str] = ()) -> str:
    out: list[str] = []
    if not groups:
        out.append("_No races to report._\n")
    for g in groups:
        label = TARGET_LABEL.get(g.target, g.target)
        n_seeds = len(set(g.seeds))
        out.append(f"### Headline: {label} (`{g.target}`), {n_seeds} seed(s)\n")
        out.append(f"- Races: {'; '.join(g.names)} · ids {', '.join(f'`{r[:8]}`' for r in g.race_ids)} · "
                   f"seeds {', '.join(str(s) for s in g.seeds)}.")
        out.append(f"- Setup: {g.gpus_per_arm} slots per arm, {g.budget_s / 60:g}-min budget"
                   + (f", {g.run_budget_s} s runs" if g.run_budget_s else "") + f", runner {g.runner}.")
        out.append(f"- Goal: held-out score {'≤' if g.direction == 'min' else '≥'} {g.goal:g} ({g.goal_source}).")
        out.append(f"- Starting code held-out: {_q(g.start_quality)} (mean of every arm's starting-config runs)"
                   + (f"; best known {_q(g.best_known)} (--best-known)." if g.best_known is not None else "."))
        if g.headline_ok:
            out.append("- Status: **eligible for the headline** (finished, benchmark mode, frozen goal, equal slots, "
                       "one fairness fingerprint, no corrections); each framing still needs its own n and its arm gates.")
        else:
            out.append("- Status: **NOT A HEADLINE RESULT**: " + "; ".join(g.blockers) + ".")
        if g.runner == "surrogate":
            out.append("- GPU $ is virtual (surrogate runner): real spend is the LLM only.")
        out.append(f"- Fairness fingerprint: `{g.fingerprint_id}` (arm set, per-arm engine features, prompt versions "
                   "called, record modes, race settings, Mubit routing; only races with the same fingerprint are pooled)."
                   if g.fingerprint_id else "- Fairness fingerprint: none (no fairness manifest).")
        if g.fingerprint and g.fingerprint.get("mubit_projects"):
            out.append(f"- Mubit project routing: `{json.dumps(g.fingerprint['mubit_projects'], sort_keys=True)}`.")
        for k, v in g.unused_prompts.items():
            out.append(f"- {column(k)}: prompts listed but never called in these races (not in the fingerprint): "
                       + ", ".join(v) + ".")
        if "mubit" in g.arms:
            served, calls = sum(s for s, _ in g.adapter), sum(n for _, n in g.adapter)
            out.append(f"- Mubit adapter evidence: {served} of {calls} Mubit-arm calls carry evidence an adapter "
                       "answered" + ("" if g.adapter_ok else f": every Mubit framing is not measured ({ADAPTER_REASON})")
                       + ".")
        for w in g.warnings:
            if "(unfair)" not in w:
                out.append(f"- ⚠️ {w}")
        out.append("- Cells: mean ± std over seeds (n = seeds with a value). Same quality / same money are "
                   "paired per race with that race's Opus arm.")
        out.append("")
        kinds = list(g.arms)
        out.append("| | " + " | ".join(column(k) for k in kinds) + " |")
        out.append("|---|" + "---|" * len(kinds))
        out.append("| Engine features (Busy Machines) | " + " | ".join(_features_cell(g, k) for k in kinds) + " |")
        out.append("| LLM setup (model · record · prompts called) | " + " | ".join(_llm_cell(g, k) for k in kinds) + " |")
        for fld, row_label, kind in ROWS:
            if fld in ("cost_to_opus_quality_usd", "time_to_opus_quality_s", "quality_at_opus_spend") and "opus" not in g.arms:
                continue
            cells = []
            for k in kinds:
                s = g.arms[k].stats[fld]
                c = _cell(s, kind)
                if fld == "fake_best_accepted" and s.n:
                    c = f"{int(sum(r.fake_best_accepted for r in g.runs if r.arm == k))} of {g.arms[k].trap_runs} trap runs"
                cells.append(c)
            out.append(f"| {row_label} | " + " | ".join(cells) + " |")
        out.append("")
        out.extend(_engine_lines(g, kinds))
        out.append("**Headline framings** (benchmark.md §6b, fixed formulas; worse results printed as worse):\n")
        for f in g.framings:
            out.append(f"- {f.sentence}")
        out.append("")
        out.append("**Evidence** behind each number (summed over races; LLM calls completed and runs started "
                   "inside the window the number uses):\n")
        out.append("| Number | " + " | ".join(column(k) for k in kinds) + " |")
        out.append("|---|" + "---|" * len(kinds))
        for fld, row_label, _ in ROWS:
            if fld in ("cost_to_opus_quality_usd", "time_to_opus_quality_s", "quality_at_opus_spend") and "opus" not in g.arms:
                continue
            out.append(f"| {row_label} | " + " | ".join(
                _ev_cell(g.arms[k].evidence[fld], _is_llm_arm(g, k)) for k in kinds) + " |")
        out.append("")
    if notes:
        out.append("#### Notes\n")
        out.extend(f"- {n}" for n in notes)
        out.append("")
    out.append("#### Excluded\n")
    if not (ex.races or ex.llm_calls or ex.experiments or ex.duplicates):
        out.append("- Nothing excluded.")
    for rid, why in ex.races:
        out.append(f"- race `{rid}`: {why}")
    for why, n in ex.llm_calls.items():
        out.append(f"- {n} LLM call row(s): {why}")
    for why, n in ex.experiments.items():
        out.append(f"- {n} experiment row(s): {why}")
    for rid in ex.duplicates:
        out.append(f"- race `{rid}`: duplicate (counted once)")
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- JSON


def _stat_json(s: Any) -> dict[str, Any]:
    return {"mean": s.mean, "std": s.std, "n": s.n, "n_total": s.n_total}


def to_json(groups: list[Group], ex: Any, notes: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "source": "scripts/headline.py",
        "min_seeds": MIN_SEEDS,
        "groups": [{
            "target": g.target, "target_label": TARGET_LABEL.get(g.target, g.target), "direction": g.direction,
            "goal": g.goal, "goal_source": g.goal_source, "budget_s": g.budget_s, "gpus_per_arm": g.gpus_per_arm,
            "run_budget_s": g.run_budget_s, "runner": g.runner, "race_names": g.names, "race_ids": g.race_ids,
            "seeds": g.seeds, "start_quality": g.start_quality, "best_known": g.best_known,
            "headline_ok": g.headline_ok, "blockers": g.blockers, "warnings": g.warnings,
            "columns": [{"arm": k, "label": column(k)} for k in g.arms],
            "arms": {k: {
                "label": column(k), "seeds": a.seeds, "race_ids": a.race_ids, "trap_runs": a.trap_runs,
                "stats": {fld: _stat_json(a.stats[fld]) for fld, _, _ in ROWS},
                "evidence": {fld: asdict(a.evidence[fld]) for fld, _, _ in ROWS},
            } for k, a in g.arms.items()},
            "fingerprint_id": g.fingerprint_id, "fingerprint": g.fingerprint, "unused_prompts": g.unused_prompts,
            "engine": {k: {**asdict(e), "slot_s_freed_note": "engine estimate; never converted to $",
                           "real_gain_stats_usable": not e.degenerate} for k, e in g.engine.items()},
            "adapter_evidence": {"per_race": [{"served": s, "calls": n} for s, n in g.adapter],
                                 "ok": g.adapter_ok, "reason": None if g.adapter_ok else ADAPTER_REASON},
            "framings": [asdict(f) for f in g.framings],
            "per_race": [{k: v for k, v in asdict(r).items() if k != "evidence"} for r in g.runs],
        } for g in groups],
        "excluded": asdict(ex),
        "notes": list(notes),
    }


# --------------------------------------------------------------------------- selection + CLI


def refuse_ablation_preset(preset: str) -> None:
    """An ablation preset never feeds the headline (SystemExit, before anything is read)."""
    from backend.benchmark.configs import PRESETS

    base = PRESETS.get(preset)
    if base is not None and (base.mode == "ablation" or any(canon_arm(a.value) in ABLATION_ARMS for a in base.arms)):
        raise SystemExit(f"refused: preset {preset!r} is an {ABLATION_REASON}; "
                         f"read it with scripts/analyze_races.py, not headline.py")


def select_preset(races: list[Rows], preset: str) -> list[Rows]:
    """Races run from ``preset`` by scripts/bench.py: named "<preset name> seed N[ tags]"."""
    from backend.benchmark.configs import PRESETS

    if preset not in PRESETS:
        raise SystemExit(f"unknown preset {preset!r}; known: {', '.join(sorted(PRESETS))}")
    base = PRESETS[preset]
    refuse_ablation_preset(preset)
    prefix = f"{base.name} seed "
    return [r for r in races if str(r["races"][0].get("name") or "").startswith(prefix)
            and r["races"][0].get("target") == base.target]


def run(*, db: str | Path | None = None, race_ids: Sequence[str] | None = None, preset: str | None = None,
        jsonl: Sequence[str | Path] = (), best_known: Mapping[str, float] | None = None, min_seeds: int = MIN_SEEDS,
        pairs: Sequence[tuple[str, str]] = DEFAULT_PAIRS) -> tuple[list[Group], Any, list[str]]:
    if preset:
        refuse_ablation_preset(preset)
    races: list[Rows] = []
    if db is not None:
        races.extend(ar.load_db(db, list(race_ids) if race_ids else None))  # opened with mode=ro
    if jsonl:
        loaded = ar.load_jsonl(jsonl)
        races.extend([r for r in loaded if not race_ids or r["races"][0]["id"] in race_ids])
    if preset:
        races = select_preset(races, preset)
    refused = [r for r in races if is_ablation_race(r)]
    races = [r for r in races if not is_ablation_race(r)]
    # Keep dev and unfinished races so they show up stamped "not measured" instead of vanishing;
    # probe/collection races and orphan rows are still removed (analyze_races.screen).
    kept, ex = ar.screen(races, allow_dev=True, include_unfinished=True)
    groups, notes = build_groups(kept, best_known=best_known, min_seeds=min_seeds, pairs=pairs)
    notes = list(notes) + [f"refused race {str(r['races'][0]['id'])[:8]}: {ABLATION_REASON}" for r in refused]
    return groups, ex, notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="night/bench headline table and 6b framings (see module docstring).")
    ap.add_argument("--db", help=f"SQLite DB (read-only; default {DEFAULT_DB.relative_to(REPO_ROOT)} when no --jsonl)")
    ap.add_argument("--race", action="append", help="race id (repeatable)")
    ap.add_argument("--preset", help="only races run from this bench.py preset (e.g. rl-wide-15m)")
    ap.add_argument("--jsonl", nargs="+", default=[], help="JSONL race exports")
    ap.add_argument("--best-known", action="append", help="TARGET=VALUE for the 'gap closed' framing")
    ap.add_argument("--min-seeds", type=int, default=MIN_SEEDS, help="seeds needed before a framing prints")
    ap.add_argument("--pair", action="append", help="ARM:BASELINE head-to-head pair (default: the D36 set)")
    ap.add_argument("--out", help="also write the markdown here")
    ap.add_argument("--json", help="also write every number here (for the slides)")
    args = ap.parse_args(argv)
    if args.min_seeds < MIN_SEEDS:
        ap.error(f"--min-seeds below {MIN_SEEDS} is not allowed (benchmark.md 6b: mean over >= 3 seeds)")
    db = args.db if args.db else (None if args.jsonl else str(DEFAULT_DB))
    pairs = DEFAULT_PAIRS
    if args.pair:
        pairs = tuple(tuple(canon_arm(x) for x in p.split(":", 1)) for p in args.pair)  # type: ignore[misc]
        if any(len(p) != 2 for p in pairs):
            ap.error("--pair expects ARM:BASELINE")
    groups, ex, notes = run(db=db, race_ids=args.race, preset=args.preset, jsonl=args.jsonl,
                            best_known=ar._kv(args.best_known), min_seeds=args.min_seeds, pairs=pairs)
    md = render_markdown(groups, ex, notes)
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    if args.json:
        data = to_json(groups, ex, notes)
        data["min_seeds"] = args.min_seeds
        Path(args.json).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
