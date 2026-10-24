"""Build the 'real gain or noise?' judge set (opportunity_benchmarks.md regime 1) from the
recorded Opus collection races, graded against many-seed re-run ground truth.

Source: the read-only collection DBs ``data/collection/collect_{rl,cifar}.db`` (R20). Every
**clean** (``is_trap = 0``) claimed new best that went to ``judge_improvement`` becomes a case,
plus a seeded sample of **trap** claims (planted contamination: ground truth ``fake``). Each
case keeps the previous best it was compared against (the last ``is_best`` run of the same race
processed before it; checked against ``summary.best_before``).

Phases (subcommands)::

    python -m uv run --extra ml python scripts/make_noise_set.py plan            # no Modal, no $
    python -m uv run --extra ml python scripts/make_noise_set.py rerun --yes --usd-cap 2.5
    python -m uv run --extra ml python scripts/make_noise_set.py build           # label + write set

1. ``rerun``: every distinct config among the clean claims (candidate and previous best) is
   re-run on the deployed Modal app ``busy-machines-runner`` (never deployed here) over
   ``--seeds`` (default 100..109, fresh seeds never used by the races), with the claim's own
   race ``run_budget_s`` (rl 30 s on CPU, cifar 45 s on L4, at most 3 concurrent L4). Results
   are appended to ``data/noise_set/reruns.jsonl`` (resumable: done (config, seed) pairs are
   skipped). A run launches only if spent + in-flight worst case stays under ``--usd-cap``.
   Collection configs predate the D29 research choices, so the new parameters are filled with
   their defaults, which are byte-identical in behaviour (D29).
2. ``build``: ground truth per clean claim, on the **held-out** score, direction-aware:
   ``real`` iff (mean_new - mean_prev) > 2 x pooled std (sqrt((s_new^2 + s_prev^2)/2)) AND a
   two-sided Welch t-test gives p < 0.05; otherwise ``noise``. Trap claims are ``fake``.
   Every statistic is stored on the case. Writes ``data/noise_set/noise_set_v1.jsonl`` and
   ``data/noise_set/summary.json``.

Judge user messages are rebuilt EXACTLY as the race loop built them
(``RepoPrompts.judge_user`` -> ``build_judge_user``): previous best score/config from the
previous best run, new score/config from the candidate, evidence = the candidate's
agent-visible run metrics (``scheduler.agent_visible_metrics``). The target spec block uses the
spec **as it was at collection time**: today's spec restricted to the parameters the collection
configs carry (D29 added parameters after the collection). The request text was not stored in
the DB, so ``eval_noise_set.py`` checks each rebuilt message against the recorded call's total
input-token count (system + spec + claim) instead.

Splits by target + race-seed group (no group is in both): dev = rl/seed-0, cifar/seed-0;
test = rl/seed-1, rl/seed-2, cifar/seed-1.

No LLM calls here. Modal only through ``workers.runners.modal_runner.ModalRunner``.
Measurements belong in docs/results_log.md (the lead logs them).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import hashlib
import json
import math
import random
import sqlite3
import statistics
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agent.prompts import CURRENT_VERSION, prompt_fingerprint  # noqa: E402
from backend.agent.prompts.build import CACHE_BREAK, JudgeClaim, build_judge_user, target_spec_block  # noqa: E402
from backend.core.types import StepType  # noqa: E402
from backend.scheduler import agent_visible_metrics  # noqa: E402
from workers.targets.base import TargetSpec, get_target  # noqa: E402

SET_VERSION = "noise_set_v1"
STEP = StepType.judge_improvement
COLLECTION_DIR = ROOT / "data" / "collection"
OUT_DIR = ROOT / "data" / "noise_set"
TARGETS: tuple[str, ...] = ("rl", "cifar")
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(100, 110))
# Split by (target, race seed): every case of a group lands in the same split.
SPLIT_OF_GROUP: dict[str, str] = {
    "rl/seed-0": "dev", "cifar/seed-0": "dev",
    "rl/seed-1": "test", "rl/seed-2": "test", "cifar/seed-1": "test",
    "lm/seed-0": "dev", "lm/seed-1": "test",
}
TRAPS_PER_GROUP = 8
SAMPLE_SEED = 20261024
# Ground-truth rule (fixed before any re-run): gain > K_SIGMA x pooled std AND Welch p < ALPHA.
K_SIGMA = 2.0
ALPHA = 0.05
MIN_SEEDS = 10
# Worst-case seconds billed per run beyond budget_s (container start + eval + grace).
OVERHEAD_S = {"CPU": 60.0, "L4": 90.0}


# --------------------------------------------------------------------------- DB


def _ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def config_key(target: str, config: dict[str, Any]) -> str:
    """Stable id of a (target, full config): sha256 of canonical JSON, 12 hex chars."""
    blob = json.dumps({"target": target, "config": config}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# Text the D29 edit changed in the spec block after the collection races (found by diffing
# today's spec block against data/trap_set/archive/trap_set_v1.pre_d29.jsonl, which was
# rendered with the collection-time spec; ``check_against_archive`` re-verifies it).
PRE_D29_TASK: dict[str, str] = {
    "cifar": ("CIFAR-10 image classification speedrun: a small airbench-style CNN trained from scratch with "
              "Nesterov SGD under a fixed wall-clock budget on one GPU. The goal is the highest test accuracy "
              "within the budget; runs that exceed the budget are cut short and evaluated as-is."),
}
PRE_D29_PARAM_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "rl": {"hidden_size": "width of the two hidden layers (actor and critic)"},
    "cifar": {"momentum": "Nesterov SGD momentum."},
}
ARCHIVE_PRE_D29 = ROOT / "data" / "trap_set" / "archive" / "trap_set_v1.pre_d29.jsonl"


def collection_spec(spec: TargetSpec, keys: Iterable[str]) -> TargetSpec:
    """The spec as the collection races saw it: today's spec restricted to ``keys`` (the
    parameters the collection configs carry), in today's order, with the pre-D29 texts."""
    keys = set(keys)
    missing = keys - set(spec.config_space)
    if missing:
        raise ValueError(f"{spec.name}: collection params not in today's spec: {sorted(missing)}")
    descs = PRE_D29_PARAM_DESCRIPTIONS.get(spec.name, {})
    space = {k: (p.model_copy(update={"description": descs[k]}) if k in descs else p)
             for k, p in spec.config_space.items() if k in keys}
    update: dict[str, Any] = {"config_space": space}
    if spec.name in PRE_D29_TASK:
        update["description"] = PRE_D29_TASK[spec.name]
    return spec.model_copy(update=update)


def check_against_archive(spec_c: TargetSpec, archive: Path = ARCHIVE_PRE_D29) -> bool | None:
    """True if ``spec_c``'s spec block equals the one in the pre-D29 trap set (rendered with
    the collection-time spec), False if not, None if the archive is missing."""
    if not archive.exists():
        return None
    for ln in archive.read_text(encoding="utf-8").splitlines():
        c = json.loads(ln) if ln.strip() else None
        if c and c.get("target") == spec_c.name:
            return c["user_message"].split(CACHE_BREAK, 1)[0] == target_spec_block(spec_c)
    return None


def full_config(spec: TargetSpec, config: dict[str, Any]) -> dict[str, Any]:
    """A collection config in today's config space (new D29 params at their defaults)."""
    return spec.validate_config({**spec.default_config(), **config})


def judge_message(spec_c: TargetSpec, cand: dict[str, Any], prev: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    """(JudgeClaim dict, user message) exactly as ``RepoPrompts.judge_user`` built it."""
    claim = JudgeClaim(
        prev_best_score=prev["score"] if prev else None,
        prev_best_config=prev["config"] if prev else None,
        new_score=float(cand["score"] or 0.0),
        new_config=cand["config"],
        evidence=dict(agent_visible_metrics(cand["metrics"])),
    )
    return claim.model_dump(mode="json"), build_judge_user(spec_c, claim)


def _judge_calls(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """experiment id -> the (last valid) recorded judge call for it."""
    out: dict[str, dict[str, Any]] = {}
    for r in conn.execute("SELECT * FROM llm_calls WHERE step_type = 'judge_improvement' ORDER BY created_at"):
        for eid in json.loads(r["experiment_ids_json"] or "[]"):
            if eid not in out or r["valid"]:
                out[eid] = dict(r)
    return out


def extract_claims(target: str, db_dir: Path = COLLECTION_DIR, traps_per_group: int = TRAPS_PER_GROUP,
                   sample_seed: int = SAMPLE_SEED) -> list[dict[str, Any]]:
    """Every clean judged claim plus a seeded sample of trap claims, with their previous best,
    the recorded judge call and the rebuilt judge user message."""
    path = db_dir / f"collect_{target}.db"
    conn = _ro(path)
    try:
        races = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM races")}
        exps = [dict(r) for r in conn.execute(
            "SELECT * FROM experiments WHERE status = 'succeeded' ORDER BY ended_at, id")]
        calls = _judge_calls(conn)
        all_keys: set[str] = set()
        for (cj,) in conn.execute("SELECT config_json FROM experiments"):
            all_keys |= set(json.loads(cj or "{}"))
    finally:
        conn.close()
    spec = get_target(target).spec
    spec_c = collection_spec(spec, all_keys)
    if check_against_archive(spec_c) is False:
        raise ValueError(f"{target}: rebuilt spec block differs from the pre-D29 archive; update PRE_D29_*")
    for e in exps:
        e["config"] = json.loads(e["config_json"] or "{}")
        e["metrics"] = json.loads(e["metrics_json"] or "{}")
        e["summary"] = json.loads(e["summary_json"] or "{}")

    claims: list[dict[str, Any]] = []
    for race_id, race in sorted(races.items(), key=lambda kv: kv[1]["seed"]):
        rcfg = json.loads(race["config_json"] or "{}")
        budget_s = int(rcfg.get("run_budget_s") or spec.budget_s)
        group = f"{target}/seed-{race['seed']}"
        best: dict[str, Any] | None = None
        for e in (x for x in exps if x["race_id"] == race_id):
            if e["judge_verdict"] is not None and not e["is_baseline"]:
                bb = e["summary"].get("best_before")
                if best is None or bb is None or abs(float(best["score"]) - float(bb)) > 1e-9:
                    raise ValueError(f"{e['id']}: previous best does not match summary.best_before")
                claim, user = judge_message(spec_c, e, best)
                call = calls.get(e["id"]) or {}
                claims.append({
                    "target": target, "group": group, "race_id": race_id, "race_seed": race["seed"],
                    "split": SPLIT_OF_GROUP[group], "kind": "trap" if e["is_trap"] else "clean",
                    "run_budget_s": budget_s, "hardware": spec.hardware,
                    "candidate": _brief(e, spec), "previous_best": _brief(best, spec),
                    "recorded_judge": {
                        "verdict": e["judge_verdict"], "reason": e["judge_reason"],
                        "llm_call_id": call.get("id"), "mubit_request_id": call.get("mubit_request_id"),
                        "session_id": call.get("session_id"), "latency_ms": call.get("latency_ms"),
                        "cost_usd": call.get("cost_usd"),
                        "total_input_tokens": (None if not call else int((call.get("input_tokens") or 0)
                                               + (call.get("cache_creation_tokens") or 0)
                                               + (call.get("cache_read_tokens") or 0))),
                    },
                    "claim": claim, "user_message": user,
                })
            if e["is_best"]:
                best = e
    clean = [c for c in claims if c["kind"] == "clean"]
    traps = [c for c in claims if c["kind"] == "trap"]
    rng = random.Random(f"{sample_seed}/{target}")
    picked: list[dict[str, Any]] = []
    for g in sorted({c["group"] for c in traps}):
        pool = sorted((c for c in traps if c["group"] == g), key=lambda c: c["candidate"]["experiment_id"])
        picked += rng.sample(pool, min(traps_per_group, len(pool)))
    keep = {id(c) for c in clean + picked}
    return [c for c in claims if id(c) in keep]


def _brief(e: dict[str, Any], spec: TargetSpec) -> dict[str, Any]:
    cfg_full = full_config(spec, e["config"])
    return {
        "experiment_id": e["id"], "config": e["config"], "config_key": config_key(spec.name, cfg_full),
        "score": e["score"], "heldout_score": e["metrics"].get("heldout_score"), "run_ref": e["run_ref"],
        "seed": e["seed"], "is_trap": bool(e["is_trap"]), "is_baseline": bool(e["is_baseline"]),
    }


def number_cases(claims: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {t: i for i, t in enumerate(TARGETS + ("lm",))}
    out = sorted(claims, key=lambda c: (order.get(c["target"], 9), c["race_seed"], c["kind"] != "clean",
                                        c["candidate"]["experiment_id"]))
    for i, c in enumerate(out, 1):
        c["id"] = f"ns_v1_{i:03d}"
    return out


# --------------------------------------------------------------------------- re-runs


def rerun_jobs(claims: Sequence[dict[str, Any]], seeds: Sequence[int]) -> list[dict[str, Any]]:
    """Distinct (target, config) groups to re-run (clean claims only), seed-major order so a
    $ cap cut leaves every config with the same number of seeds."""
    groups: dict[str, dict[str, Any]] = {}
    for c in claims:
        if c["kind"] != "clean":
            continue
        spec = get_target(c["target"]).spec
        for role in ("candidate", "previous_best"):
            b = c[role]
            groups.setdefault(b["config_key"], {
                "target": c["target"], "config_key": b["config_key"], "config": full_config(spec, b["config"]),
                "budget_s": c["run_budget_s"], "hardware": c["hardware"]})
    jobs = []
    for s in seeds:
        for g in sorted(groups.values(), key=lambda g: (g["hardware"] != "L4", g["target"], g["config_key"])):
            jobs.append(g | {"seed": int(s)})
    return jobs


def load_reruns(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def plan_cost(jobs: Sequence[dict[str, Any]], price: dict[str, float]) -> dict[str, Any]:
    by: dict[str, dict[str, float]] = {}
    for j in jobs:
        d = by.setdefault(j["target"], {"runs": 0, "worst_usd": 0.0, "typical_usd": 0.0})
        d["runs"] += 1
        d["worst_usd"] += (j["budget_s"] + OVERHEAD_S[j["hardware"]]) * price[j["hardware"]]
        d["typical_usd"] += (j["budget_s"] + 6.0) * price[j["hardware"]]
    return {"by_target": by, "worst_usd": sum(d["worst_usd"] for d in by.values()),
            "typical_usd": sum(d["typical_usd"] for d in by.values())}


class Budget:
    def __init__(self, cap: float, spent: float = 0.0) -> None:
        self.cap, self.spent, self.reserved = cap, spent, 0.0

    def try_reserve(self, usd: float) -> bool:
        if self.spent + self.reserved + usd > self.cap + 1e-12:
            return False
        self.reserved += usd
        return True

    def settle(self, reserved: float, actual: float) -> None:
        self.reserved -= reserved
        self.spent += actual


async def run_reruns(jobs: Sequence[dict[str, Any]], out_path: Path, usd_cap: float, max_l4: int,
                     max_cpu: int) -> None:
    from workers.runners.modal_runner import ModalRunner  # the only Modal entry point

    done = load_reruns(out_path)
    have = {(r["config_key"], r["seed"]) for r in done if r["status"] == "succeeded"}
    todo = [j for j in jobs if (j["config_key"], j["seed"]) not in have]
    runner = ModalRunner({"L4": max_l4, "CPU": max_cpu}, deployed=True)
    budget = Budget(usd_cap, spent=sum(r.get("est_usd") or 0.0 for r in done))
    sems = {"L4": asyncio.Semaphore(max_l4), "CPU": asyncio.Semaphore(max_cpu)}
    lock = asyncio.Lock()
    print(f"[noise] {len(todo)} runs to go ({len(jobs) - len(todo)} done), ${budget.spent:.3f} already spent, "
          f"cap ${usd_cap:.2f}", flush=True)

    async def one(j: dict[str, Any]) -> None:
        hw = j["hardware"]
        price = runner.price_per_second(hw)
        worst = (j["budget_s"] + OVERHEAD_S[hw]) * price
        async with sems[hw]:
            if not budget.try_reserve(worst):
                print(f"[noise] SKIP {j['target']} {j['config_key']} seed {j['seed']}: $ cap", flush=True)
                return
            t0 = time.time()
            row: dict[str, Any] = {k: j[k] for k in ("target", "config_key", "config", "budget_s", "hardware", "seed")}
            actual = 0.0
            try:
                h = await runner.launch(j["target"], j["config"], j["seed"], j["budget_s"], lambda _p: None)
                row["run_ref"] = h.run_ref
                r = await runner.wait(h)
                actual = r.duration_s * price
                m = dict(r.metrics)
                row.update(status=r.status.value, error=r.error, duration_s=r.duration_s, est_usd=actual,
                           heldout_score=m.get("heldout_score"), reported_score=m.get("reported_score", r.score),
                           is_trap=bool(r.is_trap), metrics=m)
            except Exception as exc:  # noqa: BLE001 (record and keep going)
                actual = (time.time() - t0) * price
                row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:400], est_usd=actual,
                           duration_s=time.time() - t0, heldout_score=None)
            finally:
                budget.settle(worst, actual)
            row["finished_at"] = _dt.datetime.now(_dt.UTC).isoformat()
            async with lock:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, default=str) + "\n")
            print(f"[noise] {j['target']} {j['config_key']} seed {j['seed']}: {row['status']} "
                  f"heldout={row.get('heldout_score')} {row.get('run_ref')} ~${actual:.4f} "
                  f"(spent ${budget.spent:.3f}) {row.get('error') or ''}", flush=True)

    await asyncio.gather(*(one(j) for j in todo))


# --------------------------------------------------------------------------- ground truth


def welch(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """(t, two-sided p) of Welch's t-test for mean(a) != mean(b)."""
    from scipy import stats as _st

    res = _st.ttest_ind(list(a), list(b), equal_var=False)
    return float(res.statistic), float(res.pvalue)


def ground_truth(new: Sequence[float], prev: Sequence[float], direction: str, *, k_sigma: float = K_SIGMA,
                 alpha: float = ALPHA, min_seeds: int = MIN_SEEDS) -> dict[str, Any]:
    """'real' iff the direction-aware gain of the means > k_sigma x pooled std AND Welch p < alpha.
    'noise' otherwise. 'insufficient' if either side has fewer than ``min_seeds`` runs."""
    new, prev = [float(x) for x in new], [float(x) for x in prev]
    out: dict[str, Any] = {"n_new": len(new), "n_prev": len(prev), "rule": f"gain > {k_sigma} x pooled std AND "
                           f"Welch two-sided p < {alpha} (held-out score, >= {min_seeds} seeds each)"}
    if len(new) < min_seeds or len(prev) < min_seeds:
        return out | {"label": "insufficient"}
    sign = 1.0 if direction == "max" else -1.0
    m_new, m_prev = statistics.fmean(new), statistics.fmean(prev)
    s_new, s_prev = statistics.stdev(new), statistics.stdev(prev)
    pooled = math.sqrt((s_new ** 2 + s_prev ** 2) / 2.0)
    gain = sign * (m_new - m_prev)
    if pooled == 0.0:
        t, p = (math.inf if gain > 0 else (-math.inf if gain < 0 else 0.0)), (0.0 if gain != 0 else 1.0)
    else:
        t, p = welch(new, prev)
    real = gain > k_sigma * pooled and p < alpha
    return out | {
        "label": "real" if real else "noise", "mean_new": m_new, "mean_prev": m_prev, "std_new": s_new,
        "std_prev": s_prev, "pooled_std": pooled, "gain": gain,
        "gain_in_pooled_std": (gain / pooled) if pooled else None, "welch_t": sign * t, "welch_p": p,
    }


def label_cases(cases: Sequence[dict[str, Any]], reruns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ok: dict[str, list[dict[str, Any]]] = {}
    for r in reruns:
        if r["status"] == "succeeded" and r.get("heldout_score") is not None and math.isfinite(r["heldout_score"]):
            ok.setdefault(r["config_key"], []).append(r)
    out = []
    for c in cases:
        c = dict(c)
        if c["kind"] == "trap":
            c["truth"] = "fake"
            c["ground_truth"] = {"label": "fake", "rule": "planted contamination trap (is_trap = 1 in the race DB)"}
        else:
            spec = get_target(c["target"]).spec
            rn = sorted(ok.get(c["candidate"]["config_key"], []), key=lambda r: r["seed"])
            rp = sorted(ok.get(c["previous_best"]["config_key"], []), key=lambda r: r["seed"])
            gt = ground_truth([r["heldout_score"] for r in rn], [r["heldout_score"] for r in rp], spec.direction)
            gt["new_runs"] = [{"seed": r["seed"], "heldout": r["heldout_score"], "run_ref": r.get("run_ref")} for r in rn]
            gt["prev_runs"] = [{"seed": r["seed"], "heldout": r["heldout_score"], "run_ref": r.get("run_ref")} for r in rp]
            if c["candidate"]["config_key"] == c["previous_best"]["config_key"]:
                gt["note"] = "same config as the previous best: any gain is seed noise"
            c["ground_truth"] = gt
            c["truth"] = gt["label"]
        out.append(c)
    return out


def summarize(cases: Sequence[dict[str, Any]], reruns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def count(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
        return dict(Counter(r["truth"] for r in rows))

    splits = {s: [c for c in cases if c["split"] == s] for s in ("dev", "test")}
    return {
        "set_version": SET_VERSION, "n_cases": len(cases), "truth": count(cases),
        "by_split": {s: count(rows) for s, rows in splits.items()},
        "by_group": {g: count(c for c in cases if c["group"] == g) for g in sorted({c["group"] for c in cases})},
        "reruns": {"runs": len(reruns), "succeeded": sum(r["status"] == "succeeded" for r in reruns),
                   "est_usd": round(sum(r.get("est_usd") or 0.0 for r in reruns), 4),
                   "by_hardware_usd": {hw: round(sum(r.get("est_usd") or 0.0 for r in reruns if r["hardware"] == hw), 4)
                                       for hw in sorted({r["hardware"] for r in reruns})}},
        "rule": {"k_sigma": K_SIGMA, "alpha": ALPHA, "min_seeds": MIN_SEEDS},
        "prompt": {"step": STEP.value, "version": CURRENT_VERSION[STEP], "fingerprint": prompt_fingerprint(STEP)},
    }


# --------------------------------------------------------------------------- CLI


def load_all_claims(targets: Sequence[str]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for t in targets:
        claims += extract_claims(t)
    return number_cases(claims)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["plan", "rerun", "build"])
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--usd-cap", type=float, default=2.5, help="GPU+CPU $ cap for re-runs (all re-runs so far count)")
    ap.add_argument("--max-l4", type=int, default=3)
    ap.add_argument("--max-cpu", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--yes", action="store_true", help="confirm Modal spend")
    ns = ap.parse_args(argv)
    targets = [t for t in ns.targets.split(",") if t]
    seeds = [int(s) for s in ns.seeds.split(",") if s.strip()]
    ns.out_dir.mkdir(parents=True, exist_ok=True)
    reruns_path = ns.out_dir / "reruns.jsonl"

    claims = load_all_claims(targets)
    jobs = rerun_jobs(claims, seeds)
    from workers.runners.base import DEFAULT_PRICE_PER_SECOND

    price = {"L4": DEFAULT_PRICE_PER_SECOND["L4"], "CPU": DEFAULT_PRICE_PER_SECOND["CPU"]}
    kinds = Counter((c["group"], c["kind"]) for c in claims)
    print(f"[noise] {len(claims)} cases: " + ", ".join(f"{g} {k} {n}" for (g, k), n in sorted(kinds.items())))
    print(f"[noise] {len(jobs) // max(1, len(seeds))} distinct configs x {len(seeds)} seeds = {len(jobs)} runs; "
          f"cost {json.dumps(plan_cost(jobs, price), default=lambda x: round(x, 4))}")
    if ns.cmd == "plan":
        return 0
    if ns.cmd == "rerun":
        if not ns.yes:
            print("refusing to spend Modal $ without --yes")
            return 2
        asyncio.run(run_reruns(jobs, reruns_path, ns.usd_cap, ns.max_l4, ns.max_cpu))
        return 0
    reruns = load_reruns(reruns_path)
    cases = label_cases(claims, reruns)
    with (ns.out_dir / f"{SET_VERSION}.jsonl").open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False, default=str) + "\n")
    summary = summarize(cases, reruns) | {"built_at": _dt.datetime.now(_dt.UTC).isoformat(),
                                          "seeds": seeds, "targets": targets}
    (ns.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
