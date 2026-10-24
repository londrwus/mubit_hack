"""D42 gate for early-stopping engine rules (offline, read-only; no LLM, no Modal).

For each target/regime it replays candidate percentile-stopping rules on recorded runs whose
full curves and final outcomes are known, then runs a small event simulation of a 30-slot race.

Regimes and data (all opened read-only, sqlite ``mode=ro``):
- ``rl30``    data/collection/collect_rl.db     (3 races, 8 slots, 30 s runs)
- ``rl15``    data/bench/bench.db race f2bee077 (R41), **opus arm only** (never early-stopped;
              the opus_bm / optuna runs there were stopped, so their curves are censored)
- ``cifar``   data/collection/collect_cifar.db  (2 races, 3 slots, 45 s runs)
- ``lm``      data/collection/collect_lm.db     (2 races, 2 slots, 60 s runs)
- ``lcbench`` data/bench/bench.db, the 3 LCBench surrogate races, opus + optuna arms (no stopping)

Rule (as ``backend.benchmark.engine_rules.MedianStopRule`` + ``backend.scheduler`` apply it, with
the threshold generalised from the median to a percentile):
- checkpoints ``c = f x run_budget_s`` on the MetricPoint.t axis (seconds since launch); a run is
  checked at ``c`` when its first curve point with ``t >= c`` arrives, using the last value at or
  before ``c`` (or the first point if none); each checkpoint is checked once (a skip for too few
  reference runs is final); the first stop wins;
- reference = runs of the same arm/race that **completed** before the decision (wall clock), were
  not judged invalid / trap, reached ``c``, and were not themselves stopped (closed loop: what the
  live scheduler sees);
- stop iff n_ref >= min_completed and the value is worse than the ``p``-th percentile of the
  reference (direction-aware: for 'min' targets the (1-p) percentile from below, i.e. the worst p);
  p = 0.5 is exactly the current median rule.
- a stop lands ``STOP_LATENCY_S`` later (R41 measured median cancel latency); if the run would have
  ended first it completes normally ("too late"), like the scheduler.

Metrics per rule: share of candidate runs stopped; share of the regime's top-10% runs stopped
(by final reported score, by held-out score, and by final curve value); slot time freed (% of the
candidates' total wall time). Event simulation: 30 slots, 900 s, launch cutoff, one sequential
planner with K = min(8, free slots) per call and latency Opus (2.48 s + 1.65 s x K, R36), 1 s, or
2 s; runs are bootstrap draws (common random numbers across rules) from the regime's unstopped
pool. Reports GPUs waiting %, completed runs, best reported score, held-out of that shipped run,
best held-out.

Gate (D42): wrong-stop of top-10% (by final score AND by held-out) <= 5%, freed >= 20%, and
GPUs waiting (Opus latency sim) not higher than no stopping (tolerance WAIT_TOL_PP).

Usage: python -m uv run --extra ml python scripts/gate_early_stop_rules.py [--seeds 20]
"""

from __future__ import annotations

import argparse
import bisect
import heapq
import json
import math
import random
import sqlite3
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "early_stop" / "rule_gate"

STOP_LATENCY_S = 1.87          # R41: median early-stop cancel latency (2,115 stops), Modal
WAIT_TOL_PP = 0.5              # gate (c) tolerance on GPUs waiting, percentage points
SLOTS = 30
WINDOW_S = 900.0
MAX_K = 8
TOP_FRAC = 0.10

PERCENTILES = (0.5, 0.25, 0.10)
CHECKPOINT_SETS = ((0.25, 0.5), (0.5,), (0.75,))
MIN_COMPLETED = (5, 10, 20)

PLANNERS = {
    "opus": lambda k: 2.48 + 1.65 * k,   # R36 fit, K <= 8
    "1s": lambda k: 1.0,
    "2s": lambda k: 2.0,
}

BENCH_DB = REPO / "data" / "bench" / "bench.db"
R41 = "f2bee0778f4f44ecab7562b993e04f56"
LCBENCH_RACES = ("5245bacc19804b3eacc3a45f68e3fa7b", "91a4e48d33eb44e8a93a0d580726c873",
                 "f98b671246594117884b7f5aa0fc1f60")

REGIMES: dict[str, dict[str, Any]] = {
    "rl30": dict(db=REPO / "data/collection/collect_rl.db", races=None, arms=None, metric="train_return",
                 direction="max", run_budget_s=30.0, stop_latency=STOP_LATENCY_S, sim_arms=None),
    "rl15": dict(db=BENCH_DB, races=(R41,), arms=("opus",), metric="train_return",
                 direction="max", run_budget_s=15.0, stop_latency=STOP_LATENCY_S, sim_arms=None),
    "cifar": dict(db=REPO / "data/collection/collect_cifar.db", races=None, arms=None, metric="val_acc",
                  direction="max", run_budget_s=45.0, stop_latency=STOP_LATENCY_S, sim_arms=None),
    "lm": dict(db=REPO / "data/collection/collect_lm.db", races=None, arms=None, metric="val_bpb",
               direction="min", run_budget_s=60.0, stop_latency=STOP_LATENCY_S, sim_arms=None),
    # surrogate runner: cancel is local and immediate; run_budget_s 300 x runtime multiplier 1
    "lcbench": dict(db=BENCH_DB, races=LCBENCH_RACES, arms=("opus", "optuna"), metric="val_balanced_accuracy",
                    direction="max", run_budget_s=300.0, stop_latency=0.0, sim_arms=("opus",)),
}


# ============================================================================ data


@dataclass
class Run:
    id: str
    stream: str                 # race:arm
    arm: str
    started: float              # wall clock
    ended: float
    curve: list[tuple[float, float]]
    score: float
    heldout: float | None
    final_curve: float
    ref_ok: bool                # may serve as a reference (not trap, not judged invalid)
    candidate: bool             # may be stopped / is graded (not baseline, not trap, not probe)
    top: dict[str, bool] = field(default_factory=dict)

    @property
    def dur(self) -> float:
        return self.ended - self.started


def connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def load_regime(name: str, cfg: dict[str, Any]) -> list[Run]:
    c = connect_ro(cfg["db"])
    probes: set[str] = set()
    if c.execute("select 1 from sqlite_master where name='collection_probes'").fetchone():
        probes = {r[0] for r in c.execute("select experiment_id from collection_probes")}
    sql = ("select e.id, e.race_id, a.kind, e.started_at, e.ended_at, e.score, e.metrics_json, e.is_trap, "
           "e.is_baseline, e.judge_verdict from experiments e join arms a on a.id = e.arm_id "
           "where e.status = 'succeeded'")
    rows = c.execute(sql).fetchall()
    curves: dict[str, list[tuple[float, float]]] = {}
    for eid, t, v in c.execute("select experiment_id, t, value from metrics where name = ?", (cfg["metric"],)):
        if t is not None and v is not None and math.isfinite(t) and math.isfinite(v):
            curves.setdefault(eid, []).append((float(t), float(v)))
    out: list[Run] = []
    for eid, race, arm, st, en, score, mj, trap, base, verdict in rows:
        if cfg["races"] is not None and race not in cfg["races"]:
            continue
        if cfg["arms"] is not None and arm not in cfg["arms"]:
            continue
        cur = sorted(curves.get(eid, []))
        if not cur or score is None or st is None or en is None:
            continue
        m = json.loads(mj or "{}")
        ho = m.get("heldout_score")
        is_trap = bool(trap) or bool(m.get("is_trap"))
        ref_ok = not is_trap and verdict != "invalid"
        out.append(Run(id=eid, stream=f"{race[:8]}:{arm}", arm=arm, started=float(st), ended=float(en),
                       curve=cur, score=float(score), heldout=float(ho) if ho is not None else None,
                       final_curve=cur[-1][1], ref_ok=ref_ok,
                       candidate=ref_ok and not base and eid not in probes))
    # top-10% per regime (candidates), by three outcomes
    cands = [r for r in out if r.candidate]
    sign = 1.0 if cfg["direction"] == "max" else -1.0
    for key, get in (("score", lambda r: r.score), ("heldout", lambda r: r.heldout),
                     ("final_curve", lambda r: r.final_curve)):
        vals = [r for r in cands if get(r) is not None]
        n_top = max(1, math.ceil(TOP_FRAC * len(vals)))
        ranked = sorted(vals, key=lambda r: -sign * get(r))
        top_ids = {r.id for r in ranked[:n_top]}
        for r in out:
            r.top[key] = r.id in top_ids
    return out


# ============================================================================ rule pieces


def ckpt_info(curve: list[tuple[float, float]], c: float) -> tuple[float, float] | None:
    """(decision t since launch, value) for checkpoint c, or None if the run never got there."""
    for i, (t, _v) in enumerate(curve):
        if t + 1e-9 >= c:
            prev = [vv for tt, vv in curve[: i + 1] if tt <= c + 1e-9]
            return t, (prev[-1] if prev else curve[0][1])
    return None


def ref_value(curve: list[tuple[float, float]], c: float) -> float | None:
    """A completed run's value at c (it went past c), as MedianStopRule.completed_values."""
    if curve[-1][0] + 1e-9 < c:
        return None
    v = None
    for t, vv in curve:
        if t <= c + 1e-9:
            v = vv
        else:
            break
    return v


def quantile(sorted_vals: list[float], p: float) -> float:
    """Linear interpolation (numpy default); p=0.5 equals statistics.median."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    x = p * (n - 1)
    lo = int(math.floor(x))
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (x - lo)


@dataclass(frozen=True)
class Rule:
    pct: float
    fractions: tuple[float, ...]
    min_completed: int

    @property
    def name(self) -> str:
        p = "median" if self.pct == 0.5 else f"p{int(round(self.pct * 100))}"
        f = "+".join(f"{int(x * 100)}%" for x in self.fractions)
        return f"{p} @ {f}, n>={self.min_completed}"

    def worse(self, v: float, ref_sorted: list[float], direction: str) -> bool:
        if direction == "max":
            return v < quantile(ref_sorted, self.pct)
        return v > quantile(ref_sorted, 1.0 - self.pct)


ALL_RULES = [Rule(p, f, m) for p in PERCENTILES for f in CHECKPOINT_SETS for m in MIN_COMPLETED]


# ============================================================================ causal replay


def replay(runs: list[Run], rule: Rule, cfg: dict[str, Any]) -> dict[str, Any]:
    budget, direction, lat = cfg["run_budget_s"], cfg["direction"], cfg["stop_latency"]
    cps = [(f, f * budget) for f in rule.fractions]
    stopped: dict[str, tuple[float, float]] = {}   # id -> (decision elapsed, fraction)
    skipped = checks = 0
    by_stream: dict[str, list[Run]] = {}
    for r in runs:
        by_stream.setdefault(r.stream, []).append(r)
    for srs in by_stream.values():
        events: list[tuple[float, int, int, str, int]] = []   # (wall t, order, seq, run id, ckpt idx)
        seq = 0
        info: dict[tuple[str, int], tuple[float, float]] = {}
        for r in srs:
            events.append((r.ended, 0, seq, r.id, -1)); seq += 1
            if r.candidate:
                for j, (_f, c) in enumerate(cps):
                    ci = ckpt_info(r.curve, c)
                    if ci is not None:
                        info[(r.id, j)] = ci
                        events.append((r.started + ci[0], 1, seq, r.id, j)); seq += 1
        events.sort()
        by_id = {r.id: r for r in srs}
        refs: list[list[float]] = [[] for _ in cps]
        for _t, kind, _s, rid, j in events:
            r = by_id[rid]
            if kind == 0:   # completion
                if rid in stopped or not r.ref_ok:
                    continue
                for jj, (_f, c) in enumerate(cps):
                    v = ref_value(r.curve, c)
                    if v is not None:
                        bisect.insort(refs[jj], v)
                continue
            if rid in stopped:
                continue
            checks += 1
            if len(refs[j]) < rule.min_completed:
                skipped += 1
                continue
            el, v = info[(rid, j)]
            if rule.worse(v, refs[j], direction):
                if el + lat < r.dur:      # otherwise the run finishes first ("too late")
                    stopped[rid] = (el, rule.fractions[j])
    cands = [r for r in runs if r.candidate]
    total = sum(r.dur for r in cands)
    freed = sum(r.dur - (stopped[r.id][0] + lat) for r in cands if r.id in stopped)
    out: dict[str, Any] = {
        "candidates": len(cands), "stopped": sum(1 for r in cands if r.id in stopped),
        "checks": checks, "skipped_few_completed": skipped,
        "stopped_share": sum(1 for r in cands if r.id in stopped) / len(cands),
        "slot_time_freed": freed / total if total else 0.0,
    }
    for key in ("score", "heldout", "final_curve"):
        tops = [r for r in cands if r.top.get(key)]
        k = sum(1 for r in tops if r.id in stopped)
        out[f"top10_{key}"] = {"stopped": k, "n": len(tops), "share": k / len(tops) if tops else 0.0}
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def predictiveness(runs: list[Run], cfg: dict[str, Any]) -> dict[str, Any]:
    """How well the curve value at each checkpoint ranks the final outcome (candidates)."""
    out: dict[str, Any] = {}
    for f in (0.25, 0.5, 0.75):
        c = f * cfg["run_budget_s"]
        pairs = [(ci[1], r.score, r.heldout) for r in runs if r.candidate
                 for ci in [ckpt_info(r.curve, c)] if ci is not None]
        out[f"{int(f * 100)}%"] = {
            "n_reached": len(pairs),
            "spearman_vs_score": spearman([p[0] for p in pairs], [p[1] for p in pairs]),
            "spearman_vs_heldout": spearman([p[0] for p in pairs if p[2] is not None],
                                            [p[2] for p in pairs if p[2] is not None]),
        }
    return out


# ============================================================================ event simulation


def simulate(pool: list[Run], rule: Rule | None, cfg: dict[str, Any], planner: str, seed: int,
             cutoff_s: float) -> dict[str, Any]:
    rng = random.Random(seed)
    draws: list[Run] = []
    lat_fn = PLANNERS[planner]
    budget, direction, stop_lat = cfg["run_budget_s"], cfg["direction"], cfg["stop_latency"]
    cps = [(f, f * budget) for f in rule.fractions] if rule else []
    cut_at = WINDOW_S - cutoff_s
    sign = 1.0 if direction == "max" else -1.0

    def draw(i: int) -> Run:
        while len(draws) <= i:   # common random numbers: the i-th proposal is the same run for every rule
            draws.append(pool[rng.randrange(len(pool))])
        return draws[i]

    heap: list[tuple[float, int, int, Any]] = []
    seq = 0

    def push(t: float, kind: int, data: Any) -> None:
        nonlocal seq
        heapq.heappush(heap, (t, kind, seq, data)); seq += 1

    free = SLOTS
    busy_time = 0.0
    planner_busy = False
    n_proposed = 0
    refs: list[list[float]] = [[] for _ in cps]
    alive: dict[int, dict[str, Any]] = {}
    completed: list[Run] = []
    n_stopped = 0
    release_t: list[float] = []   # per-slot release times after the cutoff (paid until then)

    def maybe_call(t: float) -> None:
        nonlocal planner_busy
        if planner_busy or t >= cut_at:
            return
        k = min(MAX_K, free)
        if k <= 0:
            return
        planner_busy = True
        push(t + lat_fn(k), 0, k)

    def release(t: float) -> None:
        # a slot going idle after the cutoff stops paying at max(t, cutoff)
        if t >= cut_at:
            release_t.append(min(max(t, cut_at), WINDOW_S))

    maybe_call(0.0)
    while heap:
        t, kind, _s, data = heapq.heappop(heap)
        if t > WINDOW_S:
            break
        if kind == 0:                     # planner returns K proposals
            planner_busy = False
            if t < cut_at:
                for _ in range(data):
                    r = draw(n_proposed)
                    rid = n_proposed
                    n_proposed += 1
                    free -= 1
                    alive[rid] = {"run": r, "start": t, "stop": False}
                    push(t + r.dur, 3, rid)
                    for j, (_f, c) in enumerate(cps):
                        ci = ckpt_info(r.curve, c)
                        if ci is not None:
                            push(t + ci[0], 1, (rid, j, ci[1], ci[0]))
            maybe_call(t)
        elif kind == 1:                   # checkpoint
            rid, j, v, el = data
            st = alive.get(rid)
            if st is None or st["stop"]:
                continue
            if len(refs[j]) < rule.min_completed:   # type: ignore[union-attr]
                continue
            if rule.worse(v, refs[j], direction) and el + stop_lat < st["run"].dur:  # type: ignore[union-attr]
                st["stop"] = True
                push(t + stop_lat, 2, rid)
        elif kind in (2, 3):              # stop lands / natural end
            st = alive.get(rid := data)
            if st is None:
                continue
            if kind == 3 and st["stop"]:
                continue                  # superseded by the stop (cannot happen: stop lands earlier)
            alive.pop(rid)
            busy_time += t - st["start"]
            free += 1
            if kind == 2:
                n_stopped += 1
            else:
                r = st["run"]
                completed.append(r)
                if r.ref_ok:
                    for jj, (_f, c) in enumerate(cps):
                        rv = ref_value(r.curve, c)
                        if rv is not None:
                            bisect.insort(refs[jj], rv)
            release(t)
            maybe_call(t)
    # runs still in flight at the deadline are cancelled and do not count
    for st in alive.values():
        busy_time += WINDOW_S - st["start"]
        release_t.append(WINDOW_S)
    # slots idle at the cutoff are released at the cutoff
    n_unreleased = SLOTS - len(release_t)
    paid = sum(release_t) + max(0, n_unreleased) * min(cut_at, WINDOW_S)
    waiting = 1.0 - busy_time / paid if paid else 0.0
    best = max(completed, key=lambda r: sign * r.score) if completed else None
    hos = [r.heldout for r in completed if r.heldout is not None]
    return {
        "waiting": waiting, "completed": len(completed), "stopped": n_stopped, "proposed": n_proposed,
        "best_score": best.score if best else None,
        "shipped_heldout": best.heldout if best else None,
        "best_heldout": (max(hos) if direction == "max" else min(hos)) if hos else None,
    }


def sim_summary(pool: list[Run], rule: Rule | None, cfg: dict[str, Any], planner: str, seeds: int,
                cutoff_s: float) -> dict[str, Any]:
    res = [simulate(pool, rule, cfg, planner, s, cutoff_s) for s in range(seeds)]
    def m(k: str) -> float | None:
        xs = [r[k] for r in res if r[k] is not None]
        return statistics.fmean(xs) if xs else None
    return {k: m(k) for k in ("waiting", "completed", "stopped", "proposed", "best_score",
                              "shipped_heldout", "best_heldout")} | {
        "waiting_sd": statistics.pstdev([r["waiting"] for r in res])}


# ============================================================================ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--regimes", default=",".join(REGIMES))
    args = ap.parse_args()
    t0 = time.time()
    results: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "script": "scripts/gate_early_stop_rules.py",
        "params": {"stop_latency_s": STOP_LATENCY_S, "wait_tol_pp": WAIT_TOL_PP, "slots": SLOTS,
                   "window_s": WINDOW_S, "max_k": MAX_K, "top_frac": TOP_FRAC, "seeds": args.seeds,
                   "planners": {"opus": "2.48 + 1.65*K s (R36)", "1s": "1 s", "2s": "2 s"},
                   "gate": "top10 wrong-stop (score and held-out) <= 5%, freed >= 20%, "
                           f"Opus-sim GPUs waiting <= no-stop + {WAIT_TOL_PP} pp"},
        "regimes": {},
    }
    for name in args.regimes.split(","):
        cfg = REGIMES[name]
        runs = load_regime(name, cfg)
        cands = [r for r in runs if r.candidate]
        sim_pool = [r for r in cands if cfg["sim_arms"] is None or r.arm in cfg["sim_arms"]]
        durs = sorted(r.dur for r in sim_pool)
        cutoff_s = quantile(durs, 0.9)
        print(f"\n== {name}: {len(runs)} runs ({len(cands)} candidates, {len({r.stream for r in runs})} streams), "
              f"sim pool {len(sim_pool)}, dur p50 {quantile(durs, .5):.1f}s, cutoff {cutoff_s:.1f}s")
        reg: dict[str, Any] = {
            "n_runs": len(runs), "n_candidates": len(cands), "streams": sorted({r.stream for r in runs}),
            "sim_pool": len(sim_pool), "dur_p50_s": quantile(durs, .5), "launch_cutoff_s": cutoff_s,
            "run_budget_s": cfg["run_budget_s"], "metric": cfg["metric"], "direction": cfg["direction"],
            "stop_latency_s": cfg["stop_latency"],
            "no_stop": {"sim": {p: sim_summary(sim_pool, None, cfg, p, args.seeds, cutoff_s) for p in PLANNERS}},
            "predictiveness": predictiveness(runs, cfg),
            "rules": {},
        }
        print("   checkpoint->final Spearman:", {k: (round(v["spearman_vs_score"], 2) if v["spearman_vs_score"] is not None else None, v["n_reached"]) for k, v in reg["predictiveness"].items()})
        base = reg["no_stop"]["sim"]
        print(f"   no stop: waiting opus {base['opus']['waiting']*100:.1f}% / 2s {base['2s']['waiting']*100:.1f}% / "
              f"1s {base['1s']['waiting']*100:.1f}%, best {base['opus']['best_score']:.4g}")
        for rule in ALL_RULES:
            rp = replay(runs, rule, cfg)
            sims = {p: sim_summary(sim_pool, rule, cfg, p, args.seeds, cutoff_s) for p in PLANNERS}
            dwait = (sims["opus"]["waiting"] - base["opus"]["waiting"]) * 100
            g = {
                "top10_score_ok": rp["top10_score"]["share"] <= 0.05,
                "top10_heldout_ok": rp["top10_heldout"]["share"] <= 0.05,
                "freed_ok": rp["slot_time_freed"] >= 0.20,
                "waiting_ok": dwait <= WAIT_TOL_PP,
                "waiting_delta_pp_opus": dwait,
            }
            g["pass"] = all(g[k] for k in ("top10_score_ok", "top10_heldout_ok", "freed_ok", "waiting_ok"))
            reg["rules"][rule.name] = {"rule": {"percentile": rule.pct, "fractions": list(rule.fractions),
                                                "min_completed": rule.min_completed},
                                       "replay": rp, "sim": sims, "gate": g}
            print(f"   {rule.name:28s} stop {rp['stopped_share']*100:5.1f}%  top10 val {rp['top10_score']['stopped']}/"
                  f"{rp['top10_score']['n']} ho {rp['top10_heldout']['stopped']}/{rp['top10_heldout']['n']}  "
                  f"freed {rp['slot_time_freed']*100:5.1f}%  wait opus {sims['opus']['waiting']*100:5.1f}% "
                  f"({dwait:+.1f}) 2s {sims['2s']['waiting']*100:5.1f}% 1s {sims['1s']['waiting']*100:5.1f}%  "
                  f"best {sims['opus']['best_score']:.4g}  {'PASS' if g['pass'] else ''}")
        passing = [(k, v) for k, v in reg["rules"].items() if v["gate"]["pass"]]
        if passing:
            k, v = max(passing, key=lambda kv: (kv[1]["replay"]["slot_time_freed"],
                                                 -kv[1]["replay"]["top10_heldout"]["share"]))
            reg["recommendation"] = {"rule": k, "why": "highest slot time freed among passing rules"}
        else:
            reg["recommendation"] = {"rule": "no early stopping", "why": "no candidate rule passes all D42 criteria"}
        print("   ->", reg["recommendation"])
        results["regimes"][name] = reg
    # sanity check against R41: the median rule as raced (25%+50%, n>=5) on the rl15 curves
    if "rl15" in results["regimes"]:
        r = results["regimes"]["rl15"]["rules"]["median @ 25%+50%, n>=5"]["replay"]
        results["r41_check"] = {"replayed_stopped_share": r["stopped_share"],
                                "r41_measured_opus_bm": 466 / 505, "r41_measured_optuna": 1649 / 2221}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'results.json'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
