"""Offline quality gate for 'backfill' on LCBench ($0; no LLM, no Modal): does TPE fed with the
Opus agent's runs do at least as well as Opus alone and Optuna alone?

LCBench is a deterministic surrogate (workers/targets/lcbench.py), so any config the sampler asks
for can be scored for free, with the race's exact runtime (run length = predicted train_time_s,
SurrogateRunner multiplier 1.0). An event replay of one arm (16 slots, 900 s, late runs not
counted):

- ``opus``      only the Opus arm's own proposals, at their measured proposal times, in order
                (a slot-free check delays a proposal if all slots are busy). Validates the replay
                against the measured Opus arm.
- ``optuna``    only TPE (as backend/agent/optuna_arm.py: TPESampler(seed, n_startup_trials=10,
                constant_liar=True), baseline told first); one ask per free slot. Validates
                against the measured Optuna arm.
- ``backfill``  Opus's proposals first (same times, queued while slots are busy); any free slot
                with no Opus proposal waiting gets a TPE config at once. The study is told EVERY
                finished run (Opus's via add_trial, the sampler's via tell). Opus's proposals are
                held fixed: the counterfactual cannot re-run Opus, so it does not react to the
                sampler's results (it would see them in its digest).

Checker (D40, as the scheduler): runs are processed in end order; a run whose reported score beats
the checked best is a claim -> one judge call. A trap run (checkpoint_selection=best_test) that
claims is rejected (judge v1 caught 17/17 fake claims in races, R54; 0/16 fakes on the test split,
R40) -- that is the *assumption* here, the judge is not called; genuine claims are accepted
(accept_unsure). The sampler is never told which option is the trap. What TPE is told for a run
the checker rejected (``--feedback``): ``raw`` (its reported score), ``fail`` (TrialState.FAIL),
``worst`` (the worst score seen so far). Runs that are not claims are always told their reported
score (a trap run below the best is never judged).

Outputs per mode: best held-out among counted runs (headline 'model quality'), the shipped run's
held-out (best checked run), experiments, sampler share, trap proposals by the sampler, judge
calls by source. Also ``continue``: tell TPE the Opus arm's complete 15-min history, then ask
M more configs (batch 16): does TPE improve on Opus's best?

Usage: PYTHONUTF8=1 python -m uv run --with onnxruntime==1.29.0 python scripts/replay_backfill.py
       [--seeds 10] [--feedback raw fail worst]
Writes data/backfill/replay_lcbench.json.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import optuna  # noqa: E402

from backend.agent.optuna_arm import distributions  # noqa: E402
from workers.targets.lcbench import BUDGET_S, SPEC, evaluate, get_surrogate  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

BENCH_DB = REPO / "data" / "bench" / "bench.db"
OUT = REPO / "data" / "backfill" / "replay_lcbench.json"
SLOTS = 16
WINDOW_S = 900.0
GOAL = 0.78
RACES = {"R38": ("5245bacc", "91a4e48d", "f98b6712"), "R44": ("2ded10bc", "e3fddb29", "aaedd180")}

_cache: dict[str, dict[str, Any]] = {}


def score(config: dict[str, Any]) -> dict[str, Any]:
    key = json.dumps(config, sort_keys=True)
    if key not in _cache:
        try:
            out = evaluate(config, float(BUDGET_S), surrogate=get_surrogate())
            _cache[key] = {"ok": True, "score": float(out["score"]), "heldout": float(out["heldout_score"]),
                           "dur": float(out["train_time_s"])}
        except TimeoutError:
            _cache[key] = {"ok": False, "dur": 1.0}
    return _cache[key]


def load_opus(short: str) -> dict[str, Any]:
    c = sqlite3.connect(f"file:{BENCH_DB.as_posix()}?mode=ro", uri=True)
    rid = c.execute("select id from races where id like ?", (short + "%",)).fetchone()[0]
    out: dict[str, Any] = {"race": short}
    for kind in ("opus", "optuna"):
        arm_id, a0, summ = c.execute("select id, started_at, summary_json from arms where race_id=? and kind=?",
                                     (rid, kind)).fetchone()
        rows = c.execute("select proposed_at, config_json, is_baseline, status, score, is_trap, metrics_json, judge_verdict "
                         "from experiments where arm_id=? order by proposed_at", (arm_id,)).fetchall()
        props = [(max(0.0, p - a0), json.loads(cj), bool(b)) for p, cj, b, *_ in rows if p is not None]
        counted = [json.loads(m or "{}").get("heldout_score") for _p, _c, _b, st, s, tr, m, v in rows
                   if st == "succeeded" and not tr and v != "invalid" and s is not None]
        counted = [h for h in counted if h is not None]
        out[kind] = {"props": props, "measured_best_heldout": max(counted) if counted else None,
                     "measured_summary": json.loads(summ or "{}"), "n": len(rows)}
    return out


class Arm:
    def __init__(self, mode: str, seed: int, opus_props: list[tuple[float, dict[str, Any], bool]],
                 feedback: str, opus_slots: int = SLOTS) -> None:
        self.mode, self.feedback, self.opus_slots = mode, feedback, opus_slots
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10, constant_liar=True)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)
        self.dists = distributions(SPEC)
        self.opus_props = list(opus_props) if mode != "optuna" else []
        self.results: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        heap: list[tuple[float, int, int, Any]] = []
        seq = 0

        def push(t: float, kind: int, data: Any) -> None:
            nonlocal seq
            heapq.heappush(heap, (t, kind, seq, data)); seq += 1

        free = SLOTS
        queue: list[tuple[float, dict[str, Any], bool]] = []   # Opus proposals waiting for a slot
        for p in self.opus_props:
            push(p[0], 0, p)
        if self.mode == "optuna":
            base = SPEC.default_config()
            push(0.0, 0, (0.0, base, True))
        checked_best: float | None = None
        worst: float | None = None
        n_sampler = 0
        trap_samp = 0
        judges = {"opus": 0, "samp": 0}
        trap_claims = {"opus": 0, "samp": 0}
        rejected = {"opus": 0, "samp": 0}
        delays: list[float] = []
        launched = {"opus": 0, "samp": 0}

        opus_running = 0

        def launch(t: float, cfg: dict[str, Any], src: str, trial: Any, is_base: bool) -> None:
            nonlocal free, opus_running
            free -= 1
            opus_running += src == "opus"
            r = score(cfg)
            launched[src] += 1
            push(t + r["dur"], 1, (cfg, src, trial, is_base, r, t))

        def fill(t: float) -> None:
            nonlocal n_sampler, trap_samp
            while queue and free > 0 and opus_running < self.opus_slots:
                pt, cfg, b = queue.pop(0)
                delays.append(t - pt)
                launch(t, cfg, "opus", None, b)
            if self.mode == "opus":
                return
            # the sampler leaves the Opus share free while Opus proposals wait for it
            hold = min(len(queue), max(0, self.opus_slots - opus_running)) if self.opus_slots < SLOTS else 0
            while free - hold > 0 and t < WINDOW_S:
                trial = self.study.ask()
                cfg = SPEC.validate_config(SPEC.suggest_optuna(trial))
                n_sampler += 1
                trap_samp += SPEC.is_trap(cfg)
                launch(t, cfg, "samp", trial, False)

        while heap:
            t, kind, _s, data = heapq.heappop(heap)
            if t > WINDOW_S:
                break
            if kind == 0:
                queue.append(data)
                fill(t)
                continue
            cfg, src, trial, is_base, r, t_start = data
            free += 1
            opus_running -= src == "opus"
            is_trap = SPEC.is_trap(cfg)
            row = {"t_end": t, "src": src, "trap": is_trap, "base": is_base, **r, "accepted": None,
                   "cfg": json.dumps(cfg, sort_keys=True)}
            told: float | None = r["score"] if r["ok"] else None
            if r["ok"] and not is_base and (checked_best is None or r["score"] > checked_best):
                judges[src] += 1
                if is_trap:
                    trap_claims[src] += 1
                    rejected[src] += 1
                    row["accepted"] = False
                    if self.feedback == "fail":
                        told = None
                    elif self.feedback == "worst":
                        told = worst if worst is not None else r["score"]
                else:
                    row["accepted"] = True
                    checked_best = r["score"]
            elif r["ok"] and is_base:
                row["accepted"] = True
                checked_best = r["score"] if checked_best is None else max(checked_best, r["score"])
            if r["ok"]:
                worst = r["score"] if worst is None else min(worst, r["score"])
            # tell the study
            if trial is not None:
                if told is None:
                    self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
                else:
                    self.study.tell(trial, told)
            elif told is not None and self.mode != "opus":
                params = {k: cfg[k] for k in self.dists if k in cfg}
                self.study.add_trial(optuna.trial.create_trial(
                    params=params, distributions={k: self.dists[k] for k in params}, value=told))
            self.results.append(row)
            fill(t)
        counted = [x for x in self.results if x["ok"] and not x["trap"]]
        shipped = max((x for x in self.results if x["accepted"]), key=lambda x: x["score"], default=None)
        best = max((x["heldout"] for x in counted), default=None)
        goal_t = min((x["t_end"] for x in counted if x["heldout"] >= GOAL), default=None)
        best_src = max(counted, key=lambda x: x["heldout"])["src"] if counted else None
        return {
            "best_heldout": best, "shipped_heldout": shipped["heldout"] if shipped else None,
            "shipped_src": shipped["src"] if shipped else None, "best_src": best_src,
            "experiments": len(self.results), "exp_opus": sum(1 for x in self.results if x["src"] == "opus"),
            "exp_samp": sum(1 for x in self.results if x["src"] == "samp"),
            "sampler_trap_share": trap_samp / max(1, n_sampler), "sampler_trap_runs": trap_samp,
            "judges_opus": judges["opus"], "judges_samp": judges["samp"],
            "trap_claims_opus": trap_claims["opus"], "trap_claims_samp": trap_claims["samp"],
            "opus_delay_p50": statistics.median(delays) if delays else 0.0,
            "time_to_goal": goal_t,
        }


def continue_tpe(opus_props: list[tuple[float, dict[str, Any], bool]], seed: int, m: int) -> dict[str, Any]:
    """opus_props: the Opus runs that finished within the race (from the ``opus`` replay)."""
    """Tell TPE Opus's full history (reported scores, trap claims as the checker sees them is
    irrelevant here: told raw), then ask m more in batches of 16."""
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10, constant_liar=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    dists = distributions(SPEC)
    opus_best = None
    for _t, cfg, _b in opus_props:
        r = score(cfg)
        if r["ok"]:
            study.add_trial(optuna.trial.create_trial(params={k: cfg[k] for k in dists if k in cfg},
                                                      distributions=dists, value=r["score"]))
            if not SPEC.is_trap(cfg):
                opus_best = r["heldout"] if opus_best is None else max(opus_best, r["heldout"])
    best_new = None; traps = 0
    curve: list[float] = []
    for _ in range(math.ceil(m / 16)):
        trials = [study.ask() for _ in range(16)]
        for tr in trials:
            cfg = SPEC.validate_config(SPEC.suggest_optuna(tr))
            r = score(cfg)
            traps += SPEC.is_trap(cfg)
            if r["ok"]:
                study.tell(tr, r["score"])
                if not SPEC.is_trap(cfg):
                    best_new = r["heldout"] if best_new is None else max(best_new, r["heldout"])
            else:
                study.tell(tr, state=optuna.trial.TrialState.FAIL)
        curve.append(max(best_new or 0.0, opus_best or 0.0))
    return {"opus_best": opus_best, "tpe_best_new": best_new, "combined": max(opus_best or 0, best_new or 0),
            "trap_share": traps / (16 * math.ceil(m / 16)), "curve_by_batch": curve}


def agg(vals: list[float | None]) -> dict[str, Any]:
    v = [x for x in vals if x is not None]
    return {"mean": statistics.fmean(v) if v else None, "sd": statistics.pstdev(v) if len(v) > 1 else 0.0,
            "n": len(v), "min": min(v) if v else None, "max": max(v) if v else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--feedback", nargs="+", default=["raw", "fail", "worst"])
    ap.add_argument("--continue-m", type=int, default=64)
    ap.add_argument("--split", type=int, nargs="*", default=[5],
                    help="Opus may hold at most L of the 16 slots (RL-like regime: on RL wide Opus fills ~30%%)")
    args = ap.parse_args()
    t0 = time.time()
    races = {short: load_opus(short) for g in RACES.values() for short in g}
    res: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "script": "scripts/replay_backfill.py",
                           "seeds": args.seeds, "measured": {}, "per_race": {}, "pooled": {}}
    pooled: dict[str, list[dict[str, Any]]] = {}
    for short, r in races.items():
        res["measured"][short] = {"opus_best_heldout": r["opus"]["measured_best_heldout"],
                                  "optuna_best_heldout": r["optuna"]["measured_best_heldout"]}
        props = r["opus"]["props"]
        out: dict[str, Any] = {}
        oa = Arm("opus", 0, props, "raw")
        o = oa.run()
        out["opus"] = o
        pooled.setdefault("opus", []).append(o)
        done_cfgs = [(0.0, json.loads(x["cfg"]), x["base"]) for x in oa.results]
        for L in args.split:
            ol = Arm("opus", 0, props, "raw", opus_slots=L).run()
            out[f"opus@{L}"] = [ol]
            pooled.setdefault(f"opus@{L}", []).append(ol)
        for s in range(args.seeds):
            for name, mode, fb in [("optuna", "optuna", "raw")] + [(f"backfill/{f}", "backfill", f) for f in args.feedback]:
                x = Arm(mode, s, props, fb).run()
                out.setdefault(name, []).append(x)
                pooled.setdefault(name, []).append(x)
            for L in args.split:
                for f in ("raw", "worst"):
                    x = Arm("backfill", s, props, f, opus_slots=L).run()
                    out.setdefault(f"backfill@{L}/{f}", []).append(x)
                    pooled.setdefault(f"backfill@{L}/{f}", []).append(x)
            c = continue_tpe(done_cfgs, s, args.continue_m)
            out.setdefault("continue", []).append(c)
            pooled.setdefault("continue", []).append(c)
        res["per_race"][short] = {k: (v if isinstance(v, dict) else {
            kk: agg([x.get(kk) for x in v]) for kk in v[0]
            if (isinstance(v[0].get(kk), (int, float)) and not isinstance(v[0].get(kk), bool)) or v[0].get(kk) is None
        }) for k, v in out.items()}
        bf = {f: statistics.fmean(x["best_heldout"] for x in out[f"backfill/{f}"]) for f in args.feedback}
        print(f"{short}: measured opus {r['opus']['measured_best_heldout']:.4f} optuna {r['optuna']['measured_best_heldout']:.4f} | "
              f"replay opus {o['best_heldout']:.4f} ({o['experiments']} exps) optuna "
              f"{statistics.fmean(x['best_heldout'] for x in out['optuna']):.4f} | backfill "
              + " ".join(f"{f} {v:.4f}" for f, v in bf.items())
              + f" | continue(+{args.continue_m}) {statistics.fmean(x['combined'] for x in out['continue']):.4f} "
              f"[{time.time() - t0:.0f}s]")
    for name, xs in pooled.items():
        keys = [k for k in xs[0] if isinstance(xs[0][k], (int, float)) and not isinstance(xs[0][k], bool)] + \
               [k for k in xs[0] if xs[0][k] is None]
        res["pooled"][name] = {k: agg([x.get(k) for x in xs]) for k in keys}
        res["pooled"][name]["src_counts"] = {
            key: {s: sum(1 for x in xs if x.get(key) == s) for s in ("opus", "samp")}
            for key in ("best_src", "shipped_src") if key in xs[0]}
    # per-race means (n = 6 races) for a seed-noise comparison at the race level
    per_race_means: dict[str, list[float]] = {}
    for short in races:
        pr = res["per_race"][short]
        for name, v in pr.items():
            if name == "opus" or "cfg" in v:
                per_race_means.setdefault(name, []).append(v["best_heldout"])
            elif "best_heldout" in v:
                per_race_means.setdefault(name, []).append(v["best_heldout"]["mean"])
    res["race_level"] = {k: agg(v) for k, v in per_race_means.items()}
    # rates for scripts/sim_backfill.py: judge calls per sampler run under the checker
    rates = {}
    for f in args.feedback:
        xs = pooled[f"backfill/{f}"]
        rates[f] = sum(x["judges_samp"] for x in xs) / max(1, sum(x["exp_samp"] for x in xs))
    res["sim_rates"] = rates
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: {kk: (round(vv["mean"], 4) if isinstance(vv, dict) and vv.get("mean") is not None else vv)
                          for kk, vv in v.items()} for k, v in res["pooled"].items()}, indent=1, default=str))
    print("race level:", json.dumps(res["race_level"], default=str))
    print("sim rates:", rates, f"wrote {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
