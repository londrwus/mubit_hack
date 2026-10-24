"""Offline trap-exposure / quality estimate for 'backfill' on RL wide ($0; no LLM, no Modal).

RL training cannot be re-run for free, so sampler configs are scored by a CRUDE nearest-neighbour
surrogate of logged 15 s RL runs (R48 both arms, R41's opus arm, R50's two arms in ablation.db;
~6,400 succeeded runs): normalised config space (log for log params, index for categorical
choices, the trap flag excluded); a query returns the held-out score of a random one of its
K nearest logged runs (keeps run-to-run noise), and for a trap config adds the train-levels
inflation (reported - held-out) of a random one of its K nearest logged trap runs. A kNN cannot
extrapolate beyond the logged runs, so this estimates TRAP DYNAMICS and judge load, not how much
better a sampler could get than the best logged run.

Event replay of one arm, 30 slots, 900 s, deadline rule 17 s, run length drawn from the R48 pool:
- ``optuna``    TPE alone (as backend/agent/optuna_arm.py). Validates the surrogate against the
                measured Optuna arm (trap share ~97%, best held-out of counted runs -155.2).
- ``backfill``  the Opus arm's measured proposals at their measured times with their MEASURED
                outcomes (Opus held fixed: it does not react to the sampler), queued first; every
                other free slot gets a TPE config; the study is told every finished run.
Checker as in scripts/replay_backfill.py: a run beating the checked best is a claim (one judge
call); trap claims are rejected (assumed caught, judge v1 17/17 races, 0/16 test); feedback for
a rejected claim: ``raw`` / ``worst``. The sampler is never told which option is the trap.

Usage: PYTHONUTF8=1 python -m uv run python scripts/replay_backfill_rl.py --race 8a76fa05 [--seeds 2]
Writes data/backfill/replay_rl_<race>.json; ``--merge`` writes data/backfill/replay_rl.json.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import optuna  # noqa: E402

from backend.agent.optuna_arm import distributions  # noqa: E402
from workers.targets.rl import SPEC  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

BENCH = REPO / "data" / "bench" / "bench.db"
ABL = REPO / "data" / "bench" / "ablation.db"
OUT_DIR = REPO / "data" / "backfill"
SLOTS = 30
WINDOW_S = 900.0
CUT_AT = WINDOW_S - 17.0
GOAL = -170.0
KNN = 5
R48 = ("8a76fa05", "45ab180f", "59d7f7eb")
TRAINING = {BENCH: [("8a76fa05", None), ("45ab180f", None), ("59d7f7eb", None), ("f2bee077", "opus")], ABL: [(None, None)]}

PARAMS = [k for k in SPEC.config_space if k != SPEC.trap_option]


def encode(cfg: dict[str, Any]) -> list[float]:
    out = []
    for k in PARAMS:
        p = SPEC.config_space[k]
        v = cfg.get(k, p.default)
        if p.type == "bool":
            out.append(1.0 if v else 0.0)
        elif p.type == "categorical":
            ch = list(p.choices or [])
            out.append(ch.index(v) / max(1, len(ch) - 1) if v in ch else 0.5)
        else:
            lo, hi = float(p.low), float(p.high)  # type: ignore[arg-type]
            if p.log:
                out.append((math.log(float(v)) - math.log(lo)) / (math.log(hi) - math.log(lo)))
            else:
                out.append((float(v) - lo) / (hi - lo))
    return out


def ro(p: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)


class Surrogate:
    def __init__(self) -> None:
        X: list[list[float]] = []; H: list[float] = []
        XT: list[list[float]] = []; INF: list[float] = []
        for db, sel in TRAINING.items():
            c = ro(db)
            for race, arm in sel:
                q = ("select e.config_json, e.score, e.metrics_json, e.is_trap from experiments e join arms a on a.id=e.arm_id "
                     "where e.status='succeeded' and e.score is not null")
                args: list[Any] = []
                if race:
                    q += " and e.race_id like ?"; args.append(race + "%")
                if arm:
                    q += " and a.kind=?"; args.append(arm)
                for cj, s, mj, tr in c.execute(q, args):
                    m = json.loads(mj or "{}"); h = m.get("heldout_score")
                    if h is None:
                        continue
                    cfg = json.loads(cj); x = encode(cfg)
                    X.append(x); H.append(float(h))
                    if tr:
                        XT.append(x); INF.append(float(s) - float(h))
        self.X = np.array(X); self.H = np.array(H)
        self.XT = np.array(XT); self.INF = np.array(INF)
        self.n, self.nt = len(H), len(INF)

    def query(self, cfg: dict[str, Any], rng: random.Random) -> tuple[float, float]:
        x = np.array(encode(cfg))
        d = ((self.X - x) ** 2).sum(1)
        idx = np.argpartition(d, KNN)[:KNN]
        h = float(self.H[idx[rng.randrange(KNN)]])
        if SPEC.is_trap(cfg):
            dt = ((self.XT - x) ** 2).sum(1)
            it = np.argpartition(dt, KNN)[:KNN]
            return h + float(self.INF[it[rng.randrange(KNN)]]), h
        return h, h


def load_race(short: str) -> dict[str, Any]:
    c = ro(BENCH)
    rid = c.execute("select id from races where id like ?", (short + "%",)).fetchone()[0]
    out: dict[str, Any] = {}
    durs: list[float] = []
    for kind in ("opus", "optuna"):
        arm_id, a0, summ = c.execute("select id, started_at, summary_json from arms where race_id=? and kind=?",
                                     (rid, kind)).fetchone()
        rows = c.execute("select proposed_at, started_at, ended_at, config_json, is_baseline, status, score, metrics_json "
                         "from experiments where arm_id=? order by proposed_at", (arm_id,)).fetchall()
        durs += [e - s for _p, s, e, *_ in rows if s and e]
        props = []
        for p, s, e, cj, b, st, sc, mj in rows:
            if p is None:
                continue
            m = json.loads(mj or "{}")
            ok = st == "succeeded" and sc is not None and m.get("heldout_score") is not None
            props.append({"t": max(0.0, p - a0), "cfg": json.loads(cj), "base": bool(b), "ok": ok,
                          "score": float(sc) if ok else None, "heldout": float(m["heldout_score"]) if ok else None,
                          "dur": (e - s) if (s and e) else None})
        counted = [x["heldout"] for x in props if x["ok"] and not SPEC.is_trap(x["cfg"])]
        out[kind] = {"props": props, "summary": json.loads(summ or "{}"), "best_counted": max(counted) if counted else None}
    out["durs"] = durs
    return out


def replay(mode: str, seed: int, race: dict[str, Any], sur: Surrogate, feedback: str) -> dict[str, Any]:
    rng = random.Random(1000 + seed)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10, constant_liar=True))
    dists = distributions(SPEC)
    durs = race["durs"]
    heap: list[tuple[float, int, int, Any]] = []
    seq = 0

    def push(t: float, kind: int, data: Any) -> None:
        nonlocal seq
        heapq.heappush(heap, (t, kind, seq, data)); seq += 1

    opus_props = race["opus"]["props"] if mode == "backfill" else []
    for p in opus_props:
        push(p["t"], 0, p)
    if mode == "optuna":
        base = SPEC.default_config()
        h, _ = sur.query(base, rng)
        push(0.0, 0, {"t": 0.0, "cfg": base, "base": True, "ok": True, "score": h, "heldout": h, "dur": None})
    free = SLOTS
    queue: list[dict[str, Any]] = []
    checked_best: float | None = None
    worst: float | None = None
    st = {"exp_opus": 0, "exp_samp": 0, "samp_asked": 0, "samp_trap": 0, "judges_opus": 0, "judges_samp": 0,
          "trap_claims_opus": 0, "trap_claims_samp": 0, "genuine_claims_samp": 0}
    counted: list[tuple[float, float, str]] = []
    shipped: tuple[float, float, str] | None = None
    trap_share_by_min: dict[int, list[int]] = {}

    def launch(t: float, item: dict[str, Any], src: str, trial: Any) -> None:
        nonlocal free
        free -= 1
        dur = item["dur"] if (src == "opus" and item.get("dur")) else rng.choice(durs)
        push(t + dur, 1, (item, src, trial))

    def fill(t: float) -> None:
        while queue and free > 0 and t < CUT_AT:
            launch(t, queue.pop(0), "opus", None)
        while free > 0 and t < CUT_AT and mode != "opus":
            trial = study.ask()
            cfg = SPEC.validate_config(SPEC.suggest_optuna(trial))
            s, h = sur.query(cfg, rng)
            tr = SPEC.is_trap(cfg)
            st["samp_asked"] += 1; st["samp_trap"] += tr
            trap_share_by_min.setdefault(int(t // 60), []).append(int(tr))
            launch(t, {"cfg": cfg, "base": False, "ok": True, "score": s, "heldout": h}, "samp", trial)

    while heap:
        t, kind, _s, data = heapq.heappop(heap)
        if t > WINDOW_S:
            break
        if kind == 0:
            queue.append(data); fill(t); continue
        item, src, trial = data
        free += 1
        if not item["ok"]:
            if trial is not None:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            fill(t); continue
        st[f"exp_{'opus' if src == 'opus' else 'samp'}"] += 1
        cfg, s, h = item["cfg"], item["score"], item["heldout"]
        tr = SPEC.is_trap(cfg)
        told: float | None = s
        if item["base"]:
            checked_best = s if checked_best is None else max(checked_best, s)
            shipped = (s, h, src) if shipped is None or s > shipped[0] else shipped
        elif checked_best is None or s > checked_best:
            key = "opus" if src == "opus" else "samp"
            st[f"judges_{key}"] += 1
            if tr:
                st[f"trap_claims_{key}"] += 1
                if feedback == "worst":
                    told = worst if worst is not None else s
            else:
                if key == "samp":
                    st["genuine_claims_samp"] += 1
                checked_best = s
                shipped = (s, h, src)
        worst = s if worst is None else min(worst, s)
        if not tr:
            counted.append((h, t, src))
        if trial is not None:
            study.tell(trial, told)
        elif mode != "opus":
            params = {k: cfg[k] for k in dists if k in cfg}
            study.add_trial(optuna.trial.create_trial(params=params, distributions={k: dists[k] for k in params},
                                                      value=told))
        fill(t)
    best = max(counted, default=None)
    goal = min((tt for hh, tt, _ in counted if hh >= GOAL), default=None)
    return {**st, "best_heldout": best[0] if best else None, "best_src": best[2] if best else None,
            "shipped_heldout": shipped[1] if shipped else None, "shipped_src": shipped[2] if shipped else None,
            "time_to_goal": goal, "samp_trap_share": st["samp_trap"] / max(1, st["samp_asked"]),
            "trap_share_by_min": {k: sum(v) / len(v) for k, v in sorted(trap_share_by_min.items())},
            "samp_claim_rate": st["judges_samp"] / max(1, st["exp_samp"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", default=None)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--modes", nargs="+", default=["optuna", "backfill/raw", "backfill/worst"])
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        parts = {p.stem.split("_")[-1]: json.loads(p.read_text(encoding="utf-8")) for p in OUT_DIR.glob("replay_rl_*.json")
                 if p.stem != "replay_rl"}
        pooled: dict[str, list[dict[str, Any]]] = {}
        for part in parts.values():
            for m, xs in part["runs"].items():
                pooled.setdefault(m, []).extend(xs)
        summ = {}
        for m, xs in pooled.items():
            keys = [k for k in xs[0] if isinstance(xs[0][k], (int, float)) and not isinstance(xs[0][k], bool)]
            summ[m] = {k: {"mean": statistics.fmean(x[k] for x in xs if x[k] is not None),
                           "sd": statistics.pstdev([x[k] for x in xs if x[k] is not None]) if len(xs) > 1 else 0.0,
                           "n": sum(1 for x in xs if x[k] is not None)} for k in keys
                       if any(x[k] is not None for x in xs)}
            summ[m]["best_src"] = {s: sum(1 for x in xs if x["best_src"] == s) for s in ("opus", "samp")}
            summ[m]["shipped_src"] = {s: sum(1 for x in xs if x["shipped_src"] == s) for s in ("opus", "samp")}
        rates = {m.split("/")[1]: summ[m]["samp_claim_rate"]["mean"] for m in summ if m.startswith("backfill/")}
        res = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "races": sorted(parts), "pooled": summ,
               "measured": {k: v["measured"] for k, v in parts.items()}, "surrogate": next(iter(parts.values()))["surrogate"],
               "sim_rates": rates}
        (OUT_DIR / "replay_rl.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        print(json.dumps(res, indent=1, default=str)[:6000])
        return
    t0 = time.time()
    sur = Surrogate()
    race = load_race(args.race)
    print(f"surrogate: {sur.n} runs ({sur.nt} trap); race {args.race}: opus props {len(race['opus']['props'])}, "
          f"measured best counted opus {race['opus']['best_counted']:.1f} optuna {race['optuna']['best_counted']:.1f}")
    runs: dict[str, list[dict[str, Any]]] = {}
    for s in range(args.seeds):
        for m in args.modes:
            mode, _, fb = m.partition("/")
            x = replay(mode, s, race, sur, fb or "raw")
            runs.setdefault(m, []).append(x)
            print(f"seed {s} {m}: best {x['best_heldout']:.1f} ({x['best_src']}) shipped {x['shipped_heldout']:.1f} "
                  f"({x['shipped_src']}) exps opus {x['exp_opus']} samp {x['exp_samp']} trap share {x['samp_trap_share']:.2f} "
                  f"judges opus {x['judges_opus']} samp {x['judges_samp']} (trap {x['trap_claims_samp']}, genuine "
                  f"{x['genuine_claims_samp']}) goal {x['time_to_goal']} [{time.time() - t0:.0f}s]", flush=True)
    meas = {k: {"best_counted": race[k]["best_counted"], "waiting": race[k]["summary"].get("gpu_idle_pct"),
                "experiments": race[k]["summary"].get("experiments_finished")} for k in ("opus", "optuna")}
    (OUT_DIR / f"replay_rl_{args.race}.json").write_text(json.dumps(
        {"race": args.race, "runs": runs, "measured": meas, "surrogate": {"n": sur.n, "n_trap": sur.nt, "knn": KNN}},
        indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
