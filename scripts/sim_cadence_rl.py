"""Offline pick of ``min_propose_interval_s`` (planner_cadence, D47) for Busy Machines on RL wide
($0; no LLM, no Modal). PROJECTIONS, not measurements.

One event replay of the BM arm (R67 preset ``rl-wide-bm-15m``: 30 CPU slots, 15 min, deadline
rule 17 s) that combines the two existing offline tools:
- the planner lane of scripts/sim_backfill.py (one sequential lane as ``ArmRunner._agent_loop``:
  process finished runs in order, a judge call on every claimed new best, then one propose call;
  K = min(8, LLM demand), demand counting only LLM runs + buffered LLM proposals), plus the
  cadence rule (a propose call starts only ``min_propose_interval_s`` after the previous START);
- the RL replay of scripts/replay_backfill_rl.py (sampler = real Optuna TPE as BackfillSampler,
  n_startup 10, constant_liar, checker-gated 'worst on rejected' feedback; its configs scored by
  the CRUDE kNN surrogate of logged 15 s RL runs).

Calibrated on R67's MEASURED BM arm (races 01004d9f, e648f7f6): lean-planner latency and $ per
call by K, malformed rate, judge latency and $, durations of LLM and sampler runs, and the BM
LLM proposal stream itself (configs with their measured scores / held-out scores / durations).
A cadence replay can only re-use proposals R67's BM actually made, so two bounds:
- ``first``  call n gets the next K proposals of R67's stream in order (early, less-informed
             proposals; pessimistic on quality);
- ``spread`` a call at time t gets the K earliest unused proposals R67 made at or after t (made
             with R67's richer history; optimistic on quality).
Validation (``none``, no cadence, ``first``): must reproduce R67 BM's measured LLM $, calls,
waiting, experiments and best held-out.

Surrogate fix: replay_backfill_rl.py reads ALL of data/bench/ablation.db, which now also holds
three LCBench races; this script restricts it to the RL race (R50, fa7e7a5d).

Usage: PYTHONUTF8=1 python -m uv run python scripts/sim_cadence_rl.py [--seeds 3] [--procs 8]
Writes data/backfill/cadence_rl.json.
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import sqlite3
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import optuna  # noqa: E402

import replay_backfill_rl as R  # noqa: E402
from backend.agent.optuna_arm import distributions  # noqa: E402
from workers.targets.rl import SPEC  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

# RL-only surrogate training (the original reads every ablation.db race, LCBench included).
R.TRAINING = {R.BENCH: list(R.TRAINING[R.BENCH]), R.ABL: [("fa7e7a5d", None)]}

R67 = ("01004d9f", "e648f7f6")
SLOTS, WINDOW_S, CUT_AT, KCAP = 30, 900.0, 900.0 - 17.0, 8
OUT = REPO / "data" / "backfill" / "cadence_rl.json"
INTERVALS = (15.0, 30.0, 45.0, 60.0, 90.0)


def ro(p: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)


def load() -> dict[str, Any]:
    c = ro(R.BENCH)
    lane = {"lat_by_k": {}, "cost_by_k": {}, "n_prop": 0, "n_inv": 0, "inv": [], "jlat": [], "jcost": []}
    races: dict[str, Any] = {}
    dur_samp: list[float] = []
    for short in R67:
        rid = c.execute("select id from races where id like ?", (short + "%",)).fetchone()[0]
        arms = {k: (a, t0, json.loads(s or "{}")) for a, k, t0, s in
                c.execute("select id, kind, started_at, summary_json from arms where race_id=?", (rid,))}
        aid, a0, summ = arms["opus_bm"]
        for st, lat, cost, ids, valid in c.execute(
                "select step_type, latency_ms, cost_usd, experiment_ids_json, valid from llm_calls where arm_id=? "
                "order by created_at", (aid,)):
            if st == "judge_improvement":
                lane["jlat"].append(lat / 1000); lane["jcost"].append(cost); continue
            lane["n_prop"] += 1
            k = len(json.loads(ids or "[]"))
            if not valid or k == 0:
                lane["n_inv"] += 1; lane["inv"].append((lat / 1000, cost)); continue
            lane["lat_by_k"].setdefault(k, []).append(lat / 1000)
            lane["cost_by_k"].setdefault(k, []).append(cost)
        rows = c.execute("select proposed_at, started_at, ended_at, config_json, source, status, score, is_trap, "
                         "metrics_json from experiments where arm_id=? order by proposed_at", (aid,)).fetchall()
        stream, base, meas_src = [], None, {"llm": [], "sampler": []}
        n_src = {"llm": 0, "sampler": 0}; trap_samp = 0
        for p, s, e, cj, src, stt, sc, tr, mj in rows:
            m = json.loads(mj or "{}")
            ok = stt == "succeeded" and sc is not None and m.get("heldout_score") is not None
            dur = (e - s) if (s and e) else None
            if src == "baseline":
                base = {"score": float(sc), "dur": dur}
                continue
            if src in n_src and stt in ("succeeded", "failed", "timed_out"):
                n_src[src] += 1
            if src == "sampler":
                if dur:
                    dur_samp.append(dur)
                trap_samp += bool(tr)
                if ok and not tr:
                    meas_src["sampler"].append(float(m["heldout_score"]))
                continue
            if src == "llm":
                if ok and not tr:
                    meas_src["llm"].append(float(m["heldout_score"]))
                stream.append({"t": max(0.0, (p or a0) - a0), "cfg": json.loads(cj), "ok": ok,
                               "score": float(sc) if ok else None,
                               "heldout": float(m["heldout_score"]) if ok else None, "dur": dur})
        n_j = c.execute("select count(*) from llm_calls where arm_id=? and step_type='judge_improvement'", (aid,)).fetchone()[0]
        n_p = c.execute("select count(*) from llm_calls where arm_id=? and step_type='propose_experiments'", (aid,)).fetchone()[0]
        opus = arms["opus"]
        opus_best = c.execute("select max(json_extract(metrics_json,'$.heldout_score')) from experiments where arm_id=? "
                              "and status='succeeded' and is_trap=0 and source!='baseline'", (opus[0],)).fetchone()[0]
        races[short] = {
            "stream": stream, "base": base,
            "measured_bm": {"llm_usd": summ.get("llm_cost_usd"), "waiting_pct": summ.get("gpu_idle_pct"),
                            "experiments": summ.get("experiments_finished"), "exp_llm": n_src["llm"],
                            "exp_samp": n_src["sampler"], "calls_propose": n_p, "calls_judge": n_j,
                            "best_heldout": max(meas_src["llm"] + meas_src["sampler"]),
                            "best_llm": max(meas_src["llm"]), "best_samp": max(meas_src["sampler"]),
                            "samp_trap_share": trap_samp / max(1, n_src["sampler"])},
            "measured_opus": {"llm_usd": opus[2].get("llm_cost_usd"), "waiting_pct": opus[2].get("gpu_idle_pct"),
                              "best_heldout": float(opus_best), "experiments": opus[2].get("experiments_finished")},
        }
    lane["p_inv"] = lane["n_inv"] / max(1, lane["n_prop"])
    lane["dur_samp"] = dur_samp
    # Measured launch gaps (slot free -> next run started on that slot), by the next run's source.
    gaps: dict[str, list[float]] = {"llm": [], "sampler": [], "optuna": []}
    for short in R67:
        rid = c.execute("select id from races where id like ?", (short + "%",)).fetchone()[0]
        for kind in ("opus_bm", "optuna"):
            aid = c.execute("select id from arms where race_id=? and kind=?", (rid, kind)).fetchone()[0]
            prev: dict[Any, float] = {}
            for sl, s0, e0, src in c.execute("select slot_id, started_at, ended_at, source from experiments where arm_id=? "
                                             "and started_at is not null and ended_at is not null order by slot_id, "
                                             "started_at", (aid,)):
                if sl in prev and src in gaps:
                    gaps[src].append(max(0.0, s0 - prev[sl]))
                prev[sl] = e0
    lane["gaps"] = gaps
    lane["dur_llm"] = [x["dur"] for r in races.values() for x in r["stream"] if x["dur"]]
    return {"lane": lane, "races": races}


def near_k(d: dict[int, list[float]], k: int) -> list[float]:
    ks = [kk for kk, v in d.items() if len(v) >= 5] or list(d)
    return d[min(ks, key=lambda kk: (abs(kk - k), -kk))]


def ask_time(n: int) -> float:   # as scripts/sim_backfill.py (TPE ask grows with study size)
    return 0.02 + 0.07 * n / 600


_SUR: Any = None
_DATA: Any = None


def _init() -> None:
    global _SUR, _DATA
    _SUR = R.Surrogate()
    _DATA = load()


def replay(interval: float | None, strategy: str, race: str, seed: int, gapsrc: str = "bm") -> dict[str, Any]:
    """gapsrc: launch delay of a sampler run = a measured gap before a sampler run of R67's BM arm
    (``bm``), of R67's Optuna arm (``optuna``, pessimistic: includes the CPU-load artefact R67
    notes), or the serial TPE ask-time model of sim_backfill.py (``model``)."""
    sur, data = _SUR, _DATA
    lane, rd = data["lane"], data["races"][race]
    stream = rd["stream"]
    rng = random.Random(7919 * seed + 101 * R67.index(race))
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10, constant_liar=True))
    dists = distributions(SPEC)
    heap: list[tuple[float, int, int, Any]] = []
    seq = 0

    def push(t: float, kind: int, d: Any) -> None:
        nonlocal seq
        heapq.heappush(heap, (t, kind, seq, d)); seq += 1

    used = [False] * len(stream)
    ptr = 0
    exhausted = 0

    def draw(t: float, k: int) -> list[dict[str, Any]]:
        nonlocal ptr, exhausted
        out = []
        if strategy == "first":
            for _ in range(k):
                if ptr >= len(stream):
                    exhausted += 1; ptr = 0; used[:] = [False] * len(stream)
                out.append(stream[ptr]); used[ptr] = True; ptr += 1
            return out
        # spread: earliest unused with proposed time >= t, then the latest unused before t
        idx = [i for i in range(len(stream)) if not used[i] and stream[i]["t"] >= t][:k]
        if len(idx) < k:
            back = [i for i in range(len(stream)) if not used[i] and stream[i]["t"] < t]
            idx += back[::-1][:k - len(idx)]
        for i in idx:
            used[i] = True; out.append(stream[i])
        exhausted += k - len(idx)
        return out

    free = SLOTS
    held = 0
    llm_running = 0
    buf: list[dict[str, Any]] = []
    lane_busy = False
    procq: list[tuple[dict[str, Any], str, Any]] = []
    last_start: float | None = None
    wake_at: float | None = None
    samp_busy = 0.0
    n_trials = 0
    checked_best: float | None = None
    worst: float | None = None
    shipped: tuple[float, float, str] | None = None
    counted: list[tuple[float, float, str]] = []
    busy = 0.0
    release: list[float] = []
    st = {"calls_propose": 0, "calls_invalid": 0, "calls_judge": 0, "judges_llm": 0, "judges_samp": 0,
          "trap_claims_samp": 0, "usd_propose": 0.0, "usd_judge": 0.0, "exp_llm": 0, "exp_samp": 0,
          "samp_asked": 0, "samp_trap": 0, "llm_props": 0, "k_sum": 0, "gaps": []}

    gl = lane["gaps"]["llm"]
    gs = lane["gaps"]["sampler" if gapsrc == "bm" else "optuna"]

    def launch(t: float, item: dict[str, Any], src: str, trial: Any) -> None:
        nonlocal free, llm_running
        free -= 1
        if src == "llm" and gapsrc != "model":
            t += rng.choice(gl)
        if src == "llm":
            llm_running += 1
            dur = item["dur"] or rng.choice(lane["dur_llm"])
        elif src == "base":
            dur = item["dur"] or rng.choice(lane["dur_llm"])
        else:
            dur = rng.choice(lane["dur_samp"])
        push(t + dur, 2, (item, src, trial, dur))

    def fill(t: float) -> None:
        nonlocal free, held, samp_busy, n_trials
        while buf and free > 0 and t < CUT_AT:
            launch(t, buf.pop(0), "llm", None)
        if t >= CUT_AT or buf:
            return
        while free > 0:
            if gapsrc == "model":
                samp_busy = max(t, samp_busy) + ask_time(n_trials); at = samp_busy
            else:
                at = t + rng.choice(gs)
            n_trials += 1
            free -= 1; held += 1
            push(at, 3, None)

    def dispatch(t: float) -> None:
        nonlocal lane_busy, last_start, wake_at, checked_best, shipped
        while not lane_busy and procq:
            item, src, trial = procq.pop(0)
            if process(t, item, src, trial):
                return
        if lane_busy or t >= CUT_AT:
            return
        k = min(KCAP, SLOTS - llm_running - len(buf))
        if k <= 0:
            return
        if interval is not None and last_start is not None and last_start + interval - t > 1e-3:
            due = last_start + interval
            if wake_at is None or wake_at != due:
                wake_at = due; push(due, 4, None)
            return
        if last_start is not None:
            st["gaps"].append(t - last_start)
        last_start = t
        lane_busy = True
        lat = cost = 0.0
        if rng.random() < lane["p_inv"]:          # malformed reply: retried inside the same call
            il, ic = rng.choice(lane["inv"]); lat += il; cost += ic; st["calls_invalid"] += 1
        lat += rng.choice(near_k(lane["lat_by_k"], k)); cost += rng.choice(near_k(lane["cost_by_k"], k))
        st["calls_propose"] += 1; st["usd_propose"] += cost; st["k_sum"] += k
        push(t + lat, 0, draw(t, k))

    def tell(item: dict[str, Any], src: str, trial: Any, value: float | None) -> None:
        if trial is not None:
            if value is None:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            else:
                study.tell(trial, value)
        elif value is not None:
            params = {k: item["cfg"][k] for k in dists if k in item["cfg"]}
            try:
                study.add_trial(optuna.trial.create_trial(params=params, distributions={k: dists[k] for k in params},
                                                          value=value))
            except ValueError:
                pass

    def process(t: float, item: dict[str, Any], src: str, trial: Any) -> bool:
        """Returns True when the lane went busy with a judge call (the verdict resumes processing)."""
        nonlocal checked_best, shipped, worst, lane_busy
        if src == "base":
            checked_best = item["score"]; shipped = (item["score"], item["score"], "base")
            worst = item["score"]
            tell(item, src, None, item["score"]); return False
        if not item["ok"]:
            tell(item, src, trial, None); return False
        s, h = item["score"], item["heldout"]
        tr = SPEC.is_trap(item["cfg"])
        if not tr:
            counted.append((h, t, src))
        if checked_best is None or s > checked_best:
            j = rng.randrange(len(lane["jlat"]))
            st["calls_judge"] += 1; st["usd_judge"] += lane["jcost"][j]
            st["judges_llm" if src == "llm" else "judges_samp"] += 1
            lane_busy = True
            push(t + lane["jlat"][j], 1, (item, src, trial))
            return True

        worst = s if worst is None else min(worst, s)
        tell(item, src, trial, s)

        return False

    def verdict(item: dict[str, Any], src: str, trial: Any) -> None:
        nonlocal checked_best, shipped, worst
        s, h = item["score"], item["heldout"]
        if SPEC.is_trap(item["cfg"]):             # judge v1 catches leaks (17/17 races, 0/16 test)
            if src != "llm":
                st["trap_claims_samp"] += 1
            told = worst if worst is not None else None
        else:
            checked_best = s; shipped = (s, h, src); told = s
        worst = s if worst is None else min(worst, s)
        tell(item, src, trial, told)

    launch(0.0, {"cfg": SPEC.default_config(), "ok": True, **rd["base"]}, "base", None)
    fill(0.0)
    dispatch(0.0)
    while heap:
        t, kind, _s, d = heapq.heappop(heap)
        if t > WINDOW_S:
            break
        if kind == 0:                               # propose call returned
            lane_busy = False
            if t < CUT_AT:
                buf.extend(d); st["llm_props"] += len(d)
            fill(t)
        elif kind == 1:                             # judge verdict
            lane_busy = False
            verdict(*d)
        elif kind == 3:                             # sampler ask returned: launch on the held slot
            held -= 1; free += 1
            if t < CUT_AT:
                trial = study.ask()
                cfg = SPEC.validate_config(SPEC.suggest_optuna(trial))
                s, h = sur.query(cfg, rng)
                st["samp_asked"] += 1; st["samp_trap"] += SPEC.is_trap(cfg)
                launch(t, {"cfg": cfg, "ok": True, "score": s, "heldout": h}, "samp", trial)
            fill(t)
        elif kind == 4:                             # cadence due
            wake_at = None
        elif kind == 2:                             # run finished
            item, src, trial, dur = d
            free += 1; busy += dur
            if src == "llm":
                llm_running -= 1
            if src in ("llm", "samp"):
                st["exp_llm" if src == "llm" else "exp_samp"] += 1
            if t >= CUT_AT:
                release.append(min(t, WINDOW_S))
            procq.append((item, src, trial))
            fill(t)
        dispatch(t)
    paid = sum(release) + max(0, SLOTS - len(release)) * CUT_AT
    best = max(counted, default=None)
    gaps = st.pop("gaps")
    return {**st, "interval": interval, "strategy": strategy, "race": race, "seed": seed, "gapsrc": gapsrc,
            "llm_usd": st["usd_propose"] + st["usd_judge"], "waiting": 1 - busy / paid,
            "experiments": st["exp_llm"] + st["exp_samp"],
            "best_heldout": best[0] if best else None, "best_src": best[2] if best else None,
            "shipped_heldout": shipped[1] if shipped else None, "shipped_src": shipped[2] if shipped else None,
            "samp_trap_share": st["samp_trap"] / max(1, st["samp_asked"]),
            "k_mean": st["k_sum"] / max(1, st["calls_propose"]), "gap_min": min(gaps) if gaps else None,
            "stream_exhausted": exhausted}


def _job(a: tuple[Any, ...]) -> dict[str, Any]:
    return replay(*a)


def agg(xs: list[dict[str, Any]], key: str) -> tuple[float, float]:
    v = [x[key] for x in xs if x[key] is not None]
    return statistics.fmean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    _init()
    data = _DATA
    lane = data["lane"]
    print(f"surrogate (RL only): {_SUR.n} runs ({_SUR.nt} trap), best held-out {max(_SUR.H):.1f}; lean lane: "
          f"{lane['n_prop']} calls, p_inv {lane['p_inv']:.3f}, K8 lat p50 {statistics.median(lane['lat_by_k'][8]):.2f}s "
          f"${statistics.fmean(lane['cost_by_k'][8]):.4f}/call; judge p50 {statistics.median(lane['jlat']):.2f}s "
          f"${statistics.fmean(lane['jcost']):.4f}")
    for r, rd in data["races"].items():
        print(r, "stream", len(rd["stream"]), "measured BM", rd["measured_bm"], "measured opus", rd["measured_opus"])
    settings: list[tuple[float | None, str, str]] = [(None, "first", "bm"), (None, "first", "model")]
    settings += [(i, s, "bm") for i in INTERVALS for s in ("first", "spread")]
    settings += [(i, "first", "optuna") for i in INTERVALS]
    jobs = [(i, s, r, sd, g) for i, s, g in settings for r in R67 for sd in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.procs, initializer=_init) as ex:
        runs = list(ex.map(_job, jobs))
    res: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "script": "scripts/sim_cadence_rl.py",
                           "label": "PROJECTION (offline replay, crude kNN surrogate for sampler runs)",
                           "seeds_per_race": args.seeds, "races": R67,
                           "measured": {r: {"bm": rd["measured_bm"], "opus": rd["measured_opus"]}
                                        for r, rd in data["races"].items()},
                           "surrogate": {"n": _SUR.n, "n_trap": _SUR.nt, "best": float(max(_SUR.H))},
                           "lane": {"p_inv": lane["p_inv"], "n_calls": lane["n_prop"]}, "settings": {}}
    keys = ("llm_usd", "usd_propose", "usd_judge", "calls_propose", "calls_judge", "judges_samp", "waiting",
            "experiments", "exp_llm", "exp_samp", "best_heldout", "shipped_heldout", "samp_trap_share", "k_mean",
            "stream_exhausted")
    for i, s, g in settings:
        xs = [x for x in runs if x["interval"] == i and x["strategy"] == s and x["gapsrc"] == g]
        name = f"{'none' if i is None else int(i)}/{s}/{g}"
        res["settings"][name] = {k: agg(xs, k) for k in keys}
        res["settings"][name]["best_src"] = {k: sum(1 for x in xs if x["best_src"] == k) for k in ("llm", "samp")}
        res["settings"][name]["gap_min"] = min((x["gap_min"] for x in xs if x["gap_min"] is not None), default=None)
        res["settings"][name]["runs"] = xs
    OUT.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    for name, sv in res["settings"].items():
        f = lambda k, m=1.0, p=2: f"{sv[k][0]*m:.{p}f}±{sv[k][1]*m:.{p}f}"  # noqa: E731
        print(f"{name:18s} LLM$ {f('llm_usd')} (prop {f('usd_propose')} judge {f('usd_judge')}) calls {f('calls_propose',p=0)} "
              f"judges {f('calls_judge',p=0)} (samp {f('judges_samp',p=0)}) wait% {f('waiting',100,1)} exps {f('experiments',p=0)} "
              f"(llm {f('exp_llm',p=0)} samp {f('exp_samp',p=0)}) best {f('best_heldout',p=1)} shipped {f('shipped_heldout',p=1)} "
              f"src {sv['best_src']} trap {f('samp_trap_share')} K {f('k_mean',p=1)} exh {f('stream_exhausted',p=0)}")
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
