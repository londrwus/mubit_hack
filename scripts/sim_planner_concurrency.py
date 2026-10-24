"""Offline gate: would C concurrent propose calls (an engine feature) cut GPUs waiting? ($0; no LLM, no Modal)

Extends the R43 event simulator (scripts/gate_early_stop_rules.py) of an RL wide race
(30 slots, 15 s runs, 900 s, deadline rule 17 s, K <= 8 per call, baseline on slot 0 at t0) with:
- C researcher "lanes" that may each have one LLM call in flight (C = 1 is today's sequential
  researcher in ``backend/scheduler.py``: process finished runs (judge on a claimed new best),
  then one propose call for K = min(8, N + lookahead - running - buffered));
- unclaimed demand D = N + lookahead - running - buffered - (K already requested by in-flight
  calls); a free lane asks for k = min(8, ceil(D / free lanes)) ("share") or min(8, D) ("greedy");
- pipelining (lookahead = N, as ``cfg.pipelining``): proposals wait in a buffer for a free slot;
- judge calls as measured in R48 (Bernoulli per finished run, latency/tokens bootstrapped); a judge
  occupies a lane (with C = 1 it blocks the next proposal, as in the scheduler);
- LLM $ per call from R48's measured tokens, priced by ``backend.agent.llm_base.estimate_cost_usd``:
  cached system + spec block (cache write while no call has returned yet, cache read after),
  uncached digest = fit on R48 (history rows + configs listed as 'still running', i.e. running +
  buffered), output = R48 output tokens per proposal x k;
- optional per-arm $ cap (rl-wide-15m: $8 GPU + LLM): the arm stops when an LLM call lands it there.

Run lengths: the R43 rl15 pool (R41 race f2bee077, opus arm, never early-stopped), loaded with the
gate script's own loader. Validation: C = 1, no pipelining must reproduce R48's measured Opus
waiting (69.0 +- 0.4%, range 68.6-69.3%) within ~3 points, and its LLM $ ($3.27-3.40 per seed).

Policies: share (k = ceil(D / free lanes)), greedy (k = min(8, D)), batchN (call only when D >= N,
unless no propose call is in flight). Gate: some setting with C <= 4 projects GPUs waiting <= 25% AND LLM $ per race <= 1.35 x the C = 1
Opus arm (simulated), with the simulator validated.

Usage: PYTHONUTF8=1 python -m uv run python scripts/sim_planner_concurrency.py [--seeds 20]
Reads data/bench/bench.db read-only; writes data/planner_concurrency/results.json.
"""

from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import math
import random
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.agent.llm_base import OPUS_MODEL, estimate_cost_usd  # noqa: E402

_spec = importlib.util.spec_from_file_location("gate_es", REPO / "scripts" / "gate_early_stop_rules.py")
gate = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec and _spec.loader
sys.modules["gate_es"] = gate
_spec.loader.exec_module(gate)

OUT_DIR = REPO / "data" / "planner_concurrency"
BENCH_DB = REPO / "data" / "bench" / "bench.db"
R48 = ("8a76fa052c28441cb2aea221f099a7d2", "45ab180f9fdc403e93f6f16960e6b0f9",
       "59d7f7ebe0bc4b6ba59edacd0d769f49")

SLOTS = 30
WINDOW_S = 900.0
CUTOFF_S = 17.0          # rl-wide-15m launch_cutoff_s
KCAP = 8                 # rl-wide-15m max_proposals_per_call
COST_CAP = 8.0           # rl-wide-15m cost_cap_usd (GPU $ + LLM $ per arm)
GATE_WAIT = 0.25
GATE_COST_X = 1.35
GATE_MAX_C = 4
CS = (1, 2, 3, 4, 6, 8)


def ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def ols2(x1: list[float], x2: list[float], y: list[float]) -> tuple[float, float, float]:
    """y = a + b1 x1 + b2 x2 (closed form)."""
    n = len(y)
    m1, m2, my = sum(x1) / n, sum(x2) / n, sum(y) / n
    s11 = sum((a - m1) ** 2 for a in x1); s22 = sum((b - m2) ** 2 for b in x2)
    s12 = sum((a - m1) * (b - m2) for a, b in zip(x1, x2))
    s1y = sum((a - m1) * (c - my) for a, c in zip(x1, y)); s2y = sum((b - m2) * (c - my) for b, c in zip(x2, y))
    det = s11 * s22 - s12 ** 2
    b1 = (s1y * s22 - s2y * s12) / det
    b2 = (s2y * s11 - s1y * s12) / det
    return my - b1 * m1 - b2 * m2, b1, b2


def load_r48() -> dict[str, Any]:
    c = ro(BENCH_DB)
    lat8: list[float] = []
    out_per_prop: list[float] = []
    x_rows: list[float] = []; x_run: list[float] = []; y_in: list[float] = []
    judge_lat: list[float] = []; judge_cost: list[float] = []
    n_done_total = 0; n_judge_total = 0
    per_race: dict[str, Any] = {}
    for race in R48:
        arm_id, a_start, summ = c.execute("select id, started_at, summary_json from arms where race_id=? and kind='opus'",
                                          (race,)).fetchone()
        exps = c.execute("select proposed_at, started_at, ended_at, status from experiments where arm_id=?",
                         (arm_id,)).fetchall()
        calls = c.execute("select step_type, created_at, latency_ms, input_tokens, cache_creation_tokens, "
                          "cache_read_tokens, output_tokens, experiment_ids_json, valid from llm_calls "
                          "where arm_id=? order by created_at", (arm_id,)).fetchall()
        ends = sorted(e[2] for e in exps if e[3] == "succeeded" and e[2] is not None)
        done = len(ends)
        n_done_total += done
        nj = 0
        for st, t, lat, inp, cw, cr, out, ids, valid in calls:
            if st == "judge_improvement":
                nj += 1
                judge_lat.append(lat / 1000)
                judge_cost.append(estimate_cost_usd(OPUS_MODEL, inp, cw, cr, out))
                continue
            if st != "propose_experiments" or not valid:
                continue
            k = len(json.loads(ids or "[]"))
            if k == KCAP:
                lat8.append(lat / 1000)
            if k:
                out_per_prop.append(out / k)
            n_fin = sum(1 for e in ends if e < t)
            rows = min(5, n_fin) + min(8, n_fin)
            listed = sum(1 for e in exps if e[0] is not None and e[0] < t and (e[2] is None or e[2] > t))
            x_rows.append(rows); x_run.append(listed); y_in.append(inp)
        n_judge_total += nj
        s = json.loads(summ)
        per_race[race[:8]] = {"waiting_pct": s["gpu_idle_pct"], "llm_usd": s["llm_cost_usd"],
                              "experiments": s["experiments_finished"], "gpu_usd": s["gpu_cost_usd"],
                              "judges": nj, "finished": done}
    price = c.execute("select price_per_s from arms where race_id=? and kind='opus'", (R48[0],)).fetchone()[0]
    sys_tok = statistics.median([r for r in (c.execute(
        "select cache_read_tokens + cache_creation_tokens from llm_calls where race_id in (?,?,?) and arm='opus' "
        "and step_type='propose_experiments'", R48).fetchall()) for r in r])
    a, b_rows, b_run = ols2(x_rows, x_run, y_in)
    resid = [y - (a + b_rows * r + b_run * n) for r, n, y in zip(x_rows, x_run, y_in)]
    return {
        "lat8": lat8, "out_per_prop": statistics.fmean(out_per_prop),
        "in_fit": (a, b_rows, b_run), "in_fit_resid_sd": statistics.pstdev(resid), "in_listed_max": max(x_run),
        "judge_lat": judge_lat, "judge_cost": judge_cost, "judge_rate": n_judge_total / n_done_total,
        "price_per_slot_s": price, "sys_tokens": int(sys_tok), "per_race": per_race,
    }


def latency_models(d: dict[str, Any]) -> dict[str, Callable[[random.Random, int], float]]:
    lat8 = d["lat8"]
    return {
        # R48 measured K=8 latencies, scaled to K with the R50 shape (intercept = slope: L ~ 1 + K)
        "r48": lambda rng, k: rng.choice(lat8) * (1 + k) / (1 + KCAP),
        "r50": lambda rng, k: 1.62 + 1.62 * k,
        "r36": lambda rng, k: 2.48 + 1.65 * k,
    }


def simulate(pool: list[Any], d: dict[str, Any], *, C: int, pipelining: bool, policy: str, lat: str,
             seed: int, cap: float | None = None, lat_mult: float = 1.0, judge_rate: float | None = None
             ) -> dict[str, Any]:
    rng = random.Random(seed)
    lat_fn = latency_models(d)[lat]
    a_in, b_rows, b_run = d["in_fit"]
    out_pp, sys_tok, price = d["out_per_prop"], d["sys_tokens"], d["price_per_slot_s"]
    p_judge = d["judge_rate"] if judge_rate is None else judge_rate
    lookahead = SLOTS if pipelining else 0
    cut_at = WINDOW_S - CUTOFF_S

    heap: list[tuple[float, int, int, Any]] = []
    seq = 0

    def push(t: float, kind: int, data: Any) -> None:
        nonlocal seq
        heapq.heappush(heap, (t, kind, seq, data)); seq += 1

    free_slots = SLOTS
    running: dict[int, tuple[float, float]] = {}   # rid -> (start, dur)
    buffer: list[float] = []                      # durations of proposed, not launched
    free_lanes = C
    inflight: dict[int, int] = {}                 # call id -> k (propose only)
    judge_q = 0
    n_done = 0
    busy = 0.0
    release_t: list[float] = []
    llm_usd = 0.0
    calls_p = calls_j = 0
    proposed = 0
    stale_props = 0; stale_calls = 0; blind_sum = 0
    ks: list[int] = []
    first_return: float | None = None
    stopped_at: float | None = None
    rid_seq = 0; cid_seq = 0
    dropped = 0

    def launch(t: float, dur: float) -> None:
        nonlocal free_slots, rid_seq
        free_slots -= 1
        running[rid_seq] = (t, dur)
        push(t + dur, 3, rid_seq); rid_seq += 1

    def fill(t: float) -> None:
        while buffer and free_slots > 0 and t < cut_at:
            launch(t, buffer.pop(0))

    def dispatch(t: float) -> None:
        nonlocal free_lanes, judge_q, calls_p, calls_j, stale_props, stale_calls, blind_sum, cid_seq, proposed
        while free_lanes > 0 and t < cut_at:
            if judge_q > 0:
                judge_q -= 1
                free_lanes -= 1
                j = rng.randrange(len(d["judge_lat"]))
                push(t + d["judge_lat"][j] * lat_mult, 1, ("j", None, d["judge_cost"][j]))
                calls_j += 1
                continue
            D = SLOTS + lookahead - len(running) - len(buffer) - sum(inflight.values())
            if D <= 0:
                return
            if policy.startswith("batch"):
                # min-batch lane: call only for >= kmin proposals, unless no propose call is in flight
                kmin = int(policy[5:])
                if D < kmin and inflight:
                    return
                k = min(KCAP, D)
            else:
                k = min(KCAP, D) if policy == "greedy" else min(KCAP, math.ceil(D / free_lanes))
            free_lanes -= 1
            others = len(inflight)
            if others:
                stale_calls += 1; stale_props += k; blind_sum += sum(inflight.values())
            listed = len(running) + len(buffer)
            rows = min(5, n_done) + min(8, n_done)
            inp = max(0.0, a_in + b_rows * rows + b_run * listed)
            write = first_return is None
            cost = estimate_cost_usd(OPUS_MODEL, int(inp), sys_tok if write else 0, 0 if write else sys_tok,
                                     int(out_pp * k))
            cid = cid_seq; cid_seq += 1
            inflight[cid] = k
            ks.append(k)
            calls_p += 1
            push(t + lat_fn(rng, k) * lat_mult, 0, ("p", cid, cost))

    # baseline on slot 0 at t0 (run_baseline_first), then the first call(s)
    launch(0.0, rng.choice(pool).dur)
    dispatch(0.0)
    t_end = WINDOW_S
    while heap:
        t, kind, _s, data = heapq.heappop(heap)
        if t > WINDOW_S:
            break
        if kind in (0, 1):
            _typ, cid, cost = data
            llm_usd += cost
            free_lanes += 1
            if first_return is None:
                first_return = t
            if kind == 0:
                k = inflight.pop(cid)
                if t < cut_at:
                    for _ in range(k):
                        buffer.append(rng.choice(pool).dur)
                    proposed += k
                    fill(t)
                else:
                    dropped += k
            if cap is not None:
                paid_so_far = sum(release_t) + (SLOTS - len(release_t)) * min(t, cut_at)
                if llm_usd + paid_so_far * price >= cap:
                    stopped_at = t
                    t_end = t
                    break
            dispatch(t)
        elif kind == 3:
            st, dur = running.pop(data)
            busy += dur
            free_slots += 1
            n_done += 1
            if t >= cut_at:
                release_t.append(min(max(t, cut_at), WINDOW_S))
            if rng.random() < p_judge:
                judge_q += 1
            fill(t)
            dispatch(t)
    # in flight at the deadline (or at the cap stop): busy until then, not counted
    for st, _dur in running.values():
        busy += t_end - st
        release_t.append(t_end)
    if stopped_at is not None and t_end < cut_at:
        paid = SLOTS * t_end          # every slot pays until the cap stop
    else:
        paid = sum(release_t) + max(0, SLOTS - len(release_t)) * cut_at
    gpu_usd = paid * price
    return {
        "waiting": 1.0 - busy / paid, "experiments": n_done, "proposed": proposed,
        "props_per_min": proposed / (WINDOW_S / 60), "calls_propose": calls_p, "calls_judge": calls_j,
        "mean_k": statistics.fmean(ks) if ks else 0.0, "llm_usd": llm_usd, "gpu_usd": gpu_usd,
        "stale_props": stale_props, "stale_share": stale_props / max(1, sum(ks)),
        "blind_per_stale_call": blind_sum / stale_calls if stale_calls else 0.0,
        "dropped_at_cutoff": dropped, "capped": stopped_at is not None, "stopped_at": stopped_at,
        "llm_usd_per_exp": llm_usd / max(1, n_done),
    }


def summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in runs[0]:
        vals = [r[k] for r in runs if r[k] is not None and not isinstance(r[k], bool)]
        if k == "capped":
            out["capped_share"] = sum(1 for r in runs if r[k]) / len(runs)
        elif vals:
            out[k] = statistics.fmean(vals)
            if k in ("waiting", "llm_usd", "experiments"):
                out[k + "_sd"] = statistics.pstdev(vals)
                out[k + "_min"], out[k + "_max"] = min(vals), max(vals)
    return out


def run_setting(pool: list[Any], d: dict[str, Any], seeds: int, **kw: Any) -> dict[str, Any]:
    return summary([simulate(pool, d, seed=s, **kw) for s in range(seeds)])


def fmt_row(name: str, s: dict[str, Any], base_usd: float, base_exp: float) -> str:
    return (f"| {name} | {s['waiting']*100:.1f} ± {s['waiting_sd']*100:.1f} | {s['experiments']:.0f} | "
            f"{s['props_per_min']:.1f} | {s['mean_k']:.2f} | {s['calls_propose']:.0f} | {s['calls_judge']:.0f} | "
            f"{s['llm_usd']:.2f} | {s['llm_usd']/base_usd:.2f}x | {s['llm_usd_per_exp']*1000:.2f} | "
            f"{(s['llm_usd_per_exp'])/(base_usd/base_exp):.2f}x | {s['stale_share']*100:.0f}% | "
            f"{s['blind_per_stale_call']:.1f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()
    d = load_r48()
    runs = gate.load_regime("rl15", gate.REGIMES["rl15"])
    pool = [r for r in runs if r.candidate]
    # sensitivity pool: R48's own Opus runs (the races being reproduced)
    r48cfg = dict(gate.REGIMES["rl15"]); r48cfg["races"] = R48
    pool48 = [r for r in gate.load_regime("rl15_r48", r48cfg) if r.candidate]
    durs = sorted(r.dur for r in pool)
    print(f"pool R41: {len(pool)} runs, dur p50 {gate.quantile(durs, .5):.2f} s p90 {gate.quantile(durs, .9):.2f} s; "
          f"pool R48: {len(pool48)}")
    a, br, bn = d["in_fit"]
    print(f"R48: {len(d['lat8'])} K=8 propose calls, p50 {statistics.median(d['lat8']):.2f} s; out/proposal "
          f"{d['out_per_prop']:.0f} tok; uncached input = {a:.0f} + {br:.1f} x rows + {bn:.1f} x listed "
          f"(resid sd {d['in_fit_resid_sd']:.0f}, listed max {d['in_listed_max']}); judge rate {d['judge_rate']:.4f}/run, "
          f"p50 {statistics.median(d['judge_lat']):.2f} s, ${statistics.fmean(d['judge_cost']):.4f}/call; sys tokens {d['sys_tokens']}")
    res: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "script": "scripts/sim_planner_concurrency.py",
                           "params": {"slots": SLOTS, "window_s": WINDOW_S, "cutoff_s": CUTOFF_S, "kcap": KCAP,
                                      "seeds": args.seeds, "cost_cap": COST_CAP},
                           "inputs": {k: v for k, v in d.items() if k not in ("lat8", "judge_lat", "judge_cost")} |
                                     {"lat8_p50": statistics.median(d["lat8"]), "n_lat8": len(d["lat8"]),
                                      "judge_lat_p50": statistics.median(d["judge_lat"]),
                                      "judge_cost_mean": statistics.fmean(d["judge_cost"]),
                                      "pool_r41": len(pool), "pool_r48": len(pool48)},
                           "validation": {}, "grid": {}, "sensitivity": {}}

    # ---- validation (C = 1, no pipelining)
    for lat in ("r48", "r50", "r36"):
        for pname, pl in (("r41", pool), ("r48", pool48)):
            s = run_setting(pl, d, args.seeds, C=1, pipelining=False, policy="share", lat=lat)
            res["validation"][f"{lat}/{pname}"] = s
            print(f"validate C=1 lat={lat} pool={pname}: waiting {s['waiting']*100:.1f}% "
                  f"[{s['waiting_min']*100:.1f}-{s['waiting_max']*100:.1f}], exps {s['experiments']:.0f}, "
                  f"LLM ${s['llm_usd']:.2f}, calls {s['calls_propose']:.0f}+{s['calls_judge']:.0f}")
    base = res["validation"]["r48/r41"]
    meas_wait = statistics.fmean(v["waiting_pct"] for v in d["per_race"].values())
    validated = abs(base["waiting"] * 100 - meas_wait) <= 3.0
    res["validation"]["measured"] = d["per_race"]
    res["validation"]["validated"] = validated
    print(f"measured R48 waiting {meas_wait:.1f}% -> validated {validated}")

    # ---- grid (primary: r48 latency, R41 pool, no cap)
    base_usd, base_exp = base["llm_usd"], base["experiments"]
    hdr = ("| setting | GPUs waiting % | experiments | proposals/min | mean K | propose calls | judge calls | "
           "LLM $/race | vs C=1 | LLM m$/exp | $/exp vs C=1 | stale proposals | blind configs/stale call |")
    print(hdr)
    for pipe in (False, True):
        for policy in ("share", "greedy", "batch4", "batch8"):
            for C in CS:
                if C == 1 and policy != "share":
                    continue
                name = f"C={C} pipe={'on' if pipe else 'off'} {policy if C > 1 else ''}".strip()
                s = run_setting(pool, d, args.seeds, C=C, pipelining=pipe, policy=policy, lat="r48")
                sc = run_setting(pool, d, args.seeds, C=C, pipelining=pipe, policy=policy, lat="r48", cap=COST_CAP)
                s["with_cap"] = {k: sc[k] for k in ("waiting", "experiments", "llm_usd", "capped_share", "stopped_at")
                                 if k in sc}
                s["cost_x"] = s["llm_usd"] / base_usd
                s["per_exp_x"] = s["llm_usd_per_exp"] / (base_usd / base_exp)
                s["gate"] = {"wait_ok": s["waiting"] <= GATE_WAIT, "cost_ok": s["cost_x"] <= GATE_COST_X,
                             "c_ok": C <= GATE_MAX_C}
                s["gate"]["pass"] = validated and all(s["gate"].values())
                res["grid"][name] = s
                print(fmt_row(name, s, base_usd, base_exp) +
                      f" cap: wait {sc['waiting']*100:.1f}% exps {sc['experiments']:.0f} LLM ${sc['llm_usd']:.2f} "
                      f"capped {sc['capped_share']*100:.0f}%")

    # ---- sensitivities on the candidate settings
    for C in (1, 2, 3, 4):
        for pipe in (False, True):
            for pol in (("share",) if C == 1 else ("share", "batch8")):
                for lat in ("r50", "r36"):
                    s = run_setting(pool, d, args.seeds, C=C, pipelining=pipe, policy=pol, lat=lat)
                    res["sensitivity"][f"C={C} pipe={pipe} {pol} lat={lat}"] = s
                s = run_setting(pool, d, args.seeds, C=C, pipelining=pipe, policy=pol, lat="r48",
                                lat_mult=1.25 if C > 1 else 1.0)
                res["sensitivity"][f"C={C} pipe={pipe} {pol} lat=r48 x1.25 when C>1"] = s
                s = run_setting(pool48, d, args.seeds, C=C, pipelining=pipe, policy=pol, lat="r48")
                res["sensitivity"][f"C={C} pipe={pipe} {pol} pool=r48"] = s
    for k, s in res["sensitivity"].items():
        print(f"sens {k}: waiting {s['waiting']*100:.1f}%, exps {s['experiments']:.0f}, LLM ${s['llm_usd']:.2f}")

    # lower bound: output tokens alone for the experiments that <= 25% waiting needs
    exps_needed = base_exp * (1 - GATE_WAIT) / (1 - base["waiting"])
    out_only = exps_needed * d["out_per_prop"] * estimate_cost_usd(OPUS_MODEL, 0, 0, 0, 1_000_000) / 1e6
    res["bound"] = {"experiments_needed_at_25pct": exps_needed, "output_only_usd": out_only,
                    "output_only_x": out_only / base_usd, "gate_usd": GATE_COST_X * base_usd}
    print("bound:", res["bound"])
    passing = [(k, v) for k, v in res["grid"].items() if v["gate"]["pass"]]
    res["gate"] = {"pass": bool(passing), "passing": [k for k, _ in passing], "validated": validated}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"gate pass: {bool(passing)} {[k for k, _ in passing]}; wrote {OUT_DIR / 'results.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
