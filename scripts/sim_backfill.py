"""Offline gate for 'backfill' ($0; no LLM, no Modal): fill free slots with Optuna TPE configs
whenever the LLM planner has no proposal ready.

Extends the R43/R52 event simulator (scripts/sim_planner_concurrency.py, validated in R52 at
67.5% vs 69.0% measured on RL wide) to three targets, each loaded from its own headline races in
data/bench/bench.db (read-only):

- ``rl``      R48 (30 CPU slots, 15 s runs, deadline rule 17 s, K <= 8)
- ``cifar``   R53 (3 L4, 45 s runs, no deadline rule: late runs are cancelled at the deadline)
- ``lcbench`` R44 (16 virtual L4 slots, surrogate-paced runs, no deadline rule)

Per target, from the **opus** arm: propose latency bootstrapped per K (nearest measured K, scaled
by (1 + K) / (1 + K0), the R50 shape), malformed-reply rate (a malformed call returns nothing
and the planner calls again), output tokens per proposal, the uncached digest size fit
(a + b x history rows + c x configs listed as 'still running'), judge latency / cost, and the
judge rate per finished run; run lengths of the Opus arm's runs (LLM proposals). From the
**optuna** arm: run lengths of its runs (sampler proposals) and its new-best rate per finished
run. Prices: ``backend.agent.llm_base.estimate_cost_usd`` (list price, as the races log).

Modes (one sequential planner lane, as ``ArmRunner._agent_loop``: process finished runs, a judge
call on every claimed new best, then one propose call):
- ``none``     today's arm. Validation: must reproduce each target's measured Opus waiting.
- ``fill``     backfill as specified: LLM demand counts only LLM runs/proposals
               (k = min(Kcap, N - LLM running - LLM buffered)); LLM proposals launch first; any
               slot still free with an empty LLM buffer gets a sampler config at once (TPE ask
               time modelled, growing with study size).
- ``reserve``  variant: the sampler leaves free the slots an in-flight LLM call will fill
               (keeps the LLM's share of runs; less waiting removed).

Judge calls: Bernoulli per finished run, by source: LLM runs at Opus's measured rate; sampler runs
at a scenario rate: ``naive`` = the Optuna arm's measured new-best rate per finished run (what the
gate asked for); ``replay`` = the claims per sampler run measured by the offline replays
(scripts/replay_backfill.py, data/backfill/replay_*.json) under the checker; ``checked`` = the
rate a perfect checker would see on the Optuna arm's own sequence (TPE told the inflated trap
scores; worst case).

Usage: PYTHONUTF8=1 python -m uv run python scripts/sim_backfill.py [--seeds 20]
Writes data/backfill/sim.json.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.agent.llm_base import OPUS_MODEL, estimate_cost_usd  # noqa: E402

BENCH_DB = REPO / "data" / "bench" / "bench.db"
OUT_DIR = REPO / "data" / "backfill"
WINDOW_S = 900.0
KCAP = 8

TARGETS: dict[str, dict[str, Any]] = {
    "rl": dict(races=("8a76fa05", "45ab180f", "59d7f7eb"), slots=30, cutoff=17.0, cap=8.0, measured_ref="R48"),
    "cifar": dict(races=("60477ba1", "949fce51", "c8261494"), slots=3, cutoff=None, cap=5.0, measured_ref="R53"),
    "lcbench": dict(races=("2ded10bc", "e3fddb29", "aaedd180"), slots=16, cutoff=None, cap=7.2, measured_ref="R44"),
}

GATE_WAIT_RL = 0.15
GATE_COST_X = 1.3


def ro() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{BENCH_DB.as_posix()}?mode=ro", uri=True)


def ols2(x1: list[float], x2: list[float], y: list[float]) -> tuple[float, float, float]:
    n = len(y)
    m1, m2, my = sum(x1) / n, sum(x2) / n, sum(y) / n
    s11 = sum((a - m1) ** 2 for a in x1); s22 = sum((b - m2) ** 2 for b in x2)
    s12 = sum((a - m1) * (b - m2) for a, b in zip(x1, x2))
    s1y = sum((a - m1) * (c - my) for a, c in zip(x1, y)); s2y = sum((b - m2) * (c - my) for b, c in zip(x2, y))
    det = s11 * s22 - s12 ** 2
    if abs(det) < 1e-9:   # one regressor constant: fit the other alone
        b2 = s2y / s22 if s22 else 0.0
        return my - b2 * m2, 0.0, b2
    b1 = (s1y * s22 - s2y * s12) / det
    b2 = (s2y * s11 - s1y * s12) / det
    return my - b1 * m1 - b2 * m2, b1, b2


@dataclass
class TargetData:
    name: str
    slots: int
    cutoff: float | None
    cap: float
    lat_by_k: dict[int, list[float]]
    p_invalid: float
    invalid_lat: list[float]
    out_per_prop: float
    in_fit: tuple[float, float, float]
    sys_tokens: int
    judge_lat: list[float]
    judge_cost: list[float]
    judge_rate_llm: float
    dur_llm: list[float]
    dur_samp: list[float]
    rate_samp_naive: float
    rate_samp_checked: float
    trap_share_optuna: float
    price: float
    measured: dict[str, Any] = field(default_factory=dict)


def load_target(name: str) -> TargetData:
    cfg = TARGETS[name]
    c = ro()
    lat_by_k: dict[int, list[float]] = {}
    invalid_lat: list[float] = []
    n_prop = n_inv = 0
    outpp: list[float] = []
    xr: list[float] = []; xl: list[float] = []; yi: list[float] = []
    sys_tok: list[int] = []
    jl: list[float] = []; jc: list[float] = []
    n_done_llm = n_judge = 0
    dur_llm: list[float] = []; dur_samp: list[float] = []
    n_samp = claimed_samp = checked_samp = trap_samp = 0
    measured: dict[str, Any] = {}
    price = 0.0
    for short in cfg["races"]:
        rid = c.execute("select id from races where id like ?", (short + "%",)).fetchone()[0]
        for arm_id, kind, summ, pr in c.execute("select id, kind, summary_json, price_per_s from arms where race_id=?",
                                                (rid,)).fetchall():
            if kind not in ("opus", "optuna"):
                continue
            price = pr
            exps = c.execute("select proposed_at, started_at, ended_at, status, score, is_trap, is_baseline, summary_json "
                             "from experiments where arm_id=? order by ended_at", (arm_id,)).fetchall()
            ran = [e for e in exps if e[1] is not None and e[2] is not None and e[3] in ("succeeded", "failed", "timed_out")]
            durs = [e[2] - e[1] for e in ran if not e[6]]
            s = json.loads(summ or "{}")
            measured.setdefault(kind, []).append({"race": short, "waiting_pct": s.get("gpu_idle_pct"),
                                                  "llm_usd": s.get("llm_cost_usd"),
                                                  "experiments": s.get("experiments_finished")})
            if kind == "optuna":
                dur_samp += durs
                succ = [e for e in ran if e[3] == "succeeded" and e[4] is not None]
                best = None
                for e in succ:
                    if e[6]:
                        best = e[4] if best is None or e[4] > best else best
                        continue
                    n_samp += 1
                    trap_samp += bool(e[5])
                    if json.loads(e[7] or "{}").get("claimed_new_best"):
                        claimed_samp += 1
                    if best is None or e[4] > best:     # every target here is 'max'
                        checked_samp += 1
                        if not e[5]:
                            best = e[4]
                continue
            dur_llm += durs
            ends = sorted(e[2] for e in ran if e[3] == "succeeded")
            n_done_llm += sum(1 for e in ran if e[3] == "succeeded" and not e[6])
            calls = c.execute("select step_type, created_at, latency_ms, input_tokens, cache_creation_tokens, "
                              "cache_read_tokens, output_tokens, experiment_ids_json, valid from llm_calls "
                              "where arm_id=? order by created_at", (arm_id,)).fetchall()
            for st, t, lat, inp, cw, cr, out, ids, valid in calls:
                if st == "judge_improvement":
                    n_judge += 1
                    jl.append(lat / 1000); jc.append(estimate_cost_usd(OPUS_MODEL, inp, cw, cr, out))
                    continue
                if st != "propose_experiments":
                    continue
                n_prop += 1
                k = len(json.loads(ids or "[]"))
                if not valid or k == 0:
                    n_inv += 1; invalid_lat.append(lat / 1000)
                    continue
                lat_by_k.setdefault(k, []).append(lat / 1000)
                outpp.append(out / k)
                sys_tok.append((cr or 0) + (cw or 0))
                n_fin = sum(1 for e in ends if e < t)
                xr.append(min(5, n_fin) + min(8, n_fin))
                xl.append(sum(1 for e in exps if e[0] is not None and e[0] < t and (e[2] is None or e[2] > t)))
                yi.append(inp)
    fit = ols2(xr, xl, yi)
    return TargetData(
        name=name, slots=cfg["slots"], cutoff=cfg["cutoff"], cap=cfg["cap"], lat_by_k=lat_by_k,
        p_invalid=n_inv / max(1, n_prop), invalid_lat=invalid_lat or [statistics.median(lat_by_k[min(lat_by_k)])],
        out_per_prop=statistics.fmean(outpp), in_fit=fit, sys_tokens=int(statistics.median(sys_tok)),
        judge_lat=jl, judge_cost=jc, judge_rate_llm=n_judge / max(1, n_done_llm),
        dur_llm=dur_llm, dur_samp=dur_samp,
        rate_samp_naive=claimed_samp / max(1, n_samp), rate_samp_checked=checked_samp / max(1, n_samp),
        trap_share_optuna=trap_samp / max(1, n_samp), price=price, measured=measured,
    )


def lat_sample(d: TargetData, rng: random.Random, k: int) -> float:
    ks = [kk for kk, v in d.lat_by_k.items() if len(v) >= 5] or list(d.lat_by_k)
    k0 = min(ks, key=lambda kk: (abs(kk - k), -kk))
    return rng.choice(d.lat_by_k[k0]) * (1 + k) / (1 + k0)


def ask_time(n_trials: int) -> float:
    """TPE ask() per trial: ~0.02 s at 100 trials, ~0.06 s at 600 (scheduler comment: propose(8)
    0.17 s at 100, ~0.5 s at 600+), linear growth beyond."""
    return 0.02 + 0.07 * n_trials / 600


def simulate(d: TargetData, *, mode: str, p_samp: float, seed: int, lat_mult: float = 1.0,
             judge_mult: float = 1.0, cap: bool = False, p_llm: float | None = None) -> dict[str, Any]:
    rng = random.Random(seed)
    N = d.slots
    cut_at = WINDOW_S - d.cutoff if d.cutoff is not None else WINDOW_S
    a_in, b_rows, b_run = d.in_fit
    p_llm = d.judge_rate_llm if p_llm is None else p_llm

    heap: list[tuple[float, int, int, Any]] = []
    seq = 0

    def push(t: float, kind: int, data: Any) -> None:
        nonlocal seq
        heapq.heappush(heap, (t, kind, seq, data)); seq += 1

    free = N
    running: dict[int, tuple[float, float, str]] = {}
    buf_llm: list[float] = []
    lane_busy = False
    inflight_k = 0
    judge_q: list[str] = []
    n_done = 0
    n_done_src = {"llm": 0, "samp": 0}
    launched_src = {"llm": 0, "samp": 0}
    busy = 0.0
    release_t: list[float] = []
    llm_usd = 0.0; usd_prop = 0.0; usd_judge = 0.0
    calls_p = calls_j = calls_inv = 0
    judges_src = {"llm": 0, "samp": 0}
    first_return: float | None = None
    rid = 0
    sampler_busy_until = 0.0
    n_trials = 0
    stopped_at: float | None = None
    buf_wait: list[float] = []

    def launch(t: float, dur: float, src: str) -> None:
        nonlocal free, rid
        free -= 1
        running[rid] = (t, dur, src)
        launched_src[src] += 1
        push(t + dur, 3, rid); rid += 1

    def llm_running() -> int:
        return sum(1 for v in running.values() if v[2] == "llm")

    def fill(t: float) -> None:
        nonlocal sampler_busy_until, n_trials
        while buf_llm and free > 0 and t < cut_at:
            buf_wait.append(t - buf_llm.pop(0))
            launch(t, rng.choice(d.dur_llm), "llm")
        if mode == "none" or t >= cut_at:
            return
        reserve = inflight_k if mode == "reserve" else 0
        while free - reserve > 0:
            # TPE ask runs in a worker thread, serially per arm
            t_ready = max(t, sampler_busy_until) + ask_time(n_trials)
            sampler_busy_until = t_ready
            n_trials += 1
            push(t_ready, 4, None)   # launch when the ask returns
            # reserve the slot now so the loop terminates
            free_slot_hold()

    held = 0

    def free_slot_hold() -> None:
        nonlocal free, held
        free -= 1; held += 1

    def dispatch(t: float) -> None:
        nonlocal lane_busy, calls_j, calls_p, inflight_k
        if lane_busy or t >= cut_at:
            return
        if judge_q:
            src = judge_q.pop(0)
            j = rng.randrange(len(d.judge_lat))
            lane_busy = True
            calls_j += 1; judges_src[src] += 1
            push(t + d.judge_lat[j] * lat_mult * judge_mult, 1, ("j", 0, d.judge_cost[j]))
            return
        if mode == "none":
            D = N - len(running) - len(buf_llm) - held
        else:
            D = N - llm_running() - len(buf_llm)
        k = min(KCAP, D)
        if k <= 0:
            return
        listed = len(running) + len(buf_llm)
        rows = min(5, n_done) + min(8, n_done)
        inp = max(0.0, a_in + b_rows * rows + b_run * listed)
        write = first_return is None
        cost = estimate_cost_usd(OPUS_MODEL, int(inp), d.sys_tokens if write else 0, 0 if write else d.sys_tokens,
                                 int(d.out_per_prop * k))
        lane_busy = True
        calls_p += 1
        if rng.random() < d.p_invalid:
            push(t + rng.choice(d.invalid_lat) * lat_mult, 0, ("p", 0, cost))
            inflight_k = 0
            return
        inflight_k = k
        push(t + lat_sample(d, rng, k) * lat_mult, 0, ("p", k, cost))

    # baseline on slot 0 at t0 (run_baseline_first): an LLM-arm run of the starting config
    launched_src["base"] = 0
    launch(0.0, rng.choice(d.dur_llm), "base")
    fill(0.0)
    dispatch(0.0)
    t_end = WINDOW_S
    while heap:
        t, kind, _s, data = heapq.heappop(heap)
        if t > WINDOW_S:
            break
        if kind in (0, 1):
            _typ, k, cost = data
            llm_usd += cost
            if kind == 0:
                usd_prop += cost
            else:
                usd_judge += cost
            lane_busy = False
            if first_return is None:
                first_return = t
            if kind == 0:
                if k == 0:
                    calls_inv += 1
                inflight_k = 0
                if t < cut_at:
                    for _ in range(k):
                        buf_llm.append(t)
                fill(t)
            if cap:
                paid_so_far = sum(release_t) + (N - len(release_t)) * min(t, cut_at)
                if llm_usd + paid_so_far * d.price >= d.cap:
                    stopped_at = t; t_end = t
                    break
            dispatch(t)
        elif kind == 4:
            held -= 1; free += 1
            if t < cut_at:
                launch(t, rng.choice(d.dur_samp), "samp")
            fill(t)
            dispatch(t)
        elif kind == 3:
            st, dur, src = running.pop(data)
            busy += dur
            free += 1
            n_done += 1
            s2 = "llm" if src == "base" else src
            n_done_src[s2] += 1
            if t >= cut_at:
                release_t.append(min(t, WINDOW_S))
            if src != "base" and rng.random() < (p_llm if src == "llm" else p_samp):
                judge_q.append(src)
            fill(t)
            dispatch(t)
    for st, _dur, _src in running.values():
        busy += max(0.0, t_end - st)
        release_t.append(t_end)
    if d.cutoff is None:
        paid = N * t_end
    elif stopped_at is not None and t_end < cut_at:
        paid = N * t_end
    else:
        paid = sum(release_t) + max(0, N - len(release_t)) * cut_at
    exps = n_done
    return {
        "waiting": 1.0 - busy / paid, "experiments": exps, "exp_llm": n_done_src["llm"], "exp_samp": n_done_src["samp"],
        "llm_share_launched": (launched_src["llm"]) / max(1, launched_src["llm"] + launched_src["samp"]),
        "calls_propose": calls_p, "calls_invalid": calls_inv, "calls_judge": calls_j,
        "judges_llm": judges_src["llm"], "judges_samp": judges_src["samp"],
        "llm_usd": llm_usd, "usd_propose": usd_prop, "usd_judge": usd_judge,
        "llm_props": launched_src["llm"], "buf_wait_p50": statistics.median(buf_wait) if buf_wait else 0.0,
        "capped": stopped_at is not None, "gpu_usd": paid * d.price,
    }


def summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in runs[0]:
        if k == "capped":
            out["capped_share"] = sum(1 for r in runs if r[k]) / len(runs)
            continue
        vals = [r[k] for r in runs]
        out[k] = statistics.fmean(vals)
        if k in ("waiting", "llm_usd", "experiments", "calls_judge"):
            out[k + "_sd"] = statistics.pstdev(vals)
            out[k + "_min"], out[k + "_max"] = min(vals), max(vals)
    return out


def load_replay_rates() -> dict[str, dict[str, float]]:
    """Claims per sampler run measured by scripts/replay_backfill.py (if it has been run)."""
    out: dict[str, dict[str, float]] = {}
    for tgt in ("rl", "cifar", "lcbench"):
        p = OUT_DIR / f"replay_{tgt}.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            out[tgt] = r.get("sim_rates", {})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()
    replay = load_replay_rates()
    res: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "script": "scripts/sim_backfill.py",
                           "seeds": args.seeds, "targets": {}}
    for name in TARGETS:
        d = load_target(name)
        meas = d.measured["opus"]
        mw = statistics.fmean(m["waiting_pct"] for m in meas)
        mu = statistics.fmean(m["llm_usd"] for m in meas)
        me = statistics.fmean(m["experiments"] for m in meas)
        info = {
            "lat_k": {k: [len(v), round(statistics.median(v), 2)] for k, v in sorted(d.lat_by_k.items())},
            "p_invalid": d.p_invalid, "out_per_prop": d.out_per_prop, "in_fit": d.in_fit, "sys_tokens": d.sys_tokens,
            "judge_p50": statistics.median(d.judge_lat), "judge_cost_mean": statistics.fmean(d.judge_cost),
            "judge_rate_llm": d.judge_rate_llm, "dur_llm_p50": statistics.median(d.dur_llm),
            "dur_samp_p50": statistics.median(d.dur_samp), "n_dur_llm": len(d.dur_llm), "n_dur_samp": len(d.dur_samp),
            "rate_samp_naive": d.rate_samp_naive, "rate_samp_checked": d.rate_samp_checked,
            "trap_share_optuna": d.trap_share_optuna,
            "measured_opus": {"waiting_pct": mw, "llm_usd": mu, "experiments": me, "per_race": meas},
            "measured_optuna": d.measured.get("optuna"),
        }
        print(f"\n== {name}: measured opus waiting {mw:.1f}% LLM ${mu:.2f} exps {me:.0f}; "
              f"lat by K {info['lat_k']}; p_inv {d.p_invalid:.3f}; out/prop {d.out_per_prop:.0f}; "
              f"in fit {tuple(round(x, 1) for x in d.in_fit)}; judge p50 {info['judge_p50']:.2f}s "
              f"${info['judge_cost_mean']:.4f}; judge/LLM run {d.judge_rate_llm:.4f}; samp claim naive "
              f"{d.rate_samp_naive:.4f} checked {d.rate_samp_checked:.4f} (optuna trap share {d.trap_share_optuna:.2f}); "
              f"dur p50 llm {info['dur_llm_p50']:.1f}s samp {info['dur_samp_p50']:.1f}s")
        rates = {"naive": d.rate_samp_naive, "checked": d.rate_samp_checked}
        for k, v in replay.get(name, {}).items():
            rates[f"replay_{k}"] = v
        settings: dict[str, Any] = {}
        base = summary([simulate(d, mode="none", p_samp=0.0, seed=s) for s in range(args.seeds)])
        settings["none"] = base
        for mode in ("fill", "reserve"):
            for rn, rv in rates.items():
                s = summary([simulate(d, mode=mode, p_samp=rv, seed=s) for s in range(args.seeds)])
                sc = summary([simulate(d, mode=mode, p_samp=rv, seed=s, cap=True) for s in range(args.seeds)])
                s["with_cap"] = {k: sc[k] for k in ("waiting", "experiments", "llm_usd", "capped_share")}
                settings[f"{mode}/{rn}"] = s
        # sensitivities on fill/naive and fill/replay_*: latency x1.25, judge latency x2
        for rn in [r for r in rates if r == "naive" or r.startswith("replay")]:
            settings[f"fill/{rn}/lat x1.25"] = summary([simulate(d, mode="fill", p_samp=rates[rn], seed=s, lat_mult=1.25)
                                                        for s in range(args.seeds)])
        res["targets"][name] = {"inputs": info, "rates": rates, "settings": settings,
                                "validated": abs(base["waiting"] * 100 - mw) <= 3.0}
        print(f"validation (mode none): sim {base['waiting']*100:.1f}% [{base['waiting_min']*100:.1f}-"
              f"{base['waiting_max']*100:.1f}] vs measured {mw:.1f}%; sim LLM ${base['llm_usd']:.2f} vs ${mu:.2f}; "
              f"sim exps {base['experiments']:.0f} vs {me:.0f}")
        print("| setting | waiting % | exps (LLM/sampler) | LLM share | propose | judge (LLM/samp) | LLM $ | x none | "
              "LLM $ cap-on | capped |")
        for k, s in settings.items():
            wc = s.get("with_cap", {})
            print(f"| {k} | {s['waiting']*100:.1f} ± {s['waiting_sd']*100:.1f} | {s['experiments']:.0f} "
                  f"({s['exp_llm']:.0f}/{s['exp_samp']:.0f}) | {s['llm_share_launched']*100:.0f}% | {s['calls_propose']:.0f} | "
                  f"{s['calls_judge']:.0f} ({s['judges_llm']:.0f}/{s['judges_samp']:.0f}) | {s['llm_usd']:.2f} | "
                  f"{s['llm_usd']/base['llm_usd']:.2f}x | {wc.get('llm_usd', float('nan')):.2f} | "
                  f"{wc.get('capped_share', 0)*100:.0f}% |")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sim.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'sim.json'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
