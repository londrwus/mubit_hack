"""Offline gate: a 'wide regime' on LCBench where the standard Opus agent leaves GPUs waiting
($0: no LLM, no Modal, DBs opened read-only). PROJECTION + REPLAY only, never a result.

Question (user, 24 Oct): is there a fair, pre-registered LCBench setting where the standard Opus
5.5 agent leaves GPUs waiting a lot while Busy Machines (D46: Opus + lean planner v2/effort low +
backfill sampler with checker-gated feedback + judge policy) is at ~0% waiting, BM's LLM bill
per race is LOWER than Opus's, and BM's best model is at least as good?

Knobs varied (both arms always identical): virtual L4 slots per arm N in {16, 32, 64} and the
SurrogateRunner multiplier m in {1, 0.5, 0.25, 0.1} (``BUSY_SURROGATE_RUNTIME_MULT``: a run
holds its slot for predicted train time x m). The config -> score map does NOT depend on m
(the 300 s per-run budget is in predicted seconds), so only timing changes.

One event engine (``Race``) serves every arm, so waiting, LLM $ and quality come from the same
replay:
- Opus's call timing is SIMULATED (latency bootstrapped from the measured LCBench calls of the
  arm's prompt: K=1 pool x a linear-in-K shape fitted on all calls; invalid replies at the
  measured rate), the CONTENT of its proposals is the measured proposal sequence of a real race
  (R38/R44/R65/R63 v1 for the standard agent, R63 lean for BM), in order, K per call. The
  counterfactual cannot re-run Opus, so Opus does not react to what it would have seen.
  If a regime asks for more proposals than the sequence holds, the extra ones are 'phantom'
  runs (duration drawn from the sequence, never scored, never counted): waiting and $ stay
  honest, quality is the sequence's (flagged as 'exhausted').
- Every run is scored by the deterministic surrogate (workers/targets/lcbench.py) with its
  exact duration x m; TimeoutError runs hold the slot for budget x m (SurrogateRunner).
- Checker (D40, as replay_backfill.py): runs processed in end order; a run beating the checked
  best is a claim -> one judge call on the arm's single sequential lane (judge before propose,
  as ArmRunner._agent_loop); trap claims are rejected (assumption: judge v1 17/17, R54), others
  accepted (accept_unsure; LCBench rule sigma=0).
- BM backfill: Optuna TPE (seed s, n_startup 10, constant_liar), asks serial with the modelled
  ask time (sim_backfill.ask_time), holding the slot; LLM proposals launch first; the study is
  told every finished run (rejected claims told the worst score so far = checker-gated).
- LLM $: estimate_cost_usd at list price; input = a + b*rows + c*listed (fit per prompt on the
  measured calls; 'listed' = configs still running/pending, which the digest lists in full,
  sampler runs included), system block cached, output = measured tokens per proposal x K.

BM cadence policies (the BM planner may call Opus less often because the sampler keeps GPUs busy):
  none          call whenever the LLM wants slots (demand counts only LLM runs, as scheduler.py)
  intT          at least T s between propose-call starts (K = min(8, demand))
  share25b8     LLM demand capped at ceil(0.25 N) slots and a call only when >= min(8, cap) wanted

PRE-REGISTERED SELECTION RULES (written before the grid was run):
  BM policy per regime: the policy with the MOST Opus proposals per race such that (a) projected
    BM LLM $/race <= 0.70 x the Opus arm's (margin under the 0.8 line) and (b) its Opus proposals
    fit the shortest lean sequence (no phantoms), so the quality replay stays exact. If no policy
    meets (b), the one with the fewest phantoms among those meeting (a).
  Gate (task): projected Opus waiting >= 40%, BM waiting <= 2%, BM LLM $/race <= 0.8 x Opus, replay
    BM best held-out >= Opus best (mean over replays) and >= Optuna (mean).
  Feasibility (added, real system): BM sampler ask time <= 50% of the race (one CPU thread, three
    arms share the process) and Optuna's likewise.
  Regime choice if several pass: lowest realism cost r = |log2 m| + 0.5 |log2(N/16)| (shorter
    runs change what the surrogate stands for; more slots is only a bigger pool), tie -> fewer
    slots. Never the most flattering.

Usage (LCBench needs onnxruntime):
  PYTHONUTF8=1 python -m uv run --with onnxruntime==1.29.0 python scripts/gate_wide_regime.py all
  subcommands: ceiling | opus | bm | replay | all     (--workers 7)
Writes data/wide_regime/*.json.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import multiprocessing as mp
import random
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "data" / "wide_regime"
DBS = {"bench": REPO / "data" / "bench" / "bench.db", "ablation": REPO / "data" / "bench" / "ablation.db"}
WINDOW_S = 900.0
KCAP = 8
GOAL = 0.78
STRETCH = 0.87
SLOTS = (16, 32, 64)
MULTS = (1.0, 0.5, 0.25, 0.1)
POLICIES = ("none", "int30", "int60", "int90", "share25b8")

# measured proposal sequences (arm kind 'opus' = the standard v1 agent; 'lean' = R63 lean arm)
V1_SEQS = [("bench", r, "opus") for r in ("5245bacc", "91a4e48d", "f98b6712",     # R38
                                          "2ded10bc", "e3fddb29", "aaedd180",     # R44
                                          "f76bebdd", "d7fed919")] + \
          [("ablation", r, "opus") for r in ("bd599a7c", "a78b1a16", "d78bfc3a")]  # R63 v1
LEAN_SEQS = [("ablation", r, "lean") for r in ("bd599a7c", "a78b1a16", "d78bfc3a")]  # R63 lean
R65_BM = [("bench", r, "opus_bm") for r in ("f76bebdd", "d7fed919")]               # validation only

PRICE_L4 = 0.000222


def ro(db: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DBS[db].as_posix()}?mode=ro", uri=True)


# ------------------------------------------------------------------ surrogate (lazy, per process)

_SCORE: dict[str, dict[str, Any]] = {}
_MOD: dict[str, Any] = {}


def lc() -> Any:
    if "lc" not in _MOD:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from backend.agent.optuna_arm import distributions
        from workers.targets import lcbench

        _MOD.update(lc=lcbench, optuna=optuna, dists=distributions(lcbench.SPEC))
    return _MOD["lc"]


def score(cfg: dict[str, Any]) -> dict[str, Any]:
    key = json.dumps(cfg, sort_keys=True)
    if key not in _SCORE:
        m = lc()
        try:
            out = m.evaluate(cfg, float(m.BUDGET_S), surrogate=m.get_surrogate())
            _SCORE[key] = {"ok": True, "score": float(out["score"]), "heldout": float(out["heldout_score"]),
                           "dur": float(out["train_time_s"])}
        except TimeoutError:
            _SCORE[key] = {"ok": False, "dur": float(m.BUDGET_S)}
    return _SCORE[key]


# ------------------------------------------------------------------ measured inputs


def ols(x: list[float], y: list[float]) -> tuple[float, float]:
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx if sxx else 0.0
    return my - b * mx, b


def ols2(x1: list[float], x2: list[float], y: list[float]) -> tuple[float, float, float]:
    n = len(y)
    m1, m2, my = sum(x1) / n, sum(x2) / n, sum(y) / n
    s11 = sum((a - m1) ** 2 for a in x1); s22 = sum((b - m2) ** 2 for b in x2)
    s12 = sum((a - m1) * (b - m2) for a, b in zip(x1, x2))
    s1y = sum((a - m1) * (c - my) for a, c in zip(x1, y)); s2y = sum((b - m2) * (c - my) for b, c in zip(x2, y))
    det = s11 * s22 - s12 ** 2
    b1 = (s1y * s22 - s2y * s12) / det
    b2 = (s2y * s11 - s1y * s12) / det
    return my - b1 * m1 - b2 * m2, b1, b2


def arm_id(db: str, short: str, kind: str) -> tuple[str, float, dict[str, Any]]:
    c = ro(db)
    rid = c.execute("select id from races where id like ?", (short + "%",)).fetchone()[0]
    aid, a0, summ = c.execute("select id, started_at, summary_json from arms where race_id=? and kind=?",
                              (rid, kind)).fetchone()
    return aid, a0, json.loads(summ or "{}")


def load_seq(db: str, short: str, kind: str) -> dict[str, Any]:
    """The arm's own (LLM) proposals in proposal order, without the baseline; measured summary."""
    aid, a0, summ = arm_id(db, short, kind)
    c = ro(db)
    cols = [r[1] for r in c.execute("pragma table_info(experiments)")]
    src = "source" if "source" in cols else "NULL"
    rows = c.execute(f"select proposed_at, config_json, is_baseline, {src}, status, is_trap, metrics_json, judge_verdict "
                     "from experiments where arm_id=? order by proposed_at", (aid,)).fetchall()
    props = [json.loads(cj) for p, cj, b, s, *_ in rows if p is not None and not b and s != "sampler"]
    counted = [json.loads(m or "{}").get("heldout_score") for *_x, st, tr, m, v in rows
               if st == "succeeded" and not tr and v != "invalid"]
    counted = [h for h in counted if h is not None]
    return {"race": short, "db": db, "kind": kind, "props": props, "n": len(props),
            "measured_best_heldout": max(counted) if counted else None,
            "measured": {"waiting_pct": summ.get("gpu_idle_pct"), "llm_usd": summ.get("llm_cost_usd"),
                         "experiments": summ.get("experiments_finished")}}


def load_call_stats(seqs: list[tuple[str, str, str]]) -> dict[str, Any]:
    from backend.agent.llm_base import OPUS_MODEL, estimate_cost_usd

    lat_k: dict[int, list[float]] = {}
    xs_k: list[float] = []; ys_l: list[float] = []
    inv_lat: list[float] = []
    n_prop = n_inv = 0
    outpp: list[float] = []
    xr: list[float] = []; xl: list[float] = []; yi: list[float] = []
    sys_tok: list[int] = []
    jl: list[float] = []; jc: list[float] = []
    for db, short, kind in seqs:
        aid, _a0, _s = arm_id(db, short, kind)
        c = ro(db)
        exps = c.execute("select proposed_at, ended_at, status from experiments where arm_id=?", (aid,)).fetchall()
        ends = sorted(e[1] for e in exps if e[2] == "succeeded" and e[1] is not None)
        for st, t, lat, inp, cw, cr, out, ids, valid in c.execute(
                "select step_type, created_at, latency_ms, input_tokens, cache_creation_tokens, cache_read_tokens, "
                "output_tokens, experiment_ids_json, valid from llm_calls where arm_id=? order by created_at", (aid,)):
            if st == "judge_improvement":
                jl.append(lat / 1000); jc.append(estimate_cost_usd(OPUS_MODEL, inp, cw, cr, out))
                continue
            if st != "propose_experiments":
                continue
            n_prop += 1
            k = len(json.loads(ids or "[]"))
            if not valid or k == 0:
                n_inv += 1; inv_lat.append(lat / 1000)
                continue
            lat_k.setdefault(k, []).append(lat / 1000)
            xs_k.append(k); ys_l.append(lat / 1000)
            outpp.append(out / k)
            sys_tok.append((cr or 0) + (cw or 0))
            n_fin = sum(1 for e in ends if e < t)
            xr.append(min(5, n_fin) + min(8, n_fin))
            xl.append(sum(1 for e in exps if e[0] is not None and e[0] < t and (e[1] is None or e[1] > t)))
            yi.append(inp)
    a, b = ols(xs_k, ys_l)
    return {"lat_k1": lat_k[1], "lat_fit": (a, b), "lat_k_p50": {k: (len(v), statistics.median(v)) for k, v in sorted(lat_k.items())},
            "p_invalid": n_inv / max(1, n_prop), "inv_lat": inv_lat or lat_k[1],
            "out_per_prop": statistics.fmean(outpp), "in_fit": ols2(xr, xl, yi),
            "sys_tokens": int(statistics.median(sys_tok)), "judge_lat": jl, "judge_cost": jc,
            "n_calls": n_prop}


def ask_time(n_trials: int) -> float:
    """TPE ask per trial (sim_backfill.ask_time; checked here: 1,500 sequential trials took 134 s)."""
    return 0.02 + 0.07 * n_trials / 600


# ------------------------------------------------------------------ the event engine


class Race:
    """One arm of one race. arm: 'opus' (standard agent), 'bm' (backfill + checker + cadence policy),
    'optuna' (TPE only, no checker, as backend/agent/optuna_arm.py)."""

    def __init__(self, arm: str, N: int, m: float, *, stats: dict[str, Any] | None, judge: dict[str, Any],
                 seq: list[dict[str, Any]] | None, policy: str = "none", tpe_seed: int = 0, rng_seed: int = 0,
                 lat_mult: float = 1.0) -> None:
        self.arm, self.N, self.m, self.stats, self.judge = arm, N, m, stats, judge
        self.seq = list(seq or [])
        self.policy, self.lat_mult = policy, lat_mult
        self.rng = random.Random(rng_seed * 7919 + N * 31 + int(m * 1000))
        self.study = None
        if arm in ("bm", "optuna"):
            lc()
            opt = _MOD["optuna"]
            self.study = opt.create_study(direction="maximize", sampler=opt.samplers.TPESampler(
                seed=tpe_seed, n_startup_trials=10, constant_liar=True))
        self.seq_durs = [score(c)["dur"] for c in self.seq] or [60.0]

    def run(self) -> dict[str, Any]:
        lcm = lc()
        SPEC = lcm.SPEC
        opt = _MOD.get("optuna")
        dists = _MOD.get("dists")
        N, m = self.N, self.m
        heap: list[tuple[float, int, int, str, Any]] = []
        cnt = 0

        def push(t: float, kind: str, data: Any, prio: int = 1) -> None:
            nonlocal cnt
            heapq.heappush(heap, (t, prio, cnt, kind, data)); cnt += 1

        free = N
        running: dict[int, dict[str, Any]] = {}
        buf_llm: list[dict[str, Any] | None] = []      # None = phantom (sequence exhausted)
        held = 0
        seq_i = 0
        lane_busy = False
        judge_q: list[str] = []
        last_call = -1e9
        wake_pending = False
        sampler_until = 0.0
        n_trials = 0
        ask_total = 0.0
        busy = 0.0
        checked_best: float | None = None
        worst: float | None = None
        rid = 0
        res: dict[str, Any] = {"exp": {"llm": 0, "samp": 0, "base": 0, "phantom": 0}, "calls_propose": 0, "calls_invalid": 0,
                               "calls_judge": 0, "judges": {"llm": 0, "samp": 0}, "usd_propose": 0.0, "usd_judge": 0.0,
                               "llm_props": 0, "phantoms": 0, "trap_runs": {"llm": 0, "samp": 0}, "trap_claims": 0,
                               "fakes_accepted": 0}
        counted: list[dict[str, Any]] = []
        accepted: list[dict[str, Any]] = []
        n_done = 0
        first_return: float | None = None
        st = self.stats
        cap_llm = math.ceil(0.25 * N) if self.policy == "share25b8" else N
        interval = float(self.policy[3:]) if self.policy.startswith("int") else 0.0

        def launch(t: float, cfg: dict[str, Any] | None, src: str, trial: Any = None) -> None:
            nonlocal free, rid
            free -= 1
            if cfg is None:
                r = {"ok": False, "dur": self.rng.choice(self.seq_durs), "phantom": True}
            else:
                r = score(cfg)
            running[rid] = {"t0": t, "cfg": cfg, "src": src, "trial": trial, "r": r}
            push(t + r["dur"] * m, "end", rid, prio=0); rid += 1

        def llm_running() -> int:
            return sum(1 for v in running.values() if v["src"] in ("llm", "base"))

        def fill(t: float) -> None:
            nonlocal held, sampler_until, n_trials, ask_total
            while buf_llm and free > 0:
                launch(t, buf_llm.pop(0), "llm")
            if self.study is None or t >= WINDOW_S:
                return
            if self.arm == "bm" and buf_llm:
                return
            while free > 0:
                a = ask_time(n_trials)
                t_ready = max(t, sampler_until) + a
                sampler_until = t_ready
                ask_total += a
                n_trials += 1
                trial = self.study.ask()
                cfg = SPEC.validate_config(SPEC.suggest_optuna(trial))
                self_free_hold()
                push(t_ready, "ask", (trial, cfg))

        def self_free_hold() -> None:
            nonlocal free, held
            free -= 1; held += 1

        def demand() -> int:
            if self.arm == "opus":
                d = N - len(running) - len(buf_llm) - held
            else:
                d = cap_llm - llm_running() - len(buf_llm)
            return max(0, min(KCAP, d))

        def dispatch(t: float) -> None:
            nonlocal lane_busy, last_call, wake_pending
            if lane_busy or t >= WINDOW_S or self.arm == "optuna":
                return
            if judge_q:
                src = judge_q.pop(0)
                j = self.rng.randrange(len(self.judge["lat"]))
                lane_busy = True
                res["calls_judge"] += 1; res["judges"][src] += 1
                push(t + self.judge["lat"][j] * self.lat_mult, "ret", ("j", 0, self.judge["cost"][j]))
                return
            k = demand()
            if k <= 0:
                return
            if self.arm == "bm":
                if self.policy == "share25b8" and k < min(KCAP, cap_llm):
                    return
                if interval and t < last_call + interval:
                    if not wake_pending:
                        wake_pending = True
                        push(last_call + interval, "wake", None)
                    return
            from backend.agent.llm_base import OPUS_MODEL, estimate_cost_usd

            a_in, b_rows, b_run = st["in_fit"]
            listed = len(running) + len(buf_llm) + held
            rows = min(5, n_done) + min(8, n_done)
            inp = max(0.0, a_in + b_rows * rows + b_run * listed)
            write = first_return is None
            cost = estimate_cost_usd(OPUS_MODEL, int(inp), st["sys_tokens"] if write else 0,
                                     0 if write else st["sys_tokens"], int(st["out_per_prop"] * k))
            lane_busy = True
            last_call = t
            res["calls_propose"] += 1
            if self.rng.random() < st["p_invalid"]:
                push(t + self.rng.choice(st["inv_lat"]) * self.lat_mult, "ret", ("p", 0, cost))
                return
            a, b = st["lat_fit"]
            lat = self.rng.choice(st["lat_k1"]) * max(0.2, (a + b * k) / (a + b))
            push(t + lat * self.lat_mult, "ret", ("p", k, cost))

        # t0: the baseline on slot 0 (run_baseline_first), then the first propose / sampler asks
        launch(0.0, SPEC.default_config(), "base")
        if self.arm == "optuna":
            pass
        fill(0.0)
        dispatch(0.0)
        while heap:
            t, _p, _c, kind, data = heapq.heappop(heap)
            if t > WINDOW_S:
                break
            if kind == "wake":
                wake_pending = False
                dispatch(t)
            elif kind == "ret":
                typ, k, cost = data
                lane_busy = False
                if first_return is None:
                    first_return = t
                if typ == "j":
                    res["usd_judge"] += cost
                else:
                    res["usd_propose"] += cost
                    if k == 0:
                        res["calls_invalid"] += 1
                    for _ in range(k):
                        if seq_i < len(self.seq):
                            buf_llm.append(self.seq[seq_i]); seq_i += 1
                        else:
                            buf_llm.append(None); res["phantoms"] += 1
                        res["llm_props"] += 1
                    fill(t)
                dispatch(t)
            elif kind == "ask":
                trial, cfg = data
                held -= 1; free += 1
                if t < WINDOW_S:
                    launch(t, cfg, "samp", trial)
                fill(t)
                dispatch(t)
            elif kind == "end":
                x = running.pop(data)
                free += 1
                n_done += 1
                busy += x["r"]["dur"] * m
                r, src, cfg = x["r"], x["src"], x["cfg"]
                if r.get("phantom"):
                    res["exp"]["phantom"] += 1
                    fill(t); dispatch(t)
                    continue
                res["exp"][src] += 1
                trap = SPEC.is_trap(cfg)
                if trap and src in ("llm", "samp"):
                    res["trap_runs"][src] += 1
                told = r["score"] if r["ok"] else None
                row = {"t": t, "src": src, "trap": trap, **r}
                if r["ok"]:
                    if self.arm == "optuna":
                        if checked_best is None or r["score"] > checked_best:
                            checked_best = r["score"]; accepted.append(row)
                            res["fakes_accepted"] += trap
                    elif src == "base":
                        checked_best = r["score"] if checked_best is None else max(checked_best, r["score"])
                        accepted.append(row)
                    elif checked_best is None or r["score"] > checked_best:
                        judge_q.append(src)
                        if trap:
                            res["trap_claims"] += 1
                            told = worst if worst is not None else r["score"]
                        else:
                            checked_best = r["score"]; accepted.append(row)
                    worst = r["score"] if worst is None else min(worst, r["score"])
                    if not trap:
                        counted.append(row)
                if self.study is not None:
                    if x["trial"] is not None:
                        if told is None:
                            self.study.tell(x["trial"], state=opt.trial.TrialState.FAIL)
                        else:
                            self.study.tell(x["trial"], told)
                    elif told is not None and cfg is not None:
                        params = {kk: cfg[kk] for kk in dists if kk in cfg}
                        self.study.add_trial(opt.trial.create_trial(
                            params=params, distributions={kk: dists[kk] for kk in params}, value=told))
                fill(t)
                dispatch(t)
        for x in running.values():
            busy += max(0.0, WINDOW_S - x["t0"])
        shipped = max(accepted, key=lambda r: r["score"], default=None)
        best = max(counted, key=lambda r: r["heldout"], default=None)
        llm_usd = res["usd_propose"] + res["usd_judge"]
        exps = sum(v for k, v in res["exp"].items())
        res.update({
            "arm": self.arm, "N": N, "m": m, "policy": self.policy,
            "waiting": 1.0 - busy / (N * WINDOW_S), "experiments": exps, "llm_usd": llm_usd,
            "best_heldout": best["heldout"] if best else None, "best_src": best["src"] if best else None,
            "shipped_heldout": shipped["heldout"] if shipped else None,
            "shipped_trap": bool(shipped["trap"]) if shipped else None,
            "t_goal": min((r["t"] for r in counted if r["heldout"] >= GOAL), default=None),
            "t_stretch": min((r["t"] for r in counted if r["heldout"] >= STRETCH), default=None),
            "seq_len": len(self.seq), "seq_used": min(seq_i, len(self.seq)),
            "sampler_trials": n_trials, "ask_util": ask_total / WINDOW_S,
            "llm_usd_per_exp": llm_usd / max(1, exps),
        })
        return res


# ------------------------------------------------------------------ worker plumbing

_CTX: dict[str, Any] = {}


def _init(ctx: dict[str, Any]) -> None:
    _CTX.update(ctx)
    lc()
    lc().get_surrogate()


def _job(job: dict[str, Any]) -> dict[str, Any]:
    c = _CTX
    stats = c["stats"].get(job.get("prompt", ""))
    seq = c["seqs"][job["seq"]]["props"] if job.get("seq") else None
    r = Race(job["arm"], job["N"], job["m"], stats=stats, judge=c["judge"], seq=seq, policy=job.get("policy", "none"),
             tpe_seed=job.get("tpe_seed", 0), rng_seed=job.get("rng_seed", 0), lat_mult=job.get("lat_mult", 1.0)).run()
    r["job"] = job
    return r


def run_jobs(ctx: dict[str, Any], jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    t0 = time.time()
    # longest first (high throughput regimes) for better packing
    jobs = sorted(jobs, key=lambda j: -(j["N"] / j["m"]) * (j["arm"] != "opus"))
    out: list[dict[str, Any]] = []
    if workers <= 1:
        _init(ctx)
        for j in jobs:
            out.append(_job(j))
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(ctx,)) as pool:
            for i, r in enumerate(pool.imap_unordered(_job, jobs)):
                out.append(r)
                if (i + 1) % max(1, len(jobs) // 10) == 0:
                    print(f"  {i + 1}/{len(jobs)} jobs [{time.time() - t0:.0f}s]", flush=True)
    return out


def agg(vals: list[Any]) -> dict[str, Any]:
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None, "sd": None, "n": 0, "min": None, "max": None, "n_none": len(vals)}
    return {"mean": statistics.fmean(v), "sd": statistics.pstdev(v) if len(v) > 1 else 0.0, "n": len(v),
            "min": min(v), "max": max(v), "n_none": len(vals) - len(v)}


KEYS = ("waiting", "experiments", "llm_usd", "llm_usd_per_exp", "calls_propose", "calls_judge", "llm_props", "phantoms",
        "best_heldout", "shipped_heldout", "t_goal", "t_stretch", "sampler_trials", "ask_util", "trap_claims",
        "fakes_accepted", "usd_judge", "usd_propose")


def summarize(rs: list[dict[str, Any]]) -> dict[str, Any]:
    s = {k: agg([r.get(k) for r in rs]) for k in KEYS}
    s["exp_llm"] = agg([r["exp"]["llm"] for r in rs]); s["exp_samp"] = agg([r["exp"]["samp"] for r in rs])
    s["judges_samp"] = agg([r["judges"]["samp"] for r in rs]); s["trap_runs_samp"] = agg([r["trap_runs"]["samp"] for r in rs])
    s["trap_runs_llm"] = agg([r["trap_runs"]["llm"] for r in rs])
    s["best_src_samp"] = sum(1 for r in rs if r.get("best_src") == "samp")
    s["shipped_trap"] = sum(1 for r in rs if r.get("shipped_trap"))
    s["reached_stretch"] = sum(1 for r in rs if r.get("t_stretch") is not None)
    s["n"] = len(rs)
    return s


def realism(N: int, m: float) -> float:
    return abs(math.log2(m)) + 0.5 * abs(math.log2(N / 16))


# ------------------------------------------------------------------ stages


def build_ctx() -> dict[str, Any]:
    seqs = {}
    for db, short, kind in V1_SEQS + LEAN_SEQS + R65_BM:
        seqs[f"{kind}:{short}"] = load_seq(db, short, kind)
    v1 = load_call_stats(V1_SEQS)
    lean = load_call_stats(LEAN_SEQS)
    bm65 = load_call_stats(R65_BM)
    judge = {"lat": v1["judge_lat"] + lean["judge_lat"], "cost": v1["judge_cost"] + lean["judge_cost"]}
    return {"seqs": seqs, "stats": {"v1": v1, "lean": lean, "v1bm": bm65}, "judge": judge}


def stats_info(st: dict[str, Any]) -> dict[str, Any]:
    return {"lat_k1_p50": statistics.median(st["lat_k1"]), "lat_fit_a_b": st["lat_fit"], "lat_k_p50": st["lat_k_p50"],
            "lat_k8_model_p50": statistics.median(st["lat_k1"]) * (st["lat_fit"][0] + 8 * st["lat_fit"][1]) / sum(st["lat_fit"]),
            "p_invalid": st["p_invalid"], "out_per_prop": st["out_per_prop"], "in_fit": st["in_fit"],
            "sys_tokens": st["sys_tokens"], "n_calls": st["n_calls"]}


def stage_ceiling(ctx: dict[str, Any], n_random: int = 20000) -> dict[str, Any]:
    """Plausible ceiling of held-out balanced accuracy on task 167168 (non-trap configs)."""
    m = lc()
    SPEC = m.SPEC
    rng = random.Random(0)
    t0 = time.time()

    def rand_cfg() -> dict[str, Any]:
        c: dict[str, Any] = {}
        for k, p in SPEC.config_space.items():
            if p.type == "categorical":
                c[k] = rng.choice([x for x in p.choices if x != "best_test"])
            elif p.type == "int":
                c[k] = int(round(math.exp(rng.uniform(math.log(p.low), math.log(p.high))))) if p.log else rng.randint(int(p.low), int(p.high))
            else:
                c[k] = math.exp(rng.uniform(math.log(p.low), math.log(p.high))) if p.log else rng.uniform(p.low, p.high)
        return SPEC.validate_config(c)

    pool = []
    for _ in range(n_random):
        c = rand_cfg(); r = score(c)
        if r["ok"]:
            pool.append((r["heldout"], r["score"], c))
    pool.sort(key=lambda x: -x[0])
    hs = sorted(x[0] for x in pool)
    best_rand = pool[0][0]
    best_by_val = max(pool, key=lambda x: x[1])
    # hill climb on VALIDATION from the top-20-by-validation configs (what any searcher can see)
    top = sorted(pool, key=lambda x: -x[1])[:20]
    hc_best_val = (best_by_val[1], best_by_val[0], best_by_val[2])
    hc_best_held = best_rand
    for _h, v, c in top:
        cur, cur_v = dict(c), v
        for _ in range(150):
            n = dict(cur)
            k = rng.choice([k for k in SPEC.config_space if k != "checkpoint_selection"])
            p = SPEC.config_space[k]
            if p.type == "int":
                n[k] = int(min(p.high, max(p.low, n[k] + rng.choice([-2, -1, 1, 2]) * (max(1, int(n[k] * 0.1)) if p.log else 1))))
            else:
                if p.log:
                    n[k] = min(p.high, max(p.low, n[k] * math.exp(rng.gauss(0, 0.3))))
                else:
                    n[k] = min(p.high, max(p.low, n[k] + rng.gauss(0, 0.1 * (p.high - p.low))))
            n = SPEC.validate_config(n)
            r = score(n)
            if r["ok"]:
                hc_best_held = max(hc_best_held, r["heldout"])
                if r["score"] > cur_v:
                    cur, cur_v = n, r["score"]
                    if r["score"] > hc_best_val[0]:
                        hc_best_val = (r["score"], r["heldout"], n)
    # logged runs: every succeeded non-trap run in both DBs (all LCBench arms)
    logged = []
    for db in DBS:
        c = ro(db)
        for (mj,) in c.execute("select e.metrics_json from experiments e join arms a on a.id=e.arm_id join races r on "
                               "r.id=a.race_id where r.name like '%LCBench%' and e.status='succeeded' and e.is_trap=0"):
            h = json.loads(mj or "{}").get("heldout_score")
            if h is not None:
                logged.append(h)
    out = {
        "n_random_ok": len(pool), "random_heldout_p50": hs[len(hs) // 2], "random_heldout_p90": hs[int(0.9 * len(hs))],
        "random_heldout_p99": hs[int(0.99 * len(hs))], "random_best_heldout": best_rand,
        "random_share_ge_087": sum(1 for h in hs if h >= STRETCH) / len(hs),
        "random_best_val_config_heldout": best_by_val[0], "random_best_val": best_by_val[1],
        "hillclimb_best_val": hc_best_val[0], "hillclimb_best_val_config_heldout": hc_best_val[1],
        "hillclimb_best_heldout_seen": hc_best_held, "hillclimb_best_config": hc_best_val[2],
        "logged_runs": len(logged), "logged_best_heldout": max(logged), "logged_share_ge_087": sum(1 for h in logged if h >= STRETCH) / len(logged),
        "logged_n_ge_087": sum(1 for h in logged if h >= STRETCH),
        "seconds": time.time() - t0,
    }
    print("ceiling:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k != "hillclimb_best_config"}))
    return out


def seq_curves(ctx: dict[str, Any]) -> dict[str, Any]:
    """Best held-out after the first n proposals of each measured sequence (for an Opus extrapolation)."""
    out: dict[str, Any] = {}
    for key, s in ctx["seqs"].items():
        best = None; curve = []
        for c in s["props"]:
            r = score(c)
            if r["ok"] and not lc().SPEC.is_trap(c):
                best = r["heldout"] if best is None else max(best, r["heldout"])
            curve.append(best)
        out[key] = {"n": len(curve), "best_all": curve[-1] if curve else None,
                    "at": {n: curve[min(n, len(curve)) - 1] for n in (10, 25, 50, 75, 100) if curve},
                    "measured_best": s["measured_best_heldout"], "measured": s["measured"]}
    # log fit best(n) = a + b ln n, pooled over v1 sequences, n = 10..len
    xs, ys = [], []
    for key, s in ctx["seqs"].items():
        if not key.startswith("opus:"):
            continue
        best = None
        for i, c in enumerate(s["props"], 1):
            r = score(c)
            if r["ok"] and not lc().SPEC.is_trap(c):
                best = r["heldout"] if best is None else max(best, r["heldout"])
            if i >= 10 and best is not None:
                xs.append(math.log(i)); ys.append(best)
    a, b = ols(xs, ys)
    out["_v1_logfit"] = {"a": a, "b": b, "pred": {n: a + b * math.log(n) for n in (100, 200, 400, 600, 800)}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ceiling", "opus", "bm", "replay", "all"])
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--opus-rng", type=int, default=3)
    ap.add_argument("--replay-regimes", nargs="*", default=None, help="e.g. 16x0.1 64x0.25")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = build_ctx()
    info = {k: stats_info(v) for k, v in ctx["stats"].items()}
    info["judge"] = {"p50_s": statistics.median(ctx["judge"]["lat"]), "cost_mean": statistics.fmean(ctx["judge"]["cost"]),
                     "n": len(ctx["judge"]["lat"])}
    info["seq_lens"] = {k: v["n"] for k, v in ctx["seqs"].items()}
    print("inputs:", json.dumps(info, default=str)[:3000])
    stages = ["ceiling", "opus", "bm", "replay"] if args.stage == "all" else [args.stage]

    if "ceiling" in stages:
        _init(ctx)
        ce = stage_ceiling(ctx)
        cu = seq_curves(ctx)
        (OUT_DIR / "ceiling.json").write_text(json.dumps({"ceiling": ce, "seq_curves": cu}, indent=1, default=str), encoding="utf-8")
        print("v1 log fit:", cu["_v1_logfit"])
        for k, v in cu.items():
            if not k.startswith("_"):
                print(f"  {k}: n={v['n']} best_all={v['best_all']:.4f} measured={v['measured_best']} at={ {n: round(x, 4) for n, x in v['at'].items()} }")

    v1_keys = [f"opus:{s}" for _d, s, _k in V1_SEQS]
    lean_keys = [f"lean:{s}" for _d, s, _k in LEAN_SEQS]
    bm65_keys = [f"opus_bm:{s}" for _d, s, _k in R65_BM]

    if "opus" in stages:
        jobs = [{"arm": "opus", "prompt": "v1", "seq": k, "N": N, "m": m, "rng_seed": r}
                for N in SLOTS for m in MULTS for k in v1_keys for r in range(args.opus_rng)]
        rs = run_jobs(ctx, jobs, args.workers)
        grid = {}
        for N in SLOTS:
            for m in MULTS:
                g = [r for r in rs if r["N"] == N and r["m"] == m]
                grid[f"{N}x{m:g}"] = summarize(g)
        meas = [ctx["seqs"][k]["measured"] for k in v1_keys]
        val = {"measured_waiting_pct": agg([x["waiting_pct"] for x in meas]), "measured_llm_usd": agg([x["llm_usd"] for x in meas]),
               "measured_experiments": agg([x["experiments"] for x in meas])}
        (OUT_DIR / "opus_grid.json").write_text(json.dumps({"inputs": info, "grid": grid, "validation_measured": val}, indent=1, default=str), encoding="utf-8")
        print(f"validation 16x1: sim waiting {grid['16x1']['waiting']['mean']*100:.1f}% vs measured "
              f"{val['measured_waiting_pct']['mean']:.1f}%; LLM ${grid['16x1']['llm_usd']['mean']:.2f} vs "
              f"${val['measured_llm_usd']['mean']:.2f}; exps {grid['16x1']['experiments']['mean']:.0f} vs {val['measured_experiments']['mean']:.0f}")
        print("| N x m | Opus waiting % | exps | LLM $ | calls | props (phantom) | judge | realism |")
        for key, s in grid.items():
            N, m = key.split("x")
            print(f"| {key} | {s['waiting']['mean']*100:.1f} ± {s['waiting']['sd']*100:.1f} | {s['experiments']['mean']:.0f} | "
                  f"{s['llm_usd']['mean']:.2f} | {s['calls_propose']['mean']:.0f} | {s['llm_props']['mean']:.0f} "
                  f"({s['phantoms']['mean']:.0f}) | {s['calls_judge']['mean']:.0f} | {realism(int(N), float(m)):.1f} |")

    if "bm" in stages:
        og = json.loads((OUT_DIR / "opus_grid.json").read_text(encoding="utf-8"))["grid"]
        mid = sorted(lean_keys, key=lambda k: ctx["seqs"][k]["n"])[1]   # median-length lean sequence
        regimes = [(N, m) for N in SLOTS for m in MULTS if og[f"{N}x{m:g}"]["waiting"]["mean"] >= 0.30 or (N, m) == (16, 1.0)]
        jobs = [{"arm": "bm", "prompt": "lean", "seq": mid, "N": N, "m": m, "policy": p, "tpe_seed": 0, "rng_seed": 0}
                for N, m in regimes for p in POLICIES]
        # validation: BM (v1 prompt, R65 sequences) at 16 x 1 vs R65 measured
        jobs += [{"arm": "bm", "prompt": "v1bm", "seq": k, "N": 16, "m": 1.0, "policy": "none", "tpe_seed": s, "rng_seed": s}
                 for k in bm65_keys for s in range(2)]
        rs = run_jobs(ctx, jobs, args.workers)
        grid: dict[str, Any] = {}
        for N, m in regimes:
            for p in POLICIES:
                g = [r for r in rs if r["N"] == N and r["m"] == m and r["policy"] == p and r["job"]["prompt"] == "lean"]
                grid[f"{N}x{m:g}/{p}"] = summarize(g)
        v = [r for r in rs if r["job"]["prompt"] == "v1bm"]
        val = {"sim": summarize(v), "measured": [ctx["seqs"][k]["measured"] for k in bm65_keys]}
        (OUT_DIR / "bm_grid.json").write_text(json.dumps({"seq": mid, "regimes": regimes, "grid": grid, "validation_r65": val},
                                                         indent=1, default=str), encoding="utf-8")
        print(f"validation R65 BM (v1, 16x1): sim waiting {val['sim']['waiting']['mean']*100:.2f}% LLM "
              f"${val['sim']['llm_usd']['mean']:.2f} exps {val['sim']['experiments']['mean']:.0f} vs measured {val['measured']}")
        print("| regime/policy | BM waiting % | exps (LLM/samp) | Opus props (phantom) | LLM $ | x Opus | judge (samp) | ask util | best (1 replay) |")
        for key, s in grid.items():
            o = og[key.split("/")[0]]
            print(f"| {key} | {s['waiting']['mean']*100:.2f} | {s['experiments']['mean']:.0f} ({s['exp_llm']['mean']:.0f}/{s['exp_samp']['mean']:.0f}) | "
                  f"{s['llm_props']['mean']:.0f} ({s['phantoms']['mean']:.0f}) | {s['llm_usd']['mean']:.2f} | "
                  f"{s['llm_usd']['mean']/o['llm_usd']['mean']:.2f} | {s['calls_judge']['mean']:.0f} ({s['judges_samp']['mean']:.0f}) | "
                  f"{s['ask_util']['mean']*100:.0f}% | {s['best_heldout']['mean']:.4f} |")

    if "replay" in stages:
        og = json.loads((OUT_DIR / "opus_grid.json").read_text(encoding="utf-8"))["grid"]
        bg = json.loads((OUT_DIR / "bm_grid.json").read_text(encoding="utf-8"))
        min_lean = min(ctx["seqs"][k]["n"] for k in lean_keys)
        choice: dict[str, Any] = {}
        for N, m in [tuple(x) for x in bg["regimes"]]:
            key = f"{N}x{m:g}"
            o = og[key]
            cands = []
            for p in POLICIES:
                s = bg["grid"][f"{key}/{p}"]
                ratio = s["llm_usd"]["mean"] / o["llm_usd"]["mean"]
                cands.append({"policy": p, "ratio": ratio, "props": s["llm_props"]["mean"], "phantoms": s["phantoms"]["mean"],
                              "fits": s["llm_props"]["mean"] <= min_lean})
            ok = [c for c in cands if c["ratio"] <= 0.70]
            fit = [c for c in ok if c["fits"]]
            pick = max(fit, key=lambda c: c["props"]) if fit else (min(ok, key=lambda c: c["phantoms"]) if ok else None)
            choice[key] = {"candidates": cands, "pick": pick}
            print(f"{key}: policy pick {pick}")
        # replays on the regimes that can still pass the sim criteria (Opus >= 40% waiting, BM <= 2%,
        # feasible asks) -- compute: one 64-slot replay takes ~15-20 CPU-min. --replay-regimes overrides.
        want = set(args.replay_regimes or [])
        for key, ch in choice.items():
            s = bg["grid"].get(f"{key}/{ch['pick']['policy']}") if ch["pick"] else None
            ch["replayed"] = bool(s) and (key in want if want else (
                og[key]["waiting"]["mean"] >= 0.40 and s["waiting"]["mean"] <= 0.02 and s["ask_util"]["mean"] <= 0.5))
        jobs = []
        for key, ch in choice.items():
            if not ch.get("replayed"):
                continue
            N, m = key.split("x"); N = int(N); m = float(m)
            for k in lean_keys:
                for s in range(3):
                    jobs.append({"arm": "bm", "prompt": "lean", "seq": k, "N": N, "m": m, "policy": ch["pick"]["policy"],
                                 "tpe_seed": s, "rng_seed": s, "tag": "bm"})
            for k in v1_keys[-3:]:   # ablation: same engine + cadence with the v1 sequences of the same R63 races
                jobs.append({"arm": "bm", "prompt": "v1", "seq": k, "N": N, "m": m, "policy": ch["pick"]["policy"],
                             "tpe_seed": 0, "rng_seed": 0, "tag": "bm_v1"})
            for k in lean_keys:   # ablation: lean planner alone, standard loop (no backfill)
                jobs.append({"arm": "opus", "prompt": "lean", "seq": k, "N": N, "m": m, "rng_seed": 0, "tag": "lean_alone"})
            for s in range(3):
                jobs.append({"arm": "optuna", "N": N, "m": m, "tpe_seed": s, "rng_seed": s, "tag": "optuna"})
            for k in lean_keys:   # latency sensitivity for BM: lean latency x1.25
                jobs.append({"arm": "bm", "prompt": "lean", "seq": k, "N": N, "m": m, "policy": ch["pick"]["policy"],
                             "tpe_seed": 0, "rng_seed": 0, "lat_mult": 1.25, "tag": "bm_lat125"})
            for k in v1_keys:   # Opus latency sensitivity x0.8 (a faster Opus waits less)
                jobs.append({"arm": "opus", "prompt": "v1", "seq": k, "N": N, "m": m, "rng_seed": 0, "lat_mult": 0.8, "tag": "opus_lat08"})
        rs = run_jobs(ctx, jobs, args.workers)
        out: dict[str, Any] = {"choice": choice, "regimes": {}}
        for key, ch in choice.items():
            if not ch.get("replayed"):
                continue
            N, m = key.split("x"); N = int(N); m = float(m)
            sel = lambda tag: [r for r in rs if r["N"] == N and r["m"] == m and r["job"].get("tag") == tag]  # noqa: E731
            reg = {tag: summarize(sel(tag)) for tag in ("bm", "bm_v1", "lean_alone", "optuna", "bm_lat125", "opus_lat08")}
            reg["opus"] = og[key]
            # per lean sequence means (race-level n = 3)
            reg["bm_per_seq"] = {k: agg([r["best_heldout"] for r in sel("bm") if r["job"]["seq"] == k]) for k in lean_keys}
            o, b, op = reg["opus"], reg["bm"], reg["optuna"]
            gate = {
                "opus_waiting_ge_40": o["waiting"]["mean"] >= 0.40,
                "bm_waiting_le_2": b["waiting"]["mean"] <= 0.02,
                "bm_usd_le_08x": b["llm_usd"]["mean"] <= 0.8 * o["llm_usd"]["mean"],
                "bm_best_ge_opus": b["best_heldout"]["mean"] >= o["best_heldout"]["mean"],
                "bm_best_ge_optuna": b["best_heldout"]["mean"] >= op["best_heldout"]["mean"],
                "feasible_asks": b["ask_util"]["mean"] <= 0.5 and op["ask_util"]["mean"] <= 0.5,
            }
            gate["pass"] = all(gate.values())
            reg["gate"] = gate
            reg["realism"] = realism(N, m)
            reg["usd_ratio"] = b["llm_usd"]["mean"] / o["llm_usd"]["mean"]
            out["regimes"][key] = reg
            print(f"\n== {key} (realism {realism(N, m):.1f}) policy {ch['pick']['policy']}: gate {gate}")
            for tag in ("opus", "bm", "bm_v1", "lean_alone", "optuna", "bm_lat125", "opus_lat08"):
                s = reg[tag]
                print(f"  {tag:11s} waiting {s['waiting']['mean']*100:5.1f}% exps {s['experiments']['mean']:6.0f} LLM ${s['llm_usd']['mean']:.2f} "
                      f"best {s['best_heldout']['mean']:.4f} ± {s['best_heldout']['sd']:.4f} [{s['best_heldout']['min']:.4f}-{s['best_heldout']['max']:.4f}] "
                      f"shipped {s['shipped_heldout']['mean']:.4f} stretch {s['reached_stretch']}/{s['n']} judge {s['calls_judge']['mean']:.0f} "
                      f"trapclaims {s['trap_claims']['mean']:.1f} ask {s['ask_util']['mean']*100:.0f}% phantoms {s['phantoms']['mean']:.0f}")
        passing = [k for k, v in out["regimes"].items() if v["gate"]["pass"]]
        pick = min(passing, key=lambda k: (out["regimes"][k]["realism"], int(k.split("x")[0]))) if passing else None
        out["passing"] = passing; out["pick"] = pick
        (OUT_DIR / "replay.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        print(f"\npassing: {passing}; pick (lowest realism cost): {pick}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
