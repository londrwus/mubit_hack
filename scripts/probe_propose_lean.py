"""Lean propose replay: v1 vs effort/diff variants on REAL R48 digests (Opus 5.5 via Mubit, record OFF).

Rebuilds the propose_experiments user message for N stored R48 Opus calls (RL wide, 30 slots,
K=8) from the race's own state in ``data/bench/bench.db`` (history = runs finished before the
call, oldest first; running = proposed and not finished; labels by proposal order; judge
'invalid' -> score hidden with the same note as ``RepoPrompts._row``), then sends the same
digest to every variant, in a per-digest shuffled order, sequentially:

  v1_med  frozen v1 prompt, effort medium (today's arm; baseline)
  v1_low  frozen v1 prompt, effort low                       (a)
  v2_med  v2 diff prompt (changes vs current best, <=10 words) (b)
  v2_low  v2, effort low                                      (c)
  v3_med  v3 diff prompt (<=6-word hypothesis)                (d)

Plus a few K=2 calls per variant (same digests) so the simulator gets a latency-vs-K fit.

Request layout is exactly ``MubitClient.build_request`` (cached system + spec block, structured
output, metadata.user_id); the header ``x-middleware-record: off`` is sent on EVERY call (never
recorded; never ANTHROPIC_API_KEY). Hard $ cap before every call.

Per call: latency, the four usage meters, $ (estimate@list), stop reason, raw reply text, visible
reply tokens (``count_tokens``, free) -> thinking tokens = billed output - visible, valid configs,
duplicates vs history/running/batch, distance from the current best, trap proposals.

Usage: PYTHONUTF8=1 python -m uv run python scripts/probe_propose_lean.py [--digests 20] [--k2 6] [--max-usd 8]
Writes data/propose_lean/calls.jsonl and summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sqlite3
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import workers.targets  # noqa: E402,F401  (registers targets)
from backend.agent.llm_base import OPUS_MODEL, estimate_cost_usd  # noqa: E402
from backend.agent.mubit_client import MubitClient, parse_and_validate  # noqa: E402
from backend.agent.prompts import load_system_prompt, prompt_fingerprint  # noqa: E402
from backend.agent.prompts import build as B  # noqa: E402
from backend.core.types import ProposeOutput, StepType  # noqa: E402
from workers.targets.base import TargetSpec, get_target  # noqa: E402

DB = REPO / "data" / "bench" / "bench.db"
OUT = REPO / "data" / "propose_lean"
R48 = ("8a76fa052c28441cb2aea221f099a7d2", "45ab180f9fdc403e93f6f16960e6b0f9",
       "59d7f7ebe0bc4b6ba59edacd0d769f49")
STEP = StepType.propose_experiments
HEADERS = {"x-middleware-record": "off"}  # never record probe calls

VARIANTS: dict[str, tuple[str, str]] = {  # name -> (prompt version, effort)
    "v1_med": ("v1", "medium"),
    "v1_low": ("v1", "low"),
    "v2_med": ("v2", "medium"),
    "v2_low": ("v2", "low"),
    "v3_med": ("v3", "medium"),
}


# --------------------------------------------------------------------------- digests from bench.db


def ro() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)


def race_spec(cfg: dict[str, Any]) -> TargetSpec:
    spec = get_target(cfg["target"]).spec
    upd: dict[str, Any] = {}
    if cfg.get("target_value") is not None:
        upd["target_value"] = float(cfg["target_value"])
    if cfg.get("run_budget_s") is not None:
        upd["budget_s"] = int(cfg["run_budget_s"])
    return spec.model_copy(update=upd)


def _row(e: dict[str, Any]) -> B.ResultRow:
    """Same note/score rules as backend.agent.loop.RepoPrompts._row."""
    note: str | None = None
    score = e["score"]
    if e["verdict"] == "invalid":
        note, score = "judge: invalid, not counted", None
    elif e["diagnosis"]:
        note = f"{e['diagnosis'].get('failure_class')}: {e['diagnosis'].get('fix')}"
    elif e["error"]:
        note = e["error"][:160]
    if e["is_baseline"]:
        note = "starting code" + (f"; {note}" if note else "")
    if e["status"] != "succeeded":
        score = None
    return B.ResultRow(label=e["label"], config=e["config"], status=e["status"], score=score, note=note,
                       hypothesis=e["hypothesis"] or None)


def load_digests(n: int) -> list[dict[str, Any]]:
    c = ro()
    per_race = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    out: list[dict[str, Any]] = []
    for race, take in zip(R48, per_race):
        cfg = json.loads(c.execute("select config_json from races where id=?", (race,)).fetchone()[0])
        spec = race_spec(cfg)
        arm_id, a_start = c.execute("select id, started_at from arms where race_id=? and kind='opus'", (race,)).fetchone()
        rstart, budget = c.execute("select started_at, budget_s from races where id=?", (race,)).fetchone()
        exps = []
        for (eid, cj, hyp, status, pa, sa, ea, score, verdict, diag, err, base) in c.execute(
                "select id, config_json, hypothesis, status, proposed_at, started_at, ended_at, score, judge_verdict, "
                "diagnosis_json, error, is_baseline from experiments where arm_id=? order by proposed_at, started_at, rowid",
                (arm_id,)):
            exps.append(dict(id=eid, config=json.loads(cj), hypothesis=hyp, status=status, proposed_at=pa, ended_at=ea,
                             score=score, verdict=verdict, diagnosis=json.loads(diag) if diag else None, error=err,
                             is_baseline=bool(base)))
        for i, e in enumerate(exps):
            e["label"] = f"#{i + 1}"
        calls = c.execute("select id, created_at, latency_ms, input_tokens, cache_read_tokens, cache_creation_tokens, "
                          "output_tokens, experiment_ids_json from llm_calls where arm_id=? and "
                          "step_type='propose_experiments' and valid=1 order by created_at", (arm_id,)).fetchall()
        calls = [x for x in calls if len(json.loads(x[7] or "[]")) == 8]
        # evenly spaced over the race, skipping the very first call (empty history)
        idx = [round(1 + j * (len(calls) - 2) / max(1, take - 1)) for j in range(take)]
        for j in idx:
            cid, t, lat, inp, cr, cw, outp, ids = calls[j]
            done = sorted((e for e in exps if e["ended_at"] is not None and e["ended_at"] <= t),
                          key=lambda e: e["ended_at"])
            running = [e["config"] for e in exps
                       if e["proposed_at"] is not None and e["proposed_at"] < t and e not in done]
            history = [_row(e) for e in done]
            time_left = max(0.0, rstart + budget - t)
            out.append(dict(race=race[:8], call_id=cid, t_rel=round(t - rstart, 1), spec=spec, history=history,
                            running=running, time_left=time_left,
                            stored=dict(latency_ms=lat, input=inp, cache_read=cr, cache_write=cw, output=outp,
                                        k=len(json.loads(ids)))))
    return out


# --------------------------------------------------------------------------- metrics helpers


def canon(cfg: dict[str, Any], spec: TargetSpec) -> tuple[Any, ...]:
    return tuple((k, B.fmt_value(cfg[k])) for k in spec.config_space)


def distance(cfg: dict[str, Any], base: dict[str, Any], spec: TargetSpec) -> tuple[int, float]:
    """(params changed, normalized L1: numeric |delta| / range, log-space for log params; 1 per changed cat/bool)."""
    ham, l1 = 0, 0.0
    for k, p in spec.config_space.items():
        a, b = cfg[k], base[k]
        if B.fmt_value(a) == B.fmt_value(b):
            continue
        ham += 1
        if p.type in ("float", "int") and p.low is not None and p.high is not None:
            if p.log and p.low > 0:
                l1 += abs(math.log(a) - math.log(b)) / (math.log(p.high) - math.log(p.low))
            else:
                l1 += abs(a - b) / ((p.high - p.low) or 1)
        else:
            l1 += 1.0
    return ham, l1


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = q * (len(xs) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


# --------------------------------------------------------------------------- main


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--digests", type=int, default=20)
    ap.add_argument("--k2", type=int, default=6, help="digests also sent at K=2 (latency-vs-K fit)")
    ap.add_argument("--max-usd", type=float, default=8.0)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v]
    OUT.mkdir(parents=True, exist_ok=True)
    digests = load_digests(args.digests)
    print(f"{len(digests)} digests; fingerprints: " + ", ".join(
        f"{v}={prompt_fingerprint(STEP, VARIANTS[v][0])}" for v in variants), flush=True)

    m = MubitClient()
    sdk = m._client_for(m.project_for("opus", STEP))
    spent = 0.0
    rng = random.Random(args.seed)
    jobs: list[tuple[int, int, str]] = []
    for i in range(len(digests)):
        order = variants[:]
        rng.shuffle(order)
        jobs += [(i, 8, v) for v in order]
    k2_idx = list(range(0, len(digests), max(1, len(digests) // max(1, args.k2))))[:args.k2]
    for i in k2_idx:
        order = variants[:]
        rng.shuffle(order)
        jobs += [(i, 2, v) for v in order]

    calls_path = OUT / "calls.jsonl"
    fh = calls_path.open("w", encoding="utf-8")
    last_cost: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for n_job, (i, k, vname) in enumerate(jobs):
        d = digests[i]
        version, effort = VARIANTS[vname]
        spec: TargetSpec = d["spec"]
        reserve = last_cost.get(vname, 0.06)
        if spent + reserve > args.max_usd:
            print(f"!! cost guard ${spent:.3f} + ~${reserve:.3f} > ${args.max_usd}; stopping", flush=True)
            break
        user = B.build_propose_user(spec, d["history"], k, running=d["running"], time_left_s=d["time_left"])
        system = load_system_prompt(STEP, version)
        wire = B.propose_wire_schema(spec, version)
        base = B.propose_base_config(spec, d["history"])
        kwargs = m.build_request(STEP, system, user, wire, f"probe-propose-lean/{vname}", OPUS_MODEL, effort, 8000)
        rec: dict[str, Any] = dict(job=n_job, digest=i, race=d["race"], src_call=d["call_id"], t_rel=d["t_rel"],
                                   variant=vname, version=version, effort=effort, k=k,
                                   fingerprint=prompt_fingerprint(STEP, version), recorded=False)
        t0 = time.perf_counter()
        try:
            raw = await sdk.messages.with_raw_response.create(**kwargs, extra_headers=HEADERS)
            msg = await raw.parse()
            rec["request_id"] = raw.headers.get("x-middleware-request-id")
        except Exception as e:  # noqa: BLE001
            rec.update(error=f"{type(e).__name__}: {str(e)[:300]}", latency_ms=(time.perf_counter() - t0) * 1000)
            print(json.dumps(rec), flush=True)
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            rows.append(rec)
            if "http 4" in rec["error"] or "BadRequest" in rec["error"] or "Authentication" in rec["error"]:
                break
            continue
        rec["latency_ms"] = (time.perf_counter() - t0) * 1000
        u = msg.usage
        rec.update(input=u.input_tokens, cache_write=u.cache_creation_input_tokens or 0,
                   cache_read=u.cache_read_input_tokens or 0, output=u.output_tokens, stop=msg.stop_reason)
        rec["cost_usd"] = estimate_cost_usd(OPUS_MODEL, rec["input"], rec["cache_write"], rec["cache_read"], rec["output"])
        spent += rec["cost_usd"]
        last_cost[vname] = max(last_cost.get(vname, 0.0), rec["cost_usd"])
        text = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), None)
        rec["text"] = text
        rec["block_types"] = [getattr(b, "type", "?") for b in msg.content]
        if text:
            ct = await sdk.messages.count_tokens(model=OPUS_MODEL, messages=[{"role": "user", "content": text}],
                                                 extra_headers=HEADERS)
            ct0 = await sdk.messages.count_tokens(model=OPUS_MODEL, messages=[{"role": "user", "content": "x"}],
                                                  extra_headers=HEADERS)
            rec["visible_tokens"] = ct.input_tokens - ct0.input_tokens + 1
            rec["thinking_tokens"] = rec["output"] - rec["visible_tokens"]
        schema = B.ProposeDiffOutput if B.is_diff_version(version) else ProposeOutput
        n_raw = 0
        try:
            n_raw = len(json.loads(text or "{}").get("experiments", []))
        except ValueError:
            pass
        rec["n_returned"] = min(n_raw, k)
        parsed, err = parse_and_validate(text, schema, B.propose_validator(spec, k, base=base))
        rec["parse_error"] = err
        exps = parsed.experiments if parsed is not None else []
        rec["n_valid"] = len(exps)
        hist = {canon(r.config, spec) for r in d["history"]} | {canon(cfg, spec) for cfg in d["running"]}
        seen: set[tuple[Any, ...]] = set()
        props = []
        for e in exps:
            key = canon(e.config, spec)
            ham, l1 = distance(e.config, base, spec)
            props.append(dict(dup_hist=key in hist, dup_batch=key in seen, same_as_best=ham == 0, ham=ham, l1=l1,
                              trap=bool(spec.is_trap(e.config)), words=len(e.hypothesis.split()),
                              hypothesis=e.hypothesis))
            seen.add(key)
        rec["proposals"] = props
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        rows.append(rec)
        print(f"[{n_job + 1}/{len(jobs)}] d{i:02d} K={k} {vname:7s} {rec['latency_ms']/1000:6.2f}s out={rec['output']:5d} "
              f"vis={rec.get('visible_tokens')} valid={rec['n_valid']}/{k} ${rec['cost_usd']:.4f} total ${spent:.3f}",
              flush=True)
    fh.close()
    await m.aclose()

    # ---- summary
    summ: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "digests": len(digests),
                            "spent_usd": spent, "record": "off", "model": OPUS_MODEL, "variants": {}}
    stored = [d["stored"] for d in digests]
    summ["stored_r48_same_digests"] = {
        "latency_p50_s": st.median(s["latency_ms"] for s in stored) / 1000,
        "output_per_proposal": st.fmean(s["output"] / s["k"] for s in stored),
        "input_uncached_p50": st.median(s["input"] for s in stored)}
    for v in variants:
        for k in (8, 2):
            rs = [r for r in rows if r["variant"] == v and r["k"] == k and "output" in r]
            if not rs:
                continue
            props = [p for r in rs for p in r["proposals"]]
            nv = sum(r["n_valid"] for r in rs)
            s = {
                "calls": len(rs), "errors": sum(1 for r in rows if r["variant"] == v and r["k"] == k and "error" in r),
                "latency_p50_s": st.median(r["latency_ms"] for r in rs) / 1000,
                "latency_p90_s": pct([r["latency_ms"] for r in rs], 0.9) / 1000,
                "latency_mean_s": st.fmean(r["latency_ms"] for r in rs) / 1000,
                "output_per_call_mean": st.fmean(r["output"] for r in rs),
                "output_per_valid_proposal": sum(r["output"] for r in rs) / max(1, nv),
                "visible_per_valid_proposal": sum(r.get("visible_tokens", 0) for r in rs) / max(1, nv),
                "thinking_per_call_mean": st.fmean(r.get("thinking_tokens", 0) for r in rs),
                "usd_per_call": st.fmean(r["cost_usd"] for r in rs),
                "usd_per_valid_proposal": sum(r["cost_usd"] for r in rs) / max(1, nv),
                "output_usd_per_valid_proposal": sum(r["output"] for r in rs) * 20 / 1e6 / max(1, nv),
                "input_uncached_p50": st.median(r["input"] for r in rs),
                "valid_calls": sum(1 for r in rs if r["n_valid"] > 0),
                "slot_valid_rate": nv / (k * len(rs)),
                "returned_rate": sum(r["n_returned"] for r in rs) / (k * len(rs)),
                "dropped_invalid": sum(r["n_returned"] - r["n_valid"] for r in rs),
                "dup_hist_rate": sum(p["dup_hist"] for p in props) / max(1, len(props)),
                "dup_batch_rate": sum(p["dup_batch"] for p in props) / max(1, len(props)),
                "dup_any_rate": sum(p["dup_hist"] or p["dup_batch"] for p in props) / max(1, len(props)),
                "same_as_best": sum(p["same_as_best"] for p in props),
                "ham_mean": st.fmean(p["ham"] for p in props) if props else None,
                "ham_ge3_share": sum(p["ham"] >= 3 for p in props) / max(1, len(props)),
                "l1_mean": st.fmean(p["l1"] for p in props) if props else None,
                "trap_rate": sum(p["trap"] for p in props) / max(1, len(props)),
                "hyp_words_mean": st.fmean(p["words"] for p in props) if props else None,
                "stop_reasons": sorted({str(r["stop"]) for r in rs}),
            }
            summ["variants"].setdefault(v, {})[f"K{k}"] = s
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
