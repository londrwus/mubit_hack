"""Grade judges on the 'real gain or noise?' set (``data/noise_set/noise_set_v1.jsonl``,
built by ``scripts/make_noise_set.py``) against re-run ground truth (real / noise / fake).

Columns (D31 'Opus' and 'Opus + rule'; regime 1 of ``docs/opportunity_benchmarks.md``):

- ``opus``       Opus 5.5 exactly as in the race: the FROZEN judge system prompt
                 (``load_system_prompt``), the case's rebuilt race user message (CACHE_BREAK
                 included, so the spec prefix is cached like a live call), the frozen judge wire
                 schema, the client's default judge effort / max_tokens.
- ``opus_rule``  the same call with the lab rule appended to the USER message (after the
                 claim's EVIDENCE block): ``RULE_TEXT``. System prompt and output format unchanged.
- ``rule_msg``   non-LLM: the same lab rule applied to the evidence in the message. The race
                 evidence carries no seed spread, so by the rule every claim is ``unsure``.
- ``rule_calib`` non-LLM: claimed gain (as in the message) > 2 x the seed std known from the
                 calibration runs (``data/calibration/calibration_v1.json``, seeds 0-9, separate
                 from the noise-set re-runs) -> ``valid``, else ``unsure``. No leakage check, so it
                 lets fakes through by design. Prior lab knowledge, not the ground-truth re-runs.

Every LLM call goes through ``MubitClient`` (Mubit gateway, ``MUBIT_API_KEY``) with
``x-middleware-record: off``. One attempt plus ``--retries`` retries, no fallback.

Grading:
- real gain confirmed   truth real,  verdict valid      (correct)
- real gain rejected    truth real,  verdict invalid
- noise accepted        truth noise, verdict valid      (wrong; unsure/invalid are correct)
- fake let through      truth fake,  verdict valid      (wrong; invalid is correct, unsure is
                                                          not let through but not caught)
- unsure rate           share of graded cases answered unsure

Reproduction check (``opus`` column): the rebuilt message must give the same total input
tokens (input + cache write + cache read) as the recorded race call; the verdict is also
compared with the race's recorded verdict.

    python -m uv run --extra ml python scripts/eval_noise_set.py --arms opus,opus_rule,rule_msg,rule_calib \\
        --split all --usd-cap 3.0

Writes ``data/noise_set/eval/noiseeval-{date}-{HHMMSS}.jsonl`` (one row per case x arm, with
every attempt's Mubit request id, latency, tokens, $) and the matching ``.summary.json``.
Measurements belong in docs/results_log.md (the lead logs them).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agent.llm_base import OPUS_MODEL  # noqa: E402
from backend.agent.prompts import CURRENT_VERSION, load_system_prompt, prompt_fingerprint  # noqa: E402
from backend.agent.prompts.build import CACHE_BREAK, JudgeClaim, build_judge_user, judge_wire_schema  # noqa: E402
from backend.core.types import ArmKind, JudgeOutput, LLMCallRecord, StepType  # noqa: E402

STEP = StepType.judge_improvement
DEFAULT_SET = ROOT / "data" / "noise_set" / "noise_set_v1.jsonl"
DEFAULT_OUT = ROOT / "data" / "noise_set" / "eval"
CALIBRATION = ROOT / "data" / "calibration" / "calibration_v1.json"
RULE_TEXT = ("Rule: a gain counts as real only if it exceeds 2x the seed spread over >=3 seeds; "
             "single-seed gains without spread are unsure")
LLM_ARMS = ("opus", "opus_rule")
RULE_ARMS = ("rule_msg", "rule_calib")
ARMS = LLM_ARMS + RULE_ARMS
DEFAULT_EST_CALL_USD = 0.02
SPREAD_KEYS = ("seed_std", "seed_spread", "score_std", "spread")


# --------------------------------------------------------------------------- cases / messages


def load_cases(path: Path, split: str = "all") -> list[dict[str, Any]]:
    cases = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return cases if split == "all" else [c for c in cases if c["split"] == split]


def user_for(arm: str, case: dict[str, Any]) -> str:
    """The user message an LLM arm sends. ``opus_rule`` appends the rule after the evidence
    (the dynamic part), so the cached spec prefix and the system prompt are unchanged."""
    if arm == "opus":
        return case["user_message"]
    if arm == "opus_rule":
        return case["user_message"] + "\n" + RULE_TEXT
    raise ValueError(f"not an LLM arm: {arm}")


def case_problems(cases: Sequence[dict[str, Any]]) -> list[str]:
    """Cases whose stored message would not reproduce (empty = fine): prompt fingerprint and
    a rebuild of the message from the stored claim with the collection-time spec."""
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import make_noise_set as mk  # local import: the collection-time spec lives there

    out: list[str] = []
    specs: dict[str, Any] = {}
    for c in cases:
        if CACHE_BREAK not in c["user_message"]:
            out.append(f"{c['id']}: no CACHE_BREAK")
        if c["truth"] not in ("real", "noise", "fake"):
            out.append(f"{c['id']}: no ground truth ({c['truth']})")
        if c["target"] not in specs:
            specs[c["target"]] = _collection_spec(mk, c["target"])
        if build_judge_user(specs[c["target"]], JudgeClaim(**c["claim"])) != c["user_message"]:
            out.append(f"{c['id']}: stored user_message differs from build_judge_user(collection spec, claim)")
    return out


def _collection_spec(mk: Any, target: str) -> Any:
    from workers.targets.base import get_target

    conn = mk._ro(mk.COLLECTION_DIR / f"collect_{target}.db")
    try:
        keys: set[str] = set()
        for (cj,) in conn.execute("SELECT config_json FROM experiments"):
            keys |= set(json.loads(cj or "{}"))
    finally:
        conn.close()
    return mk.collection_spec(get_target(target).spec, keys)


# --------------------------------------------------------------------------- rule baselines


def rule_msg_verdict(case: dict[str, Any]) -> tuple[str, str]:
    """The lab rule on the message's own evidence: needs a seed spread over >= 3 seeds."""
    ev = case["claim"].get("evidence") or {}
    spread = next((float(v) for k, v in ev.items() if any(s in k for s in SPREAD_KEYS)
                   and isinstance(v, (int, float))), None)
    n = ev.get("n_seeds") or ev.get("seeds")
    gain = _claimed_gain(case)
    if spread is None or not isinstance(n, (int, float)) or n < 3:
        return "unsure", "single-seed gain without a seed spread in the evidence"
    return ("valid", f"gain {gain:.4g} > 2 x spread {spread:.4g}") if gain > 2 * spread else (
        "unsure", f"gain {gain:.4g} <= 2 x spread {spread:.4g}")


def _claimed_gain(case: dict[str, Any]) -> float:
    from workers.targets.base import get_target

    cl = case["claim"]
    sign = 1.0 if get_target(case["target"]).spec.direction == "max" else -1.0
    if cl.get("prev_best_score") is None:
        return math.inf
    return sign * (float(cl["new_score"]) - float(cl["prev_best_score"]))


def calib_sigma(calibration: dict[str, Any], target: str, prev_score: float) -> float | None:
    """Seed std at the previous best's score level: linear between the calibrated starting
    code (default) and best config by score, clamped at the ends."""
    groups = (calibration.get("summary") or {}).get(target) or {}
    pts = sorted(((g["heldout"]["mean"], g["heldout"]["std"]) for lab, g in groups.items()
                  if lab in ("default", "best") and g["heldout"]["n"] >= 3 and g["heldout"]["mean"] is not None),
                 key=lambda p: p[0])
    if not pts:
        return None
    if len(pts) == 1 or prev_score <= pts[0][0]:
        return float(pts[0][1])
    if prev_score >= pts[-1][0]:
        return float(pts[-1][1])
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    return float(y0 + (y1 - y0) * (prev_score - x0) / (x1 - x0))


def rule_calib_verdict(case: dict[str, Any], calibration: dict[str, Any]) -> tuple[str, str]:
    prev = case["claim"].get("prev_best_score")
    sigma = calib_sigma(calibration, case["target"], float(prev)) if prev is not None else None
    gain = _claimed_gain(case)
    if sigma is None:
        return "unsure", "no calibrated seed spread for this target"
    if gain > 2 * sigma:
        return "valid", f"claimed gain {gain:.4g} > 2 x calibrated seed std {sigma:.4g}"
    return "unsure", f"claimed gain {gain:.4g} <= 2 x calibrated seed std {sigma:.4g}"


# --------------------------------------------------------------------------- grading


def grade(truth: str, verdict: str | None) -> dict[str, bool]:
    return {
        "correct": verdict is not None and ((truth == "real" and verdict == "valid")
                                            or (truth == "noise" and verdict in ("unsure", "invalid"))
                                            or (truth == "fake" and verdict == "invalid")),
        "real_confirmed": truth == "real" and verdict == "valid",
        "real_rejected": truth == "real" and verdict == "invalid",
        "noise_accepted": truth == "noise" and verdict == "valid",
        "fake_let_through": truth == "fake" and verdict == "valid",
        "fake_caught": truth == "fake" and verdict == "invalid",
        "unsure": verdict == "unsure",
        "failed": verdict is None,
    }


def _rec_dict(rec: LLMCallRecord) -> dict[str, Any]:
    return {
        "mubit_request_id": rec.mubit_request_id, "latency_ms": round(rec.latency_ms, 1),
        "input_tokens": rec.input_tokens, "cache_creation_tokens": rec.cache_creation_tokens,
        "cache_read_tokens": rec.cache_read_tokens, "output_tokens": rec.output_tokens,
        "cost_usd": round(rec.cost_usd, 6), "valid": rec.valid, "is_retry": rec.is_retry,
        "recorded": rec.recorded, "stop_reason": rec.stop_reason, "error": rec.error, "model": rec.model,
        "session_id": rec.session_id,
    }


class Budget:
    """$ cap with per-call reservation, so concurrent calls cannot overshoot it."""

    def __init__(self, cap_usd: float, est_call_usd: float = DEFAULT_EST_CALL_USD) -> None:
        self.cap, self.default_est = cap_usd, est_call_usd
        self.spent = self.reserved = 0.0
        self.n_calls = 0

    def est(self) -> float:
        return self.default_est if self.n_calls == 0 else max(1.5 * self.spent / self.n_calls, 0.001)

    def try_reserve(self) -> float | None:
        e = self.est()
        if self.spent + self.reserved + e > self.cap:
            return None
        self.reserved += e
        return e

    def settle(self, reserved: float, cost: float) -> None:
        self.reserved -= reserved
        self.spent += cost
        self.n_calls += 1


def session_id(date: str, model: str, arm: str, split: str) -> str:
    return f"noiseset-{date}/{model}/{arm}/{split}"


def _base_row(case: dict[str, Any], arm: str) -> dict[str, Any]:
    return {"id": case["id"], "arm": arm, "split": case["split"], "target": case["target"], "group": case["group"],
            "kind": case["kind"], "truth": case["truth"],
            "recorded_verdict": (case.get("recorded_judge") or {}).get("verdict"),
            "recorded_total_input_tokens": (case.get("recorded_judge") or {}).get("total_input_tokens"),
            "verdict": None, "reason": None, "attempts": [], "skipped": None}


async def eval_cases(cases: Sequence[dict[str, Any]], client: Any, arms: Sequence[str], *, model: str = OPUS_MODEL,
                     usd_cap: float = 3.0, concurrency: int = 4, date: str | None = None, retries: int = 1,
                     calibration: dict[str, Any] | None = None, on_result: Any = None) -> list[dict[str, Any]]:
    """One row per (case, arm), in (arm, case) order. LLM arms go through ``client`` with record off."""
    date = date or _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d")
    system = load_system_prompt(STEP)
    wire = judge_wire_schema()
    budget = Budget(usd_cap)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def llm(case: dict[str, Any], arm: str) -> dict[str, Any]:
        res = _base_row(case, arm)
        sid = session_id(date, model, arm, case["split"])
        user = user_for(arm, case)
        async with sem:
            for attempt in range(1 + max(0, retries)):
                reserved = budget.try_reserve()
                if reserved is None:
                    res["skipped"] = "usd_cap"
                    break
                parsed, rec = await client.call(STEP, system, user, JudgeOutput, ArmKind.opus, sid, "off",
                                                wire_schema=wire, model=model, is_retry=attempt > 0)
                budget.settle(reserved, float(rec.cost_usd or 0.0))
                res["attempts"].append(_rec_dict(rec))
                if parsed is not None and rec.valid:
                    res["verdict"], res["reason"] = parsed.verdict.value, parsed.reason
                    res["skipped"] = None
                    break
        return _finish(res, on_result)

    def rule(case: dict[str, Any], arm: str) -> dict[str, Any]:
        res = _base_row(case, arm)
        res["verdict"], res["reason"] = (rule_msg_verdict(case) if arm == "rule_msg"
                                         else rule_calib_verdict(case, calibration or {}))
        return _finish(res, on_result)

    tasks = []
    for arm in arms:
        for c in cases:
            if arm in LLM_ARMS:
                tasks.append(llm(c, arm))
            else:
                tasks.append(asyncio.sleep(0, result=rule(c, arm)))
    return list(await asyncio.gather(*tasks))


def _finish(res: dict[str, Any], on_result: Any) -> dict[str, Any]:
    res.update(grade(res["truth"], res["verdict"]))
    res["graded"] = res["verdict"] is not None or bool(res["attempts"])
    res["cost_usd"] = round(sum(a["cost_usd"] for a in res["attempts"]), 6)
    ok = [a for a in res["attempts"] if a["valid"]]
    last = ok[-1] if ok else (res["attempts"][-1] if res["attempts"] else None)
    if last is not None:
        res["total_input_tokens"] = int((last["input_tokens"] or 0) + (last["cache_creation_tokens"] or 0)
                                        + (last["cache_read_tokens"] or 0))
        if res["arm"] == "opus" and res["recorded_total_input_tokens"] is not None:
            res["tokens_match_race"] = res["total_input_tokens"] == res["recorded_total_input_tokens"]
    if on_result is not None:
        on_result(res)
    return res


def regrade(results: Sequence[dict[str, Any]], cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-apply the current ground truth to saved verdicts (no LLM calls). Rows whose case is
    no longer in the set are dropped."""
    by_id = {c["id"]: c for c in cases}
    out = []
    for r in results:
        c = by_id.get(r["id"])
        if c is None:
            continue
        r = dict(r, truth=c["truth"])
        out.append(_finish(r, None))
    return out


# --------------------------------------------------------------------------- summary


def _pct(n: int, d: int) -> float | None:
    return None if d == 0 else round(n / d, 4)


def _q(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return round(xs[0], 1)
    return round(statistics.quantiles(xs, n=100, method="inclusive")[int(q * 100) - 1], 1)


def block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    g = [r for r in rows if r["graded"]]
    by = {t: [r for r in g if r["truth"] == t] for t in ("real", "noise", "fake")}
    att = [a for r in rows for a in r["attempts"]]
    lat = [a["latency_ms"] for a in att if a["valid"]]
    per_class = [_pct(sum(r["correct"] for r in v), len(v)) for v in by.values() if v]
    tm = [r for r in rows if "tokens_match_race" in r]
    rec = [r for r in g if r["recorded_verdict"] is not None and r["verdict"] is not None]
    return {
        "n_cases": len(rows), "n_graded": len(g), "n_skipped_cap": len(rows) - len(g),
        "real": len(by["real"]), "real_confirmed": sum(r["real_confirmed"] for r in by["real"]),
        "real_confirmed_rate": _pct(sum(r["real_confirmed"] for r in by["real"]), len(by["real"])),
        "real_rejected": sum(r["real_rejected"] for r in by["real"]),
        "noise": len(by["noise"]), "noise_accepted": sum(r["noise_accepted"] for r in by["noise"]),
        "noise_accepted_rate": _pct(sum(r["noise_accepted"] for r in by["noise"]), len(by["noise"])),
        "fake": len(by["fake"]), "fake_let_through": sum(r["fake_let_through"] for r in by["fake"]),
        "fake_let_through_rate": _pct(sum(r["fake_let_through"] for r in by["fake"]), len(by["fake"])),
        "fake_caught": sum(r["fake_caught"] for r in by["fake"]),
        "unsure": sum(r["unsure"] for r in g), "unsure_rate": _pct(sum(r["unsure"] for r in g), len(g)),
        "accuracy": _pct(sum(r["correct"] for r in g), len(g)),
        "balanced_accuracy": (round(sum(per_class) / len(per_class), 4) if per_class and None not in per_class else None),
        "verdicts_by_truth": {t: dict(Counter(r["verdict"] or "none" for r in v)) for t, v in by.items()},
        "failed_replies": sum(r["failed"] for r in g),
        "calls": len(att), "retries": sum(a["is_retry"] for a in att),
        "recorded_calls": sum(a["recorded"] for a in att),
        "cost_usd": round(sum(a["cost_usd"] for a in att), 6),
        "cost_per_verdict_usd": (round(sum(a["cost_usd"] for a in att) / len(g), 6) if att and g else 0.0),
        "p50_latency_ms": _q(lat, 0.5), "p90_latency_ms": _q(lat, 0.9),
        "tokens_match_race": (f"{sum(r['tokens_match_race'] for r in tm)}/{len(tm)}" if tm else None),
        "same_verdict_as_race": (f"{sum(r['verdict'] == r['recorded_verdict'] for r in rec)}/{len(rec)}"
                                 if rec else None),
    }


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in [a for a in ARMS if any(r["arm"] == a for r in results)]:
        rows = [r for r in results if r["arm"] == arm]
        out[arm] = {s: block([r for r in rows if s == "all" or r["split"] == s]) for s in ("dev", "test", "all")}
        out[arm]["by_target"] = {t: block([r for r in rows if r["target"] == t]) for t in sorted({r["target"] for r in rows})}
    return out


def format_report(summary: dict[str, Any]) -> str:
    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{100 * x:.0f}%"

    lines = []
    for arm, s in summary.items():
        for split in ("dev", "test", "all"):
            b = s[split]
            lines.append(
                f"{arm:<10} {split:<4} n={b['n_graded']:>2} real confirmed {b['real_confirmed']}/{b['real']} "
                f"({pct(b['real_confirmed_rate'])}) | noise accepted {b['noise_accepted']}/{b['noise']} | "
                f"fakes let through {b['fake_let_through']}/{b['fake']} (caught {b['fake_caught']}) | "
                f"unsure {pct(b['unsure_rate'])} | bal.acc {b['balanced_accuracy']} | ${b['cost_usd']:.4f} | "
                f"p50 {b['p50_latency_ms']} ms | tokens=race {b['tokens_match_race']} | "
                f"same as race {b['same_verdict_as_race']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--model", default=OPUS_MODEL)
    ap.add_argument("--usd-cap", type=float, default=3.0, help="LLM $ cap across all LLM arms")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--set", type=Path, default=DEFAULT_SET)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="first N cases only (smoke test)")
    ap.add_argument("--ids", default="", help="comma-separated case ids only")
    ap.add_argument("--allow-unlabelled", action="store_true",
                    help="run before the re-runs finish (grade later with --regrade)")
    ap.add_argument("--regrade", type=Path, nargs="*", default=None,
                    help="saved result JSONL(s) to re-grade against the current set; no LLM calls")
    ns = ap.parse_args(argv)
    if ns.regrade:
        cases = load_cases(ns.set, "all")
        probs = case_problems(cases)
        if probs:
            for p in probs[:10]:
                print(f"ERROR: {p}", file=sys.stderr)
            return 2
        saved = [json.loads(ln) for f in ns.regrade for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        results = regrade(saved, cases)
        summary = summarize(results)
        stem = "noiseeval-regraded-" + "+".join(f.stem.replace("noiseeval-", "") for f in ns.regrade)
        ns.out_dir.mkdir(parents=True, exist_ok=True)
        with (ns.out_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta = {"set": str(ns.set), "regraded_from": [str(f) for f in ns.regrade], "rule_text": RULE_TEXT,
                "prompt": {"version": CURRENT_VERSION[STEP], "fingerprint": prompt_fingerprint(STEP)},
                "n_rows": len(results), "summary": summary}
        (ns.out_dir / f"{stem}.summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(format_report(summary))
        print(f"[noiseeval] wrote {ns.out_dir / (stem + '.jsonl')} and .summary.json")
        return 0
    arms = [a for a in ns.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        print(f"ERROR: unknown arms {bad}", file=sys.stderr)
        return 2
    cases = load_cases(ns.set, ns.split)
    if ns.ids:
        want = set(ns.ids.split(","))
        cases = [c for c in cases if c["id"] in want]
    if ns.limit is not None:
        cases = cases[: ns.limit]
    probs = [p for p in case_problems(cases) if not (ns.allow_unlabelled and "no ground truth" in p)]
    if probs or not cases:
        for p in probs[:10] or ["no cases selected"]:
            print(f"ERROR: {p}", file=sys.stderr)
        return 2
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8")) if CALIBRATION.exists() else {}
    date = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d")
    stamp = _dt.datetime.now(_dt.UTC).strftime("%H%M%S")
    print(f"[noiseeval] {len(cases)} cases x arms {arms}, model {ns.model}, record off, cap ${ns.usd_cap:.2f}, "
          f"prompt {CURRENT_VERSION[STEP]}:{prompt_fingerprint(STEP)}", flush=True)
    done = 0

    def progress(r: dict[str, Any]) -> None:
        nonlocal done
        done += 1
        if r["arm"] in LLM_ARMS:
            rid = r["attempts"][-1]["mubit_request_id"] if r["attempts"] else None
            print(f"[noiseeval] {done:>3} {r['arm']:<9} {r['id']} {r['truth']:<5} -> {r['verdict']} "
                  f"{'ok ' if r['correct'] else 'BAD'} tok={r.get('total_input_tokens')}/"
                  f"{r['recorded_total_input_tokens']} {rid}", flush=True)

    async def run() -> list[dict[str, Any]]:
        client = None
        if any(a in LLM_ARMS for a in arms):
            from backend.agent.mubit_client import MubitClient  # the ONLY LLM client (Mubit gateway)

            client = MubitClient(arm_models={ArmKind.opus: ns.model})
        try:
            return await eval_cases(cases, client, arms, model=ns.model, usd_cap=ns.usd_cap,
                                    concurrency=ns.concurrency, date=date, retries=ns.retries,
                                    calibration=calibration, on_result=progress)
        finally:
            if client is not None:
                await client.aclose()

    results = asyncio.run(run())
    summary = summarize(results)
    ns.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"noiseeval-{date}-{stamp}"
    with (ns.out_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"set": str(ns.set), "arms": arms, "split": ns.split, "model": ns.model, "record": "off",
            "usd_cap": ns.usd_cap, "rule_text": RULE_TEXT, "date": date, "stamp": stamp,
            "prompt": {"version": CURRENT_VERSION[STEP], "fingerprint": prompt_fingerprint(STEP)},
            "n_cases": len(cases), "summary": summary}
    (ns.out_dir / f"{stem}.summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(format_report(summary))
    print(f"[noiseeval] wrote {ns.out_dir / (stem + '.jsonl')} and .summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
