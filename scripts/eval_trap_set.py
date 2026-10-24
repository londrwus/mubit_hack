"""Grade a judge model on the contamination-trap set (benchmark.md Claim 2).

Sends every case of ``data/trap_set/trap_set_v1.jsonl`` to ``judge_improvement`` exactly as
the race loop does: the FROZEN judge system prompt (``load_system_prompt``), the case's
stored user message (``build_judge_user`` output, CACHE_BREAK included, so the target-spec
prefix is cached like a live call), the frozen judge wire schema (``judge_wire_schema``),
the client's default judge effort/max_tokens (``DEFAULT_STEP_CONFIGS``) and
``metadata.user_id`` = session id ``trapeval-{date}/{model}/{split}``. Every call goes
through ``MubitClient`` (the Mubit gateway, ``MUBIT_API_KEY``).

Grading is against the planted ground truth (``acceptable_verdicts``), not against Opus:
- correct           verdict in ``acceptable_verdicts`` (a failed reply counts as wrong)
- fake approved     a fake case (label ``invalid``) judged ``valid``
- genuine rejected  a genuine case (label ``valid``) judged ``invalid``
- unsure rate       share of graded cases answered ``unsure``

Policy: one attempt plus ``--retries`` retries (same model) on an invalid reply. There is NO
Opus fallback: a fallback would hide the graded model's failures (for an adapter eval) and
the retry is flagged ``is_retry`` on its record.

``--record off`` (default) sends ``x-middleware-record: off``. ``--record full`` makes the
calls Mubit examples for the judge task and is refused unless the judge prompt is FROZEN and
every case was rendered with the current prompt fingerprint and builder. Use it on ``dev``
only; ``test`` is never recorded (refused) so it stays a clean held-out split.

Usage::

    python -m uv run --extra ml python scripts/eval_trap_set.py --model claude-opus-5-5 \\
        --split dev --record full --usd-cap 1.0
    python -m uv run --extra ml python scripts/eval_trap_set.py --model claude-opus-5-5 \\
        --split test --usd-cap 1.0

Per-case results (verdict, reason, every attempt's Mubit request id, latency, tokens, $) go
to ``data/trap_eval/trapeval-{date}-{model}-{split}-{HHMMSS}.jsonl`` and the summary to the
matching ``.summary.json``. Measurements belong in docs/results_log.md (the lead logs them).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agent.llm_base import OPUS_MODEL  # noqa: E402
from backend.agent.prompts import CURRENT_VERSION, FROZEN, load_system_prompt, prompt_fingerprint  # noqa: E402
from backend.agent.prompts.build import CACHE_BREAK, JudgeClaim, build_judge_user, judge_wire_schema  # noqa: E402
from backend.core.types import ArmKind, JudgeOutput, LLMCallRecord, StepType  # noqa: E402

STEP = StepType.judge_improvement
DEFAULT_SET = ROOT / "data" / "trap_set" / "trap_set_v1.jsonl"
DEFAULT_OUT = ROOT / "data" / "trap_eval"
SPLITS = ("dev", "test")
# Judge prompt versions whose user message is the same build_judge_user output.
SAME_USER_BUILDER_VERSIONS = ("v0", "v1")
# Reserved per call before the first real cost is known (Opus judge ~ $0.01-0.02 per call).
DEFAULT_EST_CALL_USD = 0.04


# --------------------------------------------------------------------------- cases


def load_cases(path: Path, split: str = "all") -> list[dict[str, Any]]:
    cases = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if split != "all":
        cases = [c for c in cases if c["split"] == split]
    return cases


def case_problems(cases: Iterable[dict[str, Any]], *, rebuild: bool = True) -> list[str]:
    """Why these cases would not reproduce a live judge call (empty = fine).

    Checks the stored prompt fingerprint against the current judge prompt and, with
    ``rebuild``, that ``build_judge_user(spec, claim)`` still renders the stored message."""
    problems: list[str] = []
    fp = prompt_fingerprint(STEP)
    specs: dict[str, Any] = {}
    for c in cases:
        cid = c.get("id", "?")
        p = c.get("prompt") or {}
        # The user message does not depend on the system prompt, so a case built for any judge
        # version that shares build_judge_user (v0, v1) still reproduces a live call.
        if (p.get("step") != STEP.value or p.get("version") not in SAME_USER_BUILDER_VERSIONS
                or p.get("fingerprint") != prompt_fingerprint(STEP, p.get("version"))):
            problems.append(f"{cid}: built for prompt {p}, current is {CURRENT_VERSION[STEP]}:{fp}")
        if CACHE_BREAK not in c.get("user_message", ""):
            problems.append(f"{cid}: user_message has no CACHE_BREAK marker")
        if rebuild:
            if c["target"] not in specs:
                from workers.targets.base import get_target

                specs[c["target"]] = get_target(c["target"]).spec
            if build_judge_user(specs[c["target"]], JudgeClaim(**c["claim"])) != c["user_message"]:
                problems.append(f"{cid}: user_message differs from build_judge_user(spec, claim); rebuild the set")
    return problems


def record_problems(record: str, splits: Iterable[str]) -> list[str]:
    """Why ``--record full`` is not allowed (empty = allowed)."""
    if record == "off":
        return []
    out: list[str] = []
    if FROZEN[STEP] is None:
        out.append(f"{STEP.value}: prompt {CURRENT_VERSION[STEP]} is a DRAFT (FROZEN is None)")
    elif FROZEN[STEP] != CURRENT_VERSION[STEP]:
        out.append(f"{STEP.value}: FROZEN={FROZEN[STEP]} but CURRENT_VERSION={CURRENT_VERSION[STEP]}")
    if "test" in set(splits):
        out.append("the test split is never recorded (it is the held-out split); use --split dev")
    return out


def session_id(date: str, model: str, split: str) -> str:
    """Mubit session per (date, model, split): ``trapeval-{date}/{model}/{split}``."""
    return f"trapeval-{date}/{model}/{split}"


def utc_date() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- grading


def grade(case: dict[str, Any], verdict: str | None) -> dict[str, bool]:
    fake = case["label"] == "invalid"
    return {
        "correct": verdict is not None and verdict in case["acceptable_verdicts"],
        "fake_approved": fake and verdict == "valid",
        "genuine_rejected": (not fake) and verdict == "invalid",
        "unsure": verdict == "unsure",
        "failed": verdict is None,
    }


def _rec_dict(rec: LLMCallRecord) -> dict[str, Any]:
    return {
        "mubit_request_id": rec.mubit_request_id, "latency_ms": round(rec.latency_ms, 1),
        "input_tokens": rec.input_tokens, "cache_creation_tokens": rec.cache_creation_tokens,
        "cache_read_tokens": rec.cache_read_tokens, "output_tokens": rec.output_tokens,
        "cost_usd": round(rec.cost_usd, 6), "valid": rec.valid, "is_retry": rec.is_retry,
        "recorded": rec.recorded, "stop_reason": rec.stop_reason, "error": rec.error,
        "model": rec.model, "session_id": rec.session_id,
    }


class Budget:
    """$ cap with per-call reservation, so concurrent calls cannot overshoot it."""

    def __init__(self, cap_usd: float, est_call_usd: float = DEFAULT_EST_CALL_USD) -> None:
        self.cap = cap_usd
        self.default_est = est_call_usd
        self.spent = 0.0
        self.reserved = 0.0
        self.n_calls = 0

    def est(self) -> float:
        if self.n_calls == 0:
            return self.default_est
        return max(1.5 * self.spent / self.n_calls, 0.001)

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


async def eval_cases(
    cases: Sequence[dict[str, Any]],
    client: Any,
    *,
    model: str,
    record: str = "off",
    usd_cap: float = 1.0,
    concurrency: int = 4,
    date: str | None = None,
    retries: int = 1,
    on_result: Any = None,
) -> list[dict[str, Any]]:
    """Judge every case through ``client`` (a ``MubitClient``). Returns one result per case,
    in input order. Cases skipped by the $ cap have ``skipped: "usd_cap"``."""
    date = date or utc_date()
    system = load_system_prompt(STEP)
    wire = judge_wire_schema()
    budget = Budget(usd_cap)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(case: dict[str, Any]) -> dict[str, Any]:
        sid = session_id(date, model, case["split"])
        res: dict[str, Any] = {
            k: case[k] for k in ("id", "split", "target", "label", "category", "family", "difficulty",
                                 "evidence_profile", "aggregation", "acceptable_verdicts")
        }
        res.update(session_id=sid, verdict=None, reason=None, attempts=[], skipped=None)
        async with sem:
            for attempt in range(1 + max(0, retries)):
                reserved = budget.try_reserve()
                if reserved is None:
                    res["skipped"] = "usd_cap"
                    break
                parsed, rec = await client.call(
                    STEP, system, case["user_message"], JudgeOutput, ArmKind.opus, sid, record,
                    wire_schema=wire, model=model, is_retry=attempt > 0,
                )
                budget.settle(reserved, float(rec.cost_usd or 0.0))
                res["attempts"].append(_rec_dict(rec))
                if parsed is not None and rec.valid:
                    res["verdict"], res["reason"] = parsed.verdict.value, parsed.reason
                    break
        if res["verdict"] is not None:
            res["skipped"] = None
        res.update(grade(case, res["verdict"]))
        res["graded"] = not (res["skipped"] and not res["attempts"])
        res["cost_usd"] = round(sum(a["cost_usd"] for a in res["attempts"]), 6)
        if on_result is not None:
            on_result(res)
        return res

    return list(await asyncio.gather(*(one(c) for c in cases)))


# --------------------------------------------------------------------------- summary


def _pct(num: int, den: int) -> float | None:
    return None if den == 0 else round(num / den, 4)


def _quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return round(xs[0], 1)
    return round(statistics.quantiles(xs, n=100, method="inclusive")[int(q * 100) - 1], 1)


def _block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [r for r in rows if r["graded"]]
    fakes = [r for r in graded if r["label"] == "invalid"]
    reals = [r for r in graded if r["label"] == "valid"]
    attempts = [a for r in rows for a in r["attempts"]]
    lat_ok = [a["latency_ms"] for a in attempts if a["valid"]]
    return {
        "n_cases": len(rows),
        "n_graded": len(graded),
        "n_skipped_cap": sum(1 for r in rows if not r["graded"]),
        "correct": sum(r["correct"] for r in graded),
        "accuracy": _pct(sum(r["correct"] for r in graded), len(graded)),
        "fakes": len(fakes),
        "fake_caught": sum(r["correct"] for r in fakes),
        "fake_approved": sum(r["fake_approved"] for r in fakes),
        "fake_approved_rate": _pct(sum(r["fake_approved"] for r in fakes), len(fakes)),
        "genuine": len(reals),
        "genuine_accepted": sum(r["correct"] for r in reals),
        "genuine_rejected": sum(r["genuine_rejected"] for r in reals),
        "genuine_rejected_rate": _pct(sum(r["genuine_rejected"] for r in reals), len(reals)),
        "unsure": sum(r["unsure"] for r in graded),
        "unsure_rate": _pct(sum(r["unsure"] for r in graded), len(graded)),
        "failed_replies": sum(r["failed"] for r in graded),
        "verdicts": dict(Counter(r["verdict"] or "none" for r in graded)),
        "calls": len(attempts),
        "retries": sum(a["is_retry"] for a in attempts),
        "recorded_calls": sum(a["recorded"] for a in attempts),
        "p50_latency_ms": _quantile(lat_ok, 0.50),
        "p90_latency_ms": _quantile(lat_ok, 0.90),
        "cost_usd": round(sum(a["cost_usd"] for a in attempts), 6),
        "input_tokens": sum(a["input_tokens"] for a in attempts),
        "cache_creation_tokens": sum(a["cache_creation_tokens"] for a in attempts),
        "cache_read_tokens": sum(a["cache_read_tokens"] for a in attempts),
        "output_tokens": sum(a["output_tokens"] for a in attempts),
    }


def _breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r[key])].append(r)
    out = {}
    for k in sorted(groups):
        g = [r for r in groups[k] if r["graded"]]
        out[k] = {"label": groups[k][0]["label"] if key == "category" else None,
                  "n": len(g), "correct": sum(r["correct"] for r in g),
                  "accuracy": _pct(sum(r["correct"] for r in g), len(g)),
                  "verdicts": dict(Counter(r["verdict"] or "none" for r in g))}
    return out


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    splits = sorted({r["split"] for r in results})
    groups = {s: [r for r in results if r["split"] == s] for s in splits}
    if len(splits) > 1:
        groups["all"] = list(results)
    return {
        s: _block(rows) | {"by_category": _breakdown(rows, "category"),
                           "by_difficulty": _breakdown(rows, "difficulty"),
                           "by_target": _breakdown(rows, "target")}
        for s, rows in groups.items()
    }


def format_report(summary: dict[str, Any], results: list[dict[str, Any]], n_ids: int = 5) -> str:
    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{100 * x:.1f}%"

    lines: list[str] = []
    for split, b in summary.items():
        lines += [
            f"== split {split}: {b['n_graded']}/{b['n_cases']} graded"
            + (f" ({b['n_skipped_cap']} skipped by $ cap)" if b["n_skipped_cap"] else ""),
            f"  accuracy          {b['correct']}/{b['n_graded']} = {pct(b['accuracy'])}",
            f"  fake caught       {b['fake_caught']}/{b['fakes']}",
            f"  fake approved     {b['fake_approved']}/{b['fakes']} = {pct(b['fake_approved_rate'])}",
            f"  genuine accepted  {b['genuine_accepted']}/{b['genuine']}",
            f"  genuine rejected  {b['genuine_rejected']}/{b['genuine']} = {pct(b['genuine_rejected_rate'])}",
            f"  unsure            {b['unsure']}/{b['n_graded']} = {pct(b['unsure_rate'])}",
            f"  failed replies    {b['failed_replies']}   calls {b['calls']} (retries {b['retries']}, "
            f"recorded {b['recorded_calls']})",
            f"  latency p50 {b['p50_latency_ms']} ms, p90 {b['p90_latency_ms']} ms",
            f"  cost ${b['cost_usd']:.4f}  tokens in {b['input_tokens']} / cache write "
            f"{b['cache_creation_tokens']} / cache read {b['cache_read_tokens']} / out {b['output_tokens']}",
            "  by category:",
        ]
        for cat, c in b["by_category"].items():
            lines.append(f"    {cat:<28} ({c['label']:<7}) {c['correct']}/{c['n']}  {c['verdicts']}")
        lines.append("  by difficulty: " + ", ".join(f"{k} {v['correct']}/{v['n']}" for k, v in b["by_difficulty"].items()))
    ids = [a["mubit_request_id"] for r in results for a in r["attempts"] if a["mubit_request_id"]]
    lines.append(f"first request ids: {ids[:n_ids]}")
    wrong = [r for r in results if r["graded"] and not r["correct"]]
    if wrong:
        lines.append("wrong cases:")
        for r in wrong:
            lines.append(f"  {r['id']} {r['split']} {r['category']} ({r['label']}) -> {r['verdict']}: "
                         f"{(r['reason'] or (r['attempts'][-1]['error'] if r['attempts'] else ''))[:160]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default=OPUS_MODEL, help="provider model ID as sent through Mubit")
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--record", choices=["full", "off"], default="off",
                   help="full = Mubit records the calls as examples (dev only, frozen prompt)")
    p.add_argument("--usd-cap", type=float, default=1.0, help="stop sending calls past this LLM $")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--retries", type=int, default=1, help="retries on an invalid reply (no fallback)")
    p.add_argument("--set", type=Path, default=DEFAULT_SET, help="trap set JSONL")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=None, help="only the first N cases (smoke test)")
    p.add_argument("--allow-stale", action="store_true",
                   help="run even if cases do not match the current prompt/builder (record off only)")
    return p


async def _run(ns: argparse.Namespace, cases: list[dict[str, Any]], date: str) -> list[dict[str, Any]]:
    from backend.agent.mubit_client import MubitClient  # the ONLY LLM client (Mubit gateway)

    client = MubitClient(arm_models={ArmKind.opus: ns.model})
    done = 0

    def progress(r: dict[str, Any]) -> None:
        nonlocal done
        done += 1
        ok = "ok " if r["correct"] else ("CAP" if not r["graded"] else "BAD")
        rid = r["attempts"][-1]["mubit_request_id"] if r["attempts"] else None
        print(f"[trapeval] {done:>2}/{len(cases)} {ok} {r['id']} {r['category']:<26} -> {r['verdict']}  {rid}",
              flush=True)

    try:
        return await eval_cases(cases, client, model=ns.model, record=ns.record, usd_cap=ns.usd_cap,
                                concurrency=ns.concurrency, date=date, retries=ns.retries, on_result=progress)
    finally:
        await client.aclose()


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    cases = load_cases(ns.set, ns.split)
    if ns.limit is not None:
        cases = cases[: ns.limit]
    if not cases:
        print("ERROR: no cases selected", file=sys.stderr)
        return 2
    errs = record_problems(ns.record, {c["split"] for c in cases})
    stale = case_problems(cases)
    if stale and (ns.record == "full" or not ns.allow_stale):
        errs += stale[:10]
    if errs:
        for e in errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 2
    date = utc_date()
    stamp = _dt.datetime.now(_dt.UTC).strftime("%H%M%S")
    sessions = sorted({session_id(date, ns.model, c["split"]) for c in cases})
    print(f"[trapeval] {len(cases)} cases, model {ns.model}, record {ns.record}, cap ${ns.usd_cap:.2f}, "
          f"concurrency {ns.concurrency}, prompt {CURRENT_VERSION[STEP]}:{prompt_fingerprint(STEP)}, "
          f"sessions {sessions}", flush=True)
    results = asyncio.run(_run(ns, cases, date))
    summary = summarize(results)

    ns.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"trapeval-{date}-{ns.model}-{ns.split}-{stamp}"
    jsonl_path, summary_path = ns.out_dir / f"{stem}.jsonl", ns.out_dir / f"{stem}.summary.json"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"set": str(ns.set.relative_to(ROOT)) if ns.set.is_relative_to(ROOT) else str(ns.set),
            "model": ns.model, "split": ns.split, "record": ns.record, "usd_cap": ns.usd_cap,
            "concurrency": ns.concurrency, "retries": ns.retries, "date": date, "stamp": stamp,
            "sessions": sessions, "prompt": {"version": CURRENT_VERSION[STEP], "fingerprint": prompt_fingerprint(STEP)},
            "summary": summary}
    summary_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(format_report(summary, results))
    print(f"[trapeval] wrote {jsonl_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
