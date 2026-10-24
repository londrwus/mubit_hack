"""Grade the judge calls that Mubit's own model answered, next to the ones Opus answered.

The API response does not say who answered a call; the console's Requests page does
('Served by'). So: push traffic with ``scripts/push_judge_traffic.py``, look the request ids up in
the console, paste the ones marked 'Served by: Mubit model' into a text file (any text works: every
``mreq_...`` / ``mock_...`` token in it is read), and run

    python -m uv run --extra ml python scripts/eval_served_calls.py \\
        data/adapter_golive/C/traffic-<stamp>.jsonl --served-ids served_by_mubit.txt

Report (no LLM calls, $0):
- per group (Mubit model / Opus), split into **held-out** cases (trap/noise test splits, never
  recorded anywhere) and **training pool** cases (the 177 recorded inputs: the adapter may have
  trained on them, so its accuracy there is optimistic):
  accuracy vs ground truth; trap set: fake caught / fake let through / real confirmed; noise set:
  real kept (not rejected) / real confirmed / noise accepted / fake caught; race and collection
  claims: fake caught and not-fake wrongly rejected;
- agreement with Opus's reference answer on the same case (verdict, and verdict + reason_code).
  This is what Mubit grades an adapter on (it is graded against Opus's answers, not ground truth);
- the same cases answered by Opus in this traffic (paired comparison, if any);
- latency p50/p90 per group (valid answers only).

Failed and invalid calls count as WRONG (as Mubit grades them): every call with a Mubit request id
stays in the accuracy and agreement denominators, valid or not. ``invalid`` = a reply came back
but it was unusable (no reason_code, malformed JSON, a bad stop_reason); ``failed`` = no reply
(HTTP error / transport, no tokens). Both are counted per group. A ``valid_only`` accuracy and
agreement (answered calls only) is also reported, labelled as such: never quote it alone.

Without ``--served-ids`` it reports the overall numbers and the latency distribution. A clearly
bimodal latency can hint that some calls were answered by a small model: that is printed as a
HINT ONLY, never as an attribution.

A JSON copy of the report goes next to the (first) traffic file as ``<name>.served-eval.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ID_TOKEN = re.compile(r"\b(?:mreq_[0-9a-zA-Z]+|mock_(?:adp|opus|err)_[0-9a-f]+)\b")
GROUPS = ("mubit_model", "opus")


# --------------------------------------------------------------------------- loading


def load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        for ln in Path(p).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


def read_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return set(ID_TOKEN.findall(Path(path).read_text(encoding="utf-8")))


def served_calls(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every attempted call that has a Mubit request id, one per id, valid or not. A failed or
    invalid reply is a call the answerer got wrong, so it must stay in the denominators."""
    seen: set[str] = set()
    out = []
    for r in rows:
        rid = r.get("mubit_request_id")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        out.append(r)
    return out


def outcome(r: dict[str, Any]) -> str:
    """'valid' (a usable answer), 'invalid' (a reply came back, unusable) or 'failed' (no reply)."""
    if r.get("valid"):
        return "valid"
    tokens = int(r.get("input_tokens") or 0) + int(r.get("output_tokens") or 0)
    return "invalid" if tokens > 0 else "failed"


# --------------------------------------------------------------------------- grading


def grade(r: dict[str, Any]) -> dict[str, Any]:
    """Ground-truth flags for one call (empty when the case has no ground truth). A call with no
    valid answer is wrong: correct=False and no positive flag (it caught nothing, kept nothing)."""
    gt = r.get("ground_truth")
    if not gt:
        return {}
    ok = bool(r.get("valid"))
    v = r.get("verdict") if ok else None
    t = gt["truth"]
    return {
        "set": gt["set"], "truth": t, "valid": ok, "correct": ok and v in gt["acceptable"],
        "fake_caught": t == "fake" and v == "invalid",
        "fake_let_through": t == "fake" and v == "valid",
        "real_confirmed": t == "real" and v == "valid",
        "real_kept": t == "real" and v != "invalid",
        "real_rejected": t == "real" and v == "invalid",
        "noise_accepted": t == "noise" and v == "valid",
        "not_fake_rejected": t == "not_fake" and v == "invalid",
    }


def agreement(r: dict[str, Any]) -> dict[str, bool] | None:
    """Agreement with Opus's reference answer; a call with no valid answer agrees on nothing."""
    ref = r.get("opus_ref")
    if not ref:
        return None
    if not r.get("valid"):
        return {"verdict": False, "full": False, "valid": False}
    return {"verdict": r.get("verdict") == ref.get("verdict"),
            "full": (r.get("verdict"), r.get("reason_code")) == (ref.get("verdict"), ref.get("reason_code")),
            "valid": True}


def _q(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return round(xs[0], 1)
    return round(statistics.quantiles(xs, n=100, method="inclusive")[int(q * 100) - 1], 1)


def _rate(a: int, b: int) -> float | None:
    return None if b == 0 else round(a / b, 4)


def block(calls: Sequence[dict[str, Any]]) -> dict[str, Any]:
    g = [(c, grade(c)) for c in calls]
    graded = [x for _, x in g if x]
    by_set: dict[str, dict[str, Any]] = {}
    for s in ("trap", "noise", "race", "collection"):
        xs = [x for x in graded if x["set"] == s]
        if not xs:
            continue
        n_t = Counter(x["truth"] for x in xs)
        b: dict[str, Any] = {"n": len(xs), "correct": sum(x["correct"] for x in xs),
                             "accuracy": _rate(sum(x["correct"] for x in xs), len(xs)), "by_truth": dict(n_t),
                             "no_valid_answer": sum(not x["valid"] for x in xs)}
        if n_t["fake"]:
            b |= {"fake_caught": f"{sum(x['fake_caught'] for x in xs)}/{n_t['fake']}",
                  "fake_let_through": f"{sum(x['fake_let_through'] for x in xs)}/{n_t['fake']}"}
        if n_t["real"]:
            b |= {"real_confirmed": f"{sum(x['real_confirmed'] for x in xs)}/{n_t['real']}",
                  "real_kept": f"{sum(x['real_kept'] for x in xs)}/{n_t['real']}",
                  "real_rejected": f"{sum(x['real_rejected'] for x in xs)}/{n_t['real']}"}
        if n_t["noise"]:
            b["noise_accepted"] = f"{sum(x['noise_accepted'] for x in xs)}/{n_t['noise']}"
        if n_t["not_fake"]:
            b["not_fake_rejected"] = f"{sum(x['not_fake_rejected'] for x in xs)}/{n_t['not_fake']}"
        by_set[s] = b
    ag = [a for a in (agreement(c) for c in calls) if a is not None]
    oc = Counter(outcome(c) for c in calls)
    graded_ok = [x for x in graded if x["valid"]]
    ag_ok = [a for a in ag if a["valid"]]
    lat = [float(c["latency_ms"]) for c in calls if c.get("valid") and c.get("latency_ms")]
    return {
        "calls": len(calls), "cases": len({c["case_id"] for c in calls}),
        "valid": oc["valid"], "invalid": oc["invalid"], "failed": oc["failed"],
        # every call counts; an invalid or failed call is wrong
        "graded": len(graded), "correct": sum(x["correct"] for x in graded),
        "accuracy": _rate(sum(x["correct"] for x in graded), len(graded)),
        "by_set": by_set,
        "agree_with_opus_ref": {"n": len(ag), "verdict": _rate(sum(a["verdict"] for a in ag), len(ag)),
                                "verdict_and_code": _rate(sum(a["full"] for a in ag), len(ag)),
                                "verdict_and_code_count": f"{sum(a['full'] for a in ag)}/{len(ag)}"},
        # labelled secondary figure: valid answers only (drops the invalid/failed calls; optimistic)
        "valid_only": {"graded": len(graded_ok), "correct": sum(x["correct"] for x in graded_ok),
                       "accuracy": _rate(sum(x["correct"] for x in graded_ok), len(graded_ok)),
                       "agree_n": len(ag_ok),
                       "agree_verdict_and_code": _rate(sum(a["full"] for a in ag_ok), len(ag_ok)),
                       "label": "VALID ANSWERS ONLY (invalid/failed calls dropped): optimistic, never quote alone"},
        "verdicts": dict(Counter(c.get("verdict") if c.get("valid") else f"<{outcome(c)}>" for c in calls)),
        "latency_p50_ms": _q(lat, 0.5), "latency_p90_ms": _q(lat, 0.9),
    }


def split_pool(calls: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {"held_out": [c for c in calls if not c.get("in_training_pool")],
            "training_pool": [c for c in calls if c.get("in_training_pool")],
            "all": list(calls)}


# --------------------------------------------------------------------------- latency hint


def latency_histogram(lat: Sequence[float]) -> list[tuple[str, int]]:
    edges = [0, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 5000, 10000, math.inf]
    out = []
    for lo, hi in zip(edges, edges[1:]):
        n = sum(lo <= x < hi for x in lat)
        out.append((f"{lo:>5.0f}-{'' if hi == math.inf else f'{hi:.0f}'} ms", n))
    return out


def bimodal_hint(lat: Sequence[float]) -> dict[str, Any] | None:
    """Two-means on log latency. Reported only when the clusters are far apart (fast mean < 0.5x
    the slow mean), well separated (gap) and the small cluster holds >= 5% of calls (>= 3 calls).
    A hint, never an attribution: cache hits, retries and load also change latency."""
    xs = sorted(math.log(x) for x in lat if x > 0)
    if len(xs) < 10:
        return None
    best = None
    for k in range(1, len(xs)):
        a, b = xs[:k], xs[k:]
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        sse = sum((x - ma) ** 2 for x in a) + sum((x - mb) ** 2 for x in b)
        if best is None or sse < best[0]:
            best = (sse, k, ma, mb)
    _, k, ma, mb = best  # type: ignore[misc]
    fast, slow = xs[:k], xs[k:]
    ratio = math.exp(ma) / math.exp(mb)
    share = len(fast) / len(xs)
    # Ashman's D > 2: the two groups are cleanly separated, not one skewed bump
    ashman_d = math.sqrt(2) * (mb - ma) / math.sqrt(statistics.pvariance(fast) + statistics.pvariance(slow) + 1e-9)
    if ratio < 0.5 and min(len(fast), len(slow)) >= max(3, 0.05 * len(xs)) and ashman_d > 2:
        return {"fast_calls": len(fast), "share": round(share, 4), "ashman_d": round(ashman_d, 2),
                "fast_p50_ms": round(math.exp(statistics.median(fast)), 1),
                "slow_p50_ms": round(math.exp(statistics.median(slow)), 1),
                "split_below_ms": round(math.exp((fast[-1] + slow[0]) / 2), 1),
                "label": "HINT ONLY: the fast group may be Mubit-model answers; confirm with 'Served by' in the console"}
    return None


# --------------------------------------------------------------------------- report


def report(rows: Sequence[dict[str, Any]], served_ids: set[str] | None) -> dict[str, Any]:
    calls = served_calls(rows)
    oc = Counter(outcome(c) for c in calls)
    meta = {"rows": len(rows), "calls": len(calls), "valid_calls": oc["valid"], "invalid_calls": oc["invalid"],
            "failed_calls": oc["failed"], "rows_without_request_id": sum(not r.get("mubit_request_id") for r in rows),
            "workspaces": sorted({str(r.get("workspace")) for r in rows}),
            "record_modes": sorted({str(r.get("record")) for r in rows}),
            "dry_run": any(r.get("dry_run") for r in rows),
            "prompt_fingerprints": sorted({str(r.get("prompt_fingerprint")) for r in rows})}
    if served_ids:
        known = {r["mubit_request_id"] for r in rows if r.get("mubit_request_id")}
        groups = {"mubit_model": [c for c in calls if c["mubit_request_id"] in served_ids],
                  "opus": [c for c in calls if c["mubit_request_id"] not in served_ids]}
        out: dict[str, Any] = {"mode": "served_by", **meta, "served_ids_given": len(served_ids),
                               "served_ids_not_in_traffic": sorted(served_ids - known)[:20],
                               "n_served_ids_not_in_traffic": len(served_ids - known),
                               "groups": {g: {k: block(v) for k, v in split_pool(cs).items()} for g, cs in groups.items()}}
        # paired: the cases the Mubit model answered, as answered by Opus (in this traffic)
        adp_cases = {c["case_id"] for c in groups["mubit_model"]}
        paired_opus = [c for c in groups["opus"] if c["case_id"] in adp_cases]
        paired_adp = [c for c in groups["mubit_model"] if c["case_id"] in {p["case_id"] for p in paired_opus}]
        out["paired_same_cases"] = {"cases": len({c["case_id"] for c in paired_adp}),
                                    "mubit_model": block(paired_adp), "opus": block(paired_opus)}
        out["disagreements_with_opus_ref"] = [
            {"case_id": c["case_id"], "source": c["source"], "answer": [c.get("verdict"), c.get("reason_code")],
             "outcome": outcome(c), "error": None if c.get("valid") else c.get("error"),
             "opus_ref": [c["opus_ref"]["verdict"], c["opus_ref"]["reason_code"]],
             "truth": (c.get("ground_truth") or {}).get("truth"), "mubit_request_id": c["mubit_request_id"]}
            for c in groups["mubit_model"] if c.get("opus_ref") and not agreement(c)["full"]][:30]  # type: ignore[index]
        return out
    lat = [float(c["latency_ms"]) for c in calls if c.get("valid") and c.get("latency_ms")]
    return {"mode": "overall", **meta, "overall": {k: block(v) for k, v in split_pool(calls).items()},
            "latency_histogram": latency_histogram(lat), "latency_hint": bimodal_hint(lat)}


def _fmt_block(name: str, b: dict[str, Any]) -> list[str]:
    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{100 * x:.1f}%"

    ag = b["agree_with_opus_ref"]
    vo = b["valid_only"]
    lines = [f"  {name:<14} calls {b['calls']:>4} (cases {b['cases']:>3}; invalid {b['invalid']}, failed "
             f"{b['failed']})  accuracy {b['correct']}/{b['graded']} = {pct(b['accuracy'])}  | agrees with Opus: "
             f"verdict {pct(ag['verdict'])}, verdict+code {ag['verdict_and_code_count']} = "
             f"{pct(ag['verdict_and_code'])}  | latency p50 {b['latency_p50_ms']} ms, p90 {b['latency_p90_ms']} ms"]
    if b["invalid"] or b["failed"]:
        lines.append(f"      [valid answers only, optimistic] accuracy {vo['correct']}/{vo['graded']} = "
                     f"{pct(vo['accuracy'])}, agrees verdict+code {pct(vo['agree_verdict_and_code'])} (n {vo['agree_n']})")
    for s, x in b["by_set"].items():
        extra = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in x.items()
                          if k not in ("n", "correct", "accuracy", "by_truth", "no_valid_answer"))
        if x["no_valid_answer"]:
            extra += f"; no valid answer (counted wrong) {x['no_valid_answer']}"
        lines.append(f"      {s:<10} {x['correct']}/{x['n']} correct; {extra}")
    return lines


def format_report(rep: dict[str, Any]) -> str:
    lines = [f"traffic: {rep['rows']} rows, {rep['calls']} calls ({rep['valid_calls']} valid, {rep['invalid_calls']} "
             f"invalid, {rep['failed_calls']} failed; invalid/failed count as wrong), workspaces {rep['workspaces']}, "
             f"record {rep['record_modes']}, prompt {rep['prompt_fingerprints']}"
             + ("  [DRY RUN data: mock answers, not a measurement]" if rep["dry_run"] else "")]
    if rep["mode"] == "served_by":
        lines.append(f"served-by ids given: {rep['served_ids_given']}; not found in this traffic: "
                     f"{rep['n_served_ids_not_in_traffic']}")
        for g in GROUPS:
            lines.append("== answered by Mubit model (in --served-ids)" if g == "mubit_model" else
                         "== not in --served-ids, counted as Opus (wrong if the pasted console list was incomplete)")
            for k in ("held_out", "training_pool", "all"):
                if rep["groups"][g][k]["calls"]:
                    lines += _fmt_block(k, rep["groups"][g][k])
        p = rep["paired_same_cases"]
        lines.append(f"== same cases answered by both ({p['cases']} cases)")
        lines += _fmt_block("Mubit model", p["mubit_model"]) + _fmt_block("Opus", p["opus"])
        if rep["disagreements_with_opus_ref"]:
            lines.append("== Mubit-model answers that differ from Opus's reference answer")
            for d in rep["disagreements_with_opus_ref"]:
                lines.append(f"  {d['case_id']:<20} {d['source']:<10} {d['answer']} vs Opus {d['opus_ref']} "
                             f"(truth {d['truth']}) {d['mubit_request_id']}")
        lines.append("note: 'training_pool' cases were recorded into the workspace, so the adapter may have trained "
                     "on them; quote 'held_out' numbers, with counts. Invalid/failed calls count as wrong; the "
                     "'valid answers only' line is optimistic and never quoted alone. Failed calls (no reply, e.g. a "
                     "gateway 503) not in --served-ids land in the Opus group.")
    else:
        lines.append("== all calls (who answered is unknown without --served-ids)")
        for k in ("held_out", "training_pool", "all"):
            if rep["overall"][k]["calls"]:
                lines += _fmt_block(k, rep["overall"][k])
        lines.append("latency distribution:")
        mx = max((n for _, n in rep["latency_histogram"]), default=0) or 1
        for label, n in rep["latency_histogram"]:
            lines.append(f"  {label:<16} {n:>4} {'#' * round(40 * n / mx)}")
        h = rep["latency_hint"]
        lines.append(f"latency hint: {h['label']}: {h['fast_calls']} calls ({100 * h['share']:.1f}%) below ~"
                     f"{h['split_below_ms']:.0f} ms (p50 {h['fast_p50_ms']} vs {h['slow_p50_ms']} ms)" if h
                     else "latency hint: none (latency looks unimodal; no sign of a second, faster answerer)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traffic", nargs="+", type=Path, help="traffic-*.jsonl file(s) from push_judge_traffic.py")
    ap.add_argument("--served-ids", type=Path, default=None,
                    help="text file with the request ids the console shows as 'Served by: Mubit model'")
    ap.add_argument("--out", type=Path, default=None, help="JSON report path (default: next to the first traffic file)")
    ns = ap.parse_args(argv)
    rows = load_rows(ns.traffic)
    if not rows:
        print("ERROR: no rows in the traffic file(s)", file=sys.stderr)
        return 2
    ids = read_ids(ns.served_ids)
    if ns.served_ids is not None and not ids:
        print(f"ERROR: no request ids found in {ns.served_ids}", file=sys.stderr)
        return 2
    rep = report(rows, ids or None)
    out = ns.out or ns.traffic[0].with_suffix(".served-eval.json")
    out.write_text(json.dumps(rep, indent=1, default=list), encoding="utf-8")
    print(format_report(rep))
    print(f"[eval] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
