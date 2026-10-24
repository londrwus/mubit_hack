"""D39 gate for a closed-output judge_improvement v1 (D37). Record OFF, nothing is recorded.

Measures, on the same 40+ judge inputs:
- Opus 5.5 self-agreement (two passes) on the full closed output (verdict + reason_code) and on
  the verdict alone;
- Haiku 5.5 zero-shot agreement with Opus (lower bound for an adapter);
- class balance of verdict and reason_code; which codes Opus flips between;
- accuracy vs ground truth (trap set ``acceptable_verdicts``; noise set truth).

Inputs: ``data/trap_set/trap_set_v1.jsonl`` (all 50) + a deterministic, balanced pick from
``data/noise_set/noise_set_v1.jsonl`` (race-format evidence: run metrics only).

Every call goes through ``backend.agent.mubit_client.MubitClient`` (the Mubit gateway,
``MUBIT_API_KEY``), ``record=False`` -> ``x-middleware-record: off``, step judge_improvement
(effort low). The draft prompt lives in ``data/gates/judge_v1/`` and is NOT a frozen prompt.

    python -m uv run --extra ml python scripts/gate_judge_v1.py run --variant v1 --model claude-opus-5-5 --pass 1
    python -m uv run --extra ml python scripts/gate_judge_v1.py run --variant v1 --model claude-opus-5-5 --pass 2
    python -m uv run --extra ml python scripts/gate_judge_v1.py run --variant v1 --model claude-haiku-5-5 --pass 1
    python -m uv run --extra ml python scripts/gate_judge_v1.py report

Runs are resumable (a case with a valid reply in its calls file is skipped). The total spend
over every calls file is capped at ``LLM_CAP_USD``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pydantic import BaseModel, create_model  # noqa: E402

from backend.core.types import ArmKind, StepType  # noqa: E402

GATE_DIR = REPO / "data" / "gates" / "judge_v1"
CALLS_DIR = GATE_DIR / "calls"
TRAP_SET = REPO / "data" / "trap_set" / "trap_set_v1.jsonl"
NOISE_SET = REPO / "data" / "noise_set" / "noise_set_v1.jsonl"
DATE = "2026-10-24"
LLM_CAP_USD = 1.50
OPUS = "claude-opus-5-5"
HAIKU = "claude-haiku-5-5"
STEP = StepType.judge_improvement
SELF_AGREEMENT_TARGET = 0.97
MAX_CLASS_SHARE = 0.70

# variant -> (prompt file, {reason_code: verdict})
VARIANTS: dict[str, tuple[str, dict[str, str]]] = {
    "v1": (
        "judge_improvement.v1.draft.md",
        {
            "contamination_or_eval_leak": "invalid",
            "eval_setup_changed": "invalid",
            "gain_within_noise": "unsure",
            "insufficient_evidence": "unsure",
            "clean_gain": "valid",
        },
    ),
    # v1 + one sentence in case 4 (when leakage counts as ruled out); tuned on 2 trap DEV cases
    "v1b": (
        "judge_improvement.v1b.draft.md",
        {
            "contamination_or_eval_leak": "invalid",
            "eval_setup_changed": "invalid",
            "gain_within_noise": "unsure",
            "insufficient_evidence": "unsure",
            "clean_gain": "valid",
        },
    ),
    # fallback if v1 fails self-agreement (not needed: v1 passed); prompt file not written
    "v1m": (
        "judge_improvement.v1m.draft.md",
        {
            "eval_not_trustworthy": "invalid",
            "not_proven": "unsure",
            "clean_gain": "valid",
        },
    ),
}
VERDICTS = ["valid", "invalid", "unsure"]

_HEADER_RE = re.compile(r"\A(?:\s*<!--.*?-->\s*)+", re.DOTALL)


def load_prompt(variant: str) -> str:
    """Same normalisation as backend.agent.prompts._load (header stripped, LF, one trailing NL)."""
    raw = (GATE_DIR / VARIANTS[variant][0]).read_text(encoding="utf-8")
    return _HEADER_RE.sub("", raw, count=1).strip().replace("\r\n", "\n") + "\n"


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def wire_schema(variant: str) -> dict[str, Any]:
    codes = list(VARIANTS[variant][1])
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": VERDICTS},
            "reason_code": {"type": "string", "enum": codes},
        },
        "required": ["verdict", "reason_code"],
        "additionalProperties": False,
    }


def output_model(variant: str) -> type[BaseModel]:
    codes = tuple(VARIANTS[variant][1])
    return create_model(  # type: ignore[call-overload]
        f"JudgeV1Output_{variant}",
        verdict=(Literal[tuple(VERDICTS)], ...),  # type: ignore[valid-type]
        reason_code=(Literal[codes], ...),  # type: ignore[valid-type]
    )


# --------------------------------------------------------------------------- cases

NOISE_PICK = {"real": 7, "noise": 7, "fake": 4}  # all real cases; noise and fake split rl/cifar


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in TRAP_SET.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        cases.append({
            "id": r["id"], "source": "trap_set_v1", "target": r["target"], "split": r["split"],
            "truth": r["label"], "category": r["category"],
            "acceptable": list(r["acceptable_verdicts"]), "user_message": r["user_message"],
        })
    noise = [json.loads(l) for l in NOISE_SET.read_text(encoding="utf-8").splitlines()]
    noise.sort(key=lambda r: r["id"])
    acceptable = {"fake": ["invalid"], "noise": ["unsure", "invalid"], "real": ["valid"]}
    for truth, n in NOISE_PICK.items():
        pool = [r for r in noise if r["truth"] == truth]
        # alternate targets so both rl and cifar appear
        by_t: dict[str, list[dict[str, Any]]] = {}
        for r in pool:
            by_t.setdefault(r["target"], []).append(r)
        picked: list[dict[str, Any]] = []
        i = 0
        while len(picked) < min(n, len(pool)):
            for t in sorted(by_t):
                if i < len(by_t[t]) and len(picked) < n:
                    picked.append(by_t[t][i])
            i += 1
        for r in picked:
            cases.append({
                "id": r["id"], "source": "noise_set_v1", "target": r["target"], "split": r["split"],
                "truth": truth, "category": f"race_{truth}", "acceptable": acceptable[truth],
                "user_message": r["user_message"],
            })
    return cases


# --------------------------------------------------------------------------- run

def calls_path(variant: str, model: str, pass_n: int) -> Path:
    return CALLS_DIR / f"{variant}.{model}.pass{pass_n}.jsonl"


def read_calls(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def total_spend() -> float:
    return sum(r.get("cost_usd") or 0.0 for p in CALLS_DIR.glob("*.jsonl") for r in read_calls(p))


async def run(variant: str, model: str, pass_n: int, concurrency: int, limit: int | None) -> None:
    from backend.agent.mubit_client import MubitClient  # the ONLY LLM client (Mubit gateway)

    CALLS_DIR.mkdir(parents=True, exist_ok=True)
    system = load_prompt(variant)
    fp = fingerprint(system)
    schema = wire_schema(variant)
    out_model = output_model(variant)
    path = calls_path(variant, model, pass_n)
    done = {r["case_id"] for r in read_calls(path) if r.get("valid")}
    cases = [c for c in load_cases() if c["id"] not in done]
    if limit:
        cases = cases[:limit]
    spent = total_spend()
    print(f"variant {variant} fp {fp} model {model} pass {pass_n}: {len(cases)} to run, "
          f"spent so far ${spent:.4f} (cap ${LLM_CAP_USD})", flush=True)
    session = f"gate-judgev1-{DATE}/{model}/{variant}/pass-{pass_n}"
    client = MubitClient(arm_models={ArmKind.opus: model})
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    state = {"spent": spent, "stopped": False}

    async def one(case: dict[str, Any]) -> None:
        async with sem:
            if state["spent"] >= LLM_CAP_USD:
                state["stopped"] = True
                return
            for attempt in range(2):  # one retry on a failed reply (counts toward spend)
                parsed, rec = await client.call(
                    STEP, system, case["user_message"], out_model, ArmKind.opus, session, False,
                    wire_schema=schema, model=model, is_retry=attempt > 0,
                )
                row = {
                    "case_id": case["id"], "variant": variant, "prompt_fingerprint": fp,
                    "model": model, "pass": pass_n, "attempt": attempt, "valid": parsed is not None,
                    "verdict": getattr(parsed, "verdict", None),
                    "reason_code": getattr(parsed, "reason_code", None),
                    "error": rec.error, "mubit_request_id": rec.mubit_request_id,
                    "recorded": rec.recorded, "session_id": session, "latency_ms": rec.latency_ms,
                    "input_tokens": rec.input_tokens, "cache_creation_tokens": rec.cache_creation_tokens,
                    "cache_read_tokens": rec.cache_read_tokens, "output_tokens": rec.output_tokens,
                    "cost_usd": rec.cost_usd, "stop_reason": rec.stop_reason,
                }
                async with lock:
                    state["spent"] += rec.cost_usd or 0.0
                    with path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                if parsed is not None:
                    return

    try:
        await asyncio.gather(*(one(c) for c in cases))
    finally:
        await client.aclose()
    print(f"done; spent total ${state['spent']:.4f}{' (CAP HIT, stopped)' if state['stopped'] else ''}")


# --------------------------------------------------------------------------- report

def final_outputs(variant: str, model: str, pass_n: int) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for r in read_calls(calls_path(variant, model, pass_n)):
        if r.get("valid"):
            out[r["case_id"]] = (r["verdict"], r["reason_code"])
    return out


def agreement(a: dict[str, tuple[str, str]], b: dict[str, tuple[str, str]]) -> dict[str, Any]:
    ids = sorted(set(a) & set(b))
    full = sum(a[i] == b[i] for i in ids)
    verd = sum(a[i][0] == b[i][0] for i in ids)
    flips = Counter(
        " <-> ".join(sorted([f"{a[i][0]}/{a[i][1]}", f"{b[i][0]}/{b[i][1]}"])) for i in ids if a[i] != b[i]
    )
    return {
        "n": len(ids),
        "full_agree": full, "full_rate": round(full / len(ids), 4) if ids else None,
        "verdict_agree": verd, "verdict_rate": round(verd / len(ids), 4) if ids else None,
        "disagreements": {i: {"a": list(a[i]), "b": list(b[i])} for i in ids if a[i] != b[i]},
        "flip_pairs": dict(flips.most_common()),
    }


def balance(outs: dict[str, tuple[str, str]]) -> dict[str, Any]:
    n = len(outs)
    v = Counter(o[0] for o in outs.values())
    c = Counter(o[1] for o in outs.values())
    return {
        "n": n,
        "verdict": dict(v.most_common()), "reason_code": dict(c.most_common()),
        "max_verdict_share": round(max(v.values()) / n, 4) if n else None,
        "max_reason_code_share": round(max(c.values()) / n, 4) if n else None,
    }


def accuracy(outs: dict[str, tuple[str, str]], cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    res: dict[str, Any] = {}
    groups: dict[str, list[bool]] = {}
    by_cat: dict[str, Counter[str]] = {}
    for cid, (verdict, code) in outs.items():
        c = cases[cid]
        ok = verdict in c["acceptable"]
        for g in ("all", c["source"], f"{c['source']}/{c['truth']}"):
            groups.setdefault(g, []).append(ok)
        by_cat.setdefault(f"{c['source']}/{c['category']}", Counter())[f"{verdict}/{code}"] += 1
    for g, oks in sorted(groups.items()):
        res[g] = {"n": len(oks), "correct": sum(oks), "rate": round(sum(oks) / len(oks), 4)}
    res["outputs_by_category"] = {k: dict(v.most_common()) for k, v in sorted(by_cat.items())}
    return res


def code_verdict_mismatch(variant: str, outs: dict[str, tuple[str, str]]) -> int:
    m = VARIANTS[variant][1]
    return sum(m[c] != v for v, c in outs.values())


def run_stats(variant: str, model: str, pass_n: int) -> dict[str, Any]:
    rows = read_calls(calls_path(variant, model, pass_n))
    valid = [r for r in rows if r.get("valid")]
    lat = sorted(r["latency_ms"] for r in valid if r.get("latency_ms"))
    return {
        "calls": len(rows), "valid": len(valid), "failed": len(rows) - len(valid),
        "recorded_calls": sum(bool(r.get("recorded")) for r in rows),
        "cost_usd": round(sum(r.get("cost_usd") or 0 for r in rows), 4),
        "p50_latency_ms": round(lat[len(lat) // 2], 1) if lat else None,
        "mean_output_tokens": round(sum(r["output_tokens"] for r in valid) / len(valid), 1) if valid else None,
        "first_request_ids": [r["mubit_request_id"] for r in rows[:3]],
        "session_id": rows[0]["session_id"] if rows else None,
        "prompt_fingerprint": rows[0]["prompt_fingerprint"] if rows else None,
    }


def report() -> dict[str, Any]:
    cases = {c["id"]: c for c in load_cases()}
    out: dict[str, Any] = {
        "gate": "D39", "date": DATE, "step": STEP.value, "record": "off",
        "cases": {
            "n": len(cases),
            "by_source_truth": dict(Counter(f"{c['source']}/{c['truth']}" for c in cases.values())),
            "by_target": dict(Counter(c["target"] for c in cases.values())),
            "ids": sorted(cases),
        },
        "criteria": {"opus_self_agreement_full_min": SELF_AGREEMENT_TARGET, "max_class_share": MAX_CLASS_SHARE},
        "variants": {},
    }
    for variant in VARIANTS:
        o1, o2 = final_outputs(variant, OPUS, 1), final_outputs(variant, OPUS, 2)
        h1 = final_outputs(variant, HAIKU, 1)
        if not o1:
            continue
        prompt = load_prompt(variant)
        v: dict[str, Any] = {
            "prompt_file": str((GATE_DIR / VARIANTS[variant][0]).relative_to(REPO)),
            "prompt_fingerprint": fingerprint(prompt),
            "prompt_chars": len(prompt),
            "codes": VARIANTS[variant][1],
            "runs": {f"{m}/pass{p}": run_stats(variant, m, p)
                     for m, p in ((OPUS, 1), (OPUS, 2), (HAIKU, 1)) if calls_path(variant, m, p).exists()},
            "opus_self_agreement": agreement(o1, o2) if o2 else None,
            "haiku_vs_opus_pass1": agreement(h1, o1) if h1 else None,
            "haiku_vs_opus_pass2": agreement(h1, o2) if h1 and o2 else None,
            "balance_opus_pass1": balance(o1),
            "balance_opus_pass2": balance(o2) if o2 else None,
            "balance_haiku": balance(h1) if h1 else None,
            "accuracy_opus_pass1": accuracy(o1, cases),
            "accuracy_opus_pass2": accuracy(o2, cases) if o2 else None,
            "accuracy_haiku": accuracy(h1, cases) if h1 else None,
            "code_verdict_mismatch": {"opus_pass1": code_verdict_mismatch(variant, o1),
                                      "opus_pass2": code_verdict_mismatch(variant, o2),
                                      "haiku": code_verdict_mismatch(variant, h1)},
        }
        sa = v["opus_self_agreement"]
        bal = [b for b in (v["balance_opus_pass1"], v["balance_opus_pass2"]) if b]
        max_share = max(max(b["max_verdict_share"], b["max_reason_code_share"]) for b in bal)
        v["gate"] = {
            "self_agreement_full": sa["full_rate"] if sa else None,
            "self_agreement_pass": bool(sa and sa["n"] == len(cases) and sa["full_rate"] >= SELF_AGREEMENT_TARGET),
            "max_class_share": max_share,
            "balance_pass": max_share <= MAX_CLASS_SHARE,
        }
        v["gate"]["pass_on_this_set"] = v["gate"]["self_agreement_pass"] and v["gate"]["balance_pass"]
        out["variants"][variant] = v
    out["total_cost_usd"] = round(total_spend(), 4)
    (GATE_DIR / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--variant", choices=list(VARIANTS), default="v1")
    r.add_argument("--model", default=OPUS)
    r.add_argument("--pass", dest="pass_n", type=int, default=1)
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--limit", type=int, default=None)
    sub.add_parser("cases")
    sub.add_parser("report")
    ns = ap.parse_args()
    if ns.cmd == "run":
        asyncio.run(run(ns.variant, ns.model, ns.pass_n, ns.concurrency, ns.limit))
    elif ns.cmd == "cases":
        cs = load_cases()
        print(len(cs), Counter(f"{c['source']}/{c['truth']}" for c in cs))
        for v in VARIANTS:
            if (GATE_DIR / VARIANTS[v][0]).exists():
                p = load_prompt(v)
                print(v, fingerprint(p), len(p), "chars")
    else:
        res = report()
        for v, d in res["variants"].items():
            print(v, json.dumps(d["gate"]), "self", d["opus_self_agreement"] and
                  {k: d["opus_self_agreement"][k] for k in ("n", "full_rate", "verdict_rate", "flip_pairs")})
            if d["haiku_vs_opus_pass1"]:
                print("  haiku vs opus1", {k: d["haiku_vs_opus_pass1"][k] for k in ("n", "full_rate", "verdict_rate")})
            print("  balance", d["balance_opus_pass1"])
            print("  acc opus1", {k: x for k, x in d["accuracy_opus_pass1"].items() if k != "outputs_by_category"})
        print("total $", res["total_cost_usd"])


if __name__ == "__main__":
    main()
