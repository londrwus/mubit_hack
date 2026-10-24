"""Record judge_improvement v1 (closed output, D37) examples into Mubit project B (D38), gated by D39.

Inputs (``build``), deduplicated on the exact user message:
- ``trap_set``    data/trap_set/trap_set_v1.jsonl, DEV split only (the 20 test cases are the held-out
                  split of the benchmark's fake-improvement metric and are never recorded);
- ``noise_set``   data/noise_set/noise_set_v1.jsonl, DEV split only (same reason: its test split is
                  the held-out split of the real-vs-noise metric);
- ``collection``  every judged claim in data/collection/collect_{rl,cifar,lm}.db that is not already
                  a noise-set case, rebuilt exactly like ``scripts/make_noise_set.py`` (collection-time
                  spec, agent-visible metrics as evidence);
- ``race``        every LLM-judged claim in data/bench/bench.db (read-only), rebuilt like the race
                  loop: ``build_judge_user(race_spec(cfg), JudgeClaim(...))`` with the previous best
                  checked against ``summary.best_before``.

Phases (``run --phase``):
1. ``first``   a stratified batch of 40, Opus 5.5, ``record: full`` on project B
               (``MUBIT_API_KEY`` is set to ``MUBIT_API_KEY_B`` from .env for this process only);
2. ``recheck`` the same 40 with record OFF on project A (``MUBIT_API_KEY`` from .env); D39
               self-agreement on verdict + reason_code must be >= 97%;
3. ``rest``    the remaining inputs, ``record: full`` on project B; refused unless the recheck passed.

Every call goes through ``backend.agent.mubit_client.MubitClient`` (the Mubit gateway). The frozen
prompt (``judge_improvement`` v1, fingerprint ``5c1d3cfd2941``) and its wire schema come from the
prompt registry. Total spend over every calls file is capped at ``LLM_CAP_USD``.

    python -m uv run --extra ml python scripts/record_judge_v1.py build
    python -m uv run --extra ml python scripts/record_judge_v1.py run --phase first --yes
    python -m uv run --extra ml python scripts/record_judge_v1.py run --phase recheck
    python -m uv run --extra ml python scripts/record_judge_v1.py run --phase rest --yes
    python -m uv run --extra ml python scripts/record_judge_v1.py report
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.agent.prompts import CURRENT_VERSION, FROZEN, load_system_prompt, prompt_fingerprint  # noqa: E402
from backend.agent.prompts.build import JudgeClaim, build_judge_user, judge_wire_schema  # noqa: E402
from backend.core.types import JUDGE_REASON_VERDICT, ArmKind, JudgeOutput, JudgeReasonCode, StepType  # noqa: E402

STEP = StepType.judge_improvement
VERSION = "v1"
EXPECTED_FINGERPRINT = "5c1d3cfd2941"
DATE = "2026-10-24"
OPUS = "claude-opus-5-5"
LLM_CAP_USD = 1.50
FIRST_BATCH = 40
FIRST_SEED = 20261024
SELF_AGREEMENT_MIN = 0.97
MAX_CLASS_SHARE = 0.70
MIN_DISTINCT = 150

OUT = REPO / "data" / "gates" / "judge_v1" / "record"
INPUTS = OUT / "inputs.jsonl"
CALLS = {"first": OUT / "calls.first.B.record-full.jsonl",
         "recheck": OUT / "calls.recheck.A.record-off.jsonl",
         "rest": OUT / "calls.rest.B.record-full.jsonl"}
TRAP_SET = REPO / "data" / "trap_set" / "trap_set_v1.jsonl"
NOISE_SET = REPO / "data" / "noise_set" / "noise_set_v1.jsonl"
BENCH_DB = REPO / "data" / "bench" / "bench.db"
SOURCES = ("trap_set", "noise_set", "collection", "race")


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def _make_noise_set() -> Any:
    spec = importlib.util.spec_from_file_location("make_noise_set", REPO / "scripts" / "make_noise_set.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# --------------------------------------------------------------------------- inputs


def race_claims(db: Path = BENCH_DB) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Every LLM-judged claim in the bench DB, rebuilt as the race loop built it."""
    from backend.benchmark.configs import RaceConfig
    from backend.scheduler import agent_visible_metrics, race_spec
    from workers.targets.base import get_target

    conn = _ro(db)
    try:
        races = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM races")}
        arms = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM arms")}
        judged: dict[str, dict[str, Any]] = {}
        for r in conn.execute("SELECT * FROM llm_calls WHERE step_type = 'judge_improvement' ORDER BY created_at"):
            for eid in json.loads(r["experiment_ids_json"] or "[]"):
                if eid not in judged or r["valid"]:
                    judged[eid] = dict(r)
        exps = [dict(r) for r in conn.execute(
            "SELECT * FROM experiments WHERE status = 'succeeded' ORDER BY ended_at, id")]
    finally:
        conn.close()
    stats: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for arm_id, arm in sorted(arms.items(), key=lambda kv: (races[kv[1]["race_id"]]["created_at"], kv[0])):
        race = races[arm["race_id"]]
        cfg = RaceConfig.model_validate(json.loads(race["config_json"] or "{}"))
        spec = race_spec(cfg, get_target(race["target"]).spec)
        best: dict[str, Any] | None = None
        for e in (x for x in exps if x["arm_id"] == arm_id):
            e["config"] = json.loads(e["config_json"] or "{}")
            e["metrics"] = json.loads(e["metrics_json"] or "{}")
            e["summary"] = json.loads(e["summary_json"] or "{}")
            if e["id"] in judged and not e["is_baseline"]:
                bb = e["summary"].get("best_before")
                if best is None or bb is None or abs(float(best["score"]) - float(bb)) > 1e-9:
                    stats["skipped_prev_best_mismatch"] += 1
                else:
                    claim = JudgeClaim(prev_best_score=best["score"], prev_best_config=best["config"],
                                       new_score=float(e["score"] or 0.0), new_config=e["config"],
                                       evidence=dict(agent_visible_metrics(e["metrics"])))
                    call = judged[e["id"]]
                    out.append({
                        "orig_id": e["id"], "target": race["target"], "split": None,
                        "group": f"race/{race['target']}/{race['id'][:8]}",
                        "expected": {"recorded_v0_verdict": e["judge_verdict"], "is_trap": bool(e["is_trap"])},
                        "recorded_v0": {"mubit_request_id": call.get("mubit_request_id"),
                                        "total_input_tokens": int((call.get("input_tokens") or 0)
                                                                  + (call.get("cache_creation_tokens") or 0)
                                                                  + (call.get("cache_read_tokens") or 0))},
                        "user_message": build_judge_user(spec, claim),
                    })
                    stats["rebuilt"] += 1
            if e["is_best"]:
                best = e
    return out, dict(stats)


def collection_claims(exclude_experiment_ids: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mns = _make_noise_set()
    out: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for target in ("rl", "cifar", "lm"):
        try:
            claims = mns.extract_claims(target, traps_per_group=10**6)
        except Exception as e:  # noqa: BLE001
            stats[target] = f"skipped: {type(e).__name__}: {e}"
            continue
        kept = [c for c in claims if c["candidate"]["experiment_id"] not in exclude_experiment_ids]
        stats[target] = {"claims": len(claims), "not_in_noise_set": len(kept)}
        for c in kept:
            out.append({
                "orig_id": c["candidate"]["experiment_id"], "target": target, "split": c["split"],
                "group": c["group"], "expected": {"kind": c["kind"], "recorded_v0_verdict": c["recorded_judge"]["verdict"]},
                "recorded_v0": {"mubit_request_id": c["recorded_judge"]["mubit_request_id"],
                                "total_input_tokens": c["recorded_judge"]["total_input_tokens"]},
                "user_message": c["user_message"],
            })
    return out, stats


def build_inputs() -> dict[str, Any]:
    if any(p.exists() for p in CALLS.values()):
        raise SystemExit("calls already exist for these inputs; refusing to rebuild (ids and the first batch would move)")
    trap = _jsonl(TRAP_SET)
    noise = _jsonl(NOISE_SET)
    rows: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for r in trap:
        if r["split"] != "dev":
            excluded["trap_set/test"] += 1
            continue
        rows.append({"source": "trap_set", "orig_id": r["id"], "target": r["target"], "split": r["split"],
                     "group": "trap_set", "expected": {"label": r["label"], "acceptable": r["acceptable_verdicts"],
                                                       "category": r["category"]},
                     "user_message": r["user_message"]})
    for r in noise:
        if r["split"] != "dev":
            excluded["noise_set/test"] += 1
            continue
        rows.append({"source": "noise_set", "orig_id": r["id"], "target": r["target"], "split": r["split"],
                     "group": r["group"], "expected": {"truth": r["truth"],
                                                       "recorded_v0_verdict": (r.get("recorded_judge") or {}).get("verdict")},
                     "recorded_v0": {"total_input_tokens": (r.get("recorded_judge") or {}).get("total_input_tokens")},
                     "user_message": r["user_message"]})
    noise_eids = {r["candidate"]["experiment_id"] for r in noise}
    coll, coll_stats = collection_claims(noise_eids)
    rows += [{"source": "collection", **c} for c in coll]
    race, race_stats = race_claims()
    rows += [{"source": "race", **c} for c in race]

    # Held-out messages (test splits) must not slip in through another source.
    held_out = {_sha(r["user_message"]) for r in trap + noise if r["split"] != "dev"}
    seen: set[str] = set()
    inputs: list[dict[str, Any]] = []
    per_source: Counter[str] = Counter()
    for r in rows:
        h = _sha(r["user_message"])
        if h in held_out:
            excluded[f"{r['source']}/equals_held_out_message"] += 1
            continue
        if h in seen:
            excluded[f"{r['source']}/duplicate"] += 1
            continue
        seen.add(h)
        per_source[r["source"]] += 1
        r["n"] = per_source[r["source"]]
        r["id"] = f"jv1_{r['source']}_{r['n']:03d}"
        r["msg_sha256"] = h
        r["session_id"] = f"collect-judgev1-{DATE}/{r['source']}/{r['n']}"
        inputs.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    INPUTS.write_text("".join(json.dumps(r) + "\n" for r in inputs), encoding="utf-8")
    pred = Counter(predicted_verdict(r) for r in inputs)
    summary = {"n": len(inputs), "by_source": dict(per_source), "excluded": dict(excluded),
               "collection": coll_stats, "race": race_stats, "predicted_verdict": dict(pred),
               "predicted_max_share": round(max(pred.values()) / len(inputs), 4) if inputs else None}
    return summary


def predicted_verdict(r: dict[str, Any]) -> str:
    """A prior for balance planning only (the real classes come from Opus's answers)."""
    e = r["expected"]
    if r["source"] == "trap_set":
        return e["label"]
    if r["source"] == "noise_set":
        return "invalid" if e["truth"] == "fake" else "unsure"
    if r["source"] == "collection":
        return "invalid" if e["kind"] == "trap" else "unsure"
    return e.get("recorded_v0_verdict") or "unsure"


def first_batch_ids(inputs: list[dict[str, Any]], n: int = FIRST_BATCH) -> list[str]:
    """Stratified by source (largest remainder), seeded sample inside each source."""
    by: dict[str, list[dict[str, Any]]] = {}
    for r in inputs:
        by.setdefault(r["source"], []).append(r)
    total = len(inputs)
    quota = {s: n * len(v) / total for s, v in by.items()}
    alloc = {s: int(q) for s, q in quota.items()}
    for s in sorted(quota, key=lambda s: -(quota[s] - alloc[s]))[: n - sum(alloc.values())]:
        alloc[s] += 1
    rng = random.Random(FIRST_SEED)
    ids: list[str] = []
    for s in sorted(by):
        ids += [r["id"] for r in rng.sample(sorted(by[s], key=lambda r: r["id"]), alloc[s])]
    return sorted(ids)


# --------------------------------------------------------------------------- run


def total_spend() -> float:
    return sum(r.get("cost_usd") or 0.0 for p in CALLS.values() for r in _jsonl(p))


def final_outputs(phase: str) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for r in _jsonl(CALLS[phase]):
        if r.get("valid"):
            out[r["case_id"]] = (r["verdict"], r["reason_code"])
    return out


def recheck_result() -> dict[str, Any]:
    a, b = final_outputs("first"), final_outputs("recheck")
    ids = sorted(set(a) & set(b))
    full = sum(a[i] == b[i] for i in ids)
    verd = sum(a[i][0] == b[i][0] for i in ids)
    return {"n": len(ids), "n_first": len(a), "n_recheck": len(b),
            "full_agree": full, "full_rate": round(full / len(ids), 4) if ids else None,
            "verdict_agree": verd, "verdict_rate": round(verd / len(ids), 4) if ids else None,
            "disagreements": {i: {"first_B": list(a[i]), "recheck_A": list(b[i])} for i in ids if a[i] != b[i]},
            "pass": bool(ids) and len(ids) >= FIRST_BATCH and full / len(ids) >= SELF_AGREEMENT_MIN}


def preflight() -> tuple[str, dict[str, Any]]:
    if CURRENT_VERSION[STEP] != VERSION or FROZEN[STEP] != VERSION:
        raise SystemExit(f"judge prompt is {CURRENT_VERSION[STEP]} (FROZEN={FROZEN[STEP]}), expected frozen {VERSION}")
    fp = prompt_fingerprint(STEP)
    if fp != EXPECTED_FINGERPRINT:
        raise SystemExit(f"judge v1 fingerprint {fp} != {EXPECTED_FINGERPRINT}")
    return load_system_prompt(STEP), judge_wire_schema(VERSION)


def set_project_key(project: str) -> None:
    """Point MUBIT_API_KEY (this process only) at project A or B. The key is never printed."""
    from dotenv import dotenv_values

    env = dotenv_values(REPO / ".env")
    name = "MUBIT_API_KEY_B" if project == "B" else "MUBIT_API_KEY"
    key = env.get(name)
    if not key:
        raise SystemExit(f"{name} missing from .env")
    if project == "B" and key == env.get("MUBIT_API_KEY"):
        raise SystemExit("MUBIT_API_KEY_B equals MUBIT_API_KEY; refusing")
    os.environ["MUBIT_API_KEY"] = key


async def run(phase: str, concurrency: int, yes: bool, limit: int | None = None) -> None:
    from backend.agent.mubit_client import MubitClient  # the ONLY LLM client (Mubit gateway)

    system, schema = preflight()
    inputs = _jsonl(INPUTS)
    if len(inputs) < MIN_DISTINCT:
        raise SystemExit(f"only {len(inputs)} distinct inputs (< {MIN_DISTINCT}); D39 (3) not met")
    first = set(first_batch_ids(inputs))
    if phase in ("first", "recheck"):
        todo = [r for r in inputs if r["id"] in first]
    else:
        rc = recheck_result()
        if not rc["pass"]:
            raise SystemExit(f"recheck has not passed ({rc['full_agree']}/{rc['n']}); refusing to record the rest")
        todo = [r for r in inputs if r["id"] not in first]
    record = "off" if phase == "recheck" else "full"
    project = "A" if phase == "recheck" else "B"
    if record == "full" and not yes:
        raise SystemExit("recording needs --yes")
    set_project_key(project)
    path = CALLS[phase]
    done = {r["case_id"] for r in _jsonl(path) if r.get("valid")}
    todo = [r for r in todo if r["id"] not in done]
    if limit:
        todo = todo[:limit]
    spent = total_spend()
    print(f"phase {phase}: project {project}, record {record}, {len(todo)} to run, prompt {VERSION}:"
          f"{prompt_fingerprint(STEP)}, spent so far ${spent:.4f} (cap ${LLM_CAP_USD})", flush=True)
    client = MubitClient(arm_models={ArmKind.opus: OPUS})
    sem = asyncio.Semaphore(min(concurrency, 4))
    lock = asyncio.Lock()
    state = {"spent": spent, "stopped": False}

    async def one(case: dict[str, Any]) -> None:
        async with sem:
            sid = case["session_id"] if phase != "recheck" else case["session_id"].replace("collect-", "recheck-", 1)
            for attempt in range(2):  # one retry on a failed reply
                if state["spent"] >= LLM_CAP_USD:
                    state["stopped"] = True
                    return
                parsed, rec = await client.call(STEP, system, case["user_message"], JudgeOutput, ArmKind.opus, sid,
                                                record, wire_schema=schema, model=OPUS, is_retry=attempt > 0)
                ok = parsed is not None and parsed.reason_code is not None
                row = {
                    "case_id": case["id"], "source": case["source"], "phase": phase, "project": project,
                    "record": record, "prompt_version": VERSION, "prompt_fingerprint": prompt_fingerprint(STEP),
                    "model": OPUS, "attempt": attempt, "valid": ok,
                    "verdict": parsed.verdict.value if ok else None,
                    "reason_code": parsed.reason_code.value if ok and parsed.reason_code else None,
                    "error": rec.error if parsed is not None or rec.error else "no reason_code",
                    "mubit_request_id": rec.mubit_request_id, "recorded": rec.recorded, "session_id": sid,
                    "latency_ms": rec.latency_ms, "input_tokens": rec.input_tokens,
                    "cache_creation_tokens": rec.cache_creation_tokens, "cache_read_tokens": rec.cache_read_tokens,
                    "output_tokens": rec.output_tokens, "cost_usd": rec.cost_usd, "stop_reason": rec.stop_reason,
                }
                async with lock:
                    state["spent"] += rec.cost_usd or 0.0
                    with path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                if ok:
                    return

    try:
        await asyncio.gather(*(one(c) for c in todo))
    finally:
        await client.aclose()
    print(f"done; spent total ${state['spent']:.4f}{' (CAP HIT, stopped)' if state['stopped'] else ''}", flush=True)


# --------------------------------------------------------------------------- report


def _p50(xs: list[float]) -> float | None:
    return round(sorted(xs)[len(xs) // 2], 1) if xs else None


def report() -> dict[str, Any]:
    inputs = {r["id"]: r for r in _jsonl(INPUTS)}
    rec = {**final_outputs("first"), **final_outputs("rest")}
    recorded_rows = [r for p in ("first", "rest") for r in _jsonl(CALLS[p])]
    by_src: dict[str, Counter[str]] = {}
    for cid, (v, c) in rec.items():
        by_src.setdefault(inputs[cid]["source"], Counter())[f"{v}/{c}"] += 1
    v = Counter(o[0] for o in rec.values())
    c = Counter(o[1] for o in rec.values())
    n = len(rec)
    # token check: v1 total input minus the recorded v0 total should be one constant
    deltas: dict[str, Counter[int]] = {}
    for r in recorded_rows:
        if not r.get("valid"):
            continue
        old = (inputs[r["case_id"]].get("recorded_v0") or {}).get("total_input_tokens")
        if old:
            new = r["input_tokens"] + r["cache_creation_tokens"] + r["cache_read_tokens"]
            deltas.setdefault(inputs[r["case_id"]]["source"], Counter())[new - old] += 1
    acc: dict[str, Counter[str]] = {}
    for cid, (verdict, _) in rec.items():
        i = inputs[cid]
        if i["source"] == "trap_set":
            acc.setdefault("trap_set_dev", Counter())["correct" if verdict in i["expected"]["acceptable"] else "wrong"] += 1
    def first_ids(p: str) -> list[str]:
        return [r["mubit_request_id"] for r in _jsonl(CALLS[p]) if r.get("valid")][:3]
    out = {
        "step": STEP.value, "prompt_version": VERSION, "prompt_fingerprint": prompt_fingerprint(STEP),
        "wire_schema": judge_wire_schema(VERSION),
        "inputs": {"n": len(inputs), "by_source": dict(Counter(r["source"] for r in inputs.values()))},
        "recorded": {"n_cases": n, "calls": len(recorded_rows),
                     # a failed call (e.g. gateway 503 before the provider) yields no example
                     "recorded_valid_calls": sum(bool(r.get("recorded")) and bool(r.get("valid")) for r in recorded_rows),
                     "failed_calls": sum(not r.get("valid") for r in recorded_rows),
                     "by_source": {s: dict(x.most_common()) for s, x in sorted(by_src.items())},
                     "verdict": dict(v.most_common()), "reason_code": dict(c.most_common()),
                     "max_verdict_share": round(max(v.values()) / n, 4) if n else None,
                     "code_verdict_mismatch": sum(JUDGE_REASON_VERDICT[JudgeReasonCode(k)].value != vv
                                                  for vv, k in rec.values())},
        "recheck": recheck_result(),
        "trap_set_dev_accuracy": {k: dict(x) for k, x in acc.items()},
        "token_delta_v1_minus_v0_by_source": {s: dict(x.most_common()) for s, x in deltas.items()},
        "cost_usd": {p: round(sum(r.get("cost_usd") or 0 for r in _jsonl(f)), 4) for p, f in CALLS.items()},
        "total_cost_usd": round(total_spend(), 4),
        "first_request_ids_valid": {p: first_ids(p) for p in CALLS},
        "first_request_id_any": {p: (_jsonl(f)[0]["mubit_request_id"] if _jsonl(f) else None) for p, f in CALLS.items()},
        "latency_p50_ms": {p: _p50([r["latency_ms"] for r in _jsonl(f) if r.get("valid") and r.get("latency_ms")])
                           for p, f in CALLS.items()},
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--phase", choices=list(CALLS), required=True)
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--yes", action="store_true")
    r.add_argument("--limit", type=int, default=None)
    sub.add_parser("report")
    ns = ap.parse_args()
    if ns.cmd == "build":
        print(json.dumps(build_inputs(), indent=1))
    elif ns.cmd == "run":
        asyncio.run(run(ns.phase, ns.concurrency, ns.yes, ns.limit))
    else:
        print(json.dumps(report(), indent=1))


if __name__ == "__main__":
    main()
