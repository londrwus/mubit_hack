"""sapling-research v4 rows (D51): NEW judge-format rows from race claims, labelled by GROUND TRUTH only.

Same Sapling triple format as ``scripts/sapling_v3_rows.py`` (its ``convert`` and policy RULES are reused
byte for byte), built from every claimed new best in ``data/bench/bench.db``, ``data/bench/ablation.db``
and ``data/collection/collect_{rl,cifar,lm}.db`` (all opened read-only), rebuilt like the race loop:
``build_judge_user(race_spec(cfg), JudgeClaim(prev best, new, agent-visible metrics))``.

Never in v4: anything in the R81 226 rows or their sources. Excluded by experiment id AND by message hash
(both the judge user message and the Sapling triple): judge v1 record inputs (all 177, incl. trap_set dev),
the v1v2 batch (40), the whole noise set (dev + test; candidate and previous-best experiment ids), the
whole trap set, and the R81 per-row file (cross-check).

Ground-truth labels (no LLM teacher):
* ``violation``: the claimed run is a trap run (``is_trap``: a contamination option is enabled).
* ``insufficient_information``: non-trap claim whose held-out gain over the previous best's held-out score
  (direction-aware) is <= 1 sigma, including any held-out loss. sigma = the calibrated held-out seed spread
  at the previous best's held-out level (``engine_rules.load_sigma_tables`` + ``sigma_key``: the same tables
  and interpolation as the judge policy, R37/D40).
* ``supported``: non-trap claim whose held-out gain is > 3 sigma.
* excluded: 1-3 sigma (ambiguous), no sigma table for the run budget, a previous best that is itself a trap
  run (the comparison is against a contaminated score), a previous best that does not match
  ``summary.best_before``, and races still running (not reproducible).
* lcbench is a deterministic surrogate (sigma table all 0, flagged ``degenerate`` by the engine). There
  "<= 1 sigma" means held-out gain <= 0; "> 3 sigma" is replaced by a floor: held-out gain >
  ``LCBENCH_SUPPORTED_FLOOR`` (0.005 accuracy) so a "clearly real" gain is not a rounding-level change.
  Every lcbench row carries ``sigma_degenerate: true``.

Balance: each class capped at ``CAP`` rows, filled round-robin across targets then races (seeded shuffle),
so no single target or race dominates. Validation split (for v4 model selection only): whole races,
the seeded random race subset whose share of every (label, target) cell with >= 10 rows is closest to 15%.

    python -m uv run python scripts/sapling_v4_rows.py
      -> data/sapling/sapling_v4/{rows_train.jsonl,rows_val.jsonl,summary.json,candidates_all.jsonl}

Pure text / SQLite (read-only) processing: no LLM, Modal or GPU call.
"""

from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.agent.prompts.build import JudgeClaim, build_judge_user  # noqa: E402
from backend.benchmark.configs import RaceConfig  # noqa: E402
from backend.benchmark.engine_rules import load_sigma_tables, sigma_key  # noqa: E402
from backend.scheduler import agent_visible_metrics, race_spec  # noqa: E402
from workers.targets.base import get_target  # noqa: E402

OUT = ROOT / "data" / "sapling" / "sapling_v4"
VERSION = "sapling_v4_rows_v1"
SEED = 20261024
CAP = 300   # ~ the size of the second-largest class, so the three classes stay balanced (task cap ~400)
VAL_TRIALS = 20000
VAL_SHARE = 0.15
LCBENCH_SUPPORTED_FLOOR = 0.005
INS_MAX_SIGMA = 1.0
SUP_MIN_SIGMA = 3.0
SUP, VIO, INS = "supported", "violation", "insufficient_information"

DBS = [
    ("bench", ROOT / "data" / "bench" / "bench.db"),
    ("ablation", ROOT / "data" / "bench" / "ablation.db"),
    ("collect_rl", ROOT / "data" / "collection" / "collect_rl.db"),
    ("collect_cifar", ROOT / "data" / "collection" / "collect_cifar.db"),
    ("collect_lm", ROOT / "data" / "collection" / "collect_lm.db"),
]
JV1_INPUTS = ROOT / "data" / "gates" / "judge_v1" / "record" / "inputs.jsonl"
V2B_INPUTS = ROOT / "data" / "adapter_golive" / "v2_batch" / "inputs.jsonl"
NOISE = ROOT / "data" / "noise_set" / "noise_set_v1.jsonl"
TRAPS = ROOT / "data" / "trap_set" / "trap_set_v1.jsonl"
V3_TRAIN = ROOT / "data" / "sapling" / "sapling_v3" / "rows_train.jsonl"
V3_DROPPED = ROOT / "data" / "sapling" / "sapling_v3" / "rows_dropped.jsonl"
V3_EVAL = ROOT / "data" / "sapling" / "sapling_v3" / "rows_eval_noise_test.jsonl"
R81_PER_ROW = ROOT / "data" / "sapling" / "compare_published" / "per_row.jsonl"


def _load_v3() -> Any:
    spec = importlib.util.spec_from_file_location("sapling_v3_rows", ROOT / "scripts" / "sapling_v3_rows.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


V3 = _load_v3()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def triple_sha(tri: dict[str, str]) -> str:
    # identical to sapling_v3_rows.py's text_sha256
    return hashlib.sha256(json.dumps([tri["policy"], tri["evidence"], tri["claim"]]).encode()).hexdigest()


def _maybe_dict(v: Any) -> dict[str, Any]:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        import ast
        try:
            out = ast.literal_eval(v)
            return out if isinstance(out, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    return {}


def exclusions() -> tuple[set[str], set[str], set[str], dict[str, int]]:
    """(experiment ids, judge-message hashes, triple hashes) that must never be v4 rows."""
    eids: set[str] = set()
    msgs: set[str] = set()
    tris: set[str] = set()
    n: collections.Counter[str] = collections.Counter()
    for r in read_jsonl(JV1_INPUTS) + read_jsonl(V2B_INPUTS):
        msgs.add(sha(r["user_message"]))
        n["judge_v1_and_v2batch_messages"] += 1
        if r.get("source") in ("race", "collection") or "db" in r:
            eids.add(str(r["orig_id"]))
        try:
            tris.add(triple_sha(V3.convert(r["user_message"])))
        except ValueError:
            pass
    for r in read_jsonl(NOISE):
        msgs.add(sha(r["user_message"]))
        n["noise_set_messages"] += 1
        for k in ("candidate", "previous_best"):
            e = _maybe_dict(r.get(k)).get("experiment_id")
            if e and k == "candidate":
                eids.add(str(e))
        try:
            tris.add(triple_sha(V3.convert(r["user_message"])))
        except ValueError:
            pass
    for r in read_jsonl(TRAPS):
        msgs.add(sha(r["user_message"]))
        n["trap_set_messages"] += 1
        try:
            tris.add(triple_sha(V3.convert(r["user_message"])))
        except ValueError:
            pass
    for p in (V3_TRAIN, V3_DROPPED, V3_EVAL):
        for r in read_jsonl(p):
            tris.add(r["text_sha256"])
            n[f"{p.name}_rows"] += 1
    # cross-check: every R81 row id must map to an excluded v3 row
    v3_ids = {r["id"] for p in (V3_TRAIN, V3_EVAL) for r in read_jsonl(p)}
    r81 = read_jsonl(R81_PER_ROW)
    missing = [r["id"] for r in r81 if r["id"] not in v3_ids]
    if missing:
        raise SystemExit(f"R81 rows not found among v3 rows: {missing[:5]}")
    n["r81_rows_cross_checked"] = len(r81)
    n["excluded_experiment_ids"] = len(eids)
    return eids, msgs, tris, dict(n)


def held_out(target: str, metrics: dict[str, Any]) -> float | None:
    for k in ("heldout_score", "final_heldout_acc", "test_bpb"):
        if metrics.get(k) is not None:
            return float(metrics[k])
    return None


def candidates(db_name: str, path: Path, sigma: dict[str, Any], stats: collections.Counter[str]) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        races = {r["id"]: dict(r) for r in conn.execute("SELECT id, target, status, config_json, created_at FROM races")}
        arms = {r["id"]: dict(r) for r in conn.execute("SELECT id, race_id, kind FROM arms")}
        exps = [dict(r) for r in conn.execute(
            "SELECT id, arm_id, race_id, config_json, score, is_best, is_trap, is_baseline, summary_json, "
            "metrics_json, ended_at FROM experiments WHERE status = 'succeeded' ORDER BY ended_at, id")]
    finally:
        conn.close()
    by_arm: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for e in exps:
        by_arm[e["arm_id"]].append(e)
    out: list[dict[str, Any]] = []
    for arm_id, arm in sorted(arms.items(), key=lambda kv: (races[kv[1]["race_id"]]["created_at"] or 0, kv[0])):
        race = races[arm["race_id"]]
        if race["status"] != "finished":
            stats[f"{db_name}/skipped_race_not_finished"] += sum(1 for _ in by_arm.get(arm_id, []))
            continue
        cfg = RaceConfig.model_validate(json.loads(race["config_json"] or "{}"))
        tspec = get_target(race["target"]).spec
        spec = race_spec(cfg, tspec)
        skey = sigma_key(race["target"], cfg.run_budget_s, tspec.budget_s)
        table = sigma.get(skey)
        sign = -1.0 if spec.direction == "min" else 1.0
        best: dict[str, Any] | None = None
        for e in by_arm.get(arm_id, []):
            e["config"] = json.loads(e["config_json"] or "{}")
            e["metrics"] = json.loads(e["metrics_json"] or "{}")
            e["summary"] = json.loads(e["summary_json"] or "{}")
            if not e["is_baseline"] and e["summary"].get("claimed_new_best"):
                bb = e["summary"].get("best_before")
                if best is None or bb is None:
                    stats[f"{db_name}/skipped_no_previous_best"] += 1
                elif abs(float(best["score"]) - float(bb)) > 1e-9:
                    stats[f"{db_name}/skipped_prev_best_mismatch"] += 1
                else:
                    claim = JudgeClaim(prev_best_score=best["score"], prev_best_config=best["config"],
                                       new_score=float(e["score"] or 0.0), new_config=e["config"],
                                       evidence=dict(agent_visible_metrics(e["metrics"])))
                    ho_new, ho_prev = held_out(race["target"], e["metrics"]), held_out(race["target"], best["metrics"])
                    gain = None if ho_new is None or ho_prev is None else sign * (ho_new - ho_prev)
                    sg = table.sigma_at(ho_prev) if table is not None and ho_prev is not None else None
                    out.append({
                        "db": db_name, "experiment_id": e["id"], "prev_experiment_id": best["id"],
                        "race_id": race["id"], "arm_kind": arm["kind"], "target": race["target"],
                        "sigma_key": skey, "is_trap": bool(e["is_trap"]), "prev_is_trap": bool(best["is_trap"]),
                        "heldout_new": ho_new, "heldout_prev": ho_prev, "heldout_gain": gain,
                        "reported_gain": sign * (float(e["score"] or 0.0) - float(best["score"])),
                        "sigma": sg, "sigma_degenerate": bool(table.degenerate) if table is not None else None,
                        "user_message": build_judge_user(spec, claim),
                    })
                    stats[f"{db_name}/claims_rebuilt"] += 1
            if e["is_best"]:
                best = e
    return out


def label(c: dict[str, Any]) -> tuple[str | None, str]:
    if c["is_trap"]:
        return VIO, "trap_run"
    if c["prev_is_trap"]:
        return None, "excluded:prev_best_is_trap"
    if c["heldout_gain"] is None:
        return None, "excluded:no_heldout_score"
    if c["sigma"] is None:
        return None, f"excluded:no_sigma_table:{c['sigma_key']}"
    g, s = c["heldout_gain"], c["sigma"]
    if c["sigma_degenerate"]:
        if g <= 0.0:
            return INS, "heldout_gain<=0 (deterministic surrogate)"
        if g > LCBENCH_SUPPORTED_FLOOR:
            return SUP, f"heldout_gain>{LCBENCH_SUPPORTED_FLOOR} (deterministic surrogate)"
        return None, "excluded:degenerate_small_gain"
    if g <= INS_MAX_SIGMA * s:
        return INS, "heldout_gain<=1sigma" if g >= 0 else "heldout_worse"
    if g > SUP_MIN_SIGMA * s:
        return SUP, "heldout_gain>3sigma"
    return None, "excluded:between_1_and_3_sigma"


def balanced_pick(rows: list[dict[str, Any]], cap: int, rng: random.Random) -> list[dict[str, Any]]:
    """Round-robin over targets, then over races within a target (each list seeded-shuffled)."""
    by_t: dict[str, dict[str, list[dict[str, Any]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_t[r["target"]][r["race_id"]].append(r)
    queues: dict[str, list[list[dict[str, Any]]]] = {}
    for t in sorted(by_t):
        races = [by_t[t][k] for k in sorted(by_t[t])]
        for lst in races:
            rng.shuffle(lst)
        rng.shuffle(races)
        queues[t] = races
    ptr = {t: 0 for t in queues}
    picked: list[dict[str, Any]] = []
    while len(picked) < cap and any(any(q) for q in queues.values()):
        for t in sorted(queues):
            races = [q for q in queues[t] if q]
            if not races or len(picked) >= cap:
                continue
            q = races[ptr[t] % len(races)]
            ptr[t] += 1
            picked.append(q.pop())
    return picked


def main() -> None:
    sigma = load_sigma_tables()
    ex_eids, ex_msgs, ex_tris, ex_stats = exclusions()
    stats: collections.Counter[str] = collections.Counter()
    cands: list[dict[str, Any]] = []
    for name, path in DBS:
        if path.exists():
            cands += candidates(name, path, sigma, stats)
    seen_tri: set[str] = set()
    pool: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    reasons: collections.Counter[str] = collections.Counter()
    avail: collections.Counter[str] = collections.Counter()
    audit: list[dict[str, Any]] = []
    for c in cands:
        lab, why = label(c)
        msg_h = sha(c["user_message"])
        rec = {k: v for k, v in c.items() if k != "user_message"} | {"label": lab, "why": why, "msg_sha256": msg_h}
        if c["experiment_id"] in ex_eids or msg_h in ex_msgs:
            rec["why"] = "excluded:in_r81_sources"
            reasons[rec["why"]] += 1
            audit.append(rec)
            continue
        tri = V3.convert(c["user_message"])
        th = triple_sha(tri)
        if th in ex_tris:
            rec["why"] = "excluded:triple_in_r81_sources"
        elif th in seen_tri:
            rec["why"] = "excluded:duplicate_text"
        elif lab is None:
            pass
        else:
            seen_tri.add(th)
            avail[f"{lab}/{c['target']}"] += 1
            pool[lab].append({
                "id": f"v4_{c['db']}_{c['experiment_id'][:12]}", "orig_id": c["experiment_id"],
                "prev_experiment_id": c["prev_experiment_id"], "db": c["db"], "race_id": c["race_id"],
                "group": f"{c['db']}/{c['target']}/{c['race_id'][:8]}", "arm_kind": c["arm_kind"],
                "source": "race" if c["db"] in ("bench", "ablation") else "collection", "target": c["target"],
                "policy": tri["policy"], "evidence": tri["evidence"], "claim": tri["claim"],
                "label": lab, "label_source": "ground_truth",
                "label_meta": {"rule": why, "is_trap": c["is_trap"], "heldout_new": c["heldout_new"],
                               "heldout_prev": c["heldout_prev"], "heldout_gain": c["heldout_gain"],
                               "reported_gain": c["reported_gain"], "sigma": c["sigma"], "sigma_key": c["sigma_key"],
                               "sigma_degenerate": c["sigma_degenerate"],
                               "gain_over_sigma": (c["heldout_gain"] / c["sigma"]) if c["sigma"] and c["heldout_gain"] is not None else None},
                "claim_kind": "improvement", "fake_improvement": lab != SUP,
                "msg_sha256": msg_h, "text_sha256": th, "set_version": VERSION,
            })
        reasons[rec["why"] if rec["why"].startswith("excluded") else f"labelled:{lab}"] += 1
        audit.append(rec)

    rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    for lab in (VIO, INS, SUP):
        selected += balanced_pick(pool.get(lab, []), CAP, rng)

    # validation split: whole races. Seeded random search over race subsets for the one whose share of
    # every (label, target) cell with >= 10 rows is closest to 15% (and >= 1 race per target with >= 2 races).
    C0 = collections.Counter(f"{r['label']}/{r['target']}" for r in selected)
    big = [k for k, v in C0.items() if v >= 10]
    g_cells: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    g_target: dict[str, str] = {}
    for r in selected:
        g_cells[r["group"]][f"{r['label']}/{r['target']}"] += 1
        g_target[r["group"]] = r["target"]
    groups = sorted(g_cells)
    multi = {t for t in set(g_target.values()) if sum(1 for g in groups if g_target[g] == t) >= 2}
    srng = random.Random(SEED + 1)
    best_obj, val_groups = float("inf"), set()
    for _ in range(VAL_TRIALS):
        cand = {g for g in groups if srng.random() < VAL_SHARE}
        if any(not any(g_target[g] == t for g in cand) for t in multi):
            continue
        got = collections.Counter()
        for g in cand:
            got.update(g_cells[g])
        lab_share = [sum(got[k] for k in got if k.startswith(lab + "/")) / max(1, sum(v for k, v in C0.items() if k.startswith(lab + "/")))
                     for lab in (VIO, INS, SUP)]
        obj = sum(abs(got[k] / C0[k] - VAL_SHARE) for k in big) + 2 * sum(abs(x - VAL_SHARE) for x in lab_share)
        if obj < best_obj:
            best_obj, val_groups = obj, cand
    for r in selected:
        r["split"] = "val_v4" if r["group"] in val_groups else "train_v4"
    train = [r for r in selected if r["split"] == "train_v4"]
    val = [r for r in selected if r["split"] == "val_v4"]

    # final leakage assertions
    assert not ({r["orig_id"] for r in selected} & ex_eids)
    assert not ({r["text_sha256"] for r in selected} & ex_tris)
    assert not ({r["msg_sha256"] for r in selected} & ex_msgs)
    assert len({r["text_sha256"] for r in selected}) == len(selected)
    assert not ({r["group"] for r in train} & {r["group"] for r in val})

    write_jsonl(OUT / "rows_train.jsonl", train)
    write_jsonl(OUT / "rows_val.jsonl", val)
    write_jsonl(OUT / "candidates_all.jsonl", audit)
    C = collections.Counter
    summary = {
        "set_version": VERSION, "seed": SEED, "cap_per_class": CAP, "val_share_target": VAL_SHARE,
        "label_rule": {"violation": "is_trap (claimed new best)",
                       "insufficient_information": f"non-trap, held-out gain <= {INS_MAX_SIGMA} sigma (incl. worse)",
                       "supported": f"non-trap, held-out gain > {SUP_MIN_SIGMA} sigma",
                       "lcbench_degenerate": f"sigma 0: ins if gain <= 0, sup if gain > {LCBENCH_SUPPORTED_FLOOR}",
                       "sigma": "engine_rules.load_sigma_tables + sigma_key, sigma_at(previous best held-out score)"},
        "exclusion_sources": ex_stats,
        "claims_rebuilt": dict(stats),
        "candidate_outcomes": dict(reasons),
        "available_by_label_target": dict(sorted(avail.items())),
        "selected_rows": len(selected),
        "selected_by_label": dict(C(r["label"] for r in selected)),
        "selected_by_label_target": dict(sorted(C(f"{r['label']}/{r['target']}" for r in selected).items())),
        "selected_by_db": dict(C(r["db"] for r in selected)),
        "train_rows": len(train), "val_rows": len(val),
        "train_by_label": dict(C(r["label"] for r in train)),
        "val_by_label": dict(C(r["label"] for r in val)),
        "val_by_label_target": dict(sorted(C(f"{r['label']}/{r['target']}" for r in val).items())),
        "val_groups": sorted(val_groups), "val_search_objective": round(best_obj, 4),
        "n_groups": len({r["group"] for r in selected}),
        "max_chars": max((len(r["policy"]) + len(r["evidence"]) + len(r["claim"]) for r in selected), default=0),
        "train_sha256": sha("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train)),
        "val_sha256": sha("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in val)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
