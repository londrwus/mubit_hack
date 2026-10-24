"""Stock Banyan Sapling (as published) vs sapling-research v2 vs Opus 5.5 on the 226 REAL judge-format claims.

Rows: the 186 ``train_v3`` + 40 ``eval_noise_test`` rows built by ``scripts/sapling_v3_rows.py`` (real race,
collection and noise-set claims in the live judge's format). Their H100 head inputs were cached in the R76 session
(``data/sapling/sapling_v3/h100_session/features.npy`` + ``records.jsonl``), so this is CPU only, from cached
features: no backbone, no GPU, no Modal, no LLM call.

* (a) stock Sapling = stock head (``data/sapling/stock_head/head.safetensors``) + its own ``calibration.json``. The
  statuses the base package itself produced on the H100 (``records.jsonl``) are used, and re-derived here as a check.
* (b) sapling-research v2 = the released 5-head delta + our ML-research calibration (alpha 0.01 and 0.05), applied
  with the release loader ``data/sapling/sapling_v2/hf_release/sapling_research.py``.
* (c) Opus 5.5 = recorded judge v1 verdicts (prompt ``5c1d3cfd2941``): majority of valid calls in
  ``data/gates/judge_v1/record/calls.*.jsonl`` + ``data/adapter_golive/B/traffic-*.jsonl`` (186 rows) and the R40
  record-off run ``data/noise_set/eval/noiseeval-2026-10-24-043502.jsonl`` arm ``opus`` (40 rows).

v2 never saw judge-format rows (these rows trained v3, not v2), so for v2 and stock they are out-of-distribution
held-out data. For the 77 rows labelled by the Opus teacher, Opus-vs-label is circular, so Opus is graded only on
ground-truth rows.

    python -m uv run --extra ml --with safetensors==0.8.0 python scripts/sapling_compare_published.py
"""

from __future__ import annotations

import glob
import importlib.util
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data" / "sapling" / "sapling_v3"
HF = ROOT / "data" / "sapling" / "sapling_v2" / "hf_release"
OUT = ROOT / "data" / "sapling" / "compare_published"
PROMPT_FP = "5c1d3cfd2941"
V2L = {"invalid": "violation", "unsure": "insufficient_information", "valid": "supported"}
L2V = {v: k for k, v in V2L.items()}


def jl(p: str | Path) -> list[dict[str, Any]]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def load_rows() -> list[dict[str, Any]]:
    rows = []
    for r in jl(V3 / "rows_train.jsonl"):
        m = r["label_meta"]
        rows.append({"id": r["id"], "orig_id": r["orig_id"], "set": "train_v3", "source": r["source"],
                     "target": r["target"], "truth": m["truth"], "allowed": m["allowed"], "label": r["label"],
                     "label_source": r["label_source"]})
    for r in jl(V3 / "rows_eval_noise_test.jsonl"):
        rows.append({"id": r["id"], "orig_id": r["orig_id"], "set": "eval_noise_test", "source": r["source"],
                     "target": r["target"], "truth": r["truth"], "allowed": r["allowed"], "label": r["label"],
                     "label_source": r["label_source"]})
    for r in rows:  # ground-truth class: fake (trap / planted leak), noise (10+10 seed re-runs), real (re-runs)
        r["gt"] = {"trap": "fake", "fake": "fake", "noise": "noise", "real": "real"}.get(r["truth"])
    return rows


def opus_verdicts() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    calls: dict[str, list[dict[str, Any]]] = {}
    for p in sorted(glob.glob(str(ROOT / "data/gates/judge_v1/record/calls.*.jsonl"))) + sorted(
            glob.glob(str(ROOT / "data/adapter_golive/B/traffic-*.jsonl"))):
        for c in jl(p):
            if c.get("valid") and c.get("verdict") in V2L and c.get("prompt_fingerprint") == PROMPT_FP:
                calls.setdefault(c["case_id"], []).append(
                    {"verdict": c["verdict"], "latency_ms": c.get("latency_ms"), "cost_usd": c.get("cost_usd"),
                     "file": Path(p).name})
    for c in jl(ROOT / "data/noise_set/eval/noiseeval-2026-10-24-043502.jsonl"):
        if c["arm"] == "opus" and c.get("verdict") in V2L:
            a = [x for x in c["attempts"] if x.get("valid")]
            calls.setdefault(c["id"], []).append(
                {"verdict": c["verdict"], "latency_ms": a[-1]["latency_ms"] if a else None, "cost_usd": c.get("cost_usd"),
                 "file": "noiseeval-2026-10-24-043502.jsonl"})
    out = {}
    for cid, cs in calls.items():
        cnt = Counter(x["verdict"] for x in cs).most_common()
        tie = len(cnt) > 1 and cnt[0][1] == cnt[1][1]
        out[cid] = {"verdict": None if tie else cnt[0][0], "votes": [x["verdict"] for x in cs], "tie": tie}
    flat = [x | {"case_id": k} for k, v in calls.items() for x in v]
    return out, flat


def gt_block(rows: list[dict[str, Any]], status: list[str]) -> dict[str, Any]:
    """On ground-truth rows only. ``status`` per row in Sapling words (or 'unsure' = no calibrated answer)."""
    g = [(r, s) for r, s in zip(rows, status) if r["gt"]]
    n = Counter(r["gt"] for r, _ in g)
    by = {k: dict(Counter(s for r, s in g if r["gt"] == k)) for k in ("fake", "noise", "real")}
    cnt = lambda k, s: sum(1 for r, x in g if r["gt"] == k and x == s)  # noqa: E731
    correct = sum(1 for r, s in g if s in r["allowed"])
    return {
        "n_ground_truth_rows": len(g), "n_fake": n["fake"], "n_noise": n["noise"], "n_real": n["real"],
        "fakes_caught (violation)": cnt("fake", "violation"),
        "fakes_approved (supported)": cnt("fake", "supported"),
        "fakes_called_not_established": cnt("fake", "insufficient_information"),
        "real_gains_accepted (supported)": cnt("real", "supported"),
        "real_gains_rejected_as_violation": cnt("real", "violation"),
        "noise_accepted (supported)": cnt("noise", "supported"),
        "noise_called_not_established (correct)": cnt("noise", "insufficient_information"),
        "unsure (no calibrated answer)": sum(1 for _, s in g if s == "unsure"),
        "unsure_rate": round(sum(1 for _, s in g if s == "unsure") / len(g), 4),
        "correct (status in the truth's allowed labels)": correct,
        "statuses_by_truth": by,
    }


def opus_agree(rows: list[dict[str, Any]], status: list[str], opus: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs = [(s, opus[r["orig_id"]]["verdict"]) for r, s in zip(rows, status)
             if r["orig_id"] in opus and opus[r["orig_id"]]["verdict"]]
    ans = [(s, o) for s, o in pairs if s != "unsure"]
    return {"rows_with_opus_verdict": len(pairs),
            "agree_strict (unsure counts as disagreement)": sum(L2V.get(s) == o for s, o in pairs),
            "agree_rate_strict": round(sum(L2V.get(s) == o for s, o in pairs) / len(pairs), 4),
            "answered": len(ans), "agree_among_answered": sum(L2V.get(s) == o for s, o in ans),
            "confusion (verifier -> opus)": dict(Counter(f"{s}->{o}" for s, o in pairs))}


def main() -> None:
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(2)  # a timing-sensitive race shares this machine
    spec = importlib.util.spec_from_file_location("sapling_research", HF / "sapling_research.py")
    sr = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(sr)

    rows = load_rows()
    assert len(rows) == 226, len(rows)
    recs = {r["id"]: r for r in jl(V3 / "h100_session/records.jsonl")}
    feats = np.load(V3 / "h100_session/features.npy")
    X = feats[[recs[r["id"]]["feature_index"] for r in rows]]

    # (a) stock: package's own statuses on the H100 + re-derived from cached features as a check
    stock_state = {k: v.float() for k, v in load_file(str(ROOT / "data/sapling/stock_head/head.safetensors")).items()}
    stock_cal = json.loads((ROOT / "data/sapling/stock_head/calibration.json").read_text(encoding="utf-8"))
    with torch.inference_mode():
        xt = torch.from_numpy(np.ascontiguousarray(X))
        stock_lg = sr._head_forward(stock_state, xt[:, 0], xt[:, 1], xt[:, 2]).numpy()
    rec_lg = np.array([recs[r["id"]]["logits"] for r in rows], np.float32)
    stock_parity = float(np.abs(stock_lg - rec_lg).max())
    stock_status = [recs[r["id"]]["status"] for r in rows]
    stock_raw = [recs[r["id"]]["raw_label"] for r in rows]
    stock_recalc = [p["status"] for p in sr.calibrate(rec_lg, stock_cal)]

    # (b) v2 via the release loader
    deltas = sr.load_deltas(HF / sr.DELTA_FILE)
    v2_lg = sr.ensemble_logits_from_features(X, stock_state, deltas)
    ids3 = {json.loads(x)["id"]: i for i, x in enumerate(open(V3 / "final_v3_ids.jsonl", encoding="utf-8"))}
    ref = np.load(V3 / "final_v2_rescored_logits.npy")[[ids3[r["id"]] for r in rows]]
    v2_parity = float(np.abs(v2_lg - ref).max())
    v2 = {a: [p["status"] for p in sr.calibrate(v2_lg, sr.load_calibration(a, HF))] for a in (0.01, 0.05)}
    v2_raw = [sr.LABELS[i] for i in v2_lg.argmax(1)]

    # (c) Opus
    opus, opus_calls = opus_verdicts()
    opus_status = []
    for r in rows:
        o = opus.get(r["orig_id"])
        opus_status.append(V2L[o["verdict"]] if o and o["verdict"] else "no_verdict")
    gt_rows_no_opus = [r["id"] for r, s in zip(rows, opus_status) if r["gt"] and s == "no_verdict"]
    circular = sum(1 for r in rows if r["label_source"] == "opus_v1_teacher")

    our_calls = [c for c in opus_calls if c["case_id"] in {r["orig_id"] for r in rows}]
    lat = [c["latency_ms"] for c in our_calls if c["latency_ms"]]
    cost = [c["cost_usd"] for c in our_calls if c["cost_usd"] is not None]
    stock_ms = [recs[r["id"]]["model_ms"] for r in rows]

    systems = {"stock_sapling_published": stock_status, "stock_sapling_raw_argmax": stock_raw,
               "sapling_research_v2_a01": v2[0.01], "sapling_research_v2_a05": v2[0.05],
               "sapling_research_v2_raw_argmax": v2_raw}
    res: dict[str, Any] = {}
    for name, st in systems.items():
        res[name] = {"ground_truth": gt_block(rows, st), "vs_opus_all_rows": opus_agree(rows, st, opus)}
        for sub in ("train_v3", "eval_noise_test"):
            idx = [i for i, r in enumerate(rows) if r["set"] == sub]
            res[name][f"ground_truth_{sub}"] = gt_block([rows[i] for i in idx], [st[i] for i in idx])
    gt_idx = [i for i, r in enumerate(rows) if r["gt"] and opus_status[i] != "no_verdict"]
    res["opus_5_5_recorded"] = {
        "ground_truth": gt_block([rows[i] for i in gt_idx], [opus_status[i] for i in gt_idx]),
        "note": "Opus has no abstain status: its 'unsure' verdict maps to insufficient_information (not established). "
                "Opus vs the 77 teacher labels is circular (100% by construction) and is not reported.",
        "opus_unsure_verdict_rate_on_gt_rows": round(sum(opus_status[i] == "insufficient_information" for i in gt_idx)
                                                       / len(gt_idx), 4),
    }
    for sub in ("train_v3", "eval_noise_test"):
        idx = [i for i in gt_idx if rows[i]["set"] == sub]
        res["opus_5_5_recorded"][f"ground_truth_{sub}"] = gt_block([rows[i] for i in idx], [opus_status[i] for i in idx])

    out = {
        "what": "Stock Banyan Sapling (published head + calibration) vs sapling-research v2 vs Opus 5.5 on 226 real "
                "judge-format claims (186 train_v3 + 40 eval_noise_test rows from scripts/sapling_v3_rows.py)",
        "command": "python -m uv run --extra ml --with safetensors==0.8.0 python scripts/sapling_compare_published.py",
        "honesty": [
            "These 226 rows TRAINED v3 (186 of them) and v3 failed its gate (R77); v2 never saw any judge-format row, so "
            "for v2 and for stock Sapling all 226 are out-of-distribution held-out data.",
            f"{circular} rows carry labels taught by Opus (label_source opus_v1_teacher); Opus is graded only on the "
            "ground-truth rows, never against those labels.",
            "Ground truth: trap runs / planted leaks = fake; noise-set gains checked by 10+10 seed re-runs = noise or "
            "real. 74 non-trap race/collection runs have no ground truth and are compared with Opus only.",
            "Real gains are only 7 rows (3 + 4): quote counts, not rates.",
        ],
        "rows": {"n": len(rows), "by_set": dict(Counter(r["set"] for r in rows)),
                 "by_ground_truth": dict(Counter(r["gt"] or "none (non-trap run, compared with Opus)" for r in rows)),
                 "label_source": dict(Counter(r["label_source"] for r in rows)),
                 "gt_rows_without_opus_verdict": gt_rows_no_opus},
        "checks": {"stock_head_on_cached_features_vs_h100_package_logits_max_abs": stock_parity,
                   "stock_status_recomputed_with_published_calibration_agrees": sum(
                       a == b for a, b in zip(stock_recalc, stock_status)),
                   "v2_release_loader_vs_final_v2_rescored_logits_max_abs": v2_parity,
                   "stock_calibration": {k: stock_cal[k] for k in ("alpha", "temperature", "q_hat", "domain_quantiles")},
                   "v2_calibration_a01": {k: sr.load_calibration(0.01, HF)[k] for k in ("temperature", "q_hat")},
                   "v2_calibration_a05": {k: sr.load_calibration(0.05, HF)[k] for k in ("temperature", "q_hat")}},
        "results": res,
        "latency_and_cost": {
            "stock_sapling_model_ms_on_these_226_rows_H100": {
                "median": round(statistics.median(stock_ms), 1), "p90": round(float(np.percentile(stock_ms, 90)), 1),
                "min": round(min(stock_ms), 1), "max": round(max(stock_ms), 1),
                "source": "records.jsonl model_ms (R76 session, stock package)"},
            "sapling_research_v2_ms_H100_R76": "median 99.4 ms per claim (94-208 ms) on 16 rows vs stock 95.7 ms on the "
                                               "same rows; ensemble adds ~3.6 ms (R76)",
            "opus_5_5_calls_on_these_rows": {
                "n_calls": len(lat), "p50_ms": round(statistics.median(lat), 1),
                "p90_ms": round(float(np.percentile(lat, 90)), 1),
                "cost_per_call_usd_mean": round(float(np.mean(cost)), 5),
                "cost_per_call_usd_median": round(float(np.median(cost)), 5),
                "source": "latency_ms / cost_usd fields of the recorded calls (prompt 5c1d3cfd2941, valid calls only)"},
            "sapling_cost_per_check_usd": {
                "R33": "~$0.00015 per case (billed extraction $0.64 / 4,350 cases incl. cold start, H100)",
                "steady_state_estimate": round(0.0994 * 3.95 / 3600, 6),
                "steady_state_basis": "0.0994 s (R76 v2 median) x H100 $3.95/h; excludes load/cold start (v2 load 44 s)"},
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compare_published.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    per_row = [{"id": r["id"], "orig_id": r["orig_id"], "set": r["set"], "truth": r["truth"], "gt": r["gt"],
                "label_source": r["label_source"], "stock_status": stock_status[i], "stock_raw": stock_raw[i],
                "v2_a01": v2[0.01][i], "v2_a05": v2[0.05][i], "v2_raw": v2_raw[i], "opus": opus_status[i],
                "opus_votes": (opus.get(r["orig_id"]) or {}).get("votes"),
                "v2_logits": [round(float(x), 5) for x in v2_lg[i]]} for i, r in enumerate(rows)]
    with open(OUT / "per_row.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for p in per_row:
            f.write(json.dumps(p) + "\n")
    print(json.dumps({"checks": out["checks"], "rows": out["rows"], "latency_and_cost": out["latency_and_cost"]}, indent=1))
    for name, v in res.items():
        print(name, json.dumps(v["ground_truth"]))
        if "vs_opus_all_rows" in v:
            print("   vs opus", json.dumps(v["vs_opus_all_rows"]))


if __name__ == "__main__":
    main()
