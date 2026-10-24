"""Check the v3 H100 session (``modal_sapling v3-session``) against what is logged. CPU only, no GPU, no LLM.

* v2 package (``hf_release/example.py`` end to end, real backbone): example logits / status vs the expected
  values in example.py, and the held-out check rows vs ``eval/v2_predictions_heldout.jsonl``.
* stock features of the check rows vs the cached v1 H100 pass (``data/sapling/sapling_v1/volume``), stock
  logits vs the cached stock logits, and the 5-head v2 ensemble on the NEW features (CPU) vs the GPU v2 logits.

    python -m uv run --extra ml --with safetensors==0.8.0 python scripts/sapling_v3_gpu_check.py [session_dir]
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workers.verifier.sapling import read_jsonl  # noqa: E402

REL = ROOT / "data" / "sapling" / "sapling_v2" / "hf_release"


def _mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main() -> None:
    from safetensors.torch import load_file

    sess = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "sapling" / "sapling_v3" / "h100_session")
    rep = json.loads((sess / "report.json").read_text(encoding="utf-8"))
    recs = read_jsonl(sess / "records.jsonl")
    feats = np.load(sess / "features.npy")
    sr = _mod("sapling_research_rel", REL / "sapling_research.py")
    ex = (REL / "example.py").read_text(encoding="utf-8")
    exp_logits = json.loads(ex.split("EXPECTED_LOGITS = ")[1].splitlines()[0])
    out: dict = {"session": str(sess), "call_id": rep.get("call_id"), "driver": rep.get("driver"),
                 "reference_stock_vs_h200": rep.get("reference"), "extract": rep.get("extract"),
                 "parity_on_gpu": rep.get("parity")}

    # --- (a) v2 package end to end
    v2 = rep.get("v2_check", {})
    out["v2_example_stdout"] = v2.get("example_stdout")
    out["v2_load_s"] = v2.get("load_s")
    logged = {r["id"]: r for r in read_jsonl(REL / "eval" / "v2_predictions_heldout.jsonl")}
    cal05 = sr.load_calibration(0.05, REL)
    rows = []
    for r in v2.get("rows", []):
        lg = logged[r["id"]]
        c05 = sr.calibrate([r["logits"]], cal05)[0]
        rows.append({"id": r["id"], "max_abs_logit_diff": float(np.max(np.abs(np.array(r["logits"]) - np.array(lg["logits"])))),
                     "label": r["label"], "logged_label": lg["raw_label"], "status_a01": r["status_a01"],
                     "logged_status_a01": lg["status_a01"], "status_a05": c05["status"], "logged_status_a05": lg["status_a05"],
                     "input_tokens": r["input_tokens"], "model_ms": r["model_ms"]})
    if rows:
        ex_row = next((x for x in rows if x["id"] == "mlc_v1_test_00453"), None)
        ex_gpu = next(r for r in v2["rows"] if r["id"] == "mlc_v1_test_00453")
        out["v2_example_row"] = {"logits": ex_gpu["logits"], "expected": exp_logits,
                                 "max_abs_diff_vs_expected": float(np.max(np.abs(np.array(ex_gpu["logits"]) - np.array(exp_logits)))),
                                 "status_a01": ex_gpu["status_a01"], "expected_status": "violation", **(ex_row or {})}
        out["v2_heldout_rows"] = {"n": len(rows), "max_abs_logit_diff": max(x["max_abs_logit_diff"] for x in rows),
                                  "median_max_abs_diff": float(np.median([x["max_abs_logit_diff"] for x in rows])),
                                  "raw_label_agree": sum(x["label"] == x["logged_label"] for x in rows),
                                  "status_a01_agree": sum(x["status_a01"] == x["logged_status_a01"] for x in rows),
                                  "status_a05_agree": sum(x["status_a05"] == x["logged_status_a05"] for x in rows),
                                  "rows": rows}

    # --- (b) stock features of the check rows vs the cached v1 pass
    v1 = _mod("sapling_stage_v1", ROOT / "scripts" / "sapling_stage.py")
    X1, L1, idx1 = v1.load_features("sapling_v1")
    by_id = {r["id"]: r for r in recs if r.get("feature_index") is not None}
    chk = [i for i in (x["id"] for x in rows) if i in by_id and i in idx1] if rows else \
        [r["id"] for r in recs if r["id"].startswith("mlc_v1_") and r["id"] in idx1]
    if chk:
        fn = np.stack([feats[by_id[i]["feature_index"]] for i in chk])
        fo = np.stack([X1[idx1[i]] for i in chk])
        ln = np.array([by_id[i]["logits"] for i in chk])
        lo = np.stack([L1[idx1[i]] for i in chk])
        stock = load_file(str(ROOT / "data" / "sapling" / "stock_head" / "head.safetensors"))
        deltas = sr.load_deltas(REL / sr.DELTA_FILE)
        ens_cpu = sr.ensemble_logits_from_features(fn, stock, deltas)
        gpu_v2 = {r["id"]: r["logits"] for r in v2.get("rows", [])}
        out["stock_features_vs_cached_v1_pass"] = {
            "rows": len(chk), "feature_max_abs_diff": float(np.abs(fn - fo).max()),
            "feature_rel_l2_max": float(max(np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(fn, fo))),
            "stock_logit_max_abs_diff": float(np.abs(ln - lo).max()),
            "stock_raw_label_agree": int(sum(a.argmax() == b.argmax() for a, b in zip(ln, lo)))}
        if gpu_v2:
            g = np.array([gpu_v2[i] for i in chk])
            out["v2_ensemble_cpu_on_new_features_vs_gpu_v2"] = {"rows": len(chk), "max_abs_logit_diff": float(np.abs(ens_cpu - g).max())}
    # --- new rows produced
    new = [r for r in recs if not r["id"].startswith("mlc_v1_")]
    out["new_feature_rows"] = {"n": len(new), "ok": sum(r.get("feature_index") is not None for r in new),
                               "errors": [r for r in new if r.get("feature_index") is None][:5],
                               "input_tokens_max": max((r.get("input_tokens", 0) for r in new), default=0),
                               "input_tokens_median": float(np.median([r["input_tokens"] for r in new if "input_tokens" in r])) if new else None,
                               "stock_status_counts": {s: sum(r.get("status") == s for r in new) for s in
                                                       sorted({r.get("status") for r in new if r.get("status")})}}
    (sess / "check.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "v2_heldout_rows"} |
                     {"v2_heldout_rows": {k: v for k, v in out.get("v2_heldout_rows", {}).items() if k != "rows"}},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
