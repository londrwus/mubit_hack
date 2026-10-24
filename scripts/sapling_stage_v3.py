"""sapling-research v3: v2's recipe + the 186 judge-format rows from the v3 H100 session ($0 GPU, CPU only).

The candidate is fixed in advance in ``data/sapling/sapling_v3/gate_predeclared.json``: v2's recipe unchanged
(LoRA rank 256 on W1, lr 3e-4, L2 pull 1e-3 toward the stock head, 18 fixed epochs, 5 seeds, logit ensemble,
LAC primary calibration on the ML-research temperature/conformal folds only), trained on v2's 5,219 rows plus
the 186 v3 rows (``rows_train.jsonl``, features in ``x_extract/volume``). Test (500), trap50 and ContractNLI are
evaluation only; the 40 noise-set test rows are evaluation only.

v2 is re-scored from its released 5-head delta on the same cached features (and must reproduce
``final_v2_logits.npy`` and the released calibrations), so both sides are scored by the same code.

    python -m uv run --extra ml --with safetensors==0.8.0 python scripts/sapling_stage_v3.py [--seeds 5] [--threads 8]

Nothing here is an LLM call or a GPU job. Nothing is published.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workers.verifier.sapling import LABELS, read_jsonl, write_jsonl  # noqa: E402

spec = importlib.util.spec_from_file_location("sapling_stage_v2", ROOT / "scripts" / "sapling_stage_v2.py")
v2 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v2)
v1 = v2.v1

V2_DIR = ROOT / "data" / "sapling" / "sapling_v2"
OUT = ROOT / "data" / "sapling" / "sapling_v3"
CONFIG = {"mode": "lora", "rank": 256, "lr": 3e-4, "l2sp": 1e-3}
EPOCHS = 18


def extend(d: Any) -> None:
    """Append the v3 train rows (sp=train_v3) and the noise-set test rows (sp=eval_noise_test) to a v2 Data."""
    import torch

    Xn, _, idx = v1.load_features("sapling_v3/x_extract")
    recs: dict[str, dict[str, Any]] = {}
    for fp in sorted((OUT / "x_extract" / "volume").glob("chunk_*_rows.jsonl")):
        recs |= {r["id"]: r for r in read_jsonl(fp)}
    new = []
    for path in ("rows_train.jsonl", "rows_eval_noise_test.jsonl"):
        for s in read_jsonl(OUT / path):
            meta = {k: v for k, v in s.items() if k not in ("policy", "evidence", "claim", "hidden")}
            if meta.get("label") is None:  # noise-set real gains: either label is right; scored by ``allowed``
                meta["label"] = "insufficient_information"
                meta["label_alt"] = [a for a in meta["allowed"] if a != "insufficient_information"]
            new.append({**meta, "rec": recs[s["id"]]})
    assert all(r["id"] in idx for r in new), "missing v3 features"
    Xf = torch.from_numpy(np.ascontiguousarray(Xn[np.array([idx[r["id"]] for r in new])]))
    head = d.ih.TriVectorInteractionHead(hidden_dim=Xf.shape[2])
    head.load_state_dict(d.stock_state)
    head.eval()
    with torch.inference_mode():
        F = head.construct_features(Xf[:, 0], Xf[:, 1], Xf[:, 2]).contiguous()
        Z0 = (F @ d.stock_state["classifier.1.weight"].T + d.stock_state["classifier.1.bias"]).contiguous()
        sl = head.classifier[4](torch.nn.functional.gelu(Z0)).numpy()
    par = float(np.abs(sl - np.array([r["rec"]["logits"] for r in new], dtype=np.float32)).max())
    assert par < 1e-3, f"v3 rows: cached first-layer path does not reproduce stock logits ({par})"
    n = len(new)
    f = np.zeros(n, bool)
    d.rows = d.rows + new
    d.F = torch.cat([d.F, F]).contiguous()
    d.Z0 = torch.cat([d.Z0, Z0]).contiguous()
    d.stock_logits = np.concatenate([d.stock_logits, sl])
    d.y = np.concatenate([d.y, [v2.LABEL_INDEX[r["label"]] for r in new]])
    d.sp = np.concatenate([d.sp, [r["split"] for r in new]])
    d.v1_logits = np.concatenate([d.v1_logits, np.zeros((n, 4), np.float32)])
    d.groups = np.concatenate([d.groups, [r["group"] for r in new]])
    for a in ("is_val", "is_tr", "is_head", "tm", "cm", "is_extra", "extra_tr"):
        setattr(d, a, np.concatenate([getattr(d, a), f]))
    d.is_v3 = d.sp == "train_v3"
    d.is_noise_test = d.sp == "eval_noise_test"
    d.parity_v3 = par


def v2_logits(d: Any) -> tuple[np.ndarray, list[np.ndarray]]:
    """Re-score v2 from its released 5-head delta: head k = stock with W1 + B@A, b1 + db1, W2, b2; mean of logits."""
    import torch
    from safetensors.torch import load_file

    t = load_file(str(V2_DIR / "release" / "heads_ensemble_delta.safetensors"))
    per = []
    with torch.no_grad():
        for k in range(5):
            A, B, db1, W2, b2 = (t[f"seed{k}.{n}"] for n in ("lora_A", "lora_B", "db1", "W2", "b2"))
            out = []
            for s in range(0, len(d.y), 1024):
                z = d.Z0[s:s + 1024] + (d.F[s:s + 1024] @ A.T) @ B.T + db1
                out.append((torch.nn.functional.gelu(z) @ W2.T + b2).numpy())
            per.append(np.concatenate(out))
    return np.mean(per, 0), per


def noise_eval(d: Any, lg: np.ndarray, cal: dict[str, Any]) -> dict[str, Any]:
    """40 noise-set test rows: correct iff the status (or raw label) is in the row's ``allowed`` set."""
    pr = v2.preds_for(d, lg, cal)
    idx = np.where(d.is_noise_test)[0]
    by: dict[str, Counter] = {}
    tot = Counter()
    for i in idx:
        r, p = d.rows[i], pr[i]
        c = by.setdefault(r["truth"], Counter())
        for cc in (c, tot):
            cc["n"] += 1
            cc["raw_ok"] += p["raw_label"] in r["allowed"]
            cc["answered"] += p["status"] != "unsure"
            cc["status_ok"] += p["status"] in r["allowed"]
            cc["supported"] += p["status"] == "supported"
            cc[f"status:{p['status']}"] += 1
    return {"all": dict(tot), "by_truth": {k: dict(v) for k, v in sorted(by.items())},
            "fake_or_noise_approved": int(sum(pr[i]["status"] == "supported" for i in idx if d.rows[i]["fake_improvement"]))}


def summary(res: dict[str, Any], sname: str) -> dict[str, Any]:
    sp = res[sname]["splits"]
    t, imp = sp["test"], sp["test"]["improvement"]
    tg = sp["trap50"]["trap_grading"]["all"]
    return {"test_raw_acc": t["raw_accuracy"], "test_coverage": t["coverage"], "test_answered_acc": t["answered_accuracy"],
            "test_fakes_approved": imp["fake_approved"], "test_fakes": imp["fakes"], "test_fakes_caught": imp["fake_caught"],
            "test_genuine_accepted": imp["genuine_accepted"], "test_genuine": imp["genuine"],
            "test_genuine_rejected": imp["genuine_rejected"], "test_fake_approved_raw": imp["fake_approved_raw"],
            "test_genuine_rejected_raw": imp["genuine_rejected_raw"],
            "unseen_targets_raw_acc": sp["test_unseen_targets"]["raw_accuracy"],
            "unseen_targets_coverage": sp["test_unseen_targets"]["coverage"],
            "trap50_correct": tg["correct"], "trap50_fake_approved": tg["fake_approved"], "trap50_unsure": tg["unsure"],
            "trap50_genuine_rejected": tg["genuine_rejected"],
            "cnli_raw_acc": sp["contractnli"]["raw_accuracy"], "cnli_heldout266_raw_acc": sp["contractnli_heldout"]["raw_accuracy"],
            "cnli_false_supported_raw": sp["contractnli"]["false_supported_raw"],
            "cnli_coverage": sp["contractnli"]["coverage"]}


def main(argv: list[str] | None = None) -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out-name", default="final_v3.json")
    ap.add_argument("--smoke", action="store_true", help="1 seed, 1 epoch, writes nothing, prints no metrics (plumbing check)")
    args = ap.parse_args(argv)
    epochs = 1 if args.smoke else EPOCHS
    if args.smoke:
        args.seeds = 1
    gate_decl = json.loads((OUT / "gate_predeclared.json").read_text(encoding="utf-8"))
    t0 = time.time()
    d = v2.Data(args.threads, "sapling_v2/x_extract")
    n_old = len(d.rows)
    extend(d)
    print(f"data: {n_old} v2 rows + {int(d.is_v3.sum())} v3 train + {int(d.is_noise_test.sum())} noise-test "
          f"(parity v2 {d.parity:.2e}, v3 {d.parity_v3:.2e}) {time.time() - t0:.0f}s", flush=True)

    # v2 baseline, re-scored; must reproduce the logged logits and calibrations
    v2_lg, _ = v2_logits(d)
    ids = [r["id"] for r in read_jsonl(V2_DIR / "final_v2_ids.jsonl")]
    ref = np.load(V2_DIR / "final_v2_logits.npy")
    v2_par = float(np.abs(v2_lg[:n_old] - ref).max()) if ids == [r["id"] for r in d.rows[:n_old]] else None
    assert v2_par is not None and v2_par < 1e-3, f"v2 re-score parity {v2_par}"
    v2_cals = v2.cals_for(d, v2_lg)
    cal_par = {}
    for a in ("a01", "a05"):
        rel = json.loads((V2_DIR / "release" / f"calibration_{a}.json").read_text(encoding="utf-8"))
        cal_par[a] = {"q_hat": [rel.get("q_hat"), v2_cals[f"lac_{a}"].get("q_hat")],
                      "temperature": [rel.get("temperature"), v2_cals[f"lac_{a}"].get("temperature")]}
    print("v2 re-score parity", v2_par, json.dumps(cal_par), flush=True)

    train_mask = d.is_head | d.is_extra | d.is_v3
    assert not (train_mask & ((d.sp == "test") | (d.sp == "trap50") | d.is_noise_test | d.tm | d.cm)).any()
    runs = []
    for s in range(args.seeds):
        r = v2.train(d, train_mask, None, seed=v2.SEED + s, fixed_epochs=epochs, **CONFIG)
        runs.append(r)
        print(f"  v3 seed{s}: {epochs} epochs, train loss {r['history'][-1]['train_loss']:.4f} ({r['train_s']:.0f}s)", flush=True)
    # parity: each seed merged to Sapling's own head module reproduces the trainer's logits
    X = d.F[:, :2048 * 3].reshape(-1, 3, 2048).numpy()
    mod_lgs = []
    for r in runs:
        hd = d.ih.TriVectorInteractionHead(hidden_dim=2048)
        hd.load_state_dict({k: v.contiguous() for k, v in v2.head_state(d, r["snap"]).items()})
        mod_lgs.append(v1._apply(hd, X))
    merged_parity = float(max(np.abs(m - r["logits"]).max() for m, r in zip(mod_lgs, runs)))
    v3_lg = np.mean(mod_lgs, 0)
    v3_cals = v2.cals_for(d, v3_lg)

    systems: dict[str, tuple[np.ndarray, dict[str, Any]]] = {"stock": (d.stock_logits, d.stock_cal)}
    for tag, lg, cs in (("v2", v2_lg, v2_cals), ("v3", v3_lg, v3_cals)):
        for a in ("a01", "a05"):
            for kind in ("lac", "binary", "asym"):
                systems[f"{tag}_{kind}_{a}"] = (lg, cs[f"{kind}_{a}"])
    res = v2.full_eval(d, systems)
    noise = {s: noise_eval(d, lg, cal) for s, (lg, cal) in systems.items()}
    summ = {s: summary(res, s) for s in systems}

    g2a01, g2a05, g3a01, g3a05 = summ["v2_lac_a01"], summ["v2_lac_a05"], summ["v3_lac_a01"], summ["v3_lac_a05"]
    checks = {
        "test_fakes_approved_a01 == 0": (g3a01["test_fakes_approved"], g3a01["test_fakes_approved"] == 0),
        "test_raw_accuracy >= v2 - 0.005": (g3a01["test_raw_acc"], g3a01["test_raw_acc"] >= g2a01["test_raw_acc"] - 0.005 - 1e-9),
        "test_coverage_a05 >= v2 - 0.01": (g3a05["test_coverage"], g3a05["test_coverage"] >= g2a05["test_coverage"] - 0.01 - 1e-9),
        "trap50_correct_a05 >= v2 (36)": (g3a05["trap50_correct"], g3a05["trap50_correct"] >= g2a05["trap50_correct"]),
        "trap50_fake_approved_a05 <= 1": (g3a05["trap50_fake_approved"], g3a05["trap50_fake_approved"] <= 1),
        "contractnli_raw_400 >= v2": (g3a01["cnli_raw_acc"], g3a01["cnli_raw_acc"] >= g2a01["cnli_raw_acc"] - 1e-9),
        "contractnli_raw_heldout266 >= v2": (g3a01["cnli_heldout266_raw_acc"],
                                              g3a01["cnli_heldout266_raw_acc"] >= g2a01["cnli_heldout266_raw_acc"] - 1e-9),
    }
    passed = all(ok for _, ok in checks.values())
    gate = {"predeclared": gate_decl, "checks": {k: {"v3": v, "pass": bool(ok)} for k, (v, ok) in checks.items()},
            "v2_reference": {"a01": g2a01, "a05": g2a05}, "passed": passed,
            "decision": "v3 replaces v2" if passed else "v3 FAILS the gate: keep v2"}

    if args.smoke:
        print("smoke ok: merged parity", merged_parity, "gate keys", list(checks), "noise rows", noise["v3_lac_a01"]["all"]["n"])
        return
    rel = OUT / "release"
    rel.mkdir(parents=True, exist_ok=True)
    rel_info = v2.ensemble_release(rel / "heads_ensemble_delta.safetensors", [r["snap"] for r in runs])
    for a in ("a01", "a05"):
        (rel / f"calibration_{a}.json").write_text(json.dumps(v3_cals[f"lac_{a}"], indent=2), encoding="utf-8")
        for kind in ("binary", "asym"):
            (rel / f"calibration_{kind}_{a}.json").write_text(json.dumps(v3_cals[f"{kind}_{a}"], indent=2), encoding="utf-8")
    out = {"name": "sapling-research-v3", "config": CONFIG, "epochs": EPOCHS, "seeds": args.seeds, "aggregate": "ensemble",
           "primary_calibration": "lac", "train_rows": int(train_mask.sum()), "train_rows_v3": int(d.is_v3.sum()),
           "v2_rescore_parity_vs_final_v2_logits": v2_par, "v2_calibration_parity": cal_par,
           "merged_head_parity_vs_sapling_module": merged_parity, "release": rel_info,
           "train_s_total": float(sum(r["train_s"] for r in runs)), "wall_s": time.time() - t0,
           "gate": gate, "summary": summ, "noise_test_40": noise, "results": res}
    (OUT / args.out_name).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    (OUT / "gate_result.json").write_text(json.dumps(gate, indent=1, default=float), encoding="utf-8")
    np.save(OUT / "final_v3_logits.npy", v3_lg)
    np.save(OUT / "final_v2_rescored_logits.npy", v2_lg)
    write_jsonl(OUT / "final_v3_ids.jsonl", [{"id": r["id"]} for r in d.rows])
    for s, m in summ.items():
        nz = noise[s]["all"]
        print(f"{s:16s} test raw {m['test_raw_acc']:.3f} cov {m['test_coverage']:.3f} ans {m['test_answered_acc']} "
              f"fakeAppr {m['test_fakes_approved']}/{m['test_fakes']} caught {m['test_fakes_caught']} "
              f"genAcc {m['test_genuine_accepted']}/{m['test_genuine']} genRej {m['test_genuine_rejected']} | "
              f"trap {m['trap50_correct']}/50 fake {m['trap50_fake_approved']} | cnli {m['cnli_raw_acc']:.3f} "
              f"h266 {m['cnli_heldout266_raw_acc']:.3f} | noise40 status_ok {nz.get('status_ok', 0)}/{nz['n']} "
              f"answered {nz.get('answered', 0)} approved_fake {noise[s]['fake_or_noise_approved']}", flush=True)
    print(json.dumps({k: v for k, v in gate.items() if k in ("checks", "passed", "decision")}, indent=1, default=float))
    print("merged parity", merged_parity, "release", json.dumps(rel_info))


if __name__ == "__main__":
    main()
