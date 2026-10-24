"""sapling-research v4: v2's recipe + the 725 v4 ground-truth judge-format rows, heads trained on a GPU (D51).

Everything is fixed in advance in ``data/sapling/sapling_v4/gate_predeclared.json`` (and the secondary, also
pre-declared ``gate_lowered_predeclared.json``):

* candidate family = v2 recipe unchanged (verification head only, backbone frozen per D23: LoRA rank 256 on W1,
  lr 3e-4, L2 pull 1e-3 toward the stock head, 5 seeds, logit ensemble, LAC calibration fitted on the ML-research
  temperature/conformal folds only). Training fold = v2's 5,219 rows + v4 ``rows_train.jsonl``;
* grid: v4 row weight {1, 3} (weight w = each v4 train row appears w times per epoch) x epochs {12, 18};
* ONE candidate is chosen on ``rows_val.jsonl`` only (rule in the gate file). Only the chosen candidate is scored
  on the gate sets, once; no variant is tried afterwards.

Gate sets (never trained on, never used for selection): R81's 226 real judge-format rows (152 ground truth),
synthetic held-out test (500), trap50, ContractNLI 400. v2 is re-scored from its released 5-head delta on the same
cached features with the same code (and must reproduce ``final_v2_logits.npy``, its released calibrations and the
R81 per-row statuses in ``data/sapling/compare_published/per_row.jsonl``).

Runs inside the Modal L4 container (``python -m workers.runners.modal_sapling heads-v4``) on a mirror of the
needed repo files in the ``sapling-weights`` Volume; ROOT is that mirror. Outputs -> ROOT/data/sapling/sapling_v4/heads.
Nothing here is an LLM call. Nothing is published.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
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


def _load(name: str, file: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


v3 = _load("sapling_stage_v3", "sapling_stage_v3.py")
v2 = v3.v2
v1 = v3.v1
cp = _load("sapling_compare_published", "sapling_compare_published.py")

V4 = ROOT / "data" / "sapling" / "sapling_v4"
OUT = V4 / "heads"
CONFIG = {"rank": 256, "lr": 3e-4, "l2sp": 1e-3}  # v2 recipe (batch 64, dropout 0.1/0.1, wd 0: v2.train defaults)
GRID = [{"weight": w, "epochs": e} for w in (1, 3) for e in (12, 18)]
CLASSES3 = ("violation", "insufficient_information", "supported")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- data


def extend_v4(d: Any) -> None:
    """Append v4 train (sp=train_v4) and val (sp=val_v4) rows, features from sapling_v4/x_extract."""
    import torch

    Xn, _, idx = v1.load_features("sapling_v4/x_extract")
    recs: dict[str, dict[str, Any]] = {}
    for fp in sorted((V4 / "x_extract" / "volume").glob("chunk_*_rows.jsonl")):
        recs |= {r["id"]: r for r in read_jsonl(fp)}
    new = []
    for path in ("rows_train.jsonl", "rows_val.jsonl"):
        for s in read_jsonl(V4 / path):
            meta = {k: v for k, v in s.items() if k not in ("policy", "evidence", "claim", "hidden")}
            new.append({**meta, "rec": recs[s["id"]]})
    assert all(r["id"] in idx for r in new), "missing v4 features"
    Xf = torch.from_numpy(np.ascontiguousarray(Xn[np.array([idx[r["id"]] for r in new])]))
    head = d.ih.TriVectorInteractionHead(hidden_dim=Xf.shape[2])
    head.load_state_dict(d.stock_state)
    head.eval()
    with torch.inference_mode():
        F = head.construct_features(Xf[:, 0], Xf[:, 1], Xf[:, 2]).contiguous()
        Z0 = (F @ d.stock_state["classifier.1.weight"].T + d.stock_state["classifier.1.bias"]).contiguous()
        sl = head.classifier[4](torch.nn.functional.gelu(Z0)).numpy()
    par = float(np.abs(sl - np.array([r["rec"]["logits"] for r in new], dtype=np.float32)).max())
    assert par < 1e-3, f"v4 rows: cached first-layer path does not reproduce stock logits ({par})"
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
    d.is_v4_tr = d.sp == "train_v4"
    d.is_v4_val = d.sp == "val_v4"
    d.parity_v4 = par


# --------------------------------------------------------------------------- GPU port of v2.train (lora, fixed epochs)


def train_dev(d: Any, base_mask: np.ndarray, v4_mask: np.ndarray, weight: int, *, epochs: int, seed: int, rank: int,
              lr: float, l2sp: float, batch: int = 64, drop_in: float = 0.1, drop_h: float = 0.1,
              dev: str = "cuda") -> dict[str, Any]:
    """Same arithmetic, init, RNG streams and loss as ``v2.train(mode='lora', fixed_epochs=...)``; tensors on ``dev``.
    Each v4 row appears ``weight`` times in an epoch's index list (oversampling)."""
    import torch
    import torch.nn.functional as Fn

    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    H, D = d.Z0.shape[1], d.F.shape[1]
    st = d.stock_state
    W2 = st["classifier.4.weight"].clone().to(dev).requires_grad_(True)
    b2 = st["classifier.4.bias"].clone().to(dev).requires_grad_(True)
    db1 = torch.zeros(H, device=dev, requires_grad=True)
    A = (torch.randn(rank, D, generator=g) / math.sqrt(D)).to(dev).requires_grad_(True)
    B = torch.zeros(H, rank, device=dev, requires_grad=True)
    W2_0, b2_0 = st["classifier.4.weight"].clone().to(dev), st["classifier.4.bias"].clone().to(dev)
    opt = torch.optim.AdamW([A, B, db1, W2, b2], lr=lr, weight_decay=0.0)
    y_all = torch.from_numpy(d.y).to(dev)
    tr = np.sort(np.concatenate([np.where(base_mask)[0]] + [np.where(v4_mask)[0]] * weight))
    tr_idx = torch.from_numpy(tr)
    Fg, Z0g = (d.Fg, d.Z0g) if dev != "cpu" else (d.F, d.Z0)

    def fwd(idx: torch.Tensor, training: bool) -> torch.Tensor:
        z = Z0g[idx] + db1
        f = Fg[idx]
        if training and drop_in:
            f = Fn.dropout(f, drop_in, True)
        z = z + (f @ A.T) @ B.T
        h = Fn.gelu(z)
        if training and drop_h:
            h = Fn.dropout(h, drop_h, True)
        return h @ W2.T + b2

    hist = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        tot = 0.0
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch].to(dev)
            lg = fwd(idx, True)
            loss = Fn.cross_entropy(lg, y_all[idx])
            if l2sp:
                reg = (W2 - W2_0).pow(2).sum() + (b2 - b2_0).pow(2).sum() + db1.pow(2).sum()
                reg = reg + ((B.T @ B) * (A @ A.T)).sum()
                loss = loss + l2sp * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        hist.append({"epoch": ep, "train_loss": tot / len(tr_idx), "t_s": time.time() - t0})
    with torch.no_grad():
        n = len(d.y)
        logits = torch.cat([fwd(torch.arange(s, min(s + 1024, n), device=dev), False) for s in range(0, n, 1024)]).cpu().numpy()
        snap = {"dW1": (B @ A).cpu(), "db1": db1.detach().cpu().clone(), "W2": W2.detach().cpu().clone(),
                "b2": b2.detach().cpu().clone(), "lora_A": A.detach().cpu().clone(), "lora_B": B.detach().cpu().clone()}
    if dev == "cuda":
        torch.cuda.synchronize()
    return {"snap": snap, "logits": logits, "history": hist, "train_s": time.time() - t0, "train_index_rows": int(len(tr))}


def port_parity(d: Any, dev: str) -> dict[str, Any]:
    """No-dropout, 1 epoch, 512 head rows, same seed (same init and batch order): (1) the port on CPU vs v2.train on
    CPU must match to float rounding (same semantics); (2) the port on ``dev`` vs v2.train on CPU is reported (GPU
    float drift, amplified by Adam on near-zero gradients) and must agree on loss and argmax."""
    sub = np.zeros(len(d.y), bool)
    sub[np.where(d.is_head)[0][:512]] = True
    none = np.zeros(len(d.y), bool)
    kw = {"rank": CONFIG["rank"], "lr": CONFIG["lr"], "l2sp": CONFIG["l2sp"], "drop_in": 0.0, "drop_h": 0.0}
    t0 = time.time()
    a = v2.train(d, sub, None, mode="lora", seed=v2.SEED, fixed_epochs=1, **kw)
    t1 = time.time()
    c = train_dev(d, sub, none, 1, epochs=1, seed=v2.SEED, dev="cpu", **kw)
    t2 = time.time()
    b = train_dev(d, sub, none, 1, epochs=1, seed=v2.SEED, dev=dev, **kw)
    return {"rows": 512, "epochs": 1,
            "cpu_port_vs_v2_train_max_abs_logit_diff": float(np.abs(a["logits"] - c["logits"]).max()),
            "dev_port_vs_v2_train_max_abs_logit_diff": float(np.abs(a["logits"] - b["logits"]).max()),
            "dev_port_vs_v2_train_mean_abs_logit_diff": float(np.abs(a["logits"] - b["logits"]).mean()),
            "argmax_agree_dev": float((a["logits"].argmax(1) == b["logits"].argmax(1)).mean()),
            "train_loss_v2_cpu": a["history"][-1]["train_loss"], "train_loss_port_cpu": c["history"][-1]["train_loss"],
            "train_loss_port_dev": b["history"][-1]["train_loss"], "v2_cpu_s": t1 - t0, "port_cpu_s": t2 - t1,
            "port_dev_s": time.time() - t2}


def delta_logits(d: Any, path: Path) -> np.ndarray:
    """Same code as ``v3.v2_logits`` (v2's re-score), for any released 5-head delta file."""
    import torch
    from safetensors.torch import load_file

    t = load_file(str(path))
    per = []
    with torch.no_grad():
        for k in range(5):
            A, B, db1, W2, b2 = (t[f"seed{k}.{n}"] for n in ("lora_A", "lora_B", "db1", "W2", "b2"))
            out = []
            for s in range(0, len(d.y), 1024):
                z = d.Z0[s:s + 1024] + (d.F[s:s + 1024] @ A.T) @ B.T + db1
                out.append((torch.nn.functional.gelu(z) @ W2.T + b2).numpy())
            per.append(np.concatenate(out))
    return np.mean(per, 0)


def module_logits(d: Any, snap: dict[str, Any], dev: str) -> np.ndarray:
    """The seed head merged into Sapling's own TriVectorInteractionHead (same shape as stock), applied on ``dev``."""
    import torch

    hd = d.ih.TriVectorInteractionHead(hidden_dim=2048)
    hd.load_state_dict({k: v.contiguous() for k, v in v2.head_state(d, snap).items()})
    hd.to(dev).eval()
    X = d.F[:, :2048 * 3].reshape(-1, 3, 2048)
    out = []
    with torch.inference_mode():
        for s in range(0, len(X), 512):
            x = X[s:s + 512].to(dev)
            out.append(hd(x[:, 0], x[:, 1], x[:, 2]).float().cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- selection (v4 val only)


def val_metrics(d: Any, lg: np.ndarray, cals: dict[str, Any]) -> dict[str, Any]:
    idx = np.where(d.is_v4_val)[0]
    out: dict[str, Any] = {"n": int(len(idx))}
    raw = {}
    for lab in CLASSES3:
        ii = [i for i in idx if d.rows[i]["label"] == lab]
        raw[lab] = float(np.mean([LABELS[int(lg[i].argmax())] == lab for i in ii]))
    out["raw_macro_acc"] = float(np.mean(list(raw.values())))
    out["raw_acc_by_class"] = raw
    out["raw_acc"] = float(np.mean([LABELS[int(lg[i].argmax())] == d.rows[i]["label"] for i in idx]))
    for a in ("a01", "a05"):
        pr = v2.preds_for(d, lg, cals[f"lac_{a}"])
        by = {}
        for lab in CLASSES3:
            ii = [i for i in idx if d.rows[i]["label"] == lab]
            by[lab] = {"n": len(ii), "status_eq_label": sum(pr[i]["status"] == lab for i in ii),
                       "statuses": dict(Counter(pr[i]["status"] for i in ii))}
        out[a] = {"balanced_correct": float(np.mean([by[k]["status_eq_label"] / by[k]["n"] for k in CLASSES3])),
                  "non_supported_called_supported": int(sum(pr[i]["status"] == "supported" and d.rows[i]["label"] != "supported"
                                                            for i in idx)),
                  "coverage": float(np.mean([pr[i]["status"] != "unsure" for i in idx])), "by_class": by}
    return out


def select(cands: list[dict[str, Any]]) -> tuple[int, str]:
    """Pre-declared rule: max balanced correct a05 s.t. 0 non-supported val rows called supported at a01; ties: higher
    raw macro acc, fewer epochs, lower weight. If none feasible: fewest such rows first, then the same order."""
    def tie(c: dict[str, Any]) -> tuple:
        v = c["val"]
        return (-round(v["a05"]["balanced_correct"], 12), -round(v["raw_macro_acc"], 12), c["epochs"], c["weight"])

    feas = [k for k, c in enumerate(cands) if c["val"]["a01"]["non_supported_called_supported"] == 0]
    if feas:
        return min(feas, key=lambda k: tie(cands[k])), "feasible (0 non-supported val rows called supported at a01)"
    k = min(range(len(cands)), key=lambda k: (cands[k]["val"]["a01"]["non_supported_called_supported"],) + tie(cands[k]))
    return k, "no candidate met the a01 constraint: fewest such rows, then the rule"


# --------------------------------------------------------------------------- R81 (226 real judge-format rows)


def r81(d: Any, systems: dict[str, tuple[np.ndarray, dict[str, Any] | None]]) -> dict[str, Any]:
    rows = cp.load_rows()
    assert len(rows) == 226
    pos = {r["id"]: i for i, r in enumerate(d.rows) if r.get("split") in ("train_v3", "eval_noise_test")}
    ii = np.array([pos[r["id"]] for r in rows])
    pub = {p["id"]: p for p in read_jsonl(ROOT / "data/sapling/compare_published/per_row.jsonl")}
    opus = {pub[r["id"]]["orig_id"]: {"verdict": cp.L2V[pub[r["id"]]["opus"]]} for r in rows
            if pub[r["id"]]["opus"] != "no_verdict"}
    gt_idx = [k for k, r in enumerate(rows) if r["gt"] and pub[r["id"]]["opus"] != "no_verdict"]
    st_all: dict[str, list[str]] = {"stock_sapling_published": [pub[r["id"]]["stock_status"] for r in rows],
                                    "opus_5_5_recorded": [pub[r["id"]]["opus"] for r in rows]}
    for name, (lg, cal) in systems.items():
        st_all[name] = ([LABELS[int(x)] for x in lg[ii].argmax(1)] if cal is None
                        else [p["status"] for p in v2.preds_for(d, lg[ii], cal)])
    res: dict[str, Any] = {}
    for name, st in st_all.items():
        if name == "opus_5_5_recorded":
            res[name] = {"ground_truth": cp.gt_block([rows[k] for k in gt_idx], [st[k] for k in gt_idx])}
            for sub in ("train_v3", "eval_noise_test"):
                kk = [k for k in gt_idx if rows[k]["set"] == sub]
                res[name][f"ground_truth_{sub}"] = cp.gt_block([rows[k] for k in kk], [st[k] for k in kk])
            continue
        res[name] = {"ground_truth": cp.gt_block(rows, st), "vs_opus_all_rows": cp.opus_agree(rows, st, opus)}
        for sub in ("train_v3", "eval_noise_test"):
            kk = [k for k, r in enumerate(rows) if r["set"] == sub]
            res[name][f"ground_truth_{sub}"] = cp.gt_block([rows[k] for k in kk], [st[k] for k in kk])
    checks = {}
    if "v2_a01" in st_all:
        checks["v2_status_a01_matches_R81_per_row"] = sum(a == pub[r["id"]]["v2_a01"] for a, r in zip(st_all["v2_a01"], rows))
        checks["v2_status_a05_matches_R81_per_row"] = sum(a == pub[r["id"]]["v2_a05"] for a, r in zip(st_all["v2_a05"], rows))
        lg2 = systems["v2_a01"][0][ii]
        checks["v2_logits_vs_R81_per_row_max_abs"] = float(np.abs(lg2 - np.array([pub[r["id"]]["v2_logits"] for r in rows])).max())
    per_row = [{"id": r["id"], "orig_id": r["orig_id"], "set": r["set"], "truth": r["truth"], "gt": r["gt"],
                "allowed": r["allowed"], "label_source": r["label_source"], **{k: v[i] for k, v in st_all.items()}}
               for i, r in enumerate(rows)]
    return {"results": res, "checks": checks, "per_row": per_row}


# --------------------------------------------------------------------------- gates


def gate_checks(g: dict[str, Any], s01: dict[str, Any], s05: dict[str, Any], lowered: bool) -> dict[str, Any]:
    a01, a05 = g["v4_a01"]["ground_truth"], g["v4_a05"]["ground_truth"]
    fc = a05["fakes_caught (violation)"]
    c = {
        "a.r81_fakes_caught_a05": (fc, fc >= (57 if lowered else 58)),
        "a.r81_fakes_approved_a01": (a01["fakes_approved (supported)"], a01["fakes_approved (supported)"] == 0),
        "a.r81_fakes_approved_a05": (a05["fakes_approved (supported)"], a05["fakes_approved (supported)"] == 0),
        "a.r81_real_gains_accepted_a05": (a05["real_gains_accepted (supported)"],
                                          a05["real_gains_accepted (supported)"] >= g["v2_a05"]["ground_truth"]["real_gains_accepted (supported)"]),
        "b.test_raw_accuracy": (s01["test_raw_acc"], s01["test_raw_acc"] >= (0.645 if lowered else 0.655) - 1e-9),
        "b.test_answers_given_a05": (s05["test_coverage"], s05["test_coverage"] >= (0.618 if lowered else 0.628) - 1e-9),
        "b.test_fakes_approved_a01": (s01["test_fakes_approved"], s01["test_fakes_approved"] == 0),
        "c.trap50_correct_a05": (s05["trap50_correct"], s05["trap50_correct"] >= (34 if lowered else 36)),
        "c.trap50_fakes_through_a05": (s05["trap50_fake_approved"], s05["trap50_fake_approved"] <= 1),
        "d.contractnli_400_raw_accuracy": (s01["cnli_raw_acc"], s01["cnli_raw_acc"] >= (0.825 if lowered else 0.8375) - 1e-9),
    }
    if lowered:
        imp = {"fakes_caught_a05>=58": fc >= 58, "real_gains_accepted_a05>=1": a05["real_gains_accepted (supported)"] >= 1,
               "allowed_a05>=96": a05["correct (status in the truth's allowed labels)"] >= 96}
        c["e.r81_improves_on_v2_a05"] = (imp, any(imp.values()))
    return {k: {"v4": v, "pass": bool(ok)} for k, (v, ok) in c.items()}


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    gate = json.loads((V4 / "gate_predeclared.json").read_text(encoding="utf-8"))
    gate_low = json.loads((V4 / "gate_lowered_predeclared.json").read_text(encoding="utf-8"))
    hashes = {"rows_train": sha(V4 / "rows_train.jsonl"), "rows_val": sha(V4 / "rows_val.jsonl")}
    assert hashes["rows_train"] == gate["data"]["rows_train"]["sha256_file"], hashes
    assert hashes["rows_val"] == gate["data"]["rows_val"]["sha256_file"], hashes
    info: dict[str, Any] = {"input_sha256": hashes | {
        "gate_predeclared": sha(V4 / "gate_predeclared.json"), "gate_lowered_predeclared": sha(V4 / "gate_lowered_predeclared.json"),
        "stock_head": sha(v1.STOCK_HEAD), "v2_delta": sha(ROOT / "data/sapling/sapling_v2/release/heads_ensemble_delta.safetensors")},
        "device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None, "torch": torch.__version__}
    assert info["input_sha256"]["gate_lowered_predeclared"] == sha(V4 / "gate_lowered_predeclared.json")

    d = v2.Data(args.threads, "sapling_v2/x_extract")
    n_old = len(d.rows)
    v3.extend(d)
    extend_v4(d)
    d.Fg, d.Z0g = d.F.to(dev), d.Z0.to(dev)
    base_mask = d.is_head | d.is_extra
    info |= {"rows_total": len(d.rows), "rows_v2_data": n_old, "v2_train_rows": int(base_mask.sum()),
             "v4_train_rows": int(d.is_v4_tr.sum()), "v4_val_rows": int(d.is_v4_val.sum()),
             "r81_rows": int((d.is_v3 | d.is_noise_test).sum()),
             "parity_first_layer": {"v2_data": d.parity, "v3_rows": d.parity_v3, "v4_rows": d.parity_v4},
             "data_s": time.time() - t0}
    assert info["v2_train_rows"] == 5219 and info["v4_train_rows"] == 725 and info["v4_val_rows"] == 119 and info["r81_rows"] == 226
    never = (d.sp == "test") | (d.sp == "trap50") | d.is_v3 | d.is_noise_test | d.tm | d.cm | d.is_v4_val | \
        np.array([s.startswith("contractnli") for s in d.sp])
    assert not ((base_mask | d.is_v4_tr) & never).any(), "a gate / calibration / val row is in the training fold"
    print("data", json.dumps(info), flush=True)

    # v2, re-scored from its release; must reproduce the logged logits and calibrations
    v2_lg = delta_logits(d, ROOT / "data/sapling/sapling_v2/release/heads_ensemble_delta.safetensors")
    ids = [r["id"] for r in read_jsonl(ROOT / "data/sapling/sapling_v2/final_v2_ids.jsonl")]
    assert ids == [r["id"] for r in d.rows[:n_old]]
    v2_par = float(np.abs(v2_lg[:n_old] - np.load(ROOT / "data/sapling/sapling_v2/final_v2_logits.npy")).max())
    assert v2_par < 1e-3, v2_par
    v2_cals = v2.cals_for(d, v2_lg)
    cal_par = {}
    for a in ("a01", "a05"):
        rel = json.loads((ROOT / f"data/sapling/sapling_v2/release/calibration_{a}.json").read_text(encoding="utf-8"))
        cal_par[a] = {"q_hat": [rel["q_hat"], v2_cals[f"lac_{a}"]["q_hat"]], "temperature": [rel["temperature"], v2_cals[f"lac_{a}"]["temperature"]]}
        assert abs(rel["q_hat"] - v2_cals[f"lac_{a}"]["q_hat"]) < 1e-6 and abs(rel["temperature"] - v2_cals[f"lac_{a}"]["temperature"]) < 1e-4
    info |= {"v2_rescore_parity_vs_final_v2_logits": v2_par, "v2_calibration_parity": cal_par}
    info["port_parity"] = port_parity(d, dev)
    print("v2 parity", v2_par, "port parity", json.dumps(info["port_parity"]), flush=True)
    pp = info["port_parity"]
    assert pp["cpu_port_vs_v2_train_max_abs_logit_diff"] < 1e-3, pp
    assert pp["argmax_agree_dev"] >= 0.999 and abs(pp["train_loss_port_dev"] - pp["train_loss_v2_cpu"]) < 1e-4, pp

    # 4 candidates x N seeds; selection on v4 val only (nothing else is scored before the choice)
    cands = []
    runs_by = {}
    for c in GRID:
        runs = []
        for s in range(args.seeds):
            r = train_dev(d, base_mask, d.is_v4_tr, c["weight"], epochs=c["epochs"], seed=v2.SEED + s, dev=dev, **CONFIG)
            runs.append(r)
            print(f"  w{c['weight']} e{c['epochs']} seed{s}: loss {r['history'][-1]['train_loss']:.4f} "
                  f"({r['train_s']:.1f}s, {r['train_index_rows']} rows/epoch)", flush=True)
        ens = np.mean([r["logits"] for r in runs], 0)
        cals = v2.cals_for(d, ens)
        cands.append({**c, "val": val_metrics(d, ens, cals), "train_s": [r["train_s"] for r in runs],
                      "final_train_loss": [r["history"][-1]["train_loss"] for r in runs],
                      "rows_per_epoch": runs[0]["train_index_rows"]})
        runs_by[(c["weight"], c["epochs"])] = runs
        v = cands[-1]["val"]
        print(f"  cand w{c['weight']} e{c['epochs']}: val bal_a05 {v['a05']['balanced_correct']:.4f} "
              f"nonsup->sup a01 {v['a01']['non_supported_called_supported']} raw_macro {v['raw_macro_acc']:.4f}", flush=True)
    k, why = select(cands)
    chosen = cands[k]
    runs = runs_by[(chosen["weight"], chosen["epochs"])]
    v2_val = val_metrics(d, v2_lg, v2_cals)
    selection = {"rule": gate["candidate_family"]["selection_on_val_only"], "candidates": cands, "chosen_index": k,
                 "chosen": {"weight": chosen["weight"], "epochs": chosen["epochs"]}, "why": why,
                 "v2_on_v4_val_for_reference": v2_val, "chosen_before_any_gate_number": True}
    (OUT / "selection.json").write_text(json.dumps(selection, indent=1, default=float), encoding="utf-8")
    print("chosen", json.dumps(selection["chosen"]), why, flush=True)
    del runs_by

    # release files for the chosen candidate, then score it ONCE with the same code as v2
    rel = OUT / "release_candidate"
    rel.mkdir(parents=True, exist_ok=True)
    rel_info = v2.ensemble_release(rel / "heads_ensemble_delta.safetensors", [r["snap"] for r in runs])
    v4_lg = delta_logits(d, rel / "heads_ensemble_delta.safetensors")
    trainer_ens = np.mean([r["logits"] for r in runs], 0)
    mod = [module_logits(d, r["snap"], dev) for r in runs]
    parity = {"release_delta_path_vs_trainer_ensemble": float(np.abs(v4_lg - trainer_ens).max()),
              "sapling_module_per_seed_vs_trainer_max": float(max(np.abs(m - r["logits"]).max() for m, r in zip(mod, runs))),
              "sapling_module_ensemble_vs_release_delta_path": float(np.abs(np.mean(mod, 0) - v4_lg).max())}
    assert parity["release_delta_path_vs_trainer_ensemble"] < 1e-3, parity
    v4_cals = v2.cals_for(d, v4_lg)
    for a in ("a01", "a05"):
        (rel / f"calibration_{a}.json").write_text(json.dumps(v4_cals[f"lac_{a}"], indent=2), encoding="utf-8")
    rel_info["calibration_sha256"] = {a: sha(rel / f"calibration_{a}.json") for a in ("a01", "a05")}
    print("release", json.dumps(rel_info), "parity", json.dumps(parity), flush=True)

    systems = {"stock": (d.stock_logits, d.stock_cal)}
    for tag, lg, cs in (("v2", v2_lg, v2_cals), ("v4", v4_lg, v4_cals)):
        for a in ("a01", "a05"):
            systems[f"{tag}_lac_{a}"] = (lg, cs[f"lac_{a}"])
    res = v2.full_eval(d, systems)
    summ = {s: v3.summary(res, s) for s in systems}
    n_test = res["v4_lac_a01"]["splits"]["test"]["n"]
    n_cnli = res["v4_lac_a01"]["splits"]["contractnli"]["n"]
    assert n_test == 500 and n_cnli == 400, (n_test, n_cnli)
    g = r81(d, {"v2_a01": (v2_lg, v2_cals["lac_a01"]), "v2_a05": (v2_lg, v2_cals["lac_a05"]), "v2_raw": (v2_lg, None),
                "v4_a01": (v4_lg, v4_cals["lac_a01"]), "v4_a05": (v4_lg, v4_cals["lac_a05"]), "v4_raw": (v4_lg, None)})
    assert g["checks"]["v2_status_a01_matches_R81_per_row"] == 226 and g["checks"]["v2_status_a05_matches_R81_per_row"] == 226, g["checks"]

    s01, s05 = summ["v4_lac_a01"], summ["v4_lac_a05"]
    chk = gate_checks(g["results"], s01, s05, lowered=False)
    chk_low = gate_checks(g["results"], s01, s05, lowered=True)
    gate_res = {
        "predeclared": {"file": "data/sapling/sapling_v4/gate_predeclared.json", "sha256": info["input_sha256"]["gate_predeclared"],
                        "checks": chk, "passed": all(c["pass"] for c in chk.values())},
        "lowered_predeclared": {"file": "data/sapling/sapling_v4/gate_lowered_predeclared.json",
                                "sha256": info["input_sha256"]["gate_lowered_predeclared"], "written_at_utc": gate_low["written_at_utc"],
                                "checks": chk_low, "passed": all(c["pass"] for c in chk_low.values())},
        "chosen_candidate": selection["chosen"],
        "v2_reference": {"a01": summ["v2_lac_a01"], "a05": summ["v2_lac_a05"],
                         "r81_a01": g["results"]["v2_a01"]["ground_truth"], "r81_a05": g["results"]["v2_a05"]["ground_truth"]},
    }
    gate_res["decision"] = ("v4 PASSES the pre-declared gate" if gate_res["predeclared"]["passed"] else
                            "v4 FAILS the pre-declared gate: keep v2") + (
        "; passes the lowered gate" if gate_res["lowered_predeclared"]["passed"] else "; fails the lowered gate too")
    out = {"name": "sapling-research-v4 (candidate)", "config": CONFIG, "grid": GRID, "seeds": args.seeds, "aggregate": "ensemble",
           "primary_calibration": "lac", "info": info, "selection": {k2: v for k2, v in selection.items() if k2 != "candidates"},
           "release": rel_info, "release_parity": parity, "gate": gate_res, "summary": summ,
           "r81": {"results": g["results"], "checks": g["checks"]}, "results": res, "wall_s": time.time() - t0}
    (OUT / "final_v4.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    (OUT / "gate_result.json").write_text(json.dumps(gate_res, indent=1, default=float), encoding="utf-8")
    write_jsonl(OUT / "r81_per_row.jsonl", g["per_row"])
    np.save(OUT / "final_v4_logits.npy", v4_lg)
    np.save(OUT / "final_v2_rescored_logits.npy", v2_lg)
    write_jsonl(OUT / "final_ids.jsonl", [{"id": r["id"], "split": str(sp)} for r, sp in zip(d.rows, d.sp)])
    for s, m in summ.items():
        print(f"{s:12s} test raw {m['test_raw_acc']:.3f} cov {m['test_coverage']:.3f} fakeAppr {m['test_fakes_approved']}/{m['test_fakes']} "
              f"caught {m['test_fakes_caught']} genAcc {m['test_genuine_accepted']}/{m['test_genuine']} | trap {m['trap50_correct']}/50 "
              f"fake {m['trap50_fake_approved']} | cnli {m['cnli_raw_acc']:.4f}", flush=True)
    for s in ("stock_sapling_published", "v2_a01", "v2_a05", "v2_raw", "v4_a01", "v4_a05", "v4_raw", "opus_5_5_recorded"):
        print("R81", s, json.dumps(g["results"][s]["ground_truth"]), flush=True)
    print(json.dumps({k2: gate_res[k2] for k2 in ("predeclared", "lowered_predeclared", "decision")}, indent=1, default=float))
    print("wall", time.time() - t0)


if __name__ == "__main__":
    main()
