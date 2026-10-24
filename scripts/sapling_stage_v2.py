"""sapling-research v2: one timeboxed iteration on the v1 cached features (backbone frozen, $0 GPU).

v1 (R33, ``scripts/sapling_stage.py``): raw accuracy 58% on held-out research claims, but only 6-18%
answered and 0-1/132 genuine gains accepted. v2 tries, on the SAME cached H100 features:

  * a template-held-out validation split (train families only: policy ``terse`` or evidence ``table``
    groups) for early stopping and every selection; test / trap50 / ContractNLI are never looked at
    before the configuration is frozen;
  * a fast head trainer: the stock head's first layer output is cached once, the update to W1 is either
    low-rank (LoRA, merged back into a full same-shape head) or full; W2/b2 and b1 are trained fully;
  * class-balanced and cost-sensitive loss (extra penalty on P(supported) for non-supported rows);
  * seed ensembles (logit average) and weight soups (one head, same shape as Sapling's);
  * label-conditional (Mondrian) conformal thresholds next to Sapling's single-threshold LAC.

Subcommands (repo root, PYTHONUTF8=1):

    hosted-parity  30 ml-claims rows -> Mubit's hosted stock Sapling endpoint (D33, 1 request at a time),
                   compared with our H100 stock logits -> data/sapling/sapling_v2/hosted_parity.json
    explore        grid on the template-held-out val split -> data/sapling/sapling_v2/explore.json
    final          refit the selected config on the full head fold (N seeds), calibrate, evaluate stock /
                   v1 / v2 on test, unseen targets, trap50 and ContractNLI -> data/sapling/sapling_v2/final.json

``python -m uv run --extra ml --with safetensors==0.8.0 python scripts/sapling_stage_v2.py <cmd>``.
Nothing here is an LLM call. Nothing is published.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workers.verifier.sapling import LABELS, calibrate, read_jsonl, write_jsonl  # noqa: E402


def _load_v1() -> Any:
    spec = importlib.util.spec_from_file_location("sapling_stage_v1", ROOT / "scripts" / "sapling_stage.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


v1 = _load_v1()
RUN = "sapling_v1"  # cached features of the v1 H100 pass
OUT = ROOT / "data" / "sapling" / "sapling_v2"
HOSTED_URL = "https://6adbee0669a00298cd4774c9.endpoints.huggingface.cloud"
SEED = 20261024
LABEL_INDEX = {lab: i for i, lab in enumerate(LABELS)}
SUP = LABEL_INDEX["supported"]
# Template-held-out validation (train families only; test families qa/memo, log/sections never used).
VAL_POLICY_FAMILIES = ("terse",)
VAL_EVIDENCE_FAMILIES = ("table",)


# --------------------------------------------------------------------------- hosted parity (D33)


def cmd_hosted_parity(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = {r["id"]: r for r in read_jsonl(v1.SAPLING_DIR / RUN / "records.jsonl")}
    src = read_jsonl(ROOT / "data/ml_claims/test.jsonl") + read_jsonl(ROOT / "data/ml_claims/trap50.jsonl")
    by_lab: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in src:
        if r["id"] in recs and recs[r["id"]].get("logits"):
            by_lab[r["label"]].append(r)
    rng = random.Random(SEED)
    want = {"supported": 9, "violation": 9, "insufficient_information": 8, "inconsistent_context": 4}
    picked = [r for lab, n in want.items() for r in rng.sample(by_lab[lab], n)][: args.n]
    out_rows = []
    for k, r in enumerate(picked):
        body = json.dumps({"inputs": {f: r[f] for f in ("policy", "evidence", "claim")}}).encode()
        res, err, t0 = None, None, time.time()
        for attempt in range(4):
            try:
                req = urllib.request.Request(HOSTED_URL, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    res = json.loads(resp.read().decode())
                break
            except (urllib.error.URLError, TimeoutError) as e:  # cold start / 5xx: back off politely
                err = repr(e)
                time.sleep(5 * (attempt + 1))
        wall = time.time() - t0
        ours = recs[r["id"]]
        row = {"id": r["id"], "label": r["label"], "wall_s": wall, "error": None if res else err}
        if res:
            hl, ol = np.array(res["logits"], float), np.array(ours["logits"], float)
            row |= {"hosted_logits": res["logits"], "h100_logits": ours["logits"], "max_abs_diff": float(np.abs(hl - ol).max()),
                    "hosted_status": res["status"], "h100_status": ours["status"], "hosted_raw": res["raw_label"],
                    "h100_raw": ours["raw_label"], "hosted_input_tokens": res["usage"]["input_tokens"],
                    "h100_input_tokens": ours["input_tokens"], "hosted_model_ms": res["timing_ms"]["model"],
                    "h100_model_ms": ours["model_ms"]}
        out_rows.append(row)
        print(k, json.dumps({kk: row.get(kk) for kk in ("id", "max_abs_diff", "hosted_raw", "h100_raw", "hosted_status", "wall_s", "error")}), flush=True)
        time.sleep(args.sleep)
    ok = [r for r in out_rows if not r["error"]]
    d = np.array([r["max_abs_diff"] for r in ok]) if ok else np.zeros(0)
    summ = {"endpoint": HOSTED_URL, "rows_sent": len(out_rows), "rows_ok": len(ok), "concurrency": 1,
            "max_abs_logit_diff": float(d.max()) if ok else None, "median_abs_logit_diff": float(np.median(d)) if ok else None,
            "mean_abs_logit_diff_all": float(np.mean([np.abs(np.array(r["hosted_logits"]) - np.array(r["h100_logits"])).mean() for r in ok])) if ok else None,
            "raw_label_agree": sum(r["hosted_raw"] == r["h100_raw"] for r in ok), "status_agree": sum(r["hosted_status"] == r["h100_status"] for r in ok),
            "input_tokens_agree": sum(r["hosted_input_tokens"] == r["h100_input_tokens"] for r in ok),
            "hosted_model_ms_median": float(np.median([r["hosted_model_ms"] for r in ok])) if ok else None,
            "hosted_wall_s_median": float(np.median([r["wall_s"] for r in ok])) if ok else None,
            "h100_model_ms_median_same_rows": float(np.median([r["h100_model_ms"] for r in ok])) if ok else None,
            "rows": out_rows}
    (OUT / "hosted_parity.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=1))


# --------------------------------------------------------------------------- data


class Data:
    """Cached v1 features joined with labels; first-layer output of the stock head cached once."""

    def __init__(self, threads: int = 8, extra_run: str | None = None) -> None:
        import torch
        from safetensors.torch import load_file

        torch.set_num_threads(threads)
        self.ih = v1._import_pkg("interaction_head")
        self.cc = v1._import_pkg("conformal_calibrator")
        rows = v1.load_labelled(RUN)
        X, _, index = v1.load_features(RUN)
        rows = [r for r in rows if r["id"] in index]
        Xs = [X[np.array([index[r["id"]] for r in rows])]]
        if extra_run:  # extra head-fold rows (scripts/make_ml_claims_extra.py), features from one more H100 pass
            recs = {r["id"]: r for r in read_jsonl(v1.SAPLING_DIR / extra_run / "records.jsonl")}
            Xe, _, ie = v1.load_features(extra_run)
            ex = [{**{k: v for k, v in s.items() if k not in ("policy", "evidence", "claim", "hidden")}, "rec": recs[s["id"]],
                   "split": "train_extra"} for s in read_jsonl(OUT / "extra_train.jsonl") if s["id"] in ie and s["id"] in recs]
            Xs.append(Xe[np.array([ie[r["id"]] for r in ex])])
            rows = rows + ex
        X = np.concatenate(Xs)
        self.rows = rows
        fi = np.arange(len(rows))
        self.y = np.array([LABEL_INDEX[r["label"]] for r in rows])
        self.sp = np.array(["train_extra" if r["split"] == "train_extra" else v1.split_of(r) for r in rows])
        self.stock_state = load_file(str(v1.STOCK_HEAD))
        self.stock_cal = json.loads(v1.STOCK_CAL.read_text(encoding="utf-8"))
        Xf = torch.from_numpy(np.ascontiguousarray(X[fi]))
        head = self.ih.TriVectorInteractionHead(hidden_dim=X.shape[2])
        head.load_state_dict(self.stock_state)
        head.eval()
        with torch.inference_mode():
            self.F = head.construct_features(Xf[:, 0], Xf[:, 1], Xf[:, 2]).contiguous()  # [n, 14336]
            W1, b1 = self.stock_state["classifier.1.weight"], self.stock_state["classifier.1.bias"]
            self.Z0 = (self.F @ W1.T + b1).contiguous()  # [n, 4096]
            self.stock_logits = head.classifier[4](torch.nn.functional.gelu(self.Z0)).numpy()
        rec = np.array([r["rec"]["logits"] for r in rows], dtype=np.float32)
        self.parity = float(np.abs(self.stock_logits - rec).max())
        assert self.parity < 1e-3, f"cached first-layer path does not reproduce stock logits ({self.parity})"
        # v1 head logits (saved by v1 stage2, same row order by id)
        ids = [r["id"] for r in read_jsonl(v1.SAPLING_DIR / RUN / "ours_ids.jsonl")]
        v1l = np.load(v1.SAPLING_DIR / RUN / "ours_logits.npy")
        pos = {i: k for k, i in enumerate(ids)}
        self.v1_logits = np.stack([v1l[pos[r["id"]]] if r["id"] in pos else np.zeros(4, np.float32) for r in rows])
        groups = [r.get("group") or r["id"] for r in rows]
        tmpl = [r.get("templates") or {} for r in rows]
        self.groups = np.array(groups)
        is_head = self.sp == "train_head"
        # a group is val if ANY of its rows uses a held-out family (policy/evidence are shared within a group)
        val_groups = {g for g, t, h in zip(groups, tmpl, is_head) if h and (
            t.get("policy_family") in VAL_POLICY_FAMILIES or t.get("evidence_family") in VAL_EVIDENCE_FAMILIES)}
        self.is_val = np.array([h and g in val_groups for g, h in zip(groups, is_head)])
        self.is_tr = is_head & ~self.is_val
        self.is_head = is_head
        assert not ({groups[i] for i in np.where(self.is_tr)[0]} & val_groups)
        self.tm, self.cm = self.sp == "train_temperature", self.sp == "train_conformal"
        self.is_extra = self.sp == "train_extra"
        # extra rows in a val family never train an explore-stage model (val_tf stays family-held-out)
        self.extra_tr = self.is_extra & np.array([not ((r.get("templates") or {}).get("policy_family") in VAL_POLICY_FAMILIES or
                                                      (r.get("templates") or {}).get("evidence_family") in VAL_EVIDENCE_FAMILIES) for r in rows])

    def mask(self, name: str) -> np.ndarray:
        if name == "test_unseen_targets":
            return (self.sp == "test") & np.array([not r.get("target_seen_in_train", True) for r in self.rows])
        if name == "contractnli":
            return self.sp == "contractnli_ev"
        if name == "contractnli_layout_pol":
            return self.sp == "contractnli_pol"
        if name in ("contractnli_dev", "contractnli_heldout"):
            third = np.array([r.get("case", -1) % 3 if r["id"].startswith("cnli_") else -1 for r in self.rows])
            ev = self.sp == "contractnli_ev"
            return ev & (third == 0) if name == "contractnli_dev" else ev & (third != 0)
        return {"test": self.sp == "test", "trap50": self.sp == "trap50", "val_tf": self.is_val,
                "conformal_fold": self.cm, "temperature_fold": self.tm}[name]


# --------------------------------------------------------------------------- fast head trainer


def train(d: Data, train_mask: np.ndarray, val_mask: np.ndarray | None, *, mode: str = "lora", rank: int = 64,
          lr: float = 3e-4, epochs: int = 30, patience: int = 4, batch: int = 64, drop_in: float = 0.1,
          drop_h: float = 0.1, balanced: bool = False, fake_penalty: float = 0.0, wd: float = 0.0, l2sp: float = 0.0,
          seed: int = SEED, fixed_epochs: int | None = None, verbose: bool = False) -> dict[str, Any]:
    """Warm start = stock head. z = Z0 + dropout(F) @ dW1^T + db1 (dW1 low-rank or full); W2/b2 trained.
    Early stopping on val CE unless ``fixed_epochs``. Returns the best delta (merged dW1) and history."""
    import torch
    import torch.nn.functional as Fn

    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    H, D = d.Z0.shape[1], d.F.shape[1]
    W2 = d.stock_state["classifier.4.weight"].clone().requires_grad_(True)
    b2 = d.stock_state["classifier.4.bias"].clone().requires_grad_(True)
    db1 = torch.zeros(H, requires_grad=True)
    if mode == "lora":
        A = (torch.randn(rank, D, generator=g) / math.sqrt(D)).requires_grad_(True)
        B = torch.zeros(H, rank, requires_grad=True)
        params = [A, B, db1, W2, b2]
    elif mode == "full":
        Dw = torch.zeros(H, D, requires_grad=True)
        params = [Dw, db1, W2, b2]
    elif mode == "last":
        params = [db1, W2, b2]
    else:
        raise ValueError(mode)
    W2_0, b2_0 = d.stock_state["classifier.4.weight"].clone(), d.stock_state["classifier.4.bias"].clone()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    ytr_all = torch.from_numpy(d.y)
    tr_idx = torch.from_numpy(np.where(train_mask)[0])
    cw = None
    if balanced:
        cnt = np.bincount(d.y[train_mask], minlength=4).astype(np.float64)
        cw = torch.tensor(cnt.sum() / (4 * np.maximum(cnt, 1)), dtype=torch.float32)

    def fwd(idx: torch.Tensor, training: bool) -> torch.Tensor:
        z = d.Z0[idx] + db1
        if mode != "last":
            f = d.F[idx]
            if training and drop_in:
                f = Fn.dropout(f, drop_in, True)
            z = z + ((f @ A.T) @ B.T if mode == "lora" else f @ Dw.T)
        h = Fn.gelu(z)
        if training and drop_h:
            h = Fn.dropout(h, drop_h, True)
        return h @ W2.T + b2

    def logits_all() -> np.ndarray:
        with torch.no_grad():
            return torch.cat([fwd(torch.arange(s, min(s + 1024, len(d.y))), False) for s in range(0, len(d.y), 1024)]).numpy()

    def snapshot() -> dict[str, torch.Tensor]:
        with torch.no_grad():
            dw = (B @ A) if mode == "lora" else (Dw.clone() if mode == "full" else None)
            snap = {"dW1": dw, "db1": db1.detach().clone(), "W2": W2.detach().clone(), "b2": b2.detach().clone()}
            if mode == "lora":  # compact form for the release (rank x (D + H) floats per seed)
                snap |= {"lora_A": A.detach().clone(), "lora_B": B.detach().clone()}
            return snap

    hist: list[dict[str, Any]] = []
    best = {"val_ce": float("inf"), "epoch": 0}
    best_snap, best_logits, bad = snapshot(), None, 0
    t0 = time.time()
    n_ep = fixed_epochs if fixed_epochs is not None else epochs
    va = np.where(val_mask)[0] if val_mask is not None else None
    if va is not None and fixed_epochs is None:
        lg0 = logits_all()
        best = {"val_ce": v1._ce(lg0[va], d.y[va]), "epoch": 0}
        best_logits = lg0
    for ep in range(1, n_ep + 1):
        perm = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        tot = 0.0
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            lg = fwd(idx, True)
            yb = ytr_all[idx]
            loss = Fn.cross_entropy(lg, yb, weight=cw)
            if fake_penalty:
                ns = yb != SUP
                if ns.any():
                    p_sup = torch.softmax(lg[ns], 1)[:, SUP]
                    loss = loss + fake_penalty * (-torch.log1p(-p_sup.clamp(max=1 - 1e-6))).mean()
            if l2sp:  # pull every parameter back toward the stock head (squared Frobenius norm of the delta)
                reg = (W2 - W2_0).pow(2).sum() + (b2 - b2_0).pow(2).sum() + db1.pow(2).sum()
                if mode == "lora":
                    reg = reg + ((B.T @ B) * (A @ A.T)).sum()  # ||B A||_F^2 without forming B A
                elif mode == "full":
                    reg = reg + Dw.pow(2).sum()
                loss = loss + l2sp * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        rec: dict[str, Any] = {"epoch": ep, "train_loss": tot / len(tr_idx), "t_s": time.time() - t0}
        if va is not None:
            lg = logits_all()
            rec |= {"val_ce": v1._ce(lg[va], d.y[va]), "val_acc": float((lg[va].argmax(1) == d.y[va]).mean())}
            if fixed_epochs is None:
                if rec["val_ce"] < best["val_ce"] - 1e-4:
                    best, best_snap, best_logits, bad = {"val_ce": rec["val_ce"], "val_acc": rec["val_acc"], "epoch": ep}, snapshot(), lg, 0
                else:
                    bad += 1
        hist.append(rec)
        if verbose:
            print("   ", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
        if fixed_epochs is None and va is not None and bad >= patience:
            break
    if fixed_epochs is not None or va is None:
        best_snap, best_logits = snapshot(), logits_all()
        best = {"epoch": n_ep, **({"val_ce": hist[-1].get("val_ce"), "val_acc": hist[-1].get("val_acc")} if va is not None else {})}
    return {"best": best, "snap": best_snap, "logits": best_logits, "history": hist, "train_s": time.time() - t0}


def ensemble_release(path: Path, snaps: list[dict[str, Any]]) -> dict[str, Any]:
    """K seed heads as compact deltas on the stock head: seed{k}.lora_A [r, 14336], seed{k}.lora_B [4096, r],
    seed{k}.db1, seed{k}.W2, seed{k}.b2. Head k = stock with W1 + B@A, b1 + db1, W2, b2; served logits = mean."""
    from safetensors.torch import save_file

    t = {}
    for k, sn in enumerate(snaps):
        for name in ("lora_A", "lora_B", "db1", "W2", "b2"):
            if name in sn:
                t[f"seed{k}.{name}"] = sn[name].contiguous()
            elif name.startswith("lora"):
                t[f"seed{k}.dW1"] = sn["dW1"].contiguous()
    save_file(t, str(path))
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "heads": len(snaps)}


def head_state(d: Data, snap: dict[str, Any]) -> dict[str, Any]:
    st = {k: v.clone() for k, v in d.stock_state.items()}
    if snap["dW1"] is not None:
        st["classifier.1.weight"] = st["classifier.1.weight"] + snap["dW1"]
    st["classifier.1.bias"] = st["classifier.1.bias"] + snap["db1"]
    st["classifier.4.weight"], st["classifier.4.bias"] = snap["W2"], snap["b2"]
    return st


def soup(snaps: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for k in ("dW1", "db1", "W2", "b2"):
        out[k] = None if snaps[0][k] is None else sum(s[k] for s in snaps) / len(snaps)
    return out


def logits_of_state(d: Data, st: dict[str, Any]) -> np.ndarray:
    import torch

    with torch.no_grad():
        out = []
        for s in range(0, len(d.y), 1024):
            z = d.F[s:s + 1024] @ st["classifier.1.weight"].T + st["classifier.1.bias"]
            out.append((torch.nn.functional.gelu(z) @ st["classifier.4.weight"].T + st["classifier.4.bias"]).numpy())
        return np.concatenate(out)


# --------------------------------------------------------------------------- calibration


def lac(d: Data, lg: np.ndarray, alpha: float) -> dict[str, Any]:
    return v1.fit_calibration(d.cc, lg[d.tm], d.y[d.tm], lg[d.cm], d.y[d.cm], alpha)


def mondrian(d: Data, lg: np.ndarray, alpha: float, T: float) -> list[float]:
    """Label-conditional split conformal: one LAC threshold per true class on the conformal fold
    (same T as the LAC fit). A class with too few rows for its rank gets threshold 1 (always in the set)."""
    from workers.verifier.sapling import log_probabilities

    lp = log_probabilities(lg[d.cm], T)
    yc = d.y[d.cm]
    qs = []
    for k in range(4):
        s = -np.expm1(lp[yc == k, k])
        rank = math.ceil((len(s) + 1) * (1 - alpha))
        qs.append(1.0 if rank > len(s) else float(np.partition(s, rank - 1)[rank - 1]))
    return qs


def predict_sets(lg: np.ndarray, T: float, qs: list[float]) -> list[dict[str, Any]]:
    """Per-class thresholds; same status rule as Sapling (singleton -> label, else unsure; empty set ->
    argmax only under the float-rounding fallback, which we treat as unsure here)."""
    from workers.verifier.sapling import log_probabilities

    lp = log_probabilities(lg, T)
    sc = -np.expm1(lp)
    out = []
    for row, s in zip(lg, sc):
        labs = [LABELS[k] for k in range(4) if s[k] <= qs[k]]
        out.append({"raw_label": LABELS[int(np.argmax(row))], "status": labs[0] if len(labs) == 1 else "unsure",
                    "prediction_set": labs})
    return out


def asym_cal(d: Data, lg: np.ndarray, alpha: float, T: float) -> dict[str, Any]:
    """Cost-sensitive version of ``binary_cal``: the approve threshold is the strictest finite conformal one
    (tau_hi = max P(supported) over non-supported conformal rows, i.e. alpha_hi = 1/(n_neg+1)), the reject
    threshold stays at ``alpha``. Approving a fake is treated as much worse than missing a real gain."""
    c = binary_cal(d, lg, alpha, T)
    from workers.verifier.sapling import log_probabilities

    p = np.exp(log_probabilities(lg[d.cm], T))[:, SUP]
    c |= {"kind": "binary", "variant": "asymmetric", "tau_hi": float(p[d.y[d.cm] != SUP].max()),
          "alpha_hi": 1.0 / (c["n_neg"] + 1)}
    return c


def binary_cal(d: Data, lg: np.ndarray, alpha: float, T: float) -> dict[str, Any]:
    """One-sided split-conformal thresholds on P(supported) (conformal fold, same T as LAC):
    approve iff p_sup > tau_hi, where tau_hi is the conformal (1-alpha) quantile of p_sup over NON-supported
    rows, so P(approve | claim not supported) <= alpha on exchangeable rows; reject iff p_sup < tau_lo,
    the mirror image over supported rows (P(reject | supported) <= alpha)."""
    from workers.verifier.sapling import log_probabilities

    p = np.exp(log_probabilities(lg[d.cm], T))[:, SUP]
    yc = d.y[d.cm]

    def q(s: np.ndarray) -> float:
        rank = math.ceil((len(s) + 1) * (1 - alpha))
        return 1.0 if rank > len(s) else float(np.partition(s, rank - 1)[rank - 1])

    tau_hi = q(p[yc != SUP])
    tau_lo = 1.0 - q(1.0 - p[yc == SUP])
    return {"kind": "binary", "alpha": alpha, "temperature": T, "tau_hi": tau_hi, "tau_lo": tau_lo,
            "n_neg": int((yc != SUP).sum()), "n_pos": int((yc == SUP).sum()), "labels": list(LABELS)}


def predict_binary(lg: np.ndarray, cal: dict[str, Any]) -> list[dict[str, Any]]:
    """approve -> 'supported'; reject -> the most likely non-supported label (a definite 'not established');
    otherwise 'unsure'. prediction_set: {supported} / non-supported labels / all labels."""
    from workers.verifier.sapling import log_probabilities

    pr = np.exp(log_probabilities(lg, cal["temperature"]))
    out = []
    for row, p in zip(lg, pr):
        if p[SUP] > cal["tau_hi"]:
            st, ps = "supported", ["supported"]
        elif p[SUP] < cal["tau_lo"]:
            alt = [k for k in range(4) if k != SUP]
            st, ps = LABELS[max(alt, key=lambda k: row[k])], [LABELS[k] for k in alt]
        else:
            st, ps = "unsure", list(LABELS)
        out.append({"raw_label": LABELS[int(np.argmax(row))], "status": st, "prediction_set": ps,
                    "verdict": {"supported": "valid", "unsure": "unsure"}.get(st, "invalid")})
    return out


def preds_for(d: Data, lg: np.ndarray, cal: dict[str, Any]) -> list[dict[str, Any]]:
    if cal.get("kind") == "mondrian":
        return predict_sets(lg, cal["temperature"], cal["q_by_class"])
    if cal.get("kind") == "binary":
        return predict_binary(lg, cal)
    return calibrate(lg, cal, empty_set_policy="rounded_argmax")


def quick(d: Data, lg: np.ndarray, cal: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    idx = np.where(mask)[0]
    pr = preds_for(d, lg, cal)
    m = v1.metrics([d.rows[i] for i in idx], [pr[i] for i in idx])
    imp = m.get("improvement", {})
    cov_ok = np.mean([LABELS[d.y[i]] in pr[i].get("prediction_set", []) for i in idx]) if "prediction_set" in pr[0] else None
    return {"n": m["n"], "raw_acc": round(m["raw_accuracy"], 4), "coverage": round(m["coverage"], 4),
            "ans_acc": None if m["answered_accuracy"] is None else round(m["answered_accuracy"], 4),
            "fake_approved": imp.get("fake_approved"), "fakes": imp.get("fakes"), "fake_caught": imp.get("fake_caught"),
            "genuine_accepted": imp.get("genuine_accepted"), "genuine": imp.get("genuine"),
            "genuine_rejected": imp.get("genuine_rejected"), "false_supported": m["false_supported"],
            "fake_approved_raw": imp.get("fake_approved_raw"), "set_contains_true": None if cov_ok is None else round(float(cov_ok), 4)}


def cals_for(d: Data, lg: np.ndarray) -> dict[str, dict[str, Any]]:
    out = {}
    for a in (0.01, 0.05):
        tag = f"a{a:g}".replace("0.", "")
        c = lac(d, lg, a)
        out[f"lac_{tag}"] = c
        out[f"mondrian_{tag}"] = {"kind": "mondrian", "alpha": a, "temperature": c["temperature"],
                                  "q_by_class": mondrian(d, lg, a, c["temperature"]), "labels": list(LABELS)}
        out[f"binary_{tag}"] = binary_cal(d, lg, a, c["temperature"])
        out[f"asym_{tag}"] = asym_cal(d, lg, a, c["temperature"])
    return out


# --------------------------------------------------------------------------- explore (val only)


GRID: list[dict[str, Any]] = [
    # 1) trainer parity vs v1's recipe (full update, lr 1e-4, batch 32) and cheaper parameterisations
    {"name": "full_lr1e-4_b32", "mode": "full", "lr": 1e-4, "batch": 32},
    {"name": "lora64_lr3e-4", "mode": "lora", "rank": 64, "lr": 3e-4},
    {"name": "lora64_lr1e-3", "mode": "lora", "rank": 64, "lr": 1e-3},
    {"name": "lora64_lr1e-4", "mode": "lora", "rank": 64, "lr": 1e-4, "epochs": 50, "patience": 6},
    {"name": "lora256_lr3e-4", "mode": "lora", "rank": 256, "lr": 3e-4},
    {"name": "last_lr1e-3", "mode": "last", "lr": 1e-3},
    # 2) loss: class-balanced, cost-sensitive (penalise P(supported) on non-supported rows)
    {"name": "lora64_lr3e-4_bal", "mode": "lora", "rank": 64, "lr": 3e-4, "balanced": True},
    {"name": "lora64_lr3e-4_bal_fp1", "mode": "lora", "rank": 64, "lr": 3e-4, "balanced": True, "fake_penalty": 1.0},
    {"name": "lora64_lr3e-4_fp1", "mode": "lora", "rank": 64, "lr": 3e-4, "fake_penalty": 1.0},
    # 3) regularisation
    {"name": "lora64_lr3e-4_drop3", "mode": "lora", "rank": 64, "lr": 3e-4, "drop_in": 0.3, "drop_h": 0.3},
    {"name": "lora64_lr3e-4_wd1e-2", "mode": "lora", "rank": 64, "lr": 3e-4, "wd": 1e-2},
    # 4) anchored to the stock head (added after the ContractNLI regression; selected on the ContractNLI dev third)
    {"name": "lora256_lr3e-4_l2sp1e-2", "mode": "lora", "rank": 256, "lr": 3e-4, "l2sp": 1e-2},
    {"name": "lora256_lr3e-4_l2sp1e-1", "mode": "lora", "rank": 256, "lr": 3e-4, "l2sp": 1e-1},
    {"name": "lora256_lr3e-4_l2sp1", "mode": "lora", "rank": 256, "lr": 3e-4, "l2sp": 1.0},
    {"name": "lora256_lr3e-4_l2sp1e-3", "mode": "lora", "rank": 256, "lr": 3e-4, "l2sp": 1e-3},
    {"name": "lora256_lr3e-4_l2sp1e-4", "mode": "lora", "rank": 256, "lr": 3e-4, "l2sp": 1e-4},
]


def cmd_explore(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = Data(args.threads, args.extra or None)
    if args.extra:
        d.is_tr = d.is_tr | d.extra_tr
    info = {"extra_run": args.extra or None, "extra_rows_used_in_explore": int(d.extra_tr.sum()),"parity_first_layer_cache": d.parity, "head_train": int(d.is_tr.sum()), "val_tf": int(d.is_val.sum()),
            "val_rule": {"policy_family": VAL_POLICY_FAMILIES, "evidence_family": VAL_EVIDENCE_FAMILIES},
            "val_label_counts": dict(Counter(LABELS[k] for k in d.y[d.is_val]))}
    print(json.dumps(info), flush=True)
    va = d.mask("val_tf")
    res: dict[str, Any] = {"info": info, "baselines": {}, "grid": [], "learning_curve": []}
    # v1 head on val_tf is NOT a clean baseline (v1 trained on some val_tf rows); reported, flagged.
    for name, lg in (("stock", d.stock_logits), ("v1_head_trained_on_part_of_val", d.v1_logits)):
        cals = cals_for(d, lg)
        res["baselines"][name] = {"val_ce": v1._ce(lg[va], d.y[va]), "val_acc": float((lg[va].argmax(1) == d.y[va]).mean()),
                                  **{c: quick(d, lg, cal, va) for c, cal in cals.items()}}
    print(json.dumps(res["baselines"], indent=0)[:3000], flush=True)
    grid = [g for g in GRID if not args.only or g["name"] in args.only.split(",")]
    seeds = list(range(args.seeds))
    for cfg in grid:
        kw = {k: v for k, v in cfg.items() if k != "name"}
        runs = []
        for s in seeds:
            r = train(d, d.is_tr, va, seed=SEED + s, verbose=args.verbose, **kw)
            runs.append(r)
            print(f"  {cfg['name']} seed{s}: best ep {r['best']['epoch']} val_ce {r['best']['val_ce']:.4f} "
                  f"val_acc {r['best'].get('val_acc', float('nan')):.4f} ({r['train_s']:.0f}s)", flush=True)
        ens = np.mean([r["logits"] for r in runs], 0)
        sp_state = head_state(d, soup([r["snap"] for r in runs]))
        sp_lg = logits_of_state(d, sp_state)
        entry = {"cfg": cfg, "seeds": [{"best": r["best"], "train_s": r["train_s"], "epochs_run": len(r["history"])} for r in runs],
                 "val_ce_mean": float(np.mean([r["best"]["val_ce"] for r in runs])),
                 "val_acc_mean": float(np.mean([r["best"].get("val_acc", np.nan) for r in runs])),
                 "best_epoch_median": float(np.median([r["best"]["epoch"] for r in runs])),
                 "ensemble": {"val_ce": v1._ce(ens[va], d.y[va]), "val_acc": float((ens[va].argmax(1) == d.y[va]).mean())},
                 "soup": {"val_ce": v1._ce(sp_lg[va], d.y[va]), "val_acc": float((sp_lg[va].argmax(1) == d.y[va]).mean())},
                 "calibrated_val": {}}
        # WiSE-FT style interpolation toward the stock head: head = stock + beta * delta (per seed, then ensemble)
        cd = d.mask("contractnli_dev")
        entry["wise"] = {}
        for beta in args.wise:
            lgs = []
            for r in runs:
                sn = {k: (None if v is None else v) for k, v in r["snap"].items()}
                sc = {"dW1": None if sn["dW1"] is None else beta * sn["dW1"], "db1": beta * sn["db1"],
                      "W2": d.stock_state["classifier.4.weight"] + beta * (sn["W2"] - d.stock_state["classifier.4.weight"]),
                      "b2": d.stock_state["classifier.4.bias"] + beta * (sn["b2"] - d.stock_state["classifier.4.bias"])}
                lgs.append(logits_of_state(d, head_state(d, sc)))
            el = np.mean(lgs, 0)
            entry["wise"][str(beta)] = {"val_ce": v1._ce(el[va], d.y[va]), "val_acc": float((el[va].argmax(1) == d.y[va]).mean()),
                                        "contractnli_dev_raw_acc": float((el[cd].argmax(1) == d.y[cd]).mean()),
                                        "calibrated_val": {c: quick(d, el, cal, va) for c, cal in cals_for(d, el).items()}}
            print("    wise", beta, json.dumps({k: v for k, v in entry["wise"][str(beta)].items() if k != "calibrated_val"}), flush=True)
        nrm = sum((v ** 2).sum() for v in d.stock_state.values()) ** 0.5
        entry["weight_drift_rel"] = [float(sum(((head_state(d, r["snap"])[k] - d.stock_state[k]) ** 2).sum() for k in d.stock_state) ** 0.5 / nrm)
                                     for r in runs]
        cd = d.mask("contractnli_dev")
        entry["contractnli_dev_raw_acc"] = {"stock": float((d.stock_logits[cd].argmax(1) == d.y[cd]).mean()),
                                            "ensemble": float((ens[cd].argmax(1) == d.y[cd]).mean()),
                                            "seeds": [float((r["logits"][cd].argmax(1) == d.y[cd]).mean()) for r in runs],
                                            "n": int(cd.sum())}
        for tag, lg in (("seed0", runs[0]["logits"]), ("ensemble", ens), ("soup", sp_lg)):
            for c, cal in cals_for(d, lg).items():
                entry["calibrated_val"][f"{tag}/{c}"] = quick(d, lg, cal, va)
        res["grid"].append(entry)
        print(json.dumps({k: entry[k] for k in ("val_ce_mean", "val_acc_mean", "best_epoch_median", "ensemble", "soup", "contractnli_dev_raw_acc", "weight_drift_rel")}), flush=True)
        for c in ("ensemble/lac_a01", "ensemble/lac_a05", "ensemble/mondrian_a05", "ensemble/binary_a01", "ensemble/binary_a05",
                  "soup/lac_a01", "soup/lac_a05", "soup/binary_a01"):
            print("    ", c, json.dumps(entry["calibrated_val"][c]), flush=True)
        (OUT / args.out_name).write_text(json.dumps(res, indent=1), encoding="utf-8")
    # learning curve (data bottleneck?): fraction of head-train GROUPS, val_tf fixed, one config
    if args.curve:
        lc_cfg = next(g for g in GRID if g["name"] == args.curve)
        tr_groups = sorted({d.groups[i] for i in np.where(d.is_tr)[0]})
        for frac in (0.25, 0.5, 0.75, 1.0):
            vals = []
            for s in seeds:
                rg = random.Random(SEED + s)
                keep = set(rg.sample(tr_groups, int(len(tr_groups) * frac)))
                m = d.is_tr & np.array([g in keep for g in d.groups])
                r = train(d, m, va, seed=SEED + s, **{k: v for k, v in lc_cfg.items() if k != "name"})
                vals.append((r["best"]["val_ce"], r["best"]["val_acc"]))
            res["learning_curve"].append({"frac": frac, "rows": int(m.sum()), "val_ce_mean": float(np.mean([v[0] for v in vals])),
                                          "val_acc_mean": float(np.mean([v[1] for v in vals]))})
            print("  curve", json.dumps(res["learning_curve"][-1]), flush=True)
        (OUT / args.out_name).write_text(json.dumps(res, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- final


def full_eval(d: Data, systems: dict[str, tuple[np.ndarray, dict[str, Any]]]) -> dict[str, Any]:
    ms = np.array([r["rec"]["model_ms"] for r in d.rows])
    out: dict[str, Any] = {}
    for sname, (lg, cal) in systems.items():
        pr = preds_for(d, lg, cal)
        out[sname] = {"calibration": {k: cal.get(k) for k in ("kind", "variant", "alpha", "alpha_hi", "temperature", "q_hat", "q_by_class",
                                                               "tau_hi", "tau_lo") if k in cal}, "splits": {}}
        for ename in ("test", "test_unseen_targets", "trap50", "contractnli", "contractnli_dev", "contractnli_heldout",
                      "contractnli_layout_pol", "conformal_fold"):
            idx = np.where(d.mask(ename))[0]
            m = v1.metrics([d.rows[i] for i in idx], [pr[i] for i in idx], ms=ms[idx].tolist())
            sets = [pr[i].get("prediction_set", pr[i].get("prediction_set")) for i in idx]
            m["set_contains_true"] = float(np.mean([LABELS[d.y[i]] in s for i, s in zip(idx, sets)]))
            if ename == "trap50":
                m["trap_grading"] = v1.trap_grade([d.rows[i] for i in idx], [pr[i] for i in idx])
            if ename == "test":
                fam: dict[str, Counter] = defaultdict(Counter)
                for i in idx:
                    r = d.rows[i]
                    if r.get("claim_kind") == "improvement" and r.get("fake_improvement"):
                        fam[r["category"]]["n"] += 1
                        fam[r["category"]]["approved"] += pr[i]["status"] == "supported"
                        fam[r["category"]]["caught"] += pr[i]["status"] not in ("supported", "unsure")
                m["fake_by_category"] = {k: dict(v) for k, v in sorted(fam.items())}
            out[sname]["splits"][ename] = m
    return out


def cmd_final(args: argparse.Namespace) -> None:
    import torch
    from safetensors.torch import save_file

    cfg = json.loads(args.config)
    name = cfg.pop("name", "v2")
    agg = cfg.pop("aggregate", "soup")  # soup (one head) | ensemble
    epochs = int(cfg.pop("epochs"))
    calk = cfg.pop("calibration", "lac")
    d = Data(args.threads, args.extra or None)
    train_mask = d.is_head | d.is_extra
    runs = []
    for s in range(args.seeds):
        r = train(d, train_mask, None, seed=SEED + s, fixed_epochs=epochs, **cfg)
        runs.append(r)
        print(f"  final seed{s}: {epochs} epochs, train loss {r['history'][-1]['train_loss']:.4f} ({r['train_s']:.0f}s)", flush=True)
    ens_lg = np.mean([r["logits"] for r in runs], 0)
    # parity: every seed head, merged to Sapling's own TriVectorInteractionHead shape, reproduces the trainer's logits
    X = d.F[:, :2048 * 3].reshape(-1, 3, 2048).numpy()  # P, E, C are the first three blocks of construct_features
    states, mod_lgs = [], []
    for r in runs:
        st_k = head_state(d, r["snap"])
        hd = d.ih.TriVectorInteractionHead(hidden_dim=2048)
        hd.load_state_dict({k: v.contiguous() for k, v in st_k.items()})
        mod_lgs.append(v1._apply(hd, X))
        states.append(st_k)
    merged_parity = float(max(np.abs(m - r["logits"]).max() for m, r in zip(mod_lgs, runs)))
    if agg == "soup":
        st = head_state(d, soup([r["snap"] for r in runs]))
        v2_lg = logits_of_state(d, st)
    else:
        st = None
        v2_lg = np.mean(mod_lgs, 0)
    systems: dict[str, tuple[np.ndarray, dict[str, Any]]] = {"stock": (d.stock_logits, d.stock_cal)}
    for tag, lg in (("stockhead", d.stock_logits), ("v1", d.v1_logits), ("v2", v2_lg)):
        cs = cals_for(d, lg)
        for a in ("a01", "a05"):
            if tag != "stockhead":
                systems[f"{tag}_lac_{a}"] = (lg, cs[f"lac_{a}"])
                systems[f"{tag}_mondrian_{a}"] = (lg, cs[f"mondrian_{a}"])
            systems[f"{tag}_binary_{a}"] = (lg, cs[f"binary_{a}"])
            systems[f"{tag}_asym_{a}"] = (lg, cs[f"asym_{a}"])
    systems["v2_seed0_lac_a01"] = (runs[0]["logits"], cals_for(d, runs[0]["logits"])["lac_a01"])
    systems["v2_seed0_binary_a01"] = (runs[0]["logits"], cals_for(d, runs[0]["logits"])["binary_a01"])
    res = full_eval(d, systems)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = Path(args.out_name).stem
    rel = OUT / ("release" if stem == "final" else f"release_{stem}")
    rel.mkdir(exist_ok=True)
    if st is not None:
        save_file({k: v.contiguous() for k, v in st.items()}, str(rel / "head.safetensors"))
        rel_info = {"file": "head.safetensors", "sha256": hashlib.sha256((rel / "head.safetensors").read_bytes()).hexdigest()}
    else:
        rel_info = ensemble_release(rel / "heads_ensemble_delta.safetensors", [r["snap"] for r in runs])
    for a in ("a01", "a05"):
        for kind in ("lac", "binary", "asym"):
            c = systems[f"v2_{kind}_{a}"][1]
            fn = f"calibration_{a}.json" if kind == calk else f"calibration_{kind}_{a}.json"
            (rel / fn).write_text(json.dumps(c, indent=2), encoding="utf-8")
    out = {"name": name, "config": cfg, "epochs": epochs, "seeds": args.seeds, "aggregate": agg, "primary_calibration": calk,
           "merged_head_parity_vs_sapling_module": merged_parity, "release": rel_info,
           "weight_drift_rel_per_seed": [float(sum(((sk[k] - d.stock_state[k]) ** 2).sum() for k in sk) ** 0.5 /
                                               sum((d.stock_state[k] ** 2).sum() for k in sk) ** 0.5) for sk in states],
           "train_rows": int(train_mask.sum()), "extra_run": args.extra or None, "train_s_total": float(sum(r["train_s"] for r in runs)),
           "results": res}
    (OUT / args.out_name).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    np.save(OUT / f"{stem}_v2_logits.npy", v2_lg)
    write_jsonl(OUT / f"{stem}_v2_ids.jsonl", [{"id": r["id"]} for r in d.rows])
    for sname, v in res.items():
        t = v["splits"]["test"]
        imp = t["improvement"]
        print(f"{sname:24s} test raw {t['raw_accuracy']:.3f} cov {t['coverage']:.3f} ans_acc {t['answered_accuracy']} "
              f"fakeAppr {imp['fake_approved']}/{imp['fakes']} caught {imp['fake_caught']} genAcc {imp['genuine_accepted']}/{imp['genuine']} "
              f"genRej {imp['genuine_rejected']} | trap {v['splits']['trap50']['trap_grading']['all']['correct']}/50 "
              f"trapFakeAppr {v['splits']['trap50']['trap_grading']['all']['fake_approved']} | cnli raw {v['splits']['contractnli']['raw_accuracy']:.3f}", flush=True)
    print("merged parity", merged_parity, "release", json.dumps(rel_info))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    hp = sub.add_parser("hosted-parity")
    hp.add_argument("--n", type=int, default=30)
    hp.add_argument("--sleep", type=float, default=0.5)
    ex = sub.add_parser("explore")
    ex.add_argument("--threads", type=int, default=8)
    ex.add_argument("--seeds", type=int, default=3)
    ex.add_argument("--only", default="")
    ex.add_argument("--curve", default="")
    ex.add_argument("--verbose", action="store_true")
    ex.add_argument("--extra", default="", help="extra head rows run (e.g. sapling_v2x)")
    ex.add_argument("--wise", type=lambda x: [float(v) for v in x.split(",")], default=[], help="interpolation betas")
    ex.add_argument("--out-name", default="explore.json")
    fi = sub.add_parser("final")
    fi.add_argument("--threads", type=int, default=8)
    fi.add_argument("--seeds", type=int, default=5)
    fi.add_argument("--extra", default="")
    fi.add_argument("--out-name", default="final.json")
    fi.add_argument("--config", required=True, help='JSON, e.g. {"mode":"lora","rank":64,"lr":3e-4,"epochs":12}')
    args = ap.parse_args(argv)
    {"hosted-parity": cmd_hosted_parity, "explore": cmd_explore, "final": cmd_final}[args.cmd](args)


if __name__ == "__main__":
    main()
