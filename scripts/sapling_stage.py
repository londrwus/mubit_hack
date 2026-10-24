"""Sapling stages 1-2 (D28, docs/sapling_research.md, docs/sapling_gate.md).

Subcommands (run from the repo root; PYTHONUTF8=1):

    prep-cnli   download ContractNLI Task A *test* (HF ``kiddothe2b/contract-nli``) and write
                (policy, evidence, claim) rows in two layouts -> data/sapling/contractnli/rows.jsonl
                (needs ``--with pyarrow``)
    stage1      stock Sapling baseline from the extraction run's records (stock logits + stock
                calibration from the same H100 pass) -> data/sapling/<run>/stage1.json
    stage2      head fine-tune on cached features (backbone frozen), temperature + conformal
                refit (ML-research only), evaluation, release files -> data/sapling/<run>/stage2.json,
                data/sapling/release/  (needs ``--extra ml --with safetensors==0.8.0``)

The H100 part is ``python -m workers.runners.modal_sapling extract ...`` (one combined run);
``fetch`` copies the cached features to data/sapling/<run>/volume. Nothing here is an LLM call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workers.verifier.sapling import LABELS, calibrate, read_jsonl, write_jsonl  # noqa: E402

SAPLING_DIR = ROOT / "data" / "sapling"
CNLI_DIR = SAPLING_DIR / "contractnli"
CNLI_REPO = "kiddothe2b/contract-nli"
CNLI_URL = f"https://huggingface.co/api/datasets/{CNLI_REPO}/parquet/contractnli_a/test/0.parquet"
# contract-nli.py in that repo: CONTRACTNLI_LABELS = ["contradiction", "entailment", "neutral"]
CNLI_LABEL_NAMES = ("contradiction", "entailment", "neutral")
CNLI_TO_SAPLING = {"entailment": "supported", "contradiction": "violation", "neutral": "insufficient_information"}
CNLI_PER_CLASS = {"entailment": 150, "contradiction": 100, "neutral": 150}
CNLI_SEED = 20261024

# Sapling's training-time ContractNLI layout is not published (docs/provenance.md: "evaluation
# inputs ... remain in the training archive"), so we try two layouts and pick, BEFORE looking at
# our head, the one where STOCK raw accuracy is higher (closest to what Sapling was trained on).
CNLI_LAYOUTS: dict[str, dict[str, str]] = {
    "ev": {  # contract excerpt as evidence
        "policy": ("Statements about a non-disclosure agreement are judged only against the contract text supplied "
                   "as evidence. A statement is established when the contract text says it, contradicted when the "
                   "contract text says the opposite, and unknown when the contract text does not address it."),
    },
    "pol": {  # contract excerpt as the policy (the agreement is the rule set)
        "evidence": ("The statement below concerns the non-disclosure agreement whose relevant text is quoted in "
                     "the policy. No other facts about the agreement are available."),
    },
}


# --------------------------------------------------------------------------- prep-cnli


def cmd_prep_cnli(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    raw = CNLI_DIR / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    pq_path = raw / "contractnli_a_test.parquet"
    if not pq_path.exists():
        urllib.request.urlretrieve(CNLI_URL, pq_path)
    sha = hashlib.sha256(pq_path.read_bytes()).hexdigest()
    t = pq.read_table(pq_path).to_pylist()
    seen: set[tuple[str, str]] = set()
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, r in enumerate(t):
        key = (r["premise"].strip(), r["hypothesis"].strip())
        if key in seen or not key[0] or not key[1]:
            continue
        seen.add(key)
        by_label[CNLI_LABEL_NAMES[int(r["label"])]].append({"src_index": i, "premise": key[0], "hypothesis": key[1]})
    rng = random.Random(CNLI_SEED)
    picked: list[tuple[str, dict[str, Any]]] = []
    for lab, n in CNLI_PER_CLASS.items():
        pool = by_label[lab]
        picked += [(lab, r) for r in rng.sample(pool, min(n, len(pool)))]
    picked.sort(key=lambda x: x[1]["src_index"])
    rows: list[dict[str, Any]] = []
    for k, (lab, r) in enumerate(picked):
        for layout, fixed in CNLI_LAYOUTS.items():
            if layout == "ev":
                policy, evidence = fixed["policy"], r["premise"]
            else:
                policy, evidence = r["premise"], fixed["evidence"]
            rows.append({"id": f"cnli_a_test_{layout}_{k:04d}", "policy": policy, "evidence": evidence,
                         "claim": r["hypothesis"], "label": CNLI_TO_SAPLING[lab], "cnli_label": lab,
                         "layout": layout, "case": k, "src_index": r["src_index"], "split": "contractnli_test"})
    write_jsonl(CNLI_DIR / "rows.jsonl", rows)
    meta = {"source": CNLI_URL, "repo": CNLI_REPO, "config": "contractnli_a", "split": "test",
            "parquet_sha256": sha, "rows_in_parquet": len(t), "unique_pairs": len(seen),
            "available": {k: len(v) for k, v in by_label.items()}, "per_class": CNLI_PER_CLASS,
            "seed": CNLI_SEED, "cases": len(picked), "layouts": list(CNLI_LAYOUTS), "rows": len(rows),
            "label_map": CNLI_TO_SAPLING, "licence": "cc-by-nc-sa-4.0 (evaluation only; not redistributed)"}
    (CNLI_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------- shared loading


def load_labelled(run: str) -> list[dict[str, Any]]:
    """Join the H100 records (stock logits, status, latency) with local labels and metadata on ``id``."""
    recs = {r["id"]: r for r in read_jsonl(SAPLING_DIR / run / "records.jsonl")}
    src: list[dict[str, Any]] = []
    for p in ("data/ml_claims/train.jsonl", "data/ml_claims/test.jsonl", "data/ml_claims/trap50.jsonl"):
        src += read_jsonl(ROOT / p)
    cnli = CNLI_DIR / "rows.jsonl"
    if cnli.exists():
        src += read_jsonl(cnli)
    out = []
    for s in src:
        r = recs.get(s["id"])
        if r is None:
            continue
        meta = {k: v for k, v in s.items() if k not in ("policy", "evidence", "claim", "hidden")}
        out.append({**meta, "rec": r})
    return out


def load_features(run: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Concatenate the fetched chunk_*.npy files; map id -> global feature row."""
    vol = SAPLING_DIR / run / "volume"
    feats, logits, index = [], [], {}
    offset = 0
    for fp in sorted(vol.glob("chunk_*_features.npy")):
        base = fp.name.replace("_features.npy", "")
        f = np.load(fp)
        lg = np.load(vol / f"{base}_logits.npy")
        rows = read_jsonl(vol / f"{base}_rows.jsonl")
        for r in rows:
            if r.get("feature_index") is not None:
                index[r["id"]] = offset + int(r["feature_index"])
        feats.append(f)
        logits.append(lg)
        offset += len(f)
    return np.concatenate(feats), np.concatenate(logits), index


# --------------------------------------------------------------------------- metrics


def metrics(rows: list[dict[str, Any]], preds: list[dict[str, Any]], *, ms: list[float] | None = None) -> dict[str, Any]:
    """Accuracy (raw argmax and delivered status), coverage, fake improvements approved, genuine rejected.

    ``preds[i]`` has ``raw_label``, ``status`` (a label or ``unsure``). Delivered accuracy counts
    ``unsure`` as wrong; answered accuracy is over non-``unsure`` rows. A label in ``label_alt``
    also counts as correct (e.g. noise -> violation), reported separately as ``*_lenient``.
    """
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def ok(r: dict[str, Any], lab: str, lenient: bool) -> bool:
        return lab == r["label"] or (lenient and lab in (r.get("label_alt") or []))

    raw_ok = sum(ok(r, p["raw_label"], False) for r, p in zip(rows, preds))
    raw_ok_l = sum(ok(r, p["raw_label"], True) for r, p in zip(rows, preds))
    answered = [(r, p) for r, p in zip(rows, preds) if p["status"] != "unsure"]
    ans_ok = sum(ok(r, p["status"], False) for r, p in answered)
    ans_ok_l = sum(ok(r, p["status"], True) for r, p in answered)
    out: dict[str, Any] = {
        "n": n,
        "raw_accuracy": raw_ok / n, "raw_accuracy_lenient": raw_ok_l / n,
        "answered": len(answered), "coverage": len(answered) / n,
        "answered_accuracy": ans_ok / len(answered) if answered else None,
        "answered_accuracy_lenient": ans_ok_l / len(answered) if answered else None,
        "delivered_accuracy": ans_ok / n, "delivered_accuracy_lenient": ans_ok_l / n,
        "false_supported_raw": sum(p["raw_label"] == "supported" and r["label"] != "supported" for r, p in zip(rows, preds)),
        "false_supported": sum(p["status"] == "supported" and r["label"] != "supported" for r, p in zip(rows, preds)),
    }
    imp = [(r, p) for r, p in zip(rows, preds) if r.get("claim_kind") == "improvement"]
    if imp:
        fakes = [(r, p) for r, p in imp if r.get("fake_improvement")]
        gen = [(r, p) for r, p in imp if not r.get("fake_improvement")]
        out["improvement"] = {
            "n": len(imp), "fakes": len(fakes), "genuine": len(gen),
            # headline: share of fake improvements that come out ``supported`` (approved)
            "fake_approved": sum(p["status"] == "supported" for _, p in fakes),
            "fake_approved_rate": sum(p["status"] == "supported" for _, p in fakes) / max(1, len(fakes)),
            "fake_approved_raw": sum(p["raw_label"] == "supported" for _, p in fakes),
            "fake_approved_raw_rate": sum(p["raw_label"] == "supported" for _, p in fakes) / max(1, len(fakes)),
            # caught = a non-supported, non-unsure status on a fake
            "fake_caught": sum(p["status"] not in ("supported", "unsure") for _, p in fakes),
            "fake_unsure": sum(p["status"] == "unsure" for _, p in fakes),
            "genuine_accepted": sum(p["status"] == "supported" for _, p in gen),
            # rejected = a definite non-supported answer on a real gain (unsure counted separately)
            "genuine_rejected": sum(p["status"] not in ("supported", "unsure") for _, p in gen),
            "genuine_unsure": sum(p["status"] == "unsure" for _, p in gen),
            "genuine_rejected_raw": sum(p["raw_label"] != "supported" for _, p in gen),
            # binary valid/invalid view (as the LLM judge is graded): supported = valid, any other
            # definite label = invalid, unsure = unsure
            "binary_accuracy_raw": sum((p["raw_label"] == "supported") == (not r.get("fake_improvement")) for r, p in imp) / len(imp),
        }
    conf: dict[str, Counter] = defaultdict(Counter)
    for r, p in zip(rows, preds):
        conf[r["label"]][p["status"]] += 1
    out["status_confusion"] = {k: dict(v) for k, v in sorted(conf.items())}
    if ms:
        a = np.asarray(ms, dtype=np.float64)
        out["latency_ms"] = {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "mean": float(a.mean())}
    return out


def split_of(r: dict[str, Any]) -> str:
    if r["id"].startswith("cnli_"):
        return f"contractnli_{r['layout']}"
    if r["split"] == "trap50":
        return "trap50"
    if r["split"] == "train":
        return f"train_{r['fold']}"
    return r["split"]


def stock_pred(rec: dict[str, Any]) -> dict[str, Any]:
    return {"raw_label": rec["raw_label"], "status": rec["status"]}


# --------------------------------------------------------------------------- stage 1


def cmd_stage1(args: argparse.Namespace) -> None:
    rows = load_labelled(args.run)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[split_of(r)].append(r)
    out: dict[str, Any] = {"run": args.run, "rows": len(rows), "calibration": "stock (T=5.1716, q_hat=0.9955)", "splits": {}}
    for name, rs in sorted(groups.items()):
        ok = [r for r in rs if r["rec"].get("feature_index") is not None]
        out["splits"][name] = metrics(ok, [stock_pred(r["rec"]) for r in ok], ms=[r["rec"]["model_ms"] for r in ok])
        out["splits"][name]["errors"] = len(rs) - len(ok)
        out["splits"][name]["input_tokens_median"] = float(np.median([r["rec"]["input_tokens"] for r in ok])) if ok else None
    trap = [r for r in groups.get("trap50", []) if r["rec"].get("feature_index") is not None]
    for sp in ("dev", "test"):
        sub = [r for r in trap if r.get("trap_split") == sp]
        out["splits"][f"trap50_{sp}"] = metrics(sub, [stock_pred(r["rec"]) for r in sub], ms=[r["rec"]["model_ms"] for r in sub])
    path = SAPLING_DIR / args.run / "stage1.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    brief = {k: {m: v.get(m) for m in ("n", "raw_accuracy", "coverage", "answered_accuracy", "delivered_accuracy")}
             | ({"fake_approved_rate": v["improvement"]["fake_approved_rate"],
                 "fake_approved_raw_rate": v["improvement"]["fake_approved_raw_rate"]} if "improvement" in v else {})
             for k, v in out["splits"].items()}
    print(json.dumps(brief, indent=1))


# --------------------------------------------------------------------------- stage 2

PKG_RUNTIME = SAPLING_DIR / "pkg" / "runtime"  # fetched from the Volume; sha256 matches distribution_manifest.json
STOCK_HEAD = SAPLING_DIR / "stock_head" / "head.safetensors"
STOCK_CAL = SAPLING_DIR / "stock_head" / "calibration.json"
RELEASE = SAPLING_DIR / "release"
LABEL_INDEX = {lab: i for i, lab in enumerate(LABELS)}
VAL_FRACTION = 0.15  # of head-fold groups, for early stopping (grouped by scenario)
SEED = 20261024
# Small, pre-declared grid. Selection = lowest validation cross-entropy (never test).
GRID: list[dict[str, float]] = [
    {"lr": 1e-5, "l2sp": 0.0}, {"lr": 3e-5, "l2sp": 0.0}, {"lr": 1e-4, "l2sp": 0.0},
    {"lr": 3e-5, "l2sp": 1e-2}, {"lr": 1e-4, "l2sp": 1e-2},
]
MAX_EPOCHS = 25
PATIENCE = 3
BATCH = 32


def _import_pkg(name: str) -> Any:
    """Import one of Sapling's own runtime modules (MIT) by file path, without touching sys.path."""
    import importlib.util

    path = PKG_RUNTIME / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"sapling_pkg_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _group_val(group: str) -> bool:
    h = int(hashlib.sha256(f"{SEED}:{group}".encode()).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF) < VAL_FRACTION


def _apply(head: Any, X: np.ndarray, bs: int = 256) -> np.ndarray:
    import torch

    head.eval()
    out = []
    with torch.inference_mode():
        for s in range(0, len(X), bs):
            x = torch.from_numpy(np.ascontiguousarray(X[s:s + bs]))
            out.append(head(x[:, 0], x[:, 1], x[:, 2]).float().numpy())
    return np.concatenate(out) if out else np.zeros((0, 4), np.float32)


def _ce(logits: np.ndarray, y: np.ndarray) -> float:
    z = logits.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-lp[np.arange(len(y)), y].mean())


def train_head(stock_state: dict[str, Any], Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
               *, lr: float, l2sp: float, seed: int = SEED, log: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Warm-start from the stock head; AdamW (no decay toward 0), optional L2-SP pull to the stock weights;
    early stopping on the grouped validation split. Returns (best state, summary)."""
    import torch

    ih = _import_pkg("interaction_head")
    torch.manual_seed(seed)
    head = ih.TriVectorInteractionHead(hidden_dim=Xtr.shape[2])
    head.load_state_dict(stock_state)
    ref = {k: v.detach().clone() for k, v in head.state_dict().items()}
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    best = {"val_ce": _ce(_apply(head, Xva), yva), "epoch": 0}
    best["val_acc"] = float((_apply(head, Xva).argmax(1) == yva).mean())
    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    hist = [{"epoch": 0, "val_ce": best["val_ce"], "val_acc": best["val_acc"]}]
    g = torch.Generator().manual_seed(seed)
    bad = 0
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        head.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for s in range(0, len(perm), BATCH):
            idx = perm[s:s + BATCH]
            x = Xt[idx]
            loss = torch.nn.functional.cross_entropy(head(x[:, 0], x[:, 1], x[:, 2]), yt[idx])
            if l2sp:
                loss = loss + l2sp * sum(((p - ref[n]) ** 2).sum() for n, p in head.named_parameters())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        va = _apply(head, Xva)
        ce, acc = _ce(va, yva), float((va.argmax(1) == yva).mean())
        hist.append({"epoch": ep, "train_loss": tot / len(Xt), "val_ce": ce, "val_acc": acc, "t_s": time.time() - t0})
        print(f"  lr={lr:g} l2sp={l2sp:g} ep{ep}: train {tot / len(Xt):.4f} val_ce {ce:.4f} val_acc {acc:.4f}", flush=True)
        if ce < best["val_ce"] - 1e-4:
            best = {"val_ce": ce, "val_acc": acc, "epoch": ep}
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    drift = float(sum(((best_state[k] - ref[k]) ** 2).sum() for k in ref) ** 0.5)
    norm = float(sum((ref[k] ** 2).sum() for k in ref) ** 0.5)
    summary = {"lr": lr, "l2sp": l2sp, "best_epoch": best["epoch"], "val_ce": best["val_ce"], "val_acc": best["val_acc"],
               "epochs_run": hist[-1]["epoch"], "train_s": time.time() - t0, "weight_drift_l2": drift,
               "weight_drift_rel": drift / norm, "history": hist}
    if log is not None:
        log.append(summary)
    return best_state, summary


def fit_calibration(cc_mod: Any, logits_t: np.ndarray, y_t: np.ndarray, logits_c: np.ndarray, y_c: np.ndarray,
                    alpha: float) -> dict[str, Any]:
    cal = cc_mod.ConformalCalibrator(alpha=alpha)
    cal.fit(logits_c, y_c, temperature_logits=logits_t, temperature_labels=y_t)
    return cal.to_dict()


def fit_multidomain(cc_mod: Any, domains_t: dict[str, tuple[np.ndarray, np.ndarray]],
                    domains_c: dict[str, tuple[np.ndarray, np.ndarray]], alpha: float) -> dict[str, Any]:
    cal = cc_mod.ConformalCalibrator(alpha=alpha)
    cal.fit_multidomain(domains_c, temperature_domains=domains_t)
    return cal.to_dict()


def trap_grade(rows: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, Any]:
    """Grade Sapling on trap50 exactly as the LLM judge is graded (scripts/eval_trap_set.py):
    supported -> valid, any other definite label -> invalid, unsure -> unsure; correct if the
    verdict is in the case's ``acceptable_verdicts``."""
    trap = {t["id"]: t for t in read_jsonl(ROOT / "data/trap_set/trap_set_v1.jsonl")}
    out: dict[str, Any] = {}
    for sp in ("dev", "test", "all"):
        sel = [(r, p) for r, p in zip(rows, preds) if sp == "all" or r.get("trap_split") == sp]
        res = {"n": len(sel), "correct": 0, "fake_approved": 0, "fakes": 0, "genuine": 0, "genuine_rejected": 0,
               "genuine_unsure": 0, "unsure": 0}
        for r, p in sel:
            t = trap[r["trap_id"]]
            v = "unsure" if p["status"] == "unsure" else ("valid" if p["status"] == "supported" else "invalid")
            res["correct"] += v in t["acceptable_verdicts"]
            res["unsure"] += v == "unsure"
            if t["label"] == "invalid":
                res["fakes"] += 1
                res["fake_approved"] += v == "valid"
            else:
                res["genuine"] += 1
                res["genuine_rejected"] += v == "invalid"
                res["genuine_unsure"] += v == "unsure"
        res["accuracy"] = res["correct"] / max(1, res["n"])
        out[sp] = res
    return out


def _cal_preds(logits: np.ndarray, cal: dict[str, Any]) -> list[dict[str, Any]]:
    return calibrate(logits, cal, empty_set_policy="rounded_argmax")


def cmd_stage2(args: argparse.Namespace) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(args.threads)
    cc_mod = _import_pkg("conformal_calibrator")
    ih = _import_pkg("interaction_head")
    run_dir = SAPLING_DIR / args.run
    rows = load_labelled(args.run)
    X, stock_logits_cached, index = load_features(args.run)
    rows = [r for r in rows if r["id"] in index]
    fi = np.array([index[r["id"]] for r in rows])
    y = np.array([LABEL_INDEX[r["label"]] for r in rows])
    out: dict[str, Any] = {"run": args.run, "rows": len(rows), "features_shape": list(X.shape)}

    # ---- parity first: stock head on cached features reproduces the stock logits from the H100 pass
    stock_state = load_file(str(STOCK_HEAD))
    stock_head = ih.TriVectorInteractionHead(hidden_dim=X.shape[2])
    stock_head.load_state_dict(stock_state)
    rec_logits = np.array([r["rec"]["logits"] for r in rows], dtype=np.float32)
    t0 = time.time()
    re_logits = _apply(stock_head, X[fi])
    head_cpu_ms = (time.time() - t0) * 1000 / len(fi)
    parity = {"vs_records": float(np.abs(re_logits - rec_logits).max()),
              "vs_cached_npy": float(np.abs(re_logits - stock_logits_cached[fi]).max()),
              "argmax_agree": float((re_logits.argmax(1) == rec_logits.argmax(1)).mean()),
              "head_cpu_ms_per_case_batched": head_cpu_ms}
    stock_cal = json.loads(STOCK_CAL.read_text(encoding="utf-8"))
    recal = _cal_preds(re_logits, stock_cal)
    parity["stock_status_agree"] = float(np.mean([a["status"] == r["rec"]["status"] for a, r in zip(recal, rows)]))
    parity["ok"] = parity["vs_records"] <= 1e-3 and parity["argmax_agree"] == 1.0 and parity["stock_status_agree"] == 1.0
    out["parity"] = parity
    print("parity", json.dumps(parity), flush=True)
    if not parity["ok"]:
        raise SystemExit("parity check failed: cached features do not reproduce the stock logits")

    # ---- splits (all grouped by scenario: the generator keeps a group inside one fold)
    sp = np.array([split_of(r) for r in rows])
    groups = [r.get("group") or r["id"] for r in rows]
    is_head = sp == "train_head"
    is_val = np.array([h and _group_val(g) for h, g in zip(is_head, groups)])
    is_tr = is_head & ~is_val
    out["splits"] = {"head_train": int(is_tr.sum()), "head_val": int(is_val.sum()),
                     "temperature": int((sp == "train_temperature").sum()), "conformal": int((sp == "train_conformal").sum()),
                     "test": int((sp == "test").sum()), "trap50": int((sp == "trap50").sum()),
                     "contractnli_ev": int((sp == "contractnli_ev").sum()), "contractnli_pol": int((sp == "contractnli_pol").sum())}
    g_tr = {groups[i] for i in np.where(is_tr)[0]}
    g_va = {groups[i] for i in np.where(is_val)[0]}
    assert not (g_tr & g_va), "group leak between head_train and head_val"

    # ---- head fine-tune (backbone frozen: only cached features are used)
    Xf = X[fi]
    log: list[dict[str, Any]] = []
    best_state, best_sum = None, None
    grid = GRID[: args.grid] if args.grid else GRID
    for cfg in grid:
        st, sm = train_head(stock_state, Xf[is_tr], y[is_tr], Xf[is_val], y[is_val], lr=cfg["lr"], l2sp=cfg["l2sp"], log=log)
        if best_sum is None or sm["val_ce"] < best_sum["val_ce"]:
            best_state, best_sum = st, sm
    out["grid"] = [{k: v for k, v in s.items() if k != "history"} for s in log]
    out["selected"] = {k: v for k, v in best_sum.items() if k != "history"}
    out["selected_history"] = best_sum["history"]
    ours = ih.TriVectorInteractionHead(hidden_dim=X.shape[2])
    ours.load_state_dict(best_state)
    our_logits = _apply(ours, Xf)

    # ---- calibration refit (ML-research only): temperature fold -> T, conformal fold -> q_hat
    tm, cm = sp == "train_temperature", sp == "train_conformal"
    cals: dict[str, dict[str, Any]] = {"stock": stock_cal}
    for a in (0.01, 0.05):
        tag = f"{a:g}".replace("0.", "")
        cals[f"stockhead_mlcal_a{tag}"] = fit_calibration(cc_mod, re_logits[tm], y[tm], re_logits[cm], y[cm], a)
        cals[f"ours_mlcal_a{tag}"] = fit_calibration(cc_mod, our_logits[tm], y[tm], our_logits[cm], y[cm], a)

    # ---- ContractNLI layout choice: the one where the STOCK head's raw accuracy is higher (decided on stock only)
    acc_layout = {}
    for lay in ("ev", "pol"):
        m = sp == f"contractnli_{lay}"
        if m.any():
            acc_layout[lay] = float((re_logits[m].argmax(1) == y[m]).mean())
    layout = max(acc_layout, key=acc_layout.get) if acc_layout else None
    out["contractnli_layout"] = {"stock_raw_accuracy": acc_layout, "chosen": layout}

    # ---- multidomain refit for transparency (ML conformal/temperature folds + ContractNLI cases split in thirds)
    if layout:
        cn = np.where(sp == f"contractnli_{layout}")[0]
        third = np.array([rows[i]["case"] % 3 for i in cn])
        cn_t, cn_c, cn_e = cn[third == 0], cn[third == 1], cn[third == 2]
        out["splits"]["contractnli_md_temperature"], out["splits"]["contractnli_md_conformal"], out["splits"]["contractnli_md_eval"] = len(cn_t), len(cn_c), len(cn_e)
        for name, lg in (("stockhead", re_logits), ("ours", our_logits)):
            dt = {"ml_research": (lg[tm], y[tm]), "contract_nli": (lg[cn_t], y[cn_t])}
            dc = {"ml_research": (lg[cm], y[cm]), "contract_nli": (lg[cn_c], y[cn_c])}
            cals[f"{name}_mdcal_a01"] = fit_multidomain(cc_mod, dt, dc, 0.01)
        md_eval = set(cn_e.tolist())
    else:
        md_eval = set()

    # ---- evaluation
    ms = np.array([r["rec"]["model_ms"] for r in rows])
    systems = {
        "stock": (re_logits, "stock"),
        "stock_head+ml_calibration_a01": (re_logits, "stockhead_mlcal_a01"),
        "stock_head+ml_calibration_a05": (re_logits, "stockhead_mlcal_a05"),
        "sapling_research_a01": (our_logits, "ours_mlcal_a01"),
        "sapling_research_a05": (our_logits, "ours_mlcal_a05"),
        "sapling_research_head+stock_calibration": (our_logits, "stock"),
    }
    if md_eval:
        systems["stock_head+multidomain_calibration_a01"] = (re_logits, "stockhead_mdcal_a01")
        systems["sapling_research+multidomain_calibration_a01"] = (our_logits, "ours_mdcal_a01")
    eval_splits = {"test": sp == "test", "trap50": sp == "trap50", "head_val": is_val,
                   "conformal_fold": cm}
    if layout:
        eval_splits["contractnli"] = sp == f"contractnli_{layout}"
        eval_splits["contractnli_md_eval"] = np.array([i in md_eval for i in range(len(rows))])
        other = "pol" if layout == "ev" else "ev"
        if (sp == f"contractnli_{other}").any():
            eval_splits[f"contractnli_layout_{other}"] = sp == f"contractnli_{other}"
    test_unseen = (sp == "test") & np.array([not r.get("target_seen_in_train", True) for r in rows])
    eval_splits["test_unseen_targets"] = test_unseen
    res: dict[str, Any] = {}
    for sname, (lg, cname) in systems.items():
        preds = _cal_preds(lg, cals[cname])
        res[sname] = {"calibration": cname, "temperature": cals[cname]["temperature"], "q_hat": cals[cname]["q_hat"], "splits": {}}
        for ename, mask in eval_splits.items():
            idx = np.where(mask)[0]
            if not len(idx):
                continue
            m = metrics([rows[i] for i in idx], [preds[i] for i in idx], ms=ms[idx].tolist())
            if ename == "trap50":
                m["trap_grading"] = trap_grade([rows[i] for i in idx], [preds[i] for i in idx])
            if ename == "test":
                fam: dict[str, Counter] = defaultdict(Counter)
                for i in idx:
                    r = rows[i]
                    if r.get("claim_kind") == "improvement" and r.get("fake_improvement"):
                        fam[r["category"]]["n"] += 1
                        fam[r["category"]]["approved"] += preds[i]["status"] == "supported"
                        fam[r["category"]]["caught"] += preds[i]["status"] not in ("supported", "unsure")
                m["fake_by_category"] = {k: dict(v) for k, v in sorted(fam.items())}
            res[sname]["splits"][ename] = m
    out["results"] = res
    out["calibrations"] = {k: {kk: v[kk] for kk in ("alpha", "temperature", "q_hat", "calibration_count", "temperature_count",
                                                     "domain_quantiles", "coverage_scope") if kk in v} for k, v in cals.items()}
    out["latency"] = {"h100_model_ms_median_all_rows": float(np.median(ms)),
                      "h100_model_ms_p90_all_rows": float(np.percentile(ms, 90)),
                      "note": "measured once for the stock forward pass; the fine-tuned head has the same shape and FLOPs"}

    # ---- release files (head-only delta + our calibration). NOT published anywhere.
    RELEASE.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in best_state.items()}, str(RELEASE / "head.safetensors"))
    (RELEASE / "calibration.json").write_text(json.dumps(cals["ours_mlcal_a01"], indent=2), encoding="utf-8")
    (RELEASE / "calibration_alpha0.05.json").write_text(json.dumps(cals["ours_mlcal_a05"], indent=2), encoding="utf-8")
    sha = hashlib.sha256((RELEASE / "head.safetensors").read_bytes()).hexdigest()
    out["release"] = {"head_sha256": sha, "head_bytes": (RELEASE / "head.safetensors").stat().st_size,
                      "base": "MubitLabs/banyan-sapling@437cf0b75fd52970464f9f22604a4b92ab0bec9f",
                      "stock_head_sha256": hashlib.sha256(STOCK_HEAD.read_bytes()).hexdigest()}
    (RELEASE / "training_config.json").write_text(json.dumps({
        "selected": out["selected"], "grid": out["grid"], "splits": out["splits"], "seed": SEED,
        "optimizer": "AdamW(weight_decay=0)", "batch": BATCH, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "selection": "lowest validation cross-entropy on head_val (15% of head-fold scenario groups)",
        "warm_start": "stock head (MubitLabs/banyan-sapling @437cf0b)", "backbone": "frozen (features cached once on H100)",
        "data": "ml-claims v1 train.jsonl fold=head (head), fold=temperature (T), fold=conformal (q_hat)",
    }, indent=2), encoding="utf-8")
    (RELEASE / "metrics.json").write_text(json.dumps({k: out[k] for k in ("parity", "splits", "selected", "contractnli_layout",
                                                                             "calibrations", "latency", "release")} |
                                                     {"results": res}, indent=2), encoding="utf-8")
    (run_dir / "stage2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.save(run_dir / "ours_logits.npy", our_logits)
    write_jsonl(run_dir / "ours_ids.jsonl", [{"id": r["id"]} for r in rows])
    print(json.dumps({"selected": out["selected"], "release": out["release"]}, indent=1))


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep-cnli")
    s1 = sub.add_parser("stage1")
    s1.add_argument("--run", default="sapling_v1")
    s2 = sub.add_parser("stage2")
    s2.add_argument("--run", default="sapling_v1")
    s2.add_argument("--threads", type=int, default=8)
    s2.add_argument("--grid", type=int, default=0, help="use only the first N grid configs (0 = all)")
    args = ap.parse_args(argv)
    {"prep-cnli": cmd_prep_cnli, "stage1": cmd_stage1, "stage2": cmd_stage2}[args.cmd](args)


if __name__ == "__main__":
    main()
