"""Banyan Sapling glue: pins, rows, calibration maths, parity and cost (pure Python + numpy).

Spec: ``docs/sapling_gate.md``. Decisions: D28 (stages 1-2 GO, ML-research-only calibration,
head-only delta, Sapling stays out of the live race loop, H100 cap $4).

Nothing here imports torch or modal. The torch side is ``workers.verifier.sapling_runtime``;
the Modal app is ``workers/runners/modal_sapling.py``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- package pins

REPO_ID = "MubitLabs/banyan-sapling"
REVISION = "437cf0b75fd52970464f9f22604a4b92ab0bec9f"
WEIGHTS_FILE = "model.safetensors"
WEIGHTS_BYTES = 46_247_530_352
# From the package's distribution_manifest.json at REVISION (also its package_validation.json).
WEIGHTS_SHA256 = "103a67f7a112589346c3e1090f5bca182d7fdafee85dfd43e67ac5d75eb04dd4"
SOURCE_RELEASE_SHA256 = "245970b0e5b0b22d64c1434ee21dd221b1f211f06bda5465f8fc108f70cd5c3f"

# Files at REVISION whose hash is NOT the one in the package's own distribution_manifest.json
# (observed 2026-10-24: requirements.txt was edited after the manifest was written; the
# package's own ``verify_files`` fails on it). We accept exactly the observed hash, nothing else.
MANIFEST_DRIFT_SHA256: dict[str, str] = {
    "requirements.txt": "620e7c7e6b0a330719bc5fe6cf162f8fb268bf9329f32d1daa01fc4fea84d207",
}
# Files we execute that the manifest does not list at all: pinned by their hash at REVISION.
UNLISTED_PINNED_SHA256: dict[str, str] = {
    "handler.py": "dcc3073011231096fe862a6eb0ee309a9e8bbd07e0135a7d9ed368d5d355e7e1",
}


def expected_sha256(manifest_files: dict[str, str]) -> dict[str, str]:
    """What the download must match: manifest hashes, overridden by the observed drift, plus
    the unlisted files we load. ``model.safetensors`` is never overridden."""
    out = {**manifest_files, **UNLISTED_PINNED_SHA256}
    for name, sha in MANIFEST_DRIFT_SHA256.items():
        if name in out:
            out[name] = sha
    if out.get(WEIGHTS_FILE) != WEIGHTS_SHA256:
        raise ValueError("manifest weights hash differs from the pinned WEIGHTS_SHA256")
    return out

# Exact runtime pins (requirements.txt + runtime_versions.json at REVISION). torch comes from
# PyPI, never the PyTorch index: inference.load_model checks exact versions ("2.14.1+cu130"
# would fail) and the PyPI Linux wheel pulls the CUDA 13 runtime (driver >= 580 needed).
RUNTIME_PINS: dict[str, str] = {
    "torch": "2.14.1",
    "transformers": "5.17.0",
    "safetensors": "0.8.0",
    "numpy": "2.5.4",
    "tokenizers": "0.23.3",
    "accelerate": "1.15.0",
    "huggingface-hub": "1.33.0",
}
MIN_DRIVER_MAJOR = 580  # CUDA 13

LABELS: tuple[str, ...] = ("supported", "violation", "insufficient_information", "inconsistent_context")
HIDDEN_SIZE = 2048  # backbone hidden size; the head consumes 3 pooled vectors of this size
N_SEGMENTS = 3  # policy, evidence, claim
MAX_TOKENS = 16_384  # encode() raises above this, never truncates
ROW_FIELDS = ("policy", "evidence", "claim")

# The package's own example (examples/policy_claim.json) and its H200 reference result
# (evidence/package_validation.json). Our first H100 run must reproduce these logits.
REFERENCE_EXAMPLE: dict[str, str] = {
    "policy": (
        "A production deployment is permitted only when the security review has been approved. "
        "A rejected security review prohibits deployment."
    ),
    "evidence": "The security review for deployment D-204 was rejected.",
    "claim": "Deployment D-204 is permitted.",
}
REFERENCE_LOGITS: tuple[float, ...] = (
    -10.317618370056152, 18.04765510559082, -8.824878692626953, -18.418155670166016,
)
REFERENCE_STATUS = "unsure"
REFERENCE_SET = ("violation", "insufficient_information")
REFERENCE_INPUT_TOKENS = 92
# H100 vs the H200 reference: same deterministic kernels are not guaranteed across GPU models,
# so we accept a small absolute drift and report the exact number.
REFERENCE_TOL = 5e-2

# --------------------------------------------------------------------------- prices (budget.md)

H100_PER_HOUR = 3.95
L4_PER_HOUR = 0.80
CPU_CORE_PER_HOUR = 0.047
RAM_GIB_PER_HOUR = 0.008
GPU_PER_HOUR: dict[str, float] = {"H100": H100_PER_HOUR, "L4": L4_PER_HOUR, "CPU": 0.0}


def estimate_cost(seconds: float, *, gpu: str = "H100", cpu_cores: float = 8.0, mem_gib: float = 64.0) -> float:
    """Billed $ for one container alive ``seconds`` (GPU + reserved CPU + RAM)."""
    hours = max(0.0, seconds) / 3600.0
    return hours * (GPU_PER_HOUR[gpu] + cpu_cores * CPU_CORE_PER_HOUR + mem_gib * RAM_GIB_PER_HOUR)


def estimate_extraction(n_cases: int, *, s_per_case: float, load_s: float, gpu: str = "H100",
                        cpu_cores: float = 8.0, mem_gib: float = 64.0) -> dict[str, float]:
    """Wall time and $ for one extraction run (one cold start + n_cases sequential forwards)."""
    seconds = load_s + n_cases * s_per_case
    return {"seconds": seconds, "minutes": seconds / 60.0,
            "usd": estimate_cost(seconds, gpu=gpu, cpu_cores=cpu_cores, mem_gib=mem_gib)}


# --------------------------------------------------------------------------- rows


class RowError(ValueError):
    """A (policy, evidence, claim) row Sapling would reject."""


def validate_row(row: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Return ``{"id", "policy", "evidence", "claim"}``; raise RowError on what encode() rejects.

    ``id`` is ours (kept out of the model request). Every other key is an error, as in
    Sapling's ``encode`` (exactly three non-empty strings).
    """
    if not isinstance(row, dict):
        raise RowError(f"row {index}: expected an object, got {type(row).__name__}")
    extra = set(row) - set(ROW_FIELDS) - {"id"}
    if extra:
        raise RowError(f"row {index}: unexpected keys {sorted(extra)}")
    out: dict[str, Any] = {"id": str(row.get("id", index))}
    for k in ROW_FIELDS:
        v = row.get(k)
        if not isinstance(v, str) or not v.strip():
            raise RowError(f"row {index} ({out['id']}): '{k}' must be a non-empty string")
        out[k] = v
    return out


def validate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [validate_row(r, i) for i, r in enumerate(rows)]
    ids = [r["id"] for r in out]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise RowError(f"duplicate row ids: {dupes[:5]}")
    return out


def project_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only ``id`` + the three model fields (drops labels and generator metadata, e.g.
    ``data/ml_claims/*.jsonl``), so labels never travel to the GPU. Join back on ``id``."""
    return [{k: r[k] for k in ("id", *ROW_FIELDS) if k in r} for r in rows]


def request_of(row: dict[str, Any]) -> dict[str, str]:
    """The exact request Sapling's ``encode`` accepts (the three fields, no id)."""
    return {k: row[k] for k in ROW_FIELDS}


def chunk_rows(rows: Sequence[dict[str, Any]], size: int) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for i, start in enumerate(range(0, len(rows), size)):
        yield i, list(rows[start:start + size])


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, allow_nan=False) + "\n")


# --------------------------------------------------------------------------- output layout


def run_dir(run_name: str) -> str:
    """Volume-relative directory of one extraction run."""
    if not run_name or any(c in run_name for c in "/\\") or run_name.startswith("."):
        raise ValueError(f"bad run name: {run_name!r}")
    return f"outputs/{run_name}"


def chunk_paths(run_name: str, chunk: int) -> dict[str, str]:
    """Volume-relative file refs for one chunk: features [n,3,2048] f32, logits [n,4] f32, rows."""
    base = f"{run_dir(run_name)}/chunk_{chunk:05d}"
    return {"features": f"{base}_features.npy", "logits": f"{base}_logits.npy", "rows": f"{base}_rows.jsonl"}


HEAD_DIR = "outputs/stock_head"
HEAD_FILE = f"{HEAD_DIR}/head.safetensors"  # model.head.state_dict(), fp32
CALIBRATION_FILE = f"{HEAD_DIR}/calibration.json"


# --------------------------------------------------------------------------- calibration (numpy)


def load_calibration(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    state = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    if state.get("labels") != list(LABELS):
        raise ValueError("calibration labels do not match Sapling's label order")
    t, q = state.get("temperature"), state.get("q_hat")
    if not (isinstance(t, (int, float)) and math.isfinite(t) and t > 0):
        raise ValueError("calibration temperature must be a positive finite number")
    if not (isinstance(q, (int, float)) and 0 <= q <= 1):
        raise ValueError("calibration q_hat must be in [0, 1]")
    return state


def log_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Same arithmetic as ConformalCalibrator._log_probabilities (float64)."""
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    with np.errstate(over="ignore", under="ignore"):
        shifted = (arr - np.max(arr, axis=1, keepdims=True)) / temperature
        return shifted - np.logaddexp.reduce(shifted, axis=1, keepdims=True)


def calibrate(logits: np.ndarray | Sequence[Sequence[float]], calibration: dict[str, Any],
              *, empty_set_policy: str = "rounded_argmax") -> list[dict[str, Any]]:
    """Numpy re-implementation of ``ConformalCalibrator.predict_detailed`` + probabilities.

    Used off-GPU to score a retrained head on cached features with any calibration state
    (stock or ours). The H100 path still uses Sapling's own calibrator; ``tests`` and the
    smoke compare the two.
    """
    if empty_set_policy not in ("strict", "rounded_argmax"):
        raise ValueError("empty_set_policy must be 'strict' or 'rounded_argmax'")
    cal = load_calibration(calibration)
    t, q = float(cal["temperature"]), float(cal["q_hat"])
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(LABELS) or not np.isfinite(arr).all():
        raise ValueError("logits must be finite with shape [n, 4]")
    log_probs = log_probabilities(arr, t)
    scores = -np.expm1(log_probs)
    out: list[dict[str, Any]] = []
    for row, log_row, score_row in zip(arr, log_probs, scores):
        labels = [lab for lab, inc in zip(LABELS, score_row <= q) if inc]
        argmax = int(np.argmax(row))
        fallback = bool(not labels and empty_set_policy == "rounded_argmax"
                        and 1.0 - np.exp(log_row[argmax]) <= q < score_row[argmax])
        selected = [LABELS[argmax]] if fallback else list(labels)
        provenance = ("rounded_empty_argmax" if fallback else
                      "conformal_singleton" if len(labels) == 1 else
                      "empty_conformal_set" if not labels else "multiple_conformal_labels")
        out.append({
            "conformal_prediction_set": labels,
            "prediction_set": selected,
            "status": selected[0] if len(selected) == 1 else "unsure",
            "raw_label": LABELS[argmax],
            "fallback_used": fallback,
            "fallback_reason": "float64_subtraction_rounding" if fallback else None,
            "status_provenance": provenance,
            "empty_set_policy": empty_set_policy,
            "conformal_singleton": len(labels) == 1,
            "probabilities": dict(zip(LABELS, np.exp(log_row).tolist())),
        })
    return out


# --------------------------------------------------------------------------- parity checks


def max_abs_diff(a: np.ndarray | Sequence[Any], b: np.ndarray | Sequence[Any]) -> float:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {x.shape} vs {y.shape}")
    return float(np.max(np.abs(x - y))) if x.size else 0.0


def parity_report(stock_logits: Any, recomputed_logits: Any, *, tol: float = 1e-5) -> dict[str, Any]:
    """Original head on captured features vs stock logits from the same forward pass."""
    d = max_abs_diff(stock_logits, recomputed_logits)
    return {"max_abs_diff": d, "exact": d == 0.0, "tol": tol, "ok": d <= tol}


def reference_report(logits: Sequence[float], *, tol: float = REFERENCE_TOL) -> dict[str, Any]:
    """Our run of the package example vs the published H200 logits."""
    d = max_abs_diff(list(logits), list(REFERENCE_LOGITS))
    return {"max_abs_diff": d, "tol": tol, "ok": d <= tol, "reference": list(REFERENCE_LOGITS),
            "observed": [float(v) for v in logits]}


def driver_major(nvidia_smi_version: str) -> int:
    """'580.65.06' -> 580."""
    return int(nvidia_smi_version.strip().split(".")[0])
