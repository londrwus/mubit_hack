"""Banyan Sapling on Modal: weights download, H100 feature extraction (D28, docs/sapling_gate.md).

A separate Modal app (``busy-machines-sapling``) so it never touches the race runner
(``busy-machines-runner``). One of the two modules allowed to import ``modal``
(rule: ``modal`` only in ``workers/runners/modal_*.py``, D28).

Pieces
- Volume ``sapling-weights`` (v1) mounted at ``/weights``. The package lives at
  ``/weights/banyan-sapling/<REVISION>``; extraction outputs under ``/weights/outputs/``.
- ``smoke_env`` (L4, cents): the GPU image builds, pinned versions import, ``nvidia-smi``
  driver >= 580 (CUDA 13), a tiny ``qwen3_moe`` with ``grouped_mm`` runs a forward.
- ``download_weights`` (CPU 8 cores / 64 GiB, 2 h timeout): ``snapshot_download`` at the
  pinned revision into the Volume, SHA-256 of every manifest file ONCE, then a marker file.
  Never on the H100 clock.
- ``h100_batch`` (H100, 60 min timeout, max 1 container, short scaledown, no retries): loads
  via ``handler.load_model_for_endpoint`` (skips the 46 GB per-boot checksum and the exact
  version check; the marker proves the checksum), ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` in the
  image env, forward pre-hook on ``model.head`` -> pooled features. One call = one chunk of
  rows: stock logits + calibrated output per row, features/logits saved as ``.npy`` in the
  Volume and returned as file refs. Optional: the package example vs the H200 reference
  logits, the head parity check, and a one-time dump of ``model.head.state_dict()`` +
  ``calibration.json``. The model is loaded lazily (not in ``@enter``) so a failed load fails
  the call instead of crash-looping an H100 container.

CLI (deploy once: ``modal deploy workers/runners/modal_sapling.py``)
    python -m workers.runners.modal_sapling smoke                  # L4 env smoke (ephemeral app)
    python -m workers.runners.modal_sapling download               # spawn on the deployed app, prints call id
    python -m workers.runners.modal_sapling wait <call_id> [timeout_s]
    python -m workers.runners.modal_sapling status                 # is the verified package in the Volume?
    python -m workers.runners.modal_sapling h100-smoke [deadline_s]  # reference + 5 rows + parity + head dump
    python -m workers.runners.modal_sapling extract <rows.jsonl> [more.jsonl ...] <run_name> [--chunk N] [--max-usd X] [--out DIR]
    python -m workers.runners.modal_sapling fetch <volume_path> <local_path>

Rows are JSONL objects with ``id``, ``policy``, ``evidence``, ``claim``; any other keys
(labels, generator metadata in ``data/ml_claims/*.jsonl``) are dropped before upload
(``project_rows``) and joined back on ``id`` locally. Sapling verifies offline only (D28 (6)); it is not
an LLM call, so nothing here goes through Mubit.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal

from workers.verifier.sapling import (
    CALIBRATION_FILE,
    HEAD_DIR,
    HEAD_FILE,
    MANIFEST_DRIFT_SHA256,
    MIN_DRIVER_MAJOR,
    REFERENCE_EXAMPLE,
    REPO_ID,
    REVISION,
    RUNTIME_PINS,
    UNLISTED_PINNED_SHA256,
    WEIGHTS_BYTES,
    WEIGHTS_FILE,
    WEIGHTS_SHA256,
    chunk_paths,
    chunk_rows,
    driver_major,
    estimate_cost,
    estimate_extraction,
    expected_sha256,
    project_rows,
    read_jsonl,
    reference_report,
    validate_rows,
    write_jsonl,
)

APP_NAME = "busy-machines-sapling"
VOLUME_NAME = "sapling-weights"
MOUNT = "/weights"
MODEL_SUBDIR = f"banyan-sapling/{REVISION}"
MODEL_DIR = f"{MOUNT}/{MODEL_SUBDIR}"
MARKER_NAME = ".busy_verified.json"
MARKER = f"{MODEL_DIR}/{MARKER_NAME}"

H100_CPU = 8.0
H100_MEMORY_MB = 32 * 1024  # safetensors mmaps the file and copies tensors straight to CUDA
DOWNLOAD_CPU = 8.0
DOWNLOAD_MEMORY_MB = 64 * 1024
H100_TIMEOUT_S = 3600
DOWNLOAD_TIMEOUT_S = 2 * 3600
SCALEDOWN_S = 30  # idle H100 seconds are billed: keep the warm window short
DEFAULT_CHUNK = 500
DEFAULT_MAX_USD = 4.0  # D28 H100 cap for all Sapling work

GPU_PIP: list[str] = [f"{k}=={v}" for k, v in RUNTIME_PINS.items()]  # torch from PyPI (CUDA 13)
DOWNLOAD_PIP: list[str] = [f"huggingface-hub=={RUNTIME_PINS['huggingface-hub']}",
                           f"numpy=={RUNTIME_PINS['numpy']}"]
# ``workers.runners`` imports ``backend.core.types`` (pydantic) at package import, so both
# images carry the ``backend`` source and pydantic, as in modal_runner.py.
EXTRA_PIP: list[str] = ["pydantic>=2"]
LOCAL_SOURCES: tuple[str, ...] = ("workers", "backend")
GPU_ENV: dict[str, str] = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",  # seed_all enables deterministic algorithms; handler.py does not set this
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
}
DOWNLOAD_ENV: dict[str, str] = {"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*GPU_PIP, *EXTRA_PIP)
    .env(GPU_ENV)
    .add_local_python_source(*LOCAL_SOURCES)
)
download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*DOWNLOAD_PIP, *EXTRA_PIP)
    .env(DOWNLOAD_ENV)
    .add_local_python_source(*LOCAL_SOURCES)
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

_CONTAINER_T0 = time.time()  # module import time inside the container ~ container start


# --------------------------------------------------------------------------- container side


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(64 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_marker() -> dict[str, Any] | None:
    p = Path(MARKER)
    return json.loads(p.read_text()) if p.exists() else None


def _driver() -> dict[str, Any]:
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        name, ver, mem = (s.strip() for s in out.splitlines()[0].split(","))
        return {"gpu": name, "driver": ver, "driver_ok": driver_major(ver) >= MIN_DRIVER_MAJOR, "memory": mem}
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc), "driver_ok": False}


def _versions() -> dict[str, Any]:
    import importlib.metadata as md

    got = {k: md.version(k) for k in RUNTIME_PINS}
    return {"installed": got, "pins_ok": got == RUNTIME_PINS, "python": sys.version.split()[0]}


@app.function(image=gpu_image, gpu="L4", timeout=900, scaledown_window=2, max_containers=1, retries=0)
def smoke_env() -> dict[str, Any]:
    """L4: image builds, pinned imports, driver >= 580, tiny qwen3_moe forward with grouped_mm."""
    t0 = time.time()
    out: dict[str, Any] = {"driver": _driver(), "versions": _versions()}
    import torch

    out["torch_cuda"] = {"available": torch.cuda.is_available(), "cuda": torch.version.cuda,
                         "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None}
    try:
        from transformers import AutoModel, Qwen3MoeConfig

        cfg = Qwen3MoeConfig(vocab_size=256, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                             num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1,
                             max_position_embeddings=256)
        torch.use_deterministic_algorithms(True)
        m = AutoModel.from_config(cfg, dtype=torch.bfloat16, attn_implementation="sdpa",
                                  experts_implementation="grouped_mm").cuda().eval()
        ids = torch.randint(0, 256, (1, 32), device="cuda")
        with torch.inference_mode():
            h = m(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        out["tiny_qwen3_moe_grouped_mm"] = {"ok": bool(torch.isfinite(h.float()).all()), "shape": list(h.shape)}
    except Exception as exc:  # noqa: BLE001 (sm89 may lack grouped_mm; informative, H100 is the real test)
        out["tiny_qwen3_moe_grouped_mm"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:800]}
    out["seconds"] = time.time() - t0
    out["container_age_s"] = time.time() - _CONTAINER_T0
    return out


@app.function(image=download_image, volumes={MOUNT: volume}, cpu=DOWNLOAD_CPU, memory=DOWNLOAD_MEMORY_MB,
              timeout=DOWNLOAD_TIMEOUT_S, max_containers=1, retries=0)
def download_weights(force: bool = False) -> dict[str, Any]:
    """Download the package at REVISION into the Volume and SHA-256 it once (writes MARKER)."""
    from huggingface_hub import snapshot_download

    volume.reload()
    marker = _read_marker()
    if marker and not force:
        return {"skipped": True, **marker}
    root = Path(MODEL_DIR)
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    snapshot_download(repo_id=REPO_ID, revision=REVISION, local_dir=str(root), max_workers=8)
    t_dl = time.time() - t0
    print(f"[sapling] downloaded in {t_dl:.0f}s", flush=True)
    volume.commit()

    t1 = time.time()
    manifest = json.loads((root / "distribution_manifest.json").read_text())
    if manifest.get("model_id") != "banyan-sapling" or manifest.get("format_version") != 1:
        raise RuntimeError("unexpected distribution manifest")
    expected_all = expected_sha256(manifest["files"])  # raises if the weights hash is not the pinned one
    size = (root / WEIGHTS_FILE).stat().st_size
    if size != WEIGHTS_BYTES:
        raise RuntimeError(f"{WEIGHTS_FILE} is {size} bytes, expected {WEIGHTS_BYTES}")
    bad: dict[str, str] = {}
    for name, expected in sorted(expected_all.items()):
        p = (root / name).resolve()
        got = _sha256(p) if p.is_relative_to(root.resolve()) and p.is_file() else "missing"
        if got != expected:
            bad[name] = got
    t_hash = time.time() - t1
    if bad:
        raise RuntimeError(f"checksum mismatch or missing: {bad}")
    marker = {"repo_id": REPO_ID, "revision": REVISION, "weights_sha256": WEIGHTS_SHA256,
              "weights_bytes": size, "verified_files": len(expected_all),
              "manifest_drift_accepted": MANIFEST_DRIFT_SHA256, "unlisted_pinned": UNLISTED_PINNED_SHA256,
              "download_s": t_dl, "sha256_s": t_hash, "verified_at": time.time()}
    Path(MARKER).write_text(json.dumps(marker, indent=2))
    volume.commit()
    return {"skipped": False, **marker}


@app.function(image=download_image, volumes={MOUNT: volume}, cpu=0.25, timeout=120, retries=0)
def volume_status() -> dict[str, Any]:
    volume.reload()
    root = Path(MODEL_DIR)
    w = root / WEIGHTS_FILE
    return {"marker": _read_marker(), "weights_present": w.exists(),
            "weights_bytes": w.stat().st_size if w.exists() else 0,
            "head_dumped": (Path(MOUNT) / HEAD_FILE).exists()}


_STATE: dict[str, Any] = {}


def _load() -> dict[str, Any]:
    """Load once per container (lazy, so a failure fails the call rather than crash-looping)."""
    if _STATE:
        return _STATE
    t0 = time.time()
    volume.reload()
    if _read_marker() is None:
        raise RuntimeError(f"no verified package at {MODEL_DIR}: run download_weights first")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 (deterministic cuBLAS)")
    if MODEL_DIR not in sys.path:
        sys.path.insert(0, MODEL_DIR)
    import handler  # the package's own loader (Apache-2.0/MIT code shipped with the weights)
    from runtime.backbone import encode
    from runtime.model import infer

    model, tokenizer, calibrator, manifest = handler.load_model_for_endpoint(Path(MODEL_DIR), device="cuda")
    import torch

    torch.cuda.synchronize()
    _STATE.update(model=model, tokenizer=tokenizer, calibrator=calibrator, manifest=manifest,
                  encode=encode, infer=infer, handler=handler,
                  load_s=time.time() - t0, loaded_at=time.time(),
                  gpu_mem_gib=torch.cuda.memory_allocated() / 2**30)
    return _STATE


@app.function(image=gpu_image, gpu="H100", volumes={MOUNT: volume}, cpu=H100_CPU, memory=H100_MEMORY_MB,
              timeout=H100_TIMEOUT_S, scaledown_window=SCALEDOWN_S, max_containers=1, retries=0)
def h100_batch(rows: list[dict[str, Any]], run_name: str, chunk: int = 0, *, reference: bool = False,
               parity: bool = True, dump_head: bool = False) -> dict[str, Any]:
    """One chunk of (policy, evidence, claim) rows -> stock logits, calibrated output, pooled features."""
    import numpy as np

    from workers.verifier import sapling_runtime as rt

    t_call = time.time()
    warm = bool(_STATE)
    st = _load()
    out: dict[str, Any] = {
        "warm": warm, "load_s": None if warm else st["load_s"],
        "container_age_at_call_s": t_call - _CONTAINER_T0, "gpu_mem_gib": st["gpu_mem_gib"],
        "driver": _driver(), "versions": _versions(),
    }
    model, tok, cal = st["model"], st["tokenizer"], st["calibrator"]

    if reference:
        res = st["handler"].predict_for_endpoint(model, tok, cal, st["manifest"], dict(REFERENCE_EXAMPLE))
        out["reference"] = {**reference_report(res["logits"]), "status": res["status"],
                            "prediction_set": res["prediction_set"], "input_tokens": res["usage"]["input_tokens"],
                            "model_ms": res["timing_ms"]["model"]}

    rows = validate_rows(rows)
    t0 = time.time()
    feats, logits, records = rt.extract_rows(model, tok, cal, rows, encode=st["encode"], infer=st["infer"])
    t_ext = time.time() - t0
    n_ok = int(len(feats))
    out["extract"] = {"rows": len(rows), "ok": n_ok, "errors": len(rows) - n_ok, "seconds": t_ext,
                      "s_per_case": t_ext / max(1, len(rows)),
                      "model_ms_median": float(np.median([r["model_ms"] for r in records if "model_ms" in r]))
                      if n_ok else None}

    refs = chunk_paths(run_name, chunk)
    base = Path(MOUNT)
    (base / refs["features"]).parent.mkdir(parents=True, exist_ok=True)
    np.save(base / refs["features"], feats)
    np.save(base / refs["logits"], logits)
    write_jsonl(base / refs["rows"], records)
    out["files"] = {"volume": VOLUME_NAME, **refs, "features_shape": list(feats.shape),
                    "features_bytes": int(feats.nbytes)}

    if parity and n_ok:
        out["parity"] = rt.head_parity(model.head, feats, logits, device="cuda")

    if dump_head and not (base / HEAD_FILE).exists():
        (base / HEAD_DIR).mkdir(parents=True, exist_ok=True)
        shapes = rt.dump_head(model, str(base / HEAD_FILE))
        (base / CALIBRATION_FILE).write_text(json.dumps(cal.to_dict(), indent=2))
        out["head_dump"] = {"file": HEAD_FILE, "calibration": CALIBRATION_FILE, "shapes": shapes,
                            "bytes": (base / HEAD_FILE).stat().st_size}
    volume.commit()
    out["records"] = records
    out["call_s"] = time.time() - t_call
    return out


# --------------------------------------------------------------------------- client side


def _fn(name: str) -> modal.Function:
    return modal.Function.from_name(APP_NAME, name)


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _summary(res: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in res.items() if k != "records"}


def _get_with_deadline(fc: modal.FunctionCall, deadline_s: float) -> Any:
    try:
        return fc.get(timeout=deadline_s)
    except TimeoutError:
        print(f"[sapling] deadline {deadline_s:.0f}s hit: cancelling {fc.object_id} and its container", file=sys.stderr)
        fc.cancel(terminate_containers=True)
        raise


def cmd_smoke(deadline_s: float = 900.0) -> dict[str, Any]:
    """Ephemeral app; the deadline stops a crash-looping container (import errors do not respect retries=0)."""
    with modal.enable_output(), app.run():
        t0 = time.time()
        res = _get_with_deadline(smoke_env.spawn(), deadline_s)
        res["client_wall_s"] = time.time() - t0
    return res


def cmd_h100_smoke(deadline_s: float = 660.0) -> dict[str, Any]:
    """Reference example + 5 rows + parity + head dump, under a client-side deadline (cost cap)."""
    rows = [{"id": f"smoke-{i}", **r} for i, r in enumerate(_smoke_rows())]
    t0 = time.time()
    fc = _fn("h100_batch").spawn(rows, "smoke", 0, reference=True, parity=True, dump_head=True)
    print(f"[sapling] h100 smoke call {fc.object_id}", file=sys.stderr)
    res = _get_with_deadline(fc, deadline_s)
    wall = time.time() - t0
    res["client_wall_s"] = wall
    res["est_usd_upper"] = estimate_cost(wall + SCALEDOWN_S, gpu="H100", cpu_cores=H100_CPU,
                                         mem_gib=H100_MEMORY_MB / 1024)
    return res


def _smoke_rows() -> list[dict[str, str]]:
    """Five short ML-research claims (smoke only; not part of any eval set)."""
    p = ("A reported improvement counts only if it is measured on a held-out test set that is disjoint "
         "from the training data and the gain exceeds run-to-run noise across seeds.")
    return [
        dict(REFERENCE_EXAMPLE),
        {"policy": p, "evidence": "Validation accuracy rose from 93.1% to 94.0% over 5 seeds (std 0.1). "
                                  "The test split shares no images with the training split.",
         "claim": "The new augmentation improves test accuracy."},
        {"policy": p, "evidence": "Test accuracy rose from 93.1% to 96.8% after adding the test images to the "
                                  "training set.",
         "claim": "The new augmentation improves test accuracy."},
        {"policy": p, "evidence": "Loss went from 3.281 to 3.279 on one seed; seed-to-seed std is 0.02.",
         "claim": "The new optimiser lowers validation loss."},
        {"policy": p, "evidence": "The summary table reports a return of 412, while the log for the same run "
                                  "reports a final return of 288.",
         "claim": "The agent reaches a return of 412."},
    ]


def cmd_extract(rows_paths: str | list[str], run_name: str, *, chunk: int = DEFAULT_CHUNK,
                max_usd: float = DEFAULT_MAX_USD, out_dir: str | None = None, s_per_case: float = 0.15,
                load_s: float = 120.0) -> dict[str, Any]:
    """Extract one or more JSONL files in ONE run (one cold start); ids must be unique across files.

    Estimate defaults are conservative vs the H100 smoke (load 44 s, ~0.07-0.16 s/case at ~120 tokens).
    """
    paths = [rows_paths] if isinstance(rows_paths, str) else list(rows_paths)
    rows = validate_rows(project_rows([r for p in paths for r in read_jsonl(p)]))
    est = estimate_extraction(len(rows), s_per_case=s_per_case, load_s=load_s, cpu_cores=H100_CPU,
                              mem_gib=H100_MEMORY_MB / 1024)
    print(f"[sapling] {len(rows)} rows, estimate {est['minutes']:.1f} min / ${est['usd']:.2f} (cap ${max_usd:.2f})",
          file=sys.stderr)
    if est["usd"] > max_usd:
        raise SystemExit(f"estimated ${est['usd']:.2f} exceeds --max-usd {max_usd:.2f}")
    fn = _fn("h100_batch")
    local = Path(out_dir or f"data/sapling/{run_name}")
    local.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summaries: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for i, part in chunk_rows(rows, chunk):
        # Re-check the real spend before each chunk (wall clock so far ~ billed container time).
        spent = estimate_cost(time.time() - t0, gpu="H100", cpu_cores=H100_CPU, mem_gib=H100_MEMORY_MB / 1024)
        if spent > max_usd:
            raise SystemExit(f"stopping before chunk {i}: ~${spent:.2f} spent >= cap ${max_usd:.2f}")
        res = _get_with_deadline(fn.spawn(part, run_name, i, reference=(i == 0), parity=True, dump_head=(i == 0)),
                                 H100_TIMEOUT_S)
        all_records.extend(res.pop("records"))
        summaries.append(res)
        e = res["extract"]
        print(f"[sapling] chunk {i}: {e['ok']}/{e['rows']} ok, {e['s_per_case']:.3f} s/case, "
              f"parity {res.get('parity', {}).get('same_device_b1', {}).get('max_abs_diff')}", file=sys.stderr)
    wall = time.time() - t0
    write_jsonl(local / "records.jsonl", all_records)
    report = {"run_name": run_name, "inputs": paths, "rows": len(rows), "chunks": summaries, "client_wall_s": wall,
              "est_usd_upper": estimate_cost(wall + SCALEDOWN_S, gpu="H100", cpu_cores=H100_CPU,
                                             mem_gib=H100_MEMORY_MB / 1024),
              "fetch": f"python -m workers.runners.modal_sapling fetch outputs/{run_name} {local}/volume"}
    (local / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def cmd_fetch(volume_path: str, local_path: str) -> list[str]:
    """Copy a file or directory from the Volume to the laptop."""
    vol = modal.Volume.from_name(VOLUME_NAME)
    got: list[str] = []
    entries = vol.listdir(volume_path, recursive=True)
    files = [e.path for e in entries if e.type == modal.volume.FileEntryType.FILE] or [volume_path]
    for p in files:
        rel = os.path.relpath(p, volume_path) if p != volume_path else os.path.basename(p)
        dest = Path(local_path) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            vol.read_file_into_fileobj(p, f)
        got.append(str(dest))
    return got


def main(argv: list[str]) -> None:
    if not argv:
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "smoke":
        _print(cmd_smoke())
    elif cmd == "download":
        fc = _fn("download_weights").spawn(force="--force" in rest)
        print(fc.object_id)
    elif cmd == "wait":
        _print(modal.FunctionCall.from_id(rest[0]).get(timeout=float(rest[1]) if len(rest) > 1 else None))
    elif cmd == "status":
        _print(_fn("volume_status").remote())
    elif cmd == "h100-smoke":
        res = cmd_h100_smoke(float(rest[0]) if rest else 660.0)
        _print(res)
    elif cmd == "extract":
        opts = {"chunk": DEFAULT_CHUNK, "max_usd": DEFAULT_MAX_USD, "out_dir": None}
        pos: list[str] = []
        it = iter(rest)
        for a in it:
            if a == "--chunk":
                opts["chunk"] = int(next(it))
            elif a == "--max-usd":
                opts["max_usd"] = float(next(it))
            elif a == "--out":
                opts["out_dir"] = next(it)
            else:
                pos.append(a)
        if len(pos) < 2:
            raise SystemExit("usage: extract <rows.jsonl> [<rows.jsonl> ...] <run_name> [options]")
        _print(_summary(cmd_extract(pos[:-1], pos[-1], **opts)))
    elif cmd == "fetch":
        _print(cmd_fetch(rest[0], rest[1]))
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
