"""Container-side torch helpers for Banyan Sapling (no modal; torch imported lazily).

The feature hook (``docs/sapling_gate.md`` section 1): a forward pre-hook on ``model.head``
receives exactly the three post-layer-norm fp32 vectors (policy, evidence, claim) the head
consumes, so the cached features are bit-identical to the stock path and the stock logits
come from the same forward pass. Re-applying the original head to the cached features must
reproduce those logits (the parity check that lets head training run off-GPU).

Sapling's own ``encode`` / ``infer`` are passed in (they live in the downloaded package),
which keeps this module importable and testable without the 46 GB package.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np

from workers.verifier.sapling import HIDDEN_SIZE, LABELS, MAX_TOKENS, N_SEGMENTS, parity_report, request_of


@contextlib.contextmanager
def capture_head_inputs(model: Any) -> Iterator[list[tuple[Any, ...]]]:
    """Yield a list that receives a CPU fp32 copy of the head's inputs on every forward."""
    store: list[tuple[Any, ...]] = []

    def hook(_module: Any, args: tuple[Any, ...]) -> None:
        store.append(tuple(a.detach().float().cpu().clone() for a in args))

    handle = model.head.register_forward_pre_hook(hook)
    try:
        yield store
    finally:
        handle.remove()


def extract_rows(
    model: Any,
    tokenizer: Any,
    calibrator: Any,
    rows: Sequence[dict[str, Any]],
    *,
    encode: Callable[..., dict[str, Any]],
    infer: Callable[..., Any],
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """One stock forward per row (batch 1, as Sapling's ``infer``), capturing pooled features.

    Returns ``features`` [n_ok, 3, H] float32, ``logits`` [n_ok, 4] float32 and one record per
    input row (``feature_index`` is None for a row Sapling rejected, with the error kept).
    """
    import torch

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    feats: list[np.ndarray] = []
    logits_out: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    with capture_head_inputs(model) as store:
        for row in rows:
            rec: dict[str, Any] = {"id": row["id"], "feature_index": None}
            try:
                enc = encode(tokenizer, request_of(row), MAX_TOKENS)
            except ValueError as exc:
                rec["error"] = f"encode: {exc}"
                records.append(rec)
                continue
            store.clear()
            t0 = time.perf_counter()
            logits = infer(model, [enc], pad, device)[0]  # float32 cpu [4]
            ms = (time.perf_counter() - t0) * 1000.0
            if len(store) != 1 or len(store[0]) != N_SEGMENTS:
                raise RuntimeError(f"head hook fired {len(store)} times / {len(store[0]) if store else 0} args")
            f = torch.cat(store[0], dim=0)  # [3, H]
            if tuple(f.shape) != (N_SEGMENTS, getattr(model.head, "hidden_dim", HIDDEN_SIZE)):
                raise RuntimeError(f"unexpected feature shape {tuple(f.shape)}")
            detail = calibrator.predict_detailed(logits.reshape(1, len(LABELS)), empty_set_policy="rounded_argmax")[0]
            probs = torch.softmax(logits.double() / calibrator.temperature, -1).tolist()
            rec.update({
                "feature_index": len(feats),
                "input_tokens": len(enc["input_ids"]),
                "model_ms": ms,
                "logits": [float(v) for v in logits.tolist()],
                **detail,
                "probabilities": dict(zip(LABELS, probs)),
            })
            feats.append(f.numpy().astype(np.float32, copy=False))
            logits_out.append(logits.float().numpy())
            records.append(rec)
    h = getattr(model.head, "hidden_dim", HIDDEN_SIZE)
    features = np.stack(feats) if feats else np.zeros((0, N_SEGMENTS, h), np.float32)
    logits_arr = np.stack(logits_out).astype(np.float32) if logits_out else np.zeros((0, len(LABELS)), np.float32)
    return features, logits_arr, records


def head_logits(head: Any, features: np.ndarray, *, device: str = "cpu", batch_size: int = 1) -> np.ndarray:
    """Apply a head to cached features [n, 3, H] -> logits [n, 4] (eval mode, no dropout)."""
    import torch

    was_training = head.training
    head.eval()
    out: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for s in range(0, len(features), batch_size):
                x = torch.from_numpy(np.ascontiguousarray(features[s:s + batch_size])).to(device=device, dtype=torch.float32)
                y = head(x[:, 0], x[:, 1], x[:, 2])
                out.append(y.float().cpu().numpy())
    finally:
        head.train(was_training)
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)), np.float32)


def head_parity(head: Any, features: np.ndarray, stock_logits: np.ndarray, *, device: str = "cuda",
                tol: float = 1e-5) -> dict[str, Any]:
    """Original head on captured features vs stock logits.

    ``same_device_b1``: on the stock device at batch 1 (the stock path) -- expect exact.
    ``cpu_batched``: on CPU in one batch (how head training / eval will run off-GPU).
    """
    import copy

    import torch

    same = head_logits(head, features, device=device, batch_size=1)
    cpu_head = copy.deepcopy(head).to("cpu")
    cpu = head_logits(cpu_head, features, device="cpu", batch_size=max(1, len(features)))
    del cpu_head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"same_device_b1": parity_report(stock_logits, same, tol=tol),
            "cpu_batched": parity_report(stock_logits, cpu, tol=1e-3)}


def dump_head(model: Any, path: str) -> dict[str, Any]:
    """Save ``model.head.state_dict()`` (fp32) as safetensors; return names and shapes."""
    from safetensors.torch import save_file

    state = {k: v.detach().float().cpu().contiguous() for k, v in model.head.state_dict().items()}
    save_file(state, path)
    return {k: list(v.shape) for k, v in state.items()}
