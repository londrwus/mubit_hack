"""Sapling glue (workers/verifier): rows, calibration maths vs the package's published example,
parity/cost helpers, the feature hook on a fake model, and the Modal app's static config."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from workers.verifier import sapling as sp

ROOT = Path(__file__).resolve().parents[1]

# evidence/package_validation.json at REVISION (H200 reference)
REF_PROBS = {
    "supported": 0.004105873121410999,
    "violation": 0.9895570416934399,
    "insufficient_information": 0.0054797604562215755,
    "inconsistent_context": 0.0008573247289275715,
}
STOCK_CAL = {
    "version": 2, "labels": list(sp.LABELS), "alpha": 0.01, "temperature_floor": 0.5,
    "temperature": 5.1715781469095745, "q_hat": 0.995477654430865,
}


# --------------------------------------------------------------------------- rows


def test_validate_rows_accepts_ids_and_rejects_what_encode_rejects() -> None:
    rows = sp.validate_rows([{"id": "a", **sp.REFERENCE_EXAMPLE}, dict(sp.REFERENCE_EXAMPLE)])
    assert [r["id"] for r in rows] == ["a", "1"]
    assert sp.request_of(rows[0]) == sp.REFERENCE_EXAMPLE
    with pytest.raises(sp.RowError, match="unexpected keys"):
        sp.validate_row({**sp.REFERENCE_EXAMPLE, "label": "violation"})
    with pytest.raises(sp.RowError, match="non-empty"):
        sp.validate_row({**sp.REFERENCE_EXAMPLE, "claim": "  "})
    with pytest.raises(sp.RowError, match="duplicate"):
        sp.validate_rows([{"id": "x", **sp.REFERENCE_EXAMPLE}] * 2)


def test_project_rows_drops_labels_and_metadata() -> None:
    raw = [{"id": "m1", "split": "test", "label": "violation", "hidden": {"x": 1}, **sp.REFERENCE_EXAMPLE}]
    rows = sp.validate_rows(sp.project_rows(raw))
    assert rows == [{"id": "m1", **sp.REFERENCE_EXAMPLE}]


def test_chunking_and_paths() -> None:
    rows = [{"id": str(i)} for i in range(7)]
    chunks = list(sp.chunk_rows(rows, 3))
    assert [i for i, _ in chunks] == [0, 1, 2] and [len(c) for _, c in chunks] == [3, 3, 1]
    p = sp.chunk_paths("run1", 2)
    assert p["features"] == "outputs/run1/chunk_00002_features.npy"
    with pytest.raises(ValueError):
        sp.run_dir("../x")


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    rows = [{"id": "1", **sp.REFERENCE_EXAMPLE}]
    sp.write_jsonl(tmp_path / "a" / "r.jsonl", rows)
    assert sp.read_jsonl(tmp_path / "a" / "r.jsonl") == rows


# --------------------------------------------------------------------------- calibration


def test_calibrate_reproduces_the_published_example() -> None:
    d = sp.calibrate(sp.REFERENCE_LOGITS, STOCK_CAL)[0]
    assert d["status"] == "unsure" == sp.REFERENCE_STATUS
    assert d["conformal_prediction_set"] == list(sp.REFERENCE_SET)
    assert d["raw_label"] == "violation" and d["status_provenance"] == "multiple_conformal_labels"
    for k, v in REF_PROBS.items():
        assert d["probabilities"][k] == pytest.approx(v, rel=1e-9)


def test_calibrate_singleton_and_ml_only_threshold() -> None:
    big_margin = [-40.0, 40.0, -40.0, -40.0]
    assert sp.calibrate(big_margin, STOCK_CAL)[0]["status"] == "violation"
    # An ML-only calibration with a lower q_hat answers the reference example.
    ours = {**STOCK_CAL, "q_hat": 0.05}
    assert sp.calibrate(sp.REFERENCE_LOGITS, ours)[0]["status"] == "violation"


def test_calibrate_rejects_bad_state_and_logits() -> None:
    with pytest.raises(ValueError):
        sp.calibrate([0, 0, 0, 0], {**STOCK_CAL, "labels": ["a", "b", "c", "d"]})
    with pytest.raises(ValueError):
        sp.calibrate([0, 0, 0], STOCK_CAL)
    with pytest.raises(ValueError):
        sp.calibrate([0, math.nan, 0, 0], STOCK_CAL)


def test_calibrate_matches_package_calibration_json_if_present() -> None:
    """The real file, when a fetched copy exists locally (data/sapling/stock_head/)."""
    p = ROOT / "data" / "sapling" / "stock_head" / "calibration.json"
    if not p.exists():
        pytest.skip("calibration.json not fetched yet")
    cal = sp.load_calibration(p)
    assert cal["temperature"] == pytest.approx(STOCK_CAL["temperature"])
    assert sp.calibrate(sp.REFERENCE_LOGITS, cal)[0]["status"] == "unsure"


# --------------------------------------------------------------------------- parity / cost


def test_parity_and_reference_reports() -> None:
    a = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert sp.parity_report(a, a)["exact"]
    r = sp.parity_report(a, a + 1e-3)
    assert not r["exact"] and not r["ok"]
    assert sp.reference_report(sp.REFERENCE_LOGITS)["ok"]
    assert not sp.reference_report([0.0, 0.0, 0.0, 0.0])["ok"]
    with pytest.raises(ValueError):
        sp.max_abs_diff([1.0], [1.0, 2.0])


def test_cost_estimates_match_budget_prices() -> None:
    assert sp.estimate_cost(3600, gpu="H100", cpu_cores=0, mem_gib=0) == pytest.approx(3.95)
    assert sp.estimate_cost(3600, gpu="H100", cpu_cores=8, mem_gib=64) == pytest.approx(3.95 + 0.376 + 0.512)
    e = sp.estimate_extraction(5500, s_per_case=0.15, load_s=360)
    assert e["minutes"] == pytest.approx((360 + 825) / 60)
    assert 1.0 < e["usd"] < 2.0
    assert sp.driver_major("580.65.06") == 580


# --------------------------------------------------------------------------- feature hook (torch)


def _fake_stack():
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.nn import functional as F

    H = 8

    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_dim = H
            self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(7 * H, 16), nn.GELU(), nn.Linear(16, 4))

        def forward(self, p, e, c):
            return self.classifier(torch.cat((p, e, c, (p - e).abs(), p * e, (e - c).abs(), e * c), -1))

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(64, H)
            self.head = Head()

        def forward(self, ids, seg):
            h = self.emb(ids)
            pools = [h[seg == s].mean(0, keepdim=True) for s in (1, 2, 3)]
            return self.head(*(F.layer_norm(p.float(), (H,)) for p in pools))

    def encode(_tok, req, _max):
        if set(req) != {"policy", "evidence", "claim"}:
            raise ValueError("bad keys")
        if len(req["claim"]) > 200:
            raise ValueError("Input has too many tokens")
        ids, seg = [1], [0]
        for s, k in ((1, "policy"), (2, "evidence"), (3, "claim")):
            toks = [ord(ch) % 64 for ch in req[k]]
            ids += toks
            seg += [s] * len(toks)
        return {"input_ids": ids, "segments": seg}

    def infer(model, rows, _pad, _device="cpu"):
        model.eval()
        with torch.inference_mode():
            r = rows[0]
            return model(torch.tensor(r["input_ids"]), torch.tensor(r["segments"])).float().cpu()

    class Tok:
        pad_token_id = 0

    class Cal:
        temperature = STOCK_CAL["temperature"]

        def predict_detailed(self, logits, empty_set_policy="strict"):
            d = sp.calibrate(logits.numpy(), STOCK_CAL, empty_set_policy=empty_set_policy)[0]
            d.pop("probabilities")
            return [d]

    torch.manual_seed(0)
    return torch, Model(), Tok(), Cal(), encode, infer


def test_extract_rows_captures_head_inputs_and_parity_is_exact() -> None:
    torch, model, tok, cal, encode, infer = _fake_stack()
    from workers.verifier import sapling_runtime as rt

    rows = sp.validate_rows([
        {"id": "a", **sp.REFERENCE_EXAMPLE},
        {"id": "b", "policy": "p", "evidence": "e", "claim": "x" * 300},  # rejected by encode
        {"id": "c", "policy": "pol", "evidence": "ev", "claim": "cl"},
    ])
    feats, logits, recs = rt.extract_rows(model, tok, cal, rows, encode=encode, infer=infer, device="cpu")
    assert feats.shape == (2, 3, 8) and feats.dtype == np.float32
    assert logits.shape == (2, 4)
    assert [r["feature_index"] for r in recs] == [0, None, 1]
    assert "encode" in recs[1]["error"]
    assert recs[0]["status"] in (*sp.LABELS, "unsure")
    assert sum(recs[0]["probabilities"].values()) == pytest.approx(1.0)
    # The hook is removed afterwards.
    assert not model.head._forward_pre_hooks
    par = rt.head_parity(model.head, feats, logits, device="cpu")
    assert par["same_device_b1"]["exact"] and par["cpu_batched"]["ok"]
    model.head.train()
    rt.head_logits(model.head, feats)
    assert model.head.training  # head_logits restores the head's mode
    # A different head does not reproduce the logits.
    other = type(model.head)()
    assert not rt.head_parity(other, feats, logits, device="cpu")["same_device_b1"]["ok"]


def test_dump_head_roundtrip(tmp_path: Path) -> None:
    torch, model, *_ = _fake_stack()
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file

    from workers.verifier import sapling_runtime as rt

    shapes = rt.dump_head(model, str(tmp_path / "head.safetensors"))
    loaded = load_file(str(tmp_path / "head.safetensors"))
    assert set(loaded) == set(model.head.state_dict()) and shapes["classifier.1.weight"] == [16, 56]


# --------------------------------------------------------------------------- Modal app (static)


def test_verifier_package_never_imports_modal_or_torch_at_top_level() -> None:
    for path in (ROOT / "workers" / "verifier").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module level only; torch is imported lazily inside functions
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not any(n.split(".")[0] in ("modal", "torch") for n in names), (path, names)


def test_modal_sapling_static_config() -> None:
    ms = pytest.importorskip("workers.runners.modal_sapling")
    assert ms.APP_NAME == "busy-machines-sapling" and ms.VOLUME_NAME == "sapling-weights"
    assert ms.APP_NAME != "busy-machines-runner"
    assert ms.GPU_ENV["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "torch==2.14.1" in ms.GPU_PIP and "transformers==5.17.0" in ms.GPU_PIP
    assert set(ms.GPU_PIP) == {f"{k}=={v}" for k, v in sp.RUNTIME_PINS.items()}
    assert sp.REVISION in ms.MODEL_DIR
    assert ms.H100_TIMEOUT_S >= 3600 and ms.DOWNLOAD_TIMEOUT_S >= 3600
    assert ms.SCALEDOWN_S <= 60
    assert ms.DEFAULT_MAX_USD <= 4.0  # D28 cap
    rows = ms._smoke_rows()
    assert len(rows) == 5 and sp.validate_rows(rows)


def test_reference_example_matches_package_file_shape() -> None:
    assert set(sp.REFERENCE_EXAMPLE) == set(sp.ROW_FIELDS)
    assert json.dumps(sp.REFERENCE_EXAMPLE)  # serialisable


def test_expected_sha256_overrides_only_the_observed_drift() -> None:
    manifest = {sp.WEIGHTS_FILE: sp.WEIGHTS_SHA256, "requirements.txt": "12177eb6", "config.json": "47ca"}
    exp = sp.expected_sha256(manifest)
    assert exp["requirements.txt"] == sp.MANIFEST_DRIFT_SHA256["requirements.txt"]
    assert exp["config.json"] == "47ca" and exp[sp.WEIGHTS_FILE] == sp.WEIGHTS_SHA256
    assert exp["handler.py"] == sp.UNLISTED_PINNED_SHA256["handler.py"]
    with pytest.raises(ValueError):
        sp.expected_sha256({**manifest, sp.WEIGHTS_FILE: "0" * 64})


def test_extract_cli_parses_many_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    ms = pytest.importorskip("workers.runners.modal_sapling")
    seen: dict[str, object] = {}

    def fake(paths, run, **kw):
        seen.update(paths=paths, run=run, **kw)
        return {"ok": True, "records": []}

    monkeypatch.setattr(ms, "cmd_extract", fake)
    ms.main(["extract", "a.jsonl", "b.jsonl", "run1", "--chunk", "250", "--max-usd", "2"])
    assert seen == {"paths": ["a.jsonl", "b.jsonl"], "run": "run1", "chunk": 250, "max_usd": 2.0, "out_dir": None}
    with pytest.raises(SystemExit):
        ms.main(["extract", "only_run"])


def test_extract_refuses_over_budget_before_touching_modal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ms = pytest.importorskip("workers.runners.modal_sapling")
    rows = [{"id": str(i), "label": "x", **sp.REFERENCE_EXAMPLE} for i in range(2000)]
    sp.write_jsonl(tmp_path / "r.jsonl", rows)
    monkeypatch.setattr(ms, "_fn", lambda name: (_ for _ in ()).throw(AssertionError("touched Modal")))
    with pytest.raises(SystemExit, match="exceeds"):
        ms.cmd_extract([str(tmp_path / "r.jsonl")], "t", max_usd=0.10, out_dir=str(tmp_path / "o"))
