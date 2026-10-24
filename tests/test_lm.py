"""Tests for workers/targets/lm.py (CPU only, synthetic corpus, no network, < 60 s)."""

from __future__ import annotations

import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from workers.targets.base import get_target
from workers.targets.lm import (
    SPEC,
    LMTarget,
    corpus_path,
    ensure_corpus,
    make_splits,
    shard_bounds,
)

TINY = {
    "n_layer": 1,
    "n_embd": 64,
    "n_head": 2,
    "block_size": 64,
    "batch_size": 16,
    "lr": 1e-2,
    "warmup_frac": 0.0,
    "weight_decay": 0.0,
    "dropout": 0.0,
}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """2000 random bytes over a 32-symbol alphabet: nothing to generalise, easy to memorise,
    so the contamination trap shows up clearly in a few CPU steps."""
    rng = random.Random(0)
    alpha = b"abcdefghijklmnopqrstuvwxyzABCDEF"
    p = tmp_path_factory.mktemp("lm") / "corpus.txt"
    p.write_bytes(bytes(rng.choice(alpha) for _ in range(2000)))
    return p


def _run(corpus: Path, trap: bool) -> tuple[dict[str, Any], list[tuple[int, str, float]]]:
    """120 steps (step-based schedule -> deterministic), generous 60 s time cap."""
    pts: list[tuple[int, str, float]] = []
    cfg = SPEC.default_config() | TINY | {"train_on_all_shards": trap}
    target = LMTarget(data_path=corpus, device="cpu", log=False, n_evals=3, max_steps=120)
    res = target.run(cfg, 0, 60, lambda s, n, v: pts.append((s, n, v)))
    return res, pts


# ---------------------------------------------------------------------- spec / registry


def test_spec_and_registry() -> None:
    t = get_target("lm")
    assert isinstance(t, LMTarget)
    assert t.name == "lm" and t.metric == "val_bpb" and t.direction == "min"
    assert t.hardware == "L4" and t.budget_s == 60
    assert t.trap_option == "train_on_all_shards"
    d = SPEC.default_config()
    assert SPEC.validate_config(d) == d
    assert not SPEC.is_trap(d)
    assert SPEC.is_trap(d | {"train_on_all_shards": True})
    assert 8 <= len(SPEC.config_space) <= 16
    schema = SPEC.config_json_schema()
    assert schema["additionalProperties"] is False and set(schema["required"]) == set(SPEC.config_space)


def test_every_head_choice_divides_every_width() -> None:
    for e in SPEC.config_space["n_embd"].choices or []:
        for h in SPEC.config_space["n_head"].choices or []:
            assert int(e) % int(h) == 0


def test_random_configs_validate() -> None:
    rng = random.Random(1)
    for _ in range(50):
        SPEC.validate_config(SPEC.random_config(rng))


def test_import_does_not_import_torch() -> None:
    code = "import sys, workers.targets.lm; assert 'torch' not in sys.modules; assert 'modal' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])


# ---------------------------------------------------------------------- data


def test_shard_bounds_partition() -> None:
    b = shard_bounds(1003)
    assert b[0][0] == 0 and b[-1][1] == 1003
    assert all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))


def test_make_splits_clean_and_trap() -> None:
    raw = bytes(range(32, 127)) * 20
    clean = make_splits(raw, trap=False)
    trap = make_splits(raw, trap=True)
    n = len(raw)
    b = shard_bounds(n)
    # clean: train is shards 0-7 only, val/test are disjoint from it
    assert len(clean.train) == b[7][1]
    assert clean.train_val_overlap == 0.0
    # trap: train also contains the val shard (as a suffix); test never enters training
    assert trap.train.endswith(trap.val) and len(trap.train) == b[8][1]
    assert trap.train_val_overlap == 1.0
    assert clean.val == trap.val and clean.test == trap.test
    assert clean.split_sha256["val"] == trap.split_sha256["val"]
    assert clean.split_sha256["train"] != trap.split_sha256["train"]
    assert len(clean.train) + len(clean.val) + len(clean.test) == n
    assert clean.vocab_size == 95 and max(clean.train) < 95


def test_corpus_path_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LM_DATA_PATH", raising=False)
    monkeypatch.setenv("LM_DATA_DIR", str(tmp_path))
    assert corpus_path() == tmp_path / "tinyshakespeare.txt"
    monkeypatch.setenv("LM_DATA_PATH", str(tmp_path / "x.txt"))
    assert corpus_path() == tmp_path / "x.txt"
    assert corpus_path(tmp_path / "y.txt") == tmp_path / "y.txt"
    with pytest.raises(FileNotFoundError):
        ensure_corpus(tmp_path / "missing.txt", download=False)


# ---------------------------------------------------------------------- training (CPU, tiny)


def test_tiny_run_clean_trap_and_determinism(corpus: Path) -> None:
    clean, pts = _run(corpus, trap=False)
    for k in ("score", "reported_score", "heldout_score", "val_bpb", "test_bpb", "is_trap",
              "train_val_overlap", "steps", "tokens_seen", "n_params", "train_time_s"):
        assert k in clean and isinstance(clean[k], float) and math.isfinite(clean[k]), k
    assert clean["score"] == clean["reported_score"] == clean["val_bpb"]
    assert clean["heldout_score"] == clean["test_bpb"]
    assert clean["is_trap"] == 0.0 and clean["steps"] == 120.0
    names = [n for _, n, _ in pts]
    assert names.count("val_bpb") >= 3  # streamed evals + final
    assert {"train_loss", "lr"} <= set(names)
    steps = [s for s, n, _ in pts if n == "val_bpb"]
    assert steps == sorted(steps)

    again, _ = _run(corpus, trap=False)
    assert again["score"] == clean["score"] and again["heldout_score"] == clean["heldout_score"]

    trap, _ = _run(corpus, trap=True)
    assert trap["is_trap"] == 1.0 and trap["train_val_overlap"] == 1.0
    # The trap inflates the reported metric (fake improvement) but not held-out quality.
    assert trap["reported_score"] < clean["reported_score"] - 1.0
    assert trap["heldout_score"] - trap["reported_score"] > 1.0
    assert abs(clean["heldout_score"] - clean["reported_score"]) < 0.6


def test_time_budget_stops_training(corpus: Path) -> None:
    pts: list[tuple[int, str, float]] = []
    cfg = SPEC.default_config() | TINY
    res = LMTarget(data_path=corpus, device="cpu", log=False, n_evals=2).run(
        cfg, 1, 2, lambda s, n, v: pts.append((s, n, v)))
    assert 2.0 <= res["train_time_s"] < 3.5
    assert res["steps"] >= 1
    assert any(n == "val_bpb" for _, n, _ in pts)


def test_invalid_config_raises(corpus: Path) -> None:
    t = LMTarget(data_path=corpus, device="cpu", log=False, max_steps=1)
    with pytest.raises(ValueError):
        t.run(SPEC.default_config() | {"n_layer": 99}, 0, 5, lambda *a: None)
    with pytest.raises(ValueError):
        t.run(SPEC.default_config() | {"bogus": 1}, 0, 5, lambda *a: None)
    with pytest.raises(ValueError):  # block longer than a 200-byte shard
        t.run(SPEC.default_config() | TINY | {"block_size": 512}, 0, 5, lambda *a: None)


# ---------------------------------------------------------------------- research choices (D29)

RESEARCH_DEFAULTS = {"pos_encoding": "learned", "mlp": "gelu", "qk_norm": False, "norm": "layernorm",
                     "tie_embeddings": True}

# Recorded from the pre-D29 code (torch 2.14.1+cpu, the module's 2000-byte corpus) on 24 Oct:
# the default of every research choice must reproduce the original code path exactly.
RECORDED_TORCH = "2.14.1"
RECORDED = [
    (TINY, 3, 120, 0, {"score": 6.223789345234526, "heldout_score": 6.002640982073661,
                       "train_loss": 2.6195634841918944, "n_params": 55680.0}),
    (TINY | {"n_layer": 2, "dropout": 0.1, "weight_decay": 0.1, "warmup_frac": 0.1}, 2, 40, 5,
     {"score": 5.263816584648501, "heldout_score": 5.283745590265181,
      "train_loss": 3.319627380371094, "n_params": 105088.0}),
]


def test_research_choice_defaults_and_descriptions() -> None:
    for name, default in RESEARCH_DEFAULTS.items():
        p = SPEC.config_space[name]
        assert p.default == default and p.description
        assert p.type in ("categorical", "bool")
    assert SPEC.config_space["pos_encoding"].choices == ["learned", "rope"]
    assert SPEC.config_space["mlp"].choices == ["gelu", "swiglu"]
    assert SPEC.config_space["norm"].choices == ["layernorm", "rmsnorm"]
    assert not SPEC.is_trap(SPEC.default_config() | {"pos_encoding": "rope", "tie_embeddings": False})


@pytest.mark.parametrize("tiny,n_evals,steps,seed,expected", RECORDED)
def test_default_config_reproduces_recorded_scores(corpus: Path, tiny: dict, n_evals: int, steps: int,
                                                   seed: int, expected: dict) -> None:
    torch = pytest.importorskip("torch")
    if not torch.__version__.startswith(RECORDED_TORCH):
        pytest.skip(f"values recorded with torch {RECORDED_TORCH}")
    t = LMTarget(data_path=corpus, device="cpu", log=False, n_evals=n_evals, max_steps=steps)
    res = t.run(SPEC.default_config() | tiny, seed, 60, lambda *a: None)
    for k, v in expected.items():
        assert res[k] == pytest.approx(v, rel=1e-9), k
    # explicit defaults == omitted research keys (same code path)
    cfg = {k: v for k, v in (SPEC.default_config() | tiny).items() if k not in RESEARCH_DEFAULTS}
    again = t.run(cfg | RESEARCH_DEFAULTS, seed, 60, lambda *a: None)
    assert again["score"] == res["score"]


def _short(corpus: Path, extra: dict) -> dict[str, Any]:
    t = LMTarget(data_path=corpus, device="cpu", log=False, n_evals=1, max_steps=20)
    return t.run(SPEC.default_config() | TINY | extra, 0, 60, lambda *a: None)


@pytest.mark.parametrize("extra", [
    {"pos_encoding": "rope"}, {"mlp": "swiglu"}, {"qk_norm": True}, {"norm": "rmsnorm"},
    {"tie_embeddings": False},
    {"pos_encoding": "rope", "mlp": "swiglu", "qk_norm": True, "norm": "rmsnorm", "tie_embeddings": False},
])
def test_research_choice_trains(corpus: Path, extra: dict) -> None:
    ref = _short(corpus, {})
    res = _short(corpus, extra)
    assert math.isfinite(res["score"]) and math.isfinite(res["heldout_score"])
    assert res["is_trap"] == 0.0 and res["steps"] == 20.0
    assert res["score"] != ref["score"]  # the option changes training


def test_research_choice_param_counts(corpus: Path) -> None:
    ref = _short(corpus, {})["n_params"]
    vocab_x_width = 32 * TINY["n_embd"]
    assert _short(corpus, {"tie_embeddings": False})["n_params"] == ref + vocab_x_width
    assert _short(corpus, {"pos_encoding": "rope"})["n_params"] == ref - TINY["block_size"] * TINY["n_embd"]
    assert abs(_short(corpus, {"mlp": "swiglu"})["n_params"] - ref) / ref < 0.05


def test_rope_rotation_preserves_norm_and_relative_position() -> None:
    torch = pytest.importorskip("torch")
    from workers.targets.lm import _torch_impl

    impl = _torch_impl()
    model = impl["GPT"](32, 16, 1, 64, 2, 0.0, pos_encoding="rope")
    assert not hasattr(model, "pos")
    cos, sin = model.rope_cos, model.rope_sin  # (block, head_dim / 2)
    assert tuple(cos.shape) == (16, 16)
    q, k = torch.randn(1, 1, 1, 32).expand(1, 1, 16, 32), torch.randn(1, 1, 1, 32).expand(1, 1, 16, 32)
    rq, rk = impl["apply_rope"](q, cos, sin), impl["apply_rope"](k, cos, sin)
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    # Same q and k at every position: <rope(q)_i, rope(k)_j> depends only on i - j.
    dots = (rq[0, 0, 3:] * rk[0, 0, :-3]).sum(-1)
    assert torch.allclose(dots, dots[0].expand_as(dots), atol=1e-4)
