"""The contamination-trap set for judge_improvement (scripts/make_trap_set.py). No LLM, no Modal."""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from backend.agent.prompts.build import CACHE_BREAK, JudgeClaim, build_judge_user
from backend.scheduler import GROUND_TRUTH_METRIC_KEYS
from workers.targets.base import get_target

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_trap_set", REPO / "scripts" / "make_trap_set.py")
assert _spec and _spec.loader
mts = importlib.util.module_from_spec(_spec)
sys.modules["make_trap_set"] = mts
_spec.loader.exec_module(mts)

CASES: list[dict[str, Any]] = mts.build_cases()


def _file_cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in mts.DEFAULT_OUT.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------------------- determinism


def test_build_is_deterministic() -> None:
    assert mts.render_jsonl(mts.build_cases()) == mts.render_jsonl(CASES)


def test_committed_file_matches_generator() -> None:
    assert mts.DEFAULT_OUT.exists(), "run scripts/make_trap_set.py"
    assert mts.DEFAULT_OUT.read_bytes().decode("utf-8") == mts.render_jsonl(CASES), \
        "data/trap_set/trap_set_v1.jsonl is stale: rerun scripts/make_trap_set.py"
    assert mts.main(["--check"]) == 0


def test_other_seed_is_refused() -> None:
    with pytest.raises(ValueError):
        mts.build_cases(seed=mts.SEED + 1)


# --------------------------------------------------------------------------- balance


def test_size_and_label_balance() -> None:
    assert len(CASES) == 50
    assert Counter(c["label"] for c in CASES) == {"valid": 25, "invalid": 25}
    assert len({c["id"] for c in CASES}) == 50


def test_dev_test_split_is_stratified() -> None:
    cnt = Counter((c["split"], c["label"]) for c in CASES)
    assert cnt == {("dev", "valid"): 15, ("dev", "invalid"): 15, ("test", "valid"): 10, ("test", "invalid"): 10}
    for cat in mts.CATEGORIES:
        splits = {c["split"] for c in CASES if c["category"] == cat}
        n = sum(c["category"] == cat for c in CASES)
        if n >= 3:
            assert splits == {"dev", "test"}, f"{cat} should appear in both splits"


def test_targets_and_categories_spread() -> None:
    assert all(n >= 15 for n in Counter(c["target"] for c in CASES).values())
    cats = Counter(c["category"] for c in CASES)
    assert set(cats) == set(mts.CATEGORIES)
    for t, cat in mts.TRAP_CATEGORY.items():
        assert all(c["target"] == t for c in CASES if c["category"] == cat)
        assert cats[cat] == 5
    for cat in ("eval_changed", "lucky_seed"):
        assert {c["target"] for c in CASES if c["category"] == cat} == {"cifar", "lm", "rl"}


def test_profiles_balanced_across_labels() -> None:
    cnt = Counter((c["label"], c["evidence_profile"]) for c in CASES)
    for label in ("valid", "invalid"):
        assert abs(cnt[(label, "full")] - cnt[(label, "lean")]) <= 1


# --------------------------------------------------------------------------- schema


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case_schema(case: dict[str, Any]) -> None:
    cat = mts.CATEGORIES[case["category"]]
    assert case["label"] == cat["label"] and case["family"] == cat["family"]
    assert case["acceptable_verdicts"] == cat["accept"]
    assert case["split"] in ("dev", "test")
    assert case["set_version"] == mts.SET_VERSION
    assert case["prompt"]["step"] == "judge_improvement"

    spec = get_target(case["target"]).spec
    claim = JudgeClaim.model_validate(case["claim"])
    assert claim.prev_best_config is not None
    assert spec.validate_config(claim.prev_best_config) == claim.prev_best_config
    assert spec.validate_config(claim.new_config) == claim.new_config
    assert claim.prev_best_score is not None and spec.is_better(claim.new_score, claim.prev_best_score)
    assert claim.new_config != claim.prev_best_config

    # The stored message is exactly what the builder renders for this claim.
    assert case["user_message"] == build_judge_user(spec, claim)
    assert CACHE_BREAK in case["user_message"]

    # No ground truth reaches the judge.
    ev = claim.evidence
    assert not (set(ev) & GROUND_TRUTH_METRIC_KEYS)
    assert not any("heldout" in k for k in ev)
    dynamic = case["user_message"].split(CACHE_BREAK, 1)[1]
    for word in ("heldout_score", "is_trap", "test_bpb", "train_levels_score", "contaminat", "trap"):
        assert word not in dynamic

    # Key set depends only on (target, profile), never on the label.
    assert set(ev) == mts.expected_evidence_keys(case["target"], case["evidence_profile"])

    # Seed bookkeeping is consistent with the reported score.
    seeds = list(ev["seed_scores"].values())
    agg = case["aggregation"]
    if agg == "single":
        assert seeds == [claim.new_score]
    elif agg == "mean":
        assert len(seeds) == 3 and abs(statistics.fmean(seeds) - claim.new_score) < 10 ** -mts.MODELS[case["target"]].prec
    else:
        best = max(seeds) if spec.direction == "max" else min(seeds)
        assert len(seeds) == 3 and best == claim.new_score

    # Deciding evidence names fields the judge can actually see.
    for k in case["deciding_evidence"]:
        assert k in ev or k in ("changed parameters", "claimed gain", "previous best")


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_ground_truth_is_consistent(case: dict[str, Any]) -> None:
    spec = get_target(case["target"]).spec
    claim = JudgeClaim.model_validate(case["claim"])
    h = case["hidden"]
    ev = claim.evidence
    std = ev["prev_best_seed_std"]
    gain = abs(claim.new_score - (claim.prev_best_score or 0.0))
    trap_case = case["family"] == "contamination"

    assert not spec.is_trap(claim.prev_best_config or {})
    assert h["trap_on"] == trap_case == spec.is_trap(claim.new_config)
    if trap_case:
        assert spec.trap_option in [k for k in claim.new_config if claim.new_config[k] != claim.prev_best_config[k]]  # type: ignore[index]

    true_gain = h["true_improvement"]
    if case["label"] == "valid":
        assert true_gain > 2 * std and gain > 2 * std          # real and outside the noise
        assert spec.is_better(h["new_heldout_score"], h["prev_heldout_score"])
    else:
        assert true_gain <= 0.5 * std + 1e-9                   # no real gain on the clean eval
    if case["category"] == "noise_within_spread":
        assert gain < std
    if case["category"] == "lucky_seed":
        mean = statistics.fmean(ev["seed_scores"].values())
        assert not spec.is_better(mean, claim.prev_best_score)
        assert gain > std                                      # the noise rule alone does not catch it


def test_trap_evidence_signals() -> None:
    for c in CASES:
        ev, t = c["claim"]["evidence"], c["target"]
        trap = c["family"] == "contamination"
        if t == "cifar":
            assert (ev["finetune_split_sha256"] == ev["test_split_sha256"]) == trap
            assert (ev["adapt_steps"] > 0) == trap
        elif t == "lm":
            assert (ev["train_shards"] == "0-8") == trap
        else:
            assert (ev["eval_level_seed"] == ev["train_level_seed"]) == trap
        changed = c["category"] == "eval_changed"
        prev_n = {"cifar": "prev_eval_examples", "lm": "prev_eval_tokens", "rl": "prev_eval_episodes"}[t]
        n = {"cifar": "eval_examples", "lm": "eval_tokens", "rl": "eval_episodes"}[t]
        assert (ev[n] != ev[prev_n]) == changed


def test_file_round_trips() -> None:
    assert _file_cases() == CASES
