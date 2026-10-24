"""Review fix: 'Shipped model' quality never uses a contaminated held-out score.

On lcbench the trap (``checkpoint_selection="best_test"``) keeps the epoch with the best TEST
score and ``heldout_score`` IS that test score, so a shipped trap run's held-out score is too
good. Before the fix ``shipped_quality = quality(shipped)`` reported it, and an arm that shipped
a fake (Optuna, D47 seed 2: 0.8645 vs its best counted 0.8428) looked like it shipped a better
model. Pure functions over plain rows; no network.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from backend.benchmark.analysis import summaries_from_rows
from backend.benchmark.metrics import TRAP_CONTAMINATES_HELDOUT, shipped_quality, summarize_arm

REPO = Path(__file__).resolve().parents[1]


def _load(name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _exps(arm: str = "a") -> list[dict[str, Any]]:
    """An honest best (0.8428 held-out), then a trap the arm accepted as its best (0.8645)."""
    return [
        {"id": "base", "arm_id": arm, "status": "succeeded", "score": 0.80, "metrics": {"heldout_score": 0.80},
         "is_trap": False, "is_best": True, "is_baseline": True, "started_at": 0.0, "ended_at": 10.0},
        {"id": "honest", "arm_id": arm, "status": "succeeded", "score": 0.85, "metrics": {"heldout_score": 0.8428},
         "is_trap": False, "is_best": True, "started_at": 5.0, "ended_at": 20.0},
        {"id": "fake", "arm_id": arm, "status": "succeeded", "score": 0.90, "metrics": {"heldout_score": 0.8645},
         "is_trap": True, "is_best": True, "started_at": 10.0, "ended_at": 30.0},
    ]


def _summ(target: str | None) -> tuple[Any, dict[str, Any]]:
    ivs = [{"state": "busy", "start": 0.0, "end": 40.0, "price_per_s": 0.001}]
    return summarize_arm("optuna", direction="max", target_value=0.78, race_start=0.0, experiments=_exps(),
                         intervals=ivs, llm_calls=[], target=target)


def test_lcbench_shipped_fake_has_no_quality() -> None:
    assert "lcbench" in TRAP_CONTAMINATES_HELDOUT
    s, x = _summ("lcbench")
    assert x["shipped_experiment_id"] == "fake" and x["shipped_is_fake"] is True
    assert x["shipped_quality"] is None                        # withheld: contaminated
    assert x["shipped_heldout_raw"] == pytest.approx(0.8645)   # audit only
    assert s.best_score == pytest.approx(0.8428)               # the counted best is unchanged


@pytest.mark.parametrize("target", ["rl", "cifar", "lm", None])
def test_targets_with_an_honest_heldout_split_keep_the_shipped_fakes_score(target: str | None) -> None:
    _, x = _summ(target)
    assert x["shipped_is_fake"] is True
    assert x["shipped_quality"] == pytest.approx(0.8645)       # honest held-out: shows the cost of a fake


def test_an_honest_shipped_model_is_unchanged() -> None:
    honest = {"status": "succeeded", "score": 0.85, "metrics": {"heldout_score": 0.8428}, "is_trap": False}
    assert shipped_quality(honest, "lcbench") == pytest.approx(0.8428)
    assert shipped_quality(None, "lcbench") is None
    exps = _exps()[:2]
    _, x = summarize_arm("opus", direction="max", target_value=0.78, race_start=0.0, experiments=exps,
                         intervals=[], llm_calls=[], target="lcbench")
    assert x["shipped_quality"] == pytest.approx(0.8428) and x["shipped_is_fake"] is False


def _rows(target: str = "lcbench") -> dict[str, Any]:
    return {"races": [{"id": "race1", "target": target, "started_at": 0.0, "seed": 2, "status": "finished",
                       "config": {"target_value": 0.78}, "budget_s": 40, "gpus_per_arm": 1}],
            "arms": [{"id": "a", "kind": "optuna", "model": None, "llm_mode": "none"}],
            "experiments": _exps(), "llm_calls": [],
            "gpu_intervals": [{"arm_id": "a", "state": "busy", "start": 0.0, "end": 40.0, "price_per_s": 0.001}]}


def test_export_analysis_passes_the_target() -> None:
    out = summaries_from_rows(_rows(), direction="max", target_value=0.78, allow_dev=True)
    _, x = out["optuna"]
    assert x["shipped_quality"] is None and x["shipped_is_fake"] is True


def test_analyze_races_and_headline_flag_the_fake_shipped_model() -> None:
    ar = _load("analyze_races")
    (run,) = ar.arm_runs_for_race(_rows(), direction="max", target_value=0.78, is_dev=False)
    assert run.shipped_quality is None and run.shipped_is_fake is True
    rep = ar.TargetReport(target="lcbench", direction="max", target_value=0.78, target_value_source="t",
                          start_score=None, start_source="", best_known=None, best_source="", is_dev=False,
                          race_ids=["race1"], arms={}, runs=[run], warnings=[])
    assert ar._fake_shipped_note(rep, "optuna") == " (1 fake shipped, held-out not counted)"
    (rl_run,) = ar.arm_runs_for_race(_rows("rl"), direction="max", target_value=0.78, is_dev=False)
    assert rl_run.shipped_quality == pytest.approx(0.8645) and rl_run.shipped_is_fake
    hl = _load("headline")
    (ra,) = hl.race_arms(_rows(), direction="max", goal=0.78, is_dev=False)
    assert ra.shipped_quality is None and ra.shipped_is_fake is True
