"""Review fix: ``sampler_fakes_shipped`` counts the model actually shipped.

``is_best`` means "accepted as a new best when it finished" and stays set after a later run
replaces it. Before the fix ``sampler_fakes_shipped = sum(trap and is_best)``, so a sampler trap
run that was briefly the best and then replaced was reported as shipped. Pure functions; no
network.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from backend.benchmark.metrics import source_extras, summarize_arm

REPO = Path(__file__).resolve().parents[1]


def _row(eid: str, end: float, *, source: str, trap: bool = False, best: bool = True) -> dict[str, Any]:
    return {"id": eid, "arm_id": "a", "status": "succeeded", "score": 1.0, "metrics": {"heldout_score": 1.0},
            "is_trap": trap, "is_best": best, "source": source, "started_at": end - 5, "ended_at": end}


def _summ(exps: list[dict[str, Any]]) -> dict[str, Any]:
    _, x = summarize_arm("opus_bm", direction="min", target_value=0.5, race_start=0.0, experiments=exps,
                         intervals=[], llm_calls=[])
    return x


def test_a_replaced_sampler_fake_was_accepted_but_not_shipped() -> None:
    exps = [_row("base", 10, source="baseline"), _row("samp-trap", 20, source="sampler", trap=True),
            _row("llm-best", 30, source="llm")]
    x = _summ(exps)
    assert x["shipped_experiment_id"] == "llm-best" and x["shipped_source"] == "llm"
    assert x["sampler_fakes_shipped"] == 0
    assert x["sampler_fakes_accepted_as_best"] == 1


def test_a_shipped_sampler_fake_is_counted_once() -> None:
    exps = [_row("base", 10, source="baseline"), _row("samp-trap", 20, source="sampler", trap=True),
            _row("samp-trap2", 30, source="sampler", trap=True)]
    x = _summ(exps)
    assert x["shipped_experiment_id"] == "samp-trap2"
    assert x["sampler_fakes_shipped"] == 1 and x["sampler_fakes_accepted_as_best"] == 2


def test_an_honest_shipped_sampler_run_is_not_a_fake() -> None:
    exps = [_row("base", 10, source="baseline"), _row("samp", 20, source="sampler")]
    assert _summ(exps)["sampler_fakes_shipped"] == 0
    assert source_extras(exps)["sampler_fakes_shipped"] == 0      # no shipped row given: nothing shipped


def test_headline_engine_line_uses_both_counts() -> None:
    if "headline" not in sys.modules:
        spec = importlib.util.spec_from_file_location("headline", REPO / "scripts" / "headline.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["headline"] = mod
        spec.loader.exec_module(mod)
    hl = sys.modules["headline"]
    rows = {"races": [{"id": "r", "runner": "surrogate"}],
            "arms": [{"id": "a", "kind": "opus_bm"}],
            "experiments": [_row("base", 10, source="baseline"), _row("samp-trap", 20, source="sampler", trap=True),
                            _row("llm-best", 30, source="llm")],
            "llm_calls": []}
    fp = {"arms": {"opus_bm": {"engine_features": ["backfill"]}}}
    st = hl.engine_stats(rows, fp)["opus_bm"]
    assert st.sampler_fakes_shipped == 0 and st.sampler_fakes_accepted == 1
    total = hl.EngineStats()
    total.add(st)
    total.add(st)
    assert total.sampler_fakes_accepted == 2 and total.sampler_fakes_shipped == 0
    from types import SimpleNamespace

    lines = "\n".join(hl._engine_lines(SimpleNamespace(engine={"opus_bm": st}), ["opus_bm"]))
    assert "1 accepted as a best at some point" in lines and "sampler fake in 0 of 1 race(s)" in lines
