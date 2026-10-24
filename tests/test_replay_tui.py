"""Terminal race replay (scripts/replay_tui.py): frame builder and renderer, offline.

A synthetic race as plain rows (no DB, no network, no Modal, no LLM), rebuilt through
backend.replay.rebuild_events, then checked frame by frame against hand-computed numbers.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from backend.core.types import ArmSummary

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("replay_tui", ROOT / "scripts" / "replay_tui.py")
R = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["replay_tui"] = R
_spec.loader.exec_module(R)  # type: ignore[union-attr]

T0 = 1_000_000.0
P = 0.001


def _exp(i: str, arm: str, s: float, e: float, status: str, score: float | None, **kw: Any) -> dict[str, Any]:
    row = {"id": i, "arm_id": arm, "race_id": "race-1", "slot_id": f"{arm}-0", "config": {"lr": s},
           "hypothesis": f"h-{i}", "status": status, "seed": 0, "hardware": "L4", "proposed_at": T0 + s - 0.5,
           "started_at": T0 + s, "ended_at": T0 + e, "score": score, "gpu_cost_usd": (e - s) * P,
           "is_trap": False, "is_best": False, "judge_verdict": None, "metrics": {}, "summary": {}, "error": None}
    row.update(kw)
    return row


def _call(i: str, arm_id: str, kind: str, step: str, start: float, end: float, cost: float,
          exps: list[str]) -> dict[str, Any]:
    return {"id": i, "arm_id": arm_id, "arm": kind, "session_id": "s", "step_type": step, "model": "claude-opus-5-5",
            "latency_ms": (end - start) * 1000, "cost_usd": cost, "valid": True, "experiment_ids": exps,
            "created_at": T0 + start, "completed_at": T0 + end, "race_id": "race-1"}


def _iv(arm: str, state: str, s: float, e: float, exp: str | None = None, hw: str = "L4") -> dict[str, Any]:
    return {"arm_id": arm, "race_id": "race-1", "slot": f"{arm}-0", "hardware": hw, "state": state,
            "start": T0 + s, "end": T0 + e, "experiment_id": exp, "price_per_s": P}


def synthetic_rows(hw: str = "L4") -> dict[str, list[dict[str, Any]]]:
    """Busy Machines (A, one slot) vs Optuna (B, one slot), 40 s, goal 0.65 (max).

    A: a1 0-10 (0.50) | a2 11-20 trap, claimed 0.90, held-out 0.45, judge 'invalid' at 21 |
       a3 21-30 (0.70, held-out 0.68, reaches the goal) | a4 31-35 cancelled (early stop) | idle to 40.
    B: b1 0-8 (0.60) | b2 8-16 failed | idle 16-30, slot closed at 30.
    A correction flags a1 as fake at 12 (by Sapling)."""
    race = {"id": "race-1", "status": "finished", "finish_reason": "budget", "target": "synthetic",
            "name": "synthetic race", "mode": "dev", "started_at": T0, "ended_at": T0 + 40, "created_at": T0,
            "config": {"arms": ["opus_bm", "optuna"], "target_value": 0.65, "seed": 0}}
    arms = [
        {"id": "A", "race_id": "race-1", "kind": "opus_bm", "hardware": hw, "n_slots": 1, "started_at": T0,
         "ended_at": T0 + 40, "summary": ArmSummary(arm="opus_bm").model_dump(mode="json"), "report": {}},
        {"id": "B", "race_id": "race-1", "kind": "optuna", "hardware": hw, "n_slots": 1, "started_at": T0,
         "ended_at": T0 + 30, "summary": ArmSummary(arm="optuna").model_dump(mode="json"), "report": {}},
    ]
    exps = [
        _exp("a1", "A", 0, 10, "succeeded", 0.50, is_best=True, metrics={"heldout_score": 0.50}),
        _exp("a2", "A", 11, 20, "succeeded", 0.90, is_trap=True, judge_verdict="invalid",
             metrics={"heldout_score": 0.45},
             summary={"claimed_new_best": True, "judge_policy": {"llm_reason": "contamination_or_eval_leak"}}),
        _exp("a3", "A", 21, 30, "succeeded", 0.70, is_best=True, judge_verdict="valid",
             metrics={"heldout_score": 0.68}),
        _exp("a4", "A", 31, 35, "cancelled", None, error="early stop: below median"),
        _exp("b1", "B", 0, 8, "succeeded", 0.60, is_best=True),
        _exp("b2", "B", 8, 16, "failed", None, error="loss is NaN"),
    ]
    calls = [
        _call("c1", "A", "opus_bm", "propose_experiments", 9.0, 10.0, 0.05, ["a2"]),
        _call("c2", "A", "opus_bm", "judge_improvement", 20.0, 21.0, 0.02, ["a2"]),
        _call("c3", "A", "opus_bm", "judge_improvement", 30.0, 30.5, 0.02, ["a3"]),
    ]
    ivs = [
        _iv("A", "busy", 0, 10, "a1", hw), _iv("A", "idle", 10, 11, None, hw), _iv("A", "busy", 11, 20, "a2", hw),
        _iv("A", "idle", 20, 21, None, hw), _iv("A", "busy", 21, 30, "a3", hw), _iv("A", "idle", 30, 31, None, hw),
        _iv("A", "busy", 31, 35, "a4", hw), _iv("A", "idle", 35, 40, None, hw),
        _iv("B", "busy", 0, 8, "b1", hw), _iv("B", "busy", 8, 16, "b2", hw), _iv("B", "idle", 16, 30, None, hw),
    ]
    corrections = [{"id": "k1", "race_id": "race-1", "arm_id": "A", "experiment_id": "a1", "source": "sapling",
                    "label": "fake_improvement", "reason": "leak", "created_at": T0 + 12}]
    return {"races": [race], "arms": arms, "experiments": exps, "metrics": [], "llm_calls": calls,
            "gpu_intervals": ivs, "corrections": corrections}


@pytest.fixture()
def model() -> Any:
    return R.build_model(synthetic_rows(), speed=1.0)


def _arm(frame: Any, kind: str) -> Any:
    return next(a for a in frame.arms if a.kind == kind)


def test_frames_follow_the_race(model: Any) -> None:
    f = model.frame()
    assert [a.kind for a in f.arms] == ["opus_bm", "optuna"]          # config arm order
    assert f.race_id == "race-1" and f.duration_s == pytest.approx(40) and f.resource == "GPUs"
    assert f.arms_line == "Busy Machines (Opus 5.5) vs No-AI tuner (Optuna)"

    model.advance_by_elapsed(5)
    a, b = _arm(model.frame(), "opus_bm"), _arm(model.frame(), "optuna")
    assert a.slots == [("A-0", R.BUSY)] and b.slots == [("B-0", R.BUSY)]
    assert a.ai_bill_usd == 0 and a.experiments_finished == 0 and a.best_quality is None
    assert a.waiting_pct == 0 and b.uses_ai is False and a.uses_ai is True

    model.advance_by_elapsed(10.5)
    a = _arm(model.frame(), "opus_bm")
    assert a.slots == [("A-0", R.WAITING)]
    assert a.ai_bill_usd == pytest.approx(0.05) and a.experiments_finished == 1
    assert a.best_quality == pytest.approx(0.50) and a.quality_changed_recently
    assert a.waiting_pct == pytest.approx(0.5 / 10.5 * 100)

    model.advance_by_elapsed(17)
    f = model.frame()
    a, b = _arm(f, "opus_bm"), _arm(f, "optuna")
    assert b.slots == [("B-0", R.CANCELLED)]                           # idle after a failed run
    assert b.best_quality == pytest.approx(0.60) and b.experiments_finished == 1
    assert a.fakes_caught == 1                                         # Sapling correction at 12
    assert "flagged by the Sapling checker" in f.feed[-1][0].text


def test_fake_caught_flashes_when_the_judge_rejects_it(model: Any) -> None:
    model.advance_by_elapsed(20.5)
    a = _arm(model.frame(), "opus_bm")
    assert a.experiments_finished == 2 and a.best_quality == pytest.approx(0.50)  # trap never counts
    assert a.fakes_caught == 1                                         # only the correction so far

    model.advance_by_elapsed(21)
    f = model.frame()
    a = _arm(f, "opus_bm")
    assert a.fakes_caught == 2 and a.ai_bill_usd == pytest.approx(0.07)
    item, hot = f.feed[-1]
    assert hot and item.kind == "fake" and item.at_s == pytest.approx(21)
    assert item.text.startswith("Fake improvement caught")
    assert "claimed 0.9000" in item.text and "real held-out 0.4500" in item.text
    assert "test data leaked in" in item.text

    model.advance_by_elapsed(25)                                       # hold = 3 s x speed 1
    assert model.frame().feed[-1][1] is False


def test_goal_waiting_and_final_numbers(model: Any) -> None:
    model.advance_by_elapsed(30.2)
    f = model.frame()
    a = _arm(f, "opus_bm")
    assert a.best_quality == pytest.approx(0.68) and a.goal_at_s == pytest.approx(30)
    assert f.feed[-1][0].kind == "goal"

    model.advance_by_elapsed(36)
    f = model.frame()
    assert _arm(f, "opus_bm").slots == [("A-0", R.CANCELLED)]          # idle after the cancelled a4
    assert _arm(f, "optuna").slots == [("B-0", R.OFF)]                 # slot closed at 30

    model.advance_by_elapsed(1e9)
    f = model.frame()
    a, b = _arm(f, "opus_bm"), _arm(f, "optuna")
    assert f.finished and f.elapsed_s == pytest.approx(40)
    assert a.waiting_pct == pytest.approx(8 / 40 * 100)
    assert b.waiting_pct == pytest.approx(14 / 30 * 100)
    assert a.ai_bill_usd == pytest.approx(0.09) and a.experiments_finished == 3 and a.fakes_caught == 2
    assert b.goal_at_s is None and b.ai_bill_usd == 0
    assert a.slots == [("A-0", R.OFF)]


def test_advance_never_goes_back(model: Any) -> None:
    model.advance_by_elapsed(21)
    n = model.pos
    model.advance_by_elapsed(5)
    assert model.pos == n and model.now == pytest.approx(T0 + 21)


def test_cpu_slots_are_labelled_machines() -> None:
    m = R.build_model(synthetic_rows(hw="CPU"), speed=30)
    assert m.frame().resource == "Machines"
    assert R.build_model(synthetic_rows(hw="CPU"), speed=30, resource="GPUs").frame().resource == "GPUs"


def test_record_layout_is_fixed_and_plain_language(model: Any) -> None:
    m = R.build_model(synthetic_rows(), speed=30)
    texts = []
    for t in (0, 21, 40):
        m.advance_by_elapsed(t)
        out = R.frame_text(m.frame())
        lines = out.rstrip("\n").split("\n")
        assert len(lines) == R.RECORD_HEIGHT
        assert all(len(ln) <= R.RECORD_WIDTH for ln in lines)
        texts.append(out)
    head = texts[0]
    for label in ("replay of a logged race, x30", "Race race-1", "Busy Machines (Opus 5.5)", "No-AI tuner (Optuna)",
                  "Model quality", "AI bill", "GPUs waiting", "Experiments finished", "Goal >= 0.6500"):
        assert label in head
    assert "Fake improvement caught" in texts[1]
    assert "Race finished" in texts[2]
    for jargon in ("val_bpb", "adapter", "idle %", "opus_bm"):
        assert jargon not in texts[1]


def test_non_benchmark_races_are_labelled_on_screen() -> None:
    rows = synthetic_rows()                                           # mode "dev"
    assert "[dev race, not a benchmark result]" in R.frame_text(R.build_model(rows, speed=30).frame())
    demo = "demo run for the video, not a benchmark result"
    rows["races"][0]["config"]["demo_label"] = demo
    assert f"[{demo}]" in R.frame_text(R.build_model(rows, speed=30).frame())
    rows["races"][0].update(mode="benchmark", config={k: v for k, v in rows["races"][0]["config"].items()
                                                      if k != "demo_label"})
    m = R.build_model(rows, speed=30)
    assert m.frame().note is None and "not a benchmark result" not in R.frame_text(m.frame())


def test_render_handles_many_arms_and_small_screens() -> None:
    rows = synthetic_rows()
    m = R.build_model(rows, speed=10)
    m.advance_by_elapsed(21)
    f = m.frame()
    f.arms = f.arms * 2                                               # 4 columns
    for w, h in ((120, 34), (80, 24), (200, 50)):
        out = R.frame_text(f, w, h).rstrip("\n").split("\n")
        assert len(out) == h


def test_grid_lines_pick_the_biggest_cells_that_fit() -> None:
    slots = [(str(i), R.BUSY) for i in range(30)]
    big = R.grid_lines(slots[:16], 36, 14)
    assert big[0].plain.startswith("█████ ")                          # 5x3 cells for 16 slots
    mid = R.grid_lines(slots, 36, 13)
    assert mid[0].plain.startswith("███ ") and len(mid) <= 13         # 3x2 cells for 30 slots
    tiny = R.grid_lines(slots, 36, 2)
    assert len(tiny) <= 2
    assert R.grid_lines([], 36, 5)[0].plain == "no slots"


def test_jsonl_reader_skips_metrics(tmp_path: Path) -> None:
    rows = synthetic_rows()
    rows["metrics"] = [{"experiment_id": "a1", "step": 0, "t": 1.0, "name": "loss", "value": 1.0}]
    p = tmp_path / "race.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for table, items in rows.items():
            for r in items:
                f.write(json.dumps({"row": r, "table": table}, sort_keys=True) + "\n")
    got, src = R.load_rows("race", jsonl=p)
    assert src == str(p) and got["metrics"] == []
    assert len(got["experiments"]) == 6 and got["races"][0]["id"] == "race-1"
    with pytest.raises(SystemExit):
        R.load_rows("other", jsonl=p)
    m = R.build_model(got, speed=30)
    m.advance_by_elapsed(1e9)
    assert _arm(m.frame(), "opus_bm").fakes_caught == 2


def test_resolve_race_id() -> None:
    ids = ["4ca4ed27aa", "4cb00000", "c1f4b1d9ff"]
    assert R.resolve_race_id(ids, "c1f4") == "c1f4b1d9ff"
    assert R.resolve_race_id(ids, "4cb00000") == "4cb00000"
    with pytest.raises(SystemExit):
        R.resolve_race_id(ids, "4c")
    with pytest.raises(SystemExit):
        R.resolve_race_id(ids, "zz")
