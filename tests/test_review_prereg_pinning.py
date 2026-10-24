"""Review fix: the pre-registered report checks the frozen settings and refuses repeated seeds.

Before the fix a setup got the "Pre-registered report (frozen ... before any race)" banner when
its race NAMES matched the preset; only the pacing and the seed set were checked, so an edited
re-run (cadence 60 s, 24 slots, ...) under the same name could print SUCCESS LINE MET. And
``_seed_values`` kept one race per seed silently, so a re-run seed could decide the verdict
while the table pooled every race. Synthetic groups and DBs only; no network.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from backend.db import Database
from test_headline import _race, hl
from test_planner_cadence import PRESET, _frozen_fp, _vals

PR = hl.PREREGISTRATIONS[PRESET]


def _group(values: dict[str, dict[int, dict[str, float]]], fp: dict[str, Any], *,
           extra: list[tuple[str, int, dict[str, float]]] = ()) -> Any:
    runs = []
    rows = [(arm, seed, v, f"r{seed}") for arm, per in values.items() for seed, v in per.items()]
    rows += [(arm, seed, v, f"rerun{seed}") for arm, seed, v in extra]
    for arm, seed, v, rid in rows:
        runs.append(hl.RaceArm(
            race_id=rid, seed=seed, arm=arm, raw_kind=arm, cost_to_goal_usd=None, time_to_goal_s=None,
            gpu_idle_pct=v["idle"], best_quality=v["best"], shipped_quality=v["best"], start_quality=0.4,
            experiments_finished=10, total_cost_usd=v["llm"] + 3.2, gpu_cost_usd=3.2, llm_cost_usd=v["llm"],
            experiments_per_usd=None, trap_runs=0, fake_best_accepted=0, quality_line=0.86))
    return hl.Group(target="lcbench", direction="max", goal=0.78, goal_source="race config (frozen)", budget_s=900,
                    gpus_per_arm=16, run_budget_s=None, runner="surrogate", names=[], race_ids=[], seeds=[0, 1, 2],
                    arms={k: None for k in values}, start_quality=0.4, best_known=None, blockers=[], warnings=[],
                    framings=[], runs=runs, fingerprint=fp)


PASSING = {"opus": _vals(45.0, 3.0, 0.862), "opus_bm": _vals(1.0, 0.4, 0.868), "optuna": _vals(1.5, 0.0, 0.857)}


def test_the_frozen_settings_match_the_preset_and_the_doc() -> None:
    fp = _frozen_fp(hl, 0.1)
    assert hl.prereg_mismatches(PR, fp) == []
    rep = hl.prereg_report(_group(PASSING, fp), PR)
    assert rep["matches_preregistration"] and rep["verdict"].startswith("SUCCESS LINE MET")
    pinned = dict(PR.frozen)
    assert pinned["race.budget_s"] == 900.0 and pinned["race.n_slots"] == 16 and pinned["race.cost_cap_usd"] == 9.2
    assert pinned["arms.opus_bm.planner_cadence.min_propose_interval_s"] == 90.0
    assert pinned["arms.opus_bm.llm_policy.propose_effort"] == "low"
    assert pinned["arm_set"] == ("optuna", "opus", "opus_bm")


@pytest.mark.parametrize(("path", "value"), [
    ("arms.opus_bm.planner_cadence.min_propose_interval_s", 60.0),   # cadence changed after the results
    ("race.n_slots", 24),                                          # --slots 24 under the same name
    ("race.budget_s", 600.0),                                      # --minutes 10
    ("race.cost_cap_usd", 12.0),                                   # --usd-cap
    ("arms.opus_bm.llm_policy.propose_effort", "medium"),
    ("arms.opus.record_mode", "metadata"),
])
def test_an_edited_setup_under_the_same_name_is_never_the_preregistered_result(path: str, value: Any) -> None:
    fp = copy.deepcopy(_frozen_fp(hl, 0.1))
    cur = fp
    *head, last = path.split(".")
    for p in head:
        cur = cur.setdefault(p, {})
    cur[last] = value
    g = _group(PASSING, fp)                     # every success-line value would pass
    rep = hl.prereg_report(g, PR)
    assert not rep["matches_preregistration"] and any(path in m for m in rep["mismatches"])
    assert rep["verdict"].startswith("NOT THE PRE-REGISTERED SETUP")
    assert not any(c["status"] in ("PASS", "FAIL") for c in rep["success_line"])
    g.prereg, g.prereg_report = PR, rep
    md = "\n".join(hl.render_prereg(g))
    assert "NOT the pre-registered setup" in md and "SUCCESS LINE MET" not in md
    assert "**Pre-registered report**" not in md


def test_a_missing_arm_or_manifest_is_a_mismatch() -> None:
    fp = copy.deepcopy(_frozen_fp(hl, 0.1))
    del fp["arms"]["optuna"]
    assert any(m.startswith("arm_set") for m in hl.prereg_mismatches(PR, fp))
    assert hl.prereg_mismatches(PR, None) == ["no fairness manifest"]


def test_a_repeated_seed_makes_the_report_not_evaluable() -> None:
    # seed 0 re-run for every arm, with a worse value: the success line could flip on it
    extra = [(arm, 0, {**PASSING[arm][0], "best": PASSING[arm][0]["best"] - 0.02}) for arm in PASSING]
    g = _group(PASSING, _frozen_fp(hl, 0.1), extra=extra)
    rep = hl.prereg_report(g, PR)
    assert rep["repeated_seeds"] and "rerun0" in " ".join(rep["repeated_seeds"])
    assert any("repeated seeds" in w for w in rep["not_evaluable_because"])
    assert all(c["status"].endswith("(not evaluable)") for c in rep["success_line"])
    assert rep["verdict"].startswith("SUCCESS LINE NOT EVALUABLE")
    best = next(e for e in rep["endpoints"] if e["field"] == "best_quality")
    assert isinstance(best["arms"]["opus_bm"]["per_seed"]["0"], list)          # both races shown, none hidden
    assert len(best["arms"]["opus_bm"]["per_seed"]["0"]) == 2


def test_repeated_seeds_block_the_setup_in_headline_grouping(tmp_path: Path) -> None:
    path = tmp_path / "bench.db"
    db = Database(path)
    for rid, seed in (("a0", 0), ("a0b", 0), ("a1", 1), ("a2", 2)):
        _race(db, rid, seed)
    db.close()
    groups, _, _ = hl.run(db=path)
    (g,) = groups
    assert not g.headline_ok and any(b.startswith("repeated seeds") for b in g.blockers)


def test_the_two_lcbench_wide_preregistrations_differ_only_in_the_cadence() -> None:
    c30 = hl.PREREGISTRATIONS["lcbench-wide-bm-c30-15m"]
    a, b = dict(PR.frozen), dict(c30.frozen)
    assert {k for k in a if a[k] != b[k]} == {"arms.opus_bm.planner_cadence.min_propose_interval_s"}
    assert b["arms.opus_bm.planner_cadence.min_propose_interval_s"] == 30.0
    fp90, fp30 = _frozen_fp(hl, 0.1), _frozen_fp(hl, 0.1, "lcbench-wide-bm-c30-15m")
    assert hl.prereg_mismatches(c30, fp30) == [] and hl.prereg_mismatches(PR, fp90) == []
    assert hl.prereg_mismatches(c30, fp90) and hl.prereg_mismatches(PR, fp30)   # never mistaken for each other
