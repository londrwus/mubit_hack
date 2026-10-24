"""scripts/headline.py on a synthetic DB built with backend/db.py. No LLM, no Modal.

Every expected number below is worked out by hand from the rows written in ``_race``, so the
test checks the section 6b formulas themselves, not just that something prints.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from backend.core.types import ArmKind, LLMCallRecord, StepType
from backend.db import Database

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("headline", REPO / "scripts" / "headline.py")
assert _spec and _spec.loader
hl = importlib.util.module_from_spec(_spec)
sys.modules["headline"] = hl
_spec.loader.exec_module(hl)

T0 = 1000.0
PRICE = 0.001  # $/s per slot; 2 slots x 100 s = $0.20 GPU per arm
GOAL = -500.0
CFG = {"target": "rl", "target_value": GOAL, "provisional": False}
PRESET_NAME = "RL wide 15 min (15 s runs)"  # backend.benchmark.configs.PRESETS["rl-wide-15m"].name


def _exp(db: Database, race: str, arm: str, eid: str, *, end: float, heldout: float, score: float | None = None,
         baseline: bool = False, trap: bool = False, best: bool = False, verdict: str | None = None) -> None:
    db.upsert_experiment({
        "id": eid, "arm_id": arm, "race_id": race, "status": "succeeded",
        "score": heldout if score is None else score, "metrics": {"heldout_score": heldout},
        "is_baseline": baseline, "is_trap": trap, "is_best": best, "judge_verdict": verdict,
        "proposed_at": T0 + end - 25, "started_at": T0 + end - 20, "ended_at": T0 + end,
        "run_ref": f"fc-{eid}", "seed": 0,
    })


def _call(db: Database, race: str, arm_id: str, i: int, *, cost: float, at: float) -> None:
    # call 1 of every arm is a judge call, so the judge prompt version is "called" (fingerprint)
    step = StepType.judge_improvement if i == 1 else StepType.propose_experiments
    rec = LLMCallRecord(arm_id=arm_id, arm=ArmKind.opus, session_id=f"race-{race}/{arm_id}/seed-0",
                        step_type=step, model="claude-opus-5-5", latency_ms=1000,
                        cost_usd=cost, valid=True, mubit_request_id=f"mreq-{arm_id}-{i}", created_at=T0 + at - 1)
    db.insert_llm_call(rec, race_id=race, completed_at=T0 + at)


KINDS = ("opus", "opus-bm", "mubit", "optuna")
FEATS = {"opus": [], "opus-bm": ["judge_policy", "median_stop"], "mubit": ["judge_policy", "median_stop"],
         "optuna": ["median_stop"]}


def _manifest(rid: str, kinds: tuple[str, ...] = KINDS, *, judge: str = "v1:5c1d3cfd2941", adapter: bool = True,
              feats: dict | None = None, runner: str = "modal", cost_cap: float = 1.0) -> dict:
    """A fairness manifest shaped like Race.fairness_manifest() (scheduler.py)."""
    feats = {k: v for k, v in (feats or FEATS).items() if k in kinds}
    llm = [k for k in kinds if k != "optuna"]
    rec = {k: ("full" if k == "mubit" else "off") for k in llm}
    per = {"budget_s": 100.0, "n_slots": 2, "runner": runner, "cost_cap_usd": cost_cap, "accept_unsure": True,
           "pipelining": False, "lookahead": 0, "max_proposals_per_call": 8, "run_baseline_first": True,
           "launch_cutoff_s": None, "target": "rl", "target_value": GOAL, "hardware": ["CPU", "CPU"],
           "price_per_s": [PRICE, PRICE], "record_modes": rec, "engine_features": feats,
           "median_stop_rule": {"fractions": [0.25, 0.5], "min_completed": 5}, "runtime_multiplier": None}
    m = {
        "identical": True, "per_arm": {k: dict(per) for k in kinds},
        "llm_policy": {k: {"researcher": "LLMResearcher", "max_retries": 1, "has_fallback": True,
                           "fallback_model": "claude-opus-5-5",
                           "prompts": f"propose_experiments=v1:fdda8c5b0919,diagnose_run=v0:22ecb45368e3,"
                                      f"judge_improvement={judge},early_stop=v2:f1f3bf21aa79"} for k in llm},
        "differs_only_in": {k: {"researcher": "OptunaResearcher" if k == "optuna" else "LLMResearcher",
                                "model": None if k == "optuna" else "claude-opus-5-5",
                                "llm_mode": "none" if k == "optuna" else "mubit", "record_mode": rec.get(k),
                                "engine_features": feats.get(k, [])} for k in kinds},
        "engine": {"features": feats, "per_arm": {}},
    }
    if adapter and "mubit" in kinds:  # request IDs checked by hand against the console's "Served by"
        m["adapter_served"] = {"request_ids": [f"mreq-{rid}-mubit-{i}" for i in range(4)]}
    return m


def _iv(db: Database, race: str, arm: str, slot: str, state: str, a: float, b: float) -> None:
    db.insert_gpu_interval(arm_id=arm, race_id=race, slot=slot, hardware="CPU", state=state,
                           start=T0 + a, end=T0 + b, experiment_id=None, price_per_s=PRICE)


def _race(db: Database, rid: str, seed: int, *, mode: str = "benchmark", llm_mode: str = "mubit",
          finish: bool = True, config: dict | None = None, name: str | None = None,
          manifest: dict | None | bool = True, runner: str = "modal") -> None:
    db.insert_race(race_id=rid, target="rl", budget_s=100, gpus_per_arm=2, seed=seed,
                   config=dict(CFG) if config is None else config, name=name or f"{PRESET_NAME} seed {seed}",
                   mode=mode, llm_mode=llm_mode, runner=runner)
    lm = llm_mode
    for kind in KINDS:
        aid = f"{rid}-{kind}"
        db.insert_arm(arm_id=aid, race_id=rid, kind=kind, model="claude-opus-5-5",
                      llm_mode="none" if kind == "optuna" else lm, n_slots=2, hardware="CPU", price_per_s=PRICE)
        _exp(db, rid, aid, f"{aid}-base", end=30, heldout=-1000.0, baseline=True)
        _iv(db, rid, aid, "s0", "busy", 0, 100)
    # Opus: slot 1 waits 50 s; reaches the goal (and its final quality -400) at t=80.
    a = f"{rid}-opus"
    _iv(db, rid, a, "s1", "idle", 0, 50)
    _iv(db, rid, a, "s1", "busy", 50, 100)
    _exp(db, rid, a, f"{a}-1", end=80, heldout=-400.0, best=True, verdict="valid")
    _call(db, rid, a, 0, cost=0.02, at=20)
    _call(db, rid, a, 1, cost=0.02, at=70)
    # Busy Machines with Opus: same LLM $, no waiting; goal at t=50, -350 at t=70.
    b = f"{rid}-opus-bm"
    _iv(db, rid, b, "s1", "busy", 0, 100)
    _exp(db, rid, b, f"{b}-1", end=50, heldout=-450.0, best=True)
    _exp(db, rid, b, f"{b}-2", end=70, heldout=-350.0, best=True)
    _call(db, rid, b, 0, cost=0.02, at=20)
    _call(db, rid, b, 1, cost=0.02, at=60)
    # Mubit + BM: goal at t=40, Opus quality at t=60, a caught trap, then an expensive call
    # ($0.10 at t=85) that pushes its spend past Opus's $0.24 before its best run (t=95).
    m = f"{rid}-mubit"
    _iv(db, rid, m, "s1", "idle", 0, 10)
    _iv(db, rid, m, "s1", "busy", 10, 100)
    _exp(db, rid, m, f"{m}-1", end=40, heldout=-450.0, best=True)
    _exp(db, rid, m, f"{m}-2", end=60, heldout=-300.0 - 10 * seed, best=True)
    _exp(db, rid, m, f"{m}-trap", end=70, heldout=-900.0, score=-50.0, trap=True, verdict="invalid")
    _exp(db, rid, m, f"{m}-3", end=95, heldout=-100.0, best=True)
    for i, (c, t) in enumerate(((0.001, 5), (0.001, 40), (0.001, 65), (0.1, 85))):
        _call(db, rid, m, i, cost=c, at=t)
    # Optuna: never reaches the goal; accepts a trap run as a new best.
    o = f"{rid}-optuna"
    _iv(db, rid, o, "s1", "busy", 0, 100)
    _exp(db, rid, o, f"{o}-trap", end=50, heldout=-700.0, score=-100.0, trap=True, best=True)
    _exp(db, rid, o, f"{o}-1", end=90, heldout=-600.0, best=True)
    if manifest is True:
        manifest = _manifest(rid, runner=runner)
    if manifest:
        db.update_race(rid, fairness=manifest)
    if finish:
        db.update_race(rid, status="finished", started_at=T0, ended_at=T0 + 100, finish_reason="budget")
    else:
        db.update_race(rid, status="running", started_at=T0)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "bench.db"
    db = Database(path)
    for s in (0, 1, 2):
        _race(db, f"bench{s}", s)
    db.close()
    return path


def _group(path: Path, **kw):
    groups, ex, notes = hl.run(db=path, **kw)
    assert len(groups) == 1, notes
    return groups[0]


def _f(g, fid: str):
    return next(f for f in g.framings if f.id == fid)


def test_columns_and_table_numbers(db_path: Path) -> None:
    g = _group(db_path)
    assert g.headline_ok and g.blockers == []
    assert list(g.arms) == ["opus", "opus_bm", "mubit", "optuna"]  # D36 order; "opus-bm" aliased
    o, b, m, p = (g.arms[k].stats for k in ("opus", "opus_bm", "mubit", "optuna"))
    # cost to the goal: GPU $ accrued by the hit + LLM $ completed by then
    assert o["cost_to_goal_usd"].mean == pytest.approx(0.16 + 0.04)
    assert b["cost_to_goal_usd"].mean == pytest.approx(0.10 + 0.02)
    assert m["cost_to_goal_usd"].mean == pytest.approx(0.08 + 0.002)
    assert p["cost_to_goal_usd"].n == 0 and p["cost_to_goal_usd"].n_total == 3
    assert m["time_to_goal_s"].mean == pytest.approx(40.0)
    assert o["gpu_idle_pct"].mean == pytest.approx(25.0)
    assert m["best_quality"].mean == pytest.approx(-100.0)
    # same quality: to the Opus arm's final held-out (-400) of the same race
    assert o["cost_to_opus_quality_usd"].mean == pytest.approx(0.20)
    assert b["cost_to_opus_quality_usd"].mean == pytest.approx(0.14 + 0.04)
    assert m["cost_to_opus_quality_usd"].mean == pytest.approx(0.12 + 0.002)
    assert m["time_to_opus_quality_s"].mean == pytest.approx(60.0)
    assert p["cost_to_opus_quality_usd"].n == 0
    # same money: best held-out while spend <= Opus's $0.24 (Mubit's t=95 run is past it)
    assert o["total_cost_usd"].mean == pytest.approx(0.24)
    assert m["total_cost_usd"].mean == pytest.approx(0.303)
    assert m["quality_at_opus_spend"].mean == pytest.approx(-310.0)
    assert m["quality_at_opus_spend"].std == pytest.approx(10.0)
    assert p["quality_at_opus_spend"].mean == pytest.approx(-600.0)  # spent less: whole race
    assert o["quality_at_opus_spend"].mean == pytest.approx(-400.0)
    assert m["experiments_per_usd"].mean == pytest.approx(5 / 0.303)
    assert o["experiments_per_usd"].mean == pytest.approx(2 / 0.24)
    assert g.arms["optuna"].trap_runs == 3 and p["fake_best_accepted"].mean == 1
    assert g.start_quality == pytest.approx(-1000.0)


def test_framings_follow_6b_formulas(db_path: Path) -> None:
    g = _group(db_path)
    cases = {
        # RL is framed as a penalty (= -return): N× = start / end
        "quality_vs_start:mubit": 1000 / 100,
        "quality_same_budget:mubit:opus": 400 / 100,
        "quality_same_money:mubit:opus": 400 / 310,
        "cost_to_goal:mubit:opus": 1 - 0.082 / 0.20,
        "time_to_goal:mubit:opus": 1 - 40 / 80,
        "cost_to_opus_quality:mubit:opus": 1 - 0.122 / 0.20,
        "time_to_opus_quality:mubit:opus": 1 - 60 / 80,
        "throughput:mubit:opus": (5 / 0.303) / (2 / 0.24),
        "cost_to_goal:opus_bm:opus": 1 - 0.12 / 0.20,
        "cost_to_opus_quality:mubit:opus_bm": 1 - 0.122 / 0.18,
        "quality_same_budget:opus:optuna": 600 / 400,
    }
    for fid, want in cases.items():
        f = _f(g, fid)
        assert f.measured, (fid, f.reasons)
        assert f.value == pytest.approx(want), fid
    s = _f(g, "cost_to_goal:mubit:opus").sentence
    assert "59% cheaper than the Opus 5.5 agent" in s and "same 2 slots" in s and "mean of 3 seeds" in s
    assert "Mubit + Busy Machines' best model has 4.0× lower penalty (75% lower) than the Opus 5.5 agent's, on" in _f(g, "quality_same_budget:mubit:opus").sentence
    assert "At the Opus 5.5 agent's total spend ($0.240" in _f(g, "quality_same_money:mubit:opus").sentence
    assert "50% faster than the Opus 5.5 agent" in _f(g, "time_to_goal:mubit:opus").sentence
    assert "final model quality 39% cheaper than the Opus 5.5 agent did" in \
        _f(g, "cost_to_opus_quality:mubit:opus").sentence
    # per-seed paired values are kept for the JSON
    assert _f(g, "quality_same_money:mubit:opus").per_seed == pytest.approx([400 / 300, 400 / 310, 400 / 320])


def test_worse_results_are_printed_as_worse(db_path: Path) -> None:
    g = _group(db_path)
    f = _f(g, "throughput:opus:optuna")  # 8.3 vs 15 per $
    assert f.measured and f.value == pytest.approx((2 / 0.24) / (3 / 0.2))
    assert "only 0.6× as many experiments per dollar (FEWER) as Optuna (no AI)" in f.sentence


def test_not_measured_when_a_side_misses_the_goal(db_path: Path) -> None:
    g = _group(db_path)
    f = _f(g, "cost_to_goal:mubit:optuna")
    assert not f.measured and f.value is None
    assert "not measured" in f.sentence and "reached the goal in 0/3 seeds" in f.sentence
    assert not _f(g, "fake_improvements").measured


def test_fewer_than_three_seeds_is_not_measured(db_path: Path) -> None:
    g = _group(db_path, race_ids=["bench0", "bench1"])
    assert all(not f.measured for f in g.framings)
    assert "needs >= 3" in _f(g, "cost_to_goal:mubit:opus").sentence
    # the table still has the numbers, with their n
    assert g.arms["mubit"].stats["cost_to_goal_usd"].n == 2


@pytest.mark.parametrize("variant", ["provisional", "dev", "unfinished"])
def test_gated_races_print_not_measured(tmp_path: Path, variant: str) -> None:
    path = tmp_path / f"{variant}.db"
    db = Database(path)
    for s in (0, 1, 2):
        if variant == "provisional":
            _race(db, f"r{s}", s, config={**CFG, "provisional": True})
        elif variant == "dev":
            _race(db, f"r{s}", s, mode="dev", llm_mode="mock")
        else:
            _race(db, f"r{s}", s, finish=(s != 2))
    db.close()
    g = _group(path)
    assert not g.headline_ok
    assert all(not f.measured for f in g.framings)
    word = {"provisional": "provisional goal", "dev": "dev/mock/replay", "unfinished": "not finished"}[variant]
    assert any(word in b for b in g.blockers)
    groups, ex, notes = hl.run(db=path)
    md = hl.render_markdown(groups, ex, notes)
    assert "NOT A HEADLINE RESULT" in md and word in md
    assert "cheaper than" not in md and "faster than" not in md


def test_evidence_counts_follow_each_window(db_path: Path) -> None:
    g = _group(db_path)
    ev = g.arms["mubit"].evidence
    whole = ev["gpu_idle_pct"]
    assert (whole.request_ids, whole.llm_calls, whole.modal_ids, whole.runs, whole.races) == (12, 12, 15, 15, 3)
    # goal at t=40: calls done by t=40 (t=5, t=40), runs started by t=40 (base, m1, m2)
    goal = ev["cost_to_goal_usd"]
    assert (goal.request_ids, goal.llm_calls, goal.modal_ids, goal.runs) == (6, 6, 9, 9)
    # same money: up to the last run inside Opus's spend (t=60): calls at 5, 40; runs started
    # by then: base, m1, m2 and the trap run (started t=50)
    money = ev["quality_at_opus_spend"]
    assert (money.request_ids, money.modal_ids) == (6, 12)
    assert g.arms["optuna"].evidence["gpu_idle_pct"].llm_calls == 0


def test_markdown_layout(db_path: Path) -> None:
    groups, ex, notes = hl.run(db=db_path)
    md = hl.render_markdown(groups, ex, notes)
    assert ("| | Standard AI assistant (Opus 5.5) | Busy Machines with Opus | Mubit + Busy Machines | "
            "Automatic search (no AI, Optuna) |") in md
    assert "| Cost to reach the goal | $0.200 ± 0.000 (n=3) | $0.120 ± 0.000 (n=3) | $0.082 ± 0.000 (n=3) | " \
           "didn't reach it (n=0/3) |" in md
    assert "-310 ± 10 (n=3)" in md
    assert "3 of 3 trap runs" in md and "0 of 3 trap runs" in md  # Optuna accepted each seed's trap; Mubit none
    assert "12/12 Mubit req IDs · 15/15 Modal IDs · 3 race(s)" in md
    assert "eligible for the headline" in md


def test_cli_json_and_preset_read_only(db_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = Database(db_path)
    _race(db, "other", 7, name="LCBench surrogate 15 min seed 7")  # not in the preset
    db.close()
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    out_json, out_md = tmp_path / "h.json", tmp_path / "h.md"
    rc = hl.main(["--db", str(db_path), "--preset", "rl-wide-15m", "--json", str(out_json), "--out", str(out_md)])
    assert rc == 0
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before  # opened mode=ro, nothing written
    text = capsys.readouterr().out
    assert out_md.read_text(encoding="utf-8") in text
    data = json.loads(out_json.read_text(encoding="utf-8"))
    (grp,) = data["groups"]
    assert grp["race_ids"] == ["bench0", "bench1", "bench2"] and grp["headline_ok"] is True
    assert [c["arm"] for c in grp["columns"]] == ["opus", "opus_bm", "mubit", "optuna"]
    fr = {f["id"]: f for f in grp["framings"]}
    assert fr["cost_to_goal:mubit:opus"]["value"] == pytest.approx(0.59)
    assert fr["cost_to_goal:mubit:optuna"]["value"] is None and not fr["cost_to_goal:mubit:optuna"]["measured"]
    assert grp["arms"]["mubit"]["evidence"]["cost_to_goal_usd"]["request_ids"] == 6
    assert grp["arms"]["mubit"]["stats"]["quality_at_opus_spend"]["n"] == 3
    with pytest.raises(SystemExit):
        hl.main(["--db", str(db_path), "--min-seeds", "2"])


def test_unknown_target_framing_and_aliases() -> None:
    assert hl.canon_arm("opus-bm") == hl.canon_arm("OPUS_BM") == "opus_bm"
    assert hl.canon_arm("mubit") == "mubit"
    # accuracy targets: (err_start - err_end) / err_start
    assert hl._framed_ratio("cifar", 0.95, 0.90) == pytest.approx((0.10 - 0.05) / 0.10)
    # bits per byte: start / end
    assert hl._framed_ratio("lm", 2.0, 3.0) == pytest.approx(1.5)
    # RL penalty undefined when the return is positive
    assert hl._framed_ratio("rl", 10.0, -100.0) is None


# --------------------------------------------------------------------------- fairness review (D41)


def _two_arm_race(db: Database, rid: str, seed: int, *, judge: str, cost_cap: float = 1.0) -> None:
    """An Opus vs Optuna race from before the engine (R38 shape): no engine features in the manifest."""
    db.insert_race(race_id=rid, target="rl", budget_s=100, gpus_per_arm=2, seed=seed, config=dict(CFG),
                   name=f"{PRESET_NAME} seed {seed}", mode="benchmark", llm_mode="mubit", runner="modal")
    for kind in ("opus", "optuna"):
        aid = f"{rid}-{kind}"
        db.insert_arm(arm_id=aid, race_id=rid, kind=kind, model="claude-opus-5-5",
                      llm_mode="none" if kind == "optuna" else "mubit", n_slots=2, hardware="CPU", price_per_s=PRICE)
        _exp(db, rid, aid, f"{aid}-base", end=30, heldout=-1000.0, baseline=True)
        _exp(db, rid, aid, f"{aid}-1", end=80, heldout=-400.0 if kind == "opus" else -600.0, best=True)
        _iv(db, rid, aid, "s0", "busy", 0, 100)
        _iv(db, rid, aid, "s1", "busy", 0, 100)
    _call(db, rid, f"{rid}-opus", 0, cost=0.02, at=20)
    _call(db, rid, f"{rid}-opus", 1, cost=0.02, at=70)
    m = _manifest(rid, ("opus", "optuna"), judge=judge, cost_cap=cost_cap)
    m.pop("engine")
    for pa in m["per_arm"].values():  # pre-engine manifests had no engine keys
        pa.pop("engine_features")
        pa.pop("median_stop_rule")
    for d in m["differs_only_in"].values():
        d.pop("engine_features")
    db.update_race(rid, fairness=m, status="finished", started_at=T0, ended_at=T0 + 100, finish_reason="budget")


def test_races_with_different_arm_sets_or_prompts_are_not_pooled(tmp_path: Path) -> None:
    path = tmp_path / "mixed.db"
    db = Database(path)
    for s in (0, 1, 2):
        _two_arm_race(db, f"old{s}", s, judge="v0:623366123052")  # R38-like: opus vs optuna, judge v0
        _race(db, f"new{s}", s)                                    # 4 arms, engine features, judge v1
    db.close()
    groups, _, notes = hl.run(db=path)
    assert sorted(sorted(g.race_ids) for g in groups) == [["new0", "new1", "new2"], ["old0", "old1", "old2"]]
    old = next(g for g in groups if g.race_ids[0].startswith("old"))
    new = next(g for g in groups if g is not old)
    assert list(old.arms) == ["opus", "optuna"] and old.headline_ok
    assert old.arms["opus"].stats["best_quality"].n == 3  # not diluted by the 4-arm races
    assert old.fingerprint_id and old.fingerprint_id != new.fingerprint_id
    (note,) = [n for n in notes if "NOT pooled" in n]
    assert "arm set" in note and "judge_improvement" in note
    assert "v0:623366123052" in note and "v1:5c1d3cfd2941" in note
    # same arm set, only the judge prompt version differs: still two setups
    path2 = tmp_path / "judge.db"
    db = Database(path2)
    for s in (0, 1, 2):
        _two_arm_race(db, f"a{s}", s, judge="v0:623366123052")
        _two_arm_race(db, f"b{s}", s, judge="v1:5c1d3cfd2941")
    db.close()
    groups, _, notes = hl.run(db=path2)
    assert len(groups) == 2 and any("judge_improvement" in n for n in notes)


def test_prompts_listed_but_never_called_do_not_split(tmp_path: Path) -> None:
    """R38: seed 0's policy has no early_stop entry; seeds 1-2 list early_stop v2 with 0 calls."""
    path = tmp_path / "unused.db"
    db = Database(path)
    for s in (0, 1, 2):
        _two_arm_race(db, f"r{s}", s, judge="v0:623366123052")
    f = db.get_race("r0")["fairness"]
    f["llm_policy"]["opus"]["prompts"] = "propose_experiments=v1:fdda8c5b0919,judge_improvement=v0:623366123052"
    db.update_race("r0", fairness=f)
    db.close()
    g = _group(path)
    assert g.race_ids == ["r0", "r1", "r2"]
    assert "early_stop=v2:f1f3bf21aa79" in g.unused_prompts["opus"]
    md = hl.render_markdown([g], hl.ar.Exclusions())
    assert "prompts listed but never called" in md


def test_mubit_framings_need_evidence_an_adapter_served(tmp_path: Path) -> None:
    path = tmp_path / "noadapter.db"
    db = Database(path)
    for s in (0, 1, 2):
        _race(db, f"r{s}", s, manifest=_manifest(f"r{s}", adapter=False))
    db.close()
    g = _group(path)
    assert g.headline_ok and not g.adapter_ok and g.adapter == [(0, 4)] * 3
    mubit = [f for f in g.framings if "mubit" in (f.arm, f.baseline)]
    assert mubit and all(not f.measured for f in mubit)
    for f in mubit:
        if f.id != "fake_improvements":
            assert "not measured: no evidence an adapter served" in f.sentence, f.sentence
    assert _f(g, "cost_to_goal:opus_bm:opus").measured  # framings without the Mubit arm are unaffected
    md = hl.render_markdown([g], hl.ar.Exclusions())
    assert "0 of 12 Mubit-arm calls carry evidence an adapter answered" in md
    assert "Mubit + Busy Machines reached" not in md and "Mubit + Busy Machines' best model" not in md
    # evidence in only some races is not enough
    path2 = tmp_path / "partial.db"
    db = Database(path2)
    for s in (0, 1, 2):
        _race(db, f"r{s}", s, manifest=_manifest(f"r{s}", adapter=(s != 1)))
    db.close()
    g2 = _group(path2)
    assert not g2.adapter_ok and not _f(g2, "cost_to_goal:mubit:opus").measured


def test_no_manifest_or_a_correction_blocks_the_setup(tmp_path: Path) -> None:
    path = tmp_path / "nomanifest.db"
    db = Database(path)
    for s in (0, 1, 2):
        _race(db, f"r{s}", s, manifest=None)
    db.close()
    g = _group(path)
    assert not g.headline_ok and g.fingerprint_id is None
    assert any("no fairness manifest" in b for b in g.blockers)

    path2 = tmp_path / "corr.db"
    db = Database(path2)
    for s in (0, 1, 2):
        _race(db, f"r{s}", s)
    db.insert_correction(experiment_id="r1-mubit-1", source="human", label="fake_improvement", race_id="r1",
                         arm_id="r1-mubit")
    db.close()
    g2 = _group(path2)
    assert not g2.headline_ok and any("correction" in b for b in g2.blockers)
    assert all(not f.measured for f in g2.framings)


def _stopped(db: Database, rid: str, aid: str, es: dict) -> None:
    db.upsert_experiment({
        "id": f"{aid}-stopped", "arm_id": aid, "race_id": rid, "status": "cancelled", "score": None,
        "proposed_at": T0 + 10, "started_at": T0 + 10, "ended_at": T0 + 20, "run_ref": f"fc-{aid}-stopped",
        "seed": 0, "summary": {"early_stop": es},
    })


def test_features_per_column_slot_time_freed_and_degenerate_rule(tmp_path: Path) -> None:
    path = tmp_path / "engine.db"
    db = Database(path)
    for s in (0, 1, 2):
        rid = f"r{s}"
        _race(db, rid, s, runner="surrogate", manifest=_manifest(rid, runner="surrogate"))
        aid = f"{rid}-opus-bm"
        # an early-stopped run (40 s of slot time freed; its "$" must never be shown) and a
        # judged claim under a sigma = 0 table (LCBench's deterministic surrogate)
        _stopped(db, rid, aid, {"outcome": "stopped", "saved_s": 40.0, "saved_usd": 0.987654})
        db.upsert_experiment({
            "id": f"{aid}-judged", "arm_id": aid, "race_id": rid, "status": "succeeded", "score": -450.0,
            "metrics": {"heldout_score": -450.0}, "proposed_at": T0 + 20, "started_at": T0 + 25, "ended_at": T0 + 45,
            "run_ref": f"fc-{aid}-judged", "seed": 0,
            "summary": {"judge_policy": {"llm_verdict": "unsure", "llm_replied": True, "rule": "real",
                                         "final": "valid", "sigma": 0.0}},
        })
    db.close()
    g = _group(path)
    md = hl.render_markdown([g], hl.ar.Exclusions())
    assert ("| Engine features (Busy Machines) | none | judge policy + median stop | judge policy + median stop "
            "| median stop |") in md
    assert "claude-opus-5-5 · record off · judge v1, propose v1" in md and "| no LLM |" in md
    e = g.engine["opus_bm"]
    assert e.early_stopped == 3 and e.slot_s_freed == pytest.approx(120.0) and e.stops_unconfirmed == 0
    assert "2.0 slot-min freed" in md
    assert "0.98" not in md and "saved" not in md.lower()  # never in $ (D41 #5)
    assert e.degenerate and "DEGENERATE" in md and "excluded from real-gain stats" in md
    assert hl.JUDGE_POLICY_NOTE in md
    data = hl.to_json([g], hl.ar.Exclusions())
    eng = data["groups"][0]["engine"]["opus_bm"]
    assert eng["real_gain_stats_usable"] is False and "saved_usd" not in json.dumps(eng)


def test_unconfirmed_remote_early_stops_gate_that_arm(tmp_path: Path) -> None:
    path = tmp_path / "remote.db"
    db = Database(path)
    for s in (0, 1, 2):
        _race(db, f"r{s}", s)  # runner modal
        _stopped(db, f"r{s}", f"r{s}-opus-bm", {"outcome": "stopped", "saved_s": 5.0})
    db.close()
    g = _group(path)
    f = _f(g, "cost_to_goal:opus_bm:opus")
    assert not f.measured and "D41 #1" in f.sentence
    assert _f(g, "cost_to_goal:mubit:opus").measured  # arms without early stops are unaffected
    # once the runner confirms each stop, the gate lifts
    path2 = tmp_path / "confirmed.db"
    db = Database(path2)
    for s in (0, 1, 2):
        _race(db, f"r{s}", s)
        _stopped(db, f"r{s}", f"r{s}-opus-bm", {"outcome": "stopped", "saved_s": 5.0, "cancel_lag_s": 1.2})
    db.close()
    assert _f(_group(path2), "cost_to_goal:opus_bm:opus").measured
    # the exact keys the scheduler writes (scheduler._early_stop_outcome) also lift the gate
    path3 = tmp_path / "scheduler_keys.db"
    db = Database(path3)
    for s in (0, 1, 2):
        _race(db, f"r{s}", s)
        _stopped(db, f"r{s}", f"r{s}-opus-bm", {"outcome": "stopped", "slot_s_freed": 5.0,
                                                 "runner_cancel_lag_s": 2.7, "cancel_confirmed": True})
    db.close()
    assert _f(_group(path3), "cost_to_goal:opus_bm:opus").measured
