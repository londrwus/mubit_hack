"""scripts/analyze_races.py on a synthetic DB built with backend/db.py. No LLM, no Modal."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from backend.benchmark.export import export_race
from backend.core.types import ArmKind, LLMCallRecord, StepType
from backend.db import Database

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analyze_races", REPO / "scripts" / "analyze_races.py")
assert _spec and _spec.loader
ar = importlib.util.module_from_spec(_spec)
sys.modules["analyze_races"] = ar
_spec.loader.exec_module(ar)

T0 = 1000.0
PRICE = 0.001  # $/s per slot
TV = {"rl": -300.0}


def _exp(db: Database, race: str, arm: str, eid: str, *, end: float, heldout: float, score: float | None = None,
         baseline: bool = False, trap: bool = False, best: bool = False, verdict: str | None = None,
         status: str = "succeeded", run_ref: str | None = "fc-x") -> None:
    db.upsert_experiment({
        "id": eid, "arm_id": arm, "race_id": race, "status": status,
        "score": (heldout if score is None else score) if status == "succeeded" else None,
        "metrics": {"heldout_score": heldout} if status == "succeeded" else {},
        "is_baseline": baseline, "is_trap": trap, "is_best": best, "judge_verdict": verdict,
        "proposed_at": end - 25, "started_at": end - 20, "ended_at": end, "run_ref": run_ref, "seed": 0,
    })


def _call(db: Database, race: str, arm_id: str, arm: ArmKind, *, latency: float, cost: float, at: float,
          valid: bool = True, fallback: bool = False, req: str | None = "mreq_1", session: str | None = None,
          step: StepType = StepType.propose_experiments) -> None:
    rec = LLMCallRecord(arm_id=arm_id, arm=arm, session_id=session or f"race-{race}/{arm.value}/seed-0",
                        step_type=step, model="claude-opus-5-5", latency_ms=latency, cost_usd=cost,
                        valid=valid, is_fallback=fallback, mubit_request_id=req, created_at=at - latency / 1000)
    db.insert_llm_call(rec, race_id=race, completed_at=at)


def _iv(db: Database, race: str, arm: str, slot: str, state: str, a: float, b: float) -> None:
    db.insert_gpu_interval(arm_id=arm, race_id=race, slot=slot, hardware="CPU", state=state,
                           start=T0 + a, end=T0 + b, experiment_id=None, price_per_s=PRICE)


def _race(db: Database, rid: str, seed: int, *, mode: str = "benchmark", llm_mode: str = "mubit",
          name: str = "rl bench", finish: bool = True, config: dict | None = None) -> None:
    db.insert_race(race_id=rid, target="rl", budget_s=100, gpus_per_arm=2, seed=seed,
                   config=config if config is not None else {"target": "rl"},
                   name=name, mode=mode, llm_mode=llm_mode, runner="modal")
    kinds = [("opus", "mubit"), ("mubit", "mubit"), ("optuna", "none")] if mode == "benchmark" else \
        [("mock", "mock"), ("optuna", "none")]
    for kind, lm in kinds:
        aid = f"{rid}-{kind}"
        db.insert_arm(arm_id=aid, race_id=rid, kind=kind, model="claude-opus-5-5", llm_mode=lm, n_slots=2,
                      hardware="CPU", price_per_s=PRICE)
        _exp(db, rid, aid, f"{aid}-base", end=T0 + 30, heldout=-1000.0, baseline=True)
        _iv(db, rid, aid, "s0", "busy", 0, 100)
    # opus: slot 1 waits 50 s, never reaches -300; 2 calls, one malformed
    a = f"{rid}-opus"
    if mode == "benchmark":
        _iv(db, rid, a, "s1", "idle", 0, 50)
        _iv(db, rid, a, "s1", "busy", 50, 100)
        _exp(db, rid, a, f"{a}-1", end=T0 + 80, heldout=-400.0, best=True, verdict="valid")
        _call(db, rid, a, ArmKind.opus, latency=6000, cost=0.02, at=T0 + 20)
        _call(db, rid, a, ArmKind.opus, latency=8000, cost=0.02, at=T0 + 70, valid=False, req=None)
        # mubit: slot 1 waits 10 s, reaches the goal at t0+60; a trap run the judge caught
        m = f"{rid}-mubit"
        _iv(db, rid, m, "s1", "idle", 0, 10)
        _iv(db, rid, m, "s1", "busy", 10, 100)
        _exp(db, rid, m, f"{m}-1", end=T0 + 60, heldout=-250.0 + 10 * seed, best=True, verdict="valid")
        _exp(db, rid, m, f"{m}-trap", end=T0 + 70, heldout=-900.0, score=-50.0, trap=True, verdict="invalid")
        _exp(db, rid, m, f"{m}-fail", end=T0 + 75, heldout=0.0, status="failed")
        _call(db, rid, m, ArmKind.mubit, latency=1000, cost=0.001, at=T0 + 5)
        _call(db, rid, m, ArmKind.mubit, latency=1500, cost=0.001, at=T0 + 40, fallback=True)
        _call(db, rid, m, ArmKind.mubit, latency=900, cost=0.001, at=T0 + 65, step=StepType.judge_improvement)
        # an example-collection probe call that leaked into the race: must be excluded
        _call(db, rid, m, ArmKind.mubit, latency=50, cost=5.0, at=T0 + 1, session="race-probe-latency/mubit/seed-0")
    o = f"{rid}-optuna"
    _iv(db, rid, o, "s1", "busy", 0, 100)
    _exp(db, rid, o, f"{o}-1", end=T0 + 90, heldout=-350.0, best=True)
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
    _race(db, "devrace", 0, mode="dev", llm_mode="mock", name="dev local race")
    _race(db, "probe1", 0, name="collection probe propose")
    _race(db, "halfway", 3, finish=False)
    db.close()
    return path


def _reports(path: Path, **kw):
    kept, ex = ar.screen(ar.load_db(path), **kw)
    return ar.analyse(kept, target_values=TV), ex


def test_goal_comes_from_race_config(tmp_path: Path) -> None:
    """Without --target-value the goal is the one each race ran with (bench.py stores it in the
    RaceConfig), not the spec placeholder: same rule as backend.benchmark.analysis."""
    path = tmp_path / "goal.db"
    db = Database(path)
    for s in (0, 1):
        _race(db, f"g{s}", s, config={"target": "rl", "target_value": -245.0, "provisional": False})
    db.close()
    kept, _ = ar.screen(ar.load_db(path))
    (rep,) = ar.analyse(kept)
    assert rep.target_value == -245.0 and rep.target_value_source.startswith("race config")
    # mubit seed 0 reaches -250 (< -245: no), seed 1 reaches -240 (yes)
    assert rep.arms["mubit"].stats["time_to_target_s"].n == 1
    (forced,) = ar.analyse(kept, target_values={"rl": -300.0})
    assert forced.target_value == -300.0 and forced.target_value_source == "--target-value"


def test_mixed_race_goals_are_not_aggregated(tmp_path: Path) -> None:
    path = tmp_path / "mixed.db"
    db = Database(path)
    _race(db, "m0", 0, config={"target": "rl", "target_value": -245.0})
    _race(db, "m1", 1, config={"target": "rl", "target_value": -165.0, "provisional": True})
    db.close()
    kept, _ = ar.screen(ar.load_db(path))
    assert ar.analyse(kept) == []


def test_screen_excludes_dev_probe_unfinished(db_path: Path) -> None:
    kept, ex = ar.screen(ar.load_db(db_path))
    assert sorted(r["races"][0]["id"] for r, _ in kept) == ["bench0", "bench1", "bench2"]
    why = dict(ex.races)
    assert "dev race" in why["devrace"] and "mock" in why["devrace"]
    assert why["probe1"] == "collection/probe race"
    assert why["halfway"].startswith("not finished")
    assert ex.llm_calls == {"probe session": 3}


def test_numbers(db_path: Path) -> None:
    reports, _ = _reports(db_path)
    assert len(reports) == 1
    rep = reports[0]
    assert rep.target == "rl" and rep.direction == "max" and rep.target_value == -300.0
    assert list(rep.arms) == ["mubit", "opus", "optuna"]
    assert rep.start_score == pytest.approx(-1000.0)
    assert rep.best_known == pytest.approx(-230.0)  # mubit seed 2
    m, o, op = (rep.arms[k].stats for k in ("mubit", "opus", "optuna"))

    assert m["gpu_idle_pct"].mean == pytest.approx(5.0)
    assert o["gpu_idle_pct"].mean == pytest.approx(25.0)
    assert op["gpu_idle_pct"].mean == pytest.approx(0.0)
    # mubit reaches the goal at t0+60: GPU 2 slots x 60 s x $0.001 + 2 LLM calls done by then
    assert m["time_to_target_s"].mean == pytest.approx(60.0)
    assert m["cost_to_target_usd"].mean == pytest.approx(0.12 + 0.002)
    assert m["cost_to_target_usd"].n == 3
    assert o["cost_to_target_usd"].n == 0 and o["cost_to_target_usd"].n_total == 3
    # held-out quality: the trap's reported -50 is ignored, the judge threw it away anyway
    assert m["best_quality"].mean == pytest.approx(-240.0)
    assert m["best_quality"].std == pytest.approx(10.0)
    assert m["normalised_quality"].mean == pytest.approx(100 * statistics_mean([750, 760, 770]) / 770)
    assert o["normalised_quality"].mean == pytest.approx(100 * 600 / 770)
    assert m["experiments_finished"].mean == 3  # baseline, good run, trap (failed one not finished)
    assert m["llm_cost_usd"].mean == pytest.approx(0.003)  # probe call ($5) excluded
    assert o["llm_cost_usd"].mean == pytest.approx(0.04)
    assert o["malformed_rate"].mean == pytest.approx(0.5)
    assert m["fallback_rate"].mean == pytest.approx(1 / 3)
    assert o["llm_p50_latency_ms"].mean == pytest.approx(7000)
    assert m["llm_p50_latency_ms"].mean == pytest.approx(1000)
    assert op["llm_calls"].mean == 0
    a = rep.arms["mubit"]
    assert a.p50_by_step["judge_improvement"].mean == pytest.approx(900)
    assert a.calls_by_step == {"judge_improvement": 3, "propose_experiments": 6}
    assert a.evidence == {"llm_calls": 9, "request_ids": 9, "experiments": 12, "run_ids": 12, "corrections": 0}
    assert rep.arms["opus"].evidence["request_ids"] == 3
    assert rep.warnings == []


def statistics_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def test_jsonl_export_gives_same_numbers(db_path: Path, tmp_path: Path) -> None:
    db = Database(db_path)
    paths = [export_race(db, f"bench{s}", tmp_path / f"bench{s}.jsonl") for s in (0, 1, 2)]
    db.close()
    from_db, _ = _reports(db_path)
    kept, _ = ar.screen(ar.load_jsonl(paths))
    from_jsonl = ar.analyse(kept, target_values=TV)
    for kind in ("mubit", "opus", "optuna"):
        for f in ar.AGG_FIELDS:
            a, b = from_db[0].arms[kind].stats[f], from_jsonl[0].arms[kind].stats[f]
            assert (a.mean is None and b.mean is None) or math.isclose(a.mean, b.mean, rel_tol=1e-9, abs_tol=1e-12), f


def test_markdown(db_path: Path) -> None:
    reports, ex = _reports(db_path)
    md = ar.render_markdown(reports, ex)
    assert "| Arm | Seeds | Cost to reach the goal |" in md
    assert "didn't reach it (0/3)" in md
    assert "$0.12 ± 0.00" in md
    assert "5.0 ± 0.0%" in md
    assert "9/9 calls with Mubit request ID" in md
    assert "race `devrace`" in md and "race `probe1`" in md and "race `halfway`" in md
    assert "3 LLM call row(s): probe session" in md
    assert "| Game-playing AI | Busy Machines (Mubit) | 3 |" in md
    assert ar.DEV_STAMP not in md


def test_allow_dev_is_stamped(db_path: Path) -> None:
    reports, ex = _reports(db_path, allow_dev=True)
    rep = reports[0]
    assert rep.is_dev and "devrace" in rep.race_ids and "probe1" not in rep.race_ids
    assert "mock" in rep.arms
    assert ar.DEV_STAMP in ar.render_markdown(reports, ex)


def test_normalise_min_direction() -> None:
    # lower is better: start 3.0, best 2.0 -> 2.5 is halfway
    assert ar.normalise(2.5, 3.0, 2.0) == pytest.approx(50.0)
    assert ar.normalise(3.2, 3.0, 2.0) == pytest.approx(-20.0)
    assert ar.normalise(None, 3.0, 2.0) is None
    assert ar.normalise(2.0, 2.0, 2.0) is None


def test_charts(db_path: Path, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    reports, _ = _reports(db_path)
    paths = ar.render_charts(reports, tmp_path / "charts")
    assert sorted(p.name for p in paths) == [
        "rl-cost-to-goal.png", "rl-experiments-finished.png", "rl-gpus-waiting.png", "rl-model-quality.png",
    ]
    assert all(p.stat().st_size > 5000 for p in paths)


def test_cli(db_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_md, out_json = tmp_path / "r.md", tmp_path / "r.json"
    rc = ar.main(["--db", str(db_path), "--race", "bench0", "--race", "bench1", "--target-value", "rl=-300",
                  "--no-charts", "--out", str(out_md), "--json", str(out_json)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "Busy Machines (Mubit)" in text and "| 0,1 |" in text
    assert out_md.read_text(encoding="utf-8") in text
    import json

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["reports"][0]["race_ids"] == ["bench0", "bench1"]
    with pytest.raises(KeyError):
        ar.load_db(db_path, ["nope"])


def _fairness(judge: str, *, prompts_extra: str = "", runtime_ref: bool = False) -> dict:
    per = {"budget_s": 100.0, "n_slots": 2, "runner": "modal", "cost_cap_usd": 1.0, "accept_unsure": True,
           "hardware": ["CPU", "CPU"], "price_per_s": [PRICE, PRICE], "record_modes": {"opus": "off", "mubit": "full"}}
    pol = {"researcher": "LLMResearcher", "max_retries": 1, "has_fallback": True, "fallback_model": "claude-opus-5-5",
           "prompts": f"propose_experiments=v1:fdda8c5b0919,judge_improvement={judge}{prompts_extra}"}
    return {"identical": True, "per_arm": {k: dict(per) for k in ("opus", "mubit", "optuna")},
            "llm_policy": {"opus": dict(pol), "mubit": dict(pol)}}


def test_fairness_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "fp.db"
    db = Database(path)
    for s in (0, 1, 2):
        _race(db, f"f{s}", s)
    db.update_race("f0", fairness=_fairness("v0:623366123052"))
    # lists a prompt the race never called: same fingerprint
    db.update_race("f1", fairness=_fairness("v0:623366123052", prompts_extra=",early_stop=v2:f1f3bf21aa79"))
    db.update_race("f2", fairness=_fairness("v1:5c1d3cfd2941"))  # a judge prompt the race did call
    db.close()
    races = {r["races"][0]["id"]: r for r in ar.load_db(path)}
    fps = {k: ar.fairness_fingerprint(v) for k, v in races.items()}
    ids = {k: ar.fingerprint_id(v) for k, v in fps.items()}
    assert ids["f0"] == ids["f1"] != ids["f2"]
    assert fps["f1"]["unused_prompts"] == {"mubit": ["early_stop=v2:f1f3bf21aa79"],
                                           "opus": ["early_stop=v2:f1f3bf21aa79"]}
    assert fps["f0"]["arms"]["opus"]["llm_policy"]["prompts"] == {
        "judge_improvement": "v0:623366123052", "propose_experiments": "v1:fdda8c5b0919"}
    assert fps["f0"]["arms"]["opus"]["record_mode"] == "off" and fps["f0"]["arms"]["optuna"]["engine_features"] == []
    diff = ar.fingerprint_diff(fps["f0"], fps["f2"])
    assert "arms.mubit.llm_policy.prompts.judge_improvement: v0:623366123052 vs v1:5c1d3cfd2941" in diff
    # no manifest -> None; analyse never pools unlike setups: f0 + f1 (same fingerprint) are one
    # report, f2 (a different judge prompt) is its own, and both say the target has two setups
    assert ar.fairness_fingerprint({"races": [{"id": "x"}], "arms": []}) is None
    kept, _ = ar.screen(list(races.values()))
    reps = ar.analyse(kept, target_values=TV)
    assert sorted(sorted(r.race_ids) for r in reps) == [["f0", "f1"], ["f2"]]
    assert all(r.n_setups == 2 and r.setup_id == ids[r.race_ids[0]] for r in reps)
    assert all(any("2 different setups" in w for w in r.warnings) for r in reps)
    assert not any("fairness fingerprint" in w and "differs" in w for r in reps for w in r.warnings)


def test_runtime_multiplier_inferred_from_run_refs() -> None:
    rows = {"races": [{"id": "s", "fairness": {"per_arm": {"opus": {"runner": "surrogate"}}}}], "arms": [],
            "experiments": [{"run_ref": "surrogate:abc:x1"}, {"run_ref": "surrogate:def:x1"}]}
    assert ar.fairness_fingerprint(rows)["race"]["runtime_multiplier"] == 1.0
    rows["races"][0]["fairness"]["per_arm"]["opus"]["runtime_multiplier"] = 1.0
    assert ar.fairness_fingerprint(rows)["race"]["runtime_multiplier"] == 1.0
