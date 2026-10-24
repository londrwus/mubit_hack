"""Fairness-review fixes (D41, 05:55 review): early-stop cancel accounting, corrections refused
in benchmark races (or recorded in the manifest), and the minors (failed-before-stop kept as a
failure, slot time freed instead of '$ saved', strict default engine features in benchmarks,
project routing only on the mubit arm + in the manifest, opus_bm model, no-reply judge ->
unsure, degenerate sigma-0 rule, cost_source on mubit-arm calls).

In-process fakes only (bm-curve target and fakes from test_busy_machines). No network."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.agent.loop import DefaultPrompts, LLMResearcher
from backend.agent.mubit_client import (
    COST_SOURCE_LIST,
    COST_SOURCE_MUBIT,
    DEFAULT_ARM_MODELS,
    ProjectRouting,
)
from backend.app import create_app
from backend.benchmark.configs import OPUS_MODEL, PRESETS, RaceConfig
from backend.benchmark.engine_rules import SeedSpreadRule, SigmaTable, combine, load_sigma_tables
from backend.benchmark.metrics import engine_extras
from backend.core.types import ArmKind, CorrectionLabel, ExperimentStatus, RunHandle, RunResult, Verdict, now
from backend.db import BenchmarkIntegrityError, Database
from backend.scheduler import (
    RUNNER_ACCOUNTING_METRIC_KEYS,
    ArmDeps,
    AsymmetricRaceError,
    CorrectionRefusedError,
    EventBus,
    RaceEngine,
    _Exp,
    agent_visible_metrics,
)
from test_busy_machines import SPEC, CurveRunner, ScriptedLLM, llm_deps, race, sigma_tables  # noqa: F401

# =========================================================================== (1) cancel accounting


class LagRunner(CurveRunner):
    """cancel() returns at once, but the run only stops ``lag_s`` later (like Modal: the remote
    input keeps training until the cancel lands) and wait() reports it then, with the runner's
    cancel_lag_s accounting metric."""

    def __init__(self, n_slots: int, *, lag_s: float, fail_on_cancel: bool = False, **kw: Any) -> None:
        super().__init__(n_slots, **kw)
        self.lag_s = lag_s
        self.fail_on_cancel = fail_on_cancel
        self.cancel_at: dict[str, float] = {}

    async def cancel(self, handle: RunHandle) -> None:
        t = self._tasks.get(handle.run_id)
        if t is None or t.done() or handle.run_id in self.cancel_at:
            return
        self.cancelled.append(handle.run_id)
        self.cancel_at[handle.run_id] = now()

        async def later() -> None:
            await asyncio.sleep(self.lag_s)
            t.cancel()

        asyncio.get_running_loop().create_task(later())

    async def wait(self, handle: RunHandle) -> RunResult:
        res = await self._tasks[handle.run_id]
        if handle.run_id in self.cancel_at and res.status == ExperimentStatus.cancelled:
            res.metrics["cancel_lag_s"] = now() - self.cancel_at[handle.run_id]
            res.metrics["cancel_confirmed"] = 1.0
            if self.fail_on_cancel:   # the run diverged just before the cancel landed
                res = res.model_copy(update={"status": ExperimentStatus.failed, "error": "nan loss at step 3"})
        return res


async def test_early_stop_keeps_the_slot_busy_until_the_runner_confirms(sigma_tables: Any) -> None:
    lag = 0.25
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus_bm], budget_s=5.0, gpus_per_arm=3, llm="mock", seed=2,
                     engine_features={ArmKind.opus_bm: ["judge_policy", "median_stop"]})   # D43: opt-in only
    runner = LagRunner(3, lag_s=lag)
    eng = await race(cfg, [llm_deps(ArmKind.opus_bm, "lag", runner, ScriptedLLM(seed=2))])
    arm = eng.arms[ArmKind.opus_bm]
    stopped = [e for e in arm.exps.values() if (e.early_stop or {}).get("outcome") == "stopped"]
    assert stopped, "the median rule never fired"
    for e in stopped:
        es = e.early_stop
        assert es["stop_latency_s"] >= lag - 0.02                    # the lag is inside the stop latency
        assert es["runner_cancel_lag_s"] >= lag - 0.02 and es["cancel_confirmed"] is True
        busy = next(iv for iv in arm.intervals if iv["experiment_id"] == e.id and iv["state"] == "busy")
        assert busy["end"] == pytest.approx(e.ended_at)                # slot busy until the stop landed
        assert e.ended_at >= es["decided_at"] + lag - 0.02
        assert e.gpu_cost_usd == pytest.approx((e.ended_at - e.started_at) * runner.price)  # paid by this arm
        assert "cancel_lag_s" not in e.snapshot().metrics             # runner accounting: hidden
    rep = eng.reports["opus_bm"]
    assert rep["early_stop_runner_cancel_lag_p50_s"] >= lag - 0.02
    assert rep["early_stop_cancel_unconfirmed"] == 0


async def test_a_failure_before_the_stop_lands_stays_a_failure_and_is_diagnosed(sigma_tables: Any) -> None:
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus_bm], budget_s=5.0, gpus_per_arm=3, llm="mock", seed=2,
                     engine_features={ArmKind.opus_bm: ["judge_policy", "median_stop"]})   # D43: opt-in only
    llm = ScriptedLLM(seed=2)
    eng = await race(cfg, [llm_deps(ArmKind.opus_bm, "fail", LagRunner(3, lag_s=0.05, fail_on_cancel=True), llm)])
    arm = eng.arms[ArmKind.opus_bm]
    es = [e for e in arm.exps.values() if e.early_stop is not None and e.early_stop.get("outcome") != "race_ended"]
    assert es, "the median rule never fired"
    assert not [e for e in es if e.early_stop["outcome"] == "stopped"]
    failed = [e for e in es if e.early_stop["outcome"] == "failed_before_stop"]
    assert failed
    for e in failed:
        assert e.status == ExperimentStatus.failed and e.error == "nan loss at step 3"
        assert "slot_s_freed" not in e.early_stop
    assert any(e.diagnosis is not None for e in failed) and llm.calls["diagnose_run"] >= 1
    assert eng.reports["opus_bm"]["early_stop_failed_first"] == len(failed)
    assert eng.reports["opus_bm"]["early_stopped"] == 0


def test_runner_accounting_metrics_are_hidden_from_researchers() -> None:
    assert {"cancel_lag_s", "cancel_confirmed"} <= RUNNER_ACCOUNTING_METRIC_KEYS
    assert agent_visible_metrics({"cancel_lag_s": 2.0, "cancel_confirmed": 1.0, "loss": 0.5}) == {"loss": 0.5}


def test_engine_extras_slot_time_freed_reads_old_and_new_records() -> None:
    new = {"summary": {"early_stop": {"outcome": "stopped", "slot_s_freed": 3.0, "freed_slot_usd_equiv": 0.3,
                                      "runner_cancel_lag_s": 2.0, "cancel_confirmed": True}}}
    old = {"summary": {"early_stop": {"outcome": "stopped", "saved_s": 1.0, "saved_usd": 0.1}}}
    x = engine_extras([new, old])
    assert x["early_stop_slot_s_freed"] == 4.0 and x["early_stop_freed_slot_usd_equiv"] == pytest.approx(0.4)
    assert x["early_stop_saved_s"] == 4.0 and "early_stop_saved_usd" not in x
    assert x["early_stop_runner_cancel_lag_p50_s"] == 2.0


# =========================================================================== (4) corrections


def _bench_cfg(**kw: Any) -> RaceConfig:
    base: dict[str, Any] = dict(target="bm-curve", arms=[ArmKind.opus, ArmKind.opus_bm], budget_s=1.5,
                                gpus_per_arm=2, mode="benchmark", llm="mubit",
                                record_mode={ArmKind.opus: "off", ArmKind.opus_bm: "off"})
    base.update(kw)
    return RaceConfig(**base)


def _bench_engine(cfg: RaceConfig, db: Database | None = None) -> RaceEngine:
    deps = [llm_deps(k, "b", CurveRunner(cfg.gpus_per_arm), ScriptedLLM(seed=3), llm_mode="mubit") for k in cfg.arms]
    return RaceEngine(cfg, deps, spec=SPEC, db=db, bus=EventBus(), tick_s=0.02)


async def _first_exp(eng: RaceEngine) -> tuple[asyncio.Task[Any], str]:
    task = asyncio.create_task(eng.run())
    for _ in range(100):
        await asyncio.sleep(0.02)
        ids = [i for a in eng.arms.values() for i in a.exps]
        if ids:
            return task, ids[0]
    raise AssertionError("no experiment started")


async def test_benchmark_race_refuses_corrections(sigma_tables: Any) -> None:
    eng = _bench_engine(_bench_cfg())
    task, exp_id = await _first_exp(eng)
    with pytest.raises(CorrectionRefusedError):
        eng.apply_correction(exp_id, CorrectionLabel.fake_improvement, correction_id="c1")
    arm = eng.find_arm(exp_id)
    assert arm is not None
    with pytest.raises(CorrectionRefusedError):   # the arm refuses an unrecorded one too
        arm.apply_correction(exp_id, CorrectionLabel.fake_improvement)
    await task
    assert eng.fairness_manifest()["corrections"] == []
    assert eng.fairness_manifest()["corrections_policy"] == "refuse"


async def test_benchmark_race_with_record_lists_corrections_in_the_manifest(sigma_tables: Any) -> None:
    db = Database(":memory:")
    eng = _bench_engine(_bench_cfg(benchmark_corrections="record"), db)
    task, exp_id = await _first_exp(eng)
    assert eng.apply_correction(exp_id, CorrectionLabel.fake_improvement, correction_id="c1", reason="leak") is not None
    await task
    corr = eng.fairness_manifest()["corrections"]
    assert [c["correction_id"] for c in corr] == ["c1"] and corr[0]["label"] == "fake_improvement"
    assert corr[0]["reason"] == "leak" and corr[0]["elapsed_s"] is not None
    stored = db.get_race(eng.race_id)["fairness"]
    assert [c["correction_id"] for c in stored["corrections"]] == ["c1"]
    assert PRESETS["demo-3m"].benchmark_corrections == "record"   # the demo shows a correction
    assert all(p.benchmark_corrections == "refuse" for n, p in PRESETS.items() if n != "demo-3m")


async def test_dev_race_accepts_corrections(sigma_tables: Any) -> None:
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus], budget_s=1.0, gpus_per_arm=2, llm="mock")
    eng = RaceEngine(cfg, [llm_deps(ArmKind.opus, "d", CurveRunner(2))], spec=SPEC, bus=EventBus(), tick_s=0.02)
    task, exp_id = await _first_exp(eng)
    assert eng.apply_correction(exp_id, CorrectionLabel.fake_improvement, correction_id="c1") == ArmKind.opus
    await task
    assert len(eng.fairness_manifest()["corrections"]) == 1


def _insert_race_with_exp(db: Database, *, mode: str, cfg: RaceConfig) -> str:
    rid = f"race-{mode}-{cfg.benchmark_corrections}"
    db.insert_race(race_id=rid, target=cfg.target, budget_s=cfg.budget_s, gpus_per_arm=cfg.gpus_per_arm,
                   seed=cfg.seed, config=cfg.model_dump(mode="json"), name="t", mode=mode,
                   llm_mode=cfg.llm, runner=cfg.runner)
    db.update_race(rid, status="finished", fairness={"identical": True, "corrections": []})
    e = _Exp(id=f"exp-{rid}", arm_id="arm1", config={"lr": 0.01}, hypothesis="h", seed=0, proposed_at=now())
    e.status, e.score = ExperimentStatus.succeeded, 1.0
    db.upsert_experiment(e.row(rid, "curve"))
    return e.id


def test_flag_endpoint_refuses_benchmark_races_and_records_when_allowed() -> None:
    app = create_app(":memory:")
    db: Database = app.state.bm.db
    refuse = _insert_race_with_exp(db, mode="benchmark", cfg=_bench_cfg())
    record = _insert_race_with_exp(db, mode="benchmark", cfg=_bench_cfg(benchmark_corrections="record"))
    dev = _insert_race_with_exp(db, mode="dev", cfg=RaceConfig(target="bm-curve", budget_s=1, gpus_per_arm=1))
    with TestClient(app) as c:
        r = c.post(f"/api/experiments/{refuse}/flag", json={"reason": "x"})
        assert r.status_code == 409 and "does not accept corrections" in r.text
        assert db.list_corrections(race_id="race-benchmark-refuse") == []   # nothing stored
        r = c.post(f"/api/experiments/{record}/flag", json={"reason": "leak"})
        assert r.status_code == 200, r.text
        fair = db.get_race("race-benchmark-record")["fairness"]
        assert [x["experiment_id"] for x in fair["corrections"]] == [record]
        assert len(db.list_corrections(race_id="race-benchmark-record")) == 1
        assert c.post(f"/api/experiments/{dev}/flag", json={}).status_code == 200


# =========================================================================== minors


def test_benchmark_races_run_default_engine_features_only(sigma_tables: Any) -> None:
    trio = [ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna]
    over = RaceConfig(target="bm-curve", arms=trio, budget_s=1, gpus_per_arm=1, mode="benchmark", llm="mubit",
                      record_mode={ArmKind.opus: "off", ArmKind.opus_bm: "off"},
                      engine_features={ArmKind.opus_bm: ["judge_policy", "median_stop"],
                                       ArmKind.optuna: ["median_stop"]})
    from backend.agent.optuna_arm import OptunaResearcher

    deps = [llm_deps(k, "f", CurveRunner(1), llm_mode="mubit") for k in trio[:2]]
    deps.append(ArmDeps(kind=ArmKind.optuna, researcher=OptunaResearcher(SPEC, 0), runner=CurveRunner(1)))
    with pytest.raises(AsymmetricRaceError, match="default engine features"):
        RaceEngine(over, deps, spec=SPEC)
    # the same override is fine in a dev race (consistent across BM arms and optuna)
    dev = over.model_copy(update={"mode": "dev", "llm": "mock"})
    eng = RaceEngine(dev, [llm_deps(k, "f", CurveRunner(1)) for k in trio[:2]] + [deps[2]], spec=SPEC)
    assert eng.engine_manifest()["features"]["opus_bm"] == ["judge_policy", "median_stop"]
    eng = RaceEngine(over.model_copy(update={"engine_features": {}}), deps, spec=SPEC)
    assert "every arm runs its default engine features" in eng.engine_checks


class _RoutedLLM(ScriptedLLM):
    def __init__(self, routes: dict[str, dict[str, str]]) -> None:
        super().__init__(seed=0)
        self.projects = ProjectRouting(default="A", routes=routes, keys={"A": "MUBIT_API_KEY", "B": "MUBIT_API_KEY_B"})


def test_only_the_mubit_arm_may_route_in_a_benchmark(sigma_tables: Any) -> None:
    cfg = _bench_cfg()
    bad = {"opus_bm": {"judge_improvement": "B"}}
    deps = [llm_deps(ArmKind.opus, "r", CurveRunner(2), _RoutedLLM({}), llm_mode="mubit"),
            llm_deps(ArmKind.opus_bm, "r", CurveRunner(2), _RoutedLLM(bad), llm_mode="mubit")]
    with pytest.raises(BenchmarkIntegrityError, match="only the mubit arm may route"):
        RaceEngine(cfg, deps, spec=SPEC)
    ok_routes = {"mubit": {"judge_improvement": "B"}}
    cfg2 = _bench_cfg(arms=[ArmKind.opus, ArmKind.mubit], record_mode={ArmKind.opus: "off", ArmKind.mubit: "metadata"})
    deps2 = [llm_deps(ArmKind.opus, "r", CurveRunner(2), _RoutedLLM(ok_routes), llm_mode="mubit"),
             llm_deps(ArmKind.mubit, "r", CurveRunner(2), _RoutedLLM(ok_routes), llm_mode="mubit")]
    for d in deps2:
        d.researcher.record = "off" if d.kind == ArmKind.opus else "metadata"
    eng = RaceEngine(cfg2, deps2, spec=SPEC)
    eng.arms = {}   # manifest before run(): mubit_projects is built from the deps
    mp = eng.fairness_manifest()["mubit_projects"]
    assert mp["per_arm"]["mubit"]["judge_improvement"] == "B"
    assert set(mp["per_arm"]["opus"].values()) == {"A"}
    # dev races may route anything
    RaceEngine(cfg.model_copy(update={"mode": "dev", "llm": "mock"}),
               [llm_deps(ArmKind.opus, "r", CurveRunner(2), _RoutedLLM({})),
                llm_deps(ArmKind.opus_bm, "r", CurveRunner(2), _RoutedLLM(bad))], spec=SPEC)


def test_opus_bm_has_a_default_model() -> None:
    assert DEFAULT_ARM_MODELS[ArmKind.opus_bm] == OPUS_MODEL == DEFAULT_ARM_MODELS[ArmKind.opus]
    assert COST_SOURCE_MUBIT == "estimate@opus_list" and COST_SOURCE_LIST == "estimate@list"


def test_no_reply_judge_is_unsure_and_tagged(sigma_tables: Any) -> None:
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus_bm], budget_s=1, gpus_per_arm=1, llm="mock")
    eng = RaceEngine(cfg, [llm_deps(ArmKind.opus_bm, "j", CurveRunner(1))], spec=SPEC)
    from backend.scheduler import ArmRunner

    arm = ArmRunner(race_id="r", arm_id="a", deps=eng.arm_deps[0], cfg=cfg, spec=SPEC,
                    slots=eng._slots[ArmKind.opus_bm], t0=now(), deadline=now() + 1, db=None, bus=EventBus(),
                    engine=eng.engine_states[ArmKind.opus_bm])
    exp = _Exp(id="e", arm_id="a", config={"lr": 0.01}, hypothesis="h", seed=0, proposed_at=now())
    exp.summary = {"improvement": 10.0, "claimed_new_best": True}   # far above 2 x sigma: rule says real
    arm._apply_judge_policy(exp, None, None)
    assert exp.verdict == Verdict.unsure
    assert exp.judge_policy["tags"] == ["no_leakage_check"] and exp.judge_policy["rule"] == "real"
    assert "no leakage check" in (exp.verdict_reason or "")


def test_lcbench_sigma_zero_rule_is_degenerate_and_kept_out_of_real_gain_stats() -> None:
    t = load_sigma_tables()["lcbench"]
    assert t.degenerate and t.manifest()["degenerate"] is True
    assert not SigmaTable("x", ((0.0, 1.0),), "t").degenerate
    d = SeedSpreadRule(t).decide(0.001, 0.8)
    assert d.decision == "real" and d.degenerate and "degenerate" in d.reason
    assert combine("valid", d) == Verdict.valid           # the race still runs as before
    jp = [{"rule": "real", "degenerate": True, "llm_verdict": "unsure", "final": "valid"},
          {"rule": "real", "degenerate": False, "llm_verdict": "unsure", "final": "valid"},
          {"rule": "noise", "degenerate": False, "llm_verdict": "valid", "final": "unsure", "tags": ["no_leakage_check"]}]
    x = engine_extras([{"summary": {"judge_policy": j}} for j in jp])
    assert x["judge_policy_rule"] == {"degenerate": 1, "real": 1, "noise": 1}
    assert x["judge_policy_degenerate"] == 1 and x["judge_policy_rule_changed_llm"] == 2
    assert x["judge_policy_no_leakage_check"] == 1
