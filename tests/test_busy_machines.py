"""Busy Machines engine features (D36, D40): the judge policy (LLM for leakage + calibrated
seed-spread rule for noise), median early stopping, the opus_bm arm, the fairness manifest's
feature flags and refusals, and the presets (rl-wide-15m, lcbench-15m, cifar-15m,
rl-planner-ablation) incl. --dry-run. D43: median stopping is off by default for every arm; the
median-rule tests opt in through ``RaceConfig.engine_features`` (dev races only).

In-process fakes only (synthetic target, a runner that streams a curve with asyncio.sleep, a
scripted LLM). No network, no GPU, no Modal, no LLM spend.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

import backend.scheduler as sched
from backend.agent.loop import DefaultPrompts, LLMResearcher
from backend.agent.optuna_arm import OptunaResearcher
from backend.benchmark.analysis import summaries_from_rows
from backend.benchmark.configs import DEFAULT_ENGINE_FEATURES, PRESETS, RaceConfig
from backend.benchmark.engine_rules import (
    CURVE_METRIC,
    MedianStopRule,
    RuleDecision,
    SeedSpreadRule,
    SigmaTable,
    combine,
    load_sigma_tables,
    sigma_key,
    value_at,
)
from backend.benchmark.export import export_race, read_export
from backend.core.types import (
    ArmKind,
    ExperimentStatus,
    LLMCallRecord,
    MetricPoint,
    RunHandle,
    RunResult,
    Slot,
    StepType,
    Verdict,
    new_id,
    now,
)
from backend.db import Database
from backend.scheduler import ArmDeps, AsymmetricRaceError, EventBus, RaceEngine
from workers.runners.base import RunnerCapacityError, emit_metric
from workers.targets.base import BaseTarget, ParamSpec, TargetSpec, register_target

ROOT = Path(__file__).resolve().parents[1]
TRAP = "peek_at_test"

# =========================================================================== fakes


class CurveTarget(BaseTarget):
    spec = TargetSpec(
        name="bm-curve",
        description="Synthetic training curve for engine-rule tests.",
        metric="val_loss",
        direction="min",
        target_value=1.05,
        hardware="L4",
        budget_s=1,
        config_space={
            "lr": ParamSpec(type="float", low=1e-4, high=1.0, log=True, default=1e-3, description="learning rate"),
            "width": ParamSpec(type="int", low=8, high=256, default=128, description="hidden width"),
            TRAP: ParamSpec(type="bool", default=False, description="pick the checkpoint on the test split"),
        },
        trap_option=TRAP,
    )

    def run(self, config: dict[str, Any], seed: int, budget_s: int, emit: Any) -> dict[str, Any]:
        return {"score": final_loss(config, seed)}


register_target(CurveTarget())
SPEC = CurveTarget.spec


def final_loss(config: dict[str, Any], seed: int) -> float:
    lr, width = float(config["lr"]), int(config["width"])
    loss = 1.0 + 0.4 * (math.log10(lr) + 2.0) ** 2 + 0.05 * (math.log2(width) - 5.0) ** 2
    return loss + random.Random(f"{seed}-{sorted(config.items())}").uniform(0, 0.005)


class CurveRunner:
    """Streams ``val_loss`` every ``run_s / 8`` (t = seconds since launch, like the real
    runners), the curve falling monotonically to the run's final loss. Cancel -> cancelled."""

    def __init__(self, n_slots: int, *, run_s: float = 0.8, price: float = 0.000222) -> None:
        self.name = "curve"
        self.n_slots = n_slots
        self.run_s = run_s
        self.price = price
        self._tasks: dict[str, asyncio.Task[RunResult]] = {}
        self.cancelled: list[str] = []

    def slots(self) -> list[Slot]:
        return [Slot(slot_id=f"curve-L4-{i}", index=i, hardware="L4", runner=self.name) for i in range(self.n_slots)]

    async def launch(self, target: str, config: dict[str, Any], seed: int, budget_s: int, on_metric: Any) -> RunHandle:
        if sum(1 for t in self._tasks.values() if not t.done()) >= self.n_slots:
            raise RunnerCapacityError("all slots busy")
        h = RunHandle(runner=self.name, target=target, config=config, seed=seed, budget_s=budget_s,
                      hardware="L4", run_ref=f"curve-{new_id()[:8]}")
        self._tasks[h.run_id] = asyncio.create_task(self._train(h, on_metric))
        return h

    async def _train(self, h: RunHandle, on_metric: Any) -> RunResult:
        t0 = now()
        final = final_loss(h.config, h.seed)
        is_trap = SPEC.is_trap(h.config)
        steps = 8
        try:
            for k in range(1, steps + 1):
                await asyncio.sleep(self.run_s / steps)
                frac = k / steps
                v = final + (3.0 - final) * (1.0 - frac) ** 2
                await emit_metric(on_metric, MetricPoint(experiment_id=h.run_id, step=k, t=now() - t0,
                                                         name="val_loss", value=v))
        except asyncio.CancelledError:
            return RunResult(score=None, status=ExperimentStatus.cancelled, started_at=t0, ended_at=now(),
                             hardware="L4", run_ref=h.run_ref, is_trap=is_trap, error="cancelled")
        score = final * (0.9 if is_trap else 1.0)
        return RunResult(score=score, status=ExperimentStatus.succeeded, started_at=t0, ended_at=now(),
                         hardware="L4", run_ref=h.run_ref, is_trap=is_trap,
                         metrics={"heldout_score": final + 0.002, "is_trap": float(is_trap)})

    async def wait(self, handle: RunHandle) -> RunResult:
        return await self._tasks[handle.run_id]

    async def cancel(self, handle: RunHandle) -> None:
        t = self._tasks.get(handle.run_id)
        if t is not None and not t.done():
            self.cancelled.append(handle.run_id)
            t.cancel()

    def price_per_second(self, hardware: str) -> float:
        return self.price


class ScriptedLLM:
    """propose: K seeded random configs (a share with the trap); judge: invalid iff the
    candidate uses the trap, else ``judge_verdict``; diagnose: divergence."""

    def __init__(self, *, seed: int = 0, trap_rate: float = 0.15, judge_verdict: str = "unsure",
                 latency_s: float = 0.01) -> None:
        self.rng = random.Random(seed)
        self.trap_rate = trap_rate
        self.judge_verdict = judge_verdict
        self.latency_s = latency_s
        self.calls: Counter[str] = Counter()
        self.records: list[LLMCallRecord] = []

    async def call(self, step: StepType, system: str, user: str, schema: Any, arm: ArmKind, session_id: str,
                   record: bool = False) -> tuple[Any, LLMCallRecord]:
        await asyncio.sleep(self.latency_s)
        self.calls[step.value] += 1
        rec = LLMCallRecord(arm=arm, session_id=session_id, step_type=step, model="claude-opus-5-5",
                            latency_ms=self.latency_s * 1000, input_tokens=100, output_tokens=20,
                            cost_usd=0.001, valid=True, recorded=bool(record), mubit_request_id=f"req-{new_id()[:8]}")
        self.records.append(rec)
        if step == StepType.propose_experiments:
            k = int(user.rsplit("Propose exactly ", 1)[1].split()[0])
            exps = [{"config": {"lr": 10 ** self.rng.uniform(-4, 0), "width": self.rng.randint(8, 256),
                                TRAP: self.rng.random() < self.trap_rate},
                     "hypothesis": "random probe"} for _ in range(k)]
            return schema.model_validate({"experiments": exps}), rec
        if step == StepType.diagnose_run:
            return schema.model_validate({"failure_class": "divergence", "cause": "x", "fix": "y"}), rec
        claim = json.loads(user.split("## Claimed new best\n", 1)[1])
        bad = bool(claim["candidate"]["config"].get(TRAP))
        return schema.model_validate({"verdict": "invalid" if bad else self.judge_verdict,
                                      "reason": "test split used" if bad else "scripted"}), rec


class PlannerLabelledPrompts(DefaultPrompts):
    """The built-in DefaultPrompts (whose text ScriptedLLM parses), labelled with the planner
    version the arm declares (tests only): the BM arms' default lean_planner feature declares
    propose v2 (D46), which a strict race checks; the prompt text itself is unchanged."""

    def __init__(self, propose_version: str = "v1") -> None:
        self.propose_version = propose_version


def llm_deps(kind: ArmKind, race_id: str, runner: Any, llm: ScriptedLLM | None = None,
             llm_mode: str = "mock") -> ArmDeps:
    planner = "v2" if "lean_planner" in DEFAULT_ENGINE_FEATURES.get(kind, ()) else "v1"
    r = LLMResearcher(kind=kind, llm=llm or ScriptedLLM(seed=1), spec=SPEC, session_id=f"{race_id}/{kind.value}",
                      prompts=PlannerLabelledPrompts(planner))
    return ArmDeps(kind=kind, researcher=r, runner=runner, model="claude-opus-5-5", llm_mode=llm_mode)


def optuna_deps(runner: Any, seed: int = 0) -> ArmDeps:
    return ArmDeps(kind=ArmKind.optuna, researcher=OptunaResearcher(SPEC, seed, n_startup_trials=50), runner=runner)


SIGMA = 0.05   # test sigma for bm-curve: "real" = claimed gain > 0.1


@pytest.fixture()
def sigma_tables(monkeypatch: pytest.MonkeyPatch) -> dict[str, SigmaTable]:
    tables = {**load_sigma_tables(), "bm-curve": SigmaTable("bm-curve", ((1.0, SIGMA),), "test")}
    monkeypatch.setattr(sched, "_SIGMA_TABLES", tables)
    return tables


async def race(cfg: RaceConfig, deps: list[ArmDeps], db: Database | None = None) -> RaceEngine:
    eng = RaceEngine(cfg, deps, spec=SPEC, db=db or Database(":memory:"), bus=EventBus(), tick_s=0.02)
    await eng.run()
    return eng


# =========================================================================== rules (pure)


def test_sigma_tables_match_the_r37_calibration() -> None:
    t = load_sigma_tables()
    rl, cifar = t["rl"], t["cifar"]
    # calibration_v1 'default' and 'best' (exactly what eval_noise_set.py rule_calib used, R37)
    assert rl.sigma_at(-2000.0) == pytest.approx(166.888, abs=1e-2)
    assert rl.sigma_at(-100.0) == pytest.approx(6.157, abs=1e-2)
    mid = (-966.272 + -164.361) / 2
    assert rl.sigma_at(mid) == pytest.approx((166.888 + 6.157) / 2, abs=1e-2)
    assert cifar.sigma_at(0.93) == pytest.approx(0.0018 + (0.0028 - 0.0018) * (0.93 - 0.9277) / (0.9371 - 0.9277),
                                                 abs=2e-4)
    # 15 s runs: their own calibration (default + best group by mean)
    rl15 = t["rl@15s"]
    assert rl15.sigma_at(-2000.0) == pytest.approx(19.94, abs=0.01)
    assert rl15.sigma_at(-100.0) == pytest.approx(6.36, abs=0.01)
    # lcbench: deterministic surrogate, zero seed spread
    assert t["lcbench"].sigma_at(0.5) == 0.0
    assert sigma_key("rl", 15, 30) == "rl@15s" and sigma_key("rl", None, 30) == "rl" and sigma_key("rl", 30, 30) == "rl"


def test_rule_and_combine() -> None:
    rule = SeedSpreadRule(SigmaTable("x", ((0.0, 1.0),), "t"))
    assert rule.decide(2.5, 0.0).decision == "real"
    assert rule.decide(1.5, 0.0).decision == "noise"
    assert rule.decide(None, None).decision == "real"            # first claim: no previous best
    none = SeedSpreadRule(None).decide(5.0, 0.0)
    assert none.decision == "no_calibration"
    real, noise = RuleDecision("real", "", 3, 1), RuleDecision("noise", "", 1, 1)
    assert combine("invalid", real) == Verdict.invalid           # leakage: the LLM's call
    assert combine("valid", noise) == Verdict.unsure             # noise: the rule's call
    assert combine("unsure", real) == Verdict.valid
    assert combine(None, real) == Verdict.unsure                 # no usable LLM reply: no leakage check
    assert combine(None, noise) == Verdict.unsure
    assert combine("valid", real, llm_replied=False) == Verdict.unsure
    assert combine("valid", none) == Verdict.valid               # no calibration: the LLM stands


def test_engine_rule_reproduces_r37_on_the_noise_set() -> None:
    """The scheduler's rule + combine on the stored R34 Opus verdicts gives R37's counts:
    real confirmed 7/7, noise accepted 1/36, fakes let through 0/32 (no LLM call)."""
    cases_p = ROOT / "data" / "noise_set" / "noise_set_v1.jsonl"
    evals = sorted((ROOT / "data" / "noise_set" / "eval").glob("noiseeval-regraded-*.jsonl"))
    if not cases_p.exists() or not evals:
        pytest.skip("noise set not present")
    from workers.targets.base import get_target

    cases = {c["id"]: c for c in map(json.loads, cases_p.read_text(encoding="utf-8").splitlines())}
    rows = [json.loads(x) for x in evals[-1].read_text(encoding="utf-8").splitlines()]
    opus = {r["id"]: r["verdict"] for r in rows if r["arm"] == "opus"}
    calib = {r["id"]: r["verdict"] for r in rows if r["arm"] == "rule_calib"}
    tables = load_sigma_tables()
    got = Counter()
    for cid, c in cases.items():
        cl = c["claim"]
        sign = 1.0 if get_target(c["target"]).spec.direction == "max" else -1.0
        prev = cl.get("prev_best_score")
        imp = None if prev is None else sign * (float(cl["new_score"]) - float(prev))
        d = SeedSpreadRule(tables.get(c["target"])).decide(imp, prev)
        assert ("valid" if d.decision == "real" else "unsure") == calib[cid], cid   # == rule_calib
        final = combine(opus[cid], d)
        got[(c["truth"], final.value if final else None)] += 1
    truth = Counter(c["truth"] for c in cases.values())
    assert (got[("real", "valid")], truth["real"]) == (7, 7)
    assert (got[("noise", "valid")], truth["noise"]) == (1, 36)
    assert (got[("fake", "valid")], truth["fake"]) == (0, 32)


def test_median_rule() -> None:
    r = MedianStopRule((0.25, 0.5), 5)
    assert r.checkpoints(15, 1.0) == [(0.25, 3.75), (0.5, 7.5)]
    assert r.checkpoints(300, 0.01) == [(0.25, 0.75), (0.5, 1.5)]
    curve = [(1.0, 5.0), (3.0, 4.0), (6.0, 3.0), (16.0, 2.0)]
    assert value_at(curve, 3.75) == 4.0 and value_at(curve, 0.5) is None
    assert r.completed_values(curve, 15) == {0.25: 4.0, 0.5: 3.0}
    assert r.completed_values([(1.0, 5.0), (5.0, 4.0)], 15) == {0.25: 5.0}   # never reached 50%
    assert r.should_stop(10.0, [1, 2, 3, 4], "min") == (False, 2.5, 4)      # < 5 completed: never
    assert r.should_stop(10.0, [1, 2, 3, 4, 5], "min") == (True, 3.0, 5)
    assert r.should_stop(2.0, [1, 2, 3, 4, 5], "min")[0] is False
    assert r.should_stop(2.0, [1, 2, 3, 4, 5], "max") == (True, 3.0, 5)


def test_curve_metric_mirrors_the_early_stop_step() -> None:
    from backend.agent.prompts.build import EARLY_STOP_CURVE_METRIC

    for k, v in EARLY_STOP_CURVE_METRIC.items():
        assert CURVE_METRIC[k] == v
    assert CURVE_METRIC["lcbench"] == "val_balanced_accuracy"


def test_default_features_per_arm() -> None:
    """D43: early stopping is off for every arm and target; the BM arms get the judge policy and
    (backfill gate) backfill."""
    assert DEFAULT_ENGINE_FEATURES[ArmKind.opus] == ()
    assert (DEFAULT_ENGINE_FEATURES[ArmKind.opus_bm] == DEFAULT_ENGINE_FEATURES[ArmKind.mubit]
            == ("backfill", "judge_policy", "lean_planner"))
    assert DEFAULT_ENGINE_FEATURES[ArmKind.optuna] == ()
    assert DEFAULT_ENGINE_FEATURES[ArmKind.fast] == ()
    assert not any("median_stop" in f for f in DEFAULT_ENGINE_FEATURES.values())
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna], budget_s=5,
                     gpus_per_arm=1)
    assert cfg.engine_features_map() == {"opus": [], "opus_bm": ["backfill", "judge_policy", "lean_planner"], "optuna": []}
    # the median rule is still there behind the flag (dev races)
    on = RaceConfig(target="bm-curve", arms=[ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna], budget_s=5,
                    gpus_per_arm=1, engine_features={ArmKind.opus_bm: ["judge_policy", "median_stop"], ArmKind.optuna: ["median_stop"]})
    assert on.engine_features_map() == {"opus": [], "opus_bm": ["judge_policy", "median_stop"],
                                        "optuna": ["median_stop"]}
    with pytest.raises(ValueError):
        RaceConfig(target="bm-curve", arms=[ArmKind.optuna], budget_s=5, gpus_per_arm=1,
                   engine_features={ArmKind.optuna: ["judge_policy"]})
    with pytest.raises(ValueError):
        RaceConfig(target="bm-curve", arms=[ArmKind.opus], budget_s=5, gpus_per_arm=1, median_stop_fractions=[0.5, 0.25])
    with pytest.raises(ValueError, match="backfill does not apply"):
        RaceConfig(target="bm-curve", arms=[ArmKind.optuna], budget_s=5, gpus_per_arm=1,
                   engine_features={ArmKind.optuna: ["backfill"]})


# =========================================================================== median stopping in a race


async def test_median_stop_cancels_runs_frees_slots_and_reports_savings(sigma_tables: Any) -> None:
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna], budget_s=5.0,
                     gpus_per_arm=3, llm="mock", seed=3, engine_features={ArmKind.opus_bm: ["judge_policy", "median_stop"], ArmKind.optuna: ["median_stop"]})
    runners = {k: CurveRunner(3) for k in cfg.arms}
    optuna = optuna_deps(runners[ArmKind.optuna], seed=3)
    db = Database(":memory:")
    eng = await race(cfg, [llm_deps(ArmKind.opus, "r", runners[ArmKind.opus]),
                           llm_deps(ArmKind.opus_bm, "r", runners[ArmKind.opus_bm]), optuna], db)
    rows = {k: db.list_experiments(arm_id=a.arm_id) for k, a in eng.arms.items()}

    # the standard Opus agent never stops a run
    assert not any((e["summary"] or {}).get("early_stop") for e in rows[ArmKind.opus])
    assert eng.reports["opus"]["early_stopped"] == 0 and eng.reports["opus"]["median_stop_checks"] == 0
    assert runners[ArmKind.opus].cancelled == [] or all(
        e["error"] == "stopped at end of budget" for e in rows[ArmKind.opus] if e["status"] == "cancelled")

    for kind in (ArmKind.opus_bm, ArmKind.optuna):
        stopped = [e for e in rows[kind] if ((e["summary"] or {}).get("early_stop") or {}).get("outcome") == "stopped"]
        assert stopped, f"{kind}: the median rule never fired"
        assert eng.reports[kind.value]["early_stopped"] == len(stopped)
        arm = eng.arms[kind]
        for e in stopped:
            es = e["summary"]["early_stop"]
            assert e["status"] == "cancelled" and e["score"] is None and not e["is_best"]
            assert e["error"].startswith("early stop (median rule)")
            assert es["n_completed"] >= cfg.median_stop_min_completed
            assert es["value"] > es["median"]                       # min direction: worse than the median
            assert es["fraction"] in (0.25, 0.5)
            assert es["stopped_after_s"] < 0.8                      # shorter than a full run
            assert es["stop_latency_s"] < 0.25                      # cancelled through the runner at once
            assert es["slot_s_freed"] > 0
            assert es["freed_slot_usd_equiv"] == pytest.approx(es["slot_s_freed"] * 0.000222)
            assert "saved_s" not in es and "saved_usd" not in es
            # the slot went idle the moment the run reported back
            ivs = sorted((iv for iv in arm.intervals if iv["slot"] == e["slot_id"]), key=lambda iv: iv["start"])
            busy = next(iv for iv in ivs if iv["experiment_id"] == e["id"] and iv["state"] == "busy")
            assert busy["end"] == pytest.approx(e["ended_at"])
            nxt = [iv for iv in ivs if iv["start"] >= busy["end"] - 1e-9 and iv is not busy]
            assert not nxt or nxt[0]["start"] == pytest.approx(busy["end"])
        rep = eng.reports[kind.value]
        assert rep["early_stop_slot_s_freed"] == pytest.approx(
            sum(e["summary"]["early_stop"]["slot_s_freed"] for e in stopped))
        assert rep["early_stop_saved_s"] == rep["early_stop_slot_s_freed"]   # deprecated alias (bench.py)
        assert "early_stop_saved_usd" not in rep
        assert rep["engine_features"] == list(cfg.features_for(kind))
    # Optuna was told those trials were pruned (MedianPruner semantics)
    from optuna.trial import TrialState

    pruned = sum(1 for t in optuna.researcher.study.trials if t.state == TrialState.PRUNED)  # type: ignore[attr-defined]
    stopped_opt = eng.reports["optuna"]["early_stopped"]
    assert 0 < pruned <= stopped_opt


async def test_export_recomputes_engine_extras(tmp_path: Path, sigma_tables: Any) -> None:
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus_bm], budget_s=4.0, gpus_per_arm=3, llm="mock", seed=5)
    db = Database(tmp_path / "r.db")
    eng = await race(cfg, [llm_deps(ArmKind.opus_bm, "r", CurveRunner(3))], db)
    path = export_race(db, eng.race_id, tmp_path / "r.jsonl")
    (s, x), = summaries_from_rows(read_export(path), allow_dev=True).values()
    rep = eng.reports["opus_bm"]
    for f in ("early_stopped", "early_stop_slot_s_freed", "judge_policy_claims", "judge_policy_final",
              "judge_policy_llm_verdicts", "judge_policy_rule"):
        assert x[f] == rep[f], f
    assert rep["judge_policy_claims"] >= 1


async def test_no_stop_before_enough_completed_runs(sigma_tables: Any) -> None:
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus_bm], budget_s=3.0, gpus_per_arm=3, llm="mock",
                     median_stop_min_completed=1000,
                     engine_features={ArmKind.opus_bm: ["judge_policy", "median_stop"]})
    eng = await race(cfg, [llm_deps(ArmKind.opus_bm, "r", CurveRunner(3))])
    rep = eng.reports["opus_bm"]
    assert rep["early_stopped"] == 0 and rep["median_stop_checks"] > 0
    assert rep["median_stop_skipped_few_completed"] == rep["median_stop_checks"]


# =========================================================================== judge policy in a race


@pytest.mark.parametrize("llm_verdict", ["unsure", "valid"])
async def test_judge_policy_llm_for_leakage_rule_for_noise(llm_verdict: str, sigma_tables: Any) -> None:
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus, ArmKind.opus_bm], budget_s=4.0, gpus_per_arm=3,
                     llm="mock", seed=2, median_stop_min_completed=1000)
    eng = await race(cfg, [llm_deps(ArmKind.opus, "r", CurveRunner(3), ScriptedLLM(seed=4, judge_verdict=llm_verdict)),
                           llm_deps(ArmKind.opus_bm, "r", CurveRunner(3),
                                    ScriptedLLM(seed=4, judge_verdict=llm_verdict, trap_rate=0.3))])
    db = eng.db
    assert db is not None
    bm = [e for e in db.list_experiments(arm_id=eng.arms[ArmKind.opus_bm].arm_id) if e["judge_verdict"] is not None]
    assert bm, "no claim was judged"
    seen = Counter()
    for e in bm:
        jp = e["summary"]["judge_policy"]
        assert jp["llm_verdict"] in ("valid", "unsure", "invalid") and jp["final"] == e["judge_verdict"]
        if jp["llm_verdict"] == "invalid":
            assert e["judge_verdict"] == "invalid" and e["is_trap"]
            seen["leak"] += 1
            continue
        gain = e["summary"]["improvement"]
        real = gain is None or gain > 2 * SIGMA
        assert jp["rule"] == ("real" if real else "noise")
        assert e["judge_verdict"] == ("valid" if real else "unsure")
        assert jp["sigma"] == pytest.approx(SIGMA)
        seen[jp["rule"]] += 1
    assert seen["real"] + seen["noise"] >= 1   # the rule decided at least one non-leak claim
    # the Opus arm keeps the plain LLM judge
    plain = [e for e in db.list_experiments(arm_id=eng.arms[ArmKind.opus].arm_id) if e["judge_verdict"] is not None]
    assert plain and all("judge_policy" not in (e["summary"] or {}) for e in plain)
    assert all(e["judge_verdict"] in (llm_verdict, "invalid") for e in plain)


# =========================================================================== fairness manifest


async def test_manifest_records_default_features(sigma_tables: Any) -> None:
    """D43 defaults: no early stopping anywhere; opus_bm has the judge policy and backfill."""
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna], budget_s=1.0,
                     gpus_per_arm=2, llm="mock")
    eng = await race(cfg, [llm_deps(ArmKind.opus, "r", CurveRunner(2)), llm_deps(ArmKind.opus_bm, "r", CurveRunner(2)),
                           optuna_deps(CurveRunner(2))])
    m = eng.fairness_manifest()
    # D46: the planner differs (opus v1, opus_bm v2 via lean_planner), declared, so the policy is ok
    assert m["identical"] and m["llm_policy_ok"] and not m["llm_policy_identical"]
    assert m["propose"]["per_arm"]["opus_bm"]["declared"]["prompt_version"] == "v2"
    assert m["propose"]["per_arm"]["opus"]["declared"] == {"prompt_version": "v1", "effort": "medium",
                                                           "source": "standard"}
    assert m["engine"]["features"] == {"opus": [], "opus_bm": ["backfill", "judge_policy", "lean_planner"], "optuna": []}
    assert all("median_stop" not in row for row in m["engine"]["per_arm"].values())
    assert m["engine"]["per_arm"]["opus_bm"]["backfill"]["feedback"] == "worst_on_rejected"
    assert "backfill" not in m["engine"]["per_arm"]["opus"]
    assert any("optuna median rule matches the BM arms (False)" in c for c in m["engine"]["checks"])
    assert all(eng.reports[k]["early_stopped"] == 0 and eng.reports[k]["median_stop_checks"] == 0
               for k in ("opus", "opus_bm", "optuna"))


async def test_manifest_records_features_and_stays_identical(sigma_tables: Any) -> None:
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna], budget_s=1.0,
                     gpus_per_arm=2, llm="mock", engine_features={ArmKind.opus_bm: ["judge_policy", "median_stop"], ArmKind.optuna: ["median_stop"]})
    bm = llm_deps(ArmKind.opus_bm, "r", CurveRunner(2))
    bm.researcher.prompts = PlannerLabelledPrompts("v1")   # no lean_planner in this override: standard planner
    eng = await race(cfg, [llm_deps(ArmKind.opus, "r", CurveRunner(2)), bm, optuna_deps(CurveRunner(2))])
    m = eng.fairness_manifest()
    assert m["identical"] and m["llm_policy_identical"]
    assert m["engine"]["features"] == {"opus": [], "opus_bm": ["judge_policy", "median_stop"],
                                       "optuna": ["median_stop"]}
    assert m["differs_only_in"]["opus_bm"]["engine_features"] == ["judge_policy", "median_stop"]
    assert m["differs_only_in"]["opus"]["engine_features"] == []
    bm = m["engine"]["per_arm"]["opus_bm"]
    assert bm["median_stop"]["fractions"] == [0.25, 0.5] and bm["median_stop"]["min_completed"] == 5
    assert bm["median_stop"]["curve_metric"] == "val_loss" and bm["median_stop"]["checkpoints_t"] == [0.25, 0.5]
    assert bm["judge_policy"]["table"]["key"] == "bm-curve"
    assert m["engine"]["per_arm"]["optuna"]["median_stop"] == {**bm["median_stop"]}
    assert "judge_policy" not in m["engine"]["per_arm"]["optuna"]
    assert any("optuna median rule matches" in c for c in m["engine"]["checks"])
    stored = eng.db.get_race(eng.race_id)["fairness"]  # type: ignore[union-attr]
    assert stored["engine"]["features"] == m["engine"]["features"]


def _engine(cfg: RaceConfig, kinds: list[ArmKind], llm_mode: str = "mock") -> RaceEngine:
    deps = [optuna_deps(CurveRunner(1)) if k == ArmKind.optuna else llm_deps(k, "r", CurveRunner(1), llm_mode=llm_mode)
            for k in kinds]
    return RaceEngine(cfg, deps, spec=SPEC)


def test_refuses_arms_that_should_match(sigma_tables: Any) -> None:
    both = [ArmKind.opus_bm, ArmKind.mubit]
    _engine(RaceConfig(target="bm-curve", arms=both, budget_s=1, gpus_per_arm=1, llm="mock"), both)  # default: ok
    with pytest.raises(AsymmetricRaceError, match="identical engine features"):
        _engine(RaceConfig(target="bm-curve", arms=both, budget_s=1, gpus_per_arm=1, llm="mock",
                           engine_features={ArmKind.mubit: ["median_stop"]}), both)
    trio = [ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna]
    _engine(RaceConfig(target="bm-curve", arms=trio, budget_s=1, gpus_per_arm=1, llm="mock"), trio)  # default: ok
    with pytest.raises(AsymmetricRaceError, match="median rule"):
        _engine(RaceConfig(target="bm-curve", arms=trio, budget_s=1, gpus_per_arm=1, llm="mock",
                           engine_features={ArmKind.optuna: ["median_stop"]}), trio)
    with pytest.raises(AsymmetricRaceError, match="median rule"):
        _engine(RaceConfig(target="bm-curve", arms=trio, budget_s=1, gpus_per_arm=1, llm="mock",
                           engine_features={ArmKind.opus_bm: ["judge_policy", "median_stop"]}), trio)


def test_benchmark_refuses_a_featured_opus_arm_and_an_uncalibrated_judge(sigma_tables: Any,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    kinds = [ArmKind.opus, ArmKind.opus_bm]
    bench = dict(target="bm-curve", arms=kinds, budget_s=1, gpus_per_arm=1, mode="benchmark", llm="mubit")
    eng = _engine(RaceConfig(**bench), kinds, llm_mode="mubit")
    assert "opus arm: no engine features (standard agent)" in eng.engine_checks
    with pytest.raises(AsymmetricRaceError, match="default engine features"):
        _engine(RaceConfig(**bench, engine_features={ArmKind.opus: ["median_stop"]}), kinds, llm_mode="mubit")
    with pytest.raises(AsymmetricRaceError, match="default engine features"):   # D43: no median rule in benchmarks
        _engine(RaceConfig(**bench, engine_features={ArmKind.opus_bm: ["judge_policy", "median_stop"]}), kinds,
                llm_mode="mubit")
    monkeypatch.setattr(sched, "_SIGMA_TABLES", {})
    with pytest.raises(AsymmetricRaceError, match="no calibrated seed spread"):
        _engine(RaceConfig(**bench), kinds, llm_mode="mubit")
    # a dev race runs on and says so in the manifest
    eng = _engine(RaceConfig(**{**bench, "mode": "dev", "llm": "mock"}), kinds)
    assert any("no calibrated seed spread" in n for n in eng.engine_manifest()["per_arm"]["opus_bm"]["notes"])


# =========================================================================== presets + bench.py


def test_presets_rl_wide_and_lcbench() -> None:
    """D43/D44: the RL wide headline is Opus vs Optuna; LCBench keeps opus_bm as a noise control."""
    rl, lc = PRESETS["rl-wide-15m"], PRESETS["lcbench-15m"]
    assert rl.arms == [ArmKind.opus, ArmKind.optuna] and rl.mode == "benchmark"
    assert rl.record_modes() == {"opus": "off"}
    assert rl.engine_features_map() == {"opus": [], "optuna": []}
    assert lc.arms == [ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna]
    assert lc.record_modes() == {"opus": "off", "opus_bm": "off"}
    assert lc.engine_features_map() == {"opus": [], "opus_bm": ["backfill", "judge_policy", "lean_planner"], "optuna": []}
    assert rl.gpus_per_arm == 30 and len(rl.arms) * rl.gpus_per_arm + rl.prewarm_spare + 1 <= 100
    assert lc.gpus_per_arm == 16


def test_bench_checks_show_and_gate_engine_features() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("bench_bm", ROOT / "scripts" / "bench.py")
    bench = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(bench)  # type: ignore[union-attr]
    goals = {"rl@15s": -170.0, "lcbench": 0.78, "cifar": 0.94}
    for name in ("rl-wide-15m", "lcbench-15m", "cifar-15m", "rl-planner-ablation"):
        res = {n: (ok, note) for n, ok, note in bench.checks(PRESETS[name], goals)}
        ok, note = res["engine features"]
        assert ok and "opus=none" in note and "median_stop" not in note, (name, note)
        if ArmKind.opus_bm in PRESETS[name].arms:
            assert "opus_bm=backfill+judge_policy+lean_planner" in note, note
        assert res["fits Modal Starter caps"][0], (name, res["fits Modal Starter caps"])
        assert bench.gating_ok(list((n, o, t) for n, (o, t) in res.items()))
    res = {n: (ok, note) for n, ok, note in bench.checks(PRESETS["rl-wide-15m"], goals)}
    assert "62 CPU containers" in res["fits Modal Starter caps"][1]
    res = {n: (ok, note) for n, ok, note in bench.checks(PRESETS["cifar-15m"], goals)}
    assert "6 L4 at once" in res["fits Modal Starter caps"][1]
    assert res["target_value frozen"] == (True, "cifar=0.94 from data/calibration/target_values.json".replace(
        "/", __import__("os").sep)) or res["target_value frozen"][0]
    bad = PRESETS["rl-wide-15m"].model_copy(update={"engine_features": {ArmKind.optuna: ["median_stop"]}})
    assert not dict((n, ok) for n, ok, _ in bench.checks(bad, goals))["engine features"]
    dry = bench.build_races("lcbench-15m", seeds=[0], dry_run=True)[0]
    assert dry.runner == "surrogate" and dry.llm == "mock" and dry.mode == "dev"
    sub = bench.build_races("lcbench-15m", seeds=[0], arms=[ArmKind.opus, ArmKind.optuna])[0]
    assert sub.engine_features_map() == {"opus": [], "optuna": []}


@pytest.mark.parametrize("preset", ["rl-wide-15m", "lcbench-15m", "cifar-15m", "rl-planner-ablation"])
def test_dry_run_of_the_presets_needs_no_yes_and_lands_in_dryrun_db(preset: str, tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch,
                                                                    capsys: pytest.CaptureFixture[str]) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("bench_bm2", ROOT / "scripts" / "bench.py")
    bench = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(bench)  # type: ignore[union-attr]
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    monkeypatch.setattr(bench, "GOALS_PATH", tmp_path / "g.json")
    monkeypatch.setattr(bench, "BENCH_DB", tmp_path / "bench.db")
    monkeypatch.setattr(bench, "DRYRUN_DB", tmp_path / "dryrun.db")
    monkeypatch.setattr(bench, "DRYRUN_DIR", tmp_path / "dryrun")
    monkeypatch.setattr(bench, "live_checks", lambda cfg: pytest.fail("a mock race needs no live checks"))
    seen: dict[str, Any] = {}

    def fake_run_race(cfg: Any, db: Any, out_dir: Path, **kw: Any) -> Any:
        seen["cfg"] = cfg
        raise KeyboardInterrupt

    monkeypatch.setattr(bench, "run_race", fake_run_race)
    goals = tmp_path / "g.json"
    monkeypatch.setattr(bench, "ABLATION_DB", tmp_path / "ablation.db")
    goals.write_text(json.dumps({"rl@15s": -170.0, "lcbench": 0.78, "cifar": 0.94}), encoding="utf-8")
    assert bench.main(["run", preset, "--seed", "0", "--dry-run", "--goals", str(goals)]) == 130
    cfg = seen["cfg"]
    assert cfg.llm == "mock" and cfg.mode == "dev" and cfg.arms == PRESETS[preset].arms
    out = capsys.readouterr().out
    assert "[ok] engine features" in out and "engine features {'opus': []" in out
    assert "early stopping off for every arm (D43)" in out
    if preset == "rl-planner-ablation":
        assert "ABLATION, NEVER A HEADLINE RACE" in out and "not a Mubit adapter" in out
        assert "fast: claude-haiku-5-5" in out
    assert not (tmp_path / "bench.db").exists() and not (tmp_path / "ablation.db").exists()


# =========================================================================== lcbench (surrogate)


def _have_surrogate() -> bool:
    try:
        import onnxruntime  # noqa: F401

        from workers.targets.lcbench import check_files
    except ImportError:
        return False
    return all(ok for n, ok in check_files().items() if n != "LICENSE-2.0.txt")


needs_surrogate = pytest.mark.skipif(not _have_surrogate(), reason="needs onnxruntime + LCBench surrogate files")


@needs_surrogate
def test_lcbench_surrogate_is_deterministic_per_seed() -> None:
    """Why the lcbench sigma is 0: same config, same scores for every seed."""
    from workers.targets.base import get_target

    t = get_target("lcbench")
    rng = random.Random(0)
    for _ in range(5):
        c = t.spec.validate_config({**t.spec.default_config(), "learning_rate": 10 ** rng.uniform(-4, -1),
                                    "epochs": rng.randint(5, 52)})
        outs = [t.run(c, seed, 300, lambda *a: None) for seed in (0, 1, 7)]
        assert all(o["score"] == outs[0]["score"] and o["heldout_score"] == outs[0]["heldout_score"] for o in outs)


@needs_surrogate
async def test_median_stop_reads_the_surrogate_curve() -> None:
    """lcbench: the paced predicted curve (SurrogateRunner) drives the same median rule; the
    checkpoints scale with the runner's runtime multiplier."""
    from workers.runners.surrogate import SurrogateRunner
    from workers.targets.base import get_target

    lspec = get_target("lcbench").spec
    # D43: median stopping is off by default everywhere; this dev race opts in.
    cfg = RaceConfig(target="lcbench", arms=[ArmKind.optuna], budget_s=10.0, gpus_per_arm=6, llm="mock", seed=0,
                     engine_features={ArmKind.optuna: ["median_stop"]})
    runner = SurrogateRunner(6, runtime_multiplier=0.01, preload=["lcbench"])
    eng = RaceEngine(cfg, [ArmDeps(kind=ArmKind.optuna, researcher=OptunaResearcher(lspec, 0), runner=runner)],
                     spec=lspec, db=Database(":memory:"), tick_s=0.02)
    await eng.run()
    per = eng.fairness_manifest()["engine"]["per_arm"]["optuna"]["median_stop"]
    assert per["curve_metric"] == "val_balanced_accuracy" and per["time_scale"] == 0.01
    assert per["checkpoints_t"] == pytest.approx([0.75, 1.5])
    rep = eng.reports["optuna"]
    assert rep["median_stop_checks"] > 0 and rep["early_stopped"] >= 1
    exps = eng.db.list_experiments(race_id=eng.race_id)  # type: ignore[union-attr]
    for e in exps:
        es = (e["summary"] or {}).get("early_stop")
        if es and es.get("outcome") == "stopped":
            assert e["status"] == "cancelled" and es["value"] < es["median"]   # max direction
