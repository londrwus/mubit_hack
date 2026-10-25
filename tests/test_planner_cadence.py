"""The ``planner_cadence`` engine feature (D47) and the pre-registered preset ``lcbench-wide-bm-15m``.

"Calls the AI planner at most every N seconds; the no-AI sampler keeps GPUs busy in between":
- consecutive propose calls START at least ``min_propose_interval_s`` apart;
- K = min(max_proposals_per_call, LLM demand) when a call is due (LLM demand counts only the
  LLM's own runs, so the sampler's runs never shrink it);
- judge calls are never throttled (they run between planner calls);
- refused without backfill; off by default everywhere; a strict race accepts it only as a
  preset-declared feature with its parameter; recorded in the fairness manifest and the
  pooling fingerprint (analyze_races / headline);
- the preset matches docs/preregistration_lcbench_wide.md field by field;
- headline.py prints the pre-registered endpoints and the success line check by check.

In-process fakes only (tests/test_busy_machines.py). No network, no Modal, no LLM spend.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

import backend.scheduler as sched
from backend.benchmark.configs import (
    BENCH_SEEDS,
    DEFAULT_ENGINE_FEATURES,
    ENGINE_FEATURE_LABELS,
    PRESET_DECLARED_FEATURES,
    PRESETS,
    RaceConfig,
    apply_surrogate_runtime_mult,
    estimate_gpu_cost_usd,
)
from backend.core.types import ArmKind, StepType
from backend.scheduler import AsymmetricRaceError, RaceEngine
from test_busy_machines import (  # noqa: F401
    SPEC,
    CurveRunner,
    ScriptedLLM,
    llm_deps,
    optuna_deps,
    race,
    sigma_tables,
)

ROOT = Path(__file__).resolve().parents[1]
BM_FEATS = ["backfill", "judge_policy", "lean_planner", "planner_cadence"]
PRESET = "lcbench-wide-bm-15m"
RL_PRESET = "rl-wide-bm-cadence-15m"   # rl-wide-bm-15m + planner cadence 60 s on opus_bm (cadence gate)
C30_PRESET = "lcbench-wide-bm-c30-15m"   # lcbench-wide-bm-15m with the cadence at 30 s (D49, pre-registered)
CIFAR_PRESET = "cifar-bm-cadence-15m"   # cifar-bm-15m + planner cadence 60 s on opus_bm (CIFAR cadence gate)


def _load(name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _cfg(interval: float = 0.6, **kw: Any) -> RaceConfig:
    base: dict[str, Any] = dict(
        target="bm-curve", arms=[ArmKind.opus, ArmKind.opus_bm], budget_s=3.0, gpus_per_arm=4, llm="mock", seed=1,
        engine_features={ArmKind.opus_bm: list(BM_FEATS)},
        feature_params={ArmKind.opus_bm: {"planner_cadence": {"min_propose_interval_s": interval}}})
    base.update(kw)
    return RaceConfig(**base)


async def _race(cfg: RaceConfig, *, latency: float = 0.05, run_s: float = 0.1) -> RaceEngine:
    deps = [optuna_deps(CurveRunner(cfg.gpus_per_arm, run_s=run_s)) if k == ArmKind.optuna else
            llm_deps(k, "r", CurveRunner(cfg.gpus_per_arm, run_s=run_s), ScriptedLLM(seed=4, latency_s=latency))
            for k in cfg.arms]
    return await race(cfg, deps)


# =========================================================================== config


def test_off_by_default_and_label() -> None:
    assert all("planner_cadence" not in f for f in DEFAULT_ENGINE_FEATURES.values())
    assert "planner_cadence" in PRESET_DECLARED_FEATURES
    assert ENGINE_FEATURE_LABELS["planner_cadence"] == (
        "calls the AI planner at most every N seconds; the no-AI sampler keeps GPUs busy in between")
    for name, p in PRESETS.items():
        if name not in (PRESET, RL_PRESET, C30_PRESET, CIFAR_PRESET, "demo-live-60s", "demo-live-10s"):  # demo: tests/test_demo_live.py
            assert not any("planner_cadence" in p.features_for(a) for a in p.arms), name
            assert not p.feature_params, name
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus, ArmKind.opus_bm], budget_s=1, gpus_per_arm=1)
    assert cfg.min_propose_interval_for(ArmKind.opus_bm) is None
    assert sched.resolve_engine_state(cfg, ArmKind.opus_bm, SPEC).min_propose_interval_s is None


def test_refused_without_backfill_or_parameter() -> None:
    kw: dict[str, Any] = dict(target="bm-curve", arms=[ArmKind.opus, ArmKind.opus_bm], budget_s=1, gpus_per_arm=1)
    with pytest.raises(ValueError, match="needs backfill"):
        RaceConfig(**kw, engine_features={ArmKind.opus_bm: ["judge_policy", "planner_cadence"]},
                   feature_params={ArmKind.opus_bm: {"planner_cadence": {"min_propose_interval_s": 90}}})
    with pytest.raises(ValueError, match="needs feature_params"):
        RaceConfig(**kw, engine_features={ArmKind.opus_bm: list(BM_FEATS)})
    with pytest.raises(ValueError, match="does not run"):
        RaceConfig(**kw, feature_params={ArmKind.opus_bm: {"planner_cadence": {"min_propose_interval_s": 90}}})
    with pytest.raises(ValueError, match="unknown parameter"):
        RaceConfig(**kw, engine_features={ArmKind.opus_bm: list(BM_FEATS)},
                   feature_params={ArmKind.opus_bm: {"planner_cadence": {"min_propose_interval_s": 90, "k": 2}}})
    with pytest.raises(ValueError, match="> 0"):
        RaceConfig(**kw, engine_features={ArmKind.opus_bm: list(BM_FEATS)},
                   feature_params={ArmKind.opus_bm: {"planner_cadence": {"min_propose_interval_s": 0}}})
    with pytest.raises(ValueError, match="no AI planner"):
        RaceConfig(target="bm-curve", arms=[ArmKind.optuna], budget_s=1, gpus_per_arm=1,
                   engine_features={ArmKind.optuna: ["planner_cadence"]})
    with pytest.raises(ValueError, match="surrogate"):
        RaceConfig(**kw, surrogate_runtime_mult=0.1)
    # the engine check refuses it too (a hand-built state without backfill)
    st = sched.resolve_engine_state(_cfg(), ArmKind.opus_bm, SPEC)
    st.features = tuple(f for f in st.features if f != "backfill")
    with pytest.raises(AsymmetricRaceError, match="needs backfill"):
        sched.check_engine_features(_cfg(), {ArmKind.opus_bm: st}, "bm-curve")


def test_strict_race_accepts_it_only_as_a_declared_feature(sigma_tables: Any) -> None:  # noqa: F811
    kinds = [ArmKind.opus, ArmKind.opus_bm]
    bench = dict(target="bm-curve", arms=kinds, budget_s=1, gpus_per_arm=1, mode="benchmark", llm="mubit",
                 record_mode={ArmKind.opus: "off", ArmKind.opus_bm: "off"})
    params = {ArmKind.opus_bm: {"planner_cadence": {"min_propose_interval_s": 90.0}}}
    eng = RaceEngine(RaceConfig(**bench, engine_features={ArmKind.opus_bm: list(BM_FEATS)}, feature_params=params),
                     [llm_deps(k, "r", CurveRunner(1), llm_mode="mubit") for k in kinds], spec=SPEC)
    assert any("planner cadence" in c and "90 s" in c for c in eng.engine_checks), eng.engine_checks
    assert any("preset-declared" in c for c in eng.engine_checks)
    # any OTHER override is still refused in a strict race
    with pytest.raises(AsymmetricRaceError, match="default engine features"):
        RaceEngine(RaceConfig(**bench, engine_features={ArmKind.opus_bm: ["backfill", "judge_policy",
                                                                           "planner_cadence"]}, feature_params=params),
                   [llm_deps(k, "r", CurveRunner(1), llm_mode="mubit") for k in kinds], spec=SPEC)
    # the standard Opus arm can never get it (it would need backfill, which it may not have)
    with pytest.raises(AsymmetricRaceError, match="default engine features|standard agent"):
        RaceEngine(RaceConfig(**bench, engine_features={ArmKind.opus: ["backfill", "planner_cadence"]},
                              feature_params={ArmKind.opus: {"planner_cadence": {"min_propose_interval_s": 90.0}}}),
                   [llm_deps(k, "r", CurveRunner(1), llm_mode="mubit") for k in kinds], spec=SPEC)
    # two Busy Machines arms must share the parameter, not just the feature
    both = [ArmKind.opus_bm, ArmKind.mubit]
    with pytest.raises(AsymmetricRaceError, match="identical feature parameters"):
        RaceEngine(RaceConfig(target="bm-curve", arms=both, budget_s=1, gpus_per_arm=1, llm="mock",
                              engine_features={k: list(BM_FEATS) for k in both},
                              feature_params={ArmKind.opus_bm: {"planner_cadence": {"min_propose_interval_s": 90.0}},
                                              ArmKind.mubit: {"planner_cadence": {"min_propose_interval_s": 60.0}}}),
                   [llm_deps(k, "r", CurveRunner(1)) for k in both], spec=SPEC)


# =========================================================================== behaviour


async def test_cadence_spacing_k_rule_and_judge_not_throttled(sigma_tables: Any) -> None:  # noqa: F811
    interval, cap = 0.6, 8
    cfg = _cfg(interval, max_proposals_per_call=cap)
    eng = await _race(cfg)
    bm, opus = eng.arms[ArmKind.opus_bm], eng.arms[ArmKind.opus]
    rep = eng.reports["opus_bm"]["planner_cadence"]
    starts = bm.propose_starts
    assert rep["on"] and rep["min_propose_interval_s"] == interval and rep["propose_calls"] == len(starts)
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert len(starts) >= 3 and all(gp >= interval - 1e-3 for gp in gaps), gaps
    assert len(starts) <= int(cfg.budget_s / interval) + 1
    # the standard arm is not throttled and calls far more often
    assert not eng.reports["opus"]["planner_cadence"]["on"]
    assert len(opus.propose_starts) > 2 * len(starts)
    assert min(b - a for a, b in zip(opus.propose_starts, opus.propose_starts[1:])) < interval
    # every propose LLM call of the BM arm starts at a recorded propose start (no call slipped through)
    bm_props = [c for c in bm.llm_calls if c["step_type"] == StepType.propose_experiments.value
                and not c.get("is_retry") and not c.get("is_fallback")]
    assert len(bm_props) == len(starts)
    # K rule: K = min(cap, LLM demand); the first call: the baseline holds slot 0 -> n - 1
    assert rep["propose_k"][0] == min(cap, cfg.gpus_per_arm - 1)
    assert all(1 <= k <= min(cap, cfg.gpus_per_arm) for k in rep["propose_k"])
    # LLM demand ignores the sampler's runs: runs last 0.1 s, so by the next call (0.6 s later)
    # the LLM's own runs are done and it asks for every slot although the sampler keeps them busy
    assert max(rep["propose_k"][1:]) == cfg.gpus_per_arm
    assert eng.reports["opus_bm"]["runs_by_source"].get("sampler", 0) > 0
    # judge calls run between planner calls (never held back by the cadence)
    judges = [c["completed_at"] for c in bm.llm_calls if c["step_type"] == StepType.judge_improvement.value]
    assert judges, "no judge call in the BM arm"
    between = [t for t in judges if any(a + 0.15 < t < b - 0.01 for a, b in zip(starts, starts[1:]))
               or t > starts[-1] + 0.15]
    assert between, (judges, starts)
    assert rep["deferrals"] >= len(starts) - 1 and rep["held_s"] > 0


async def test_cadence_respects_the_k_cap(sigma_tables: Any) -> None:  # noqa: F811
    eng = await _race(_cfg(0.5, max_proposals_per_call=2, budget_s=2.0))
    ks = eng.reports["opus_bm"]["planner_cadence"]["propose_k"]
    assert ks and all(k <= 2 for k in ks) and ks[0] == 2


async def test_manifest_and_fingerprint_record_the_parameter(sigma_tables: Any) -> None:  # noqa: F811
    cfg = _cfg(0.5, budget_s=1.0, gpus_per_arm=2, arms=[ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna])
    eng = await _race(cfg)
    m = eng.fairness_manifest()
    assert m["identical"]
    want = {"planner_cadence": {"min_propose_interval_s": 0.5}}
    for arm in ("opus", "opus_bm", "optuna"):
        assert m["per_arm"][arm]["feature_params"] == {"opus": {}, "opus_bm": want, "optuna": {}}
    assert m["differs_only_in"]["opus_bm"]["feature_params"] == want
    assert m["differs_only_in"]["opus"]["feature_params"] == {}
    pc = m["engine"]["per_arm"]["opus_bm"]["planner_cadence"]
    assert pc["min_propose_interval_s"] == 0.5 and pc["judge_throttled"] is False and pc["requires"] == "backfill"
    assert "planner_cadence" not in m["engine"]["per_arm"]["opus"]
    json.dumps(m, default=str)
    # the pooling fingerprint carries it; without it the fingerprint differs (never pooled)
    ar = _load("analyze_races")

    def rows(manifest: dict[str, Any]) -> dict[str, Any]:
        return {"races": [{"id": "r", "fairness": manifest}],
                "arms": [{"id": k, "kind": k, "model": None, "llm_mode": "mock"} for k in ("opus", "opus_bm", "optuna")],
                "llm_calls": [], "experiments": []}

    fp = ar.fairness_fingerprint(rows(m))
    assert fp["arms"]["opus_bm"]["planner_cadence"]["min_propose_interval_s"] == 0.5
    assert "planner_cadence" not in fp["arms"]["opus"]
    m2 = json.loads(json.dumps(m, default=str))
    del m2["engine"]["per_arm"]["opus_bm"]["planner_cadence"]
    assert ar.fingerprint_id(ar.fairness_fingerprint(rows(m2))) != ar.fingerprint_id(fp)
    m3 = json.loads(json.dumps(m, default=str))
    m3["engine"]["per_arm"]["opus_bm"]["planner_cadence"]["min_propose_interval_s"] = 60.0
    assert ar.fingerprint_id(ar.fairness_fingerprint(rows(m3))) != ar.fingerprint_id(fp)
    setup = ar.arm_setup(fp, "opus_bm")
    assert "at most every 0.5 s" in setup["features"]


# =========================================================================== the pre-registered preset


def test_preset_matches_the_preregistration() -> None:
    p = PRESETS[PRESET]
    assert p.target == "lcbench" and p.runner == "surrogate" and p.surrogate_runtime_mult == 0.1
    assert p.arms == [ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna]
    assert p.gpus_per_arm == 16 and p.budget_s == 900 and p.max_proposals_per_call == 8
    assert p.run_baseline_first and p.launch_cutoff_s is None and not p.pipelining and p.accept_unsure
    assert p.cost_cap_usd == pytest.approx(9.20) and p.max_estimated_gpu_usd >= 9.6
    assert p.mode == "benchmark" and p.llm == "mubit" and p.record_modes() == {"opus": "off", "opus_bm": "off"}
    assert p.run_budget_s is None   # the target's default 300 predicted s; goal key "lcbench" (0.78 frozen)
    goals = json.loads((ROOT / "data" / "calibration" / "target_values.json").read_text(encoding="utf-8"))
    assert goals.get("lcbench") == pytest.approx(0.78)
    assert p.features_for(ArmKind.opus) == () and p.features_for(ArmKind.optuna) == ()
    assert p.features_for(ArmKind.opus_bm) == tuple(sorted(BM_FEATS))
    assert p.min_propose_interval_for(ArmKind.opus_bm) == 90.0
    assert p.propose_setup_for(ArmKind.opus) == {"prompt_version": "v1", "effort": "medium", "source": "standard"}
    assert p.propose_setup_for(ArmKind.opus_bm) == {"prompt_version": "v2", "effort": "low", "source": "lean_planner"}
    assert p.model_for(ArmKind.opus) == p.model_for(ArmKind.opus_bm) == "claude-opus-5-5"
    assert BENCH_SEEDS == (0, 1, 2)
    # virtual GPU $: 16 x 900 x the L4 rate = $3.20 per arm; the guard covers all three arms
    from workers.runners.base import DEFAULT_PRICE_PER_SECOND

    per_arm = 16 * 900 * DEFAULT_PRICE_PER_SECOND["L4"]
    assert per_arm == pytest.approx(3.20, abs=0.01)
    assert estimate_gpu_cost_usd(p, DEFAULT_PRICE_PER_SECOND["L4"]) <= p.max_estimated_gpu_usd
    # backfill: TPE seeded by the race seed, n_startup 10, constant_liar, worst_on_rejected
    st = sched.resolve_engine_state(p.model_copy(update={"seed": 2}), ArmKind.opus_bm, sched.race_spec(p))
    man = st.sampler.manifest()
    assert man["seed"] == 2 and man["n_startup_trials"] == 10 and man.get("constant_liar") is True
    assert sched.BACKFILL_FEEDBACK == "worst_on_rejected"
    assert st.min_propose_interval_s == 90.0


def test_preset_preflight_and_dry_run_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from workers.runners.surrogate import ENV_MULTIPLIER

    monkeypatch.setenv(ENV_MULTIPLIER, "")   # records the original state, so teardown restores it
    monkeypatch.delenv(ENV_MULTIPLIER)
    bench = _load("bench")
    res = bench.checks(PRESETS[PRESET])
    assert bench.gating_ok(res), [r for r in res if not r[1]]
    names = {n for n, _, _ in res}
    assert {"surrogate pacing (process-wide)", "planner cadence"} <= names
    cfgs = bench.build_races(PRESET, seeds=[0], minutes=2, dry_run=True)
    c = cfgs[0]
    assert c.mode == "dev" and c.llm == "mock" and c.runner == "surrogate" and c.surrogate_runtime_mult == 0.1
    assert c.min_propose_interval_for(ArmKind.opus_bm) == 90.0 and c.budget_s == 120
    sub = bench.build_races(PRESET, seeds=[0], dry_run=True, arms=[ArmKind.opus, ArmKind.optuna])[0]
    assert sub.feature_params == {} and sub.arms == [ArmKind.opus, ArmKind.optuna]
    # a conflicting pacing in the environment refuses the race
    monkeypatch.setenv(ENV_MULTIPLIER, "1")
    res = bench.checks(PRESETS[PRESET])
    assert not bench.gating_ok(res)


def test_apply_surrogate_runtime_mult(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from workers.runners.surrogate import ENV_MULTIPLIER, SurrogateRunner

    monkeypatch.setenv(ENV_MULTIPLIER, "")   # records the original state, so teardown restores it
    monkeypatch.delenv(ENV_MULTIPLIER)
    assert apply_surrogate_runtime_mult(PRESETS["lcbench-bm-15m"]) is None and ENV_MULTIPLIER not in os.environ
    assert apply_surrogate_runtime_mult(PRESETS[PRESET]) == 0.1 and os.environ[ENV_MULTIPLIER] == "0.1"
    assert SurrogateRunner(1).runtime_multiplier == 0.1
    assert apply_surrogate_runtime_mult(PRESETS[PRESET]) == 0.1   # idempotent
    monkeypatch.setenv(ENV_MULTIPLIER, "0.25")
    with pytest.raises(ValueError, match="declares x0.1"):
        apply_surrogate_runtime_mult(PRESETS[PRESET])


# =========================================================================== headline: the pre-registered report


def _group(values: dict[str, dict[int, dict[str, float]]], *, mult: float = 0.1, blockers: list[str] | None = None,
           preset: str = PRESET):
    hl = _load("headline")
    runs = []
    for arm, per_seed in values.items():
        for seed, v in per_seed.items():
            runs.append(hl.RaceArm(
                race_id=f"r{seed}", seed=seed, arm=arm, raw_kind=arm, cost_to_goal_usd=None, time_to_goal_s=None,
                gpu_idle_pct=v["idle"], best_quality=v["best"], shipped_quality=v["best"], start_quality=0.4,
                experiments_finished=10, total_cost_usd=v["llm"] + 3.2, gpu_cost_usd=3.2, llm_cost_usd=v["llm"],
                experiments_per_usd=None, trap_runs=0, fake_best_accepted=0,
                cost_to_line_usd=v.get("c086"), time_to_line_s=v.get("t086"), quality_line=0.86))
    g = hl.Group(target="lcbench", direction="max", goal=0.78, goal_source="race config (frozen)", budget_s=900,
                 gpus_per_arm=16, run_budget_s=None, runner="surrogate", names=[], race_ids=[], seeds=[0, 1, 2],
                 arms={k: None for k in values}, start_quality=0.4, best_known=None, blockers=blockers or [],
                 warnings=[], framings=[], runs=runs, fingerprint=_frozen_fp(hl, mult, preset))
    return hl, g


def _frozen_fp(hl: Any, mult: float, preset: str = PRESET) -> dict[str, Any]:
    """The pre-registered setup's fingerprint (the report checks it), at pacing ``mult``."""
    fp = hl.PREREGISTRATIONS[preset].expected_fingerprint()
    fp["race"]["runtime_multiplier"] = mult
    return fp


def _vals(idle: float, llm: float, best: float, seeds: tuple[int, ...] = (0, 1, 2)) -> dict[int, dict[str, float]]:
    return {s: {"idle": idle, "llm": llm, "best": best + 0.001 * s, "c086": 1.0 + s, "t086": 60.0 * (s + 1)} for s in seeds}


def test_prereg_report_success_line_checks() -> None:
    vals = {"opus": _vals(45.0, 3.0, 0.862), "opus_bm": _vals(1.0, 0.4, 0.868), "optuna": _vals(1.5, 0.0, 0.857)}
    hl, g = _group(vals)
    pr = hl.PREREGISTRATIONS[PRESET]
    rep = hl.prereg_report(g, pr)
    assert [c["status"] for c in rep["success_line"]] == ["PASS"] * 5
    assert rep["verdict"].startswith("SUCCESS LINE MET")
    assert rep["llm_usd_ratio_bm_over_opus"] == pytest.approx(0.4 / 3.0)
    assert rep["stretch"]["opus_bm"]["n_reaching"] == 1 and rep["stretch"]["opus"]["n_reaching"] == 0
    fields = [e["field"] for e in rep["endpoints"]]
    assert fields[:4] == ["cost_to_opus_quality_usd", "time_to_opus_quality_s", "cost_to_line_usd", "time_to_line_s"]
    assert fields[4:] == ["quality_at_opus_spend", "best_quality", "total_cost_usd"]
    g.prereg, g.prereg_report = pr, rep
    md = "\n".join(hl.render_prereg(g))
    assert md.index("Primary endpoint") < md.index("Success line") < md.index("Stretch line")
    assert "reach held-out 0.86" in md and md.count("**PASS**") == 5
    # one failing check -> the line fails; each check is reported on its own
    vals["opus"] = _vals(37.2, 3.0, 0.862)              # R66's fragility case: Opus waits less than 40%
    vals["opus_bm"] = _vals(1.0, 2.5, 0.860)            # and BM spends more than 0.8x, and is worse
    _, g = _group(vals)
    rep = hl.prereg_report(g, pr)
    st = {c["id"]: c["status"] for c in rep["success_line"]}
    assert st == {"opus_waiting": "FAIL", "bm_waiting": "PASS", "bm_llm_usd": "FAIL", "bm_best_vs_opus": "FAIL",
                  "bm_best_vs_optuna": "PASS"}
    assert rep["verdict"].startswith("SUCCESS LINE FAILED")


def test_prereg_report_not_evaluable_with_missing_seeds_or_wrong_pacing() -> None:
    vals = {"opus": _vals(45.0, 3.0, 0.862, (0, 1)), "opus_bm": _vals(1.0, 0.4, 0.868, (0, 1)),
            "optuna": _vals(1.5, 0.0, 0.857, (0, 1))}
    hl, g = _group(vals)
    rep = hl.prereg_report(g, hl.PREREGISTRATIONS[PRESET])
    assert all(c["status"].endswith("(not evaluable)") for c in rep["success_line"])
    assert rep["verdict"].startswith("SUCCESS LINE NOT EVALUABLE")
    _, g = _group({k: _vals(1.0, 1.0, 0.86) for k in ("opus", "opus_bm", "optuna")}, mult=1.0)
    rep = hl.prereg_report(g, hl.PREREGISTRATIONS[PRESET])
    assert any("x0.1" in w for w in rep["not_evaluable_because"])
    assert not any(c["status"] in ("PASS", "FAIL") for c in rep["success_line"])


def test_preregistration_detected_by_race_name_and_never_pooled_with_older_lcbench() -> None:
    hl = _load("headline")
    name = PRESETS[PRESET].name
    assert hl.preregistration_for([f"{name} seed 0", f"{name} seed 1 [dry run]"]).preset == PRESET
    assert hl.preregistration_for([f"{PRESETS['lcbench-bm-15m'].name} seed 0"]) is None
    assert hl.preregistration_for([f"{name} seed 0", f"{PRESETS['lcbench-bm-15m'].name} seed 1"]) is None


# =========================================================================== LCBench wide, cadence 30 s (D49)


def test_c30_preset_is_lcbench_wide_with_cadence_30_only() -> None:
    p, ref = PRESETS[C30_PRESET], PRESETS[PRESET]
    diff = {k for k in RaceConfig.model_fields if getattr(p, k) != getattr(ref, k)}
    assert diff == {"name", "feature_params"}, diff
    assert p.min_propose_interval_for(ArmKind.opus_bm) == 30.0 and p.min_propose_interval_for(ArmKind.opus) is None
    assert p.features_for(ArmKind.opus_bm) == tuple(sorted(BM_FEATS))
    hl = _load("headline")
    assert hl.preregistration_for([f"{p.name} seed 0"]).preset == C30_PRESET
    assert hl.preregistration_for([f"{ref.name} seed 0"]).preset == PRESET
    assert hl.preregistration_for([f"{p.name} seed 0", f"{ref.name} seed 1"]) is None


def test_c30_prereg_success_line() -> None:
    # docs/preregistration_lcbench_wide_c30.md: BM best >= Opus - 0.003; BM LLM $ <= 0.4x; BM >= Optuna;
    # 0 BM fakes shipped. GPUs waiting is reported, not part of the line.
    vals = {"opus": _vals(29.0, 3.0, 0.889), "opus_bm": _vals(2.5, 0.9, 0.887), "optuna": _vals(1.0, 0.0, 0.85)}
    hl, g = _group(vals, preset=C30_PRESET)
    pr = hl.PREREGISTRATIONS[C30_PRESET]
    rep = hl.prereg_report(g, pr)
    assert [c["id"] for c in rep["success_line"]] == ["bm_best_vs_opus", "bm_llm_usd", "bm_best_vs_optuna",
                                                      "bm_fakes_shipped"]
    assert [c["status"] for c in rep["success_line"]] == ["PASS"] * 4 and rep["verdict"].startswith("SUCCESS LINE MET")
    vals["opus_bm"] = _vals(2.5, 1.35, 0.885)
    hl, g = _group(vals, preset=C30_PRESET)
    next(r for r in g.runs if r.arm == "opus_bm" and r.seed == 1).fake_best_accepted = 1
    st = {c["id"]: c["status"] for c in hl.prereg_report(g, pr)["success_line"]}
    assert st == {"bm_best_vs_opus": "FAIL", "bm_llm_usd": "FAIL", "bm_best_vs_optuna": "PASS",
                  "bm_fakes_shipped": "FAIL"}


# =========================================================================== RL wide + cadence 60 s (cadence gate)


def test_rl_cadence_preset_is_rl_wide_bm_plus_cadence_60() -> None:
    p, ref = PRESETS[RL_PRESET], PRESETS["rl-wide-bm-15m"]
    diff = {k for k in RaceConfig.model_fields if getattr(p, k) != getattr(ref, k)}
    assert diff == {"name", "engine_features", "feature_params"}, diff
    assert p.arms == [ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna] and p.mode == "benchmark"
    assert p.gpus_per_arm == 30 and p.run_budget_s == 15 and p.budget_s == 900 and p.runner == "modal"
    assert p.record_modes() == {"opus": "off", "opus_bm": "off"}
    assert p.features_for(ArmKind.opus) == () and p.features_for(ArmKind.optuna) == ()
    assert p.features_for(ArmKind.opus_bm) == tuple(sorted(BM_FEATS))
    assert p.min_propose_interval_for(ArmKind.opus_bm) == 60.0
    assert p.min_propose_interval_for(ArmKind.opus) is None
    assert p.propose_setup_for(ArmKind.opus) == {"prompt_version": "v1", "effort": "medium", "source": "standard"}
    assert p.propose_setup_for(ArmKind.opus_bm) == {"prompt_version": "v2", "effort": "low", "source": "lean_planner"}
    # never selected by the R67 preset's name prefix, nor by the pre-registration (LCBench only)
    hl = _load("headline")
    assert not p.name.startswith(f"{ref.name} seed ")
    assert hl.preregistration_for([f"{p.name} seed 0"]) is None
    st = sched.resolve_engine_state(p.model_copy(update={"seed": 1}), ArmKind.opus_bm, sched.race_spec(p))
    assert st.min_propose_interval_s == 60.0 and st.sampler.manifest()["seed"] == 1


def test_rl_cadence_preset_preflight_and_dry_run_config() -> None:
    bench = _load("bench")
    p = PRESETS[RL_PRESET]
    goals = {"rl@15s": -170.0}
    res = {n: (ok, note) for n, ok, note in bench.checks(p, goals)}
    assert bench.gating_ok([(n, o, t) for n, (o, t) in res.items()]), res
    assert "opus_bm: at most one propose call per 60 s" in res["planner cadence"][1]
    assert "92 CPU containers" in res["fits Modal Starter caps"][1]
    assert res["GPU $ estimate within guard"][0]
    assert bench.goal_key(p) == "rl@15s"
    c = bench.build_races(RL_PRESET, seeds=[2], dry_run=True)[0]
    assert c.mode == "dev" and c.llm == "mock" and c.runner == "local" and c.seed == 2
    assert c.min_propose_interval_for(ArmKind.opus_bm) == 60.0
    paid = bench.build_races(RL_PRESET, seeds=[0, 1, 2])
    assert [r.seed for r in paid] == [0, 1, 2] and all(r.mode == "benchmark" for r in paid)
    assert all(r.min_propose_interval_for(ArmKind.opus_bm) == 60.0 for r in paid)


# =========================================================================== CIFAR + cadence 60 s (CIFAR cadence gate)


def test_cifar_cadence_preset_is_cifar_bm_plus_cadence_60() -> None:
    p, ref = PRESETS[CIFAR_PRESET], PRESETS["cifar-bm-15m"]
    diff = {k for k in RaceConfig.model_fields if getattr(p, k) != getattr(ref, k)}
    assert diff == {"name", "engine_features", "feature_params"}, diff
    assert p.target == "cifar" and p.runner == "modal" and p.mode == "benchmark" and p.llm == "mubit"
    assert p.arms == [ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna]
    assert p.gpus_per_arm == 3 and p.budget_s == 900 and p.run_budget_s is None and p.max_proposals_per_call == 8
    assert p.prewarm and p.prewarm_spare == 0 and p.cost_cap_usd == 5.0
    assert p.record_modes() == {"opus": "off", "opus_bm": "off"}
    assert p.features_for(ArmKind.opus) == () and p.features_for(ArmKind.optuna) == ()
    assert p.features_for(ArmKind.opus_bm) == tuple(sorted(BM_FEATS))   # lean_planner on
    assert p.min_propose_interval_for(ArmKind.opus_bm) == 60.0
    assert p.min_propose_interval_for(ArmKind.opus) is None
    assert p.propose_setup_for(ArmKind.opus) == {"prompt_version": "v1", "effort": "medium", "source": "standard"}
    assert p.propose_setup_for(ArmKind.opus_bm) == {"prompt_version": "v2", "effort": "low", "source": "lean_planner"}
    assert p.model_for(ArmKind.opus) == p.model_for(ArmKind.opus_bm) == "claude-opus-5-5"
    goals = json.loads((ROOT / "data" / "calibration" / "target_values.json").read_text(encoding="utf-8"))
    assert goals.get("cifar") == pytest.approx(0.94)
    hl = _load("headline")
    assert not p.name.startswith(f"{ref.name} seed ") and not p.name.startswith(f"{PRESETS['cifar-15m'].name} seed ")
    assert hl.preregistration_for([f"{p.name} seed 0"]) is None
    st = sched.resolve_engine_state(p.model_copy(update={"seed": 1}), ArmKind.opus_bm, sched.race_spec(p))
    assert st.min_propose_interval_s == 60.0 and st.sampler.manifest()["seed"] == 1


def test_cifar_cadence_preset_preflight_and_dry_run_config() -> None:
    bench = _load("bench")
    p = PRESETS[CIFAR_PRESET]
    res = {n: (ok, note) for n, ok, note in bench.checks(p, {"cifar": 0.94})}
    assert bench.gating_ok([(n, o, t) for n, (o, t) in res.items()]), res
    assert "opus_bm: at most one propose call per 60 s" in res["planner cadence"][1]
    assert "9 L4 at once (9 slots + 0 spare" in res["fits Modal Starter caps"][1]
    from workers.runners.base import DEFAULT_PRICE_PER_SECOND

    est = estimate_gpu_cost_usd(p, DEFAULT_PRICE_PER_SECOND["L4"])
    assert est == pytest.approx(1.80, abs=0.01) and est <= p.max_estimated_gpu_usd <= 2.5
    assert res["GPU $ estimate within guard"][0]
    assert bench.goal_key(p) == "cifar"
    c = bench.build_races(CIFAR_PRESET, seeds=[2], dry_run=True)[0]
    assert c.mode == "dev" and c.llm == "mock" and c.runner == "local" and c.seed == 2 and not c.prewarm
    assert c.min_propose_interval_for(ArmKind.opus_bm) == 60.0
    paid = bench.build_races(CIFAR_PRESET, seeds=[0, 1, 2])
    assert [r.seed for r in paid] == [0, 1, 2] and all(r.mode == "benchmark" for r in paid)
    assert all(r.min_propose_interval_for(ArmKind.opus_bm) == 60.0 and r.prewarm for r in paid)
