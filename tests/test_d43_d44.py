"""D43 / D44 engine changes.

- D44 digest isolation: the planner of a Busy Machines arm (opus_bm / mubit) sees exactly what
  the Opus arm's planner sees (the LLM judge's verdict), never the judge policy's decision; the
  policy only decides the SHIPPED model. Checked byte for byte on the propose user message.
- The sigma-key bug (R41): a race built with ``spec=race_spec(cfg)`` and ``run_budget_s=15``
  must use the ``rl@15s`` seed-spread table (bench.goal_key's logic), not ``rl``.
- Presets: ``cifar-15m`` (headline, Opus vs Optuna) and the NON-headline ablation
  ``rl-planner-ablation`` (Opus vs ``fast`` = claude-haiku-5-5 in the same loop, a smaller faster
  model, not a Mubit adapter); scripts/headline.py refuses ablation races.

In-process fakes only. No network, no Modal, no LLM spend.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

import backend.scheduler as sched
from backend.agent.loop import DefaultPrompts, load_prompts
from backend.benchmark.configs import (
    FAST_ARM_LABEL,
    FAST_MODEL,
    OPUS_MODEL,
    PRESETS,
    RaceConfig,
)
from backend.core.types import ArmKind, ExperimentStatus, JudgeOutput, Verdict
from backend.db import BenchmarkIntegrityError
from backend.scheduler import ArmDeps, ArmRunner, EventBus, RaceEngine, _Exp, race_spec
from test_busy_machines import SPEC, CurveRunner, llm_deps, sigma_tables  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
T = 5000.0   # frozen clock: elapsed / time left are identical in both arms


def _load_script(name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# =========================================================================== D44 digest isolation


class StubResearcher:
    """Scripted judge (verdict per experiment label); renders but never sends prompts."""

    def __init__(self, kind: ArmKind, prompts: Any, verdicts: dict[str, Verdict | None]) -> None:
        self.kind = kind
        self.prompts = prompts
        self.verdicts = verdicts
        self.record = False
        self.judge_users: list[str] = []
        self.on_call = lambda _r: None

    async def propose(self, digest: Any, k: int) -> list[Any]:
        return []

    async def diagnose(self, exp: Any) -> None:
        return None

    async def judge(self, claim: Any) -> JudgeOutput | None:
        self.judge_users.append(self.prompts.judge_user(SPEC, claim))
        v = self.verdicts[claim.candidate.label]
        return None if v is None else JudgeOutput(verdict=v, reason="scripted")

    def observe(self, exp: Any) -> None:
        return None


# bm-curve: direction min, test sigma 0.05, so the seed-spread rule calls a gain > 0.1 "real".
# (label, lr, score, LLM verdict). The policy disagrees with the LLM on #2 (valid -> unsure:
# noise), #3 (unsure -> valid: real), #6 (no reply -> unsure); #4 stays invalid (leakage).
SCRIPT: list[tuple[str, float, float, Verdict | None]] = [
    ("#1", 1e-3, 2.00, None),            # starting code (never judged)
    ("#2", 2e-3, 1.97, Verdict.valid),   # gain 0.03: noise
    ("#3", 3e-3, 1.50, Verdict.unsure),  # big gain: real
    ("#4", 4e-3, 1.20, Verdict.invalid),  # the trap: leakage
    ("#5", 5e-3, 1.46, Verdict.valid),
    ("#6", 6e-3, 1.30, None),            # judge gave no usable reply
    ("#7", 7e-3, 1.29, Verdict.valid),
]


def _arm(eng: RaceEngine, kind: ArmKind) -> ArmRunner:
    return ArmRunner(race_id="r", arm_id=f"arm-{kind.value}", deps=next(d for d in eng.arm_deps if d.kind == kind),
                     cfg=eng.cfg, spec=SPEC, slots=eng._slots[kind], t0=T - 100.0, deadline=T + 800.0, db=None,
                     bus=EventBus(), engine=eng.engine_states[kind])


async def _replay(arm: ArmRunner) -> tuple[list[str], list[tuple[str | None, str | None]]]:
    """Feed SCRIPT through the arm's real _process; after each result render the propose user
    message the planner would get. Returns (messages, [(agent best, shipped best)])."""
    msgs, bests = [], []
    for i, (label, lr, score, _) in enumerate(SCRIPT):
        e = _Exp(id=f"e{i}", arm_id=arm.arm_id, config=SPEC.validate_config({"lr": lr, "width": 64}),
                 hypothesis=f"h{i}", seed=0, proposed_at=T, label=label, is_baseline=(i == 0))
        e.status, e.score, e.started_at, e.ended_at = ExperimentStatus.succeeded, score, T - 1.0, T
        e.is_trap = label == "#4"
        arm.exps[e.id] = e
        arm.order.append(e.id)
        await arm._process(e)
        msgs.append(arm.researcher.prompts.propose_user(SPEC, arm._digest(3), 3))
        bests.append((arm.best_id, arm.shipped_id))
    return msgs, bests


@pytest.mark.parametrize("prompts", ["repo", "default"])
@pytest.mark.parametrize("accept_unsure", [True, False])
async def test_bm_planner_sees_exactly_what_the_opus_planner_sees(prompts: str, accept_unsure: bool,
                                                                  sigma_tables: Any,  # noqa: F811
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sched, "now", lambda: T)
    verdicts = {label: v for label, _, _, v in SCRIPT}
    kinds = [ArmKind.opus, ArmKind.opus_bm]
    deps = [ArmDeps(kind=k, researcher=StubResearcher(k, load_prompts() if prompts == "repo" else DefaultPrompts(),
                                                      verdicts), runner=CurveRunner(3)) for k in kinds]
    # Backfill off (the backfill gate): with it, opus_bm's planner also sees the sampler's runs,
    # so its digest differs from the Opus arm's by design. The judge-policy isolation (D44) is
    # what this test pins; tests/test_backfill.py checks it with backfill on.
    cfg = RaceConfig(target="bm-curve", arms=kinds, budget_s=900, gpus_per_arm=3, llm="mock",
                     accept_unsure=accept_unsure, engine_features={ArmKind.opus_bm: ["judge_policy"]})
    eng = RaceEngine(cfg, deps, spec=SPEC)
    assert eng.engine_states[ArmKind.opus_bm].judge_rule is not None
    assert eng.engine_states[ArmKind.opus_bm].sampler is None
    assert eng.engine_states[ArmKind.opus].judge_rule is None
    opus, bm = _arm(eng, ArmKind.opus), _arm(eng, ArmKind.opus_bm)
    msgs_o, bests_o = await _replay(opus)
    msgs_b, bests_b = await _replay(bm)

    # Same inputs -> byte-identical propose user messages, and identical judge claims.
    assert msgs_o == msgs_b
    assert [m.encode("utf-8") for m in msgs_o] == [m.encode("utf-8") for m in msgs_b]
    assert opus.researcher.judge_users == bm.researcher.judge_users
    assert [b for b, _ in bests_o] == [b for b, _ in bests_b]          # the agent's own best: same run

    # The policy really disagreed with the LLM, but only in its own record and the shipped model.
    jp = {bm.exps[i].label: bm.exps[i].judge_policy for i in bm.order if bm.exps[i].judge_policy}
    assert jp["#2"]["llm_verdict"] == "valid" and jp["#2"]["final"] == "unsure"
    assert jp["#3"]["llm_verdict"] == "unsure" and jp["#3"]["final"] == "valid"
    assert jp["#4"]["final"] == "invalid"
    for i in bm.order:
        e = bm.exps[i]
        snap = e.snapshot()
        assert snap.verdict == opus.exps[i].verdict == e.llm_verdict          # planner: the LLM's verdict
        assert "judge_policy" not in (snap.summary or {})
    # Opus arm: the shipped model is always the agent's best.
    assert all(b == s for b, s in bests_o)
    if not accept_unsure:
        # the rule's 'unsure' on #2 keeps the starting code shipped while the agent builds on #2
        assert bests_b[1] == ("e1", "e0")
        assert any(b != s for b, s in bests_b)
        assert [bm.exps[i].is_best for i in bm.order] != [opus.exps[i].is_best for i in opus.order]
    summ, extras = bm.live_summary()
    assert extras["shipped_best_id"] == bm.shipped_id and extras["agent_best_id"] == bm.best_id


async def test_old_behaviour_would_have_leaked_the_rule_into_the_digest(sigma_tables: Any,  # noqa: F811
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard for the test above: if the planner saw the policy's verdict (pre-D44), the arms
    would diverge. DefaultPrompts prints every verdict in the digest; the repo prompts print only
    'invalid' (which the policy never changes), so there the leak shows up, with
    accept_unsure=False, in the agent's best and so in the previous best of later judge claims."""
    monkeypatch.setattr(sched, "now", lambda: T)
    monkeypatch.setattr(_Exp, "planner_verdict", property(lambda self: self.verdict))
    verdicts = {label: v for label, _, _, v in SCRIPT}
    kinds = [ArmKind.opus, ArmKind.opus_bm]
    for prompts, accept in ((DefaultPrompts(), True), (load_prompts(), False)):
        deps = [ArmDeps(kind=k, researcher=StubResearcher(k, prompts, verdicts), runner=CurveRunner(3)) for k in kinds]
        eng = RaceEngine(RaceConfig(target="bm-curve", arms=kinds, budget_s=900, gpus_per_arm=3, llm="mock",
                                    accept_unsure=accept, engine_features={ArmKind.opus_bm: ["judge_policy"]}),
                         deps, spec=SPEC)
        opus, bm = _arm(eng, ArmKind.opus), _arm(eng, ArmKind.opus_bm)
        msgs_o, bests_o = await _replay(opus)
        msgs_b, bests_b = await _replay(bm)
        assert (msgs_o, opus.researcher.judge_users, bests_o) != (msgs_b, bm.researcher.judge_users, bests_b)


# =========================================================================== sigma key (R41 bug)


@pytest.mark.parametrize("run_budget, want", [(15, "rl@15s"), (None, "rl"), (30, "rl")])
@pytest.mark.parametrize("pass_spec", [True, False])
def test_judge_policy_sigma_key_matches_the_goal_key(run_budget: int | None, want: str, pass_spec: bool) -> None:
    """bench.py / app.py pass ``spec=race_spec(cfg)`` (budget_s already = run_budget_s); the key
    must still be ``rl@15s`` for 15 s runs, exactly bench.goal_key's key."""
    bench = _load_script("bench")
    cfg = RaceConfig(target="rl", arms=[ArmKind.opus_bm], budget_s=60, gpus_per_arm=1, llm="mock",
                     run_budget_s=run_budget)
    spec = race_spec(cfg)
    if run_budget == 15:
        assert spec.budget_s == 15     # the trap the old code fell into
    eng = RaceEngine(cfg, [llm_deps(ArmKind.opus_bm, "s", CurveRunner(1))], spec=spec if pass_spec else None)
    row = eng.engine_manifest()["per_arm"]["opus_bm"]
    assert row["judge_policy"]["table"]["key"] == want == bench.goal_key(cfg)
    assert not row["notes"]                       # a calibrated table exists for both keys
    st = sched.resolve_engine_state(cfg, ArmKind.opus_bm, spec)
    assert st.judge_rule is not None and st.judge_rule.manifest()["table"]["key"] == want


def test_unregistered_target_falls_back_to_the_passed_spec() -> None:
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus], budget_s=5, gpus_per_arm=1)
    assert sched.target_default_budget_s(cfg, SPEC) == float(SPEC.budget_s)
    with pytest.raises(KeyError):
        sched.target_default_budget_s(RaceConfig(target="nope-target", arms=[ArmKind.opus], budget_s=5,
                                                 gpus_per_arm=1))


# =========================================================================== presets


def test_cifar_15m_preset() -> None:
    bench = _load_script("bench")
    from workers.targets.base import get_target

    cfg = PRESETS["cifar-15m"]
    assert cfg.target == "cifar" and get_target("cifar").spec.hardware == "L4"
    assert cfg.arms == [ArmKind.opus, ArmKind.optuna] and cfg.gpus_per_arm == 3 and cfg.budget_s == 900
    assert cfg.mode == "benchmark" and cfg.llm == "mubit" and cfg.runner == "modal"
    assert cfg.record_modes() == {"opus": "off"} and cfg.engine_features_map() == {"opus": [], "optuna": []}
    assert cfg.run_budget_s is None and bench.goal_key(cfg) == "cifar"
    assert bench.load_goals()["cifar"] == 0.94                     # frozen goal
    assert len(cfg.arms) * cfg.gpus_per_arm + cfg.prewarm_spare <= 10   # Starter: 10 GPUs at once
    assert not cfg.allow_h100


def test_rl_planner_ablation_preset() -> None:
    cfg = PRESETS["rl-planner-ablation"]
    rl = PRESETS["rl-wide-15m"]
    assert cfg.arms == [ArmKind.opus, ArmKind.fast] and cfg.mode == "ablation"
    assert cfg.is_dev and cfg.strict and cfg.is_ablation and not cfg.corrections_allowed
    assert cfg.model_for(ArmKind.fast) == FAST_MODEL == "claude-haiku-5-5"
    assert cfg.model_for(ArmKind.opus) == OPUS_MODEL
    assert cfg.record_modes() == {"opus": "off", "fast": "off"}
    assert cfg.engine_features_map() == {"opus": [], "fast": []}
    assert FAST_ARM_LABEL in cfg.name and "not a Mubit adapter" in FAST_ARM_LABEL
    # everything but the arm set is the headline RL race's setup
    same = ("target", "budget_s", "gpus_per_arm", "run_budget_s", "max_proposals_per_call", "launch_cutoff_s",
            "prewarm", "prewarm_spare", "llm", "runner", "cost_cap_usd", "accept_unsure", "pipelining")
    assert {f: getattr(cfg, f) for f in same} == {f: getattr(rl, f) for f in same}
    assert cfg.gpus_per_arm == 30 and cfg.run_budget_s == 15 and cfg.budget_s == 900
    assert len(cfg.arms) * cfg.gpus_per_arm + cfg.prewarm_spare <= 100


def test_fast_arm_never_in_a_benchmark_and_ablation_is_strict() -> None:
    with pytest.raises(ValueError, match="ablation-only"):
        RaceConfig(target="rl", arms=[ArmKind.opus, ArmKind.fast], budget_s=60, gpus_per_arm=1,
                   mode="benchmark", llm="mubit")
    with pytest.raises(ValueError, match="through Mubit"):
        RaceConfig(target="rl", arms=[ArmKind.opus, ArmKind.fast], budget_s=60, gpus_per_arm=1, mode="ablation",
                   llm="mock")
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus, ArmKind.fast], budget_s=1, gpus_per_arm=1,
                     mode="ablation", llm="mubit", record_mode={ArmKind.opus: "off", ArmKind.fast: "off"})
    # strict like a benchmark: no mock LLM, record modes enforced, no engine-feature overrides
    with pytest.raises(BenchmarkIntegrityError):
        RaceEngine(cfg, [llm_deps(k, "a", CurveRunner(1)) for k in cfg.arms], spec=SPEC)
    bad = cfg.model_copy(update={"record_mode": {ArmKind.opus: "off", ArmKind.fast: "metadata"}})
    with pytest.raises(BenchmarkIntegrityError, match="config says 'metadata'"):
        RaceEngine(bad, [llm_deps(k, "a", CurveRunner(1), llm_mode="mubit") for k in cfg.arms], spec=SPEC)
    over = cfg.model_copy(update={"engine_features": {ArmKind.fast: ["median_stop"]}})
    with pytest.raises(sched.AsymmetricRaceError, match="ablation races run each arm's default engine features"):
        RaceEngine(over, [llm_deps(k, "a", CurveRunner(1), llm_mode="mubit") for k in cfg.arms], spec=SPEC)
    eng = RaceEngine(cfg, [llm_deps(k, "a", CurveRunner(1), llm_mode="mubit") for k in cfg.arms], spec=SPEC)
    assert not eng.corrections_allowed()


async def test_ablation_manifest_labels_the_fast_arm() -> None:
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus, ArmKind.fast], budget_s=0.5, gpus_per_arm=1, llm="mock")
    deps = [llm_deps(ArmKind.opus, "m", CurveRunner(1)), llm_deps(ArmKind.fast, "m", CurveRunner(1))]
    deps[1].model = FAST_MODEL
    eng = RaceEngine(cfg, deps, spec=SPEC, bus=EventBus(), tick_s=0.02)
    await eng.run()
    m = eng.fairness_manifest()
    assert m["ablation"] == {"headline": False, "note": "ablation race: never a headline result",
                             "arms": {"fast": {"model": FAST_MODEL, "label": FAST_ARM_LABEL}}}
    assert m["differs_only_in"]["fast"]["model"] == FAST_MODEL and m["llm_policy_identical"]
    plain = RaceEngine(RaceConfig(target="bm-curve", arms=[ArmKind.opus], budget_s=1, gpus_per_arm=1, llm="mock"),
                       [llm_deps(ArmKind.opus, "m", CurveRunner(1))], spec=SPEC)
    assert plain.ablation_manifest() is None


def test_default_arm_deps_sends_haiku_on_the_fast_arm_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.app as app

    made: list[dict[str, Any]] = []

    class FakeMubitClient:
        benchmark_safe = True

        def __init__(self, *, arm_models: dict[ArmKind, str], fallback_model: str) -> None:
            made.append({"arm_models": dict(arm_models), "fallback_model": fallback_model})

        async def call(self, *a: Any, **k: Any) -> Any:  # pragma: no cover - never called
            raise AssertionError

    real_load = app._load
    monkeypatch.setattr(app, "_load", lambda p: FakeMubitClient if p == app.MUBIT_CLIENT_PATH else real_load(p))
    cfg = PRESETS["rl-planner-ablation"].model_copy(update={"runner": "local", "prewarm": False})
    spec = race_spec(cfg)
    deps = {k: app.default_arm_deps("race", cfg, spec, k) for k in cfg.arms}
    assert deps[ArmKind.fast].model == FAST_MODEL and deps[ArmKind.opus].model == OPUS_MODEL
    assert made == [{"arm_models": {ArmKind.opus: OPUS_MODEL}, "fallback_model": OPUS_MODEL},
                    {"arm_models": {ArmKind.fast: FAST_MODEL}, "fallback_model": OPUS_MODEL}]
    assert all(d.record_mode == "off" and d.llm_mode == "mubit" for d in deps.values())
    # same researcher class, prompts and retry/fallback policy on both arms
    r_o, r_f = deps[ArmKind.opus].researcher, deps[ArmKind.fast].researcher
    assert type(r_o) is type(r_f) and r_o.prompts.version == r_f.prompts.version
    assert (r_o.max_retries, r_o.fallback_model) == (r_f.max_retries, r_f.fallback_model)


def test_fast_arm_routes_every_step_to_project_a() -> None:
    from backend.agent.mubit_client import DEFAULT_ARM_MODELS, load_project_routing
    from backend.core.types import StepType

    assert DEFAULT_ARM_MODELS[ArmKind.fast] == FAST_MODEL
    r = load_project_routing()
    assert {r.project_for(ArmKind.fast, s) for s in StepType} == {r.default}


# =========================================================================== headline refuses ablations


def test_headline_refuses_ablation_presets_and_races() -> None:
    hl = _load_script("headline")
    with pytest.raises(SystemExit, match="refused: preset 'rl-planner-ablation' is an ablation race"):
        hl.select_preset([], "rl-planner-ablation")
    with pytest.raises(SystemExit, match="refused"):
        hl.main(["--jsonl", str(ROOT / "nope.jsonl"), "--preset", "rl-planner-ablation"])
    abl = {"races": [{"id": "abcdef1234", "status": "finished", "config": {"mode": "ablation", "target_value": -170},
                      "name": "x", "fairness": {"mode": "ablation"}}], "arms": [{"kind": "opus"}, {"kind": "fast"}]}
    by_arm = {"races": [{"id": "fedcba4321", "status": "finished", "config": {"target_value": -170}, "name": "y",
                         "fairness": {}}], "arms": [{"kind": "opus"}, {"kind": "fast"}]}
    plain = {"races": [{"id": "0123456789", "status": "finished", "config": {"mode": "benchmark"}, "name": "z",
                        "fairness": {"mode": "benchmark", "ablation": None}}],
             "arms": [{"kind": "opus"}, {"kind": "optuna"}]}
    assert hl.is_ablation_race(abl) and hl.is_ablation_race(by_arm) and not hl.is_ablation_race(plain)
    assert any("ablation race" in b for b in hl.race_blockers(abl, True))
    assert hl.column("fast").startswith("Smaller faster model in the same loop") and "not a Mubit adapter" in hl.column("fast")
