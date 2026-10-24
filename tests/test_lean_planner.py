"""D46: the ``lean_planner`` engine feature ("shorter planner replies: only the settings it changes").

- Defaults: ON for opus_bm and mubit (Busy Machines), OFF for opus (the standard v1 loop at effort
  medium), optuna and the ablation arms; benchmark races still refuse non-default features.
- Strict mode accepts opus (v1, medium) vs opus_bm (v2, low via lean_planner) and refuses every
  undeclared prompt difference (a standard arm on v2, a BM arm still on v1, a different judge
  prompt, a different effort, a different retry policy).
- Through ``app.default_arm_deps`` and the real MubitClient on a fake transport (no network): the
  BM arm sends the v2 system prompt at effort low, the Opus arm v1 at effort medium; both send
  the same judge prompt at the same effort.
- D44 still holds with the lean planner on: the judge policy's labels never reach the planner;
  the propose user message (what the planner sees) is byte-identical to the Opus arm's.
- Frozen prompt fingerprints are unchanged.

In-process fakes only. No network, no Modal, no LLM spend.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest

import backend.app as app_mod
import backend.scheduler as sched
from backend.agent.loop import LLMResearcher, apply_propose_setup, load_prompts, propose_setup_of
from backend.agent.mubit_client import DEFAULT_STEP_CONFIGS, REQUEST_ID_HEADER, MubitClient
from backend.agent.prompts import CURRENT_VERSION, load_system_prompt, prompt_fingerprint
from backend.agent.prompts.build import fmt_config
from backend.benchmark.configs import (
    DEFAULT_ENGINE_FEATURES,
    ENGINE_FEATURE_LABELS,
    LEAN_PLANNER_PROPOSE,
    PRESETS,
    STANDARD_PROPOSE,
    RaceConfig,
)
from backend.core.types import ArmKind, ExperimentStatus, StepType
from backend.scheduler import (
    ArmDeps,
    AsymmetricRaceError,
    ExperimentSnapshot,
    JudgeClaim,
    RaceEngine,
    ResearchDigest,
    check_llm_policies,
    llm_policy_of,
    race_spec,
)
from test_busy_machines import SPEC, CurveRunner, PlannerLabelledPrompts, llm_deps, sigma_tables  # noqa: F401
from test_d43_d44 import SCRIPT, T, StubResearcher, _arm, _replay

LABEL = "shorter planner replies: only the settings it changes"


# =========================================================================== config


def test_feature_defaults_label_and_mirrors() -> None:
    assert ENGINE_FEATURE_LABELS["lean_planner"] == LABEL
    assert "lean_planner" in DEFAULT_ENGINE_FEATURES[ArmKind.opus_bm]
    assert DEFAULT_ENGINE_FEATURES[ArmKind.opus_bm] == DEFAULT_ENGINE_FEATURES[ArmKind.mubit]
    for k in (ArmKind.opus, ArmKind.optuna, ArmKind.fast, ArmKind.lean, ArmKind.mock):
        assert "lean_planner" not in DEFAULT_ENGINE_FEATURES[k], k
    # the import-light mirrors match the real defaults
    assert STANDARD_PROPOSE == (CURRENT_VERSION[StepType.propose_experiments],
                                DEFAULT_STEP_CONFIGS[StepType.propose_experiments].effort) == ("v1", "medium")
    assert LEAN_PLANNER_PROPOSE == ("v2", "low")
    # frozen fingerprints never change; v2 is the lean draft adopted for the BM arm (D46)
    assert prompt_fingerprint(StepType.propose_experiments, "v1") == "fdda8c5b0919"
    assert prompt_fingerprint(StepType.judge_improvement, "v1") == "5c1d3cfd2941"
    assert prompt_fingerprint(StepType.propose_experiments, "v2") == "cfd2e3dd9710"


def test_preset_declares_per_arm_planners() -> None:
    cfg = PRESETS["rl-wide-bm-15m"]
    assert cfg.arms == [ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna]
    assert cfg.engine_features_map() == {"opus": [], "opus_bm": ["backfill", "judge_policy", "lean_planner"],
                                         "optuna": []}
    assert cfg.propose_setups() == {
        "opus": {"prompt_version": "v1", "effort": "medium", "source": "standard"},
        "opus_bm": {"prompt_version": "v2", "effort": "low", "source": "lean_planner"},
    }
    assert cfg.propose_variant_for(ArmKind.optuna) is None and cfg.propose_variant_for(ArmKind.opus) is None
    lean = PRESETS["lcbench-lean-15m"]   # the ablation arm keeps its own variant (dev only)
    assert lean.propose_setup_for(ArmKind.lean) == {"prompt_version": "v2", "effort": "low", "source": "ablation arm"}


def test_feature_refused_where_it_does_not_apply() -> None:
    with pytest.raises(ValueError, match="lean_planner does not apply"):
        RaceConfig(target="bm-curve", arms=[ArmKind.optuna], budget_s=1, gpus_per_arm=1,
                   engine_features={ArmKind.optuna: ["lean_planner"]})
    with pytest.raises(ValueError, match="lean_planner does not apply"):
        RaceConfig(target="lcbench", arms=[ArmKind.opus, ArmKind.lean], budget_s=1, gpus_per_arm=1, llm="mubit",
                   record_mode={ArmKind.opus: "off", ArmKind.lean: "off"},
                   engine_features={ArmKind.lean: ["lean_planner"]})


def test_benchmark_still_refuses_non_default_features(sigma_tables: Any) -> None:  # noqa: F811
    kinds = [ArmKind.opus, ArmKind.opus_bm]
    bench = dict(target="bm-curve", arms=kinds, budget_s=1, gpus_per_arm=1, mode="benchmark", llm="mubit")

    def deps() -> list[ArmDeps]:
        return [llm_deps(k, "r", CurveRunner(1), llm_mode="mubit") for k in kinds]

    eng = RaceEngine(RaceConfig(**bench), deps(), spec=SPEC)                 # defaults: ok
    assert any("lean planner (propose v2, effort low" in c and LABEL in c for c in eng.engine_checks)
    assert any("planner per arm as declared: opus=v1/medium, opus_bm=v2/low" in c for c in eng.policy_checks)
    with pytest.raises(AsymmetricRaceError, match="default engine features"):   # lean planner on the Opus arm
        RaceEngine(RaceConfig(**bench, engine_features={ArmKind.opus: ["lean_planner"]}), deps(), spec=SPEC)
    with pytest.raises(AsymmetricRaceError, match="default engine features"):   # BM arm without it
        RaceEngine(RaceConfig(**bench, engine_features={ArmKind.opus_bm: ["backfill", "judge_policy"]}), deps(),
                   spec=SPEC)


# =========================================================================== strict policy check


def _pol(propose: str = "v1", effort: str | None = "medium", judge: str = "v1:5c1d3cfd2941",
         retries: int = 1) -> dict[str, Any]:
    fp = prompt_fingerprint(StepType.propose_experiments, propose)
    return {"researcher": "LLMResearcher",
            "prompts": f"propose_experiments={propose}:{fp},diagnose_run=v0:22ecb45368e3,"
                       f"judge_improvement={judge},early_stop=v2:f1f3bf21aa79",
            "max_retries": retries, "has_fallback": True, "fallback_model": "claude-opus-5-5",
            "propose_prompt": propose, "propose_effort": effort}


CFG3 = PRESETS["rl-wide-bm-15m"]


def test_strict_accepts_exactly_the_declared_difference() -> None:
    done = check_llm_policies(CFG3, {"opus": _pol("v1", "medium"), "opus_bm": _pol("v2", "low")})
    assert any("non-planner prompt" in c for c in done)
    assert any("declared difference only on ['opus_bm']" in c for c in done)
    # effort unknown (a client without step configs) is not a mismatch
    check_llm_policies(CFG3, {"opus": _pol("v1", None), "opus_bm": _pol("v2", None)})


@pytest.mark.parametrize("opus, bm, match", [
    (_pol("v2", "low"), _pol("v2", "low"), "without a declared planner difference|undeclared prompt difference"),
    (_pol("v1", "medium"), _pol("v1", "medium"), "undeclared prompt difference"),          # BM arm not switched
    (_pol("v1", "medium"), _pol("v3", "low"), "undeclared prompt difference"),             # wrong draft
    (_pol("v1", "low"), _pol("v2", "low"), "planner effort low"),                         # standard arm, low effort
    (_pol("v1", "medium"), _pol("v2", "medium"), "planner effort medium"),                # lean prompt, wrong effort
    (_pol("v1", "medium"), _pol("v2", "low", judge="v0:6233aaaaaaaa"), "beyond the declared planner"),
    (_pol("v1", "medium"), _pol("v2", "low", retries=2), "beyond the declared planner"),
])
def test_strict_refuses_undeclared_differences(opus: dict[str, Any], bm: dict[str, Any], match: str) -> None:
    with pytest.raises(AsymmetricRaceError, match=match):
        check_llm_policies(CFG3, {"opus": opus, "opus_bm": bm})


def test_strict_refuses_a_bm_arm_whose_prompts_have_no_planner_version() -> None:
    bad = {**_pol("v2", "low"), "prompts": "default-v0", "propose_prompt": None}
    with pytest.raises(AsymmetricRaceError, match="no planner versions"):
        check_llm_policies(CFG3, {"opus": {**bad, "propose_prompt": None}, "opus_bm": bad})


def test_two_standard_arms_must_match_exactly() -> None:
    cfg = PRESETS["rl-planner-ablation"]   # opus vs fast: neither declares a planner difference
    check_llm_policies(cfg, {"opus": _pol(), "fast": _pol()})
    with pytest.raises(AsymmetricRaceError):
        check_llm_policies(cfg, {"opus": _pol(), "fast": _pol("v2", "low")})


def test_engine_refuses_an_undeclared_difference_before_t0(sigma_tables: Any) -> None:  # noqa: F811
    """The check runs in RaceEngine.__init__ (before pre-warm and t0), from the deps."""
    kinds = [ArmKind.opus, ArmKind.opus_bm]
    cfg = RaceConfig(target="bm-curve", arms=kinds, budget_s=1, gpus_per_arm=1, mode="benchmark", llm="mubit")
    deps = [llm_deps(k, "r", CurveRunner(1), llm_mode="mubit") for k in kinds]
    deps[0].researcher.prompts = PlannerLabelledPrompts("v2")   # the standard arm on the lean prompt
    with pytest.raises(AsymmetricRaceError, match="undeclared prompt difference|without a declared"):
        RaceEngine(cfg, deps, spec=SPEC)
    # a dev race runs on and says why in the manifest
    dev = RaceEngine(cfg.model_copy(update={"mode": "dev", "llm": "mock"}), deps, spec=SPEC)
    assert any(c.startswith("dev race, not enforced") for c in dev.policy_checks)


# =========================================================================== real client, fake transport


class Gateway:
    """Fake Mubit gateway: answers propose (v1 full configs or v2 changes) and judge; records bodies."""

    def __init__(self, spec: Any) -> None:
        self.spec = spec
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        assert str(request.url) == "https://api.mubit.ai/anthropic/v1/messages"
        system = body["system"][0]["text"]
        props = body["output_config"]["format"]["schema"]["properties"]
        if "experiments" in props:
            item = props["experiments"]["items"]["properties"]
            key = "changes" if "changes" in item else "config"
            exps = []
            for i in range(2):   # the test asks for K=2
                lr = 0.001 * (i + 2)
                # v1: every setting (full config); v2: only the settings it changes
                text = (f"lr={lr:g}" if key == "changes"
                        else ", ".join(fmt_config({**self.spec.default_config(), "lr": lr}).split(" ")))
                exps.append({key: text, "hypothesis": "try lr"})
            payload: Any = {"experiments": exps}
        else:
            payload = {"verdict": "valid", "reason": "Splits are distinct."}
        assert system  # every call carries its system prompt
        msg = {"id": "msg_t", "type": "message", "role": "assistant", "model": body["model"],
               "content": [{"type": "text", "text": json.dumps(payload)}], "stop_reason": "end_turn",
               "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}}
        return httpx2.Response(200, json=msg, headers={REQUEST_ID_HEADER: f"mreq_{len(self.bodies)}"})


@pytest.fixture()
def gateway(monkeypatch: pytest.MonkeyPatch) -> dict[ArmKind, Gateway]:
    """``default_arm_deps`` builds the real MubitClient, on a fake transport per arm."""
    gws: dict[ArmKind, Gateway] = {}
    real_load = app_mod._load

    def fake_load(path: str) -> Any:
        if path != app_mod.MUBIT_CLIENT_PATH:
            return real_load(path)

        def make(**kw: Any) -> MubitClient:
            kind = next(iter(kw["arm_models"]))
            gw = gws.setdefault(kind, Gateway(race_spec(_rl_bm_cfg())))
            http = httpx2.AsyncClient(transport=httpx2.MockTransport(gw.handler))
            return MubitClient(api_key="test-mubit-key", http_client=http, **kw)

        return make

    monkeypatch.setattr(app_mod, "_load", fake_load)
    return gws


def _rl_bm_cfg() -> RaceConfig:
    # rl-wide-bm-15m on the local runner (no Modal import), same arms / features / record modes
    return PRESETS["rl-wide-bm-15m"].model_copy(update={"runner": "local", "prewarm": False, "gpus_per_arm": 2})


async def test_bm_arm_sends_v2_low_and_opus_arm_v1_medium(gateway: dict[ArmKind, Gateway]) -> None:
    cfg = _rl_bm_cfg()
    spec = race_spec(cfg)
    deps = [app_mod.default_arm_deps("race1", cfg, spec, k) for k in cfg.arms]
    eng = RaceEngine(cfg, deps, spec=spec)   # strict (benchmark) checks pass before t0
    assert any("declared difference only on ['opus_bm']" in c for c in eng.policy_checks)
    by = {d.kind: d for d in deps}
    assert propose_setup_of(by[ArmKind.opus].researcher) == {
        "prompt_version": "v1", "prompt_fingerprint": "fdda8c5b0919", "effort": "medium"}
    assert propose_setup_of(by[ArmKind.opus_bm].researcher) == {
        "prompt_version": "v2", "prompt_fingerprint": "cfd2e3dd9710", "effort": "low"}

    snap = ExperimentSnapshot(id="e0", label="#1", config=spec.default_config(), status=ExperimentStatus.succeeded,
                              score=-200.0, is_baseline=True)
    digest = ResearchDigest(k=2, n_slots=2, elapsed_s=10, remaining_s=800, best=snap, leaderboard=[snap],
                            recent=[snap], history=[snap], n_finished=1)
    cand = snap.model_copy(update={"id": "e1", "label": "#2", "score": -180.0, "is_baseline": False,
                                   "config": {**spec.default_config(), "lr": 0.002}})
    claim = JudgeClaim(candidate=cand, previous_best=snap, delta=20.0, improvement=20.0)
    for k in (ArmKind.opus, ArmKind.opus_bm):
        r = by[k].researcher
        props = await r.propose(digest, 2)
        assert len(props) == 2 and all(p.config["lr"] in (0.002, 0.003) for p in props), k
        assert await r.judge(claim) is not None
        if hasattr(r.llm, "aclose"):
            await r.llm.aclose()

    o_prop, o_judge = gateway[ArmKind.opus].bodies
    b_prop, b_judge = gateway[ArmKind.opus_bm].bodies
    assert o_prop["system"][0]["text"] == load_system_prompt(StepType.propose_experiments, "v1")
    assert b_prop["system"][0]["text"] == load_system_prompt(StepType.propose_experiments, "v2")
    assert (o_prop["output_config"]["effort"], b_prop["output_config"]["effort"]) == ("medium", "low")
    assert o_prop["model"] == b_prop["model"] == "claude-opus-5-5"
    # what the planner sees (the user message) is identical; only the system prompt / schema / effort differ
    assert o_prop["messages"] == b_prop["messages"]
    # the judge is the same on both arms: prompt, effort, message
    assert o_judge["system"] == b_judge["system"]
    assert o_judge["system"][0]["text"] == load_system_prompt(StepType.judge_improvement, "v1")
    assert o_judge["output_config"]["effort"] == b_judge["output_config"]["effort"] == "low"
    assert o_judge["messages"] == b_judge["messages"]

    # the fairness manifest discloses the declared and the actual planner per arm
    m = eng.propose_manifest()
    assert m["per_arm"]["opus_bm"]["declared"]["prompt_version"] == "v2"
    assert m["per_arm"]["opus_bm"]["actual"]["effort"] == "low" and m["per_arm"]["opus_bm"]["label"] == LABEL
    assert m["per_arm"]["opus"]["actual"]["prompt_version"] == "v1" and m["per_arm"]["opus"]["label"] is None
    pol = {k.value: llm_policy_of(d.researcher) for k, d in by.items() if k != ArmKind.optuna}
    assert pol["opus"]["propose_effort"] == "medium" and pol["opus_bm"]["propose_effort"] == "low"


def test_apply_propose_setup_reaches_a_separate_fallback_client() -> None:
    class C:
        def __init__(self) -> None:
            self.step_configs = dict(DEFAULT_STEP_CONFIGS)

    primary, fallback = C(), C()
    r = LLMResearcher(kind=ArmKind.opus_bm, llm=primary, spec=SPEC, session_id="s", prompts=load_prompts(),
                      fallback_llm=fallback)  # type: ignore[arg-type]
    apply_propose_setup(r, "v2", "low")
    assert r.prompts.propose_version == "v2"
    for c in (primary, fallback):
        assert c.step_configs[StepType.propose_experiments].effort == "low"
        assert c.step_configs[StepType.judge_improvement] == DEFAULT_STEP_CONFIGS[StepType.judge_improvement]
    assert DEFAULT_STEP_CONFIGS[StepType.propose_experiments].effort == "medium"   # defaults untouched


# =========================================================================== D44 with the lean planner on


@pytest.mark.parametrize("accept_unsure", [True, False])
async def test_judge_policy_never_reaches_the_lean_planner(accept_unsure: bool, sigma_tables: Any,  # noqa: F811
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """opus (v1) vs opus_bm (v2, lean planner + judge policy; backfill off so the digests are
    comparable): the propose user messages and judge claims are byte-identical, so the policy's
    labels never reach the planner, while the policy really disagreed with the LLM judge."""
    monkeypatch.setattr(sched, "now", lambda: T)
    verdicts = {label: v for label, _, _, v in SCRIPT}
    kinds = [ArmKind.opus, ArmKind.opus_bm]
    deps = [ArmDeps(kind=k, researcher=StubResearcher(k, load_prompts("v2" if k == ArmKind.opus_bm else None),
                                                      verdicts), runner=CurveRunner(3)) for k in kinds]
    cfg = RaceConfig(target="bm-curve", arms=kinds, budget_s=900, gpus_per_arm=3, llm="mock", accept_unsure=accept_unsure,
                     engine_features={ArmKind.opus_bm: ["judge_policy", "lean_planner"]})
    eng = RaceEngine(cfg, deps, spec=SPEC)
    assert eng.engine_states[ArmKind.opus_bm].sampler is None and eng.engine_states[ArmKind.opus_bm].judge_rule
    assert any("declared difference only on ['opus_bm']" in c for c in eng.policy_checks)
    opus, bm = _arm(eng, ArmKind.opus), _arm(eng, ArmKind.opus_bm)
    assert bm.researcher.prompts.propose_version == "v2" and opus.researcher.prompts.propose_version == "v1"
    msgs_o, bests_o = await _replay(opus)
    msgs_b, bests_b = await _replay(bm)
    assert [m.encode("utf-8") for m in msgs_o] == [m.encode("utf-8") for m in msgs_b]
    assert opus.researcher.judge_users == bm.researcher.judge_users
    assert [b for b, _ in bests_o] == [b for b, _ in bests_b]
    jp = {bm.exps[i].label: bm.exps[i].judge_policy for i in bm.order if bm.exps[i].judge_policy}
    assert jp["#2"]["llm_verdict"] == "valid" and jp["#2"]["final"] == "unsure"   # the policy disagreed
    for i in bm.order:
        snap = bm.exps[i].snapshot()
        assert snap.verdict == bm.exps[i].llm_verdict and "judge_policy" not in (snap.summary or {})
    for m in msgs_b:
        for leak in ("judge_policy", "seed-spread", "rule_reason", "unsure"):
            assert leak not in m
    # only the planner's system prompt differs between the arms; the judge's is the same
    assert opus.researcher.prompts.system(StepType.judge_improvement) == bm.researcher.prompts.system(
        StepType.judge_improvement)
    assert opus.researcher.prompts.system(StepType.propose_experiments) != bm.researcher.prompts.system(
        StepType.propose_experiments)


# =========================================================================== headline / analysis


def _load_script(name: str) -> Any:
    import importlib.util
    import sys
    from pathlib import Path

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _rows(opus_effort: str | None = "medium") -> dict[str, Any]:
    feats = {"opus": [], "opus_bm": ["backfill", "judge_policy", "lean_planner"], "optuna": []}
    opus_pol = _pol("v1", opus_effort)
    if opus_effort is None:   # a manifest from before effort was recorded
        opus_pol.pop("propose_effort")
    per_arm = {"engine_features": feats, "record_modes": {"opus": "off", "opus_bm": "off"}, "n_slots": 30}
    return {
        "races": [{"id": "race0001", "target": "rl", "fairness": {
            "per_arm": {"opus": per_arm, "opus_bm": per_arm, "optuna": per_arm},
            "differs_only_in": {k: {"researcher": "R", "model": "claude-opus-5-5", "llm_mode": "mubit",
                                    "record_mode": "off", "engine_features": f} for k, f in feats.items()},
            "llm_policy": {"opus": opus_pol, "opus_bm": _pol("v2", "low")},
            "engine": {"features": feats}}}],
        "arms": [{"id": "a1", "kind": "opus"}, {"id": "a2", "kind": "opus_bm"}, {"id": "a3", "kind": "optuna"}],
        "llm_calls": [{"step_type": "propose_experiments"}, {"step_type": "judge_improvement"}],
    }


def test_fingerprint_records_lean_effort_and_keeps_standard_arms_poolable() -> None:
    ar = _load_script("analyze_races")
    fp = ar.fairness_fingerprint(_rows())
    assert fp["arms"]["opus_bm"]["llm_policy"]["propose_effort"] == "low"
    assert fp["arms"]["opus_bm"]["llm_policy"]["prompts"]["propose_experiments"].startswith("v2:cfd2e3dd9710")
    assert "propose_effort" not in fp["arms"]["opus"]["llm_policy"]           # standard effort: not a key
    # a standard arm with effort recorded pools with one from before effort was recorded
    assert ar.fingerprint_id(fp) == ar.fingerprint_id(ar.fairness_fingerprint(_rows(opus_effort=None)))
    assert ar.fingerprint_id(fp) != ar.fingerprint_id(ar.fairness_fingerprint(_rows(opus_effort="low")))
    assert ar.arm_setup(fp, "opus_bm") == {
        "features": "keeps GPUs busy with a no-AI sampler while the AI thinks + checks every claimed improvement + "
                    f"{LABEL}", "planner": "v2, effort low", "judge": "v1"}
    assert ar.arm_setup(fp, "opus") == {"features": "none", "planner": "v1, effort medium", "judge": "v1"}


def test_headline_setup_row_and_engine_table_show_the_lean_planner() -> None:
    from types import SimpleNamespace

    hl = _load_script("headline")
    fp = hl.ar.fairness_fingerprint(_rows())
    g = SimpleNamespace(fingerprint=fp, engine={"opus": hl.EngineStats(features=[]),
                                                 "opus_bm": hl.EngineStats(features=["lean_planner"]),
                                                 "optuna": hl.EngineStats(features=[])})
    assert "propose v2 (effort low)" in hl._llm_cell(g, "opus_bm") and "judge v1" in hl._llm_cell(g, "opus_bm")
    assert "propose v1 (effort medium)" in hl._llm_cell(g, "opus")
    assert LABEL in hl._features_cell(g, "opus_bm") and hl._features_cell(g, "opus") == "none"
    lines = "\n".join(hl._engine_lines(g, ["opus", "opus_bm", "optuna"]))
    assert f"| Lean planner ({LABEL}) | off · propose v1, effort medium | on · propose v2, effort low | " \
           "off (no planner) |" in lines
    assert "R63 ablation" in lines
    # the lean ABLATION arm is still refused as a headline race
    rows = {"races": [{"id": "abcdef0123", "fairness": {}, "config": {"mode": "dev", "arms": ["opus", "lean"]}}],
            "arms": [{"kind": "opus"}, {"kind": "lean"}]}
    assert hl.is_ablation_race(rows)
