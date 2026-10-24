"""The lean-planner prompt ablation arm (ArmKind.lean, preset lcbench-lean-15m): dev mode only,
record off, stored in ablation.db, refused by headline.py, and its propose prompt/effort swap."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agent.mubit_client import DEFAULT_STEP_CONFIGS, StepConfig
from backend.benchmark.configs import ARM_PROPOSE_VARIANTS, PRESETS, RaceConfig
from backend.core.types import ArmKind, StepType

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_preset_matches_lcbench_15m_except_arms() -> None:
    lean, base = PRESETS["lcbench-lean-15m"], PRESETS["lcbench-15m"]
    assert lean.arms == [ArmKind.opus, ArmKind.lean]
    assert lean.mode == "dev" and lean.llm == "mubit" and lean.runner == "surrogate"
    for f in ("target", "budget_s", "gpus_per_arm", "max_proposals_per_call", "cost_cap_usd", "pipelining",
              "run_baseline_first", "accept_unsure", "launch_cutoff_s", "prewarm", "model"):
        assert getattr(lean, f) == getattr(base, f), f
    assert lean.record_modes() == {"opus": "off", "lean": "off"}
    assert lean.model_for(ArmKind.lean) == lean.model_for(ArmKind.opus) == "claude-opus-5-5"
    assert ARM_PROPOSE_VARIANTS[ArmKind.lean] == ("v2", "low")


@pytest.mark.parametrize("mode", ["benchmark", "ablation"])
def test_lean_refused_outside_dev(mode: str) -> None:
    with pytest.raises(ValueError, match="dev-mode prompt ablations only"):
        RaceConfig(target="lcbench", arms=[ArmKind.opus, ArmKind.lean], budget_s=60, gpus_per_arm=1,
                   mode=mode, llm="mubit", record_mode={ArmKind.opus: "off", ArmKind.lean: "off"})


def test_lean_must_not_record() -> None:
    with pytest.raises(ValueError, match="record mode must be 'off'"):
        RaceConfig(target="lcbench", arms=[ArmKind.opus, ArmKind.lean], budget_s=60, gpus_per_arm=1,
                   mode="dev", llm="mubit", record=True)


def test_apply_propose_variants_swaps_prompt_and_effort() -> None:
    bench = _load("bench")
    from backend.agent.loop import load_prompts

    def dep(kind: ArmKind) -> SimpleNamespace:
        llm = SimpleNamespace(step_configs=dict(DEFAULT_STEP_CONFIGS))
        return SimpleNamespace(kind=kind, researcher=SimpleNamespace(prompts=load_prompts(), llm=llm))

    cfg = PRESETS["lcbench-lean-15m"]
    opus, lean = dep(ArmKind.opus), dep(ArmKind.lean)
    bench.apply_propose_variants(cfg, [opus, lean])
    assert opus.researcher.prompts.propose_version == "v1"
    assert opus.researcher.llm.step_configs[StepType.propose_experiments] == StepConfig("medium", 8000)
    assert lean.researcher.prompts.propose_version == "v2"
    assert lean.researcher.llm.step_configs[StepType.propose_experiments] == StepConfig("low", 8000)
    assert lean.researcher.llm.step_configs[StepType.judge_improvement] == DEFAULT_STEP_CONFIGS[StepType.judge_improvement]
    assert bench.is_prompt_ablation(cfg) and not bench.is_prompt_ablation(PRESETS["lcbench-15m"])
    # the mock LLM answers diff-form (v2) prompts too (D46): a dry run takes the same path
    dry = cfg.model_copy(update={"llm": "mock"})
    lean2 = dep(ArmKind.lean)
    bench.apply_propose_variants(dry, [lean2])
    assert lean2.researcher.prompts.propose_version == "v2"
    # idempotent
    bench.apply_propose_variants(cfg, [lean])
    assert lean.researcher.llm.step_configs[StepType.propose_experiments] == StepConfig("low", 8000)


def test_headline_refuses_lean_races() -> None:
    headline = _load("headline")
    rows = {"races": [{"id": "abcdef0123", "fairness": {}, "config": {"mode": "dev", "arms": ["opus", "lean"]}}],
            "arms": [{"kind": "opus"}, {"kind": "lean"}]}
    assert headline.is_ablation_race(rows)
