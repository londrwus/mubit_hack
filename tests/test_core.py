"""Tests for the shared contracts (backend/core, workers/*/base.py, backend/agent/llm_base.py)."""

from __future__ import annotations

import random
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from backend.agent.llm_base import LLMClient, estimate_cost_usd, session_id_for
from backend.core.events import (
    EVENT_TYPES,
    ExperimentFinishedEvent,
    GpuStateEvent,
    LLMCallEvent,
    RaceFinishedEvent,
    parse_event,
)
from backend.core.types import (
    STEP_OUTPUT_SCHEMAS,
    ArmKind,
    ArmSummary,
    DiagnoseOutput,
    ExperimentStatus,
    JudgeOutput,
    LLMCallRecord,
    MetricPoint,
    ProposeOutput,
    RunHandle,
    RunResult,
    Slot,
    SlotState,
    StepType,
)
from workers.runners.base import MetricCallback, Runner, emit_metric, price_from_table
from workers.targets.base import (
    TARGETS,
    BaseTarget,
    EmitFn,
    ParamSpec,
    Target,
    TargetSpec,
    get_target,
    register_target,
)


def _spec() -> TargetSpec:
    return TargetSpec(
        name="unit",
        description="Fit a quadratic.",
        metric="loss",
        direction="min",
        target_value=0.1,
        hardware="CPU",
        budget_s=5,
        config_space={
            "lr": ParamSpec(type="float", low=1e-4, high=1e-1, log=True, default=1e-3),
            "layers": ParamSpec(type="int", low=1, high=8, default=2),
            "opt": ParamSpec(type="categorical", choices=["adam", "sgd"], default="adam"),
            "val_from_train": ParamSpec(type="bool", default=False, description="reuse train pool for val"),
        },
        trap_option="val_from_train",
    )


# --------------------------------------------------------------------------- types


def test_enums() -> None:
    assert [a.value for a in ArmKind] == ["opus", "mubit", "opus_bm", "optuna", "mock", "fast"]
    assert set(STEP_OUTPUT_SCHEMAS) == set(StepType)


def test_propose_output_hypothesis_limit() -> None:
    ok = ProposeOutput.model_validate({"experiments": [{"config": {"lr": 0.01}, "hypothesis": "higher lr"}]})
    assert ok.experiments[0].config == {"lr": 0.01}
    # D26: over-long hypotheses are trimmed to 25 words, not rejected
    long = ProposeOutput.model_validate({"experiments": [{"config": {}, "hypothesis": " ".join(["w"] * 26)}]})
    assert len(long.experiments[0].hypothesis.split()) == 25
    with pytest.raises(ValidationError):
        ProposeOutput.model_validate({"experiments": [{"config": {}, "hypothesis": "  "}]})
    with pytest.raises(ValidationError):
        ProposeOutput.model_validate({"experiments": []})
    with pytest.raises(ValidationError):
        ProposeOutput.model_validate({"experiments": [{"config": {}, "hypothesis": "x", "why": "extra"}]})


def test_propose_settings_list_wire_form() -> None:
    """D24: the target-agnostic [{name, value}] form decodes to a typed config dict."""
    out = ProposeOutput.model_validate({"experiments": [{"hypothesis": "h", "config": [
        {"name": "lr", "value": "3e-4"}, {"name": "n_embd", "value": "128"},
        {"name": "flip", "value": "true"}, {"name": "opt", "value": "adamw"}]}]})
    assert out.experiments[0].config == {"lr": 3e-4, "n_embd": 128, "flip": True, "opt": "adamw"}
    with pytest.raises(ValidationError):
        ProposeOutput.model_validate({"experiments": [{"hypothesis": "h", "config": [
            {"name": "lr", "value": "1"}, {"name": "lr", "value": "2"}]}]})


def test_diagnose_and_judge() -> None:
    d = DiagnoseOutput.model_validate_json('{"failure_class":"oom","cause":"batch too big","fix":"halve batch"}')
    assert d.failure_class == "oom"
    j = JudgeOutput.model_validate({"verdict": "invalid", "reason": "val overlaps train"})
    assert j.verdict == "invalid"
    with pytest.raises(ValidationError):
        JudgeOutput.model_validate({"verdict": "maybe", "reason": ""})


def test_run_result_duration_and_roundtrip() -> None:
    r = RunResult(score=1.2, status=ExperimentStatus.succeeded, started_at=10.0, ended_at=70.5, hardware="L4")
    assert r.duration_s == 60.5
    back = RunResult.model_validate_json(r.model_dump_json())
    assert back.score == 1.2 and back.status == "succeeded"


def test_llm_call_record_defaults() -> None:
    rec = LLMCallRecord(arm=ArmKind.mock, session_id="s", step_type=StepType.diagnose_run, model="mock")
    assert rec.recorded is False and rec.valid is False and len(rec.id) == 32
    s = ArmSummary(arm=ArmKind.opus, gpu_cost_usd=1.0, llm_cost_usd=0.5)
    assert s.total_cost_usd == 1.5


# --------------------------------------------------------------------------- events


def test_event_roundtrip_all_types() -> None:
    rec = LLMCallRecord(arm=ArmKind.opus, session_id="s", step_type=StepType.propose_experiments, model="m")
    evs = [
        GpuStateEvent(race_id="r", arm=ArmKind.opus, slot_id="s0", state=SlotState.idle),
        ExperimentFinishedEvent(race_id="r", arm=ArmKind.mubit, experiment_id="e", slot_id="s0",
                                status=ExperimentStatus.failed, error="NaN"),
        LLMCallEvent(race_id="r", call=rec),
        RaceFinishedEvent(race_id="r", summaries=[ArmSummary(arm=ArmKind.opus)]),
    ]
    for e in evs:
        parsed = parse_event(e.model_dump_json())
        assert type(parsed) is type(e)
        assert parsed == e
    assert parse_event({"type": "metric", "race_id": "r", "arm": "opus", "experiment_id": "e",
                        "step": 1, "t": 0.5, "name": "loss", "value": 2.0}).type == "metric"
    assert len(EVENT_TYPES) == 8
    with pytest.raises(ValidationError):
        parse_event({"type": "nope", "race_id": "r"})


# --------------------------------------------------------------------------- targets


def test_param_spec_validation() -> None:
    with pytest.raises(ValidationError):
        ParamSpec(type="float", low=1.0, high=0.0, default=0.5)
    with pytest.raises(ValidationError):
        ParamSpec(type="categorical", choices=["a"], default="b")
    with pytest.raises(ValidationError):
        ParamSpec(type="float", low=0.0, high=1.0, log=True, default=0.5)


def test_target_spec_rules() -> None:
    spec = _spec()
    assert spec.default_config() == {"lr": 1e-3, "layers": 2, "opt": "adam", "val_from_train": False}
    assert not spec.is_trap(spec.default_config())
    assert spec.is_trap({"val_from_train": True})
    assert spec.is_better(0.5, 1.0) and not spec.is_better(1.0, 0.5) and spec.is_better(3.0, None)
    assert spec.reaches_target(0.1) and not spec.reaches_target(0.2)
    with pytest.raises(ValidationError):  # trap must exist in the space
        TargetSpec(**{**spec.model_dump(), "trap_option": "missing"})
    with pytest.raises(ValidationError):  # default config must not be the trap
        TargetSpec(**{**spec.model_dump(), "trap_value": False})


def test_validate_config() -> None:
    spec = _spec()
    assert spec.validate_config({"layers": 4.0})["layers"] == 4
    for bad in ({"lr": 5.0}, {"layers": 2.5}, {"opt": "rmsprop"}, {"x": 1}, {"val_from_train": 1}, {"lr": True}):
        with pytest.raises(ValueError):
            spec.validate_config(bad)


def test_random_config_is_valid() -> None:
    spec = _spec()
    rng = random.Random(0)
    for _ in range(50):
        cfg = spec.random_config(rng)
        assert spec.validate_config(cfg) == cfg


def test_suggest_optuna_duck_typed() -> None:
    class FakeTrial:
        def suggest_float(self, n: str, lo: float, hi: float, log: bool = False) -> float:
            return lo

        def suggest_int(self, n: str, lo: int, hi: int, log: bool = False) -> int:
            return hi

        def suggest_categorical(self, n: str, choices: list[Any]) -> Any:
            return choices[0]

    cfg = _spec().suggest_optuna(FakeTrial())
    assert cfg == {"lr": 1e-4, "layers": 8, "opt": "adam", "val_from_train": False}


def test_config_json_schema() -> None:
    s = _spec().config_json_schema()
    assert s["additionalProperties"] is False and set(s["required"]) == {"lr", "layers", "opt", "val_from_train"}
    assert s["properties"]["opt"] == {"enum": ["adam", "sgd"]}


def test_registry_and_protocol() -> None:
    class Unit(BaseTarget):
        spec = _spec()

        def run(self, config: dict[str, Any], seed: int, budget_s: int, emit: EmitFn) -> dict[str, Any]:
            emit(0, "loss", 1.0)
            return {"score": 0.5, "steps": 1.0}

    t = register_target(Unit())
    assert isinstance(t, Target)
    assert get_target("unit") is t and t.metric == "loss" and t.trap_option == "val_from_train"
    seen: list[tuple[int, str, float]] = []
    assert t.run({}, 0, 1, lambda s, n, v: seen.append((s, n, v)))["score"] == 0.5
    assert seen == [(0, "loss", 1.0)]
    with pytest.raises(KeyError):
        get_target("does-not-exist")
    TARGETS.pop("unit")


# --------------------------------------------------------------------------- runner / llm protocols


class _FakeRunner:
    name = "fake"

    def slots(self) -> list[Slot]:
        return [Slot(slot_id="fake-CPU-0", index=0, hardware="CPU", runner=self.name)]

    async def launch(self, target: str, config: dict[str, Any], seed: int, budget_s: int,
                     on_metric: MetricCallback) -> RunHandle:
        h = RunHandle(runner=self.name, target=target, config=config, seed=seed, budget_s=budget_s, hardware="CPU")
        await emit_metric(on_metric, MetricPoint(experiment_id=h.run_id, step=0, t=0.0, name="loss", value=1.0))
        self._h = h
        return h

    async def wait(self, handle: RunHandle) -> RunResult:
        return RunResult(score=0.5, status=ExperimentStatus.succeeded, started_at=handle.started_at,
                         ended_at=handle.started_at + 1, hardware="CPU")

    async def cancel(self, handle: RunHandle) -> None:
        return None

    def price_per_second(self, hardware: str) -> float:
        return price_from_table(hardware)


async def test_runner_protocol_sync_and_async_callbacks() -> None:
    r = _FakeRunner()
    assert isinstance(r, Runner)
    got: list[MetricPoint] = []

    async def acb(p: MetricPoint) -> None:
        got.append(p)

    h = await r.launch("unit", {}, 0, 5, got.append)
    await r.launch("unit", {}, 0, 5, acb)
    assert len(got) == 2 and got[0].experiment_id == h.run_id
    res = await r.wait(h)
    assert res.status == "succeeded"
    assert r.price_per_second("L4") == pytest.approx(0.000222)


class _MockLLM:
    async def call(self, step: StepType, system: str, user: str, schema: type[BaseModel], arm: ArmKind,
                   session_id: str, record: bool = False) -> tuple[BaseModel | None, LLMCallRecord]:
        parsed = schema.model_validate({"verdict": "unsure", "reason": "mock"})
        return parsed, LLMCallRecord(arm=arm, session_id=session_id, step_type=step, model="mock",
                                     valid=True, recorded=record)


async def test_llm_protocol_and_helpers() -> None:
    llm = _MockLLM()
    assert isinstance(llm, LLMClient)
    sid = session_id_for("42", ArmKind.mubit, 3)
    assert sid == "race-42/mubit/seed-3"
    parsed, rec = await llm.call(StepType.judge_improvement, "sys", "usr", JudgeOutput, ArmKind.mock, sid)
    assert isinstance(parsed, JudgeOutput) and rec.recorded is False
    assert estimate_cost_usd("claude-opus-5-5", 1_000_000, 0, 0, 0) == pytest.approx(4.0)
    assert estimate_cost_usd("unknown", 0, 0, 0, 1_000_000) == pytest.approx(20.0)
