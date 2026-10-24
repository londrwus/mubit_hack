"""judge_improvement v1 (D37, closed output; frozen from the D39-gated draft v1b). No network.

1. The frozen v1 prompt is the gate's v1b draft, byte for byte (fingerprint 5c1d3cfd2941); v0
   stays on disk and keeps its fingerprint; the other frozen steps are untouched.
2. The wire schema is chosen by prompt version: v0 = verdict + free-text reason, v1 = the
   closed 2-enum schema scripts/gate_judge_v1.py measured.
3. JudgeOutput parses both v0 and v1 replies; ``reason`` is filled from ``reason_code`` so the
   scheduler (which stores ``out.reason``) keeps working.
4. The race loop (LLMResearcher + RepoPrompts) sends v1 and parses a v1 reply.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError

from backend.agent.llm_base import OPUS_MODEL, REQUEST_ID_HEADER, session_id_for
from backend.agent.loop import LLMResearcher, RepoPrompts
from backend.agent.mubit_client import MubitClient
from backend.agent.prompts import CURRENT_VERSION, FROZEN, PROMPTS_DIR, load_system_prompt, prompt_fingerprint
from backend.agent.prompts.build import judge_wire_schema, wire_schema_for
from backend.core.types import (
    JUDGE_REASON_VERDICT,
    ArmKind,
    ExperimentStatus,
    JudgeOutput,
    JudgeReasonCode,
    StepType,
    Verdict,
)
from backend.scheduler import ExperimentSnapshot, JudgeClaim
from workers.targets.base import ParamSpec, TargetSpec

ROOT = Path(__file__).resolve().parents[1]
J = StepType.judge_improvement
V1_FINGERPRINT = "5c1d3cfd2941"
V0_FINGERPRINT = "623366123052"


def _gate() -> Any:
    spec = importlib.util.spec_from_file_location("gate_judge_v1", ROOT / "scripts" / "gate_judge_v1.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# --------------------------------------------------------------------------- 1. prompt


def test_v1_is_current_and_frozen() -> None:
    assert CURRENT_VERSION[J] == FROZEN[J] == "v1"
    assert prompt_fingerprint(J) == V1_FINGERPRINT
    assert prompt_fingerprint(J, "v1") == V1_FINGERPRINT


def test_v1_is_the_gated_v1b_draft_byte_for_byte() -> None:
    gate = _gate()
    assert load_system_prompt(J, "v1") == gate.load_prompt("v1b")
    assert gate.fingerprint(gate.load_prompt("v1b")) == V1_FINGERPRINT
    raw = (PROMPTS_DIR / "judge_improvement.v1.md").read_text(encoding="utf-8")
    assert raw.startswith("<!-- FROZEN v1")
    text = load_system_prompt(J)
    assert "<!--" not in text and "\r" not in text and "reason_code" in text


def test_v0_kept_on_disk_unchanged() -> None:
    assert (PROMPTS_DIR / "judge_improvement.v0.md").exists()
    assert prompt_fingerprint(J, "v0") == V0_FINGERPRINT


# --------------------------------------------------------------------------- 2. wire schema


def test_wire_schema_by_version() -> None:
    gate = _gate()
    v1 = judge_wire_schema("v1")
    assert v1 == gate.wire_schema("v1b")
    assert json.dumps(v1) == json.dumps(gate.wire_schema("v1b"))  # same key order on the wire
    assert judge_wire_schema() == v1 == wire_schema_for(J)
    v0 = judge_wire_schema("v0")
    assert v0["required"] == ["verdict", "reason"] and v0["properties"]["reason"] == {"type": "string"}
    with pytest.raises(ValueError):
        judge_wire_schema("v9")


def test_reason_codes_match_gate_and_pair_with_verdicts() -> None:
    gate = _gate()
    codes = gate.VARIANTS["v1b"][1]
    assert [c.value for c in JudgeReasonCode] == list(codes)
    assert {c.value: v.value for c, v in JUDGE_REASON_VERDICT.items()} == codes


# --------------------------------------------------------------------------- 3. output model


def test_judge_output_parses_v0_and_v1() -> None:
    v0 = JudgeOutput.model_validate({"verdict": "invalid", "reason": "val overlaps train"})
    assert v0.reason == "val overlaps train" and v0.reason_code is None
    v1 = JudgeOutput.model_validate_json('{"verdict":"unsure","reason_code":"insufficient_evidence"}')
    assert v1.verdict is Verdict.unsure and v1.reason_code is JudgeReasonCode.insufficient_evidence
    assert v1.reason == "insufficient_evidence"   # what the scheduler stores as verdict_reason
    with pytest.raises(ValidationError):
        JudgeOutput.model_validate({"verdict": "unsure", "reason_code": "made_up"})
    with pytest.raises(ValidationError):
        JudgeOutput.model_validate({"verdict": "valid"})      # neither reason nor reason_code
    with pytest.raises(ValidationError):
        JudgeOutput.model_validate({"verdict": "valid", "reason_code": "clean_gain", "extra": 1})


# --------------------------------------------------------------------------- 4. race loop


@pytest.fixture
def toy() -> TargetSpec:
    return TargetSpec(
        name="toy", description="A tiny MLP regression task for tests.", metric="val_loss", direction="min",
        target_value=0.05, hardware="CPU", budget_s=20,
        config_space={
            "lr": ParamSpec(type="float", low=1e-4, high=1e-1, log=True, default=1e-3, description="learning rate"),
            "val_from_train_pool": ParamSpec(type="bool", default=False, description="draw validation rows from the training pool"),
        },
        trap_option="val_from_train_pool",
    )


async def test_race_loop_sends_v1_and_parses_reply(toy: TargetSpec) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        body = {"id": "msg_test", "type": "message", "role": "assistant", "model": OPUS_MODEL,
                "content": [{"type": "text", "text": '{"verdict":"invalid","reason_code":"contamination_or_eval_leak"}'}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 0}}
        return httpx2.Response(200, json=body, headers={REQUEST_ID_HEADER: "mreq_test"})

    llm = MubitClient(api_key="test-mubit-key", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    r = LLMResearcher(kind=ArmKind.opus, llm=llm, spec=toy, session_id=session_id_for("r1", ArmKind.opus, 0),
                      prompts=RepoPrompts())
    cand = ExperimentSnapshot(id="e2", config={"lr": 0.01, "val_from_train_pool": True},
                              status=ExperimentStatus.succeeded, score=0.01)
    prev = ExperimentSnapshot(id="e1", config=toy.default_config(), status=ExperimentStatus.succeeded, score=0.2)
    out = await r.judge(JudgeClaim(candidate=cand, previous_best=prev, delta=-0.19, improvement=0.19,
                                   config_diff={}))
    assert out is not None and out.verdict is Verdict.invalid
    assert out.reason_code is JudgeReasonCode.contamination_or_eval_leak and out.reason == out.reason_code.value
    body = json.loads(requests[0].content)
    assert body["system"][0]["text"] == load_system_prompt(J, "v1")
    assert body["output_config"]["format"]["schema"] == judge_wire_schema("v1")
    assert "judge_improvement=v1:" + V1_FINGERPRINT in RepoPrompts().version
