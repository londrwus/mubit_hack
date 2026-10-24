"""Tests for backend/agent/mubit_client.py, mock_llm.py and prompts/ (no network).

The real client is exercised through a fake httpx2 transport that records each request.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest

from backend.agent.llm_base import DEV_MODEL, OPUS_MODEL, RECORD_HEADER, REQUEST_ID_HEADER, LLMClient, session_id_for
from backend.agent.mock_llm import MOCK_MODEL, MockLLM
from backend.agent.mubit_client import (
    MubitClient,
    _sanitize_schema,
    call_with_fallback,
    is_benchmark_safe,
    schema_from_model,
)
from backend.agent.prompts import CURRENT_VERSION, FROZEN, load_system_prompt, prompt_fingerprint
from backend.agent.prompts.build import (
    CACHE_BREAK,
    JudgeClaim,
    ResultRow,
    build_diagnose_user,
    build_judge_user,
    build_propose_user,
    diagnose_wire_schema,
    judge_wire_schema,
    propose_validator,
    propose_wire_schema,
    split_user,
    strip_cache_break,
    target_spec_block,
)
from backend.core.types import (
    ArmKind,
    DiagnoseOutput,
    FailureClass,
    JudgeOutput,
    ProposeOutput,
    StepType,
    Verdict,
)
from workers.targets.base import ParamSpec, TargetSpec

SESSION = session_id_for("r1", ArmKind.opus, 0)


@pytest.fixture
def spec() -> TargetSpec:
    return TargetSpec(
        name="toy",
        description="A tiny MLP regression task for tests.",
        metric="val_loss",
        direction="min",
        target_value=0.05,
        hardware="CPU",
        budget_s=20,
        config_space={
            "lr": ParamSpec(type="float", low=1e-4, high=1e-1, log=True, default=1e-3, description="learning rate"),
            "width": ParamSpec(type="int", low=8, high=256, default=32, description="hidden width"),
            "act": ParamSpec(type="categorical", choices=["relu", "gelu", "tanh"], default="relu", description="activation"),
            "val_from_train_pool": ParamSpec(type="bool", default=False, description="draw validation rows from the training pool"),
        },
        trap_option="val_from_train_pool",
    )


def _history(spec: TargetSpec, n: int = 12) -> list[ResultRow]:
    rows = []
    for i in range(n):
        cfg = spec.default_config() | {"lr": 1e-3 * (1 + i), "width": 16 + 8 * i}
        rows.append(ResultRow(label=f"#{i + 1}", config=cfg, score=0.5 - 0.01 * i if i != 3 else None,
                              status="succeeded" if i != 3 else "failed", note="diverged" if i == 3 else None))
    return rows


# --------------------------------------------------------------------------- fake transport


class FakeGateway:
    """Queue of responses; records every request."""

    def __init__(self, responses: list[httpx2.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    def client(self, **kw: Any) -> MubitClient:
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler))
        return MubitClient(api_key="test-mubit-key", http_client=http, **kw)

    def body(self, i: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[i].content)


def msg_response(payload: Any, *, stop_reason: str = "end_turn", req_id: str = "mreq_test1",
                 usage: dict[str, int] | None = None, raw_text: str | None = None) -> httpx2.Response:
    text = raw_text if raw_text is not None else json.dumps(payload)
    body = {
        "id": "msg_test", "type": "message", "role": "assistant", "model": OPUS_MODEL,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": usage or {"input_tokens": 1000, "output_tokens": 500,
                           "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 3000},
    }
    return httpx2.Response(200, json=body, headers={REQUEST_ID_HEADER: req_id})


GOOD_JUDGE = {"verdict": "valid", "reason": "Splits are distinct."}


# --------------------------------------------------------------------------- MubitClient


async def test_request_layout_headers_and_record(spec: TargetSpec) -> None:
    gw = FakeGateway([msg_response(GOOD_JUDGE)])
    c = gw.client()
    user = build_judge_user(spec, JudgeClaim(prev_best_score=0.3, new_score=0.2, new_config=spec.default_config()))
    parsed, rec = await c.call(StepType.judge_improvement, "SYSTEM", user, JudgeOutput, ArmKind.opus, SESSION)

    assert isinstance(parsed, JudgeOutput) and parsed.verdict is Verdict.valid
    req = gw.requests[0]
    assert str(req.url) == "https://api.mubit.ai/anthropic/v1/messages"
    assert req.headers[RECORD_HEADER] == "off"
    assert "test-mubit-key" in (req.headers.get("x-api-key", "") + req.headers.get("authorization", ""))
    body = gw.body()
    assert body["model"] == OPUS_MODEL
    assert body["system"] == [{"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}]
    blocks = body["messages"][0]["content"]
    assert len(blocks) == 2
    assert blocks[0]["cache_control"] == {"type": "ephemeral"} and blocks[0]["text"].startswith("TARGET SPEC")
    assert "cache_control" not in blocks[1] and blocks[1]["text"].startswith("CLAIM")
    assert CACHE_BREAK.strip() not in json.dumps(body)
    assert body["metadata"] == {"user_id": SESSION}
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "thinking" not in body and "temperature" not in body

    assert rec.valid and rec.error is None and not rec.recorded
    assert rec.mubit_request_id == "mreq_test1"
    assert (rec.input_tokens, rec.cache_creation_tokens, rec.cache_read_tokens, rec.output_tokens) == (1000, 2000, 3000, 500)
    # Opus 5.5: 1000*4 + 2000*5 + 3000*0.2 + 500*20 = 24600 per 1M
    assert rec.cost_usd == pytest.approx(0.0246)
    assert rec.latency_ms > 0 and rec.stop_reason == "end_turn"
    assert rec.session_id == SESSION and rec.step_type is StepType.judge_improvement


async def test_record_true_sends_no_off_header() -> None:
    gw = FakeGateway([msg_response(GOOD_JUDGE)])
    _, rec = await gw.client().call(StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.opus, SESSION, record=True)
    assert RECORD_HEADER not in gw.requests[0].headers
    assert rec.recorded


async def test_step_efforts_and_dev_model(spec: TargetSpec) -> None:
    replies = [
        msg_response({"experiments": [{"config": spec.default_config() | {"lr": 0.002}, "hypothesis": "Try a bit more lr."}]}),
        msg_response({"failure_class": "oom", "cause": "OOM.", "fix": "Smaller batch."}),
    ]
    gw = FakeGateway(replies)
    c = gw.client(dev=True)
    user = build_propose_user(spec, _history(spec), free_slots=2)
    parsed, rec = await c.call(StepType.propose_experiments, "S", user, ProposeOutput, ArmKind.mubit, SESSION,
                               wire_schema=propose_wire_schema(spec), validator=propose_validator(spec, 2))
    assert parsed is not None and parsed.experiments[0].config["lr"] == 0.002
    assert gw.body(0)["output_config"]["effort"] == "medium"
    assert gw.body(0)["model"] == DEV_MODEL and rec.model == DEV_MODEL
    parsed2, _ = await c.call(StepType.diagnose_run, "S", "u", DiagnoseOutput, ArmKind.opus, SESSION)
    assert parsed2 is not None and parsed2.failure_class is FailureClass.oom
    assert gw.body(1)["output_config"]["effort"] == "low"


async def test_mubit_arm_sends_same_model_as_opus_arm() -> None:
    gw = FakeGateway([msg_response(GOOD_JUDGE), msg_response(GOOD_JUDGE)])
    c = gw.client()
    await c.call(StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.opus, SESSION)
    await c.call(StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.mubit, SESSION)
    assert gw.body(0)["model"] == gw.body(1)["model"] == OPUS_MODEL


async def test_cost_source_tags_mubit_arm_cost_as_an_opus_list_estimate() -> None:
    """The mubit arm's cost is Opus list price x usage (Mubit may answer with an adapter whose
    price we do not see), so it is tagged; the other arms are list-price estimates."""
    gw = FakeGateway([msg_response(GOOD_JUDGE) for _ in range(3)])
    c = gw.client()
    recs = {}
    for arm in (ArmKind.opus, ArmKind.opus_bm, ArmKind.mubit):
        _, recs[arm] = await c.call(StepType.judge_improvement, "S", "u", JudgeOutput, arm, SESSION)
    assert recs[ArmKind.mubit].cost_source == "estimate@opus_list"
    assert recs[ArmKind.opus].cost_source == recs[ArmKind.opus_bm].cost_source == "estimate@list"
    assert recs[ArmKind.opus_bm].model == OPUS_MODEL     # DEFAULT_ARM_MODELS has opus_bm
    assert recs[ArmKind.mubit].cost_usd == recs[ArmKind.opus].cost_usd > 0


@pytest.mark.parametrize(
    ("reply", "fragment"),
    [
        (msg_response(None, raw_text="not json"), "invalid JSON"),
        (msg_response({"verdict": "maybe", "reason": "x"}), "schema"),
        (msg_response({"verdict": "valid", "reason": "x", "extra": 1}), "schema"),
        (msg_response(GOOD_JUDGE, stop_reason="max_tokens"), "max_tokens"),
        (msg_response(GOOD_JUDGE, stop_reason="refusal"), "refusal"),
    ],
)
async def test_invalid_replies_never_raise(reply: httpx2.Response, fragment: str) -> None:
    gw = FakeGateway([reply])
    parsed, rec = await gw.client().call(StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.opus, SESSION)
    assert parsed is None and not rec.valid
    assert rec.error is not None and fragment in rec.error
    assert rec.input_tokens == 1000  # usage still counted (it was paid for)


async def test_validator_rejects_out_of_range_config(spec: TargetSpec) -> None:
    bad = {"experiments": [{"config": spec.default_config() | {"lr": 5.0}, "hypothesis": "Huge lr."}]}
    gw = FakeGateway([msg_response(bad)])
    parsed, rec = await gw.client().call(StepType.propose_experiments, "S", "u", ProposeOutput, ArmKind.opus, SESSION,
                                         wire_schema=propose_wire_schema(spec), validator=propose_validator(spec))
    assert parsed is None and rec.error is not None and "lr" in rec.error


async def test_long_hypothesis_is_trimmed_not_rejected(spec: TargetSpec) -> None:
    """D26: an over-long hypothesis is trimmed to 25 words; the reply stays valid (no retry)."""
    settings = [{"name": k, "value": json.dumps(v)} for k, v in spec.default_config().items()]
    long = {"experiments": [{"config": settings, "hypothesis": " ".join(["word"] * 26)}]}
    gw = FakeGateway([msg_response(long)])
    parsed, rec = await gw.client().call(StepType.propose_experiments, "S", "u", ProposeOutput, ArmKind.opus, SESSION,
                                         wire_schema=propose_wire_schema(spec))
    assert parsed is not None and rec.valid
    assert len(parsed.experiments[0].hypothesis.split()) == 25
    assert parsed.experiments[0].config == spec.default_config()


async def test_propose_without_wire_schema_fails_without_request(spec: TargetSpec) -> None:
    gw = FakeGateway([])
    parsed, rec = await gw.client().call(StepType.propose_experiments, "S", "u", ProposeOutput, ArmKind.opus, SESSION)
    assert parsed is None and "wire_schema" in (rec.error or "") and gw.requests == []


async def test_http_error_one_request_no_raise() -> None:
    err = httpx2.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}},
                          headers={REQUEST_ID_HEADER: "mreq_err"})
    gw = FakeGateway([err])
    parsed, rec = await gw.client().call(StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.opus, SESSION)
    assert parsed is None and rec.error is not None and "500" in rec.error
    assert rec.mubit_request_id == "mreq_err"
    assert len(gw.requests) == 1  # SDK auto-retry disabled


async def test_call_with_fallback_retry_then_opus() -> None:
    gw = FakeGateway([msg_response(None, raw_text="{"), msg_response({"verdict": "x"}), msg_response(GOOD_JUDGE)])
    c = gw.client(dev=True)
    parsed, recs = await call_with_fallback(c, StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.mubit, SESSION)
    assert parsed is not None and parsed.verdict is Verdict.valid
    assert [(r.is_retry, r.is_fallback, r.valid) for r in recs] == [(False, False, False), (True, False, False), (False, True, True)]
    assert [gw.body(i)["model"] for i in range(3)] == [DEV_MODEL, DEV_MODEL, OPUS_MODEL]
    assert all(r.headers[RECORD_HEADER] == "off" for r in gw.requests)


async def test_call_with_fallback_stops_at_first_valid() -> None:
    gw = FakeGateway([msg_response(GOOD_JUDGE)])
    parsed, recs = await call_with_fallback(gw.client(), StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.opus, SESSION)
    assert parsed is not None and len(recs) == 1


def test_base_url_guard() -> None:
    with pytest.raises(ValueError):
        MubitClient(api_key="k", base_url="https://api.anthropic.com")


def test_base_url_guard_applies_with_custom_http_client() -> None:
    # A custom transport must not open a path around the Mubit gateway.
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(lambda r: httpx2.Response(500)))
    with pytest.raises(ValueError):
        MubitClient(api_key="k", base_url="https://api.anthropic.com", http_client=http)


@pytest.mark.parametrize(("mode", "header", "recorded"),
                         [("metadata", "metadata", False), ("off", "off", False), ("full", None, True),
                          ("bogus", "off", False)])
async def test_record_modes(mode: str, header: str | None, recorded: bool) -> None:
    gw = FakeGateway([msg_response(GOOD_JUDGE)])
    _, rec = await gw.client().call(StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.mubit, SESSION,
                                    record=mode)  # type: ignore[arg-type]
    assert gw.requests[0].headers.get(RECORD_HEADER) == header
    assert rec.recorded is recorded


async def test_validator_crash_is_an_invalid_reply_not_an_exception() -> None:
    def boom(_: JudgeOutput) -> JudgeOutput:
        raise KeyError("lr")

    gw = FakeGateway([msg_response(GOOD_JUDGE)])
    parsed, rec = await gw.client().call(StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.opus, SESSION,
                                         validator=boom)
    assert parsed is None and not rec.valid and rec.error is not None and "KeyError" in rec.error


def test_client_satisfies_protocol() -> None:
    gw = FakeGateway([])
    assert isinstance(gw.client(), LLMClient)
    assert isinstance(MockLLM(), LLMClient)
    assert is_benchmark_safe(gw.client()) and not is_benchmark_safe(MockLLM())


def _walk(schema: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(schema, dict):
        out.append(schema)
        for v in schema.values():
            out += _walk(v)
    elif isinstance(schema, list):
        for v in schema:
            out += _walk(v)
    return out


BANNED = {"minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems", "multipleOf"}


def test_wire_schemas_are_closed_and_supported(spec: TargetSpec) -> None:
    # JudgeOutput parses both judge versions, so its own schema is never sent: the loop always
    # passes the versioned judge_wire_schema (v0 and v1 checked here).
    for s in (propose_wire_schema(spec), diagnose_wire_schema(), judge_wire_schema("v0"), judge_wire_schema("v1"),
              schema_from_model(DiagnoseOutput)):
        for node in _walk(s):
            assert not (BANNED & set(node)), node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False and set(node["required"]) == set(node["properties"])
    assert _sanitize_schema({"type": "array", "minItems": 1, "items": {"type": "integer", "minimum": 0}}) == {
        "type": "array", "items": {"type": "integer"}}
    # K must not change the output format (it would split the Mubit task)
    assert json.dumps(propose_wire_schema(spec)) == json.dumps(propose_wire_schema(spec))


# --------------------------------------------------------------------------- prompts / builders


@pytest.mark.parametrize("step", list(StepType))
def test_system_prompts_cacheable_and_static(step: StepType) -> None:
    text = load_system_prompt(step)
    assert len(text) / 4 >= 600, "system prompt should be >= ~600 tokens so it caches"
    assert "DRAFT" not in text and "<!--" not in text  # header stripped
    assert "\r" not in text
    assert prompt_fingerprint(step) == prompt_fingerprint(step)
    # D27: all three prompts are frozen, and the frozen version is the one actually sent
    # (a CURRENT_VERSION bump without a human re-freeze would split the Mubit task).
    assert FROZEN[step] is not None and FROZEN[step] == CURRENT_VERSION[step]


def test_builders_deterministic_and_hide_trap(spec: TargetSpec) -> None:
    h = _history(spec)
    u1 = build_propose_user(spec, h, free_slots=3, running=[spec.default_config()])
    assert u1 == build_propose_user(spec, h, free_slots=3, running=[spec.default_config()])
    prefix, rest = split_user(u1)
    assert prefix == target_spec_block(spec)
    assert "trap" not in prefix.lower() and "val_from_train_pool" in prefix
    assert "FREE SLOTS (K): 3" in rest
    lb = rest.split("leaderboard")[1].split("recent results")[0]
    assert lb.count("\n  #") == 5 and "#12" in lb.splitlines()[1]  # best (lowest) first
    assert rest.split("recent results")[1].split("current best")[0].count("\n  #") == 8
    assert "current best config: lr=0.012" in rest
    assert CACHE_BREAK not in strip_cache_break(u1)
    # identical spec block across steps -> same cached prefix within a race
    d = build_diagnose_user(spec, spec.default_config(), "failed", "nan loss", [f"step {i}" for i in range(80)])
    assert split_user(d)[0] == prefix
    assert split_user(d)[1].count("\n  step ") == 50 and "step 79" in d and "step 29" not in d
    j = build_judge_user(spec, JudgeClaim(prev_best_score=0.3, prev_best_config=spec.default_config(), new_score=0.1,
                                          new_config=spec.default_config() | {"val_from_train_pool": True},
                                          evidence={"val_train_overlap": 512, "val_hash": "ab12"}))
    assert "changed parameters: val_from_train_pool false -> true" in j and "claimed gain: -0.2" in j
    assert "  val_hash: ab12\n  val_train_overlap: 512" in j


def test_free_slots_must_be_positive(spec: TargetSpec) -> None:
    with pytest.raises(ValueError):
        build_propose_user(spec, [], free_slots=0)


# --------------------------------------------------------------------------- MockLLM


async def test_mock_propose_deterministic_valid_k(spec: TargetSpec) -> None:
    m = MockLLM(spec, seed=7, trap_rate=0.0)
    user = build_propose_user(spec, _history(spec), free_slots=4)
    sys = load_system_prompt(StepType.propose_experiments)
    p1, r1 = await m.call(StepType.propose_experiments, sys, user, ProposeOutput, ArmKind.mock, SESSION,
                          validator=propose_validator(spec, 4))
    p2, _ = await MockLLM(spec, seed=7, trap_rate=0.0).call(StepType.propose_experiments, sys, user, ProposeOutput,
                                                            ArmKind.mock, SESSION)
    assert p1 is not None and p2 is not None and p1 == p2
    assert len(p1.experiments) == 4
    assert len({json.dumps(e.config, sort_keys=True) for e in p1.experiments}) == 4
    for e in p1.experiments:
        assert spec.validate_config(e.config) == e.config and not spec.is_trap(e.config)
    assert r1.valid and r1.is_replay and r1.model == MOCK_MODEL and r1.cost_usd == 0 and not r1.recorded
    p3, _ = await MockLLM(spec, seed=8, trap_rate=0.0).call(StepType.propose_experiments, sys, user, ProposeOutput,
                                                            ArmKind.mock, SESSION)
    assert p3 != p1


async def test_mock_trap_rate_one(spec: TargetSpec) -> None:
    m = MockLLM(spec, trap_rate=1.0)
    p, _ = await m.call(StepType.propose_experiments, "S", build_propose_user(spec, [], 2), ProposeOutput, ArmKind.mock, SESSION)
    assert p is not None and all(spec.is_trap(e.config) for e in p.experiments)


@pytest.mark.parametrize(
    ("error", "logs", "expected"),
    [
        ("RuntimeError: CUDA out of memory", ["step 1"], FailureClass.oom),
        (None, ["step 10 loss=nan"], FailureClass.divergence),
        ("Traceback (most recent call last)", ["KeyError: 'x'"], FailureClass.crash),
        (None, ["step 10 loss=1.2", "step 20 loss=0.9"], FailureClass.no_failure),
    ],
)
async def test_mock_diagnose(spec: TargetSpec, error: str | None, logs: list[str], expected: FailureClass) -> None:
    user = build_diagnose_user(spec, spec.default_config(), "failed" if error or expected != FailureClass.no_failure else "succeeded", error, logs)
    p, _ = await MockLLM(spec).call(StepType.diagnose_run, "S", user, DiagnoseOutput, ArmKind.mock, SESSION)
    assert p is not None and p.failure_class is expected


async def test_mock_judge(spec: TargetSpec) -> None:
    m = MockLLM(spec)
    base = spec.default_config()
    trap = JudgeClaim(prev_best_score=0.3, prev_best_config=base, new_score=0.1,
                      new_config=base | {"val_from_train_pool": True}, evidence={"split_hashes": "a,b,c"})
    clean = JudgeClaim(prev_best_score=0.3, prev_best_config=base, new_score=0.25,
                       new_config=base | {"lr": 0.003}, evidence={"val_train_overlap": 0})
    overlap = JudgeClaim(prev_best_score=0.3, new_score=0.2, new_config=base | {"lr": 0.003},
                         evidence={"val_train_overlap": 12})
    verdicts = []
    for claim in (trap, clean, overlap):
        p, _ = await m.call(StepType.judge_improvement, "S", build_judge_user(spec, claim), JudgeOutput, ArmKind.mock, SESSION)
        assert p is not None
        verdicts.append(p.verdict)
    assert verdicts == [Verdict.invalid, Verdict.valid, Verdict.invalid]


async def test_mock_invalid_rate_exercises_fallback(spec: TargetSpec) -> None:
    m = MockLLM(spec, invalid_rate=1.0, latency_s=0.01)
    parsed, recs = await call_with_fallback(m, StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.mock, SESSION)
    assert parsed is not None
    assert [r.valid for r in recs] == [False, False, True] and recs[-1].is_fallback
    assert all(r.latency_ms >= 5 for r in recs)  # sleep(0.01); Windows timers can wake ~1 ms early
    assert len(m.calls) == 3


def test_app_refuses_recording_draft_prompts() -> None:
    """Hard-rule hygiene: a real-LLM race with record=True is refused while any prompt is a draft."""
    from backend.agent.prompts import is_frozen
    from backend.app import default_arm_deps
    from backend.benchmark.configs import RaceConfig
    from workers.targets.base import get_target

    if all(is_frozen(s) for s in StepType):
        pytest.skip("prompts are frozen; the guard no longer applies")
    cfg = RaceConfig(target="lm", budget_s=10, gpus_per_arm=1, llm="mubit", record=True)
    with pytest.raises(ValueError, match="frozen"):
        default_arm_deps("r", cfg, get_target("lm").spec, ArmKind.opus)
