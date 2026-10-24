"""Review fix: the mubit arm's "fallback to Opus" is a provider call, not the same request again.

Before the fix the first try, the retry and the fallback were the same request (same model,
same record mode, same project, same prompt, same adapter header), so a routed adapter that
answered badly could answer all three; the records still said ``is_fallback`` and cost Opus.
Now a fallback goes out with ``x-middleware-record: off`` (no task fingerprint: not routable)
and without adapter-pinning headers. Fake httpx2 transport; the keys are test strings.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2

from backend.agent.llm_base import OPUS_MODEL, RECORD_HEADER, REQUEST_ID_HEADER
from backend.agent.loop import LLMResearcher
from backend.agent.mock_llm import MockLLM
from backend.agent.mubit_client import MubitClient, ProjectRouting, call_with_fallback
from backend.core.types import ArmKind, JudgeOutput, StepType
from backend.scheduler import llm_policy_of
from test_busy_machines import SPEC

KEY_A, KEY_B = "test-key-project-a", "test-key-project-b"
ADAPTER = {"x-adapter-id": "ad_lane_1"}
ROUTING = ProjectRouting.from_dict({
    "default": "A", "keys": {"A": "MUBIT_API_KEY", "B": "MUBIT_API_KEY_B"},
    "routes": {"mubit": {"judge_improvement": "B"}},
    "adapter_headers": {"mubit": {"judge_improvement": ADAPTER}},
})
GOOD = {"verdict": "valid", "reason": "Splits are distinct."}


def _reply(payload: Any = None, raw: str | None = None) -> httpx2.Response:
    body = {"id": "msg", "type": "message", "role": "assistant", "model": OPUS_MODEL,
            "content": [{"type": "text", "text": raw if raw is not None else json.dumps(payload)}],
            "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}}
    return httpx2.Response(200, json=body, headers={REQUEST_ID_HEADER: "mreq_t"})


class Gateway:
    def __init__(self, responses: list[httpx2.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    def client(self) -> MubitClient:
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler))
        return MubitClient(api_key=KEY_A, project_keys={"B": KEY_B}, projects=ROUTING, http_client=http)

    def sent(self) -> list[tuple[str | None, str | None]]:
        return [(q.headers.get(RECORD_HEADER), q.headers.get("x-adapter-id")) for q in self.requests]

    def keys(self) -> list[str | None]:
        out: list[str | None] = []
        for q in self.requests:
            auth = q.headers.get("authorization", "")
            out.append(q.headers.get("x-api-key") or (auth.removeprefix("Bearer ") if auth else None))
        return out


async def test_a_fallback_call_is_sent_record_off_without_adapter_headers() -> None:
    gw = Gateway([_reply(GOOD), _reply(GOOD)])
    c = gw.client()
    _, first = await c.call(StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.mubit, "s", "metadata")
    _, fb = await c.call(StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.mubit, "s", "metadata",
                         is_fallback=True, model=OPUS_MODEL)
    assert gw.sent() == [("metadata", "ad_lane_1"), ("off", None)]
    assert gw.keys() == [KEY_B, KEY_B]                  # the routed project's key, never another project's
    assert fb.is_fallback and not fb.recorded and fb.mubit_project == "B" and fb.valid
    await c.aclose()


async def test_a_full_recording_arm_also_falls_back_with_record_off() -> None:
    gw = Gateway([_reply(GOOD)])
    c = gw.client()
    await c.call(StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.mubit, "s", True, is_fallback=True)
    assert gw.sent() == [("off", None)]
    await c.aclose()


async def test_call_with_fallback_sends_the_third_attempt_to_the_provider() -> None:
    gw = Gateway([_reply(raw="not json"), _reply(raw="{}"), _reply(GOOD)])
    c = gw.client()
    out, recs = await call_with_fallback(c, StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.mubit, "s",
                                         "metadata")
    assert out is not None and [r.is_fallback for r in recs] == [False, False, True]
    assert gw.sent() == [("metadata", "ad_lane_1"), ("metadata", "ad_lane_1"), ("off", None)]
    await c.aclose()


async def test_the_researcher_policy_falls_back_record_off_and_says_so_in_the_manifest() -> None:
    gw = Gateway([_reply(raw="not json"), _reply(raw="not json"), _reply(GOOD)])
    c = gw.client()
    r = LLMResearcher(kind=ArmKind.mubit, llm=c, spec=SPEC, session_id="s", fallback_model=OPUS_MODEL,
                      record="metadata")
    parsed, recs = await r._call(StepType.judge_improvement, "u", JudgeOutput)
    assert parsed is not None and len(recs) == 3 and recs[-1].is_fallback
    assert gw.sent() == [("metadata", "ad_lane_1"), ("metadata", "ad_lane_1"), ("off", None)]
    assert llm_policy_of(r)["fallback_record"] == "off"
    mock = LLMResearcher(kind=ArmKind.opus, llm=MockLLM(spec=SPEC), spec=SPEC, session_id="s",
                         fallback_model=OPUS_MODEL)
    assert llm_policy_of(mock)["fallback_record"] is None
    await c.aclose()


def test_the_fingerprint_marks_the_rule_only_on_arms_that_record() -> None:
    import importlib.util
    import sys
    from pathlib import Path

    if "analyze_races" not in sys.modules:
        p = Path(__file__).resolve().parents[1] / "scripts" / "analyze_races.py"
        spec = importlib.util.spec_from_file_location("analyze_races", p)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["analyze_races"] = mod
        spec.loader.exec_module(mod)
    ar = sys.modules["analyze_races"]
    pol = {"researcher": "LLMResearcher", "max_retries": 1, "has_fallback": True, "fallback_model": OPUS_MODEL,
           "prompts": "judge_improvement=v1:5c1d3cfd2941"}

    def rows(fallback_record: str | None) -> dict[str, Any]:
        p = {**pol, **({"fallback_record": fallback_record} if fallback_record else {})}
        per = {"record_modes": {"opus": "off", "mubit": "metadata"}}
        return {"races": [{"id": "r", "fairness": {"per_arm": {"opus": per, "mubit": per},
                                                   "llm_policy": {"opus": p, "mubit": p}}}],
                "arms": [{"id": k, "kind": k, "model": OPUS_MODEL, "llm_mode": "mubit"} for k in ("opus", "mubit")],
                "llm_calls": [{"step_type": "judge_improvement"}], "experiments": []}

    old, new = ar.fairness_fingerprint(rows(None)), ar.fairness_fingerprint(rows("off"))
    assert "fallback_record" not in new["arms"]["opus"]["llm_policy"]          # record off arm: unchanged
    assert new["arms"]["mubit"]["llm_policy"]["fallback_record"] == "off"      # metadata arm: a new setup
    assert old["arms"]["opus"] == new["arms"]["opus"]
    assert ar.fingerprint_id(old) != ar.fingerprint_id(new)
