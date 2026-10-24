"""Tests for scripts/eval_trap_set.py (no network: fake httpx2 transport on the Mubit URL)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx2
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_trap_set as ets  # noqa: E402

from backend.agent.llm_base import OPUS_MODEL, RECORD_HEADER, REQUEST_ID_HEADER  # noqa: E402
from backend.agent.mubit_client import MubitClient  # noqa: E402
from backend.agent.prompts import load_system_prompt  # noqa: E402
from backend.agent.prompts.build import CACHE_BREAK, judge_wire_schema, split_user  # noqa: E402
from backend.core.types import StepType  # noqa: E402

CASES = ets.load_cases(ets.DEFAULT_SET, "all")


def _reply(verdict: str | None, req_id: str, *, raw: str | None = None, out_tokens: int = 100) -> httpx2.Response:
    text = raw if raw is not None else json.dumps({"verdict": verdict, "reason": "because"})
    body = {
        "id": "msg_t", "type": "message", "role": "assistant", "model": OPUS_MODEL,
        "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 1000, "output_tokens": out_tokens,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }
    return httpx2.Response(200, json=body, headers={REQUEST_ID_HEADER: req_id})


class Gateway:
    """Answers each request via ``answer(user_text) -> Response``; records requests."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.requests: list[httpx2.Request] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        user = "".join(b["text"] for b in body["messages"][0]["content"])
        return self.answer(user, len(self.requests))

    def client(self) -> MubitClient:
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler))
        return MubitClient(api_key="test-mubit-key", http_client=http)

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


def _case_by_user(user: str) -> dict[str, Any]:
    for c in CASES:
        prefix, rest = split_user(c["user_message"])
        if (prefix or "") + rest == user:
            return c
    raise AssertionError("unknown user message")


def oracle(user: str, i: int) -> httpx2.Response:
    c = _case_by_user(user)
    return _reply(c["acceptable_verdicts"][0], f"mreq_{c['id']}")


# --------------------------------------------------------------------------- set + guards


def test_set_matches_current_prompt_and_builder() -> None:
    assert len(CASES) == 50
    assert ets.case_problems(CASES) == []


def test_stale_case_is_flagged() -> None:
    bad = dict(CASES[0], prompt={"step": "judge_improvement", "version": "v0", "fingerprint": "deadbeef0000"})
    bad["user_message"] = bad["user_message"].replace(CACHE_BREAK, "\n")
    probs = ets.case_problems([bad])
    assert any("fingerprint" in p or "built for prompt" in p for p in probs)
    assert any("CACHE_BREAK" in p for p in probs)
    assert any("differs" in p for p in probs)


def test_split_filter() -> None:
    assert len(ets.load_cases(ets.DEFAULT_SET, "dev")) == 30
    assert len(ets.load_cases(ets.DEFAULT_SET, "test")) == 20


def test_record_guard() -> None:
    assert ets.record_problems("off", {"test"}) == []
    assert ets.record_problems("full", {"dev"}) == []  # judge prompt is frozen (D27)
    assert any("test split" in p for p in ets.record_problems("full", {"test"}))
    assert any("test split" in p for p in ets.record_problems("full", {"dev", "test"}))


def test_cli_refuses_recording_test_split() -> None:
    assert ets.main(["--split", "test", "--record", "full"]) == 2


def test_session_id() -> None:
    assert ets.session_id("2026-10-24", "claude-opus-5-5", "dev") == "trapeval-2026-10-24/claude-opus-5-5/dev"


def test_grade() -> None:
    fake = {"label": "invalid", "acceptable_verdicts": ["invalid", "unsure"]}
    real = {"label": "valid", "acceptable_verdicts": ["valid"]}
    assert ets.grade(fake, "valid") == {"correct": False, "fake_approved": True, "genuine_rejected": False,
                                        "unsure": False, "failed": False}
    assert ets.grade(fake, "unsure")["correct"]
    assert ets.grade(real, "invalid")["genuine_rejected"] and not ets.grade(real, "invalid")["correct"]
    assert not ets.grade(real, "unsure")["correct"]
    assert ets.grade(real, None) == {"correct": False, "fake_approved": False, "genuine_rejected": False,
                                     "unsure": False, "failed": True}


# --------------------------------------------------------------------------- eval loop


async def test_request_layout_matches_race_loop() -> None:
    gw = Gateway(oracle)
    cases = [c for c in CASES if c["split"] == "dev"][:3]
    res = await ets.eval_cases(cases, gw.client(), model=OPUS_MODEL, record="off", usd_cap=5,
                               concurrency=2, date="2026-10-24")
    assert [r["id"] for r in res] == [c["id"] for c in cases]
    system = load_system_prompt(StepType.judge_improvement)
    for req, body, c in zip(gw.requests, gw.bodies(), cases, strict=False):
        assert str(req.url).startswith("https://api.mubit.ai/anthropic")
        assert req.headers[RECORD_HEADER] == "off"
        assert body["model"] == OPUS_MODEL
        assert body["system"][0]["text"] == system
        assert body["output_config"]["effort"] == "low"
        assert body["output_config"]["format"]["schema"] == judge_wire_schema()
        assert body["metadata"] == {"user_id": "trapeval-2026-10-24/claude-opus-5-5/dev"}
        content = body["messages"][0]["content"]
        assert len(content) == 2 and "cache_control" in content[0]  # target-spec prefix is cached
    assert all(r["correct"] for r in res)
    assert res[0]["attempts"][0]["mubit_request_id"].startswith("mreq_")


async def test_record_full_sends_no_off_header() -> None:
    gw = Gateway(oracle)
    case = next(c for c in CASES if c["split"] == "dev")
    res = await ets.eval_cases([case], gw.client(), model=OPUS_MODEL, record="full", usd_cap=5)
    assert RECORD_HEADER not in gw.requests[0].headers
    assert res[0]["attempts"][0]["recorded"] is True


async def test_retry_once_then_fail_no_fallback() -> None:
    gw = Gateway(lambda u, i: _reply(None, f"mreq_{i}", raw="not json"))
    case = CASES[0]
    res = await ets.eval_cases([case], gw.client(), model="claude-haiku-5-5", usd_cap=5, retries=1)
    assert len(gw.requests) == 2
    assert all(b["model"] == "claude-haiku-5-5" for b in gw.bodies())  # never falls back to Opus
    r = res[0]
    assert r["verdict"] is None and r["failed"] and not r["correct"] and r["graded"]
    assert [a["is_retry"] for a in r["attempts"]] == [False, True]


async def test_retry_recovers() -> None:
    gw = Gateway(lambda u, i: _reply(None, "x", raw="{") if i == 1 else oracle(u, i))
    res = await ets.eval_cases([CASES[0]], gw.client(), model=OPUS_MODEL, usd_cap=5)
    assert res[0]["correct"] and len(res[0]["attempts"]) == 2


async def test_usd_cap_stops_calls() -> None:
    gw = Gateway(oracle)  # each reply: 1000 in + 100 out on Opus = $0.006
    res = await ets.eval_cases(CASES[:10], gw.client(), model=OPUS_MODEL, usd_cap=0.05, concurrency=1)
    sent = len(gw.requests)
    assert 0 < sent < 10
    spent = sum(r["cost_usd"] for r in res)
    assert spent <= 0.05
    skipped = [r for r in res if not r["graded"]]
    assert len(skipped) == 10 - sent and all(r["skipped"] == "usd_cap" for r in skipped)
    s = ets.summarize(res)
    assert s["dev" if CASES[0]["split"] == "dev" else CASES[0]["split"]]["n_skipped_cap"] >= 0


async def test_summary_counts() -> None:
    # Answer 'valid' to everything: every fake approved, every genuine accepted.
    gw = Gateway(lambda u, i: _reply("valid", f"mreq_{i}"))
    res = await ets.eval_cases(CASES, gw.client(), model=OPUS_MODEL, usd_cap=5, concurrency=8)
    s = ets.summarize(res)
    assert set(s) == {"dev", "test", "all"}
    for split, n_fake, n_real in (("dev", 15, 15), ("test", 10, 10), ("all", 25, 25)):
        b = s[split]
        assert b["fakes"] == n_fake and b["genuine"] == n_real
        assert b["fake_approved"] == n_fake and b["fake_approved_rate"] == 1.0
        assert b["genuine_rejected"] == 0 and b["genuine_accepted"] == n_real
        assert b["accuracy"] == 0.5 and b["unsure_rate"] == 0.0
        assert b["p50_latency_ms"] is not None and b["cost_usd"] > 0
    cats = s["all"]["by_category"]
    assert cats["clean_gain"]["accuracy"] == 1.0 and cats["tune_on_test"]["accuracy"] == 0.0
    assert sum(c["n"] for c in cats.values()) == 50
    report = ets.format_report(s, res)
    assert "fake approved" in report and "mreq_" in report


@pytest.mark.parametrize("q,expect", [(0.5, 2.0), (0.9, 2.8)])
def test_quantile(q: float, expect: float) -> None:
    assert ets._quantile([1.0, 2.0, 3.0], q) == expect
