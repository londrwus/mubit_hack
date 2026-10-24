"""Per-step Mubit project routing (D38): data/mubit_projects.json, MubitClient key selection,
LLMCallRecord.mubit_project, the llm_calls column + migration and the manifest helper.

No network: a fake httpx2 transport records the auth header each request carried. The fake
keys below are test strings, not real keys.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx2
import pytest

from backend.agent.llm_base import OPUS_MODEL, RECORD_HEADER, REQUEST_ID_HEADER
from backend.agent.mubit_client import (
    DEFAULT_PROJECT_ROUTING,
    PROJECTS_CONFIG_PATH,
    MubitClient,
    ProjectRouting,
    call_with_fallback,
    load_mubit_api_key,
    load_project_routing,
    mubit_projects_manifest,
)
from backend.core.types import ArmKind, DiagnoseOutput, JudgeOutput, LLMCallRecord, StepType
from backend.db import Database

KEY_A, KEY_B = "test-key-project-a", "test-key-project-b"
JUDGE_REPLY = {"verdict": "valid", "reason": "Splits are distinct."}
DIAG_REPLY = {"failure_class": "divergence", "cause": "lr too high.", "fix": "lower lr."}


def _reply(payload: Any, req_id: str = "mreq_t", raw: str | None = None) -> httpx2.Response:
    body = {"id": "msg", "type": "message", "role": "assistant", "model": OPUS_MODEL,
            "content": [{"type": "text", "text": raw if raw is not None else json.dumps(payload)}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5}}
    return httpx2.Response(200, json=body, headers={REQUEST_ID_HEADER: req_id})


class Gateway:
    def __init__(self, responses: list[httpx2.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    def client(self, **kw: Any) -> MubitClient:
        kw.setdefault("api_key", KEY_A)
        kw.setdefault("project_keys", {"B": KEY_B})
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler))
        return MubitClient(http_client=http, **kw)

    def keys_sent(self) -> list[str | None]:
        """The credential each request carried (``x-api-key`` or ``Authorization: Bearer``)."""
        out: list[str | None] = []
        for r in self.requests:
            auth = r.headers.get("authorization", "")
            out.append(r.headers.get("x-api-key") or (auth.removeprefix("Bearer ") if auth else None))
        return out


# --------------------------------------------------------------------------- config


def test_repo_config_is_the_d38_rule() -> None:
    raw = json.loads(PROJECTS_CONFIG_PATH.read_text(encoding="utf-8"))
    assert raw == {"default": "A",
                   "routes": {"mubit": {"judge_improvement": "B", "early_stop": "B"}},
                   "keys": {"A": "MUBIT_API_KEY", "B": "MUBIT_API_KEY_B"}}
    r = load_project_routing()
    assert r.source == "data/mubit_projects.json"
    assert r.to_dict() | {"source": "built-in"} == DEFAULT_PROJECT_ROUTING.to_dict()


@pytest.mark.parametrize("arm", list(ArmKind))
@pytest.mark.parametrize("step", list(StepType))
def test_project_for_every_arm_and_step(arm: ArmKind, step: StepType) -> None:
    want = "B" if arm is ArmKind.mubit and step in (StepType.judge_improvement, StepType.early_stop) else "A"
    assert load_project_routing().project_for(arm, step) == want


def test_missing_default_file_falls_back_but_explicit_path_must_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.agent.mubit_client as mc

    monkeypatch.setattr(mc, "PROJECTS_CONFIG_PATH", tmp_path / "nope.json")
    assert mc.load_project_routing() is DEFAULT_PROJECT_ROUTING
    with pytest.raises(FileNotFoundError):
        mc.load_project_routing(tmp_path / "also-nope.json")


@pytest.mark.parametrize("bad", [
    {"default": "C", "keys": {"A": "MUBIT_API_KEY"}},                                  # default has no key
    {"default": "A", "keys": {"A": "mbt_live_abc123"}},                                # a key, not a var name
    {"default": "A", "keys": {"A": "MUBIT_API_KEY"}, "routes": {"mubitt": {"early_stop": "A"}}},  # arm typo
    {"default": "A", "keys": {"A": "MUBIT_API_KEY"}, "routes": {"mubit": {"judge": "A"}}},        # step typo
    {"default": "A", "keys": {"A": "MUBIT_API_KEY"}, "routes": {"mubit": {"early_stop": "B"}}},   # unknown project
    {"default": "A", "keys": {"A": "MUBIT_API_KEY"}, "route": {}},                     # unknown top-level key
])
def test_bad_configs_are_rejected(bad: dict[str, Any], tmp_path: Path) -> None:
    p = tmp_path / "p.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_project_routing(p)


def test_load_key_by_env_var_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("MUBIT_API_KEY=from-file-a\nMUBIT_API_KEY_B=from-file-b\nOTHER_SECRET=x\n", encoding="utf-8")
    monkeypatch.delenv("MUBIT_API_KEY", raising=False)
    monkeypatch.delenv("MUBIT_API_KEY_B", raising=False)
    monkeypatch.delenv("OTHER_SECRET", raising=False)
    assert load_mubit_api_key(env) == "from-file-a"
    assert load_mubit_api_key(env, env_var="MUBIT_API_KEY_B") == "from-file-b"
    monkeypatch.setenv("MUBIT_API_KEY_B", "from-env-b")
    assert load_mubit_api_key(env, env_var="MUBIT_API_KEY_B") == "from-env-b"
    import os

    assert "OTHER_SECRET" not in os.environ  # dotenv_values, not load_dotenv
    with pytest.raises(RuntimeError, match="MUBIT_API_KEY_C") as ei:
        load_mubit_api_key(env, env_var="MUBIT_API_KEY_C")
    assert "from-file" not in str(ei.value)


# --------------------------------------------------------------------------- client routing


async def test_mubit_judge_goes_to_b_everything_else_to_a() -> None:
    gw = Gateway([_reply(JUDGE_REPLY, "mreq_b1"), _reply(DIAG_REPLY, "mreq_a1"),
                  _reply(JUDGE_REPLY, "mreq_a2"), _reply(DIAG_REPLY, "mreq_a3")])
    c = gw.client()
    _, r1 = await c.call(StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.mubit, "s")
    _, r2 = await c.call(StepType.diagnose_run, "S", "u", DiagnoseOutput, ArmKind.mubit, "s")
    _, r3 = await c.call(StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.opus, "s")
    _, r4 = await c.call(StepType.diagnose_run, "S", "u", DiagnoseOutput, ArmKind.opus_bm, "s")
    assert gw.keys_sent() == [KEY_B, KEY_A, KEY_A, KEY_A]
    assert [r.mubit_project for r in (r1, r2, r3, r4)] == ["B", "A", "A", "A"]
    assert [r.mubit_request_id for r in (r1, r2, r3, r4)] == ["mreq_b1", "mreq_a1", "mreq_a2", "mreq_a3"]
    assert all(r.valid for r in (r1, r2, r3, r4))
    # Unchanged behaviour on both projects: Mubit URL, record-off header, metadata session.
    for q in gw.requests:
        assert str(q.url).startswith("https://api.mubit.ai/anthropic/")
        assert q.headers.get(RECORD_HEADER) == "off"
        assert json.loads(q.content)["metadata"] == {"user_id": "s"}
    await c.aclose()


async def test_record_mode_header_reaches_project_b() -> None:
    gw = Gateway([_reply(JUDGE_REPLY), _reply(JUDGE_REPLY)])
    c = gw.client()
    await c.call(StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.mubit, "s", "metadata")
    _, rec = await c.call(StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.mubit, "s", True)
    assert [q.headers.get(RECORD_HEADER) for q in gw.requests] == ["metadata", None]
    assert rec.recorded is True and rec.mubit_project == "B"
    await c.aclose()


async def test_retry_and_fallback_stay_on_the_routed_project() -> None:
    gw = Gateway([_reply(None, raw="not json"), _reply(None, raw="{}"), _reply(JUDGE_REPLY)])
    c = gw.client()
    out, recs = await call_with_fallback(c, StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.mubit, "s")
    assert out is not None and len(recs) == 3
    assert gw.keys_sent() == [KEY_B] * 3
    assert [r.mubit_project for r in recs] == ["B"] * 3 and recs[2].is_fallback
    await c.aclose()


async def test_missing_b_key_fails_the_call_and_never_falls_back_to_a(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import backend.agent.mubit_client as mc

    monkeypatch.delenv("MUBIT_API_KEY_B", raising=False)
    monkeypatch.setattr(mc, "REPO_ROOT", tmp_path)   # no .env there
    gw = Gateway([_reply(DIAG_REPLY)])
    c = gw.client(project_keys=None)                 # A from api_key, B lazily from env/.env
    parsed, rec = await c.call(StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.mubit, "s")
    assert parsed is None and not rec.valid and rec.mubit_project == "B"
    assert "MUBIT_API_KEY_B" in (rec.error or "") and KEY_A not in (rec.error or "")
    assert gw.requests == []                          # nothing sent anywhere
    _, ok = await c.call(StepType.diagnose_run, "S", "u", DiagnoseOutput, ArmKind.mubit, "s")
    assert ok.valid and ok.mubit_project == "A" and gw.keys_sent() == [KEY_A]
    await c.aclose()


async def test_b_key_equal_to_a_key_is_refused() -> None:
    gw = Gateway([])
    c = gw.client(project_keys={"B": KEY_A})
    parsed, rec = await c.call(StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.mubit, "s")
    assert parsed is None and "equals project A" in (rec.error or "") and gw.requests == []
    assert KEY_A not in (rec.error or "")
    await c.aclose()


async def test_b_key_loaded_lazily_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUBIT_API_KEY_B", "lazy-b-key")
    gw = Gateway([_reply(DIAG_REPLY), _reply(JUDGE_REPLY)])
    c = gw.client(project_keys=None)
    assert set(c._clients) == {"A"}                   # B not built until first routed call
    await c.call(StepType.diagnose_run, "S", "u", DiagnoseOutput, ArmKind.mubit, "s")
    await c.call(StepType.judge_improvement, "S", "u", JudgeOutput, ArmKind.mubit, "s")
    assert gw.keys_sent() == [KEY_A, "lazy-b-key"] and set(c._clients) == {"A", "B"}
    await c.aclose()


def test_base_url_guard_unchanged() -> None:
    with pytest.raises(ValueError):
        MubitClient(api_key=KEY_A, project_keys={"B": KEY_B}, base_url="https://api.anthropic.com")


async def test_custom_routing_object() -> None:
    routing = ProjectRouting(default="A", routes={"opus": {"propose_experiments": "B"}},
                             keys={"A": "MUBIT_API_KEY", "B": "MUBIT_API_KEY_B"}, source="test")
    gw = Gateway([_reply(DIAG_REPLY)])
    c = gw.client(projects=routing)
    _, rec = await c.call(StepType.diagnose_run, "S", "u", DiagnoseOutput, ArmKind.opus, "s")
    assert rec.mubit_project == "A" and c.project_for(ArmKind.opus, StepType.propose_experiments) == "B"
    await c.aclose()


# --------------------------------------------------------------------------- record + db


def test_record_field_is_optional() -> None:
    rec = LLMCallRecord(arm=ArmKind.opus, session_id="s", step_type=StepType.diagnose_run, model="m")
    assert rec.mubit_project is None
    assert LLMCallRecord.model_validate(rec.model_dump() | {"mubit_project": "B"}).mubit_project == "B"


def test_db_stores_project_and_migrates_old_files(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    # An llm_calls table from schema 1 (no mubit_project column), with one existing row.
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE llm_calls (id TEXT PRIMARY KEY, arm_id TEXT, race_id TEXT, arm TEXT,
        session_id TEXT, step_type TEXT, mubit_request_id TEXT, model TEXT, latency_ms REAL,
        input_tokens INTEGER, cache_creation_tokens INTEGER, cache_read_tokens INTEGER,
        output_tokens INTEGER, cost_usd REAL, valid INTEGER, is_retry INTEGER, is_fallback INTEGER,
        is_replay INTEGER, recorded INTEGER, stop_reason TEXT, error TEXT, experiment_ids_json TEXT,
        created_at REAL, completed_at REAL)""")
    con.execute("INSERT INTO llm_calls (id, race_id, arm, step_type, model) VALUES ('old', 'r', 'opus', 'diagnose_run', 'm')")
    con.commit()
    con.close()

    db = Database(path)
    db.insert_race(race_id="r", target="toy", budget_s=1, gpus_per_arm=1, seed=0, config={})
    rec = LLMCallRecord(arm=ArmKind.mubit, session_id="s", step_type=StepType.judge_improvement,
                        model=OPUS_MODEL, mubit_project="B", cost_source="estimate@opus_list")
    db.insert_llm_call(rec, race_id="r")
    rows = {r["id"]: r for r in db.list_llm_calls(race_id="r")}
    assert rows["old"]["mubit_project"] is None and rows["old"]["cost_source"] is None
    assert rows[rec.id]["mubit_project"] == "B" and rows[rec.id]["cost_source"] == "estimate@opus_list"
    db.close()
    Database(path).close()  # re-opening a migrated file is a no-op


# --------------------------------------------------------------------------- manifest


async def test_manifest_helper() -> None:
    gw = Gateway([])
    opus, mubit = gw.client(), gw.client()
    m = mubit_projects_manifest({ArmKind.opus: opus, ArmKind.mubit: mubit, ArmKind.optuna: None,
                                 "mock": object()})
    assert m["per_arm"]["opus"] == {s.value: "A" for s in StepType}
    assert m["per_arm"]["mubit"] == {"propose_experiments": "A", "diagnose_run": "A",
                                     "judge_improvement": "B", "early_stop": "B"}
    assert m["per_arm"]["optuna"] is None and m["per_arm"]["mock"] is None
    assert m["config_identical"] and m["config"]["keys"] == {"A": "MUBIT_API_KEY", "B": "MUBIT_API_KEY_B"}
    blob = json.dumps(m)
    assert KEY_A not in blob and KEY_B not in blob     # env var names only, never keys
    await opus.aclose()
    await mubit.aclose()
