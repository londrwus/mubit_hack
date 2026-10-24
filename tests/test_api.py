"""API smoke test: create -> start -> WebSocket stream -> state -> export -> flag -> adapters.
Fakes come from test_engine (fake target, runner and LLM); no network, no GPU."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.agent.optuna_arm import OptunaResearcher
from backend.app import _construct, create_app
from backend.benchmark.configs import RaceConfig
from backend.core.events import parse_event
from backend.core.types import ArmKind
from backend.scheduler import ArmDeps
from test_engine import SPEC, FakeLLM, FakeRunner, llm_arm
from workers.targets.base import TargetSpec


def fake_factory(race_id: str, cfg: RaceConfig, spec: TargetSpec, kind: ArmKind) -> ArmDeps:
    runner = FakeRunner(cfg.gpus_per_arm, run_s=0.15)
    if kind == ArmKind.optuna:
        return ArmDeps(kind=kind, researcher=OptunaResearcher(SPEC, cfg.seed), runner=runner)
    latency = 0.3 if kind == ArmKind.opus else 0.02
    return llm_arm(kind, FakeLLM(latency, seed=cfg.seed), race_id, runner)


@pytest.fixture()
def client() -> Any:
    app = create_app(":memory:", arm_deps_factory=fake_factory, tick_s=0.02)
    with TestClient(app) as c:
        yield c


RACE = {"name": "api smoke", "target": "fake-quad", "arms": ["opus", "mubit", "optuna"], "budget_s": 1.5,
        "gpus_per_arm": 2, "seed": 0, "llm": "mock", "runner": "local"}


def test_full_race_over_http_and_websocket(client: TestClient) -> None:
    r = client.post("/api/races", json=RACE)
    assert r.status_code == 201, r.text
    race_id = r.json()["id"]
    assert client.get(f"/api/races/{race_id}").json()["race"]["status"] == "created"

    r = client.post(f"/api/races/{race_id}/start")
    assert r.status_code == 200, r.text
    assert client.post(f"/api/races/{race_id}/start").status_code == 409

    types: list[str] = []
    with client.websocket_connect(f"/ws/races/{race_id}") as ws:
        while True:
            ev = parse_event(ws.receive_text())
            assert ev.race_id == race_id
            types.append(ev.type)
            if ev.type == "race_finished":
                finished = ev
                break
    for t in ("gpu_state", "experiment_started", "metric", "experiment_finished", "llm_call", "new_best"):
        assert t in types, t
    assert {s.arm for s in finished.summaries} == {ArmKind.opus, ArmKind.mubit, ArmKind.optuna}

    state = client.get(f"/api/races/{race_id}").json()
    assert state["race"]["status"] == "finished"
    assert state["race"]["fairness"]["identical"] is True
    arms = {a["kind"]: a for a in state["arms"]}
    assert set(arms) == {"opus", "mubit", "optuna"}
    for a in arms.values():
        assert a["summary"]["experiments_finished"] >= 1
        assert a["stop_reason"] == "budget"
    assert arms["opus"]["summary"]["gpu_idle_pct"] > arms["mubit"]["summary"]["gpu_idle_pct"]
    assert arms["optuna"]["summary"]["llm_calls"] == 0

    r = client.get(f"/api/races/{race_id}/export.jsonl")
    assert r.status_code == 200
    lines = [json.loads(x) for x in r.text.splitlines() if x]
    tables = {x["table"] for x in lines}
    assert {"races", "arms", "experiments", "metrics", "llm_calls", "gpu_intervals"} <= tables
    assert lines[0]["table"] == "races"

    exp_id = next(e["id"] for e in state["experiments"] if e["status"] == "succeeded")
    r = client.post(f"/api/experiments/{exp_id}/flag", json={"reason": "eval overlaps train"})
    assert r.status_code == 200, r.text
    assert r.json()["experiment_id"] == exp_id
    corr = client.get(f"/api/races/{race_id}").json()["corrections"]
    assert [c["experiment_id"] for c in corr] == [exp_id]
    assert corr[0]["label"] == "fake_improvement" and corr[0]["source"] == "human"
    assert corr[0]["sent_to_mubit_at"] is None
    assert client.post("/api/experiments/nope/flag", json={}).status_code == 404

    # Late subscriber: replays the history and ends with race_finished.
    with client.websocket_connect(f"/ws/races/{race_id}") as ws:
        first = parse_event(ws.receive_text())
        assert first.type == "gpu_state"


def test_adapters_endpoint(client: TestClient) -> None:
    assert client.get("/api/adapters").json()["adapters"] == []
    r = client.post("/api/adapters", json={"step_type": "judge_improvement", "version": "v1", "status": "candidate",
                                           "notes": "toy exam 70%"})
    assert r.status_code == 201
    client.post("/api/adapters", json={"step_type": "judge_improvement", "version": "v1", "status": "shadow"})
    body = client.get("/api/adapters").json()
    assert len(body["adapters"]) == 1 and body["adapters"][0]["status"] == "shadow"
    assert set(body["by_step"]) == {"propose_experiments", "diagnose_run", "judge_improvement", "early_stop"}


def test_validation_and_404s(client: TestClient) -> None:
    bad = {**RACE, "mode": "benchmark"}  # benchmark + mock LLM is refused
    assert client.post("/api/races", json=bad).status_code == 422
    assert client.post("/api/races", json={**RACE, "target": "no-such-target"}).status_code == 400
    assert client.get("/api/races/missing").status_code == 404
    assert client.post("/api/races/missing/start").status_code == 404
    assert client.get("/api/races/missing/export.jsonl").status_code == 404
    assert client.get("/api/health").json()["ok"] is True


def test_construct_passes_only_accepted_kwargs() -> None:
    class A:
        def __init__(self, n_slots: int, hardware: str = "L4") -> None:
            self.n, self.hw = n_slots, hardware

    class B:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw

    a = _construct(A, n_slots=3, hardware="CPU", spec=None)
    assert (a.n, a.hw) == (3, "CPU")
    assert _construct(B, x=1).kw == {"x": 1}


def test_default_dev_wiring_builds_mock_llm_and_local_runner() -> None:
    from backend.agent.loop import LLMResearcher, RepoPrompts
    from backend.agent.mock_llm import MockLLM
    from backend.app import default_arm_deps
    from workers.runners.local import LocalRunner

    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.opus, ArmKind.optuna], budget_s=5, gpus_per_arm=3,
                     llm="mock", runner="local")
    d = default_arm_deps("r", cfg, SPEC, ArmKind.opus)
    assert isinstance(d.runner, LocalRunner) and len(d.runner.slots()) == 3
    assert isinstance(d.researcher, LLMResearcher) and isinstance(d.researcher.llm, MockLLM)
    assert isinstance(d.researcher.prompts, RepoPrompts)
    assert d.researcher.session_id == "race-r/opus/seed-0" and d.researcher.record is False
    assert d.llm_mode == "mock"
    o = default_arm_deps("r", cfg, SPEC, ArmKind.optuna)
    assert isinstance(o.researcher, OptunaResearcher) and o.llm_mode == "none"
