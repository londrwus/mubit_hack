"""Race replay (demo fallback): rebuild the WebSocket stream of a finished race from storage.

1. A synthetic race written with backend/db.py functions: every event checked by hand.
2. Fidelity: a fake in-process race (fakes from test_engine), live bus events vs the rebuild.
3. API: POST /api/races/{id}/replay, WS /ws/replays/{id}, GET /api/races?replayable=true.
4. Real data, read-only: any finished race in data/bench/*.db or data/bench/dryrun/*.jsonl.
No network, no GPU, no LLM.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from test_api import RACE, fake_factory
from test_engine import SPEC, FakeLLM, FakeRunner, llm_arm

from backend.app import create_app
from backend.benchmark.configs import RaceConfig
from backend.benchmark.export import export_race
from backend.core.events import parse_event
from backend.core.types import ArmKind, ArmSummary, CorrectionLabel, LLMCallRecord, StepType, new_id
from backend.db import Database
from backend.replay import (
    ReadOnlyDB,
    ReplayCatalog,
    ReplayError,
    ReplayStore,
    check_stream,
    main,
    play,
    read_jsonl_rows,
    rebuild_events,
    schedule,
    wire_payload,
)
from backend.scheduler import EventBus, RaceEngine

REPO = Path(__file__).resolve().parents[1]
T0 = 1_000_000.0
P = 0.001  # $/s


# =========================================================================== 1. synthetic race


def _synthetic_db(path: str | Path = ":memory:") -> tuple[Database, str]:
    """Two arms, one slot each, on an unknown target (direction inferred: max).

    opus:   baseline e1 (0-10, 0.50, best) -> judge -> e2 (12-20, 0.70, best, later flagged fake
            at 25 -> best falls back to e1) -> e3 (22-30, 0.60, best again) -> e4 running at the
            end (30-40, cancelled "stopped at end of budget").
    optuna: f1 (0-8, failed) -> f2 (8-16 trap, 0.90, best, no judge) -> slot released at 30.
    """
    db = Database(path)
    rid = new_id()
    cfg = {"target": "synthetic", "arms": ["opus", "optuna"], "seed": 7, "target_value": 0.65}
    db.insert_race(race_id=rid, target="synthetic", budget_s=40, gpus_per_arm=1, seed=7, config=cfg,
                   name="synthetic", mode="dev", llm_mode="mock", runner="local")
    db.insert_arm(arm_id="A", race_id=rid, kind="opus", model="m", llm_mode="mock", n_slots=1, hardware="L4",
                  price_per_s=P)
    db.insert_arm(arm_id="B", race_id=rid, kind="optuna", model=None, llm_mode="none", n_slots=1, hardware="L4",
                  price_per_s=P)

    def exp(i: str, arm: str, s: float, e: float | None, status: str, score: float | None, **kw: Any) -> None:
        db.upsert_experiment({"id": i, "arm_id": arm, "race_id": rid, "slot_id": f"{arm}-0",
                              "config": {"lr": s}, "hypothesis": f"h-{i}", "status": status, "seed": 7,
                              "hardware": "L4", "proposed_at": T0 + s - 0.5, "started_at": T0 + s,
                              "ended_at": None if e is None else T0 + e, "score": score,
                              "gpu_cost_usd": 0.0 if e is None else (e - s) * P, **kw})

    exp("e1", "A", 0, 10, "succeeded", 0.50, is_best=True, is_baseline=True, summary={"claimed_new_best": True})
    exp("e2", "A", 12, 20, "succeeded", 0.70, is_best=False, judge_verdict="valid",
        summary={"claimed_new_best": True})  # is_best reset by the correction
    exp("e3", "A", 22, 30, "succeeded", 0.60, is_best=True, judge_verdict="valid", summary={"claimed_new_best": True})
    exp("e4", "A", 30, 40, "cancelled", None, error="stopped at end of budget")
    db.upsert_experiment({"id": "e5", "arm_id": "A", "race_id": rid, "status": "cancelled", "seed": 7,
                          "proposed_at": T0 + 35, "error": "deadline rule: not launched"})
    exp("f1", "B", 0, 8, "failed", None, error="loss is NaN")
    exp("f2", "B", 8, 16, "succeeded", 0.90, is_best=True, is_trap=True, metrics={"heldout_score": 0.40})

    for e_id, step, t, v in [("e1", 0, 2.0, 1.5), ("e1", 1, 99.0, 1.2), ("f2", 0, 3.0, 0.9)]:
        db.add_metric(e_id, step, t, "train_loss", v)

    def call(step: StepType, start: float, end: float, exps: list[str]) -> None:
        rec = LLMCallRecord(arm=ArmKind.opus, arm_id="A", session_id="s", step_type=step, model="m",
                            latency_ms=(end - start) * 1000, cost_usd=0.01, valid=True, experiment_ids=exps,
                            created_at=T0 + start)
        db.insert_llm_call(rec, race_id=rid, completed_at=T0 + end)

    call(StepType.propose_experiments, 9.0, 11.5, ["e2"])       # in flight when e1 ends -> new best at 11.5
    call(StepType.judge_improvement, 20.0, 21.0, ["e2"])        # e2 announced after its judge call
    call(StepType.judge_improvement, 30.0, 30.5, ["e3"])

    for arm, ivs in {"A": [("idle", 0, 0, None), ("busy", 0, 10, "e1"), ("idle", 10, 12, None),
                           ("busy", 12, 20, "e2"), ("busy", 20, 30, "e3"), ("busy", 30, 40, "e4")],
                     "B": [("busy", 0, 8, "f1"), ("busy", 8, 16, "f2"), ("idle", 16, 30, None)]}.items():
        for state, s, e, x in ivs:
            if e > s:
                db.insert_gpu_interval(arm_id=arm, race_id=rid, slot=f"{arm}-0", hardware="L4", state=state,
                                       start=T0 + s, end=T0 + e, experiment_id=x, price_per_s=P)

    c1 = db.insert_correction(experiment_id="e2", source="human", label="fake_improvement", reason="leak",
                              race_id=rid, arm_id="A")
    c2 = db.insert_correction(experiment_id="e3", source="sapling", label="genuine", race_id=rid, arm_id="A")
    db._exec("UPDATE corrections SET created_at = ? WHERE id = ?", (T0 + 25, c1))
    db._exec("UPDATE corrections SET created_at = ? WHERE id = ?", (T0 + 99, c2))  # after the race
    db.update_race(rid, status="finished", finish_reason="budget", started_at=T0, ended_at=T0 + 40)
    for arm, kind in (("A", ArmKind.opus), ("B", ArmKind.optuna)):
        db.update_arm(arm, status="finished", started_at=T0, ended_at=T0 + (40 if arm == "A" else 30),
                      summary=ArmSummary(arm=kind, best_score=0.6).model_dump(mode="json"))
    db.flush_metrics()
    return db, rid


@pytest.fixture()
def synthetic() -> tuple[Database, str]:
    return _synthetic_db()


def test_synthetic_race_rebuilds_every_event(synthetic: tuple[Database, str]) -> None:
    db, rid = synthetic
    tl = rebuild_events(db.race_rows(rid))
    ev = tl.events
    assert check_stream(ev) == []
    assert tl.t0 == T0 and tl.direction == "max" and tl.target_value == 0.65
    assert all(e.race_id == rid for e in ev)
    assert ev[-1].type == "race_finished" and ev[-1].ts == T0 + 40 and ev[-1].reason == "budget"
    assert [s.arm for s in ev[-1].summaries] == [ArmKind.opus, ArmKind.optuna]  # config arm order

    def rel(e: Any) -> float:
        return round(e.ts - T0, 6)

    # gpu_state: idle at t0, each interval start, zero-length idle between back-to-back runs, closing idle.
    gpu: dict[str, list[tuple[float, str, str | None]]] = defaultdict(list)
    for e in ev:
        if e.type == "gpu_state":
            gpu[e.slot_id].append((rel(e), e.state.value, e.experiment_id))
    assert gpu["A-0"] == [(0, "idle", None), (0, "busy", "e1"), (10, "idle", None), (12, "busy", "e2"),
                          (20, "idle", None), (20, "busy", "e3"), (30, "idle", None), (30, "busy", "e4"),
                          (40, "idle", None)]
    assert gpu["B-0"] == [(0, "idle", None), (0, "busy", "f1"), (8, "idle", None), (8, "busy", "f2"),
                          (16, "idle", None), (30, "idle", None)]

    started = [e.experiment_id for e in ev if e.type == "experiment_started"]
    assert sorted(started) == ["e1", "e2", "e3", "e4", "f1", "f2"]          # e5 never launched
    fin = {e.experiment_id: e for e in ev if e.type == "experiment_finished"}
    assert set(fin) == {"e1", "e2", "e3", "f1", "f2"}                       # e4 ended after the budget
    assert fin["f2"].is_trap and fin["f2"].score == 0.90 and fin["f2"].duration_s == 8
    assert fin["f1"].status.value == "failed" and fin["f1"].error == "loss is NaN"
    assert fin["e2"].gpu_cost_usd == pytest.approx(8 * P)

    metrics = [(e.experiment_id, e.step, rel(e)) for e in ev if e.type == "metric"]
    assert metrics == [("e1", 0, 2.0), ("e1", 1, 10.0), ("f2", 0, 11.0)]  # t=99 clamped to the run end

    bests = [(e.arm.value, e.experiment_id, rel(e), e.previous_best, e.reached_target)
             for e in ev if e.type == "new_best"]
    assert bests == [
        ("opus", "e1", 11.5, None, False),     # e1 ended at 10 while a propose call ran until 11.5
        ("optuna", "f2", 16, None, False),     # a trap never reaches the goal (held-out 0.40 anyway)
        ("opus", "e2", 21.0, 0.50, True),      # after its judge call; later flagged fake
        ("opus", "e3", 30.5, 0.50, False),     # previous best is e1 again after the correction at 25
    ]
    corr = [e for e in ev if e.type == "correction"]
    assert len(corr) == 1 and corr[0].experiment_id == "e2" and corr[0].arm == ArmKind.opus and rel(corr[0]) == 25
    assert tl.dropped == Counter(correction_after_race=1)
    assert sum(e.type == "llm_call" for e in ev) == 3


def test_rebuild_is_read_only_and_matches_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    db, rid = _synthetic_db(path)
    before = json.dumps(db.race_rows(rid), sort_keys=True, default=str)
    export = export_race(db, rid, tmp_path / "race.jsonl")
    db.close()
    with ReadOnlyDB(path) as ro:
        from_db = rebuild_events(ro.race_rows(rid))
        with pytest.raises(sqlite3.OperationalError):
            ro._conn.execute("CREATE TABLE x (a)")
    from_jsonl = rebuild_events(read_jsonl_rows(export))
    assert [e.model_dump(mode="json") for e in from_db.events] == [e.model_dump(mode="json") for e in from_jsonl.events]
    db2 = Database(path)
    assert json.dumps(db2.race_rows(rid), sort_keys=True, default=str) == before
    db2.close()


def test_unfinished_race_is_refused() -> None:
    db = Database()
    rid = new_id()
    db.insert_race(race_id=rid, target="x", budget_s=10, gpus_per_arm=1, seed=0, config={})
    with pytest.raises(ReplayError):
        rebuild_events(db.race_rows(rid))
    db.update_race(rid, status="running", started_at=T0)
    with pytest.raises(ReplayError):
        rebuild_events(db.race_rows(rid))


def test_schedule_speed_and_gap_cap(synthetic: tuple[Database, str]) -> None:
    db, rid = synthetic
    ev = rebuild_events(db.race_rows(rid)).events
    offs = schedule(ev, 10.0)
    assert offs[0] == 0 and offs[-1] == pytest.approx(4.0)                  # 40 s race at 10x
    assert all(b >= a for a, b in zip(offs, offs[1:]))
    capped = schedule(ev, 1.0, max_gap_s=0.5)
    assert all(b - a <= 0.5 + 1e-9 for a, b in zip(capped, capped[1:]))
    with pytest.raises(ValueError):
        schedule(ev, 0)


async def test_play_paces_events_and_flags_them(synthetic: tuple[Database, str]) -> None:
    db, rid = synthetic
    tl = rebuild_events(db.race_rows(rid))
    offs = schedule(tl.events, 4.0)
    clock = [0.0]
    sent: list[tuple[float, dict[str, Any]]] = []

    async def fake_sleep(s: float) -> None:
        clock[0] += s

    async def send(text: str) -> None:
        sent.append((clock[0], json.loads(text)))

    n = await play(tl.events, offs, send, replay_id="r1", t0=tl.t0, clock=lambda: clock[0], sleep=fake_sleep)
    assert n == len(tl.events) == len(sent)
    for (at, msg), off, e in zip(sent, offs, tl.events):
        assert at == pytest.approx(off, abs=0.003)
        assert msg["is_replay"] is True and msg["replay_id"] == "r1"
        assert msg["race_elapsed_s"] == pytest.approx(e.ts - T0)
        assert parse_event(msg) == e                                        # extra keys ignored
    # The flag never leaks into the event model or the stored LLM record.
    call_ev = next(e for e in tl.events if e.type == "llm_call")
    assert wire_payload(call_ev, replay_id="r", t0=T0)["call"]["is_replay"] is False
    assert "is_replay" not in call_ev.model_dump()


# =========================================================================== 2. fidelity vs a live race


def _collapse(seq: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


async def _live_race(db: Database) -> tuple[str, list[Any]]:
    # The mubit arm stands for "the same loop with a faster planner" here: backfill (the BM arms'
    # default) off, so the replay heuristics are checked on the plain engine.
    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.opus, ArmKind.mubit], budget_s=2.0, gpus_per_arm=2,
                     seed=0, llm="mock", engine_features={ArmKind.mubit: ["judge_policy"]})
    rid = new_id()
    bus = EventBus()
    live: list[Any] = []
    bus.add_listener(live.append)
    deps = [llm_arm(ArmKind.opus, FakeLLM(0.3, seed=1), rid, FakeRunner(2, run_s=0.2, report_heldout=True)),
            llm_arm(ArmKind.mubit, FakeLLM(0.02, seed=1, trap_rate=0.0), rid,
                    FakeRunner(2, run_s=0.2, report_heldout=True))]
    eng = RaceEngine(cfg, deps, spec=SPEC, race_id=rid, db=db, bus=bus, tick_s=0.02)
    task = asyncio.create_task(eng.run())
    while not (eng.arms and eng.arms[ArmKind.mubit].best_id and len(eng.arms[ArmKind.mubit].processed) >= 3):
        await asyncio.sleep(0.02)
    victim = eng.arms[ArmKind.mubit].best_id
    exp = db.get_experiment(victim)
    cid = db.insert_correction(experiment_id=victim, source="human", label="fake_improvement", reason="test",
                               race_id=rid, arm_id=exp["arm_id"])            # same order as the API route
    eng.apply_correction(victim, CorrectionLabel.fake_improvement, correction_id=cid, reason="test")
    await task
    return rid, live


def test_rebuild_matches_the_live_stream() -> None:
    db = Database()
    rid, live = asyncio.run(_live_race(db))
    tl = rebuild_events(db.race_rows(rid))
    rebuilt = tl.events
    assert check_stream(rebuilt) == []
    def non_gpu(evs: list[Any]) -> Counter[str]:
        return Counter(e.type for e in evs if e.type != "gpu_state")   # gpu_state compared per slot below

    assert non_gpu(rebuilt) == non_gpu(live)

    def exact(t: str) -> list[dict[str, Any]]:
        return sorted((e.model_dump(mode="json") for e in live if e.type == t), key=lambda d: json.dumps(d, sort_keys=True))

    def rb(t: str) -> list[dict[str, Any]]:
        return sorted((e.model_dump(mode="json") for e in rebuilt if e.type == t),
                      key=lambda d: json.dumps(d, sort_keys=True))

    for t in ("experiment_started", "experiment_finished", "llm_call"):
        assert rb(t) == exact(t), t

    def slots(evs: list[Any]) -> dict[tuple[str, str], list[tuple[Any, ...]]]:
        out: dict[tuple[str, str], list[tuple[Any, ...]]] = defaultdict(list)
        for e in evs:
            if e.type == "gpu_state":
                out[(e.arm.value, e.slot_id)].append((e.state.value, e.experiment_id, e.ts))
        return {k: _collapse(v) for k, v in out.items()}

    assert slots(rebuilt) == slots(live)

    def metric_key(evs: list[Any]) -> Counter[tuple[Any, ...]]:
        return Counter((e.experiment_id, e.step, e.name, e.value, e.t) for e in evs if e.type == "metric")

    assert metric_key(rebuilt) == metric_key(live)

    live_best = [e for e in live if e.type == "new_best"]
    rb_best = [e for e in rebuilt if e.type == "new_best"]
    assert len(live_best) >= 2
    assert ({e.experiment_id: e.model_dump(exclude={"ts"}) for e in rb_best}
            == {e.experiment_id: e.model_dump(exclude={"ts"}) for e in live_best})
    lb = {e.experiment_id: e.ts for e in live_best}
    for e in rb_best:
        assert e.ts <= lb[e.experiment_id] + 1e-6 and lb[e.experiment_id] - e.ts < 0.5

    lc = [e.model_dump(exclude={"ts"}) for e in live if e.type == "correction"]
    assert lc and [e.model_dump(exclude={"ts"}) for e in rebuilt if e.type == "correction"] == lc

    lf = next(e for e in live if e.type == "race_finished")
    assert rebuilt[-1].model_dump(exclude={"ts"}) == lf.model_dump(exclude={"ts"})


# =========================================================================== 3. API


@pytest.fixture()
def client_with_sources(tmp_path: Path) -> Any:
    sdb, srid = _synthetic_db(tmp_path / "other.db")
    export_race(sdb, srid, tmp_path / "exports" / "race.jsonl")
    sdb.close()
    app = create_app(":memory:", arm_deps_factory=fake_factory, tick_s=0.02,
                     replay_sources=[str(tmp_path / "other.db"), str(tmp_path / "exports")])
    with TestClient(app) as c:
        yield c, srid


def _run_race(c: TestClient) -> str:
    rid = c.post("/api/races", json={**RACE, "budget_s": 1.0}).json()["id"]
    assert c.post(f"/api/races/{rid}/replay").status_code == 409          # created, not finished
    assert c.post(f"/api/races/{rid}/start").status_code == 200
    with c.websocket_connect(f"/ws/races/{rid}") as ws:
        while parse_event(ws.receive_text()).type != "race_finished":
            pass
    return rid


def test_api_replays_a_finished_race(client_with_sources: Any) -> None:
    c, srid = client_with_sources
    rid = _run_race(c)

    listed = c.get("/api/races", params={"replayable": True}).json()
    by_id = {r["id"]: r for r in listed}
    assert by_id[rid]["source"] == "app" and by_id[rid]["replayable"] is True
    assert srid in by_id                                                    # from the read-only source
    assert len(c.get("/api/races").json()) == 1                             # default listing unchanged

    assert c.post("/api/races/nope/replay").status_code == 404
    assert c.post(f"/api/races/{rid}/replay", params={"speed": 0}).status_code == 422

    r = c.post(f"/api/races/{rid}/replay", params={"speed": 50})
    assert r.status_code == 201, r.text
    info = r.json()
    assert info["race_id"] == rid and info["speed"] == 50 and info["is_replay"] is True
    assert info["replay_duration_s"] == pytest.approx(info["race_duration_s"] / 50, abs=2e-3)  # both rounded to ms
    assert c.get(f"/api/replays/{info['replay_id']}").json()["n_events"] == info["n_events"]
    assert [x["replay_id"] for x in c.get("/api/replays").json()] == [info["replay_id"]]

    msgs: list[dict[str, Any]] = []
    with c.websocket_connect(info["ws"]) as ws:
        while True:
            msgs.append(json.loads(ws.receive_text()))
            if msgs[-1]["type"] == "race_finished":
                break
    assert len(msgs) == info["n_events"]
    assert all(m["is_replay"] is True and m["replay_id"] == info["replay_id"] and m["race_id"] == rid for m in msgs)
    evs = [parse_event(m) for m in msgs]
    assert check_stream(evs) == []
    assert Counter(e.type for e in evs) == Counter(info["counts"])
    for t in ("gpu_state", "experiment_started", "metric", "experiment_finished", "llm_call", "new_best"):
        assert t in info["counts"], t

    # The stored race is untouched by a replay.
    assert c.get(f"/api/races/{rid}").json()["race"]["status"] == "finished"

    # A race from the extra (read-only) DB source.
    r2 = c.post(f"/api/races/{srid}/replay", params={"speed": 1000, "max_gap_s": 0})
    assert r2.status_code == 201 and r2.json()["replay_duration_s"] == 0
    with c.websocket_connect(r2.json()["ws"]) as ws:
        first = json.loads(ws.receive_text())
        assert first["is_replay"] is True and first["race_id"] == srid

    with pytest.raises(WebSocketDisconnect) as exc, c.websocket_connect("/ws/replays/missing") as ws:
        ws.receive_text()
    assert exc.value.code == 4404


def test_replay_store_is_bounded(synthetic: tuple[Database, str]) -> None:
    db, rid = synthetic
    tl = rebuild_events(db.race_rows(rid))
    store = ReplayStore(max_sessions=2)
    ids = [store.create(tl, speed=1.0).id for _ in range(3)]
    assert store.get(ids[0]) is None and store.get(ids[2]) is not None


# =========================================================================== 4. real data (read-only)


def _real_sources() -> list[Path]:
    bench = REPO / "data" / "bench"
    return [p for p in [*sorted(bench.glob("*.db")), bench / "dryrun"] if p.exists()]


def test_real_finished_races_replay_cleanly() -> None:
    sources = _real_sources()
    if not sources:
        pytest.skip("no data/bench sources")
    cat = ReplayCatalog(None, sources)
    races = cat.list_races(replayable_only=True)
    if not races:
        pytest.skip("no finished race in data/bench yet")
    for race in races[:4]:
        rows, _ = cat.load(race["id"])
        tl = rebuild_events(rows)
        assert check_stream(tl.events) == [], race["id"]
        counts = tl.counts()
        assert counts["experiment_started"] == sum(1 for e in rows["experiments"] if e.get("started_at")
                                                   and e.get("slot_id"))
        assert counts["llm_call"] == len(rows["llm_calls"]) - tl.dropped.get("llm_call_unparsed", 0)
        assert counts["race_finished"] == 1


def test_bench_db_opens_read_only_next_to_live_races() -> None:
    path = REPO / "data" / "bench" / "bench.db"
    if not path.exists():
        pytest.skip("no bench.db")
    with ReadOnlyDB(path) as ro:
        races = ro.list_races()
        with pytest.raises(sqlite3.OperationalError):
            ro._conn.execute("CREATE TABLE replay_probe (a)")
    for r in races:
        if r["status"] in ("created", "running"):
            with ReadOnlyDB(path) as ro, pytest.raises(ReplayError):
                rebuild_events(ro.race_rows(r["id"]))
            break


def test_cli_prints_a_timeline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db, rid = _synthetic_db(tmp_path / "cli.db")
    db.close()
    assert main([rid[:10], "--speed", "10", "--db", str(tmp_path / "cli.db")]) == 0
    out = capsys.readouterr().out
    assert "events over 40.0s race time -> 4.0s at 10x" in out
    assert "NEW BEST" in out and "CORRECT" in out and "RACE FINISHED" in out and "checks: ok" in out
    assert main(["--list", "--db", str(tmp_path / "cli.db")]) == 0
    assert rid in capsys.readouterr().out
