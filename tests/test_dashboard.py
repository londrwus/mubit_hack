"""Dashboard API (backend/dashboard.py, docs/dashboard_api.md).

1. Reducer on hand-made events: slot states, waiting %, GPU $, model quality (incl. a claim the
   checker rejects), fakes caught, planner thinking, final summary.
2. A fake in-process race: the reducer's numbers equal the engine's ArmSummary; the new engine
   events (judge_verdict, planner_*) and fields (source, quality) are rebuilt by the replay.
3. API: snapshot endpoint, live dashboard WS, replay dashboard WS, prefix ids, metric_stride=0.
4. Real logged races (read-only, skipped without data/bench/bench.db): R78 / R82 seed 0 frames
   equal the values in docs/dashboard_spec.md.
No network, no GPU, no LLM.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_api import RACE, fake_factory
from test_engine import SPEC, FakeLLM, FakeRunner, llm_arm

from backend.app import create_app
from backend.benchmark.configs import RaceConfig
from backend.core.events import (
    ALL_EVENT_TYPES,
    EVENT_TYPES,
    ExperimentFinishedEvent,
    ExperimentStartedEvent,
    GpuStateEvent,
    JudgeVerdictEvent,
    LLMCallEvent,
    PlannerFinishedEvent,
    PlannerStartedEvent,
    RaceFinishedEvent,
    SlotEvent,
    TickEvent,
    parse_event,
)
from backend.core.types import ArmKind, ArmSummary, ExperimentStatus, LLMCallRecord, SlotState, StepType, new_id
from backend.dashboard import DashboardState, build_header, fmt_score, fold, logged_race_ref, play_dashboard
from backend.db import Database
from backend.replay import ReplayCatalog, check_stream, rebuild_events, schedule
from backend.scheduler import EventBus, RaceEngine

REPO = Path(__file__).resolve().parents[1]
BENCH_DB = REPO / "data" / "bench" / "bench.db"
T0 = 2_000_000.0
PRICE = 0.001


# =========================================================================== 1. reducer


def _header(goal: float = 0.65) -> dict[str, Any]:
    race = {"id": "r1", "name": "unit", "target": "fake-quad", "budget_s": 100.0, "gpus_per_arm": 2,
            "runner": "modal", "mode": "dev", "llm_mode": "mock", "started_at": T0,
            "config": {"arms": ["opus", "opus_bm"], "target_value": goal, "llm": "mock"}}
    arms = [{"kind": k, "n_slots": 2, "hardware": "L4", "price_per_s": PRICE} for k in ("opus", "opus_bm")]
    h = build_header(race, arms)
    h["direction"], h["goal"] = "max", goal     # fake-quad is not a registered target here
    return h


def _gpu(arm: str, slot: str, state: str, t: float, exp: str | None = None) -> GpuStateEvent:
    return GpuStateEvent(race_id="r1", arm=arm, slot_id=slot, state=state, experiment_id=exp, ts=T0 + t)


def _fin(arm: str, exp: str, slot: str, t: float, status: str = "succeeded", score: float | None = 0.5,
         quality: float | None = None, trap: bool = False) -> ExperimentFinishedEvent:
    return ExperimentFinishedEvent(race_id="r1", arm=arm, experiment_id=exp, slot_id=slot, status=status,
                                   score=score, quality=quality if quality is not None else score, is_trap=trap,
                                   ts=T0 + t)


def _start(arm: str, exp: str, slot: str, t: float, source: str) -> ExperimentStartedEvent:
    return ExperimentStartedEvent(race_id="r1", arm=arm, experiment_id=exp, slot_id=slot, config={}, seed=0,
                                  hardware="L4", source=source, ts=T0 + t)


def test_reducer_slots_money_quality_fakes() -> None:
    d = DashboardState(_header())
    ev: list[Any] = [
        _gpu("opus", "s0", "idle", 0), _gpu("opus", "s1", "idle", 0),
        _gpu("opus_bm", "b0", "idle", 0), _gpu("opus_bm", "b1", "idle", 0),
        PlannerStartedEvent(race_id="r1", arm="opus", k=2, ts=T0 + 0),
        _gpu("opus", "s0", "busy", 0, "base"), _start("opus", "base", "s0", 0, "baseline"),
        _gpu("opus_bm", "b0", "busy", 0, "x1"), _start("opus_bm", "x1", "b0", 0, "sampler"),
        _gpu("opus_bm", "b1", "busy", 0, "x2"), _start("opus_bm", "x2", "b1", 0, "sampler"),
        _gpu("opus", "s0", "idle", 10), _fin("opus", "base", "s0", 10, score=0.30),
        PlannerFinishedEvent(race_id="r1", arm="opus", started_at=T0, k=2, n_added=1, ts=T0 + 12),
        _gpu("opus", "s1", "busy", 12, "p1"), _start("opus", "p1", "s1", 12, "llm"),
        _gpu("opus_bm", "b0", "idle", 20), _fin("opus_bm", "x1", "b0", 20, score=0.90, quality=0.40, trap=True),
        _gpu("opus_bm", "b1", "idle", 25), _fin("opus_bm", "x2", "b1", 25, status="cancelled", score=None),
        JudgeVerdictEvent(race_id="r1", arm="opus_bm", experiment_id="x1", verdict="invalid", llm_verdict="invalid",
                          llm_reason="contamination_or_eval_leak", claimed_score=0.90, quality=0.40, is_trap=True,
                          source="sampler", fake_caught=True, ts=T0 + 26),
        _gpu("opus", "s1", "idle", 30), _fin("opus", "p1", "s1", 30, score=0.70),          # goal 0.65 reached
        JudgeVerdictEvent(race_id="r1", arm="opus", experiment_id="p1", verdict="invalid", claimed_score=0.70,
                          quality=0.70, source="llm", ts=T0 + 32),                        # a real run rejected
        LLMCallEvent(race_id="r1", call=LLMCallRecord(arm="opus", session_id="s", step_type=StepType.judge_improvement,
                                                      model="m", cost_usd=0.25, valid=True), ts=T0 + 32),
        _gpu("opus", "s0", "idle", 40),                                                   # idle -> idle: closed
    ]
    derived: list[SlotEvent] = []
    for e in ev:
        derived += d.apply(e)
    s = d.snapshot(T0 + 50)
    assert s["status"] == "running" and s["clock"] == "00:50 / 01:40" and s["race_elapsed_s"] == 50
    opus, bm = s["arms"]
    assert (opus["kind"], opus["side"], opus["label"]) == ("opus", "left", "Standard AI assistant")
    assert (bm["kind"], bm["side"], bm["label"]) == ("opus_bm", "right", "Busy Machines")

    # slots: s0 off since 40, s1 waiting; b0 waiting, b1 cancelled (its run was stopped)
    assert {x["slot_id"]: x["state"] for x in opus["slots"]} == {"s0": "off", "s1": "waiting"}
    assert {x["slot_id"]: x["state"] for x in bm["slots"]} == {"b0": "waiting", "b1": "cancelled"}
    assert next(x for x in bm["slots"] if x["slot_id"] == "b1")["experiment_id"] == "x2"
    assert [(e.slot_id, e.state, e.race_elapsed_s) for e in derived if e.arm == "opus" and e.slot_id == "s0"] == [
        ("s0", "waiting", 0), ("s0", "busy", 0), ("s0", "waiting", 10), ("s0", "off", 40)]
    assert any(e.state == "busy" and e.source == "sampler" for e in derived)

    # money and waiting: opus paid s0 0-40 (busy 10) + s1 0-50 (busy 12-30): busy 28, idle 62
    assert opus["busy_s"] == pytest.approx(28) and opus["idle_s"] == pytest.approx(62)
    assert opus["waiting_so_far_pct"] == pytest.approx(100 * 62 / 90, abs=1e-3)
    assert opus["gpu_bill_usd"] == pytest.approx(90 * PRICE) and opus["ai_bill_usd"] == pytest.approx(0.25)
    assert opus["working_now"] == 0 and opus["open_now"] == 1 and opus["waiting_now_pct"] == 100.0

    # quality: p1 (0.70) counted at 30 (goal reached), retracted at 32 by the checker -> back to 0.30
    mq = opus["model_quality"]
    assert mq["best"] == 0.30 and mq["baseline"] == 0.30 and mq["goal_reached_at_s"] is None
    assert mq["series"] == [[10, 0.30], [30, 0.70], [32, 0.30]]
    assert opus["checks"]["invalid"] == 1 and opus["checks"]["claims_rejected_not_fake"] == 1
    assert opus["finished_by_source"] == {"baseline": 1, "llm": 1} and opus["experiments_finished"] == 2

    # the trap never counts; the checker caught it
    assert bm["model_quality"]["best"] is None
    assert bm["fakes_caught"] == 1 and opus["fakes_caught"] == 0
    row = bm["fakes"][0]
    assert row["text"] == "0:26 · claimed 0.9000, real score 0.4000 (tested on data it had already seen)"
    assert bm["experiments_cancelled"] == 1 and bm["finished_by_source"] == {"sampler": 1}
    assert [f["kind"] for f in s["feed"]] == ["fake_caught", "goal_reached"]

    # planner: thinking 0-12 then idle
    assert opus["planner"]["thinking"] is False and opus["planner"]["thinking_s"] == pytest.approx(12)
    assert d.snapshot(T0 + 5)["arms"][0]["planner"]["thinking"] is False   # state is "as of the last event"

    # race end: every slot off, the stored summary's numbers win
    fin = RaceFinishedEvent(race_id="r1", reason="budget", ts=T0 + 100, summaries=[
        ArmSummary(arm="opus", llm_cost_usd=0.25, gpu_cost_usd=0.111, gpu_idle_pct=70.0, experiments_finished=2,
                   best_score=0.30, time_to_target_s=None),
        ArmSummary(arm="opus_bm", gpu_cost_usd=0.2)])
    out = d.apply(fin)
    assert {e.state for e in out} == {"off"}
    f = d.snapshot()
    assert f["status"] == "finished" and f["finished_text"] == "Finished · 01:40" and f["clock"] == "01:40 / 01:40"
    assert f["arms"][0]["gpu_bill_usd"] == 0.111 and f["arms"][0]["waiting_so_far_pct"] == 70.0
    assert all(x["state"] == "off" for a in f["arms"] for x in a["slots"])
    json.dumps(f, allow_nan=False)   # valid JSON for the browser


def test_header_labels_and_formats() -> None:
    race = {"id": "4ca4ed27-not-real", "name": "RL wide 15 min (15 s runs), Busy Machines with Opus + planner "
            "cadence 60 s seed 0", "target": "rl", "budget_s": 900, "gpus_per_arm": 30, "runner": "modal",
            "config": {"arms": ["opus", "opus_bm", "optuna"], "cost_cap_usd": 8.0, "target_value": -170.0}}
    h = build_header(race, [{"kind": k, "n_slots": 30, "hardware": "CPU", "price_per_s": 1.31e-05}
                            for k in ("optuna", "opus_bm", "opus")], mode="replay", replay={"speed": 30})
    assert h["task"] == "Game-playing AI" and h["goal_text"] == "Goal: −170"
    assert h["budget_text"] == "15 min · up to $8.00 per side · 30 machines each"
    assert h["machines"]["caption"] == "30 cloud CPU workers" and h["machines"]["waiting_label"] == "Machines waiting"
    assert [a["kind"] for a in h["arms"]] == ["opus", "opus_bm", "optuna"]       # left, right, secondary
    assert h["preset"] == "rl-wide-bm-cadence-15m" and h["score_decimals"] == 1
    assert h["mode_tag"].startswith("Replay of") and h["mode_tag"].endswith("30× speed")
    demo = build_header({"id": "x", "target": "cifar", "budget_s": 300, "gpus_per_arm": 3, "runner": "modal",
                         "config": {"arms": ["opus", "opus_bm"], "demo_label": "demo run", "cost_cap_usd": 3.0}},
                        [{"kind": "opus", "n_slots": 3, "hardware": "L4", "price_per_s": 0.000222}])
    assert demo["demo_run"] is True and demo["mode_tag"] == "Demo run, not a benchmark result"
    assert demo["budget_text"] == "5 min · up to $3.00 per side · 3 L4 GPUs each"
    assert demo["machines"]["label"] == "GPUs" and demo["machines"]["caption"] == "3 L4 GPUs on Modal"
    assert fmt_score(-1068.66, 1) == "−1,068.7" and fmt_score(0.89046, 4) == "0.8905"


def test_event_types_are_parseable() -> None:
    assert len(EVENT_TYPES) == 8 and set(EVENT_TYPES) < set(ALL_EVENT_TYPES)
    for ev in (JudgeVerdictEvent(race_id="r", arm="opus", experiment_id="e", verdict="invalid", fake_caught=True),
               PlannerStartedEvent(race_id="r", arm="opus", k=3),
               PlannerFinishedEvent(race_id="r", arm="opus", started_at=1.0, n_added=2),
               SlotEvent(race_id="r", arm="opus", slot_id="s", state="cancelled"),
               TickEvent(race_id="r", snapshot={"a": 1})):
        assert parse_event(ev.model_dump_json()) == ev
    old = {"type": "experiment_started", "race_id": "r", "arm": "opus", "experiment_id": "e", "slot_id": "s",
           "config": {}, "seed": 0, "hardware": "L4"}
    assert parse_event(old).source is None                     # events from before the field still parse


# =========================================================================== 2. engine + replay


async def _race(db: Database) -> tuple[str, list[Any], RaceEngine]:
    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.opus, ArmKind.mubit], budget_s=2.0, gpus_per_arm=2,
                     seed=0, llm="mock", engine_features={ArmKind.mubit: ["judge_policy"]})
    rid = new_id()
    bus = EventBus()
    live: list[Any] = []
    bus.add_listener(live.append)
    deps = [llm_arm(ArmKind.opus, FakeLLM(0.3, seed=1, trap_rate=0.5), rid, FakeRunner(2, run_s=0.2, report_heldout=True)),
            llm_arm(ArmKind.mubit, FakeLLM(0.02, seed=1, trap_rate=0.5), rid,
                    FakeRunner(2, run_s=0.2, report_heldout=True))]
    eng = RaceEngine(cfg, deps, spec=SPEC, race_id=rid, db=db, bus=bus, tick_s=0.02)
    await eng.run()
    return rid, live, eng


def test_live_race_reducer_matches_summaries_and_replay_rebuilds_new_events() -> None:
    db = Database()
    rid, live, eng = asyncio.run(_race(db))
    race, arms = db.get_race(rid), db.get_arms(rid)
    header = build_header(race, arms)
    header["direction"], header["goal"] = SPEC.direction, SPEC.target_value
    d = DashboardState(header)
    before_final: dict[str, dict[str, Any]] = {}
    for ev in live:
        if ev.type == "race_finished":
            before_final = {a["kind"]: a for a in d.snapshot(ev.ts)["arms"]}
        d.apply(ev)
    for s in eng.summaries:
        a = before_final[s.arm.value]
        assert a["ai_bill_usd"] == pytest.approx(s.llm_cost_usd, abs=1e-9)
        assert a["experiments_finished"] == s.experiments_finished
        assert a["gpu_bill_usd"] == pytest.approx(s.gpu_cost_usd, rel=1e-6, abs=1e-9)
        assert a["waiting_so_far_pct"] == pytest.approx(s.gpu_idle_pct, abs=1e-3)
        assert a["model_quality"]["best"] == s.best_score

    # new engine events and fields
    started = [e for e in live if e.type == "experiment_started"]
    assert started and all(e.source in ("baseline", "llm") for e in started)
    fin = [e for e in live if e.type == "experiment_finished" and e.status == ExperimentStatus.succeeded]
    assert fin and all(e.quality is not None for e in fin if not e.is_trap)
    verdicts = [e for e in live if e.type == "judge_verdict"]
    assert verdicts and any(e.fake_caught for e in verdicts)
    assert all(e.fake_caught == (e.is_trap and e.verdict is not None and e.verdict.value == "invalid")
               for e in verdicts)
    for kind in (ArmKind.opus, ArmKind.mubit):
        n = eng.reports[kind.value]["planner_cadence"]["propose_calls"]
        assert sum(1 for e in live if e.type == "planner_started" and e.arm == kind) == n

    # the replay rebuilds them from storage
    tl = rebuild_events(db.race_rows(rid))
    assert check_stream(tl.events) == []
    rb = tl.events

    def by(evs: list[Any], t: str) -> dict[str, dict[str, Any]]:
        return {e.experiment_id: e.model_dump(mode="json", exclude={"ts"}) for e in evs if e.type == t}

    assert by(rb, "judge_verdict") == by(live, "judge_verdict")
    assert by(rb, "experiment_started") == by(live, "experiment_started")
    assert by(rb, "experiment_finished") == by(live, "experiment_finished")
    ls = sorted((e.arm.value, e.ts, e.k) for e in live if e.type == "planner_started")
    rs = sorted((e.arm.value, e.ts, e.k) for e in rb if e.type == "planner_started")
    assert [(a, k) for a, _, k in ls] == [(a, k) for a, _, k in rs]
    assert all(abs(x[1] - y[1]) <= 0.0006 for x, y in zip(ls, rs, strict=True))    # report rounds to ms
    lf = Counter(e.arm.value for e in live if e.type == "planner_finished")
    rf = Counter(e.arm.value for e in rb if e.type == "planner_finished")
    assert all(abs(lf[k] - rf[k]) <= 1 for k in set(lf) | set(rf))

    # folding the replay gives the same final frame as the live stream
    f_live = d.snapshot()
    f_rb = fold(rb, header, t0=tl.t0).snapshot()
    for a, b in zip(f_live["arms"], f_rb["arms"], strict=True):
        for key in ("ai_bill_usd", "gpu_bill_usd", "experiments_finished", "fakes_caught", "waiting_so_far_pct"):
            assert a[key] == b[key], key
        assert a["model_quality"]["best"] == b["model_quality"]["best"]


def test_play_dashboard_ticks_and_start_offset() -> None:
    h = _header()
    evs = [_gpu("opus", "s0", "idle", 0), _gpu("opus", "s0", "busy", 0, "e"), _start("opus", "e", "s0", 0, "llm"),
           _gpu("opus", "s0", "idle", 50), _fin("opus", "e", "s0", 50, score=0.7),
           RaceFinishedEvent(race_id="r1", summaries=[], ts=T0 + 100)]
    sent: list[dict[str, Any]] = []
    now = [0.0]

    async def send(m: str) -> None:
        sent.append(json.loads(m))

    async def sleep(dt: float) -> None:
        now[0] += dt

    offs = schedule(evs, 10.0)    # 100 s race at 10x = 10 s of wall time
    n = asyncio.run(play_dashboard(evs, offs, send, DashboardState(h, t0=T0), payload=lambda e: e.model_dump(mode="json"),
                                   speed=10.0, tick_s=2.0, raw=False, clock=lambda: now[0], sleep=sleep))
    assert n == len(sent)
    ticks = [m for m in sent if m["type"] == "tick"]
    assert [round(t["race_elapsed_s"]) for t in ticks] == [0, 20, 40, 60, 80, 100, 100]
    assert ticks[-1]["snapshot"]["status"] == "finished"
    assert {m["type"] for m in sent} == {"tick", "slot", "experiment_finished", "race_finished"}   # raw=False

    sent.clear()
    now[0] = 0.0
    asyncio.run(play_dashboard(evs, offs, send, DashboardState(h, t0=T0), payload=lambda e: e.model_dump(mode="json"),
                               speed=10.0, tick_s=100.0, raw=True, start_s=60, clock=lambda: now[0], sleep=sleep))
    first = sent[0]
    assert first["type"] == "tick" and first["snapshot"]["arms"][0]["model_quality"]["best"] == 0.7
    assert first["race_elapsed_s"] == 60 and now[0] == pytest.approx(4.0)      # 60 -> 100 s at 10x
    assert [m["type"] for m in sent[1:]] == ["race_finished", "slot", "tick"]


# =========================================================================== 3. API


@pytest.fixture()
def client() -> Any:
    app = create_app(":memory:", arm_deps_factory=fake_factory, tick_s=0.02, replay_sources=[])
    with TestClient(app) as c:
        yield c


def test_api_live_dashboard_snapshot_and_replay(client: TestClient) -> None:
    rid = client.post("/api/races", json={**RACE, "arms": ["opus", "mubit"], "budget_s": 1.0}).json()["id"]
    assert client.get("/api/races/nope/snapshot").status_code == 404
    assert client.get(f"/api/races/{rid}/snapshot").status_code == 409            # created, not finished
    with client.websocket_connect(f"/ws/races/{rid}/dashboard?tick_s=0.1&raw=false") as ws:
        assert client.post(f"/api/races/{rid}/start").status_code == 200
        msgs: list[dict[str, Any]] = []
        while True:
            msgs.append(json.loads(ws.receive_text()))
            if msgs[-1]["type"] == "race_finished":
                while msgs[-1]["type"] != "tick":                                # slots off, then the final tick
                    msgs.append(json.loads(ws.receive_text()))
                break
    types = Counter(m["type"] for m in msgs)
    assert types["tick"] >= 3 and types["slot"] > 0 and "gpu_state" not in types and "metric" not in types
    final = msgs[-1]
    assert final["type"] == "tick" and final["snapshot"]["status"] == "finished"
    snap = final["snapshot"]
    race = client.get(f"/api/races/{rid}").json()
    for a in snap["arms"]:
        arm = next(x for x in race["arms"] if x["kind"] == a["kind"])
        assert a["experiments_finished"] == arm["summary"]["experiments_finished"]
        assert a["ai_bill_usd"] == pytest.approx(arm["summary"]["llm_cost_usd"])
    assert snap["header"]["mode_tag"] == "Dev run (mock AI), not a benchmark result"
    assert [a["side"] for a in snap["header"]["arms"]] == ["left", "right"]

    # the same race, now finished: REST snapshot (at the end, and at 0.5 s), prefix ids
    r = client.get(f"/api/races/{rid[:10]}/snapshot")
    assert r.status_code == 200 and r.json()["status"] == "finished"
    mid = client.get(f"/api/races/{rid}/snapshot", params={"at_s": 0.5}).json()
    assert mid["race_elapsed_s"] == pytest.approx(0.5, abs=1e-6) and mid["status"] == "running"

    # replay through the dashboard stream, no metric events loaded
    info = client.post(f"/api/races/{rid[:10]}/replay", params={"speed": 1000, "metric_stride": 0}).json()
    assert info["race_id"] == rid and "metric" not in info["counts"]
    with client.websocket_connect(f"/ws/replays/{info['replay_id']}/dashboard?tick_s=0.05") as ws:
        rmsgs: list[dict[str, Any]] = []
        while True:
            m = json.loads(ws.receive_text())
            rmsgs.append(m)
            if m["type"] == "tick" and m["snapshot"]["status"] == "finished":
                break
    assert all(m["is_replay"] is True and m["replay_id"] == info["replay_id"] for m in rmsgs)
    assert rmsgs[-1]["snapshot"]["header"]["is_replay"] is True
    end = rmsgs[-1]["snapshot"]
    for a, b in zip(end["arms"], snap["arms"], strict=True):
        assert (a["experiments_finished"], a["ai_bill_usd"], a["fakes_caught"]) == \
               (b["experiments_finished"], b["ai_bill_usd"], b["fakes_caught"])

    # a finished race on the live dashboard socket: one final tick, then close
    with client.websocket_connect(f"/ws/races/{rid}/dashboard?raw=false") as ws:
        m = json.loads(ws.receive_text())
        while m["type"] != "tick":
            m = json.loads(ws.receive_text())
        assert m["snapshot"]["status"] == "finished"


# =========================================================================== 4. real logged races


def _real(prefix: str) -> Any:
    cat = ReplayCatalog(None, [BENCH_DB])
    rid = cat.resolve(prefix)
    rows, _ = cat.load(rid, metrics=False)
    return rebuild_events(rows)


@pytest.mark.skipif(not BENCH_DB.is_file(), reason="no data/bench/bench.db")
def test_r78_and_r82_seed0_frames_match_the_dashboard_spec() -> None:
    # Values from docs/dashboard_spec.md section 3 (computed there from the same logged races).
    want = {
        "4ca4ed27": {300: {"opus": ("$1.04", 153, "−157.3", 0, 71.9), "opus_bm": ("$0.30", 523, "−156.3", 14, None)},
                     None: {"opus": ("$3.30", 497, "−155.0", 0, 69.0), "opus_bm": ("$0.70", 1594, "−154.9", 21, 1.4)}},
        "c1f4b1d9": {300: {"opus": ("$0.97", 153, "0.8864", 0, 36.5), "opus_bm": ("$0.28", 415, "0.8845", 4, None)},
                     None: {"opus": ("$3.01", 420, "0.8905", 0, 26.2), "opus_bm": ("$0.67", 761, "0.8903", 4, 0.5)}},
    }
    goals = {"4ca4ed27": {"opus": 74.6, "opus_bm": 16.9}, "c1f4b1d9": {"opus": 24.4, "opus_bm": 33.8}}
    for prefix, frames in want.items():
        tl = _real(prefix)
        h = build_header(tl.race, tl.arms, mode="replay")
        assert h["logged_race"]["run"] == 1 and h["logged_race"]["ref"] == {"4ca4ed27": "R78", "c1f4b1d9": "R82"}[prefix]
        for at, arms in frames.items():
            until = tl.t0 + at if at is not None else None
            snap = fold(tl.events, h, t0=tl.t0, until=until).snapshot(until)
            got = {a["kind"]: a for a in snap["arms"]}
            for kind, (bill, done, quality, fakes, waiting) in arms.items():
                a = got[kind]
                assert f"${a['ai_bill_usd']:.2f}" == bill, (prefix, at, kind)
                assert a["experiments_finished"] == done, (prefix, at, kind)
                assert a["model_quality"]["best_text"] == quality, (prefix, at, kind)
                assert a["fakes_caught"] == fakes, (prefix, at, kind)
                if waiting is not None:
                    assert round(a["waiting_so_far_pct"], 1) == waiting, (prefix, at, kind)
                assert round(a["model_quality"]["goal_reached_at_s"], 1) == goals[prefix][kind]
        # the reducer's own accounting (before the summary override) equals the stored summaries
        d = fold(tl.events, h, t0=tl.t0)
        for kind, arm in d.arms.items():
            busy = sum(s.busy_s for s in arm.slots.values())
            idle = sum(s.idle_s for s in arm.slots.values())
            assert (busy + idle) * arm.price == pytest.approx(arm.final["gpu_cost_usd"], rel=1e-6)
            assert 100 * idle / (busy + idle) == pytest.approx(arm.final["gpu_idle_pct"], abs=1e-3)
            assert arm.ai_bill == pytest.approx(arm.final["llm_cost_usd"], abs=1e-9)
            assert (arm.best[1] if arm.best else None) == arm.final["best_score"]


def test_logged_race_ref_reads_the_showcase_race_sets(tmp_path: Path) -> None:
    p = tmp_path / "showcase.json"
    p.write_text(json.dumps({"sources": {"race_sets": {"R1": ["aaaa", "bbbb"]}}}), encoding="utf-8")
    assert logged_race_ref("bbbb1234", p) == {"ref": "R1", "run": 2, "runs": 2}
    assert logged_race_ref("cccc", p) is None
    assert logged_race_ref("x", tmp_path / "missing.json") is None


def test_slot_state_enum_unchanged() -> None:
    # The engine's slot states stay busy/idle; the display states are derived (no engine change).
    assert {s.value for s in SlotState} == {"busy", "idle"}
