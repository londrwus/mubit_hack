"""Live 60-second demo preset 'demo-live-60s' and the one-click endpoint POST /api/demo/start
(offline: fakes only, no Mubit call, no Modal call, no LCBench model).

- the preset's fields: dev mode, demo label, real Opus through Mubit with record off, the LCBench
  task of lcbench-wide-bm-c30-15m at faster pacing, 16 slots, 60 s, $1 per arm, BM's cadence;
- scripts/bench.py labels it a demo in its checks and plan; scripts/headline.py refuses it;
- the endpoint starts only the allow-listed demo presets, refuses a second race while one runs,
  refuses a real race without the Mubit key, and returns the race id and WebSocket URL;
- every benchmark / ablation preset's stored config is byte-for-byte what it was.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_api import fake_factory

from backend.app import create_app
from backend.benchmark.configs import (
    DEMO_START_PRESETS,
    LIVE_DEMO_LABEL,
    PRESETS,
    STANDARD_PROPOSE,
    LEAN_PLANNER_PROPOSE,
)
from backend.core.types import ArmKind
from backend.demo import demo_config

ROOT = Path(__file__).resolve().parents[1]
LIVE = "demo-live-60s"
REF = "lcbench-wide-bm-c30-15m"


def _load(name: str, rel: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bench = _load("bench_live_demo_t", "scripts/bench.py")
hl = _load("headline_live_demo_t", "scripts/headline.py")


@pytest.fixture(autouse=True)
def _fake_quad_sigma(monkeypatch: pytest.MonkeyPatch) -> None:
    """The judge policy needs a calibrated seed spread for the fake target the endpoint test runs."""
    import backend.scheduler as sched
    from backend.benchmark.engine_rules import SigmaTable

    monkeypatch.setattr(sched, "_SIGMA_TABLES",
                        {**sched.sigma_tables(), "fake-quad": SigmaTable("fake-quad", ((0.0, 0.05),), "test")})


# --------------------------------------------------------------------------- preset


def test_live_demo_preset_fields() -> None:
    cfg, ref = PRESETS[LIVE], PRESETS[REF]
    assert cfg.mode == "dev" and cfg.is_dev and not cfg.strict and cfg.is_demo
    assert cfg.demo_label == LIVE_DEMO_LABEL == "demo run, not a benchmark result"
    assert LIVE_DEMO_LABEL in cfg.name and not cfg.modal_apps
    assert cfg.llm == "mubit" and cfg.model == "claude-opus-5-5"
    assert cfg.record_modes() == {"opus": "off", "opus_bm": "off"}
    assert cfg.arms == [ArmKind.opus, ArmKind.opus_bm]
    # the LCBench task of the c30 preset; only the pacing, length, slots-per-arm budget and cadence differ
    assert cfg.target == ref.target == "lcbench" and cfg.runner == ref.runner == "surrogate"
    assert cfg.gpus_per_arm == ref.gpus_per_arm == 16
    for f in ("max_proposals_per_call", "run_baseline_first", "launch_cutoff_s", "pipelining", "accept_unsure",
              "run_budget_s", "target_value", "median_stop_fractions", "benchmark_corrections"):
        assert getattr(cfg, f) == getattr(ref, f), f
    assert cfg.launch_cutoff_s is None  # no deadline rule: late runs are cancelled at the deadline, both arms
    assert cfg.budget_s == 60 and cfg.cost_cap_usd == 1.0
    assert cfg.surrogate_runtime_mult is not None and cfg.surrogate_runtime_mult < (ref.surrogate_runtime_mult or 1)
    assert cfg.features_for(ArmKind.opus) == ()
    assert cfg.features_for(ArmKind.opus_bm) == ("backfill", "judge_policy", "lean_planner", "planner_cadence")
    assert cfg.min_propose_interval_for(ArmKind.opus_bm) == 10.0 and cfg.min_propose_interval_for(ArmKind.opus) is None
    assert cfg.propose_variant_for(ArmKind.opus) is None
    assert cfg.propose_setup_for(ArmKind.opus)["prompt_version"] == STANDARD_PROPOSE[0]
    assert cfg.propose_variant_for(ArmKind.opus_bm) == LEAN_PLANNER_PROPOSE
    est = bench.estimate_gpu_cost_usd(cfg, bench.price_per_s("L4"))  # virtual GPU $
    assert est == pytest.approx(2 * 16 * 60 * 0.000222) and est <= (cfg.max_estimated_gpu_usd or 0)
    assert set(DEMO_START_PRESETS) == {LIVE, "demo-cifar-modal-5m"}


def test_bench_labels_the_live_demo() -> None:
    cfg = PRESETS[LIVE]
    res = {n: (ok, note) for n, ok, note in bench.checks(cfg, {"lcbench": 0.78})}
    assert bench.gating_ok([(n, ok, note) for n, (ok, note) in res.items()])
    assert LIVE_DEMO_LABEL in res["demo (never a headline race)"][1]
    assert "opus_bm: at most one propose call per 10 s" in res["planner cadence"][1]
    races = bench.build_races(LIVE)
    assert [c.seed for c in races] == [0] and "[demo run]" in races[0].name and races[0].is_demo
    dry = bench.build_races(LIVE, dry_run=True, slots=16)[0]
    assert dry.llm == "mock" and dry.runner == "surrogate" and dry.is_demo and dry.gpus_per_arm == 16


def test_headline_refuses_the_live_demo() -> None:
    with pytest.raises(SystemExit, match="not a benchmark result"):
        hl.refuse_ablation_preset(LIVE)
    with pytest.raises(SystemExit, match="not a benchmark result"):
        hl.run(preset=LIVE)
    cfg = demo_config(LIVE).model_dump(mode="json")
    rows = {"races": [{"id": "abcdef0123", "name": demo_config(LIVE).name, "config": cfg, "status": "finished",
                       "fairness": {"demo": {"headline": False, "label": LIVE_DEMO_LABEL}}}], "arms": []}
    assert hl.is_demo_race(rows)
    assert any("not a benchmark result" in w for w in hl.race_blockers(rows, False))
    # even without the manifest entry or the name tag, the stored config's demo_label is enough
    bare = {"races": [{"id": "abcdef0123", "name": "x", "config": cfg, "fairness": {}}], "arms": []}
    assert hl.is_demo_race(bare)


# Stored config of every preset other than the live demo, as committed (sha256 of model_dump_json,
# first 16 hex). A change here means a benchmark preset (and so its fingerprint) moved.
PRESET_DUMP_SHA = {
    "dev-local": "b3b204f2547524ca",
    "lm-30m": "0b712ba65b33385a",
    "cifar-30m": "566a126c0e0ec9e4",
    "rl-cpu-30m": "7bf05749f074ced8",
    "rl-wide-15m": "bf1c1e576c4d2f28",
    "cifar-15m": "125b4257efb86701",
    "rl-planner-ablation": "36cab9467ecadfb1",
    "lcbench-lean-15m": "829d63a9db1e4bac",
    "lcbench-15m": "cabf06593d07d939",
    "rl-wide-bm-15m": "3be46ac4c5a42bf9",
    "rl-wide-bm-cadence-15m": "c696dafedb917453",
    "cifar-bm-15m": "592eb2bb8d72e7eb",
    "cifar-bm-cadence-15m": "c3a1cdacf84ebf83",
    "lcbench-bm-15m": "62ccf141fa3b3f18",
    "lcbench-wide-bm-15m": "e51a92c14a2af73d",
    "lcbench-wide-bm-c30-15m": "e12f7eeccc701e89",
    "demo-3m": "6425efa26ca0e911",
    # The Modal demo gained gpu_telemetry_s=1.0 (live GPU % on our dashboard, docs/dashboard_modal.md);
    # without that one field its dump is exactly the committed one (d0c935c2895d1b6c, checked below).
    "demo-cifar-modal-5m": "9627ea49191aeff4",
}


def test_other_presets_unchanged() -> None:
    assert set(PRESETS) - set(PRESET_DUMP_SHA) == {LIVE}
    for name, sha in PRESET_DUMP_SHA.items():
        got = hashlib.sha256(PRESETS[name].model_dump_json().encode()).hexdigest()[:16]
        assert got == sha, name
    no_flag = PRESETS["demo-cifar-modal-5m"].model_copy(update={"gpu_telemetry_s": None})
    assert hashlib.sha256(no_flag.model_dump_json().encode()).hexdigest()[:16] == "d0c935c2895d1b6c"


# --------------------------------------------------------------------------- endpoint


def test_demo_config_allow_list() -> None:
    cfg = demo_config(LIVE)
    assert cfg.name.endswith(" seed 0 [demo run]") and cfg.is_demo and cfg.llm == "mubit"
    cif = demo_config("demo-cifar-modal-5m")
    assert cif.runner == "modal" and cif.modal_apps and cif.prewarm
    dry = demo_config(LIVE, dry_run=True)
    assert dry.llm == "mock" and dry.runner == "surrogate" and "[dry run] [demo run]" in dry.name
    dry_cif = demo_config("demo-cifar-modal-5m", dry_run=True)
    assert dry_cif.runner == "local" and not dry_cif.prewarm and dry_cif.llm == "mock"
    for bad in ("lcbench-wide-bm-c30-15m", "rl-wide-bm-cadence-15m", "demo-3m", "dev-local", "nope"):
        with pytest.raises(ValueError, match="allowed"):
            demo_config(bad)


def _short_live_demo(monkeypatch: pytest.MonkeyPatch, budget_s: float = 1.2) -> None:
    """The live demo preset on the fake target and runner (same arms, features, labels, record modes)."""
    base = PRESETS[LIVE]
    monkeypatch.setitem(PRESETS, LIVE, base.model_copy(update={
        "target": "fake-quad", "budget_s": budget_s, "gpus_per_arm": 2, "runner": "local",
        "surrogate_runtime_mult": None}))


def test_endpoint_refuses_presets_outside_the_allow_list() -> None:
    app = create_app(":memory:", arm_deps_factory=fake_factory, tick_s=0.02)
    with TestClient(app) as c:
        for bad in ("lcbench-wide-bm-c30-15m", "cifar-bm-cadence-15m", "demo-3m", "nope"):
            r = c.post(f"/api/demo/start?preset={bad}")
            assert r.status_code == 400 and "allowed" in r.text, (bad, r.text)
        assert c.post("/api/demo/start").status_code == 422
        assert c.get("/api/races").json() == []   # nothing was created
        listed = c.get("/api/demo/presets").json()
        assert [p["preset"] for p in listed] == list(DEMO_START_PRESETS)
        assert all(p["demo_label"] for p in listed)


def test_endpoint_refuses_a_real_demo_without_the_mubit_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.agent.mubit_client as mc

    def no_key(env_file: Any = None, env_var: str = "MUBIT_API_KEY") -> str:
        raise RuntimeError(f"{env_var} is not set")

    monkeypatch.setattr(mc, "load_mubit_api_key", no_key)
    app = create_app(":memory:", tick_s=0.02)   # default (real) wiring: the key is checked
    with TestClient(app) as c:
        r = c.post(f"/api/demo/start?preset={LIVE}")
        assert r.status_code == 503 and "MUBIT_API_KEY is not set" in r.text
        assert c.get("/api/races").json() == []


def test_endpoint_starts_the_demo_and_streams_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _short_live_demo(monkeypatch)
    app = create_app(":memory:", arm_deps_factory=fake_factory, tick_s=0.02)
    with TestClient(app) as c:
        r = c.post(f"/api/demo/start?preset={LIVE}")
        assert r.status_code == 201, r.text
        body = r.json()
        rid = body["race_id"]
        assert body["status"] == "running" and body["demo_label"] == LIVE_DEMO_LABEL and not body["dry_run"]
        assert body["ws"] == f"/ws/races/{rid}/dashboard?tick_s=1&raw=false"
        assert body["ws_url"].startswith("ws://") and body["ws_url"].endswith(body["ws"])
        assert body["arms"] == ["opus", "opus_bm"] and body["llm"] == "mubit"
        assert "[demo run]" in body["name"]
        assert body["modal"] is None   # simulated machines: no live Modal panel, no Modal call
        # one race at a time: a second click while it runs is refused, nothing new is created
        r2 = c.post(f"/api/demo/start?preset={LIVE}")
        assert r2.status_code == 409
        last: dict[str, Any] = {}
        with c.websocket_connect(body["ws"]) as ws:
            while True:
                msg = ws.receive_json()
                if msg.get("type") == "tick":
                    last = msg["snapshot"]
                    if last.get("status") == "finished":
                        break
        assert last["header"]["demo_run"] is True
        assert last["header"]["mode_tag"] == "Demo run, not a benchmark result"
        assert last["header"]["preset"] == LIVE
        deadline = time.time() + 10
        while time.time() < deadline and c.get("/api/health").json()["live_races"]:
            time.sleep(0.05)
        state = c.get(f"/api/races/{rid}").json()
        assert state["race"]["status"] == "finished"
        assert state["race"]["is_dev"]
        assert state["race"]["fairness"]["demo"]["label"] == LIVE_DEMO_LABEL
        assert state["race"]["config"]["demo_label"] == LIVE_DEMO_LABEL
        assert len(c.get("/api/races").json()) == 1
        # the race is over, so the next click starts a new one
        r3 = c.post(f"/api/demo/start?preset={LIVE}")
        assert r3.status_code == 201
        c.post(f"/api/races/{r3.json()['race_id']}/stop")
