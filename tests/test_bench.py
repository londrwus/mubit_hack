"""night/bench runner tests: per-arm recording mode, frozen goals, overrides, and a full
benchmark-mode race through ``scripts/bench.py``'s ``run_race`` with fakes (no network,
no Modal, no LLM spend)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest

from backend.agent.llm_base import OPUS_MODEL, RECORD_HEADER, REQUEST_ID_HEADER
from backend.agent.loop import DefaultPrompts, LLMResearcher
from backend.agent.mock_llm import MockLLM
from backend.agent.mubit_client import MubitClient
from backend.benchmark.configs import BENCH_RECORD_MODES, PRESETS, RaceConfig
from backend.core.types import (
    ArmKind,
    ExperimentStatus,
    RunHandle,
    RunResult,
    Slot,
    new_id,
    now,
)
from backend.db import BenchmarkIntegrityError, Database
from backend.scheduler import ArmDeps, ExperimentSnapshot, RaceEngine, normalize_record, race_spec
from workers.targets.base import BaseTarget, ParamSpec, TargetSpec, register_target

ROOT = Path(__file__).resolve().parents[1]


def _load_bench() -> Any:
    spec_ = importlib.util.spec_from_file_location("bench_script_t", ROOT / "scripts" / "bench.py")
    assert spec_ is not None and spec_.loader is not None
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


bench = _load_bench()


# --------------------------------------------------------------------------- fakes


class BenchToyTarget(BaseTarget):
    spec = TargetSpec(
        name="bench-toy",
        description="Synthetic target for bench.py tests.",
        metric="val_loss",
        direction="min",
        target_value=0.0001,      # spec placeholder: unreachable
        hardware="L4",
        budget_s=1,
        config_space={
            "lr": ParamSpec(type="float", low=1e-4, high=1.0, log=True, default=1e-2, description="learning rate"),
            "width": ParamSpec(type="int", low=8, high=256, default=32, description="hidden width"),
            "leak": ParamSpec(type="bool", default=False, description="draw validation rows from train"),
        },
        trap_option="leak",
    )

    def run(self, config: dict[str, Any], seed: int, budget_s: int, emit: Any) -> dict[str, Any]:
        return {"score": 1.0}


register_target(BenchToyTarget())
TOY = BenchToyTarget.spec


class ToyRunner:
    name = "fake"

    def __init__(self, n: int, run_s: float = 0.15) -> None:
        self.n, self.run_s = n, run_s
        self._tasks: dict[str, asyncio.Task[RunResult]] = {}

    def slots(self) -> list[Slot]:
        return [Slot(slot_id=f"toy-{i}", index=i, hardware="L4", runner=self.name) for i in range(self.n)]

    async def launch(self, target: str, config: dict[str, Any], seed: int, budget_s: int, on_metric: Any) -> RunHandle:
        h = RunHandle(runner=self.name, target=target, config=config, seed=seed, budget_s=budget_s,
                      hardware="L4", run_ref=f"toy-{new_id()[:8]}")
        self._tasks[h.run_id] = asyncio.create_task(self._train(h))
        return h

    async def _train(self, h: RunHandle) -> RunResult:
        t0 = now()
        try:
            await asyncio.sleep(self.run_s)
        except asyncio.CancelledError:
            return RunResult(score=None, status=ExperimentStatus.cancelled, started_at=t0, ended_at=now(),
                             hardware="L4", run_ref=h.run_ref)
        score = 1.0 + abs(float(h.config["lr"]) - 0.01)
        return RunResult(score=score, status=ExperimentStatus.succeeded, started_at=t0, ended_at=now(),
                         hardware="L4", run_ref=h.run_ref, metrics={"heldout_score": score + 0.01})

    async def wait(self, handle: RunHandle) -> RunResult:
        return await self._tasks[handle.run_id]

    async def cancel(self, handle: RunHandle) -> None:
        t = self._tasks.get(handle.run_id)
        if t is not None and not t.done():
            t.cancel()

    def price_per_second(self, hardware: str) -> float:
        return 0.000222


class SafeMock(MockLLM):
    """MockLLM dressed as a real client (benchmark-safe, not replay), so a benchmark-mode race
    can run in-process. Records which ``record`` value every call received."""

    benchmark_safe = True

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.record_seen: list[Any] = []

    async def call(self, step, system, user, schema, arm, session_id, record=False, **kw):  # type: ignore[no-untyped-def,override]
        self.record_seen.append(record)
        parsed, rec = await super().call(step, system, user, schema, arm, session_id, record, **kw)
        return parsed, rec.model_copy(update={"is_replay": False, "model": OPUS_MODEL, "cost_usd": 0.001})


def toy_factory(made: dict[ArmKind, Any]) -> Any:
    """Stand-in for backend.app.default_arm_deps: same record wiring, fake LLM + runner."""

    def factory(race_id: str, cfg: RaceConfig, spec: TargetSpec, kind: ArmKind) -> ArmDeps:
        mode = cfg.record_mode_for(kind)
        record: bool | str = {"off": False, "full": True}.get(mode, mode)
        llm = SafeMock(spec=spec, seed=cfg.seed)
        made[kind] = llm
        r = LLMResearcher(kind=kind, llm=llm, spec=spec, session_id=f"race-{race_id}/{kind.value}/seed-0",
                          prompts=DefaultPrompts(), fallback_model=OPUS_MODEL, record=record)
        return ArmDeps(kind=kind, researcher=r, runner=ToyRunner(cfg.gpus_per_arm), model=OPUS_MODEL,
                       llm_mode="mubit", record_mode=mode)

    return factory


@pytest.fixture(autouse=True)
def _bench_toy_sigma(monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.scheduler as sched
    from backend.benchmark.engine_rules import SigmaTable

    monkeypatch.setattr(sched, "_SIGMA_TABLES",
                        {**sched.sigma_tables(), "bench-toy": SigmaTable("bench-toy", ((0.0, 0.05),), "test")})


def bench_cfg(**kw: Any) -> RaceConfig:
    base: dict[str, Any] = dict(name="toy bench", target="bench-toy", arms=[ArmKind.opus, ArmKind.mubit],
                                budget_s=1.5, gpus_per_arm=2, mode="benchmark", llm="mubit", runner="modal",
                                record_mode=dict(BENCH_RECORD_MODES), run_baseline_first=True)
    # Benchmark races run every arm's default engine features (no overrides); bench-toy gets a
    # calibrated seed spread from the autouse fixture below so the judge policy can run.
    base.update(kw)
    return RaceConfig(**base)


# --------------------------------------------------------------------------- record modes


def test_benchmark_presets_keep_the_opus_arm_out_of_routing() -> None:
    for name, cfg in PRESETS.items():
        assert cfg.record_mode_for(ArmKind.opus) == "off", name
        assert ArmKind.mubit not in cfg.arms or cfg.record_mode_for(ArmKind.mubit) == "metadata", name
        assert "optuna" not in cfg.record_modes()
        assert not cfg.record  # legacy flag stays off


def test_record_mode_defaults_and_validation() -> None:
    cfg = RaceConfig(target="rl", budget_s=10, gpus_per_arm=1)
    assert cfg.record_modes() == {"opus": "off", "mubit": "off"}
    legacy = RaceConfig(target="rl", budget_s=10, gpus_per_arm=1, record=True,
                        record_mode={ArmKind.opus: "off"})
    assert legacy.record_modes() == {"opus": "off", "mubit": "full"}
    with pytest.raises(ValueError):
        RaceConfig(target="rl", budget_s=10, gpus_per_arm=1, record_mode={ArmKind.optuna: "off"})
    with pytest.raises(ValueError):
        RaceConfig(target="rl", arms=[ArmKind.opus], budget_s=10, gpus_per_arm=1,
                   record_mode={ArmKind.mubit: "metadata"})
    with pytest.raises(ValueError):
        RaceConfig(target="rl", budget_s=10, gpus_per_arm=1, record_mode={ArmKind.opus: "bogus"})  # type: ignore[dict-item]
    assert [normalize_record(x) for x in (False, True, None, "metadata")] == ["off", "full", "off", "metadata"]


def test_default_arm_deps_wires_the_record_mode_per_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app import default_arm_deps

    monkeypatch.setenv("MUBIT_API_KEY", "test-key")
    cfg = RaceConfig(target="bench-toy", arms=[ArmKind.opus, ArmKind.mubit, ArmKind.optuna], budget_s=5,
                     gpus_per_arm=1, llm="mubit", runner="local", record_mode=dict(BENCH_RECORD_MODES))
    o = default_arm_deps("r", cfg, TOY, ArmKind.opus)
    m = default_arm_deps("r", cfg, TOY, ArmKind.mubit)
    opt = default_arm_deps("r", cfg, TOY, ArmKind.optuna)
    assert isinstance(o.researcher.llm, MubitClient) and isinstance(m.researcher.llm, MubitClient)
    assert o.researcher.record is False and o.record_mode == "off"
    assert m.researcher.record == "metadata" and m.record_mode == "metadata"
    assert opt.record_mode is None


def _gateway(responses: list[httpx2.Response]) -> tuple[MubitClient, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        seen.append(req)
        return responses.pop(0)

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return MubitClient(api_key="k", http_client=http), seen


def _reply(payload: Any | None, raw: str | None = None) -> httpx2.Response:
    body = {"id": "msg", "type": "message", "role": "assistant", "model": OPUS_MODEL,
            "content": [{"type": "text", "text": raw if raw is not None else json.dumps(payload)}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 10}}
    return httpx2.Response(200, json=body, headers={REQUEST_ID_HEADER: "mreq_t"})


@pytest.mark.parametrize(("record", "header"), [(False, "off"), ("metadata", "metadata"), (True, None)])
async def test_researcher_record_mode_reaches_the_wire_on_every_attempt(record: Any, header: str | None) -> None:
    """Attempt, retry and Opus fallback all send the arm's x-middleware-record value."""
    good = {"failure_class": "divergence", "cause": "lr too high.", "fix": "lower lr."}
    client, seen = _gateway([_reply(None, raw="not json"), _reply(None, raw="{}"), _reply(good)])
    r = LLMResearcher(kind=ArmKind.mubit, llm=client, spec=TOY, session_id="s", prompts=DefaultPrompts(),
                      fallback_model=OPUS_MODEL, record=record)
    snap = ExperimentSnapshot(id="e1", config=TOY.default_config(), status=ExperimentStatus.failed,
                              error="loss is NaN")
    out = await r.diagnose(snap)
    assert out is not None and len(seen) == 3
    assert [q.headers.get(RECORD_HEADER) for q in seen] == [header] * 3
    await client.aclose()


def test_benchmark_race_refuses_a_researcher_with_the_wrong_record_mode() -> None:
    cfg = bench_cfg()
    made: dict[ArmKind, Any] = {}
    deps = [toy_factory(made)("r", cfg, TOY, k) for k in cfg.arms]
    deps[1].researcher.record = False   # mubit arm should send "metadata"
    deps[1].record_mode = None
    with pytest.raises(BenchmarkIntegrityError):
        RaceEngine(cfg, deps, spec=TOY)
    deps = [toy_factory(made)("r", cfg, TOY, k) for k in cfg.arms]
    deps[0].record_mode = "metadata"    # label says one thing, researcher sends another
    with pytest.raises(BenchmarkIntegrityError):
        RaceEngine(cfg, deps, spec=TOY)
    RaceEngine(cfg, [toy_factory(made)("r", cfg, TOY, k) for k in cfg.arms], spec=TOY)  # consistent: ok


# --------------------------------------------------------------------------- goals


def test_load_goals(tmp_path: Path) -> None:
    assert bench.load_goals(tmp_path / "missing.json") == {}
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"lm": 2.5, "cifar": {"target_value": 0.93, "note": "x"}, "rl": None}), encoding="utf-8")
    assert bench.load_goals(p) == {"lm": 2.5, "cifar": 0.93}


def test_resolve_goal() -> None:
    g = bench.resolve_goal("lm", {"lm": 2.5}, None, dry_run=False)
    assert (g.value, g.provisional, g.source) == (2.5, False, "frozen")
    assert bench.resolve_goal("lm", {"lm": 2.5}, 2.5, dry_run=False).provisional is False
    with pytest.raises(bench.Refused):  # never move a frozen goal
        bench.resolve_goal("lm", {"lm": 2.5}, 2.4, dry_run=False)
    g = bench.resolve_goal("lm", {}, 2.4, dry_run=False)
    assert (g.value, g.provisional) == (2.4, True)
    with pytest.raises(bench.Refused):
        bench.resolve_goal("lm", {}, None, dry_run=False)
    g = bench.resolve_goal("bench-toy", {}, None, dry_run=True)
    assert (g.value, g.provisional) == (TOY.target_value, True)


def test_build_races_applies_overrides_and_dry_run_caps() -> None:
    goal = bench.Goal(2.6, True, "--target-value")
    cfgs = bench.build_races("lm-30m", seeds=[1], minutes=10, slots=3, usd_cap=5.0, goal=goal)
    (c,) = cfgs
    assert (c.seed, c.budget_s, c.gpus_per_arm, c.cost_cap_usd) == (1, 600.0, 3, 5.0)
    assert c.target_value == 2.6 and c.provisional and "provisional" in c.name
    assert c.mode == "benchmark" and c.llm == "mubit" and c.runner == "modal"
    assert c.record_modes() == {"opus": "off", "mubit": "metadata"}
    assert [x.seed for x in bench.build_races("lm-30m")] == [0, 1, 2]
    (d,) = bench.build_races("rl-cpu-30m", seeds=[0], dry_run=True)
    assert (d.mode, d.llm, d.runner) == ("dev", "mock", "local")
    assert d.gpus_per_arm == bench.DRY_RUN_MAX_SLOTS and d.budget_s == bench.DRY_RUN_MAX_MINUTES * 60
    with pytest.raises(ValueError):
        bench.build_races("lm-30m", minutes=0)


def test_checks_gate_on_caps_and_guard() -> None:
    cfg = PRESETS["lm-30m"]
    res = {n: ok for n, ok, _ in bench.checks(cfg, {})}
    assert not res["target_value frozen"] and bench.gating_ok(bench.checks(cfg, {}))  # info only
    too_many = cfg.model_copy(update={"gpus_per_arm": 6})  # 12 L4 > Starter cap
    assert not {n: ok for n, ok, _ in bench.checks(too_many, {})}["fits Modal Starter caps"]
    long = cfg.model_copy(update={"budget_s": 7200.0})
    assert not bench.gating_ok(bench.checks(long, {}))  # over the preset's GPU $ guard


def test_run_refuses_without_goal_and_needs_yes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "none.json"
    assert bench.main(["run", "demo-3m", "--goals", str(missing)]) == 2
    assert "refused" in capsys.readouterr().err
    goals = tmp_path / "g.json"
    goals.write_text(json.dumps({"cifar": 0.93}), encoding="utf-8")
    assert bench.main(["run", "demo-3m", "--goals", str(goals), "--target-value", "0.9"]) == 2  # moves a frozen goal
    monkeypatch.setattr(bench, "live_checks", lambda cfg: [])  # no Modal / key lookup in tests

    def boom(*a: Any, **kw: Any) -> Any:
        raise AssertionError("a paid race must not start without --yes")

    monkeypatch.setattr(bench, "run_race", boom)
    assert bench.main(["run", "demo-3m", "--goals", str(goals), "--seed", "0"]) == 3
    assert "--yes" in capsys.readouterr().err
    assert bench.main(["run", "demo-3m", "--goals", str(missing), "--target-value", "0.9", "--seed", "0"]) == 3
    assert "PROVISIONAL" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["run", "dev-local", "--seed", "0"], ["run", "rl-cpu-30m", "--seed", "0", "--dry-run"]])
def test_mock_races_never_land_in_bench_db(argv: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    """A mock-LLM race (--dry-run, or the dev-local mock preset without it) goes to dryrun.db
    and data/bench/dryrun/, is labelled MOCK, and needs no --yes; bench.db stays real races only."""
    monkeypatch.setattr(bench, "ROOT", tmp_path)  # help text prints paths relative to ROOT
    monkeypatch.setattr(bench, "GOALS_PATH", tmp_path / "g.json")
    monkeypatch.setattr(bench, "BENCH_DB", tmp_path / "bench.db")
    monkeypatch.setattr(bench, "DRYRUN_DB", tmp_path / "dryrun.db")
    monkeypatch.setattr(bench, "DRYRUN_DIR", tmp_path / "dryrun")
    monkeypatch.setattr(bench, "live_checks", lambda cfg: pytest.fail("a mock race needs no live checks"))
    seen: dict[str, Any] = {}

    def fake_run_race(cfg: Any, db: Any, out_dir: Path, **kw: Any) -> Any:
        seen.update(cfg=cfg, out_dir=out_dir)
        raise KeyboardInterrupt  # stop before a race starts

    monkeypatch.setattr(bench, "run_race", fake_run_race)
    goals = tmp_path / "g.json"
    goals.write_text(json.dumps({"rl": -165.0}), encoding="utf-8")
    assert bench.main([*argv, "--goals", str(goals)]) == 130
    assert seen["cfg"].llm == "mock" and seen["cfg"].mode == "dev"
    assert seen["out_dir"] == tmp_path / "dryrun"
    assert (tmp_path / "dryrun.db").exists() and not (tmp_path / "bench.db").exists()
    assert "MOCK NUMBERS: never a benchmark result" in capsys.readouterr().out


# --------------------------------------------------------------------------- a full race


def test_benchmark_race_records_modes_goal_and_recomputes_metrics(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.app

    made: dict[ArmKind, Any] = {}
    monkeypatch.setattr(backend.app, "default_arm_deps", toy_factory(made))
    goal = 1.5  # reachable, unlike the spec placeholder 0.0001
    cfg = bench_cfg(target_value=goal, provisional=True)
    db = Database(tmp_path / "bench.db")
    race_id, path, status, manifest, engine_sums = asyncio.run(
        bench.run_race(cfg, db, tmp_path, progress_every_s=0.5, tick_s=0.02))
    assert status == "finished"
    assert manifest["identical"] and manifest["llm_policy_identical"]
    assert {k: v["record_mode"] for k, v in manifest["differs_only_in"].items()} == {"opus": "off", "mubit": "metadata"}
    assert manifest["goal"] == {"target_value": goal, "provisional": True, "from_config": True}
    assert set(made[ArmKind.opus].record_seen) == {False}
    assert set(made[ArmKind.mubit].record_seen) == {"metadata"}

    db.assert_benchmark_clean(race_id)
    race = db.get_race(race_id)
    assert race["config"]["target_value"] == goal and race["config"]["provisional"] is True
    assert race["config"]["record_mode"] == {"opus": "off", "mubit": "metadata"}
    assert race["fairness"]["goal"]["target_value"] == goal
    reports = {a["kind"]: a["report"] for a in db.get_arms(race_id)}
    assert reports["opus"]["record_mode"] == "off" and reports["mubit"]["record_mode"] == "metadata"
    assert all(r["target_value"] == goal for r in reports.values())

    recomputed = bench.print_summary(path, allow_dev=False)
    assert bench.summary_mismatches(engine_sums, recomputed) == []
    # The goal from the config is used, so the reachable goal is reached by someone.
    assert any(s.time_to_target_s is not None for s in recomputed.values())
    assert race_spec(cfg, TOY).target_value == goal and race_spec(RaceConfig(
        target="bench-toy", budget_s=1, gpus_per_arm=1), TOY).target_value == TOY.target_value
    db.close()


# --------------------------------------------------------------------------- regime #2 (D32)


def test_goal_key_separates_run_budgets() -> None:
    assert bench.goal_key(PRESETS["rl-cpu-30m"]) == "rl"
    assert bench.goal_key(PRESETS["rl-wide-15m"]) == "rl@15s"
    goals = {"rl": -165.0, "rl@15s": -300.0}
    g = bench.resolve_goal("rl@15s", goals, None, dry_run=False, spec_target="rl")
    assert g == bench.Goal(-300.0, False, "frozen")
    with pytest.raises(bench.Refused):
        bench.resolve_goal("rl@15s", {"rl": -165.0}, None, dry_run=False, spec_target="rl")
    assert bench.resolve_goal("rl@15s", {"rl": -165.0}, None, dry_run=True, spec_target="rl").provisional
    res = dict((n, (ok, note)) for n, ok, note in bench.checks(PRESETS["rl-wide-15m"], goals))
    assert res["target_value frozen"][0] and "rl@15s=-300" in res["target_value frozen"][1]


def test_arms_override_filters_record_modes() -> None:
    assert bench.parse_arms(None) is None
    assert bench.parse_arms("opus, optuna") == [ArmKind.opus, ArmKind.optuna]
    cfgs = bench.build_races("rl-wide-15m", seeds=[0], arms=[ArmKind.opus, ArmKind.optuna])
    c = cfgs[0]
    assert c.arms == [ArmKind.opus, ArmKind.optuna] and c.record_modes() == {"opus": "off"}
    assert c.run_budget_s == 15 and c.prewarm and c.launch_cutoff_s == PRESETS["rl-wide-15m"].launch_cutoff_s
    with pytest.raises(ValueError):
        bench.build_races("cifar-30m", arms=[ArmKind.optuna])
    dry = bench.build_races("rl-wide-15m", seeds=[0], dry_run=True)[0]
    assert not dry.prewarm and dry.runner == "local" and dry.budget_s == 120


class WarmToyRunner(ToyRunner):
    """ToyRunner with ModalRunner's pre-warm surface; logs the call order."""

    log: list[tuple[str, Any]] = []

    def __init__(self, n: int, run_s: float = 0.15, warm_ok: bool = True) -> None:
        super().__init__(n, run_s)
        self.hardware_for: dict[str, str] = {}
        self.warm_ok = warm_ok

    async def prewarm(self, hardware: str, n: int) -> None:
        self.log.append(("prewarm", (hardware, n, now())))

    async def warm_up(self, hardware: str, n: int, **kw: Any) -> dict[str, Any]:
        self.log.append(("warm_up", (hardware, n, now())))
        warm = n if self.warm_ok else n - 1
        return {"hardware": hardware, "wanted": n, "warm": warm, "ok": self.warm_ok, "rounds": [{}], "seconds": 0.0}

    async def cooldown(self, hardware: str, *, reset_scaledown: bool = False) -> None:
        self.log.append(("cooldown", (hardware, reset_scaledown, now())))

    async def launch(self, *a: Any, **kw: Any) -> RunHandle:
        self.log.append(("launch", now()))
        return await super().launch(*a, **kw)


def warm_factory(warm_ok: bool = True) -> Any:
    inner = toy_factory({})

    def factory(race_id: str, cfg: RaceConfig, spec: TargetSpec, kind: ArmKind) -> ArmDeps:
        d = inner(race_id, cfg, spec, kind)
        d.runner = WarmToyRunner(cfg.gpus_per_arm, warm_ok=warm_ok)
        return d

    return factory


def test_bench_race_prewarms_before_t0_and_cools_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.app

    WarmToyRunner.log = []
    monkeypatch.setattr(backend.app, "default_arm_deps", warm_factory())
    cfg = bench_cfg(target_value=1.5, provisional=True, prewarm=True, prewarm_spare=1, launch_cutoff_s=0.3)
    db = Database(tmp_path / "bench.db")
    race_id, path, status, manifest, _ = asyncio.run(bench.run_race(cfg, db, tmp_path, progress_every_s=0.5,
                                                                    tick_s=0.02))
    assert status == "finished"
    kinds = [k for k, _ in WarmToyRunner.log]
    assert kinds[0] == "prewarm" and kinds[1] == "warm_up" and kinds[-1] == "cooldown"
    assert kinds.count("prewarm") == 1 and kinds.count("cooldown") == 1
    assert WarmToyRunner.log[0][1][:2] == ("L4", 2 * 2 + 1)  # every slot of every arm + spare
    warm_done = WarmToyRunner.log[1][1][2]
    race = db.get_race(race_id)
    assert race["started_at"] >= warm_done  # t0 only after the pool is warm
    assert min(t for k, t in WarmToyRunner.log if k == "launch") >= race["started_at"]
    assert manifest["prewarm"]["ok"] and manifest["prewarm"]["pinned"] == 5
    assert race["fairness"]["prewarm"]["warm"] == 5
    assert all(p["launch_cutoff_s"] == 0.3 for p in manifest["per_arm"].values())
    db.close()


def test_bench_race_refuses_when_the_pool_cannot_be_warmed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.app

    WarmToyRunner.log = []
    monkeypatch.setattr(backend.app, "default_arm_deps", warm_factory(warm_ok=False))
    cfg = bench_cfg(target_value=1.5, provisional=True, prewarm=True)
    db = Database(tmp_path / "bench.db")
    with pytest.raises(bench.Refused):
        asyncio.run(bench.run_race(cfg, db, tmp_path, progress_every_s=0.5, tick_s=0.02))
    kinds = [k for k, _ in WarmToyRunner.log]
    assert "launch" not in kinds and kinds[-1] == "cooldown"  # unpinned even when refused
    db.close()
