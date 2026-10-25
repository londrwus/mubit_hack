"""Demo preset 'demo-cifar-modal-5m' and per-arm Modal apps (offline, fakes only: no Modal call).

- per-arm routing: each arm's ModalRunner looks its functions up in its own deployed app;
- the default single-app path is unchanged (same constructor kwargs, app, function names, calls);
- benchmark / ablation presets cannot enable per-arm apps, and their stored config has no new keys;
- the demo preset is dev mode, labelled, and refused by scripts/headline.py.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import backend.app as app_mod
from backend.benchmark.configs import DEMO_LABEL, PER_ARM_MODAL_APPS, PRESETS, RaceConfig
from backend.core.types import ArmKind
from workers.targets.base import get_target

modal_runner = pytest.importorskip("workers.runners.modal_runner")
ModalRunner = modal_runner.ModalRunner

ROOT = Path(__file__).resolve().parents[1]
DEMO = "demo-cifar-modal-5m"
LIVE_TEST = "demo-live-10s"   # the "Get started" live test: a demo label, no per-arm apps
LIVE_DEMO = "demo-live-60s"   # demo label, no per-arm apps (tests/test_demo_live.py)


def _load(name: str, rel: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bench = _load("bench_demo_t", "scripts/bench.py")
hl = _load("headline_demo_t", "scripts/headline.py")


class _FakeFn:
    def __init__(self, app: str, name: str, log: list[Any]) -> None:
        self.app, self.name, self.log = app, name, log

    @property
    def update_autoscaler(self) -> Any:
        fn = self

        class _U:
            async def aio(self, **kw: Any) -> None:
                fn.log.append(("update_autoscaler", fn.app, fn.name, kw))

        return _U()


@pytest.fixture
def from_name(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Replace modal.Function.from_name (a name lookup) with a recorder; never touches Modal."""
    log: list[Any] = []

    def fake(app: str, name: str, *a: Any, **kw: Any) -> _FakeFn:
        log.append(("from_name", app, name))
        return _FakeFn(app, name, log)

    monkeypatch.setattr(modal_runner.modal.Function, "from_name", staticmethod(fake))
    return log


def _recording_construct(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    real = app_mod._construct

    def rec(cls: Any, **kw: Any) -> Any:
        calls.append(dict(kw))
        return real(cls, **kw)

    monkeypatch.setattr(app_mod, "_construct", rec)
    return calls


# --------------------------------------------------------------------------- routing


def test_per_arm_runner_routes_to_its_own_app(monkeypatch: pytest.MonkeyPatch, from_name: list[Any]) -> None:
    cfg = PRESETS[DEMO]
    spec = get_target(cfg.target).spec
    calls = _recording_construct(monkeypatch)
    runners = {k: app_mod._make_runner(cfg, spec, k) for k in cfg.arms}
    assert calls == [{"slots": {"L4": 3}, "app_name": "busy-machines-opus", "gpu_telemetry_s": 1.0},
                     {"slots": {"L4": 3}, "app_name": "busy-machines-bm", "gpu_telemetry_s": 1.0}]
    assert runners[ArmKind.opus].gpu_telemetry_s == runners[ArmKind.opus_bm].gpu_telemetry_s == 1.0
    assert runners[ArmKind.opus].app_name == "busy-machines-opus"
    assert runners[ArmKind.opus_bm].app_name == "busy-machines-bm"
    runners[ArmKind.opus]._function("L4")
    runners[ArmKind.opus_bm]._function("L4")
    runners[ArmKind.opus_bm]._resolve("prepare_data")
    assert from_name == [("from_name", "busy-machines-opus", "run_l4"), ("from_name", "busy-machines-bm", "run_l4"),
                         ("from_name", "busy-machines-bm", "prepare_data")]


def test_default_single_app_path_unchanged(monkeypatch: pytest.MonkeyPatch, from_name: list[Any]) -> None:
    for name in ("cifar-15m", "cifar-bm-15m", "rl-wide-15m"):
        cfg = PRESETS[name]
        spec = get_target(cfg.target).spec
        calls = _recording_construct(monkeypatch)
        runners = [app_mod._make_runner(cfg, spec, k) for k in cfg.arms]
        # exactly the old constructor call: slots only, no app_name
        assert calls == [{"slots": {spec.hardware: cfg.gpus_per_arm}}] * len(cfg.arms), name
        assert all(r.app_name == modal_runner.APP_NAME for r in runners)
        assert app_mod._make_runner(cfg, spec).app_name == modal_runner.APP_NAME  # old 2-arg call still works
    assert modal_runner.DEFAULT_APP_NAME == "busy-machines-runner"
    assert modal_runner.HARDWARE_FUNCTIONS == {"L4": "run_l4", "CPU": "run_cpu"}
    from_name.clear()
    r = ModalRunner({"L4": 2})
    r._function("L4")
    r._resolve("prepare_data")
    asyncio.run(r.prewarm("L4", 2))
    asyncio.run(r.cooldown("L4", reset_scaledown=True))
    assert from_name == [
        ("from_name", modal_runner.APP_NAME, "run_l4"), ("from_name", modal_runner.APP_NAME, "prepare_data"),
        ("update_autoscaler", modal_runner.APP_NAME, "run_l4", {"min_containers": 2}),
        ("update_autoscaler", modal_runner.APP_NAME, "run_l4",
         {"min_containers": 0, "scaledown_window": modal_runner.SCALEDOWN_WINDOW_S}),
    ]


def test_configs_mirror_the_modal_runner_app_names() -> None:
    assert {k.value: v for k, v in PER_ARM_MODAL_APPS.items()} == modal_runner.PER_ARM_APP_NAMES
    assert modal_runner.DEFAULT_APP_NAME not in modal_runner.PER_ARM_APP_NAMES.values()


# --------------------------------------------------------------------------- pre-warm


class _WarmRunner:
    def __init__(self, label: str, log: list[Any]) -> None:
        self.label, self.log = label, log
        self.hardware_for: dict[str, str] = {}

    async def prewarm(self, hw: str, n: int) -> None:
        self.log.append(("prewarm", self.label, hw, n))

    async def warm_up(self, hw: str, n: int, **kw: Any) -> dict[str, Any]:
        self.log.append(("warm_up", self.label, hw, n, kw.get("target")))
        return {"hardware": hw, "wanted": n, "warm": n, "ok": True, "rounds": [{"round": 0}], "seconds": 1.0}

    async def cooldown(self, hw: str, *, reset_scaledown: bool = False) -> None:
        self.log.append(("cooldown", self.label, hw, reset_scaledown))

    async def drain_background(self) -> None:
        self.log.append(("drain", self.label))


def _deps(cfg: RaceConfig, log: list[Any]) -> list[Any]:
    return [SimpleNamespace(kind=k, runner=_WarmRunner(k.value, log)) for k in cfg.arms]


def test_prewarm_per_arm_on_each_app() -> None:
    cfg = PRESETS[DEMO]
    log: list[Any] = []
    deps = _deps(cfg, log)
    res = asyncio.run(bench.prewarm_pool(cfg, deps, log=lambda m: None))
    assert res is not None and res["ok"] and res["warm"] == res["wanted"] == 6
    assert set(res["per_arm"]) == {"opus", "opus_bm"}
    assert res["per_arm"]["opus"]["app"] == "busy-machines-opus" and res["per_arm"]["opus_bm"]["app"] == "busy-machines-bm"
    assert sorted(e for e in log if e[0] == "prewarm") == [("prewarm", "opus", "L4", 3), ("prewarm", "opus_bm", "L4", 3)]
    assert sorted(e for e in log if e[0] == "warm_up") == [("warm_up", "opus", "L4", 3, "cifar"),
                                                           ("warm_up", "opus_bm", "L4", 3, "cifar")]
    log.clear()
    asyncio.run(bench.cooldown_pool(cfg, deps))
    assert log == [("cooldown", "opus", "L4", True), ("cooldown", "opus_bm", "L4", True),
                   ("drain", "opus"), ("drain", "opus_bm")]


def test_prewarm_default_path_unchanged() -> None:
    cfg = PRESETS["cifar-bm-15m"]
    log: list[Any] = []
    deps = _deps(cfg, log)
    res = asyncio.run(bench.prewarm_pool(cfg, deps, log=lambda m: None))
    # one shared pool on the first arm's runner (the single app): 3 arms x 3 slots
    assert [e[:4] for e in log] == [("prewarm", "opus", "L4", 9), ("warm_up", "opus", "L4", 9)]
    assert res is not None and "per_arm" not in res and res["pinned"] == 9
    log.clear()
    asyncio.run(bench.cooldown_pool(cfg, deps))
    assert log == [("cooldown", "opus", "L4", True), ("drain", "opus"), ("drain", "opus_bm"), ("drain", "optuna")]


# --------------------------------------------------------------------------- presets and config


def test_benchmark_presets_cannot_enable_per_arm_apps() -> None:
    for name, p in PRESETS.items():
        if name in (DEMO, LIVE_DEMO, LIVE_TEST):
            continue
        assert not p.modal_apps and p.demo_label is None, name
        if not p.strict:
            continue
        apps = {k: f"busy-machines-{k.value.replace('_', '-')}" for k in p.arms}
        with pytest.raises(ValueError, match="per-arm Modal apps|demo"):
            RaceConfig.model_validate({**p.model_dump(), "modal_apps": apps, "demo_label": DEMO_LABEL})
        with pytest.raises(ValueError, match="per-arm Modal apps"):
            RaceConfig.model_validate({**p.model_dump(), "modal_apps": apps})
        with pytest.raises(ValueError, match="dev mode only"):
            RaceConfig.model_validate({**p.model_dump(), "demo_label": DEMO_LABEL})
    # dev races without a demo label cannot either
    dev = PRESETS["lcbench-lean-15m"]
    with pytest.raises(ValueError, match="demo_label"):
        RaceConfig.model_validate({**dev.model_dump(), "modal_apps": {k: f"app-{k.value}" for k in dev.arms}})


def test_stored_config_of_other_presets_has_no_new_keys() -> None:
    for name, p in PRESETS.items():
        dumped = p.model_dump(mode="json")
        if name == DEMO:
            assert dumped["modal_apps"] == {"opus": "busy-machines-opus", "opus_bm": "busy-machines-bm"}
            assert dumped["demo_label"] == DEMO_LABEL
            continue
        if name in (LIVE_DEMO, LIVE_TEST):  # a demo label, no per-arm apps
            assert "modal_apps" not in dumped and dumped["demo_label"] == p.demo_label, name
            assert "gpu_telemetry_s" not in dumped, name
            continue
        assert "modal_apps" not in dumped and "demo_label" not in dumped, name
        assert "modal_apps" not in p.model_dump_json() and "demo_label" not in p.model_dump_json(), name
        assert "gpu_telemetry_s" not in dumped and "gpu_telemetry_s" not in p.model_dump_json(), name
        assert set(dumped) == set(RaceConfig.model_fields) - {"modal_apps", "demo_label", "gpu_telemetry_s"}, name
        for c in bench.build_races(name, seeds=[0]):
            assert "modal_apps" not in c.model_dump(mode="json") and not c.is_demo, name


def test_modal_apps_validation() -> None:
    base = PRESETS[DEMO].model_dump()
    with pytest.raises(ValueError, match="every arm"):
        RaceConfig.model_validate({**base, "modal_apps": {"opus": "busy-machines-opus"}})
    with pytest.raises(ValueError, match="its own app"):
        RaceConfig.model_validate({**base, "modal_apps": {"opus": "same", "opus_bm": "same"}})
    with pytest.raises(ValueError, match="invalid Modal app name"):
        RaceConfig.model_validate({**base, "modal_apps": {"opus": "Bad Name", "opus_bm": "ok"}})


def test_demo_preset_shape_and_estimate() -> None:
    cfg = PRESETS[DEMO]
    assert cfg.mode == "dev" and cfg.is_dev and cfg.is_demo and not cfg.strict
    assert cfg.arms == [ArmKind.opus, ArmKind.opus_bm] and cfg.target == "cifar" and cfg.runner == "modal"
    assert cfg.gpus_per_arm == 3 and cfg.budget_s == 300 and cfg.prewarm and cfg.llm == "mubit"
    assert cfg.record_modes() == {"opus": "off", "opus_bm": "off"}
    assert cfg.modal_app_for(ArmKind.opus) == "busy-machines-opus"
    assert cfg.modal_app_for("opus_bm") == "busy-machines-bm"
    est = bench.estimate_gpu_cost_usd(cfg, bench.price_per_s("L4"))
    assert est == pytest.approx(6 * 300 * 0.000222)
    assert est <= (cfg.max_estimated_gpu_usd or 0)
    assert bench.demo_warm_allowance_usd(cfg, 0.000222) == pytest.approx(6 * (300 + bench.MODAL_SCALEDOWN_S) * 0.000222)
    res = {n: (ok, note) for n, ok, note in bench.checks(cfg, {"cifar": 0.94})}
    assert bench.gating_ok([(n, ok, note) for n, (ok, note) in res.items()])
    assert DEMO_LABEL in res["demo (never a headline race)"][1]
    assert "busy-machines-opus" in res["Modal app per arm"][1]
    races = bench.build_races(DEMO)
    assert [c.seed for c in races] == [0] and "[demo run]" in races[0].name
    dry = bench.build_races(DEMO, dry_run=True)[0]
    assert dry.runner == "local" and dry.llm == "mock" and not dry.prewarm and dry.is_demo
    sub = bench.build_races(DEMO, arms=[ArmKind.opus, ArmKind.opus_bm])[0]
    assert sub.modal_apps == cfg.modal_apps


def test_arm_app_deploy_and_stop_commands() -> None:
    dep = modal_runner.arm_app_commands("deploy")
    assert dep == [({"BUSY_MODAL_APP": "busy-machines-opus"},
                    ["deploy", "--name", "busy-machines-opus", "workers/runners/modal_runner.py"]),
                   ({"BUSY_MODAL_APP": "busy-machines-bm"},
                    ["deploy", "--name", "busy-machines-bm", "workers/runners/modal_runner.py"])]
    assert modal_runner.arm_app_commands("stop") == [({}, ["app", "stop", "-y", "busy-machines-opus"]),
                                                     ({}, ["app", "stop", "-y", "busy-machines-bm"])]
    text = modal_runner.arm_apps_text()
    assert "busy-machines-opus" in text and "busy-machines-bm" in text and "app stop -y" in text
    with pytest.raises(ValueError):
        modal_runner.arm_app_commands("delete")


# --------------------------------------------------------------------------- headline refusal


def test_headline_refuses_the_demo_preset() -> None:
    with pytest.raises(SystemExit, match="demo run for the video, not a benchmark result"):
        hl.refuse_ablation_preset(DEMO)
    with pytest.raises(SystemExit, match="demo run for the video"):
        hl.run(preset=DEMO)
    hl.refuse_ablation_preset("cifar-15m")  # benchmark presets pass


def test_headline_drops_demo_races() -> None:
    cfg = PRESETS[DEMO].model_dump(mode="json")
    rows = {"races": [{"id": "abcdef0123", "name": "x seed 0 [demo run]", "config": cfg, "status": "finished",
                       "fairness": {"demo": {"headline": False, "label": DEMO_LABEL}}}], "arms": []}
    assert hl.is_demo_race(rows)
    assert any("demo run for the video" in w for w in hl.race_blockers(rows, False))
    plain = {"races": [{"id": "1234567890", "name": "CIFAR speedrun 15 min seed 0",
                        "config": PRESETS["cifar-15m"].model_dump(mode="json"), "fairness": {}}], "arms": []}
    assert not hl.is_demo_race(plain)
    assert not any("demo" in w for w in hl.race_blockers(plain, False))
