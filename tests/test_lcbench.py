"""lcbench target (YAHPO LCBench surrogate) and SurrogateRunner.

Needs onnxruntime (not in the lock file: run with ``uv run --with onnxruntime==1.29.0``) and the
cached surrogate files (``python -m workers.targets.data.prep_lcbench fetch``); skipped otherwise.
The runner tests that need no surrogate use a tiny in-test target.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from backend.benchmark.configs import PRESETS
from backend.core.types import ArmKind, ExperimentStatus, MetricPoint
from backend.scheduler import agent_visible_metrics
from workers.runners.base import DEFAULT_PRICE_PER_SECOND, RunnerCapacityError
from workers.runners.surrogate import SurrogateRunner
from workers.targets.base import TARGET_MODULES, BaseTarget, ParamSpec, TargetSpec, get_target, register_target
from workers.targets.lcbench import SPEC, TASK_ID, check_files

HAVE_ORT = True
try:
    import onnxruntime  # noqa: F401
except ImportError:
    HAVE_ORT = False
HAVE_FILES = all(v for k, v in check_files().items() if k != "LICENSE-2.0.txt")
needs_surrogate = pytest.mark.skipif(not (HAVE_ORT and HAVE_FILES),
                                     reason="needs onnxruntime and the cached LCBench surrogate files")


# --------------------------------------------------------------------------- spec


def test_spec_is_registered_and_neutral() -> None:
    assert TARGET_MODULES["lcbench"] == "workers.targets.lcbench"
    assert get_target("lcbench").spec is SPEC
    assert SPEC.direction == "max" and SPEC.hardware == "L4" and SPEC.trap_option == "checkpoint_selection"
    assert not SPEC.is_trap(SPEC.default_config())
    assert SPEC.is_trap({**SPEC.default_config(), "checkpoint_selection": "best_test"})
    # The LLM never sees the benchmark's or dataset's name.
    text = SPEC.description.lower() + " ".join(p.description.lower() for p in SPEC.config_space.values())
    assert "lcbench" not in text and "yahpo" not in text and TASK_ID not in text


def test_preset_lcbench_15m() -> None:
    cfg = PRESETS["lcbench-15m"]
    assert cfg.target == "lcbench" and cfg.runner == "surrogate" and cfg.gpus_per_arm == 16
    assert cfg.arms == [ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna] and cfg.budget_s == 900
    assert cfg.record_mode_for(ArmKind.opus) == "off" and cfg.record_mode_for(ArmKind.opus_bm) == "off"
    virtual_gpu = cfg.gpus_per_arm * cfg.budget_s * DEFAULT_PRICE_PER_SECOND["L4"]
    assert cfg.cost_cap_usd == pytest.approx(virtual_gpu + 4.0, abs=0.01)  # virtual GPU $ + $4 LLM


# --------------------------------------------------------------------------- surrogate


@needs_surrogate
def test_surrogate_predictions_are_plausible() -> None:
    from workers.targets.lcbench import evaluate, get_surrogate

    s = get_surrogate()
    hp = {k: float(v) for k, v in SPEC.default_config().items() if k not in ("epochs", "checkpoint_selection")}
    y = s.predict([{**hp, "epoch": float(e)} for e in (1, 10, 26, 52)])
    times = [float(r[0]) for r in y]
    assert times == sorted(times) and times[0] > 0          # cumulative time grows with epochs
    for r in y:
        assert 0 <= r[1] <= 100 and 0 <= r[3] <= 1 and 0 <= r[5] <= 1  # val_acc %, balanced accs
    assert float(y[-1][3]) > float(y[0][3])                  # it learns
    d = evaluate(SPEC.default_config(), SPEC.budget_s)
    assert d["heldout_score"] == pytest.approx(0.4369, abs=2e-3)   # calibration value (prep_lcbench)
    assert d["score"] == pytest.approx(0.4409, abs=2e-3)
    assert d["epochs_trained"] == 26 and not d["budget_hit"]


@needs_surrogate
def test_budget_truncates_epochs_and_timeout() -> None:
    from workers.targets.lcbench import evaluate

    big = {**SPEC.default_config(), "epochs": 52, "num_layers": 5, "max_units": 1024, "batch_size": 16}
    full = evaluate(big, 1e9)
    cut = evaluate(big, full["train_time_s"] / 2)
    assert cut["budget_hit"] and cut["epochs_trained"] < full["epochs_trained"]
    assert cut["train_time_s"] <= full["train_time_s"] / 2
    with pytest.raises(TimeoutError):
        evaluate(big, 0.5)


@needs_surrogate
def test_checkpoint_selection_and_trap() -> None:
    from workers.targets.lcbench import evaluate

    cfg = {**SPEC.default_config(), "epochs": 52, "learning_rate": 0.09, "max_dropout": 0.0}
    last = evaluate(cfg, 1e9)
    bv = evaluate({**cfg, "checkpoint_selection": "best_validation"}, 1e9)
    bt = evaluate({**cfg, "checkpoint_selection": "best_test"}, 1e9)
    assert bv["score"] >= last["score"]
    assert bt["score"] == bt["heldout_score"] >= bv["heldout_score"]  # selected and reported on test
    # all epochs are trained (and charged) whatever is kept
    assert last["train_time_s"] == bv["train_time_s"] == bt["train_time_s"]


@needs_surrogate
def test_ground_truth_metrics_are_hidden_from_the_agent() -> None:
    from workers.targets.lcbench import evaluate

    out = evaluate(SPEC.default_config(), SPEC.budget_s)
    visible = agent_visible_metrics({k: float(v) for k, v in out.items() if k != "score"})
    assert "heldout_score" not in visible and "heldout_cross_entropy" not in visible
    assert not any(k.startswith("test") for k in visible)
    assert "train_time_s" in visible and "val_accuracy" in visible


# --------------------------------------------------------------------------- runner


class _Toy(BaseTarget):
    """Instant target with a configurable 'predicted' runtime (no surrogate needed)."""

    spec = TargetSpec(
        name="surrogate-toy", description="toy", metric="m", direction="max", target_value=1.0, hardware="L4",
        budget_s=10, trap_option="trap",
        config_space={"t": ParamSpec(type="float", low=0.0, high=100.0, default=2.0),
                      "fail": ParamSpec(type="categorical", choices=["no", "timeout", "crash"], default="no"),
                      "trap": ParamSpec(type="bool", default=False)},
    )

    def run(self, config: dict[str, Any], seed: int, budget_s: int, emit: Any) -> dict[str, Any]:
        if config["fail"] == "timeout":
            raise TimeoutError("too slow")
        if config["fail"] == "crash":
            raise ValueError("bad")
        for step in (1, 2):
            emit(step, "train_time_s", config["t"] * step / 2)
            emit(step, "acc", 0.1 * step)
        return {"score": 0.5, "heldout_score": 0.4, "train_time_s": config["t"]}


register_target(_Toy())


def _cfg(**kw: Any) -> dict[str, Any]:
    return {**_Toy.spec.default_config(), **kw}


async def test_runner_paces_by_predicted_time_times_multiplier() -> None:
    r = SurrogateRunner(2, runtime_multiplier=0.1)
    assert [s.hardware for s in r.slots()] == ["L4", "L4"] and r.price_per_second("L4") == DEFAULT_PRICE_PER_SECOND["L4"]
    seen: list[tuple[float, MetricPoint]] = []
    t0 = time.time()
    h = await r.launch("surrogate-toy", _cfg(t=4.0), 0, 10, lambda p: seen.append((time.time() - t0, p)))
    assert h.run_ref and h.run_ref.endswith(":x0.1")
    res = await r.wait(h)
    took = time.time() - t0
    assert res.status == ExperimentStatus.succeeded and res.score == 0.5
    assert res.metrics["heldout_score"] == 0.4
    assert 0.38 <= took < 1.5                                   # 4 s x 0.1
    assert [p.name for _, p in seen] == ["train_time_s", "acc", "train_time_s", "acc"]
    assert [p.t for _, p in seen] == pytest.approx([0.2, 0.2, 0.4, 0.4])
    assert seen[0][0] >= 0.15                                   # delivered at its scaled time, not at once
    assert r.busy_slots() == 0


async def test_runner_caps_slots_and_cancels() -> None:
    r = SurrogateRunner(1, runtime_multiplier=1.0)
    h = await r.launch("surrogate-toy", _cfg(t=50.0), 0, 10, lambda p: None)
    with pytest.raises(RunnerCapacityError):
        await r.launch("surrogate-toy", _cfg(), 0, 10, lambda p: None)
    t0 = time.time()
    waiter = asyncio.create_task(r.wait(h))
    await asyncio.sleep(0.05)
    await r.cancel(h)
    res = await waiter
    assert res.status == ExperimentStatus.cancelled and time.time() - t0 < 2
    assert r.busy_slots() == 0
    await r.cancel(h)  # idempotent


async def test_runner_charges_failures() -> None:
    r = SurrogateRunner(2, runtime_multiplier=0.05)
    t0 = time.time()
    res = await r.wait(await r.launch("surrogate-toy", _cfg(fail="timeout"), 0, 10, lambda p: None))
    assert res.status == ExperimentStatus.failed and "TimeoutError" in (res.error or "")
    assert time.time() - t0 >= 0.45                              # full budget (10 s) x 0.05
    t0 = time.time()
    res = await r.wait(await r.launch("surrogate-toy", _cfg(fail="crash"), 0, 10, lambda p: None))
    assert res.status == ExperimentStatus.failed and time.time() - t0 < 0.3   # config-time failure: 0 s
    assert any("ValueError" in ln for ln in res.logs_tail)


def test_runner_multiplier_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUSY_SURROGATE_RUNTIME_MULT", "0.25")
    assert SurrogateRunner(1).runtime_multiplier == 0.25
    with pytest.raises(ValueError):
        SurrogateRunner(1, runtime_multiplier=-1)


@needs_surrogate
async def test_engine_race_on_surrogate_runner(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A short mock-LLM race through the real app wiring (runner="surrogate"), fast pacing."""
    from backend.app import default_arm_deps
    from backend.benchmark.configs import RaceConfig
    from backend.db import Database
    from backend.scheduler import RaceEngine, race_spec

    monkeypatch.setenv("BUSY_SURROGATE_RUNTIME_MULT", "0.01")
    cfg = RaceConfig(target="lcbench", arms=[ArmKind.mock, ArmKind.optuna], budget_s=4, gpus_per_arm=4,
                     runner="surrogate", llm="mock", mode="dev")
    spec = race_spec(cfg)
    deps = [default_arm_deps("r", cfg, spec, k) for k in cfg.arms]
    assert all(type(d.runner).__name__ == "SurrogateRunner" for d in deps)
    db = Database(tmp_path / "t.db")
    eng = RaceEngine(cfg, deps, spec=spec, db=db, tick_s=0.02)
    await eng.run()
    sums = {s.arm: s for s in eng.summaries}
    assert eng.status == "finished"
    for s in sums.values():
        assert s.experiments_finished >= 4 and s.best_score is not None and 0 < s.best_score <= 1
        assert s.gpu_cost_usd == pytest.approx(4 * 4 * DEFAULT_PRICE_PER_SECOND["L4"], rel=0.1)
    db.close()
