"""scripts/collect_examples.py: pre-flight refusals, probe recipes and policy, and a full
in-process collection run (fake target + fake runner + MockLLM). No network, no Modal."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import pytest

from backend.agent import prompts as prompts_pkg
from backend.agent.mock_llm import MockLLM
from backend.core.types import ExperimentStatus, MetricPoint, RunHandle, RunResult, Slot, StepType, new_id, now
from backend.db import BenchmarkIntegrityError, Database
from workers.runners.base import RunnerCapacityError, emit_metric
from workers.targets.base import BaseTarget, ParamSpec, TargetSpec, get_target, register_target

ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> Any:
    name = "collect_examples_script"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "collect_examples.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


ce = _load_script()

# =========================================================================== fakes

TRAP = "eval_rows_from_train"


class FakeCollectTarget(BaseTarget):
    spec = TargetSpec(
        name="fake-collect",
        description="Synthetic loss surface for collection tests.",
        metric="val_loss",
        direction="min",
        target_value=1.0,
        hardware="L4",
        budget_s=1,
        config_space={
            "lr": ParamSpec(type="float", low=1e-4, high=1.0, log=True, default=1e-3, description="learning rate"),
            "width": ParamSpec(type="int", low=8, high=256, default=32, description="hidden width"),
            TRAP: ParamSpec(type="bool", default=False, description="reuse rows for evaluation"),
        },
        trap_option=TRAP,
    )

    def run(self, config: dict[str, Any], seed: int, budget_s: int, emit: Any) -> dict[str, Any]:
        return {"score": 1.0}


register_target(FakeCollectTarget())
SPEC = FakeCollectTarget.spec


class FakeRunner:
    """lr > 0.3 diverges (failed, NaN log); the trap halves the reported loss."""

    name = "fake"

    def __init__(self, n_slots: int, run_s: float = 0.1, price: float = 0.000222, crash: bool = False) -> None:
        self.n_slots, self.run_s, self.price, self.crash = n_slots, run_s, price, crash
        self._tasks: dict[str, asyncio.Task[RunResult]] = {}
        self.budgets: list[int] = []

    def slots(self) -> list[Slot]:
        return [Slot(slot_id=f"fake-L4-{i}", index=i, hardware="L4", runner=self.name) for i in range(self.n_slots)]

    def price_per_second(self, hardware: str) -> float:
        return self.price

    async def launch(self, target: str, config: dict[str, Any], seed: int, budget_s: int, on_metric: Any) -> RunHandle:
        if sum(1 for t in self._tasks.values() if not t.done()) >= self.n_slots:
            raise RunnerCapacityError("busy")
        self.budgets.append(budget_s)
        h = RunHandle(runner=self.name, target=target, config=config, seed=seed, budget_s=budget_s,
                      hardware="L4", run_ref=f"fake-{new_id()[:6]}")
        self._tasks[h.run_id] = asyncio.create_task(self._train(h, on_metric))
        return h

    async def _train(self, h: RunHandle, on_metric: Any) -> RunResult:
        t0 = now()
        trap = SPEC.is_trap(h.config)
        try:
            await asyncio.sleep(self.run_s)
            await emit_metric(on_metric, MetricPoint(experiment_id=h.run_id, step=1, t=now() - t0,
                                                     name="train_loss", value=1.5))
        except asyncio.CancelledError:
            return RunResult(score=None, status=ExperimentStatus.cancelled, started_at=t0, ended_at=now(),
                             hardware="L4", is_trap=trap, error="cancelled")
        if self.crash:  # like a broken worker image: every run dies at import
            return RunResult(score=None, status=ExperimentStatus.failed, started_at=t0, ended_at=now(),
                             hardware="L4", is_trap=trap, error="worker exited with code 1 without a result",
                             logs_tail=["SyntaxError: unterminated string literal (detected at line 187)"])
        lr = float(h.config["lr"])
        if lr > 0.3:
            return RunResult(score=None, status=ExperimentStatus.failed, started_at=t0, ended_at=now(),
                             hardware="L4", is_trap=trap, error="loss is nan at step 3",
                             logs_tail=["step 1 loss 2.1", "step 3 loss nan"])
        clean = 1.0 + 0.3 * (math.log10(lr) + 2.5) ** 2 + random.Random(f"{h.seed}{h.config}").uniform(0, 0.01)
        return RunResult(score=clean * (0.5 if trap else 1.0), status=ExperimentStatus.succeeded, started_at=t0,
                         ended_at=now(), hardware="L4", is_trap=trap,
                         metrics={"train_loss": 1.2, "eval_train_overlap": 1.0 if trap else 0.0})

    async def wait(self, handle: RunHandle) -> RunResult:
        return await self._tasks[handle.run_id]

    async def cancel(self, handle: RunHandle) -> None:
        t = self._tasks.get(handle.run_id)
        if t is not None and not t.done():
            t.cancel()


def _freeze_all(monkeypatch: pytest.MonkeyPatch) -> None:
    for step in StepType:
        monkeypatch.setitem(prompts_pkg.FROZEN, step, prompts_pkg.CURRENT_VERSION[step])


# =========================================================================== pre-flight


def test_session_id_format() -> None:
    assert ce.session_id("2026-10-24", "cifar", 2) == "collect-2026-10-24/cifar/seed-2"


def test_record_refused_while_prompts_are_drafts(capsys: pytest.CaptureFixture[str],
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    for step in StepType:
        monkeypatch.setitem(prompts_pkg.FROZEN, step, None)

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("no client may be built before the pre-flight passes")

    monkeypatch.setattr(ce, "default_llm_factory", boom)
    monkeypatch.setattr(ce, "Collector", boom)
    assert ce.record_problems()  # drafts
    rc = ce.main(["--target", "rl", "--runner", "local", "--minutes", "1", "--yes"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "not FROZEN" in err and "propose_experiments" in err and "judge_improvement" in err


def test_record_allowed_only_when_frozen_version_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    _freeze_all(monkeypatch)
    assert ce.record_problems() == []
    assert all(v["ok"] for v in ce.prompt_freeze_status().values())
    monkeypatch.setitem(prompts_pkg.FROZEN, StepType.diagnose_run, "v-old")
    probs = ce.record_problems()
    assert len(probs) == 1 and "diagnose_run" in probs[0]


def test_paid_run_needs_yes_and_opus(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    _freeze_all(monkeypatch)
    monkeypatch.setattr(ce, "Collector", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("built")))
    assert ce.main(["--target", "rl", "--runner", "local"]) == 2
    assert "--yes" in capsys.readouterr().err
    assert ce.main(["--target", "rl", "--runner", "local", "--yes", "--model", "claude-haiku-5-5"]) == 2
    assert "different Mubit task" in capsys.readouterr().err


def test_preflight_guards(capsys: pytest.CaptureFixture[str]) -> None:
    assert ce.main(["--dry-run", "--runner", "modal"]) == 2
    assert "never uses Modal" in capsys.readouterr().err
    assert ce.main(["--dry-run", "--target", "cifar", "--runner", "local", "--slots", "0"]) == 2
    ns = ce.build_parser().parse_args(["--target", "cifar", "--runner", "modal", "--slots", "12", "--no-record"])
    errs = ce.preflight(ns, ce.args_from_namespace(ns))
    assert any("10 concurrent GPUs" in e for e in errs)
    assert ce.main(["--target", "nope", "--dry-run"]) == 2


def test_dry_run_defaults() -> None:
    a = ce.args_from_namespace(ce.build_parser().parse_args(["--dry-run"]))
    assert a.dry_run and not a.record and a.runner == "local" and a.minutes == 2.0
    b = ce.args_from_namespace(ce.build_parser().parse_args([]))
    assert b.record and b.runner == "modal" and b.model == "claude-opus-5-5"


# =========================================================================== probes


@pytest.mark.parametrize("target", ["rl", "cifar", "lm"])
def test_probe_recipes_are_valid_for_real_targets(target: str) -> None:
    spec = get_target(target).spec
    rng = random.Random(0)
    for kind in ce.PROBE_KINDS:
        built = ce.probe_config(spec, kind, spec.default_config(), rng, set())
        assert built is not None, (target, kind)
        cfg, overrides = built
        assert spec.validate_config(cfg) == cfg
        assert spec.is_trap(cfg) == (kind == "trap")
        assert all(cfg[k] == v for k, v in overrides.items())


def test_probe_config_avoids_repeats_and_keeps_trap_off() -> None:
    rng = random.Random(1)
    base = {**SPEC.default_config(), TRAP: True}
    cfg, _ = ce.probe_config(SPEC, "diverge", base, rng, set())
    assert cfg["lr"] == 1.0 and cfg[TRAP] is False  # generic recipe: max lr, trap off
    again = ce.probe_config(SPEC, "diverge", base, rng, {ce.canonical(cfg)})
    assert again is not None and ce.canonical(again[0]) != ce.canonical(cfg) and again[0]["lr"] == 1.0
    assert ce.probe_config(SPEC, "overfit", base, rng, set()) is None  # no recipe for this target
    trap_cfg, _ = ce.probe_config(SPEC, "trap", SPEC.default_config(), rng, set())
    assert SPEC.is_trap(trap_cfg)


def test_coverage_policy() -> None:
    pol = ce.CoveragePolicy(goal=10, ratio=0.5, rate=0.25, kinds=ce.PROBE_KINDS)
    P, D, J = StepType.propose_experiments, StepType.diagnose_run, StepType.judge_improvement
    ex = {P: 8, D: 0, J: 3}
    # diagnose lags most (0/8), then judge (3/8 < 0.5)
    assert pol.choose(examples=ex, inflight={}, launched=20, probes=0, k=2) == ["diverge", "trap", "overfit"]
    # judge kinds are balanced 2 trap : 1 overfit (trap wins ties)
    pol.kind_counts["overfit"] += 1
    assert pol.choose(examples={P: 8, D: 8, J: 0}, inflight={}, launched=20, probes=0, k=1) == ["trap", "overfit"]
    # rate cap: 5 probes of 20 launched (+1 batch) is the limit at 25 %
    assert pol.choose(examples=ex, inflight={}, launched=19, probes=5, k=1) == []
    # in-flight probes count toward coverage; goals stop probing
    assert pol.choose(examples=ex, inflight={D: 4}, launched=20, probes=0, k=1)[0] == "trap"
    assert pol.choose(examples={P: 100, D: 10, J: 10}, inflight={}, launched=200, probes=0, k=4) == []
    assert pol.choose(examples={P: 0, D: 0, J: 0}, inflight={}, launched=0, probes=0, k=4) == []  # warm-up
    only_trap = ce.CoveragePolicy(goal=10, ratio=0.5, rate=1.0, kinds=("trap",))
    assert only_trap.choose(examples=ex, inflight={}, launched=4, probes=0, k=1) == ["trap"]


def test_policy_backs_off_probe_kinds_that_miss() -> None:
    pol = ce.CoveragePolicy(goal=50, ratio=1.0, rate=1.0, kinds=ce.PROBE_KINDS)
    P, D, J = StepType.propose_experiments, StepType.diagnose_run, StepType.judge_improvement
    ex = {P: 10, D: 0, J: 0}
    assert "diverge" in pol.choose(examples=ex, inflight={}, launched=10, probes=0, k=1)
    assert [pol.record_outcome("diverge", False) for _ in range(3)] == [False, False, True]
    assert "diverge" in pol.disabled
    assert "diverge" not in pol.choose(examples=ex, inflight={}, launched=10, probes=0, k=1)
    for hit in (True, False, False, True):  # 2/4 hits: stays on
        assert pol.record_outcome("trap", hit) is False
    assert "trap" not in pol.disabled


async def test_budget_override_runner() -> None:
    inner = FakeRunner(2, run_s=0.01)
    r = ce.BudgetOverrideRunner(inner)
    cfg = SPEC.default_config()
    r.override(cfg, 7)
    h1 = await r.launch("fake-collect", dict(cfg), 0, 30, lambda _p: None)
    h2 = await r.launch("fake-collect", dict(cfg), 0, 30, lambda _p: None)
    assert inner.budgets == [7, 30] and r.name == "fake" and len(r.slots()) == 2
    await r.wait(h1)
    await r.wait(h2)


# =========================================================================== full run


def test_collection_run_end_to_end(tmp_path: Path) -> None:
    db = Database(":memory:")
    runners: list[FakeRunner] = []

    def runner_factory(_a: Any, _s: TargetSpec) -> FakeRunner:
        runners.append(FakeRunner(2, run_s=0.12))
        return runners[-1]

    args = ce.CollectArgs(target="fake-collect", minutes=0.1, race_minutes=0.05, slots=2, usd_cap=5.0,
                          runner="local", seeds=(0, 1), dry_run=True, record=False, run_budget_s=1,
                          log_every_min=0.0, goal=1000, coverage_ratio=1.0, probe_rate=0.5,
                          probe_kinds=("diverge", "trap"), out_dir=str(tmp_path), storm_n=10_000)
    out: list[str] = []

    class _Out:
        def write(self, s: str) -> int:
            out.append(s)
            return len(s)

        def flush(self) -> None:
            return None

    col = ce.Collector(args, spec=SPEC, db=db, runner_factory=runner_factory, stdout=_Out(),  # type: ignore[arg-type]
                       llm_factory=lambda _a, spec, seed: MockLLM(spec, seed=seed, trap_rate=0.0))
    final = asyncio.run(col.run())

    # progress JSONL: start, two races, periodic progress, done
    events = [json.loads(ln) for ln in col.jsonl_path.read_text(encoding="utf-8").splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert kinds.count("race_start") == 2 and kinds.count("race_end") == 2 and "progress" in kinds
    last = events[-1]
    for step in ce.STEPS:   # the three race steps (early_stop has its own collector, D34)
        s = last["steps"][step.value]
        assert s["calls"] >= 1 and s["valid_rate"] is not None and "examples" in s
    assert last["est_usd"]["llm"] == 0.0 and last["est_usd"]["gpu"] > 0.0  # fake L4 price, mock LLM
    assert "est $ LLM" in "".join(out)
    assert final["stop"] == "minutes"

    # races: dev only (never benchmark results), one session per seed
    races = col.store.races()
    assert [r["session_id"] for r in races] == [f"collect-{col.date}/fake-collect/seed-0",
                                                f"collect-{col.date}/fake-collect/seed-1"]
    for r in races:
        assert db.get_race(r["race_id"])["is_dev"] is True
        with pytest.raises(BenchmarkIntegrityError):
            db.assert_benchmark_clean(r["race_id"])
        calls = db.list_llm_calls(race_id=r["race_id"])
        assert calls and all(c["session_id"] == r["session_id"] for c in calls)
        assert not any(c["recorded"] for c in calls)
        assert (tmp_path / "races" / f"{r['race_id']}.jsonl").exists()

    # probes flagged in the side table, every one a real experiment row; neutral hypotheses
    probes = col.store.probes()
    assert probes and {p["kind"] for p in probes} <= {"diverge", "trap"}
    for p in probes:
        exp = db.get_experiment(p["experiment_id"])
        assert exp is not None
        assert "probe" not in exp["hypothesis"].lower() and "trap" not in exp["hypothesis"].lower()
        if p["kind"] == "trap":
            assert exp["config"][TRAP] is True
        if p["kind"] == "diverge" and p["status"] is not None:
            assert p["status"] == "failed" and p["diagnosed"] == 1
    assert final["probes"]["total"] == len(probes)
    assert final["probes"]["total"] <= 0.5 * final["launched"] + 1

    # coverage: both rare steps were exercised
    assert col.tracker.steps[StepType.diagnose_run].calls >= 1
    assert col.tracker.steps[StepType.judge_improvement].calls >= 1
    assert len(runners) == 2  # a fresh runner per race


def test_collection_stops_at_usd_cap(tmp_path: Path) -> None:
    # 2 slots x $1/s: the cap buys ~2 s per race, below the 2 x run budget minimum -> no race
    args = ce.CollectArgs(target="fake-collect", minutes=1.0, slots=2, usd_cap=4.0, runner="local", seeds=(0,),
                          dry_run=True, record=False, run_budget_s=1, log_every_min=1.0, out_dir=str(tmp_path))
    col = ce.Collector(args, spec=SPEC, db=Database(":memory:"), stdout=io.StringIO(),
                       runner_factory=lambda _a, _s: FakeRunner(2, price=1.5),
                       llm_factory=lambda _a, spec, seed: MockLLM(spec, seed=seed))
    final = asyncio.run(col.run())
    assert final["stop"] == "usd_cap" and final["races"] == 0


def test_failure_storm_stops_the_collection(tmp_path: Path) -> None:
    args = ce.CollectArgs(target="fake-collect", minutes=0.2, slots=2, usd_cap=5.0, runner="local", seeds=(0, 1),
                          dry_run=True, record=False, run_budget_s=1, log_every_min=1.0, out_dir=str(tmp_path),
                          storm_n=4)
    col = ce.Collector(args, spec=SPEC, db=Database(":memory:"), stdout=io.StringIO(),
                       runner_factory=lambda _a, _s: FakeRunner(2, run_s=0.02, crash=True),
                       llm_factory=lambda _a, spec, seed: MockLLM(spec, seed=seed))
    t0 = now()
    final = asyncio.run(col.run())
    assert final["stop"] == "failure_storm" and final["races"] == 1
    assert now() - t0 < 6.0  # stopped early, not at the 12 s deadline
    assert col.tracker.steps[StepType.diagnose_run].calls <= args.storm_n + 2  # bounded spend
    assert any("failure storm" in n for n in final["notes"])
