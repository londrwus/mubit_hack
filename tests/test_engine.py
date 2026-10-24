"""Race engine tests: fairness, idle accounting, step gating, trap handling, metrics.

Everything here is in-process and fake: a synthetic target, a runner that "trains" with
asyncio.sleep, and an LLM with configurable latency. No network, no GPU, no Modal.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
from collections import Counter
from typing import Any

import pytest

from backend.agent.loop import DefaultPrompts, LLMResearcher
from backend.agent.optuna_arm import OptunaResearcher
from backend.benchmark.analysis import check_benchmark_rows, summaries_from_rows
from backend.benchmark.configs import PRESETS, RaceConfig, estimate_gpu_cost_usd, races_for_preset
from backend.benchmark.export import parse_jsonl
from backend.benchmark.metrics import summarize_arm
from backend.core.events import parse_event
from backend.core.types import (
    ArmKind,
    CorrectionLabel,
    ExperimentStatus,
    LLMCallRecord,
    MetricPoint,
    RunHandle,
    RunResult,
    Slot,
    StepType,
    new_id,
    now,
)
from backend.db import BenchmarkIntegrityError, Database
from backend.scheduler import (
    ArmDeps,
    AsymmetricRaceError,
    EventBus,
    ExperimentSnapshot,
    Proposal,
    RaceEngine,
    ResearchDigest,
    summarize_result,
)
from workers.runners.base import RunnerCapacityError, emit_metric
from workers.targets.base import BaseTarget, ParamSpec, TargetSpec, register_target

# =========================================================================== fakes

TRAP = "val_from_train_pool"
PRICE_L4 = 0.000222


class FakeQuadTarget(BaseTarget):
    spec = TargetSpec(
        name="fake-quad",
        description="Synthetic loss surface for scheduler tests.",
        metric="val_loss",
        direction="min",
        target_value=1.06,
        hardware="L4",
        budget_s=1,
        config_space={
            "lr": ParamSpec(type="float", low=1e-4, high=1.0, log=True, default=1e-3, description="learning rate"),
            "width": ParamSpec(type="int", low=8, high=256, default=32, description="hidden width"),
            "opt": ParamSpec(type="categorical", choices=["sgd", "adam"], default="sgd", description="optimizer"),
            TRAP: ParamSpec(type="bool", default=False, description="draw validation rows from the train pool"),
        },
        trap_option=TRAP,
    )

    def run(self, config: dict[str, Any], seed: int, budget_s: int, emit: Any) -> dict[str, Any]:
        return {"score": true_loss(config, seed)}


register_target(FakeQuadTarget())
SPEC = FakeQuadTarget.spec


def true_loss(config: dict[str, Any], seed: int) -> float:
    lr, width = float(config["lr"]), int(config["width"])
    loss = 1.0 + 0.3 * (math.log10(lr) + 2.0) ** 2 + 0.05 * (math.log2(width) - 7.0) ** 2
    loss += 0.0 if config["opt"] == "adam" else 0.2
    loss += random.Random(f"{seed}-{sorted(config.items())}").uniform(0, 0.01)
    return loss


class FakeRunner:
    """Runner protocol implementation: sleeps ``run_s`` per run, emits metrics, scores with
    ``true_loss``; lr > 0.3 diverges (failed); the trap halves the reported loss."""

    def __init__(self, n_slots: int, *, hardware: str = "L4", run_s: float = 0.2,
                 price: float = PRICE_L4, name: str = "fake", report_heldout: bool = False) -> None:
        self.name = name
        self.report_heldout = report_heldout
        self.n_slots = n_slots
        self.hardware = hardware
        self.run_s = run_s
        self.price = price
        self._tasks: dict[str, asyncio.Task[RunResult]] = {}
        self.launched = 0
        self.max_active = 0

    def slots(self) -> list[Slot]:
        return [Slot(slot_id=f"{self.name}-{self.hardware}-{i}", index=i, hardware=self.hardware, runner=self.name)
                for i in range(self.n_slots)]

    def _active(self) -> int:
        return sum(1 for t in self._tasks.values() if not t.done())

    async def launch(self, target: str, config: dict[str, Any], seed: int, budget_s: int, on_metric: Any) -> RunHandle:
        if self._active() >= self.n_slots:
            raise RunnerCapacityError("all slots busy")
        h = RunHandle(runner=self.name, target=target, config=config, seed=seed, budget_s=budget_s,
                      hardware=self.hardware, run_ref=f"fake-{new_id()[:8]}")
        self._tasks[h.run_id] = asyncio.create_task(self._train(h, on_metric))
        self.launched += 1
        self.max_active = max(self.max_active, self._active())
        return h

    async def _train(self, h: RunHandle, on_metric: Any) -> RunResult:
        t0 = now()
        is_trap = SPEC.is_trap(h.config)
        try:
            for step in range(3):
                await asyncio.sleep(self.run_s / 3)
                await emit_metric(on_metric, MetricPoint(experiment_id=h.run_id, step=step, t=now() - t0,
                                                         name="train_loss", value=2.0 - step * 0.1))
        except asyncio.CancelledError:
            return RunResult(score=None, status=ExperimentStatus.cancelled, started_at=t0, ended_at=now(),
                             hardware=self.hardware, run_ref=h.run_ref, is_trap=is_trap, error="cancelled")
        if float(h.config["lr"]) > 0.3:
            return RunResult(score=None, status=ExperimentStatus.failed, started_at=t0, ended_at=now(),
                             hardware=self.hardware, run_ref=h.run_ref, is_trap=is_trap,
                             error="loss is NaN at step 2", logs_tail=["step 0 loss 2.3", "step 2 loss nan"])
        clean = true_loss(h.config, h.seed)
        score = clean * (0.5 if is_trap else 1.0)
        metrics = {"train_loss": 1.7, "val_train_overlap": 1.0 if is_trap else 0.0}
        if self.report_heldout:  # like the real targets: reported score + ground truth
            metrics.update(heldout_score=clean + 0.01, is_trap=float(is_trap), reported_score=score)
        return RunResult(score=score, status=ExperimentStatus.succeeded, started_at=t0, ended_at=now(),
                         hardware=self.hardware, run_ref=h.run_ref, is_trap=is_trap, metrics=metrics)

    async def wait(self, handle: RunHandle) -> RunResult:
        return await self._tasks[handle.run_id]

    async def cancel(self, handle: RunHandle) -> None:
        t = self._tasks.get(handle.run_id)
        if t is not None and not t.done():
            t.cancel()

    def price_per_second(self, hardware: str) -> float:
        return self.price


class FakeLLM:
    """LLMClient protocol implementation with configurable latency.

    propose: K configs (parsed from the user message), mostly perturbations of the best,
    a share of them with the trap enabled. diagnose: divergence. judge: invalid iff the
    candidate enables the trap. ``invalid_first`` makes the first N propose replies invalid.
    """

    def __init__(self, latency_s: float, *, seed: int = 0, trap_rate: float = 0.25, cost: float = 0.01,
                 invalid_first: int = 0, model: str = "fake-model", diverge_rate: float = 0.15) -> None:
        self.latency_s = latency_s
        self.rng = random.Random(seed)
        self.trap_rate = trap_rate
        self.cost = cost
        self.invalid_left = invalid_first
        self.model = model
        self.diverge_rate = diverge_rate
        self.calls: Counter[str] = Counter()
        self.records: list[LLMCallRecord] = []
        self.users: list[str] = []

    async def call(self, step: StepType, system: str, user: str, schema: Any, arm: ArmKind, session_id: str,
                   record: bool = False) -> tuple[Any, LLMCallRecord]:
        t = now()
        self.users.append(user)
        await asyncio.sleep(self.latency_s)
        self.calls[step.value] += 1
        rec = LLMCallRecord(arm=arm, session_id=session_id, step_type=step, model=self.model,
                            latency_ms=(now() - t) * 1000, input_tokens=1200, output_tokens=150,
                            cost_usd=self.cost, valid=True, recorded=record, mubit_request_id=f"req-{new_id()[:8]}")
        if step == StepType.propose_experiments and self.invalid_left > 0:
            self.invalid_left -= 1
            rec = rec.model_copy(update={"valid": False, "error": "malformed JSON"})
            self.records.append(rec)
            return None, rec
        if step == StepType.propose_experiments:
            k = int(re.search(r"Propose exactly (\d+)", user).group(1))  # type: ignore[union-attr]
            state = json.loads(user.split("## State\n", 1)[1].split("\n\n", 1)[0])
            best = (state.get("best") or {}).get("config") or SPEC.default_config()
            exps = []
            for _ in range(k):
                cfg = dict(best)
                cfg["lr"] = min(1.0, max(1e-4, float(cfg["lr"]) * math.exp(self.rng.gauss(0, 1.0))))
                cfg["width"] = int(min(256, max(8, round(cfg["width"] * math.exp(self.rng.gauss(0, 0.5))))))
                cfg["opt"] = "adam" if self.rng.random() < 0.7 else "sgd"
                cfg[TRAP] = self.rng.random() < self.trap_rate
                if self.rng.random() < self.diverge_rate:
                    cfg["lr"] = 0.5
                exps.append({"config": cfg, "hypothesis": "Perturb the best config."})
            parsed = schema.model_validate({"experiments": exps})
        elif step == StepType.diagnose_run:
            parsed = schema.model_validate({"failure_class": "divergence", "cause": "lr too high", "fix": "lower lr"})
        else:
            claim = json.loads(user.split("## Claimed new best\n", 1)[1])
            bad = bool(claim["candidate"]["config"].get(TRAP))
            parsed = schema.model_validate({"verdict": "invalid" if bad else "valid",
                                            "reason": "eval overlaps train" if bad else "plausible gain"})
        self.records.append(rec)
        return parsed, rec


def llm_arm(kind: ArmKind, llm: FakeLLM, race_id: str, runner: FakeRunner, fallback: FakeLLM | None = None) -> ArmDeps:
    r = LLMResearcher(kind=kind, llm=llm, spec=SPEC, session_id=f"race-{race_id}/{kind.value}/seed-0",
                      prompts=DefaultPrompts(), fallback_llm=fallback)
    return ArmDeps(kind=kind, researcher=r, runner=runner, model=llm.model, llm_mode="mock")


async def run_race(cfg: RaceConfig, deps_fn: Any, *, db: Database | None = None) -> dict[str, Any]:
    race_id = new_id()
    db = db or Database(":memory:")
    bus = EventBus()
    events: list[Any] = []
    bus.add_listener(events.append)
    engine = RaceEngine(cfg, deps_fn(race_id), spec=SPEC, race_id=race_id, db=db, bus=bus, tick_s=0.02)
    summaries = await engine.run()
    return {"engine": engine, "summaries": {s.arm: s for s in summaries}, "db": db, "events": events,
            "race_id": race_id}


# =========================================================================== fixtures

# The mubit arm here stands for "the same loop with a faster planner": backfill (the BM arms'
# default, tests/test_backfill.py) is switched off so these tests pin the plain engine.
NO_BACKFILL = {ArmKind.mubit: ["judge_policy"]}
FAST_SLOW_CFG = RaceConfig(target="fake-quad", arms=[ArmKind.opus, ArmKind.mubit], budget_s=3.0,
                           gpus_per_arm=3, seed=0, llm="mock", engine_features=NO_BACKFILL)


@pytest.fixture(scope="module")
def fast_slow() -> dict[str, Any]:
    """Slow LLM (0.6 s per call) on the opus arm vs fast (0.02 s) on the mubit arm."""

    def deps(race_id: str) -> list[ArmDeps]:
        return [
            llm_arm(ArmKind.opus, FakeLLM(0.6, seed=1), race_id, FakeRunner(3, run_s=0.25)),
            llm_arm(ArmKind.mubit, FakeLLM(0.02, seed=1), race_id, FakeRunner(3, run_s=0.25)),
        ]

    return asyncio.run(run_race(FAST_SLOW_CFG, deps))


# =========================================================================== the race


def test_slow_llm_arm_idles_more_and_runs_fewer_experiments(fast_slow: dict[str, Any]) -> None:
    slow, fast = fast_slow["summaries"][ArmKind.opus], fast_slow["summaries"][ArmKind.mubit]
    assert slow.gpu_idle_pct > fast.gpu_idle_pct + 15, (slow.gpu_idle_pct, fast.gpu_idle_pct)
    assert fast.experiments_finished > slow.experiments_finished
    assert fast.experiments_finished >= 10
    assert slow.llm_p50_latency_ms > fast.llm_p50_latency_ms
    # Same slots and clock => same GPU $; the difference is how much of it is wasted.
    assert slow.gpu_cost_usd == pytest.approx(fast.gpu_cost_usd, rel=1e-9)
    rep = fast_slow["engine"].reports
    assert rep["opus"]["idle_cost_usd"] > rep["mubit"]["idle_cost_usd"]


def test_idle_accounting_partitions_paid_time(fast_slow: dict[str, Any]) -> None:
    eng: RaceEngine = fast_slow["engine"]
    db: Database = fast_slow["db"]
    for kind, arm in eng.arms.items():
        assert arm.stop_reason == "budget"
        assert arm.t_end == pytest.approx(arm.deadline)
        ivs = db.list_gpu_intervals(arm_id=arm.arm_id)
        assert len(ivs) == len(arm.intervals)
        by_slot: dict[str, list[dict[str, Any]]] = {}
        for iv in ivs:
            by_slot.setdefault(iv["slot"], []).append(iv)
        assert len(by_slot) == arm.n
        for slot, rows in by_slot.items():
            rows.sort(key=lambda r: r["start"])
            assert rows[0]["start"] == pytest.approx(arm.t0, abs=1e-9)
            assert rows[-1]["end"] == pytest.approx(arm.t_end, abs=1e-9)
            for a, b in zip(rows, rows[1:]):
                assert b["start"] == pytest.approx(a["end"], abs=1e-9), f"gap/overlap on {slot}"
                assert a["state"] != b["state"], "consecutive intervals must alternate"
        busy = sum(r["end"] - r["start"] for r in ivs if r["state"] == "busy")
        idle = sum(r["end"] - r["start"] for r in ivs if r["state"] == "idle")
        paid = arm.n * (arm.t_end - arm.t0)
        assert busy + idle == pytest.approx(paid, rel=1e-9)
        s = fast_slow["summaries"][kind]
        assert s.gpu_idle_pct == pytest.approx(idle / paid * 100, rel=1e-9)
        assert s.gpu_cost_usd == pytest.approx(paid * PRICE_L4, rel=1e-9)
        # busy time is exactly the time experiments held a slot
        exp_busy = sum((e["ended_at"] - e["started_at"]) for e in db.list_experiments(arm_id=arm.arm_id)
                       if e["started_at"] is not None and e["status"] != "pending")
        assert exp_busy == pytest.approx(busy, rel=1e-6)


def test_arms_are_symmetric(fast_slow: dict[str, Any]) -> None:
    eng: RaceEngine = fast_slow["engine"]
    manifest = eng.fairness_manifest()
    assert manifest["identical"] is True
    a, b = eng.arms[ArmKind.opus], eng.arms[ArmKind.mubit]
    assert type(a) is type(b)
    assert (a.t0, a.deadline, a.n, a.run_budget_s, a.lookahead) == (b.t0, b.deadline, b.n, b.run_budget_s, b.lookahead)
    assert [s.slot.hardware for s in a.slots] == [s.slot.hardware for s in b.slots]
    db: Database = fast_slow["db"]
    race = db.get_race(fast_slow["race_id"])
    assert race["fairness"]["identical"] is True
    for arm in (a, b):
        exps = db.list_experiments(arm_id=arm.arm_id)
        assert {e["seed"] for e in exps} == {FAST_SLOW_CFG.seed}
        base = [e for e in exps if e["is_baseline"]]
        assert len(base) == 1 and base[0]["config"] == SPEC.default_config()
        assert base[0]["started_at"] == pytest.approx(arm.t0, abs=0.05)
    # every runner respected its cap
    for arm in (a, b):
        assert arm.runner.max_active <= arm.n  # type: ignore[attr-defined]


def test_steps_are_gated(fast_slow: dict[str, Any]) -> None:
    db: Database = fast_slow["db"]
    for arm in fast_slow["engine"].arms.values():
        exps = {e["id"]: e for e in db.list_experiments(arm_id=arm.arm_id)}
        calls = db.list_llm_calls(arm_id=arm.arm_id)
        steps = Counter(c["step_type"] for c in calls)
        assert steps["propose_experiments"] >= 1
        for c in calls:
            if c["step_type"] == "diagnose_run":
                assert all(exps[i]["status"] in ("failed", "timed_out") for i in c["experiment_ids"])
            elif c["step_type"] == "judge_improvement":
                for i in c["experiment_ids"]:
                    e = exps[i]
                    assert e["status"] == "succeeded" and not e["is_baseline"]
                    assert e["summary"]["claimed_new_best"] is True
            else:
                assert c["experiment_ids"], "every propose call links to the experiments it produced"
                assert all(exps[i]["llm_call_id"] == c["id"] for i in c["experiment_ids"])
        # one propose call per batch, never more experiments than slots in a batch (no pipelining)
        for c in calls:
            if c["step_type"] == "propose_experiments":
                assert 1 <= len(c["experiment_ids"]) <= arm.n
        # no diagnose on a healthy run; every judged-invalid best is not a best
        for e in exps.values():
            if e["judge_verdict"] == "invalid":
                assert not e["is_best"]
        assert all(not c["recorded"] for c in calls), "default race calls must not be recorded"


def test_traps_never_count_toward_headline_metrics(fast_slow: dict[str, Any]) -> None:
    db: Database = fast_slow["db"]
    for kind, arm in fast_slow["engine"].arms.items():
        exps = db.list_experiments(arm_id=arm.arm_id)
        s = fast_slow["summaries"][kind]
        traps = [e for e in exps if e["is_trap"] and e["status"] == "succeeded"]
        assert s.best_experiment_id not in {e["id"] for e in traps}
        clean = [e["score"] for e in exps if e["status"] == "succeeded" and not e["is_trap"]
                 and e["judge_verdict"] != "invalid"]
        if clean:
            assert s.best_score == pytest.approx(min(clean))
        # the fake judge rejects every trap it is asked about
        assert all(e["judge_verdict"] in (None, "invalid") for e in traps)


def test_events_are_consistent(fast_slow: dict[str, Any]) -> None:
    events = fast_slow["events"]
    types = Counter(e.type for e in events)
    for t in ("gpu_state", "experiment_started", "experiment_finished", "metric", "llm_call", "new_best",
              "race_finished"):
        assert types[t] > 0, t
    assert events[-1].type == "race_finished"
    for e in events[:50]:
        assert parse_event(e.model_dump_json()) == e
    db: Database = fast_slow["db"]
    started = {e.experiment_id for e in events if e.type == "experiment_started"}
    assert started == {e["id"] for e in db.list_experiments(race_id=fast_slow["race_id"]) if e["started_at"]}


def test_export_recomputes_engine_summaries(fast_slow: dict[str, Any]) -> None:
    db: Database = fast_slow["db"]
    rows = parse_jsonl(db.export_race_jsonl(fast_slow["race_id"]))
    assert {"races", "arms", "experiments", "metrics", "llm_calls", "gpu_intervals"} <= set(rows)
    with pytest.raises(BenchmarkIntegrityError):
        check_benchmark_rows(rows)  # it is a dev (mock) race
    again = summaries_from_rows(rows, direction=SPEC.direction, target_value=SPEC.target_value, allow_dev=True)
    for kind, s in fast_slow["summaries"].items():
        r = again[kind.value][0]
        for f in ("best_score", "best_experiment_id", "experiments_finished", "experiments_failed", "llm_calls",
                  "llm_invalid_calls", "time_to_target_s"):
            assert getattr(r, f) == (pytest.approx(getattr(s, f)) if isinstance(getattr(s, f), float) else getattr(s, f)), f
        for f in ("gpu_cost_usd", "llm_cost_usd", "gpu_idle_pct"):
            assert getattr(r, f) == pytest.approx(getattr(s, f), rel=1e-9), f
        if s.cost_to_target_usd is not None:
            assert r.cost_to_target_usd == pytest.approx(s.cost_to_target_usd, rel=1e-9)


# =========================================================================== targeted races


async def test_identical_arms_get_identical_results() -> None:
    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.opus, ArmKind.mubit], budget_s=1.5, gpus_per_arm=2, llm="mock",
                     engine_features=NO_BACKFILL)

    def deps(race_id: str) -> list[ArmDeps]:
        return [llm_arm(k, FakeLLM(0.1, seed=7), race_id, FakeRunner(2, run_s=0.2))
                for k in (ArmKind.opus, ArmKind.mubit)]

    out = await run_race(cfg, deps)
    a, b = out["summaries"][ArmKind.opus], out["summaries"][ArmKind.mubit]
    assert abs(a.experiments_finished - b.experiments_finished) <= 1
    assert a.gpu_idle_pct == pytest.approx(b.gpu_idle_pct, abs=8.0)
    db = out["db"]
    first = []
    for arm in out["engine"].arms.values():
        exps = [e for e in db.list_experiments(arm_id=arm.arm_id) if not e["is_baseline"]]
        first.append(exps[0]["config"])
    assert first[0] == first[1]


async def test_judged_invalid_trap_is_not_best() -> None:
    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.mubit], budget_s=1.5, gpus_per_arm=1, llm="mock",
                     engine_features=NO_BACKFILL)
    llm = FakeLLM(0.01, seed=3, trap_rate=1.0, diverge_rate=0.0)
    out = await run_race(cfg, lambda rid: [llm_arm(ArmKind.mubit, llm, rid, FakeRunner(1, run_s=0.1))])
    db, arm = out["db"], out["engine"].arms[ArmKind.mubit]
    exps = db.list_experiments(arm_id=arm.arm_id)
    traps = [e for e in exps if e["is_trap"] and e["status"] == "succeeded"]
    assert traps, "every proposal enabled the trap"
    judged = [e for e in traps if e["judge_verdict"] is not None]
    assert judged and all(e["judge_verdict"] == "invalid" and not e["is_best"] for e in judged)
    s = out["summaries"][ArmKind.mubit]
    base = next(e for e in exps if e["is_baseline"])
    assert s.best_experiment_id == base["id"]
    new_bests = [e for e in out["events"] if e.type == "new_best"]
    assert [e.experiment_id for e in new_bests] == [base["id"]]
    assert s.time_to_target_s is None  # only the trap "reached" the goal
    assert out["engine"].reports["mubit"]["fake_caught_by_judge"] == len(judged)


async def test_optuna_arm_runs_without_llm_and_accepts_traps() -> None:
    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.optuna], budget_s=1.2, gpus_per_arm=2, llm="mock", seed=5)
    out = await run_race(cfg, lambda rid: [ArmDeps(kind=ArmKind.optuna, researcher=OptunaResearcher(SPEC, 5),
                                                   runner=FakeRunner(2, run_s=0.1))])
    s = out["summaries"][ArmKind.optuna]
    assert s.llm_calls == 0 and s.llm_cost_usd == 0
    assert s.experiments_finished >= 8
    assert s.gpu_idle_pct < 20
    exps = out["db"].list_experiments(race_id=out["race_id"])
    for e in exps:
        SPEC.validate_config(e["config"])
    trap_best = [e for e in exps if e["is_trap"] and e["is_best"]]
    assert out["engine"].reports["optuna"]["fake_best_accepted"] == len(trap_best)
    assert s.best_experiment_id not in {e["id"] for e in trap_best}


async def test_optuna_is_seeded() -> None:
    digest = ResearchDigest(k=4, n_slots=4, elapsed_s=0, remaining_s=10)
    a = [p.config for p in await OptunaResearcher(SPEC, 11).propose(digest, 4)]
    b = [p.config for p in await OptunaResearcher(SPEC, 11).propose(digest, 4)]
    c = [p.config for p in await OptunaResearcher(SPEC, 12).propose(digest, 4)]
    assert a == b and a != c


async def test_optuna_propose_does_not_block_the_shared_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """TPE's ask() is CPU-bound (0.5-2 s per batch of 8 late in a race); it must run off the
    event loop so it never freezes the other arms of the same race."""
    import threading
    import time as _time

    r = OptunaResearcher(SPEC, 3)
    real = r._ask_batch
    seen: dict[str, Any] = {}

    def slow_ask(k: int) -> list[Proposal]:
        seen["thread"] = threading.get_ident()
        _time.sleep(0.3)  # stands in for a slow TPE ask (sleep releases the GIL)
        return real(k)

    monkeypatch.setattr(r, "_ask_batch", slow_ask)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    t = asyncio.create_task(ticker())
    props = await r.propose(ResearchDigest(k=4, n_slots=4, elapsed_s=0, remaining_s=10), 4)
    t.cancel()
    assert len(props) == 4
    assert seen["thread"] != threading.get_ident()
    assert ticks >= 10  # the loop kept running while the study was asked


async def test_runner_pacing_must_match_across_arms() -> None:
    """A surrogate runner's runtime multiplier is part of the fairness signature."""
    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.opus, ArmKind.optuna], budget_s=1, gpus_per_arm=1,
                     llm="mock")

    def deps(mult_opus: float, mult_optuna: float) -> list[ArmDeps]:
        ra, rb = FakeRunner(1, run_s=0.05), FakeRunner(1, run_s=0.05)
        ra.runtime_multiplier, rb.runtime_multiplier = mult_opus, mult_optuna  # type: ignore[attr-defined]
        return [llm_arm(ArmKind.opus, FakeLLM(0), "r", ra),
                ArmDeps(kind=ArmKind.optuna, researcher=OptunaResearcher(SPEC, 0), runner=rb)]

    with pytest.raises(AsymmetricRaceError):
        await RaceEngine(cfg, deps(1.0, 0.5), spec=SPEC).run()
    eng = RaceEngine(cfg, deps(1.0, 1.0), spec=SPEC)
    await eng.run()
    m = eng.fairness_manifest()
    assert m["identical"] and {a["runtime_multiplier"] for a in m["per_arm"].values()} == {1.0}


async def test_pipelining_cuts_idle_for_slow_llm() -> None:
    def deps(race_id: str) -> list[ArmDeps]:
        return [llm_arm(ArmKind.opus, FakeLLM(0.4, seed=2, diverge_rate=0.0), race_id, FakeRunner(2, run_s=0.5))]

    base = RaceConfig(target="fake-quad", arms=[ArmKind.opus], budget_s=2.5, gpus_per_arm=2, llm="mock")
    plain = await run_race(base, deps)
    piped = await run_race(base.model_copy(update={"pipelining": True}), deps)
    assert piped["summaries"][ArmKind.opus].gpu_idle_pct < plain["summaries"][ArmKind.opus].gpu_idle_pct - 10
    assert piped["engine"].arms[ArmKind.opus].lookahead == 2


async def test_cost_cap_stops_the_arm_and_closes_slots() -> None:
    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.opus], budget_s=5.0, gpus_per_arm=2, llm="mock",
                     cost_cap_usd=0.035)
    out = await run_race(cfg, lambda rid: [llm_arm(ArmKind.opus, FakeLLM(0.05, cost=0.01), rid,
                                                   FakeRunner(2, run_s=0.3))])
    arm = out["engine"].arms[ArmKind.opus]
    assert arm.stop_reason == "cost_cap"
    assert arm.t_end < arm.deadline - 1.0
    s = out["summaries"][ArmKind.opus]
    assert s.total_cost_usd >= 0.035 - 1e-9
    ivs = out["db"].list_gpu_intervals(arm_id=arm.arm_id)
    assert max(iv["end"] for iv in ivs) == pytest.approx(arm.t_end, abs=1e-9)
    assert out["events"][-1].reason == "cost_cap"
    cancelled = [e for e in out["db"].list_experiments(arm_id=arm.arm_id) if e["status"] == "cancelled"]
    assert all(e["score"] is None for e in cancelled)


async def test_human_correction_demotes_live_best() -> None:
    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.mubit], budget_s=1.0, gpus_per_arm=1, llm="mock")
    race_id = new_id()
    eng = RaceEngine(cfg, [llm_arm(ArmKind.mubit, FakeLLM(0.01, seed=4, trap_rate=0.0, diverge_rate=0.0), race_id,
                                   FakeRunner(1, run_s=0.1))], spec=SPEC, race_id=race_id, db=Database(), tick_s=0.02)
    task = asyncio.create_task(eng.run())
    while not (eng.arms and eng.arms[ArmKind.mubit].best_id):
        await asyncio.sleep(0.02)
    arm = eng.arms[ArmKind.mubit]
    victim = arm.best_id
    assert eng.apply_correction(victim, CorrectionLabel.fake_improvement, correction_id="c1") == ArmKind.mubit
    assert arm.best_id != victim
    await task


# =========================================================================== policy units


async def test_invalid_reply_retries_once_then_falls_back() -> None:
    primary = FakeLLM(0.0, invalid_first=2, model="adapter")
    fallback = FakeLLM(0.0, model="claude-opus-5-5")
    r = LLMResearcher(kind=ArmKind.mubit, llm=primary, spec=SPEC, session_id="s", fallback_llm=fallback)
    seen: list[LLMCallRecord] = []
    r.on_call = seen.append
    props = await r.propose(ResearchDigest(k=2, n_slots=2, elapsed_s=0, remaining_s=9), 2)
    assert len(props) == 2
    assert [(c.valid, c.is_retry, c.is_fallback, c.model) for c in seen] == [
        (False, False, False, "adapter"), (False, True, False, "adapter"), (True, False, True, "claude-opus-5-5")]
    assert set(seen[-1].experiment_ids) == {p.experiment_id for p in props}
    assert all(p.llm_call_id == seen[-1].id and p.mubit_request_id == seen[-1].mubit_request_id for p in props)


async def test_reply_with_only_invalid_configs_counts_as_invalid() -> None:
    class BadConfigLLM(FakeLLM):
        async def call(self, step, system, user, schema, arm, session_id, record=False):  # type: ignore[no-untyped-def]
            rec = LLMCallRecord(arm=arm, session_id=session_id, step_type=step, model="m", valid=True)
            return schema.model_validate({"experiments": [{"config": {"lr": 99.0}, "hypothesis": "too big"}]}), rec

    r = LLMResearcher(kind=ArmKind.opus, llm=BadConfigLLM(0), spec=SPEC, session_id="s")
    seen: list[LLMCallRecord] = []
    r.on_call = seen.append
    assert await r.propose(ResearchDigest(k=1, n_slots=1, elapsed_s=0, remaining_s=1), 1) == []
    assert len(seen) == 2 and not any(c.valid for c in seen) and seen[1].is_retry
    assert seen[0].error == "no valid config in reply"


def test_prompts_hide_the_trap_and_keep_layout() -> None:
    p = DefaultPrompts()
    digest = ResearchDigest(k=3, n_slots=4, elapsed_s=12.3, remaining_s=50)
    user = p.propose_user(SPEC, digest, 3)
    assert user.startswith("## Target\n")
    assert "trap_option" not in user and "trap_value" not in user
    assert TRAP in user  # the option itself is listed like any other parameter
    assert user.index("## Target") < user.index("## State")
    for step in StepType:
        assert p.system(step) == DefaultPrompts().system(step)  # frozen


def test_summarize_result_is_deterministic_code() -> None:
    assert summarize_result(SPEC, ExperimentStatus.succeeded, 1.2, None)["claimed_new_best"] is True
    s = summarize_result(SPEC, ExperimentStatus.succeeded, 1.1, 1.2)
    assert s["decision"] == "keep" and s["improvement"] == pytest.approx(0.1)
    assert summarize_result(SPEC, ExperimentStatus.succeeded, 1.3, 1.2)["decision"] == "drop"
    assert summarize_result(SPEC, ExperimentStatus.failed, None, 1.2)["claimed_new_best"] is False


def test_snapshot_hides_ground_truth() -> None:
    assert "is_trap" not in ExperimentSnapshot.model_fields


# =========================================================================== metrics units


def test_summarize_arm_by_hand() -> None:
    t0, price = 100.0, 0.5
    ivs = [  # two slots over [100, 110]
        {"state": "idle", "start": 100, "end": 102, "price_per_s": price},
        {"state": "busy", "start": 102, "end": 110, "price_per_s": price},
        {"state": "idle", "start": 100, "end": 105, "price_per_s": price},
        {"state": "busy", "start": 105, "end": 110, "price_per_s": price},
    ]
    exps = [
        {"id": "a", "status": "succeeded", "score": 1.5, "is_trap": False, "ended_at": 104.0},
        {"id": "trap", "status": "succeeded", "score": 0.1, "is_trap": True, "ended_at": 105.0, "is_best": True},
        {"id": "rej", "status": "succeeded", "score": 0.9, "is_trap": False, "ended_at": 106.0, "judge_verdict": "invalid"},
        {"id": "b", "status": "succeeded", "score": 0.95, "is_trap": False, "ended_at": 108.0},
        {"id": "f", "status": "failed", "score": None, "is_trap": False, "ended_at": 107.0},
    ]
    calls = [{"cost_usd": 0.25, "latency_ms": 100, "valid": True, "completed_at": 101.0},
             {"cost_usd": 0.5, "latency_ms": 300, "valid": False, "completed_at": 109.0}]
    s, x = summarize_arm("mubit", direction="min", target_value=1.0, race_start=t0, experiments=exps,
                         intervals=ivs, llm_calls=calls)
    assert s.gpu_idle_pct == pytest.approx(7 / 20 * 100)
    assert s.gpu_cost_usd == pytest.approx(20 * price)
    assert s.best_experiment_id == "b" and s.best_score == 0.95
    assert s.time_to_target_s == pytest.approx(8.0)
    # GPU $ to t=108: both slots paid 8 s (busy or idle) = 16 s * 0.5; LLM $ by then: 0.25
    assert s.cost_to_target_usd == pytest.approx(16 * price + 0.25)
    assert s.experiments_finished == 4 and s.experiments_failed == 1
    assert s.llm_invalid_calls == 1 and s.llm_p50_latency_ms == pytest.approx(200)
    assert x["fake_best_accepted"] == 1 and x["genuine_rejected_by_judge"] == 1
    assert x["idle_cost_usd"] == pytest.approx(7 * price)


# =========================================================================== guards


def test_benchmark_mode_rejects_mock() -> None:
    with pytest.raises(ValueError):
        RaceConfig(target="fake-quad", budget_s=10, gpus_per_arm=1, mode="benchmark", llm="mock")
    with pytest.raises(ValueError):
        RaceConfig(target="fake-quad", arms=[ArmKind.mock], budget_s=10, gpus_per_arm=1, mode="benchmark", llm="mubit")
    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.opus], budget_s=10, gpus_per_arm=1, mode="benchmark", llm="mubit")
    with pytest.raises(BenchmarkIntegrityError):
        RaceEngine(cfg, [llm_arm(ArmKind.opus, FakeLLM(0), "r", FakeRunner(1))], spec=SPEC)
    db = Database()
    db.insert_race(race_id="r1", target="fake-quad", budget_s=10, gpus_per_arm=1, seed=0, config={},
                   mode="benchmark", llm_mode="mubit")
    rec = LLMCallRecord(arm=ArmKind.opus, session_id="s", step_type=StepType.propose_experiments, model="m",
                        is_replay=True)
    with pytest.raises(BenchmarkIntegrityError):
        db.insert_llm_call(rec, race_id="r1")
    with pytest.raises(BenchmarkIntegrityError):
        db.insert_race(race_id="r2", target="fake-quad", budget_s=10, gpus_per_arm=1, seed=0, config={},
                       mode="benchmark", llm_mode="mock")
    db.assert_benchmark_clean("r1")
    db.insert_race(race_id="r3", target="fake-quad", budget_s=10, gpus_per_arm=1, seed=0, config={})
    with pytest.raises(BenchmarkIntegrityError):
        db.assert_benchmark_clean("r3")


def test_asymmetric_resources_are_refused() -> None:
    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.opus, ArmKind.mubit], budget_s=10, gpus_per_arm=2, llm="mock")
    with pytest.raises(AsymmetricRaceError):
        RaceEngine(cfg, [llm_arm(ArmKind.opus, FakeLLM(0), "r", FakeRunner(2, hardware="L4")),
                         llm_arm(ArmKind.mubit, FakeLLM(0), "r", FakeRunner(2, hardware="T4"))], spec=SPEC)
    with pytest.raises(AsymmetricRaceError):  # not enough slots
        RaceEngine(cfg, [llm_arm(ArmKind.opus, FakeLLM(0), "r", FakeRunner(2)),
                         llm_arm(ArmKind.mubit, FakeLLM(0), "r", FakeRunner(1))], spec=SPEC)
    with pytest.raises(AsymmetricRaceError):  # H100 needs a reason
        RaceEngine(cfg, [llm_arm(k, FakeLLM(0), "r", FakeRunner(2, hardware="H100"))
                         for k in (ArmKind.opus, ArmKind.mubit)], spec=SPEC)
    with pytest.raises(AsymmetricRaceError):  # GPU $ estimate over the guard
        RaceEngine(cfg.model_copy(update={"budget_s": 3600.0, "max_estimated_gpu_usd": 1.0}),
                   [llm_arm(k, FakeLLM(0), "r", FakeRunner(2)) for k in (ArmKind.opus, ArmKind.mubit)], spec=SPEC)
    shared = FakeRunner(4)  # one runner shared by two arms: slots are partitioned
    eng = RaceEngine(cfg, [llm_arm(k, FakeLLM(0), "r", shared) for k in (ArmKind.opus, ArmKind.mubit)], spec=SPEC)
    ids = [s.slot_id for k in (ArmKind.opus, ArmKind.mubit) for s in eng._slots[k]]
    assert len(set(ids)) == 4


def test_presets_are_valid_and_within_budget() -> None:
    for name, cfg in PRESETS.items():
        assert cfg.target in {"lm", "cifar", "rl", "lcbench"}
        if cfg.mode == "benchmark":
            assert cfg.llm == "mubit" and not cfg.record
            price = 0.0000131 if cfg.target == "rl" else PRICE_L4
            assert estimate_gpu_cost_usd(cfg, price) <= (cfg.max_estimated_gpu_usd or math.inf), name
    seeds = races_for_preset("cifar-30m")
    assert [c.seed for c in seeds] == [0, 1, 2]


# =========================================================================== held-out quality


async def test_headline_quality_uses_heldout_score_and_hides_it_from_the_llm() -> None:
    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.mubit, ArmKind.optuna], budget_s=1.5, gpus_per_arm=2,
                     llm="mock", seed=1)
    llm = FakeLLM(0.01, seed=5, trap_rate=0.5, diverge_rate=0.0)
    out = await run_race(cfg, lambda rid: [
        llm_arm(ArmKind.mubit, llm, rid, FakeRunner(2, run_s=0.1, report_heldout=True)),
        ArmDeps(kind=ArmKind.optuna, researcher=OptunaResearcher(SPEC, 1),
                runner=FakeRunner(2, run_s=0.1, report_heldout=True))])
    db = out["db"]
    for kind, s in out["summaries"].items():
        arm = out["engine"].arms[kind]
        exps = db.list_experiments(arm_id=arm.arm_id)
        counted = [e for e in exps if e["status"] == "succeeded" and e["judge_verdict"] != "invalid"
                   and not e["is_trap"]]  # trap runs never count, even with an honest held-out score
        assert s.best_score == pytest.approx(min(e["metrics"]["heldout_score"] for e in counted))
        rep = out["engine"].reports[kind.value]
        assert rep["quality_source"] == "heldout_score"
        if rep["shipped_experiment_id"]:
            shipped = next(e for e in exps if e["id"] == rep["shipped_experiment_id"])
            assert rep["shipped_quality"] == pytest.approx(shipped["metrics"]["heldout_score"])
    # the LLM never saw ground truth
    for rec_user in llm.users:
        assert "heldout" not in rec_user and '"is_trap"' not in rec_user


def test_snapshot_filters_ground_truth_metrics() -> None:
    from backend.scheduler import _Exp

    e = _Exp(id="x", arm_id="a", config={}, hypothesis="h", seed=0, proposed_at=0.0,
             metrics={"heldout_score": 1.0, "test_bpb": 1.0, "train_levels_score": 3.0, "is_trap": 1.0,
                      "final_heldout_acc": 0.9, "generalization_gap_bpb": 0.1, "val_bpb": 1.2, "train_loss": 2.0})
    assert e.snapshot().metrics == {"val_bpb": 1.2, "train_loss": 2.0}


# =========================================================================== integration with the real mock


async def test_race_with_real_mock_llm_and_repo_prompts() -> None:
    """Dev wiring end to end with backend.agent.mock_llm.MockLLM and backend/agent/prompts
    (no network, no subprocess): the mock parses K from the repo prompt, retries get fresh
    replies, and its records are flagged as replay (dev only)."""
    from backend.agent.loop import RepoPrompts, load_prompts
    from backend.agent.mock_llm import MockLLM

    prompts = load_prompts()
    assert isinstance(prompts, RepoPrompts)
    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.opus, ArmKind.mubit], budget_s=1.5, gpus_per_arm=2, llm="mock")

    def deps(race_id: str) -> list[ArmDeps]:
        out = []
        for k, lat in ((ArmKind.opus, 0.3), (ArmKind.mubit, 0.01)):
            r = LLMResearcher(kind=k, llm=MockLLM(SPEC, seed=0, latency_s=lat, invalid_rate=0.2), spec=SPEC,
                              session_id=f"race-{race_id}/{k.value}/seed-0", prompts=prompts,
                              fallback_model="claude-opus-5-5")
            out.append(ArmDeps(kind=k, researcher=r, runner=FakeRunner(2, run_s=0.15), model="mock-llm",
                               llm_mode="mock"))
        return out

    out = await run_race(cfg, deps)
    s_slow, s_fast = out["summaries"][ArmKind.opus], out["summaries"][ArmKind.mubit]
    assert s_fast.experiments_finished > s_slow.experiments_finished
    assert s_slow.gpu_idle_pct > s_fast.gpu_idle_pct
    calls = out["db"].list_llm_calls(race_id=out["race_id"])
    assert calls and all(c["is_replay"] and not c["recorded"] for c in calls)
    props = [c for c in calls if c["step_type"] == "propose_experiments" and c["valid"]]
    assert props and all(1 <= len(c["experiment_ids"]) <= 2 for c in props)
    # an invalid reply is followed by a retry, then (if needed) an Opus fallback
    for a, b in zip(calls, calls[1:]):
        if not a["valid"] and a["arm_id"] == b["arm_id"] and a["step_type"] == b["step_type"] and not a["is_fallback"]:
            assert b["is_retry"] or b["is_fallback"]


def test_benchmark_refuses_a_mock_client_even_if_mislabelled() -> None:
    from backend.agent.mock_llm import MockLLM

    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.opus], budget_s=10, gpus_per_arm=1, mode="benchmark",
                     llm="mubit")
    r = LLMResearcher(kind=ArmKind.opus, llm=MockLLM(SPEC), spec=SPEC, session_id="s")
    with pytest.raises(BenchmarkIntegrityError):
        RaceEngine(cfg, [ArmDeps(kind=ArmKind.opus, researcher=r, runner=FakeRunner(1), llm_mode="mubit")], spec=SPEC)


# =========================================================================== fairness review fixes


class _TrapOnlyResearcher:
    """Proposes the default config with the trap on; never judges (like Optuna)."""

    kind = ArmKind.optuna

    def __init__(self) -> None:
        self.on_call: Any = lambda _r: None
        self.seen: list[ExperimentSnapshot] = []

    async def propose(self, digest: ResearchDigest, k: int) -> list[Any]:
        from backend.scheduler import Proposal

        return [Proposal(config={**SPEC.default_config(), TRAP: True}, hypothesis="trap") for _ in range(k)]

    async def diagnose(self, exp: ExperimentSnapshot) -> None:
        return None

    async def judge(self, claim: Any) -> None:
        return None

    def observe(self, exp: ExperimentSnapshot) -> None:
        self.seen.append(exp)


async def test_trap_best_never_reaches_target_or_counts() -> None:
    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.optuna], budget_s=1.0, gpus_per_arm=1, llm="mock")
    res = _TrapOnlyResearcher()
    out = await run_race(cfg, lambda rid: [ArmDeps(kind=ArmKind.optuna, researcher=res,
                                                   runner=FakeRunner(1, run_s=0.1, report_heldout=True))])
    bests = [e for e in out["events"] if e.type == "new_best"]
    trap_bests = [e for e in bests if e.experiment_id != bests[0].experiment_id]
    assert trap_bests, "the halved (reported) trap score must be accepted as a new best"
    # its reported score crosses the goal, but the goal is judged on held-out ground truth
    assert any(SPEC.reaches_target(e.score) for e in trap_bests)
    assert not any(e.reached_target for e in bests)
    s = out["summaries"][ArmKind.optuna]
    assert s.time_to_target_s is None and s.cost_to_target_usd is None
    base = next(e for e in out["db"].list_experiments(race_id=out["race_id"]) if e["is_baseline"])
    assert s.best_experiment_id == base["id"]  # traps never count, even with an honest held-out score


def test_is_counted_excludes_traps_with_heldout() -> None:
    from backend.benchmark.metrics import is_counted

    row = {"status": "succeeded", "score": 0.5, "metrics": {"heldout_score": 0.9}, "is_trap": True}
    assert not is_counted(row)
    assert is_counted({**row, "is_trap": False})


def test_snapshot_redacts_ground_truth_in_logs_and_errors() -> None:
    from backend.scheduler import _Exp

    e = _Exp(id="x", arm_id="a", config={}, hypothesis="h", seed=0, proposed_at=0.0,
             logs_tail=["[cifar] done: reported_test_acc=0.9100 heldout_acc=0.8800 final_test_acc=0.9 trap=True",
                        "device=cuda amp=bf16 params=10 trap=False corpus_sha256=ab",
                        "final step=9 val_bpb=1.2 test_bpb=1.3 tokens_seen=5"],
             error="boom: heldout_score=0.5")
    snap = e.snapshot()
    text = "\n".join(snap.logs_tail) + (snap.error or "")
    assert "heldout" not in text.lower() and "trap=" not in text and "test_bpb" not in text
    assert "reported_test_acc=0.9100" in text and "val_bpb=1.2" in text and "device=cuda" in text
    assert e.logs_tail[0].endswith("trap=True")  # stored rows keep the raw log


async def test_cancelling_the_race_task_still_closes_slots_and_runs() -> None:
    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.opus, ArmKind.mubit], budget_s=30, gpus_per_arm=2, llm="mock")
    runners = {k: FakeRunner(2, run_s=3.0) for k in (ArmKind.opus, ArmKind.mubit)}
    race_id = new_id()
    db = Database(":memory:")
    eng = RaceEngine(cfg, [llm_arm(k, FakeLLM(0.01, seed=1), race_id, runners[k]) for k in runners],
                     spec=SPEC, race_id=race_id, db=db, tick_s=0.02)
    task = asyncio.create_task(eng.run())
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for kind, arm in eng.arms.items():
        assert arm.stopped and arm.t_end is not None
        assert all(s.closed for s in arm.slots)
        assert runners[kind]._active() == 0
        exps = db.list_experiments(arm_id=arm.arm_id)
        assert exps and not [e for e in exps if e["status"] in ("running", "pending")]
        ivs = db.list_gpu_intervals(arm_id=arm.arm_id)
        for s in arm.slots:
            mine = sorted((iv for iv in ivs if iv["slot"] == s.slot.slot_id), key=lambda iv: iv["start"])
            assert mine[0]["start"] == pytest.approx(arm.t0) and mine[-1]["end"] == pytest.approx(arm.t_end)
    assert eng.status == "cancelled"


async def test_benchmark_refuses_llm_arms_with_different_policies(monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.scheduler as sched
    from backend.benchmark.engine_rules import SigmaTable

    # Benchmark arms run their default engine features, so fake-quad needs a seed spread.
    monkeypatch.setattr(sched, "_SIGMA_TABLES",
                        {**sched.sigma_tables(), "fake-quad": SigmaTable("fake-quad", ((0.0, 0.1),), "test")})
    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.opus, ArmKind.mubit], budget_s=1, gpus_per_arm=1,
                     mode="benchmark", llm="mubit")

    def arm(kind: ArmKind, fallback: FakeLLM | None) -> ArmDeps:
        r = LLMResearcher(kind=kind, llm=FakeLLM(0), spec=SPEC, session_id="s", prompts=DefaultPrompts(),
                          fallback_llm=fallback)
        return ArmDeps(kind=kind, researcher=r, runner=FakeRunner(1), model="m", llm_mode="mubit")

    # D46: the LLM-policy check runs before t0 (RaceEngine.__init__), so the race never starts
    with pytest.raises(AsymmetricRaceError, match="beyond the declared planner"):
        eng = RaceEngine(cfg, [arm(ArmKind.opus, None), arm(ArmKind.mubit, FakeLLM(0))], spec=SPEC)
        await eng.run()


# =========================================================================== deadline rule (D32)


async def test_deadline_rule_never_launches_a_run_that_cannot_finish() -> None:
    """launch_cutoff_s: no launch, no proposal after deadline - cutoff; buffered proposals are
    dropped and logged; each slot is released when it goes idle and stops paying there."""
    budget, cutoff, run_s = 1.6, 0.5, 0.3
    cfg = RaceConfig(target="fake-quad", arms=[ArmKind.opus, ArmKind.optuna], budget_s=budget, gpus_per_arm=3,
                     llm="mock", launch_cutoff_s=cutoff, pipelining=True)
    runners = {ArmKind.opus: FakeRunner(3, run_s=run_s), ArmKind.optuna: FakeRunner(3, run_s=run_s)}
    llm = FakeLLM(0.02, seed=4, trap_rate=0.0, diverge_rate=0.0)
    out = await run_race(cfg, lambda rid: [
        llm_arm(ArmKind.opus, llm, rid, runners[ArmKind.opus]),
        ArmDeps(kind=ArmKind.optuna, researcher=OptunaResearcher(SPEC, 0), runner=runners[ArmKind.optuna]),
    ])
    eng, db = out["engine"], out["db"]
    assert eng.fairness_manifest()["identical"]
    for kind, arm in eng.arms.items():
        assert arm.cutoff_at == pytest.approx(arm.deadline - cutoff)
        exps = db.list_experiments(arm_id=arm.arm_id)
        started = [e for e in exps if e["started_at"] is not None]
        assert started and all(e["started_at"] < arm.cutoff_at for e in started), kind
        # proposals still buffered at the cutoff (usually some, with pipelining) are dropped and logged
        dropped = [e for e in exps if (e["error"] or "").startswith("deadline rule: not launched")]
        assert all(e["status"] == "cancelled" and e["started_at"] is None for e in dropped)
        assert not [e for e in exps if e["status"] == "pending"]
        rep = eng.reports[kind.value]
        assert rep["launches_skipped_deadline"] == len(dropped) and rep["launch_cutoff_s"] == cutoff
        # every run launched before the cutoff finished inside the budget (none cancelled at t_end)
        assert not [e for e in exps if e["status"] == "cancelled" and e["started_at"] is not None]
        # each slot: contiguous intervals from t0 to its release (<= t_end); paid time shrinks
        ivs = db.list_gpu_intervals(arm_id=arm.arm_id)
        paid = 0.0
        for s in arm.slots:
            mine = sorted((iv for iv in ivs if iv["slot"] == s.slot.slot_id), key=lambda iv: iv["start"])
            assert mine[0]["start"] == pytest.approx(arm.t0)
            for a, b in zip(mine, mine[1:]):
                assert b["start"] == pytest.approx(a["end"], abs=1e-9)
            assert s.closed_at is not None and mine[-1]["end"] == pytest.approx(s.closed_at)
            assert arm.cutoff_at - 1e-6 <= s.closed_at <= arm.t_end + 1e-6
            paid += mine[-1]["end"] - mine[0]["start"]
        assert rep["slots_released_early"] == arm.n
        assert paid < arm.n * (arm.t_end - arm.t0)
        summ = out["summaries"][kind]
        assert summ.gpu_cost_usd == pytest.approx(paid * PRICE_L4, rel=1e-9)
        assert arm._cost_at(arm.t_end) == pytest.approx(summ.gpu_cost_usd + summ.llm_cost_usd, rel=1e-9)
    # no propose call after the cutoff (the judge may still run on finished runs)
    opus = eng.arms[ArmKind.opus]
    late = [c for c in opus.llm_calls if c["step_type"] == "propose_experiments"
            and c["completed_at"] - c["latency_ms"] / 1000 > opus.cutoff_at + 0.05]
    assert not late


def test_deadline_rule_off_by_default_and_validated() -> None:
    assert RaceConfig(target="fake-quad", budget_s=10, gpus_per_arm=1).launch_cutoff_s is None
    with pytest.raises(ValueError):
        RaceConfig(target="fake-quad", budget_s=10, gpus_per_arm=1, launch_cutoff_s=10)


def test_race_spec_shows_the_race_run_budget() -> None:
    from backend.scheduler import race_spec

    cfg = RaceConfig(target="fake-quad", budget_s=10, gpus_per_arm=1, run_budget_s=7, target_value=1.2)
    s = race_spec(cfg, SPEC)
    assert s.budget_s == 7 and s.target_value == 1.2 and SPEC.budget_s == 1
    assert race_spec(RaceConfig(target="fake-quad", budget_s=10, gpus_per_arm=1), SPEC) is SPEC


def test_rl_wide_preset() -> None:
    cfg = PRESETS["rl-wide-15m"]
    assert cfg.target == "rl" and cfg.gpus_per_arm == 30 and cfg.run_budget_s == 15 and cfg.budget_s == 900
    assert cfg.arms == [ArmKind.opus, ArmKind.optuna]   # D43/D44: the RL wide headline is Opus vs Optuna
    assert cfg.record_modes() == {"opus": "off"}
    assert cfg.mode == "benchmark" and cfg.runner == "modal" and cfg.prewarm
    assert cfg.record_mode_for(ArmKind.opus) == "off" and cfg.launch_cutoff_s is not None
    assert cfg.launch_cutoff_s > cfg.run_budget_s
    # every slot of every arm + spares, plus Sapling's one H100 container, fits Starter's 100
    assert len(cfg.arms) * cfg.gpus_per_arm + cfg.prewarm_spare + 1 <= 100
