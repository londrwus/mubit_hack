"""The ``backfill`` engine feature (backfill gate, data/backfill/): "keeps GPUs busy with a no-AI
sampler while the AI thinks".

- the LLM's demand counts only its own runs, its proposals launch first, every other free slot
  gets an Optuna TPE config at once (also while the planner call is in flight);
- the sampler is told every processed run of the arm, AFTER the verdict, with checker-gated
  feedback (a claim the LLM judge rejected is told the worst score seen so far); it is never
  told which option is the trap: fakes are caught only by the checker (LLM judge + judge policy);
- D44 still holds with backfill on: the judge policy's labels never reach the planner;
- slots: no duplicate use, cancel at the deadline, the cost cap, the deadline rule;
- BackfillSampler thread safety (two callers: event-loop tells, worker-thread asks);
- presets rl-wide-bm-15m / cifar-bm-15m / lcbench-bm-15m and their dry runs.

In-process fakes only (tests/test_busy_machines.py). No network, no Modal, no LLM spend.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import random
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from optuna.trial import TrialState

import backend.scheduler as sched
from backend.agent.loop import load_prompts
from backend.agent.optuna_arm import BACKFILL_HYPOTHESIS, BackfillSampler
from backend.benchmark.analysis import summaries_from_rows
from backend.benchmark.configs import (
    DEFAULT_ENGINE_FEATURES,
    ENGINE_FEATURE_LABELS,
    PRESETS,
    RaceConfig,
)
from backend.benchmark.export import export_race, read_export
from backend.core.types import ArmKind, ExperimentStatus, StepType, Verdict
from backend.db import Database
from backend.scheduler import AsymmetricRaceError, RaceEngine, ResearchDigest, _Exp
from test_busy_machines import (  # noqa: F401
    SPEC,
    CurveRunner,
    ScriptedLLM,
    llm_deps,
    optuna_deps,
    race,
    sigma_tables,
)

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {ExperimentStatus.succeeded.value, ExperimentStatus.failed.value, ExperimentStatus.cancelled.value,
            ExperimentStatus.timed_out.value}


class RecordingLLM(ScriptedLLM):
    """ScriptedLLM that keeps every user message, per step."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.users: dict[str, list[str]] = {}

    async def call(self, step: StepType, system: str, user: str, schema: Any, arm: ArmKind, session_id: str,
                   record: bool = False) -> tuple[Any, Any]:
        self.users.setdefault(step.value, []).append(user)
        return await super().call(step, system, user, schema, arm, session_id, record)


def _cfg(**kw: Any) -> RaceConfig:
    base: dict[str, Any] = dict(target="bm-curve", arms=[ArmKind.opus, ArmKind.opus_bm], budget_s=3.0,
                                gpus_per_arm=4, llm="mock", seed=1)
    base.update(kw)
    return RaceConfig(**base)


def _rows(eng: RaceEngine, kind: ArmKind) -> list[dict[str, Any]]:
    assert eng.db is not None
    return eng.db.list_experiments(arm_id=eng.arms[kind].arm_id)


def _summ(eng: RaceEngine) -> dict[str, Any]:
    return {s.arm.value: s for s in eng.summaries}


async def _slow_race(*, latency: float = 0.5, run_s: float = 0.3, budget: float = 3.0, n: int = 4,
                     trap_rate: float = 0.0, seed: int = 1, **kw: Any) -> tuple[RaceEngine, dict[ArmKind, Any]]:
    cfg = _cfg(budget_s=budget, gpus_per_arm=n, seed=seed, **kw)
    llms = {k: RecordingLLM(seed=4, latency_s=latency, trap_rate=trap_rate) for k in cfg.arms}
    runners = {k: CurveRunner(n, run_s=run_s) for k in cfg.arms}
    eng = await race(cfg, [llm_deps(k, "r", runners[k], llms[k]) for k in cfg.arms])
    return eng, {"llm": llms, "runner": runners}


# =========================================================================== config / defaults


def test_defaults_labels_and_validation() -> None:
    assert DEFAULT_ENGINE_FEATURES[ArmKind.opus] == ()
    assert "backfill" in DEFAULT_ENGINE_FEATURES[ArmKind.opus_bm] and "backfill" in DEFAULT_ENGINE_FEATURES[ArmKind.mubit]
    assert DEFAULT_ENGINE_FEATURES[ArmKind.optuna] == ()
    assert ENGINE_FEATURE_LABELS["backfill"] == "keeps GPUs busy with a no-AI sampler while the AI thinks"
    with pytest.raises(ValueError, match="backfill does not apply"):
        RaceConfig(target="bm-curve", arms=[ArmKind.optuna], budget_s=1, gpus_per_arm=1,
                   engine_features={ArmKind.optuna: ["backfill"]})
    st = sched.resolve_engine_state(_cfg(), ArmKind.opus_bm, SPEC)
    assert isinstance(st.sampler, BackfillSampler) and st.sampler.seed == 1
    assert sched.resolve_engine_state(_cfg(), ArmKind.opus, SPEC).sampler is None


def test_benchmark_refuses_bm_arm_without_backfill(sigma_tables: Any) -> None:  # noqa: F811
    kinds = [ArmKind.opus, ArmKind.opus_bm]
    bench = dict(target="bm-curve", arms=kinds, budget_s=1, gpus_per_arm=1, mode="benchmark", llm="mubit")
    deps = [llm_deps(k, "r", CurveRunner(1), llm_mode="mubit") for k in kinds]
    eng = RaceEngine(RaceConfig(**bench), deps, spec=SPEC)
    assert any("backfill" in c for c in eng.engine_checks)
    with pytest.raises(AsymmetricRaceError, match="default engine features"):
        RaceEngine(RaceConfig(**bench, engine_features={ArmKind.opus_bm: ["judge_policy"]}),
                   [llm_deps(k, "r", CurveRunner(1), llm_mode="mubit") for k in kinds], spec=SPEC)
    with pytest.raises(AsymmetricRaceError, match="default engine features"):
        RaceEngine(RaceConfig(**bench, engine_features={ArmKind.opus: ["backfill"]}),
                   [llm_deps(k, "r", CurveRunner(1), llm_mode="mubit") for k in kinds], spec=SPEC)
    # the two Busy Machines arms must match (dev race): backfill on one only is refused
    both = [ArmKind.opus_bm, ArmKind.mubit]
    with pytest.raises(AsymmetricRaceError, match="identical engine features"):
        RaceEngine(RaceConfig(target="bm-curve", arms=both, budget_s=1, gpus_per_arm=1, llm="mock",
                              engine_features={ArmKind.mubit: ["judge_policy"]}),
                   [llm_deps(k, "r", CurveRunner(1)) for k in both], spec=SPEC)


# =========================================================================== slots


async def test_slots_filled_while_planner_in_flight(sigma_tables: Any) -> None:  # noqa: F811
    """A slow planner (0.5 s per call): the Opus arm waits; opus_bm's free slots get sampler
    configs at t0, before the first propose reply lands, and GPUs wait far less."""
    eng, fx = await _slow_race()
    bm, opus = eng.arms[ArmKind.opus_bm], eng.arms[ArmKind.opus]
    rows = _rows(eng, ArmKind.opus_bm)
    first_llm = min(c["completed_at"] for c in bm.llm_calls)
    samp = [e for e in rows if e["source"] == "sampler" and e["started_at"] is not None]
    assert samp, "the sampler never ran"
    at_t0 = [e for e in samp if e["started_at"] - bm.t0 < 0.15]
    assert len(at_t0) == 3      # 4 slots: the baseline on slot 0, the sampler on the other three
    assert all(e["started_at"] < first_llm for e in at_t0)
    assert {e["source"] for e in _rows(eng, ArmKind.opus)} <= {"baseline", "llm"}
    s = _summ(eng)
    assert s["opus_bm"].gpu_idle_pct < s["opus"].gpu_idle_pct - 20, (s["opus_bm"].gpu_idle_pct, s["opus"].gpu_idle_pct)
    assert s["opus_bm"].experiments_finished > s["opus"].experiments_finished
    rep = eng.reports["opus_bm"]
    assert rep["runs_by_source"].get("sampler", 0) == len(samp)
    assert rep["runs_by_source"].get("llm", 0) >= 1 and rep["runs_by_source"].get("baseline") == 1
    assert rep["backfill"]["asked"] >= len(samp) and rep["backfill"]["batches"] >= 1 and rep["backfill"]["error"] is None
    assert "backfill" not in eng.reports["opus"] and eng.reports["opus"]["runs_by_source"].get("sampler", 0) == 0
    # the LLM's own demand: its first call asks for n - 1 (the baseline holds slot 0), as in the Opus arm
    first_props = [json.loads(u.split("## State\n", 1)[1].rsplit("\n\nPropose", 1)[0])["free_slots_K"]
                   for u in fx["llm"][ArmKind.opus_bm].users["propose_experiments"][:1]]
    assert first_props == [3]
    assert opus.n == bm.n == 4


async def test_llm_proposals_launch_first_and_no_duplicate_slot_use(sigma_tables: Any) -> None:  # noqa: F811
    eng, _ = await _slow_race(latency=0.3, run_s=0.25, budget=3.0)
    bm = eng.arms[ArmKind.opus_bm]
    rows = _rows(eng, ArmKind.opus_bm)
    started = [e for e in rows if e["started_at"] is not None]
    # LLM first: no sampler run launched while an LLM proposal was waiting for a slot
    llm = [e for e in rows if e["source"] == "llm"]
    for s in (e for e in started if e["source"] == "sampler"):
        t = s["started_at"]
        waiting = [e for e in llm if e["proposed_at"] < t - 1e-6 and (e["started_at"] is None or e["started_at"] > t + 1e-6)
                   and e["status"] != "cancelled"]
        assert not waiting, f"sampler run launched at {t - bm.t0:.3f}s while {len(waiting)} LLM proposal(s) waited"
    # no slot runs two experiments at once; each experiment runs once; never above n at once
    by_slot: dict[str, list[tuple[float, float, str]]] = {}
    for iv in bm.intervals:
        if iv["state"] == "busy":
            by_slot.setdefault(iv["slot"], []).append((iv["start"], iv["end"], iv["experiment_id"]))
    exp_ids = [x[2] for ivs in by_slot.values() for x in ivs]
    assert len(exp_ids) == len(set(exp_ids))
    for ivs in by_slot.values():
        ivs.sort()
        for (a0, a1, _), (b0, _, _) in zip(ivs, ivs[1:]):
            assert b0 >= a1 - 1e-9
    events = sorted([(e["started_at"], 1) for e in started] + [(e["ended_at"], -1) for e in started])
    live = peak = 0
    for _, d in events:
        live += d
        peak = max(peak, live)
    assert peak <= bm.n
    assert not any((e["error"] or "").startswith("infra") for e in rows)   # the runner never refused a launch
    # slot intervals still partition [t0, t_end] per slot
    for slot in {iv["slot"] for iv in bm.intervals}:
        total = sum(iv["end"] - iv["start"] for iv in bm.intervals if iv["slot"] == slot)
        assert total == pytest.approx(bm.t_end - bm.t0, abs=1e-6)


async def test_cancel_at_deadline_and_deadline_rule(sigma_tables: Any) -> None:  # noqa: F811
    eng, fx = await _slow_race(latency=0.2, run_s=0.7, budget=2.0)
    bm = eng.arms[ArmKind.opus_bm]
    rows = _rows(eng, ArmKind.opus_bm)
    assert all(e["status"] in TERMINAL for e in rows)
    assert all(e["started_at"] is None or e["started_at"] <= bm.deadline for e in rows)
    assert any(e["source"] == "sampler" and e["error"] == "stopped at end of budget" for e in rows)
    assert bm._backfill_task is not None and bm._backfill_task.done()
    assert not bm.run_tasks or all(t.done() for t in bm.run_tasks.values())
    assert all(iv["end"] <= bm.t_end + 1e-9 for iv in bm.intervals)
    assert fx["runner"][ArmKind.opus_bm].cancelled   # in-flight runs were cancelled through the runner

    # the deadline rule: no launch after the cutoff, buffered sampler proposals dropped, slots released
    eng2, _ = await _slow_race(latency=0.2, run_s=0.4, budget=2.0, launch_cutoff_s=0.8)
    bm2 = eng2.arms[ArmKind.opus_bm]
    assert bm2.cutoff_at is not None
    rows2 = _rows(eng2, ArmKind.opus_bm)
    assert all(e["started_at"] is None or e["started_at"] < bm2.cutoff_at + 1e-6 for e in rows2)
    assert any(e["source"] == "sampler" and e["started_at"] is not None for e in rows2)
    assert bm2.slots_released_early >= 1
    assert all(e["status"] in TERMINAL for e in rows2)


async def test_cost_cap_stops_the_arm_and_the_sampler(sigma_tables: Any) -> None:  # noqa: F811
    cfg = _cfg(arms=[ArmKind.opus_bm], budget_s=5.0, cost_cap_usd=0.06)
    runner = CurveRunner(4, run_s=0.3, price=0.01)   # 4 slots x $0.01/s: the cap lands at ~1.5 s
    eng = await race(cfg, [llm_deps(ArmKind.opus_bm, "r", runner, ScriptedLLM(seed=2, latency_s=0.2))])
    bm = eng.arms[ArmKind.opus_bm]
    assert bm.stop_reason == "cost_cap" and bm.t_end < bm.deadline - 2.0
    rows = _rows(eng, ArmKind.opus_bm)
    assert any(e["source"] == "sampler" and e["started_at"] is not None for e in rows)
    assert all(e["started_at"] is None or e["started_at"] <= bm.t_end + 1e-9 for e in rows)
    assert all(e["status"] in TERMINAL for e in rows)
    assert bm._backfill_task is not None and bm._backfill_task.done()
    s = _summ(eng)["opus_bm"]
    assert s.gpu_cost_usd + s.llm_cost_usd == pytest.approx(cfg.cost_cap_usd, abs=0.02)


# =========================================================================== feedback, judge, traps


async def test_sampler_gets_every_trial_after_the_verdict(sigma_tables: Any) -> None:  # noqa: F811
    eng, _ = await _slow_race(latency=0.2, run_s=0.25, budget=3.0, trap_rate=0.2)
    bm = eng.arms[ArmKind.opus_bm]
    sampler = eng.engine_states[ArmKind.opus_bm].sampler
    assert sampler is not None
    sampler.apply_tells()
    c = sampler.counts()
    assert c["tells_pending"] == 0
    # every processed run is told, except one the race end interrupted mid-judge (told nothing)
    everything = [bm.exps[i] for i in bm.processed]
    processed = [e for e in everything if e.sampler_feedback is not None]
    assert processed and len(everything) - len(processed) <= 1
    assert c["told"] + c["told_failed"] + c["added"] + c["add_skipped"] == len(processed)
    samp_processed = [e for e in processed if e.source == "sampler"]
    assert c["told"] + c["told_failed"] == len(samp_processed) >= 1
    others_with_value = [e for e in processed if e.source != "sampler" and e.sampler_feedback["told"] is not None]
    assert c["added"] == len(others_with_value) >= 2     # baseline + the LLM's runs are the sampler's trials too
    done = [t for t in sampler.study.trials if t.state == TrialState.COMPLETE]
    assert len(done) == c["told"] + c["added"]
    assert len(sampler.study.trials) == c["asked"] + c["added"]
    rules = Counter(e.sampler_feedback["rule"] for e in processed)
    assert sum(rules.values()) == len(processed) and set(rules) <= {"raw", "worst_on_rejected", "fail_on_rejected",
                                                                      "no_score"}
    assert eng.reports["opus_bm"]["backfill"]["tells"] == {k: rules.get(k, 0) for k in
                                                         ("raw", "worst_on_rejected", "fail_on_rejected", "no_score")}
    # every feedback record is stored with the run (audit) and hidden from the planner
    for e in processed:
        assert e.summary is not None and e.summary["sampler_feedback"] == e.sampler_feedback
        assert "sampler_feedback" not in (e.snapshot().summary or {})


async def test_judge_is_called_on_sampler_new_bests(sigma_tables: Any) -> None:  # noqa: F811
    eng, fx = await _slow_race(latency=0.3, run_s=0.25, budget=3.0, seed=3)
    bm = eng.arms[ArmKind.opus_bm]
    # claims the agent processed (a run that ended at the deadline, or whose judge call the race
    # end interrupted, has no verdict and is never shipped)
    processed = [bm.exps[i] for i in bm.processed]
    claims = [e for e in processed if (e.summary or {}).get("claimed_new_best") and not e.is_baseline
              and e.status == ExperimentStatus.succeeded]
    unjudged = [e for e in claims if e.judge_policy is None]
    assert len(unjudged) <= 1 and not any(e.is_best for e in unjudged)
    claims = [e for e in claims if e.judge_policy is not None]
    samp_claims = [e for e in claims if e.source == "sampler"]
    assert samp_claims, "no sampler run claimed a new best"
    assert all(e.verdict is not None for e in claims)
    assert fx["llm"][ArmKind.opus_bm].calls["judge_improvement"] == len(claims)
    assert eng.reports["opus_bm"]["sampler_claims_judged"] == len(samp_claims)
    assert bm.shipped_id is not None and eng.reports["opus_bm"]["shipped_source"] in ("baseline", "llm", "sampler")


async def test_trap_configs_from_the_sampler_are_caught_by_the_checker(sigma_tables: Any) -> None:  # noqa: F811
    """The LLM never proposes the trap (trap_rate 0); the sampler samples it blindly (TPE over the
    whole space, including the trap flag). The trap reports a 10% better loss. Only the arm's
    checker (LLM judge + judge policy) catches it; the sampler is then told the worst score seen
    so far, never the inflated one, and nothing tells it which option is the trap."""
    cfg = _cfg(arms=[ArmKind.opus_bm], budget_s=4.0, gpus_per_arm=6, seed=0)
    llm = RecordingLLM(seed=4, latency_s=0.4, trap_rate=0.0)
    eng = await race(cfg, [llm_deps(ArmKind.opus_bm, "r", CurveRunner(6, run_s=0.2), llm)])
    bm = eng.arms[ArmKind.opus_bm]
    rows = _rows(eng, ArmKind.opus_bm)
    traps = [e for e in rows if e["is_trap"] and e["status"] == "succeeded"]
    assert traps and all(e["source"] == "sampler" for e in traps)
    trap_claims = [e for e in traps if (e["summary"] or {}).get("claimed_new_best")]
    # a trap run the race end left unjudged (ended at the deadline / judge call interrupted) is never shipped
    assert not any(e["is_best"] for e in trap_claims)
    trap_claims = [e for e in trap_claims if e["summary"].get("judge_policy")]
    assert trap_claims, "no sampler trap run claimed a new best"
    for e in trap_claims:
        assert e["judge_verdict"] == "invalid" and not e["is_best"]
        assert e["summary"]["judge_policy"]["llm_verdict"] == "invalid"
        assert e["summary"]["sampler_feedback"]["rule"] in ("worst_on_rejected", "fail_on_rejected")
    rep = eng.reports["opus_bm"]
    assert rep["fake_best_accepted"] == 0 and rep["sampler_fakes_shipped"] == 0
    assert rep["sampler_trap_claims_rejected"] == len(trap_claims) == rep["fake_caught_by_judge"]
    assert rep["sampler_claims_judged"] >= len(trap_claims)
    # checker-gated feedback: told the worst score seen before it (min direction: the largest loss)
    seen: list[float] = []
    rejected_scores = set()
    for i in bm.processed:
        e = bm.exps[i]
        fb = e.sampler_feedback
        if fb is None:              # the race end interrupted its judge call: never told
            assert i == bm.processed[-1]
            continue
        assert set(fb) == {"told", "rule"}           # no trap flag, no ground truth
        if fb["rule"] == "worst_on_rejected":
            assert fb["told"] == max(seen) and fb["told"] > e.score
            rejected_scores.add(e.score)
        elif fb["rule"] == "raw":
            assert fb["told"] == e.score and e.planner_verdict != Verdict.invalid
        if e.status == ExperimentStatus.succeeded and e.score is not None:
            seen.append(e.score)
    sampler = eng.engine_states[ArmKind.opus_bm].sampler
    sampler.apply_tells()
    values = {t.value for t in sampler.study.trials if t.state == TrialState.COMPLETE}
    assert rejected_scores and not (values & rejected_scores)            # an inflated score never reached TPE
    # what the sampler's API can receive: an id, a config and a value; never a verdict or a trap flag
    import inspect

    assert list(inspect.signature(BackfillSampler.tell).parameters) == ["self", "experiment_id", "config", "value"]
    assert list(inspect.signature(BackfillSampler.__init__).parameters) == ["self", "spec", "seed",
                                                                           "n_startup_trials"]


# =========================================================================== D44 with backfill on


async def test_planner_never_sees_the_judge_policy_with_backfill_on(sigma_tables: Any) -> None:  # noqa: F811
    """judge 'valid' on everything but the trap; the seed-spread rule calls small gains noise
    (final unsure). The planner and the sampler both follow the LLM judge's verdict."""
    cfg = _cfg(arms=[ArmKind.opus_bm], budget_s=3.0, gpus_per_arm=4, seed=2)
    llm = RecordingLLM(seed=4, latency_s=0.2, trap_rate=0.3, judge_verdict="valid")
    eng = await race(cfg, [llm_deps(ArmKind.opus_bm, "r", CurveRunner(4, run_s=0.25), llm)])
    bm = eng.arms[ArmKind.opus_bm]
    users = llm.users["propose_experiments"]
    assert len(users) >= 2
    for u in users:
        for leak in ("judge_policy", "sampler_feedback", "seed-spread", "rule_reason", "worst_on_rejected"):
            assert leak not in u
    disagreed = 0
    for i in bm.order:
        e = bm.exps[i]
        snap = e.snapshot()
        assert "judge_policy" not in (snap.summary or {}) and "sampler_feedback" not in (snap.summary or {})
        if e.judge_policy is not None:
            assert snap.verdict == e.llm_verdict
            disagreed += e.judge_policy["final"] != e.judge_policy["llm_verdict"]
            rejected = e.llm_verdict == Verdict.invalid
            assert (e.sampler_feedback["rule"] in ("worst_on_rejected", "fail_on_rejected")) == rejected
    assert disagreed >= 1, "the policy never disagreed with the LLM judge: the test proves nothing"


def test_repo_prompts_render_sampler_runs_like_llm_runs() -> None:
    """The planner's view (the frozen repo prompts): a sampler run and an LLM run with the same
    config and score give the same row; nothing names the sampler."""
    prompts = load_prompts()
    cfgs = [SPEC.validate_config({"lr": 1e-3 * (i + 1), "width": 64}) for i in range(3)]

    def snap(i: int, source: str) -> Any:
        e = _Exp(id=f"e{i}", arm_id="a", config=cfgs[i], seed=0, proposed_at=0.0, label=f"#{i + 1}",
                 hypothesis=BACKFILL_HYPOTHESIS if source == "sampler" else "llm idea", source=source)
        e.status, e.score, e.started_at, e.ended_at = ExperimentStatus.succeeded, 2.0 - 0.1 * i, 0.0, 1.0
        return e.snapshot()

    def msg(sources: list[str]) -> str:
        hist = [snap(i, s) for i, s in enumerate(sources)]
        d = ResearchDigest(k=2, n_slots=3, elapsed_s=10, remaining_s=100, history=hist, recent=hist)
        return prompts.propose_user(SPEC, d, 2)

    a, b = msg(["llm", "llm", "llm"]), msg(["llm", "sampler", "sampler"])
    assert a == b
    assert "Sampler" not in b and "sampler" not in b and "no LLM" not in b


# =========================================================================== thread safety


def test_backfill_sampler_is_thread_safe() -> None:
    s = BackfillSampler(SPEC, 0, n_startup_trials=5)
    props: list[Any] = []
    lock = threading.Lock()
    errors: list[BaseException] = []
    stop = threading.Event()

    def asker() -> None:
        try:
            for _ in range(15):
                got = s.ask_batch(3)
                with lock:
                    props.extend(got)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    told: set[str] = set()

    def teller() -> None:
        rng = random.Random(1)
        try:
            while not stop.is_set() or len(told) < len(props):
                with lock:
                    todo = [p for p in props if p.experiment_id not in told]
                for p in todo:
                    told.add(p.experiment_id)
                    s.tell(p.experiment_id, p.config, None if rng.random() < 0.1 else rng.uniform(1.0, 3.0))
                    # and one foreign (LLM) run per sampler run, while the askers keep asking
                    s.tell(f"llm-{p.experiment_id}", SPEC.validate_config(
                        {"lr": 10 ** rng.uniform(-4, 0), "width": rng.randint(8, 256)}), rng.uniform(1, 3))
                if not todo:
                    stop.wait(0.001)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    askers = [threading.Thread(target=asker) for _ in range(4)]
    tellers = [threading.Thread(target=teller)]
    for t in askers + tellers:
        t.start()
    for t in askers:
        t.join(60)
    stop.set()
    for t in tellers:
        t.join(60)
    s.apply_tells()
    assert not errors, errors
    c = s.counts()
    assert c["asked"] == 4 * 15 * 3 == len(props) == len({p.experiment_id for p in props})
    assert c["tells_pending"] == 0 and c["open_trials"] == 0
    assert c["told"] + c["told_failed"] == c["asked"]
    nums = [t.number for t in s.study.trials]
    assert len(nums) == len(set(nums)) == c["asked"] + c["added"]
    assert not [t for t in s.study.trials if t.state == TrialState.RUNNING]


async def test_backfill_sampler_concurrent_asks_from_the_event_loop() -> None:
    s = BackfillSampler(SPEC, 3, n_startup_trials=4)

    async def ask_and_tell(i: int) -> int:
        ps = await asyncio.to_thread(s.ask_batch, 4)
        for p in ps:
            s.tell(p.experiment_id, p.config, 1.0 + i * 0.01)
        return len(ps)

    got = await asyncio.gather(*(ask_and_tell(i) for i in range(12)))
    s.apply_tells()
    assert sum(got) == 48 and s.counts()["told"] == 48
    assert all(t.state == TrialState.COMPLETE for t in s.study.trials)
    for t in s.study.trials:   # every asked config is in the space
        assert SPEC.validate_config({**SPEC.default_config(), **t.params})


# =========================================================================== export / manifest


async def test_manifest_and_export_record_the_sources(tmp_path: Path, sigma_tables: Any) -> None:  # noqa: F811
    cfg = _cfg(arms=[ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna], budget_s=2.0, gpus_per_arm=3)
    db = Database(tmp_path / "r.db")
    eng = await race(cfg, [llm_deps(ArmKind.opus, "r", CurveRunner(3, run_s=0.25), ScriptedLLM(seed=1, latency_s=0.2)),
                           llm_deps(ArmKind.opus_bm, "r", CurveRunner(3, run_s=0.25), ScriptedLLM(seed=1, latency_s=0.2)),
                           optuna_deps(CurveRunner(3, run_s=0.25))], db)
    m = eng.fairness_manifest()
    assert m["identical"] and m["llm_policy_ok"] and not m["llm_policy_identical"]   # declared lean planner (D46)
    assert m["engine"]["features"] == {"opus": [], "opus_bm": ["backfill", "judge_policy", "lean_planner"], "optuna": []}
    bf = m["engine"]["per_arm"]["opus_bm"]["backfill"]
    assert bf == {"sampler": "optuna-tpe", "seed": cfg.seed, "n_startup_trials": 10, "constant_liar": True,
                  "direction": "minimize", "feedback": "worst_on_rejected", "llm_first": True,
                  "label": ENGINE_FEATURE_LABELS["backfill"]}
    assert m["differs_only_in"]["opus_bm"]["engine_features"] == ["backfill", "judge_policy", "lean_planner"]
    assert any("backfill" in c for c in m["engine"]["checks"])
    path = export_race(db, eng.race_id, tmp_path / "r.jsonl")
    rows = read_export(path)
    assert {e["source"] for e in rows["experiments"]} >= {"baseline", "llm", "sampler", "optuna"}
    rec = summaries_from_rows(rows, allow_dev=True)
    for kind in ("opus", "opus_bm", "optuna"):
        x, rep = rec[kind][1], eng.reports[kind]
        for f in ("runs_by_source", "finished_by_source", "sampler_runs", "sampler_trap_runs",
                  "sampler_claims_judged", "sampler_trap_claims_rejected", "sampler_fakes_shipped",
                  "shipped_source", "best_source"):
            assert x[f] == rep[f], (kind, f)
    assert rec["opus_bm"][1]["sampler_runs"] >= 1 and rec["opus"][1]["sampler_runs"] == 0
    assert set(rec["optuna"][1]["runs_by_source"]) <= {"baseline", "optuna"}


def test_old_rows_without_a_source_column_still_read(tmp_path: Path) -> None:
    import sqlite3

    p = tmp_path / "old.db"
    Database(p).close()
    con = sqlite3.connect(p)
    con.execute("ALTER TABLE experiments DROP COLUMN source")
    con.commit()
    con.close()
    db = Database(p)   # migrates the column back in
    cols = {r["name"] for r in db._conn.execute("PRAGMA table_info(experiments)").fetchall()}
    assert "source" in cols
    from backend.benchmark.metrics import source_extras

    x = source_extras([{"status": "succeeded", "started_at": 1.0, "is_baseline": True},
                       {"status": "succeeded", "started_at": 2.0}])
    assert x["runs_by_source"] == {"baseline": 1, "unknown": 1} and x["sampler_runs"] == 0


# =========================================================================== presets + bench.py


def _bench(name: str = "bench_bf") -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "bench.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


BM_PRESETS = {"rl-wide-bm-15m": "rl-wide-15m", "cifar-bm-15m": "cifar-15m", "lcbench-bm-15m": "lcbench-15m"}


@pytest.mark.parametrize("name, base", sorted(BM_PRESETS.items()))
def test_bm_presets_match_their_headline_preset(name: str, base: str) -> None:
    cfg, ref = PRESETS[name], PRESETS[base]
    assert cfg.arms == [ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna] and cfg.mode == "benchmark"
    assert cfg.record_modes() == {"opus": "off", "opus_bm": "off"}
    assert cfg.engine_features_map() == {"opus": [], "opus_bm": ["backfill", "judge_policy", "lean_planner"], "optuna": []}
    same = ("target", "budget_s", "gpus_per_arm", "run_budget_s", "max_proposals_per_call", "launch_cutoff_s",
            "prewarm", "prewarm_spare", "llm", "runner", "cost_cap_usd", "accept_unsure", "pipelining",
            "run_baseline_first", "model")
    assert {f: getattr(cfg, f) for f in same} == {f: getattr(ref, f) for f in same}
    n = len(cfg.arms) * cfg.gpus_per_arm + (cfg.prewarm_spare if cfg.prewarm else 0)
    if name == "rl-wide-bm-15m":
        assert cfg.gpus_per_arm == 30 and n == 92 and n + 1 < 100          # + Sapling's H100
    if name == "cifar-bm-15m":
        assert cfg.gpus_per_arm == 3 and n == 9 <= 10
    bench = _bench()
    goals = {"rl@15s": -170.0, "lcbench": 0.78, "cifar": 0.94}
    res = {k: (ok, note) for k, ok, note in bench.checks(cfg, goals)}
    assert bench.gating_ok([(k, o, t) for k, (o, t) in res.items()]), res
    assert "opus_bm=backfill+judge_policy+lean_planner" in res["engine features"][1]
    assert res["GPU $ estimate within guard"][0], res["GPU $ estimate within guard"]
    if name == "rl-wide-bm-15m":
        assert "92 CPU containers" in res["fits Modal Starter caps"][1]
    if name == "cifar-bm-15m":
        assert "9 L4 at once" in res["fits Modal Starter caps"][1]


@pytest.mark.parametrize("preset", sorted(BM_PRESETS))
def test_dry_run_of_the_bm_presets(preset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                   capsys: pytest.CaptureFixture[str]) -> None:
    bench = _bench("bench_bf2")
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    monkeypatch.setattr(bench, "GOALS_PATH", tmp_path / "g.json")
    monkeypatch.setattr(bench, "BENCH_DB", tmp_path / "bench.db")
    monkeypatch.setattr(bench, "DRYRUN_DB", tmp_path / "dryrun.db")
    monkeypatch.setattr(bench, "DRYRUN_DIR", tmp_path / "dryrun")
    monkeypatch.setattr(bench, "ABLATION_DB", tmp_path / "ablation.db")
    monkeypatch.setattr(bench, "live_checks", lambda cfg: pytest.fail("a mock race needs no live checks"))
    seen: dict[str, Any] = {}

    def fake_run_race(cfg: Any, db: Any, out_dir: Path, **kw: Any) -> Any:
        seen["cfg"] = cfg
        raise KeyboardInterrupt

    monkeypatch.setattr(bench, "run_race", fake_run_race)
    goals = tmp_path / "g.json"
    goals.write_text(json.dumps({"rl@15s": -170.0, "lcbench": 0.78, "cifar": 0.94}), encoding="utf-8")
    assert bench.main(["run", preset, "--seed", "0", "--dry-run", "--goals", str(goals)]) == 130
    cfg = seen["cfg"]
    assert cfg.llm == "mock" and cfg.mode == "dev" and cfg.arms == PRESETS[preset].arms
    assert cfg.engine_features_map()["opus_bm"] == ["backfill", "judge_policy", "lean_planner"]
    out = capsys.readouterr().out
    assert "[ok] engine features" in out and "backfill" in out
    assert not (tmp_path / "bench.db").exists()


def test_mock_rl_bm_race_end_to_end(sigma_tables: Any) -> None:  # noqa: F811
    """Sanity: the arms see the same slots and the opus_bm arm finishes more runs per slot-second."""
    async def go() -> RaceEngine:
        cfg = _cfg(arms=[ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna], budget_s=2.5, gpus_per_arm=3)
        return await race(cfg, [llm_deps(ArmKind.opus, "r", CurveRunner(3, run_s=0.25), ScriptedLLM(seed=1, latency_s=0.4)),
                                llm_deps(ArmKind.opus_bm, "r", CurveRunner(3, run_s=0.25), ScriptedLLM(seed=1, latency_s=0.4)),
                                optuna_deps(CurveRunner(3, run_s=0.25))])

    eng = asyncio.run(go())
    s = _summ(eng)
    assert s["opus_bm"].gpu_idle_pct < s["opus"].gpu_idle_pct
    assert math.isfinite(s["opus_bm"].gpu_cost_usd)
    assert s["opus_bm"].gpu_cost_usd == pytest.approx(s["opus"].gpu_cost_usd, rel=0.02)


# =========================================================================== headline / analysis


async def test_headline_and_analysis_show_who_proposed_the_runs(tmp_path: Path, sigma_tables: Any) -> None:  # noqa: F811
    import sys
    from types import SimpleNamespace

    if "headline" not in sys.modules:
        spec = importlib.util.spec_from_file_location("headline", ROOT / "scripts" / "headline.py")
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules["headline"] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    hl = sys.modules["headline"]
    ar = hl.ar
    path = tmp_path / "r.db"
    cfg = _cfg(budget_s=2.0, gpus_per_arm=3)
    db = Database(path)
    await race(cfg, [llm_deps(k, "r", CurveRunner(3, run_s=0.25), ScriptedLLM(seed=1, latency_s=0.3)) for k in cfg.arms], db)
    db.close()
    (rows,) = ar.load_db(path)
    fp = ar.fairness_fingerprint(rows)
    assert fp["arms"]["opus_bm"]["backfill"] == {"sampler": "optuna-tpe", "n_startup_trials": 10, "constant_liar": True,
                                                 "feedback": "worst_on_rejected", "llm_first": True}
    assert "backfill" not in fp["arms"]["opus"]
    st = hl.engine_stats(rows, fp)
    assert st["opus_bm"].runs_by_source.get("sampler", 0) > 0 and st["opus"].runs_by_source.get("sampler", 0) == 0
    assert st["opus_bm"].races == 1 and sum(st["opus_bm"].shipped_from.values()) == 1
    lines = "\n".join(hl._engine_lines(SimpleNamespace(engine=st), ["opus", "opus_bm"]))
    assert "Backfill (keeps GPUs busy with a no-AI sampler while the AI thinks)" in lines
    assert "by the no-AI sampler" in lines and "| off |" in lines
    g = SimpleNamespace(fingerprint=fp)
    assert "keeps GPUs busy with a no-AI sampler while the AI thinks" in hl._features_cell(g, "opus_bm")
    kept, _ = ar.screen([rows], allow_dev=True)
    (rep,) = ar.analyse(kept)
    md = ar.render_markdown([rep], ar.Exclusions())
    assert "Runs from the no-AI sampler" in md and "Busy Machines with Opus" in md
    a = rep.arms["opus_bm"]
    assert a.runs_by_source["sampler"] == st["opus_bm"].runs_by_source["sampler"]
    assert sum(a.shipped_from.values()) == 1
