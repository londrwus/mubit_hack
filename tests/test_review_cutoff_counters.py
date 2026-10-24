"""Review fix: proposals that land after the deadline-rule cutoff are dropped AND counted.

Before the fix ``_apply_cutoff`` did its work once: a propose call or a sampler ask that was in
flight when the cutoff passed added its proposals afterwards, the second ``_apply_cutoff`` call
returned at once, and they sat "pending" until the shutdown labelled them "stopped at end of
budget". ``launches_skipped_deadline`` / ``backfill_dropped`` undercounted them and their TPE
trials stayed open. In-process fakes only; no network, no Modal, no LLM spend.
"""

from __future__ import annotations

import time
from typing import Any

import backend.scheduler as sched
from backend.benchmark.configs import RaceConfig
from backend.core.types import ArmKind, ExperimentStatus
from backend.db import Database
from backend.scheduler import ArmRunner, EventBus, Proposal, RaceEngine
from test_busy_machines import (  # noqa: F401
    SPEC,
    CurveRunner,
    ScriptedLLM,
    llm_deps,
    sigma_tables,
)

DEADLINE_PREFIX = "deadline rule: not launched"


def _arm(n: int = 2) -> ArmRunner:
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus, ArmKind.opus_bm], budget_s=10.0, gpus_per_arm=n,
                     llm="mock", seed=1, launch_cutoff_s=9.0)
    runner = CurveRunner(n)
    t0 = sched.now()
    return ArmRunner(race_id="r", arm_id="a", deps=llm_deps(ArmKind.opus_bm, "r", runner), cfg=cfg, spec=SPEC,
                     slots=runner.slots(), t0=t0, deadline=t0 + 10.0, db=None, bus=EventBus(),
                     engine=sched.resolve_engine_state(cfg, ArmKind.opus_bm, SPEC))


def test_late_proposals_after_the_cutoff_are_dropped_and_counted(sigma_tables: Any) -> None:  # noqa: F811
    arm = _arm()
    assert arm.sampler is not None
    arm.cutoff_at = sched.now() - 0.5            # the cutoff has passed; the deadline has not
    arm._apply_cutoff(sched.now())               # the one-time work: nothing buffered yet
    assert arm.cutoff_applied and arm.slots_released_early == arm.n
    assert arm.launches_skipped_deadline == 0 and arm.backfill_dropped == 0
    # a planner reply and a sampler batch that were in flight at the cutoff land now
    late_llm = Proposal(config=SPEC.default_config(), hypothesis="late planner reply")
    (late_sampler,) = arm.sampler.ask_batch(1)
    assert arm._add_proposal(late_llm) and arm._add_proposal(late_sampler, source="sampler")
    arm._fill_slots()
    for pid in (late_llm.experiment_id, late_sampler.experiment_id):
        e = arm.exps[pid]
        assert e.status == ExperimentStatus.cancelled and e.started_at is None
        assert (e.error or "").startswith(DEADLINE_PREFIX)
    assert not arm.buffer and not arm.sampler_buffer
    assert arm.launches_skipped_deadline == 2 and arm.backfill_dropped == 1
    # the dropped sampler proposal is told as no score: its TPE trial closes (FAIL)
    arm.sampler.apply_tells()
    c = arm.sampler.counts()
    assert c["open_trials"] == 0 and c["told_failed"] == 1 and c["tells_pending"] == 0


def test_no_cutoff_configured_is_a_no_op(sigma_tables: Any) -> None:  # noqa: F811
    arm = _arm()
    arm.cutoff_at = None
    arm._add_proposal(Proposal(config=SPEC.default_config(), hypothesis="x"))
    arm._apply_cutoff(sched.now())
    assert len(arm.buffer) == 1 and not arm.cutoff_applied and arm.launches_skipped_deadline == 0


async def test_a_sampler_batch_that_straddles_the_cutoff_is_counted(sigma_tables: Any) -> None:  # noqa: F811
    """End to end: the sampler's second ask starts before the cutoff and returns after it."""
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus, ArmKind.opus_bm], budget_s=2.0, gpus_per_arm=4,
                     llm="mock", seed=1, launch_cutoff_s=0.8)
    deps = [llm_deps(k, "r", CurveRunner(4, run_s=0.3), ScriptedLLM(seed=4, latency_s=5.0)) for k in cfg.arms]
    eng = RaceEngine(cfg, deps, spec=SPEC, db=Database(":memory:"), bus=EventBus(), tick_s=0.02)
    sampler = eng.engine_states[ArmKind.opus_bm].sampler
    assert sampler is not None
    orig = sampler.ask_batch
    calls: list[float] = []

    def slow_after_first(k: int) -> list[Proposal]:
        calls.append(sched.now())
        if len(calls) >= 2:   # hold this ask until the cutoff has passed (worker thread)
            arm = eng.arms[ArmKind.opus_bm]
            assert arm.cutoff_at is not None
            while sched.now() < arm.cutoff_at + 0.05:
                time.sleep(0.01)
        return orig(k)

    sampler.ask_batch = slow_after_first  # type: ignore[method-assign]
    await eng.run()
    bm = eng.arms[ArmKind.opus_bm]
    assert bm.cutoff_at is not None and len(calls) >= 2 and calls[1] < bm.cutoff_at
    rows = [bm.exps[i] for i in bm.order if bm.exps[i].source == "sampler"]
    late = [e for e in rows if e.proposed_at >= bm.cutoff_at]
    assert late, "the straddling batch must land after the cutoff"
    assert all(e.status == ExperimentStatus.cancelled and (e.error or "").startswith(DEADLINE_PREFIX) for e in late)
    # nothing proposed in budget and never launched is labelled "stopped at end of budget"
    never = [e for e in rows if e.started_at is None and e.proposed_at < bm.deadline]
    assert all((e.error or "").startswith(DEADLINE_PREFIX) for e in never)
    assert bm.backfill_dropped == len(never) >= len(late)
    assert bm.launches_skipped_deadline >= bm.backfill_dropped
    rep = eng.reports["opus_bm"]
    assert rep["backfill"]["dropped_by_deadline_rule"] == bm.backfill_dropped
    assert rep["launches_skipped_deadline"] == bm.launches_skipped_deadline
    # the shutdown applied the queued tells: no dropped proposal keeps a TPE trial open
    assert rep["backfill"]["tells_pending"] == 0
    assert all(e.id not in sampler._trials for e in never)
