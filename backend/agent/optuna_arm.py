"""Arm C: the no-LLM baseline (benchmark.md "Arms"). Same scheduler, Optuna TPE (seeded)
as the proposer over the target's config space. No LLM calls, so no diagnose and no judge:
every claimed new best is accepted, contaminated or not (that is the point of comparing
against it; the headline metrics exclude ground-truth trap runs for every arm alike).

Batching: ``propose(k)`` asks the study for k trials at once (``constant_liar`` keeps
parallel asks from collapsing onto one point). Results come back through ``observe``.
The scheduler's starting-code baseline is added to the study as a completed trial.

Early stopping: the arm gets the same median rule as the Busy Machines arms (engine code in
the scheduler, ``RaceConfig.features_for(optuna)`` = median_stop, flagged in the manifest):
the standard MedianPruner practice. A run the rule stopped is told to the study as PRUNED.
"""

from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Callable
from typing import Any

import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import TrialState

from backend.core.types import ArmKind, DiagnoseOutput, ExperimentStatus, JudgeOutput, LLMCallRecord
from backend.scheduler import ExperimentSnapshot, JudgeClaim, Proposal, ResearchDigest
from workers.targets.base import TargetSpec

optuna.logging.set_verbosity(optuna.logging.WARNING)

HYPOTHESIS = "Optuna TPE sample (no LLM)."


def distributions(spec: TargetSpec) -> dict[str, BaseDistribution]:
    out: dict[str, BaseDistribution] = {}
    for name, p in spec.config_space.items():
        if p.type == "bool":
            out[name] = CategoricalDistribution([False, True])
        elif p.type == "categorical":
            out[name] = CategoricalDistribution(list(p.choices or []))
        elif p.type == "int":
            assert p.low is not None and p.high is not None
            out[name] = IntDistribution(int(p.low), int(p.high), log=p.log)
        else:
            assert p.low is not None and p.high is not None
            out[name] = FloatDistribution(float(p.low), float(p.high), log=p.log)
    return out


class OptunaResearcher:
    kind = ArmKind.optuna

    def __init__(self, spec: TargetSpec, seed: int, *, n_startup_trials: int = 10) -> None:
        self.spec = spec
        self.seed = seed
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials, constant_liar=True)
        self.study = optuna.create_study(
            direction="minimize" if spec.direction == "min" else "maximize", sampler=sampler
        )
        self._dists = distributions(spec)
        self._trials: dict[str, optuna.trial.Trial] = {}
        self.on_call: Callable[[LLMCallRecord], None] = lambda _r: None

    async def propose(self, digest: ResearchDigest, k: int) -> list[Proposal]:
        # TPE's ask() is CPU-bound and grows with the study (measured on the rl-wide space:
        # propose(8) takes ~0.17 s at 100 trials and ~0.5 s at 600+). Run on the event loop
        # it would freeze every other arm in the same race (run-end detection, slot refills,
        # LLM latency timing) for that long. A worker thread keeps that time this arm's own
        # wall time only. Safe: the scheduler awaits propose() and calls observe() from the
        # same per-arm agent loop, so the study is never used from two threads at once. (The
        # backfill sampler of an LLM arm has two callers, so it uses BackfillSampler below.)
        return await asyncio.to_thread(self._ask_batch, k)

    def _ask_batch(self, k: int) -> list[Proposal]:
        out: list[Proposal] = []
        for _ in range(k):
            trial = self.study.ask()
            config = self.spec.validate_config(self.spec.suggest_optuna(trial))
            p = Proposal(config=config, hypothesis=HYPOTHESIS)
            self._trials[p.experiment_id] = trial
            out.append(p)
        return out

    def observe(self, exp: ExperimentSnapshot) -> None:
        ok = exp.status == ExperimentStatus.succeeded and exp.score is not None and math.isfinite(exp.score)
        trial = self._trials.pop(exp.id, None)
        if trial is not None:
            if ok:
                self.study.tell(trial, float(exp.score))  # type: ignore[arg-type]
            elif (exp.summary or {}).get("early_stop", {}).get("outcome") == "stopped":
                # The engine's median rule stopped it (MedianPruner semantics): a pruned trial.
                self.study.tell(trial, state=TrialState.PRUNED)
            else:
                self.study.tell(trial, state=TrialState.FAIL)
            return
        if ok:  # e.g. the scheduler's baseline run: add it as a completed trial
            params: dict[str, Any] = {k: exp.config[k] for k in self._dists if k in exp.config}
            self.study.add_trial(optuna.trial.create_trial(
                params=params, distributions={k: self._dists[k] for k in params}, value=float(exp.score),  # type: ignore[arg-type]
            ))

    async def diagnose(self, exp: ExperimentSnapshot) -> DiagnoseOutput | None:
        return None

    async def judge(self, claim: JudgeClaim) -> JudgeOutput | None:
        return None


# =========================================================================== backfill sampler

BACKFILL_HYPOTHESIS = "Sampler proposal (no LLM)."


class BackfillSampler:
    """The ``backfill`` engine feature's no-AI sampler (configs.ENGINE_FEATURE_LABELS: "keeps
    GPUs busy with a no-AI sampler while the AI thinks"). Optuna TPE over the target's config
    space, seeded, ``constant_liar`` like the Optuna arm. Never an LLM call.

    Feedback (the backfill gate's "checker-gated feedback"): the scheduler tells it every
    finished run of the arm, the LLM's and its own, AFTER the arm's checker has ruled on it. A
    claim the LLM judge rejected (planner verdict ``invalid``) is told the worst score seen so far
    (the scheduler decides the value); the sampler itself never sees a verdict, a trap flag or
    which option is the trap.

    Thread safety: two callers. The event loop queues tells (``tell``, cheap, under a lock); the
    scheduler's backfill task runs ``ask_batch`` in a worker thread, which applies the queued
    tells and then asks, under a second lock, so the study is only ever used by one thread at a
    time (Optuna's InMemoryStorage locks its own calls, but trial objects and ask/tell order do
    not belong to it).
    """

    kind = "sampler"

    def __init__(self, spec: TargetSpec, seed: int, *, n_startup_trials: int = 10) -> None:
        self.spec = spec
        self.seed = seed
        self.n_startup_trials = n_startup_trials
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials, constant_liar=True)
        self.study = optuna.create_study(
            direction="minimize" if spec.direction == "min" else "maximize", sampler=sampler
        )
        self._dists = distributions(spec)
        self._trials: dict[str, optuna.trial.Trial] = {}   # experiment id -> its open trial
        self._queue: list[tuple[str, dict[str, Any], float | None]] = []
        self._queue_lock = threading.Lock()
        self._study_lock = threading.Lock()
        self.asked = 0
        self.told = 0            # sampler trials told a value
        self.told_failed = 0     # sampler trials told FAIL (no score)
        self.added = 0           # other runs (LLM / baseline) added as completed trials
        self.add_skipped = 0     # other runs that could not be added (no score / outside the space)

    def manifest(self) -> dict[str, Any]:
        return {"sampler": "optuna-tpe", "seed": self.seed, "n_startup_trials": self.n_startup_trials,
                "constant_liar": True, "direction": self.study.direction.name.lower()}

    def counts(self) -> dict[str, int]:
        with self._queue_lock:
            pending = len(self._queue)
        return {"asked": self.asked, "told": self.told, "told_failed": self.told_failed, "added": self.added,
                "add_skipped": self.add_skipped, "tells_pending": pending, "open_trials": len(self._trials)}

    def tell(self, experiment_id: str, config: dict[str, Any], value: float | None) -> None:
        """Queue one finished run (any source). ``value`` None = no usable score (failed run)."""
        v = float(value) if value is not None and math.isfinite(float(value)) else None
        with self._queue_lock:
            self._queue.append((experiment_id, dict(config), v))

    def _drain(self) -> None:
        with self._queue_lock:
            batch, self._queue = self._queue, []
        for exp_id, config, value in batch:
            trial = self._trials.pop(exp_id, None)
            if trial is not None:
                if value is None:
                    self.study.tell(trial, state=TrialState.FAIL)
                    self.told_failed += 1
                else:
                    self.study.tell(trial, value)
                    self.told += 1
                continue
            if value is None:
                self.add_skipped += 1
                continue
            params: dict[str, Any] = {k: config[k] for k in self._dists if k in config}
            try:
                self.study.add_trial(optuna.trial.create_trial(
                    params=params, distributions={k: self._dists[k] for k in params}, value=value))
            except ValueError:   # a value outside the sampler's space: skip it, never crash the arm
                self.add_skipped += 1
                continue
            self.added += 1

    def apply_tells(self) -> None:
        """Apply queued tells now (tests / shutdown); same lock as ``ask_batch``."""
        with self._study_lock:
            self._drain()

    def ask_batch(self, k: int) -> list[Proposal]:
        """Apply the queued tells, then ask ``k`` configs. CPU-bound: call it in a worker thread."""
        out: list[Proposal] = []
        with self._study_lock:
            self._drain()
            for _ in range(max(0, k)):
                trial = self.study.ask()
                config = self.spec.validate_config(self.spec.suggest_optuna(trial))
                p = Proposal(config=config, hypothesis=BACKFILL_HYPOTHESIS)
                self._trials[p.experiment_id] = trial
                self.asked += 1
                out.append(p)
        return out
