"""Review fix: the planner's 'current best config' follows human corrections.

Before the fix ``ExperimentSnapshot`` had no ``flagged_fake``, so ``RepoPrompts._row`` hid a
row's score only for an LLM-judge ``invalid``. After a human "fake improvement" correction the
engine dropped the run from ``best_id`` (``_eligible``) but the planner was still told it was the
current best config, and the lean (diff) planner expanded every proposal on top of it, trap
setting included. In-process fakes only; no network, no LLM spend.
"""

from __future__ import annotations

from typing import Any

from backend.agent.loop import DefaultPrompts, RepoPrompts, _exp_brief
from backend.agent.prompts.build import ProposeDiffOutput, fmt_config
from backend.benchmark.configs import RaceConfig
from backend.core.types import ArmKind, CorrectionLabel, ExperimentStatus, Verdict
from backend.scheduler import ExperimentSnapshot, ResearchDigest
from test_busy_machines import (  # noqa: F401
    SPEC,
    TRAP,
    CurveRunner,
    ScriptedLLM,
    llm_deps,
    race,
    sigma_tables,
)

ORDER = list(SPEC.config_space)


def _snap(i: int, score: float, *, trap: bool = False, **kw: Any) -> ExperimentSnapshot:
    cfg = {**SPEC.default_config(), "lr": 10 ** -(i + 1), "width": 16 * (i + 1), TRAP: trap}
    return ExperimentSnapshot(id=f"e{i}", label=f"#{i}", config=cfg, status=ExperimentStatus.succeeded,
                              score=score, verdict=kw.pop("verdict", Verdict.valid), **kw)


def _digest(history: list[ExperimentSnapshot]) -> ResearchDigest:
    return ResearchDigest(k=2, n_slots=2, elapsed_s=10.0, remaining_s=100.0, history=history)


def _best_line(user: str) -> str:
    return next(ln for ln in user.splitlines() if ln.startswith("current best config: "))


def test_flagged_fake_is_not_the_current_best_config() -> None:
    honest, fake, other = _snap(1, 1.30), _snap(2, 1.00, trap=True, flagged_fake=True), _snap(3, 1.50)
    d = _digest([honest, fake, other])
    for version in ("v1", "v2"):
        p = RepoPrompts(version)
        user = p.propose_user(SPEC, d, 2)
        assert _best_line(user) == "current best config: " + fmt_config(honest.config, ORDER)
        assert "human: fake improvement, not counted" in user
        row = p._row(fake)
        assert row.score is None and row.note == "human: fake improvement, not counted"


def test_lean_planner_diffs_expand_on_the_honest_best_not_the_flagged_trap() -> None:
    honest, fake = _snap(1, 1.30), _snap(2, 1.00, trap=True, flagged_fake=True)
    p = RepoPrompts("v2")
    reply = ProposeDiffOutput.model_validate({"experiments": [{"changes": {"width": 64}, "hypothesis": "wider"}]})
    out = p.expand_proposals(SPEC, _digest([honest, fake]), reply)
    (exp,) = out.experiments
    assert exp.config == {**honest.config, "width": 64}
    assert exp.config[TRAP] is False and not SPEC.is_trap(exp.config)


def test_human_genuine_overrides_an_llm_invalid_verdict() -> None:
    judged_invalid = _snap(1, 1.00, verdict=Verdict.invalid)
    assert RepoPrompts("v1")._row(judged_invalid).score is None          # the judge's call stands...
    genuine = _snap(1, 1.00, verdict=Verdict.invalid, override_genuine=True)
    row = RepoPrompts("v1")._row(genuine)                                # ...until a human marks it genuine
    assert row.score == 1.00 and row.note is None
    user = RepoPrompts("v1").propose_user(SPEC, _digest([_snap(2, 1.4), genuine]), 2)
    assert _best_line(user) == "current best config: " + fmt_config(genuine.config, ORDER)


def test_unflagged_rows_render_as_before() -> None:
    rows = [_snap(1, 1.3), _snap(2, 1.0, verdict=Verdict.invalid), _snap(3, 1.5)]
    p = RepoPrompts("v1")
    assert [(r.score, r.note) for r in map(p._row, rows)] == [(1.3, None), (None, "judge: invalid, not counted"),
                                                              (1.5, None)]


async def test_a_correction_in_a_live_race_reaches_the_planner_digest(sigma_tables: Any) -> None:  # noqa: F811
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus], budget_s=1.5, gpus_per_arm=2, llm="mock", seed=1)
    eng = await race(cfg, [llm_deps(ArmKind.opus, "r", CurveRunner(2, run_s=0.2), ScriptedLLM(seed=2, trap_rate=0.0))])
    arm = eng.arms[ArmKind.opus]
    before = arm.best_id
    assert before is not None
    assert arm.apply_correction(before, CorrectionLabel.fake_improvement)
    assert arm.best_id != before
    digest = arm._digest(2)
    snap = next(s for s in digest.history if s.id == before)
    assert snap.flagged_fake and not snap.override_genuine
    assert all(s.id != before for s in digest.leaderboard)
    user = RepoPrompts("v2").propose_user(SPEC, digest, 2)
    flagged_cfg = fmt_config(arm.exps[before].config, ORDER)
    assert _best_line(user) != "current best config: " + flagged_cfg
    # the built-in prompt set marks it too in its recent-results brief (best / leaderboard were
    # already filtered by the engine)
    assert _exp_brief(snap)["human"] == "fake improvement, not counted"
    assert DefaultPrompts().propose_user(SPEC, digest, 2)
