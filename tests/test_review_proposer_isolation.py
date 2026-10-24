"""Review fix: the CPU-bound TPE proposers run in their own processes (GIL isolation).

The backfill sampler of a Busy Machines arm and the Optuna arm's study shared one interpreter
(and its GIL) with each other and with the event loop that times every arm's runs and LLM
replies. With ``isolation="process"`` (benchmark races) each study lives in a child process.
The behaviour must be identical apart from the process: same seed, same TPE settings, same
order of operations, so the same configs. Child processes on this machine only; no network.
"""

from __future__ import annotations

import math
import os
import random
from typing import Any

import pytest

from backend.agent.optuna_arm import BackfillSampler, OptunaResearcher
from backend.agent.study_worker import ISOLATION_ENV, StudyWorker, proposer_isolation
from backend.benchmark.configs import RaceConfig
from backend.core.types import ArmKind, ExperimentStatus
from backend.db import Database
from backend.scheduler import ArmDeps, EventBus, ExperimentSnapshot, RaceEngine, ResearchDigest
from test_busy_machines import SPEC, CurveRunner, ScriptedLLM, llm_deps, sigma_tables  # noqa: F401

DIGEST = ResearchDigest(k=1, n_slots=1, elapsed_s=0.0, remaining_s=0.0)


def _score(cfg: dict[str, Any]) -> float:
    return 1.0 + (math.log10(float(cfg["lr"])) + 2.0) ** 2 + 0.001 * int(cfg["width"])


def test_isolation_follows_the_race_mode_and_the_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ISOLATION_ENV, raising=False)
    assert proposer_isolation("benchmark") == "process" and proposer_isolation("dev") == "thread"
    monkeypatch.setenv(ISOLATION_ENV, "thread")
    assert proposer_isolation("benchmark") == "thread"
    monkeypatch.setenv(ISOLATION_ENV, "process")
    assert proposer_isolation("dev") == "process"
    monkeypatch.setenv(ISOLATION_ENV, "bogus")
    assert proposer_isolation("dev") == "thread"


def _drive_sampler(s: BackfillSampler) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The same sequence of operations the scheduler produces: asks, tells of own trials
    (values and FAILs), tells of other runs (added; one outside the space; one without a score)."""
    rng = random.Random(7)
    seen: list[dict[str, Any]] = []
    for k in (4, 6, 3, 8):
        batch = s.ask_batch(k)
        seen += [p.config for p in batch]
        for i, p in enumerate(batch):
            s.tell(p.experiment_id, p.config, None if i == 1 else _score(p.config) + rng.random() * 0.01)
        other = {**SPEC.default_config(), "lr": 10 ** rng.uniform(-4, 0), "width": rng.randint(8, 256)}
        s.tell(f"llm-{k}", other, _score(other))
        s.tell(f"llm-out-{k}", {**other, "width": 10_000}, 1.0)      # outside the space: skipped
        s.tell(f"llm-none-{k}", other, None)                          # no score: skipped
    s.apply_tells()
    return seen, s.counts()


def test_backfill_sampler_gives_the_same_configs_in_a_child_process() -> None:
    a = BackfillSampler(SPEC, 3, n_startup_trials=5)
    b = BackfillSampler(SPEC, 3, n_startup_trials=5, isolation="process")
    try:
        ca, na = _drive_sampler(a)
        cb, nb = _drive_sampler(b)
        assert cb == ca and nb == na
        assert na["asked"] == 21 and na["added"] == 4 and na["add_skipped"] == 8 and na["open_trials"] == 0
        assert b.study is None and b._worker is not None and b._worker.pid not in (None, os.getpid())
        assert b.manifest() == {**a.manifest(), "isolation": "process"}
    finally:
        b.close()
    assert b.counts() == nb                       # counts stay readable after the child exits


def test_optuna_arm_gives_the_same_configs_in_a_child_process() -> None:
    async def drive(r: OptunaResearcher) -> list[dict[str, Any]]:
        seen: list[dict[str, Any]] = []
        r.observe(ExperimentSnapshot(id="baseline", config=SPEC.default_config(), status=ExperimentStatus.succeeded,
                                     score=_score(SPEC.default_config()), is_baseline=True))
        for k in (4, 4, 6):
            props = await r.propose(DIGEST, k)
            seen += [p.config for p in props]
            for i, p in enumerate(props):
                if i == 0:      # stopped by the median rule: PRUNED
                    snap = ExperimentSnapshot(id=p.experiment_id, config=p.config, status=ExperimentStatus.cancelled,
                                              summary={"early_stop": {"outcome": "stopped"}})
                elif i == 1:    # failed: FAIL
                    snap = ExperimentSnapshot(id=p.experiment_id, config=p.config, status=ExperimentStatus.failed)
                else:
                    snap = ExperimentSnapshot(id=p.experiment_id, config=p.config,
                                              status=ExperimentStatus.succeeded, score=_score(p.config))
                r.observe(snap)
        return seen

    import asyncio

    a = OptunaResearcher(SPEC, 11, n_startup_trials=3)
    b = OptunaResearcher(SPEC, 11, n_startup_trials=3, isolation="process")
    try:
        assert asyncio.run(drive(b)) == asyncio.run(drive(a))
    finally:
        b.close()


def test_child_errors_reach_the_caller_and_close_stops_the_child() -> None:
    w = StudyWorker(SPEC, 0, 10, "minimize")
    w.start()
    proc = w._proc
    assert proc is not None and proc.is_alive() and w.pid != os.getpid()
    with pytest.raises(RuntimeError, match="KeyError"):
        w.run([("tell", 999, 1.0)], 0)               # no such open trial
    assert w.run([], 2)["numbers"] == [0, 1]           # still serving after an error
    w.close()
    proc.join(5)
    assert not proc.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        w.run([], 1)


async def test_a_race_with_process_isolation_records_it_and_its_timings(monkeypatch: pytest.MonkeyPatch,
                                                                        sigma_tables: Any) -> None:  # noqa: F811
    monkeypatch.setenv(ISOLATION_ENV, "process")
    cfg = RaceConfig(target="bm-curve", arms=[ArmKind.opus, ArmKind.opus_bm, ArmKind.optuna], budget_s=2.0,
                     gpus_per_arm=3, llm="mock", seed=1)
    deps = [llm_deps(k, "r", CurveRunner(3, run_s=0.2), ScriptedLLM(seed=1, latency_s=0.3, trap_rate=0.0))
            for k in (ArmKind.opus, ArmKind.opus_bm)]
    optuna_r = OptunaResearcher(SPEC, cfg.seed, isolation=proposer_isolation(cfg.mode))
    deps.append(ArmDeps(kind=ArmKind.optuna, researcher=optuna_r, runner=CurveRunner(3, run_s=0.2)))
    eng = RaceEngine(cfg, deps, spec=SPEC, db=Database(":memory:"), bus=EventBus(), tick_s=0.02)
    sampler = eng.engine_states[ArmKind.opus_bm].sampler
    assert sampler is not None and sampler.isolation == "process"
    await eng.run()
    m = eng.fairness_manifest()
    assert m["engine"]["per_arm"]["opus_bm"]["proposer_isolation"] == "process"
    assert m["engine"]["per_arm"]["optuna"]["proposer_isolation"] == "process"
    assert "proposer_isolation" not in m["engine"]["per_arm"]["opus"]
    rep = eng.reports
    assert rep["optuna"]["proposer"]["isolation"] == "process" and rep["optuna"]["proposer"]["asks"] >= 1
    assert rep["optuna"]["proposer"]["ask_p50_s"] is not None
    assert rep["opus_bm"]["backfill"]["isolation"] == "process" and rep["opus_bm"]["backfill"]["batches"] >= 1
    assert rep["opus_bm"]["backfill"]["ask_max_s"] is not None
    for kind in ("opus", "opus_bm", "optuna"):
        lag = rep[kind]["event_loop_lag"]
        assert lag["samples"] > 10 and lag["lag_p50_s"] is not None and lag["lag_max_s"] >= lag["lag_p50_s"] >= 0
    assert eng.arms[ArmKind.optuna].exps
    # the children were closed at the end of the race
    for w in (sampler._worker, optuna_r._worker):
        assert w is not None and w.closed


def test_fingerprint_keeps_thread_races_and_splits_process_races() -> None:
    import importlib.util
    import sys
    from pathlib import Path

    if "analyze_races" not in sys.modules:
        p = Path(__file__).resolve().parents[1] / "scripts" / "analyze_races.py"
        spec = importlib.util.spec_from_file_location("analyze_races", p)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["analyze_races"] = mod
        spec.loader.exec_module(mod)
    ar = sys.modules["analyze_races"]

    def rows(iso: str | None) -> dict[str, Any]:
        per_arm: dict[str, Any] = {"optuna": {"features": []}}
        if iso is not None:
            per_arm["optuna"]["proposer_isolation"] = iso
        return {"races": [{"id": "r", "fairness": {"per_arm": {"optuna": {}}, "engine": {"per_arm": per_arm}}}],
                "arms": [{"id": "o", "kind": "optuna", "model": None, "llm_mode": "none"}],
                "llm_calls": [], "experiments": []}

    old, thread, proc = (ar.fairness_fingerprint(rows(x)) for x in (None, "thread", "process"))
    assert ar.fingerprint_id(old) == ar.fingerprint_id(thread) != ar.fingerprint_id(proc)
    assert proc["arms"]["optuna"]["proposer_isolation"] == "process"


def test_the_contention_micro_benchmark_runs(tmp_path: Any, capsys: pytest.CaptureFixture[str]) -> None:
    """Smoke run of scripts/bench_proposer_contention.py (tiny study; the real run is manual)."""
    import importlib.util
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "scripts" / "bench_proposer_contention.py"
    spec = importlib.util.spec_from_file_location("bench_proposer_contention", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "rows.json"
    assert mod.main(["--target", "bm-curve", "--trials", "10", "--repeats", "1", "--k-sampler", "5",
                     "--json", str(out)]) == 0
    rows = json.loads(out.read_text(encoding="utf-8"))["rows"]
    assert [(r["isolation"], r["contended"]) for r in rows] == [("thread", False), ("thread", True),
                                                                ("process", False), ("process", True)]
    assert all(r["optuna_asks"] == 1 and r["optuna_ask_p50_s"] > 0 for r in rows)
    assert all(r["sampler_asks"] >= 1 for r in rows if r["contended"])
    assert "contention slows the Optuna ask" in capsys.readouterr().out
