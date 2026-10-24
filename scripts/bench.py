"""night/bench: run a fair race between the arms of a preset and print the headline metrics.

    uv run --extra ml python scripts/bench.py list
    uv run --extra ml python scripts/bench.py check lm-30m
    uv run --extra ml python scripts/bench.py run dev-local --dry-run           # mock LLM + laptop CPU, $0
    uv run --extra ml python scripts/bench.py run cifar-30m --seed 0 --yes      # PAID: Modal GPUs + Mubit

What ``run`` does (docs/benchmark.md section 4, fairness rules):
- every arm of the preset starts at the same t0 with the same slots, hardware, wall-clock,
  $ cap, seed, per-run budget, prompts and retry/fallback policy (RaceEngine checks this and
  writes the fairness manifest into the race row); only the LLM client and its recording
  mode differ;
- real runs use the DEPLOYED Modal app (``ModalRunner(deployed=True)``, looked up by name,
  never deployed from here) and ``MubitClient`` (every call through the Mubit gateway);
- seeds run one after another (benchmark.md "Seed and time plan");
- every race goes into the benchmark SQLite DB (``data/bench/bench.db``; dry runs go to
  ``data/bench/dryrun.db``) and a JSONL export next to it; the summary printed at the end is
  recomputed from that export with ``backend.benchmark.metrics`` (via ``analysis``).

Recording mode per arm (``RaceConfig.record_mode``, x-middleware-record): benchmark presets
send ``off`` on the Opus arm (drops the task fingerprint, so Mubit cannot route those calls
to an adapter) and ``metadata`` on the Mubit arm (keeps the fingerprint so routing can
happen, stores no content). PENDING Mubit's confirmation (docs/mubit_office_hours.md item 2).

Goals: the threshold for time-to-target / $-to-target comes from
``data/calibration/target_values.json`` (``{"lm": 3.28, ...}``, frozen from measured
starting-code runs, never moved). The key is the target name, or ``"<target>@<N>s"`` when
the preset's per-run budget differs from the target's default (e.g. ``"rl@15s"``: a goal
calibrated at 15 s runs never stands in for the 30 s one). ``run`` refuses a target without
one unless ``--target-value`` is given; such a race is marked provisional
(``RaceConfig.provisional``, race name, manifest). A ``--dry-run`` without a goal uses the
target spec's placeholder, also provisional.

Pre-warm (``RaceConfig.prewarm``, Modal only): before the race clock starts, one container
per slot of every arm (plus ``prewarm_spare``) is pinned (``min_containers``) and warmed
(``ModalRunner.warm_up``: ML libs imported, backend initialised, >= N distinct containers
answered). t0 is taken only after that, so pre-warm time is never busy/idle/race $. The
pin is removed in a ``finally`` (also on Ctrl-C or an error). If the pool cannot be warmed
the race is refused (exit 2).

Engine features (D36, D40, D43, D44; ``RaceConfig.features_for``): the Opus arm is the standard
agent; ``opus_bm`` ("Busy Machines with Opus") and ``mubit`` ("Mubit + Busy Machines") add the
judge policy only (LLM for leakage + calibrated seed-spread rule for noise; it decides the
shipped model, never what the planner sees); median early stopping is off for every arm and
target (D43), Optuna included. The flags are printed in the plan, gated by a pre-flight check
(the arms that should match do) and stored in the fairness manifest.

Ablation presets (``mode="ablation"``, e.g. ``rl-planner-ablation``: Opus vs ``fast`` =
claude-haiku-5-5 in the same loop, a smaller faster model, NOT a Mubit adapter) get the same
strict checks as a benchmark but are never headline races: they go to ``data/bench/ablation.db``
(exports in ``data/bench/ablation/``), are stored as dev (``is_dev=1``), and
``scripts/headline.py`` refuses them.

Paid runs need ``--yes``. ``--dry-run`` swaps in the mock LLM and the local CPU runner (the
in-process surrogate runner stays for surrogate presets: it is $0 either way) (dev race, $0,
mock numbers: never a benchmark result) and, unless overridden, caps the race at 2 slots x
2 minutes.

Exit codes: 0 ok, 1 a race ended in error, 2 refused by a pre-flight check, 3 needs --yes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.agent.prompts import is_frozen  # noqa: E402
from backend.benchmark.configs import (  # noqa: E402
    ABLATION_ONLY_ARMS,
    BENCH_SEEDS,
    FAST_ARM_LABEL,
    PRESETS,
    RaceConfig,
    estimate_gpu_cost_usd,
)
from backend.core.types import ArmKind, StepType  # noqa: E402
from workers.runners.base import DEFAULT_PRICE_PER_SECOND  # noqa: E402
from workers.targets.base import get_target  # noqa: E402

GOALS_PATH = ROOT / "data" / "calibration" / "target_values.json"
BENCH_DIR = ROOT / "data" / "bench"
BENCH_DB = BENCH_DIR / "bench.db"
DRYRUN_DB = BENCH_DIR / "dryrun.db"
DRYRUN_DIR = BENCH_DIR / "dryrun"  # mock-race JSONL exports (data/bench/*.jsonl stays real races only)
ABLATION_DB = BENCH_DIR / "ablation.db"   # real (paid) ablation races: never next to the headline races
ABLATION_DIR = BENCH_DIR / "ablation"

# Mirrors modal_runner.CPU_CORES (not imported: that would import modal).
CPU_CORES_PER_SLOT = float(os.environ.get("BUSY_MODAL_CPU_CORES", "1"))
# Mirrors modal_runner.STARTER_MAX_GPUS / STARTER_MAX_CONTAINERS (Modal Starter plan).
MODAL_MAX_GPUS = 10
MODAL_MAX_CONTAINERS = 100
DRY_RUN_MAX_SLOTS = 2
DRY_RUN_MAX_MINUTES = 2.0
# Checks that only inform (a provisional goal; an ablation label); every other check gates ``run``.
INFO_ONLY_CHECKS = frozenset({"target_value frozen", "ablation (never a headline race)"})


class Refused(RuntimeError):
    """A pre-flight check failed: the race must not start."""


def price_per_s(hw: str) -> float:
    p = DEFAULT_PRICE_PER_SECOND[hw]
    return p * CPU_CORES_PER_SLOT if hw == "CPU" else p


# --------------------------------------------------------------------------- goals


def load_goals(path: Path | None = None) -> dict[str, float]:
    """``{target: value}`` from the calibration file; {} if it does not exist. A value may
    also be an object with ``target_value`` or ``value`` (calibration scripts may add notes)."""
    p = GOALS_PATH if path is None else Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: expected an object {{target: value}}")
    out: dict[str, float] = {}
    for target, v in raw.items():
        if isinstance(v, Mapping):
            v = v.get("target_value", v.get("value"))
        if v is None:
            continue
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"{p}: goal for {target!r} is not finite")
        out[str(target)] = f
    return out


class Goal(NamedTuple):
    value: float
    provisional: bool
    source: str


def goal_key(cfg: RaceConfig) -> str:
    """Goal-file key: the target, or ``target@Ns`` when the per-run budget is not the
    target's default (a goal is only valid at the run budget it was calibrated at)."""
    spec = get_target(cfg.target).spec
    if cfg.run_budget_s is not None and int(cfg.run_budget_s) != int(spec.budget_s):
        return f"{cfg.target}@{int(cfg.run_budget_s)}s"
    return cfg.target


def resolve_goal(target: str, goals: Mapping[str, float], cli_value: float | None, *, dry_run: bool,
                 goals_path: Path | None = None, spec_target: str | None = None) -> Goal:
    """Frozen goal from the goal file; else ``--target-value`` (provisional); else, for a dry
    run only, the spec placeholder (provisional). A real run without either is refused, and
    ``--target-value`` may never move a frozen goal."""
    where = goals_path or GOALS_PATH
    frozen = goals.get(target)
    if frozen is not None:
        if cli_value is not None and float(cli_value) != frozen:
            raise Refused(f"the goal for {target} is frozen at {frozen:g} ({where}); "
                          f"--target-value {cli_value:g} would move it")
        return Goal(frozen, False, "frozen")
    if cli_value is not None:
        return Goal(float(cli_value), True, "--target-value")
    if dry_run:
        return Goal(float(get_target(spec_target or target).spec.target_value), True, "spec placeholder (dry run)")
    raise Refused(f"no frozen goal for {target} in {where}; pass --target-value to run a provisional race")


# --------------------------------------------------------------------------- checks


def checks(cfg: RaceConfig, goals: Mapping[str, float] | None = None, *,
           goal: Goal | None = None) -> list[tuple[str, bool, str]]:
    """Pre-flight checks for one race. ``target_value frozen`` is informational (a race
    without a frozen goal runs only as provisional); every other check gates ``run``."""
    goals = load_goals() if goals is None else goals
    spec = get_target(cfg.target).spec
    local = cfg.runner == "local"
    est = estimate_gpu_cost_usd(cfg, 0.0 if local else price_per_s(spec.hardware))
    key = goal_key(cfg)
    frozen = goals.get(key)
    out = [(f"prompt {s.value} frozen", is_frozen(s), "ok" if is_frozen(s) else "DRAFT") for s in StepType]
    out.append(("target_value frozen", frozen is not None,
                f"{key}={frozen:g} from {GOALS_PATH.relative_to(ROOT)}" if frozen is not None
                else f"none for {key!r} (spec placeholder {spec.target_value:g})"))
    if goal is not None:
        out.append(("goal set", True, f"{goal.value:g} ({goal.source}{', PROVISIONAL' if goal.provisional else ''})"))
    out.append(engine_check(cfg))
    guard = cfg.max_estimated_gpu_usd
    out.append(("GPU $ estimate within guard", guard is None or est <= guard,
                f"${est:.2f} per seed (guard {guard})"))
    out.append(("no H100", spec.hardware != "H100" and not cfg.allow_h100, spec.hardware))
    n = len(cfg.arms) * cfg.gpus_per_arm
    if local:
        out.append(("fits Modal Starter caps", True, "local runner"))
    elif cfg.runner == "surrogate":
        out.append(("fits Modal Starter caps", True, "surrogate runner: in-process, no containers, GPU $ virtual"))
    elif spec.hardware == "CPU":
        m = n + (cfg.prewarm_spare if cfg.prewarm else 0)
        out.append(("fits Modal Starter caps", m <= MODAL_MAX_CONTAINERS,
                    f"{m} CPU containers at once ({n} slots + {m - n} spare; cap {MODAL_MAX_CONTAINERS})"))
    else:
        m = n + (cfg.prewarm_spare if cfg.prewarm else 0)
        out.append(("fits Modal Starter caps", m <= MODAL_MAX_GPUS,
                    f"{m} {spec.hardware} at once ({n} slots + {m - n} spare; cap {MODAL_MAX_GPUS}, "
                    f"shared with anything else running)"))
    if cfg.strict:
        out.append(("record modes", all(m != "full" for m in cfg.record_modes().values()),
                    ", ".join(f"{a}={m}" for a, m in cfg.record_modes().items())))
    if cfg.is_ablation or any(a in ABLATION_ONLY_ARMS for a in cfg.arms):
        out.append(("ablation (never a headline race)", True, ablation_note(cfg)))
    return out


def ablation_note(cfg: RaceConfig) -> str:
    """How an ablation race is labelled everywhere (plan, checks)."""
    arms = [f"{a.value} = {cfg.model_for(a)} ({FAST_ARM_LABEL})" for a in cfg.arms if a in ABLATION_ONLY_ARMS]
    return (f"mode={cfg.mode}; " + ("; ".join(arms) if arms else "no ablation-only arm")
            + f"; stored in {ABLATION_DB.relative_to(ROOT)}, refused by scripts/headline.py")


def engine_check(cfg: RaceConfig) -> tuple[str, bool, str]:
    """Engine features per arm, and the same refusal rules RaceEngine applies at start."""
    from backend.scheduler import AsymmetricRaceError, check_engine_features, race_spec, resolve_engine_state

    spec = race_spec(cfg)
    states = {k: resolve_engine_state(cfg, k, spec) for k in cfg.arms}
    flags = "; ".join(f"{k.value}={'+'.join(st.features) or 'none'}" for k, st in states.items())
    try:
        check_engine_features(cfg, states, spec.name)
    except AsymmetricRaceError as e:
        return ("engine features", False, f"{flags}: {e}")
    notes = [n for st in states.values() for n in st.notes]
    return ("engine features", True, flags + (f" ({'; '.join(sorted(set(notes)))})" if notes else ""))


def gating_ok(res: list[tuple[str, bool, str]]) -> bool:
    return all(ok for name, ok, _ in res if name not in INFO_ONLY_CHECKS)


# --------------------------------------------------------------------------- race configs


def parse_arms(value: str | None) -> list[ArmKind] | None:
    """``--arms opus,optuna`` -> [ArmKind.opus, ArmKind.optuna] (None: the preset's arms)."""
    if not value:
        return None
    out = [ArmKind(a.strip()) for a in value.split(",") if a.strip()]
    if not out:
        raise ValueError("--arms is empty")
    return out


def build_races(preset: str, *, seeds: list[int] | None = None, minutes: float | None = None,
                slots: int | None = None, usd_cap: float | None = None, max_gpu_usd: float | None = None,
                dry_run: bool = False, goal: Goal | None = None,
                arms: list[ArmKind] | None = None) -> list[RaceConfig]:
    base = PRESETS[preset]
    upd: dict[str, Any] = {}
    if arms is not None:
        bad = [a.value for a in arms if a not in base.arms]
        if bad:
            raise ValueError(f"--arms {bad} not in preset {preset} ({[a.value for a in base.arms]})")
        upd["arms"] = list(arms)
        upd["record_mode"] = {k: v for k, v in base.record_mode.items() if k in arms}
        upd["engine_features"] = {k: v for k, v in base.engine_features.items() if k in arms}
    if dry_run:
        # The surrogate runner is in-process and $0, so a surrogate preset keeps it (its pacing
        # is what the median rule reads); every other preset swaps in the local CPU runner.
        upd.update(mode="dev", llm="mock", runner="surrogate" if base.runner == "surrogate" else "local",
                   prewarm=False)
        if minutes is None:
            minutes = min(base.budget_s / 60.0, DRY_RUN_MAX_MINUTES)
        if slots is None:
            slots = min(base.gpus_per_arm, DRY_RUN_MAX_SLOTS)
    if minutes is not None:
        if minutes <= 0:
            raise ValueError("--minutes must be > 0")
        upd["budget_s"] = float(minutes) * 60.0
    if slots is not None:
        upd["gpus_per_arm"] = int(slots)
    if usd_cap is not None:
        upd["cost_cap_usd"] = float(usd_cap)
    if max_gpu_usd is not None:
        upd["max_estimated_gpu_usd"] = float(max_gpu_usd)
    if goal is not None:
        upd.update(target_value=goal.value, provisional=goal.provisional)
    tags = (" [dry run]" if dry_run else "") + (" [provisional goal]" if goal is not None and goal.provisional else "")
    if dry_run and base.launch_cutoff_s is not None and "budget_s" in upd and base.launch_cutoff_s >= upd["budget_s"]:
        upd["launch_cutoff_s"] = None  # a 2-minute dry run of a short race: keep it valid
    out = []
    for s in (list(seeds) if seeds else list(BENCH_SEEDS)):
        d = {**base.model_dump(), **upd, "seed": int(s), "name": f"{base.name} seed {s}{tags}"}
        out.append(RaceConfig.model_validate(d))  # re-validated: overrides go through the same checks
    return out


# --------------------------------------------------------------------------- live pre-flight (paid runs)


def live_checks(cfg: RaceConfig) -> list[tuple[str, bool, str]]:
    """Free checks that touch the outside world: the Mubit key exists and the deployed Modal
    app has the function this target runs on. Nothing is deployed or launched."""
    out: list[tuple[str, bool, str]] = []
    if cfg.llm == "mubit":
        try:
            from backend.agent.mubit_client import load_mubit_api_key

            load_mubit_api_key()
            out.append(("MUBIT_API_KEY available", True, "env or .env"))
        except Exception as e:  # noqa: BLE001
            out.append(("MUBIT_API_KEY available", False, str(e)))
    if cfg.runner == "modal":
        spec = get_target(cfg.target).spec
        try:
            from workers.runners.modal_runner import APP_NAME, ModalRunner

            runner = ModalRunner({spec.hardware: 1}, deployed=True)
            fn = runner._function(spec.hardware)
            fn.hydrate()  # name lookup only (no container starts)
            out.append(("Modal app deployed", True, f"{APP_NAME}: {spec.hardware} function found"))
        except Exception as e:  # noqa: BLE001
            out.append(("Modal app deployed", False, f"{type(e).__name__}: {str(e)[:200]}"))
    return out


# --------------------------------------------------------------------------- pre-warm (Modal)


def _arm_hardware(cfg: RaceConfig, deps: list[Any]) -> str:
    spec = get_target(cfg.target).spec
    runner = deps[0].runner
    return str(getattr(runner, "hardware_for", {}).get(cfg.target, spec.hardware))


async def prewarm_pool(cfg: RaceConfig, deps: list[Any], *, log: Any = print) -> dict[str, Any] | None:
    """Pin and warm one container per slot of every arm (+ spare) before t0. All arms share
    the deployed function's pool, so warmth is symmetric. Returns the record stored in the
    fairness manifest, or None when the race does not pre-warm."""
    if not cfg.prewarm or cfg.runner != "modal":
        return None
    runner = deps[0].runner
    if not hasattr(runner, "warm_up"):
        return None
    hw = _arm_hardware(cfg, deps)
    n = cfg.gpus_per_arm * len(cfg.arms) + cfg.prewarm_spare
    t = time.time()
    await runner.prewarm(hw, n)
    res = await runner.warm_up(hw, n, target=cfg.target, log=log)
    res.update(pinned=n, slots_total=cfg.gpus_per_arm * len(cfg.arms), spare=cfg.prewarm_spare,
               started_at=t, ended_at=time.time(), note="before t0: not in busy/idle/race $")
    return res


async def cooldown_pool(cfg: RaceConfig, deps: list[Any]) -> None:
    if not cfg.prewarm or cfg.runner != "modal" or not deps:
        return
    runner = deps[0].runner
    if hasattr(runner, "cooldown"):
        await runner.cooldown(_arm_hardware(cfg, deps), reset_scaledown=True)
    for d in deps:
        drain = getattr(d.runner, "drain_background", None)
        if drain is not None:
            await drain()


# --------------------------------------------------------------------------- running a race


def _fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


async def run_race(cfg: RaceConfig, db: Any, out_dir: Path, *, verbose: bool = False,
                   progress_every_s: float = 30.0, tick_s: float = 0.05,
                   ) -> tuple[str, Path, str, dict[str, Any], dict[str, Any]]:
    """One race, all arms at the same t0. Returns (race_id, export path, status, fairness
    manifest, the engine's own ArmSummary per arm kind)."""
    from backend.app import default_arm_deps
    from backend.benchmark.export import export_race
    from backend.core.types import new_id
    from backend.scheduler import EventBus, RaceEngine, race_spec

    race_id = new_id()
    spec = race_spec(cfg)  # the race's goal applied; researchers and metrics see the same spec
    deps = [default_arm_deps(race_id, cfg, spec, kind) for kind in cfg.arms]
    bus = EventBus()
    if verbose:
        t_start = time.time()

        def show(ev: Any) -> None:
            if ev.type in ("experiment_finished", "new_best"):
                arm = getattr(getattr(ev, "arm", None), "value", "")
                status = getattr(getattr(ev, "status", None), "value", "")
                extra = f"{status} score={_fmt(getattr(ev, 'score', None), 4)}"
                print(f"  t={time.time() - t_start:7.1f}s {ev.type:<20} {arm:<7} {extra}", flush=True)

        bus.add_listener(show)
    engine = RaceEngine(cfg, deps, spec=spec, race_id=race_id, db=db, bus=bus, tick_s=tick_s)

    async def progress() -> None:
        while True:
            await asyncio.sleep(progress_every_s)
            if engine.status != "running":
                continue
            parts = []
            for s, _ in engine.live_summaries():
                parts.append(f"{s.arm.value}: {s.experiments_finished} done, idle {s.gpu_idle_pct:.0f}%, "
                             f"${s.gpu_cost_usd + s.llm_cost_usd:.2f}, best {_fmt(s.best_score, 4)}")
            elapsed = time.time() - (engine.t0 or time.time())
            print(f"  [{elapsed:6.0f}s / {cfg.budget_s:.0f}s] " + " | ".join(parts), flush=True)

    prog: asyncio.Task[None] | None = None
    try:
        warm = await prewarm_pool(cfg, deps)
        if warm is not None:
            engine.prewarm_info = warm
            print(f"pre-warm: {warm['warm']}/{warm['wanted']} distinct warm containers in {warm['seconds']:.1f}s "
                  f"({len(warm['rounds'])} round(s)); race clock starts now", flush=True)
            if not warm["ok"]:
                raise Refused(f"pre-warm reached {warm['warm']} of {warm['wanted']} containers")
        prog = asyncio.create_task(progress())
        await engine.run()
    finally:
        if prog is not None:
            prog.cancel()
        try:
            await cooldown_pool(cfg, deps)
        except Exception as e:  # noqa: BLE001 - must not hide the race result, but say so loudly
            print(f"WARNING: cooldown failed ({e!r}); unpin with ModalRunner.cooldown or redeploy", file=sys.stderr)
        for d in deps:
            for attr in ("llm", "fallback_llm"):
                client = getattr(d.researcher, attr, None)
                close = getattr(client, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except Exception:  # noqa: BLE001 - closing must not hide the race result
                        pass
    manifest = engine.fairness_manifest()
    if cfg.mode == "benchmark":
        db.assert_benchmark_clean(race_id)
    out = export_race(db, race_id, out_dir / f"race-{cfg.target}-seed{cfg.seed}-{race_id[:8]}.jsonl")
    return race_id, out, engine.status, manifest, {s.arm.value: s for s in engine.summaries}


def run_bounded(coro: Any, *, teardown_s: float = 10.0) -> Any:
    """``asyncio.run`` with a bounded teardown: tasks still pending at the end (e.g. a Modal
    call that ignores cancellation, abandoned by ``ModalRunner``) are cancelled and given
    ``teardown_s``, then left behind, so the CLI can never hang after a race."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(asyncio.wait(pending, timeout=teardown_s))
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def summary_mismatches(engine: Mapping[str, Any], recomputed: Mapping[str, Any]) -> list[str]:
    """Headline fields where the engine and the export recomputation disagree (should be none)."""
    bad = []
    for arm, s in engine.items():
        r = recomputed.get(arm)
        if r is None:
            bad.append(f"{arm}: missing from export")
            continue
        for f in ("best_score", "time_to_target_s", "cost_to_target_usd", "gpu_cost_usd", "llm_cost_usd",
                  "gpu_idle_pct", "experiments_finished", "llm_calls"):
            a, b = getattr(s, f), getattr(r, f)
            if (a is None) != (b is None) or (a is not None and not math.isclose(float(a), float(b),
                                                                                 rel_tol=1e-6, abs_tol=1e-6)):
                bad.append(f"{arm}.{f}: engine {a} vs export {b}")
    return bad


def print_summary(path: Path, *, allow_dev: bool) -> dict[str, Any]:
    """Recompute every headline number from the JSONL export (backend.benchmark.metrics)."""
    from backend.benchmark.analysis import summaries_from_rows
    from backend.benchmark.export import read_export

    rows = read_export(path)
    sums = summaries_from_rows(rows, allow_dev=allow_dev)
    cols = [("arm", 7), ("record", 9), ("best", 11), ("t->goal s", 9), ("$->goal", 8), ("idle %", 6),
            ("done", 5), ("failed", 6), ("GPU $", 7), ("LLM $", 7), ("calls", 5), ("p50 ms", 7), ("bad %", 5),
            ("fake ok", 7), ("stopped", 7), ("freed s", 7)]   # slot time freed by early stops (not $ saved)
    hdr = " ".join(f"{n:<{w}}" if i == 0 else f"{n:>{w}}" for i, (n, w) in enumerate(cols))
    print(hdr)
    print("-" * len(hdr))
    arms = {a["kind"]: a for a in rows.get("arms", [])}
    for kind, (s, x) in sums.items():
        rec = (arms.get(kind, {}).get("report") or {}).get("record_mode") or "-"
        vals = [kind, rec, _fmt(s.best_score, 4), _fmt(s.time_to_target_s, 0), _fmt(s.cost_to_target_usd),
                _fmt(s.gpu_idle_pct, 1), s.experiments_finished, s.experiments_failed, _fmt(s.gpu_cost_usd),
                _fmt(s.llm_cost_usd), s.llm_calls, _fmt(s.llm_p50_latency_ms, 0),
                _fmt(x["malformed_reply_rate"] * 100, 1), x["fake_best_accepted"], x.get("early_stopped", 0),
                _fmt(x.get("early_stop_slot_s_freed", x.get("early_stop_saved_s")), 0)]
        print(" ".join(f"{v!s:<{w}}" if i == 0 else f"{v!s:>{w}}" for i, (v, (_, w)) in enumerate(zip(vals, cols))))
    return {k: v[0] for k, v in sums.items()}


def cmd_run(args: argparse.Namespace) -> int:
    goals = load_goals(args.goals)
    preset = PRESETS[args.preset]
    try:
        goal = resolve_goal(goal_key(preset), goals, args.target_value, dry_run=args.dry_run, goals_path=args.goals,
                            spec_target=preset.target)
        cfgs = build_races(args.preset, seeds=args.seed, minutes=args.minutes, slots=args.slots,
                           usd_cap=args.usd_cap, max_gpu_usd=args.max_gpu_usd, dry_run=args.dry_run, goal=goal,
                           arms=parse_arms(args.arms))
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"refused: invalid override: {e}", file=sys.stderr)
        return 2
    cfg = cfgs[0]
    res = checks(cfg, goals, goal=goal)
    for name, ok, note in res:
        print(f"[{'ok' if ok else 'NO'}] {name}: {note}")
    if not gating_ok(res):
        print("refused: pre-flight checks failed", file=sys.stderr)
        return 2

    spec = get_target(cfg.target).spec
    est = estimate_gpu_cost_usd(cfg, 0.0 if cfg.runner == "local" else price_per_s(spec.hardware))
    ceiling = (cfg.cost_cap_usd or 0.0) * len(cfg.arms)  # the cap is per arm, GPU $ + LLM $
    print(f"\nplan: {len(cfgs)} race(s) one after another, seeds {[c.seed for c in cfgs]}; "
          f"{len(cfg.arms)} arms ({', '.join(a.value for a in cfg.arms)}) x {cfg.gpus_per_arm} {spec.hardware} "
          f"x {cfg.budget_s / 60:.1f} min; llm={cfg.llm} runner={cfg.runner} mode={cfg.mode}")
    print(f"      run budget {cfg.run_budget_s or spec.budget_s}s; K cap {cfg.max_proposals_per_call}; "
          f"deadline rule {cfg.launch_cutoff_s}s; pre-warm {cfg.prewarm} (+{cfg.prewarm_spare} spare)")
    print(f"      record modes {cfg.record_modes()} (pending Mubit confirmation, office hours item 2)")
    stops = [a for a, f in cfg.engine_features_map().items() if "median_stop" in f]
    print(f"      engine features {cfg.engine_features_map()}; "
          + (f"median rule ({', '.join(stops)}) at {[f'{f:.0%}' for f in cfg.median_stop_fractions]} of the "
             f"run budget, >= {cfg.median_stop_min_completed} completed runs" if stops
             else "early stopping off for every arm (D43)"))
    print(f"      models {{{', '.join(f'{a.value}: {cfg.model_for(a)}' for a in cfg.llm_arms)}}}")
    if cfg.is_ablation or any(a in ABLATION_ONLY_ARMS for a in cfg.arms):
        print(f"      ABLATION, NEVER A HEADLINE RACE: {ablation_note(cfg)}")
    print(f"      goal {spec.metric} {'<=' if spec.direction == 'min' else '>='} {goal.value:g} "
          f"({goal.source}{', PROVISIONAL' if goal.provisional else ''})")
    if cfg.runner != "local":
        print(f"      GPU $ <= {est:.2f} per race, <= {est * len(cfgs):.2f} total; per-arm GPU+LLM cap "
              f"${cfg.cost_cap_usd}, so hard ceiling ${ceiling * len(cfgs):.2f} over all races")
        if cfg.runner == "surrogate":
            print("      surrogate runner: GPU $ above is VIRTUAL (predicted training time priced at the "
                  f"{spec.hardware} rate); real spend is the LLM calls only")
    # Any mock-LLM race (``--dry-run``, or a mock preset such as dev-local run without it) is
    # a dev race: it goes to dryrun.db, never bench.db, and is labelled MOCK NUMBERS.
    mock = cfg.llm == "mock"
    if mock and cfg.runner in ("local", "surrogate"):
        print(f"      {'DRY RUN' if args.dry_run else 'MOCK PRESET'}: mock LLM + {cfg.runner} runner (in-process), $0. "
              "MOCK NUMBERS: never a benchmark result.")
    else:
        live = live_checks(cfg)
        for name, ok, note in live:
            print(f"[{'ok' if ok else 'NO'}] {name}: {note}")
        if not all(ok for _, ok, _ in live):
            print("refused: live pre-flight checks failed", file=sys.stderr)
            return 2
        if not args.yes:
            print("\nthis is a PAID run (Modal GPUs + LLM calls through Mubit). Re-run with --yes to start.",
                  file=sys.stderr)
            return 3
    if cfg.target == "cifar" and cfg.runner == "local":
        os.environ.setdefault("BUSY_CIFAR_SYNTHETIC", "1")  # laptop CPU: tiny synthetic data

    from backend.benchmark.analysis import aggregate
    from backend.db import BenchmarkIntegrityError, Database
    from backend.scheduler import AsymmetricRaceError

    default_db, default_dir = (DRYRUN_DB, DRYRUN_DIR) if mock else (
        (ABLATION_DB, ABLATION_DIR) if cfg.is_ablation else (BENCH_DB, None))
    db_path = Path(args.db) if args.db else default_db
    out_dir = Path(args.out_dir) if args.out_dir else (default_dir or db_path.parent)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    db = Database(db_path)
    per_seed: list[dict[str, Any]] = []
    rc = 0
    try:
        for c in cfgs:
            print(f"\n=== {c.name}: starting all arms together", flush=True)
            race_id, path, status, manifest, engine_sums = run_bounded(
                run_race(c, db, out_dir, verbose=args.verbose, progress_every_s=args.progress_s))
            print(f"race {race_id} {status}; fairness identical={manifest['identical']} "
                  f"llm_policy_identical={manifest['llm_policy_identical']}; "
                  f"record modes {{{', '.join(f'{k}: {v['record_mode']}' for k, v in manifest['differs_only_in'].items())}}}")
            print(f"db: {db_path}  export: {path}")
            eng = manifest.get("engine") or {}
            print(f"engine features {eng.get('features')}; checks: {'; '.join(eng.get('checks') or []) or '-'}")
            if mock:
                print("MOCK NUMBERS (mock LLM, laptop CPU, $0) - not a benchmark result")
            elif c.is_ablation:
                print(f"ABLATION (never a headline result): {ablation_note(c)}")
            elif c.provisional:
                print("PROVISIONAL goal (not frozen): time/$ to goal are not headline numbers")
            recomputed = print_summary(path, allow_dev=c.is_dev)
            per_seed.append(recomputed)
            for m in summary_mismatches(engine_sums, recomputed):
                print(f"WARNING engine/export mismatch: {m}", file=sys.stderr)
                rc = 1
            if status not in ("finished",):
                rc = 1
    except KeyboardInterrupt:
        print("\ninterrupted: the running race was cancelled (slots closed, runs cancelled)", file=sys.stderr)
        return 130
    except (AsymmetricRaceError, BenchmarkIntegrityError, Refused) as e:
        print(f"refused: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    finally:
        db.close()
    if len(per_seed) > 1:
        print(f"\n=== {len(per_seed)} seeds, mean +/- std (n = seeds with a value, missing = never reached)")
        for arm, fields in aggregate(per_seed).items():
            shown = []
            for f in ("best_score", "time_to_target_s", "cost_to_target_usd", "gpu_idle_pct", "total_cost_usd"):
                st = fields[f]
                v = "n/a" if st["mean"] is None else f"{st['mean']:.4g}+/-{st['std']:.2g}"
                shown.append(f"{f}={v} (n={st['n']}, missing={st['n_missing']})")
            print(f"{arm:<7} " + "  ".join(shown))
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    c = sub.add_parser("check")
    c.add_argument("preset", choices=sorted(PRESETS))
    c.add_argument("--goals", type=Path, default=None, help=f"goal file (default {GOALS_PATH.relative_to(ROOT)})")
    r = sub.add_parser("run")
    r.add_argument("preset", choices=sorted(PRESETS))
    r.add_argument("--seed", type=int, action="append", default=None,
                   help=f"seed to run (repeatable); default {list(BENCH_SEEDS)}, one after another")
    r.add_argument("--minutes", type=float, default=None, help="wall-clock per race (all arms), overrides the preset")
    r.add_argument("--slots", type=int, default=None, help="slots per arm, overrides the preset")
    r.add_argument("--usd-cap", type=float, default=None, help="per-arm GPU $ + LLM $ cap, overrides the preset")
    r.add_argument("--max-gpu-usd", type=float, default=None,
                   help="raise/lower the preset's GPU $ estimate guard (per race)")
    r.add_argument("--arms", default=None,
                   help="comma-separated subset of the preset's arms, e.g. opus,optuna (default: all)")
    r.add_argument("--target-value", type=float, default=None,
                   help="goal when none is frozen in the goal file; the race is marked provisional")
    r.add_argument("--goals", type=Path, default=None, help=f"goal file (default {GOALS_PATH.relative_to(ROOT)})")
    r.add_argument("--dry-run", action="store_true", help="mock LLM + local CPU runner ($0, dev race)")
    r.add_argument("--yes", action="store_true", help="confirm a paid run (Modal GPUs + Mubit LLM calls)")
    r.add_argument("--db", default=None, help=f"SQLite path (default {BENCH_DB.relative_to(ROOT)}; dry run "
                                              f"{DRYRUN_DB.relative_to(ROOT)})")
    r.add_argument("--out-dir", default=None, help="where JSONL exports go (default: next to the DB)")
    r.add_argument("--progress-s", type=float, default=30.0, help="seconds between progress lines")
    r.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        goals = load_goals()
        for name, cfg in PRESETS.items():
            hw = get_target(cfg.target).spec.hardware
            est = estimate_gpu_cost_usd(cfg, 0.0 if cfg.runner == "local" else price_per_s(hw))
            arms = ",".join(a.value for a in cfg.arms)
            goal = goals.get(goal_key(cfg))
            print(f"{name:<12} {cfg.target:<6} {hw:<4} {cfg.gpus_per_arm:>3} slots/arm  {cfg.budget_s / 60:>5.1f} min  "
                  f"arms={arms:<20} mode={cfg.mode:<9} est GPU $ <= {est:.2f} per seed  "
                  f"goal={'frozen ' + format(goal, 'g') if goal is not None else 'none'}  "
                  f"record={cfg.record_modes()}")
        return 0
    if args.cmd == "check":
        cfg = PRESETS[args.preset]
        res = checks(cfg, load_goals(args.goals))
        for name, ok, note in res:
            print(f"[{'ok' if ok else 'NO'}] {name}: {note}")
        print(f"{len(BENCH_SEEDS)} races (seeds {list(BENCH_SEEDS)})")
        return 0 if all(ok for _, ok, _ in res) else 1
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
