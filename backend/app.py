"""FastAPI app: REST + WebSocket for races (architecture.md "API").

Run:  uv run uvicorn backend.app:app --reload        (DB at $BUSY_MACHINES_DB, default data/busy_machines.db)
  or: uv run uvicorn --factory backend.app:create_app

Dev mode (``llm="mock"``, ``runner="local"`` in the race body) wires the mock LLM and the
local runner, imported lazily by module path (``backend.agent.mock_llm.MockLLM``,
``workers.runners.local.LocalRunner``) so this file works before they exist. Real mode
(``llm="mubit"``) uses ``backend.agent.mubit_client.MubitClient``; ``runner="modal"`` uses
``workers.runners.modal_runner.ModalRunner``. Tests inject their own ``arm_deps_factory``.

Live Modal metrics (DEMO RACES ONLY, docs/dashboard_modal.md): for a demo race on per-arm Modal
apps (``RaceConfig.modal_apps``, runner ``modal``) ``ModalStatsHub`` polls each arm's function
stats every ~2 s while the race is live (``workers.runners.modal_stats.ModalStatsPoller``), joins
them with the in-container GPU samples of the arm's runs (``gpu_util_pct`` / ``gpu_mem_mb`` metric
points, ``RaceConfig.gpu_telemetry_s``), publishes one ``modal_stats`` event per arm per poll (raw
and dashboard WebSockets; never stored) and serves ``GET /api/races/{id}/modal``. Every other race
never polls Modal and gets ``{"enabled": false}``. Tests inject ``modal_stats_fn`` (no Modal
call). ``BUSY_MODAL_STATS=0`` turns polling off.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.agent.llm_base import OPUS_MODEL, session_id_for
from backend.benchmark.configs import RaceConfig, apply_surrogate_runtime_mult
from backend.core.events import ModalStatsEvent, RaceFinishedEvent
from backend.core.types import ArmKind, ArmSummary, CorrectionLabel, CorrectionSource, StepType, new_id
from backend.db import BenchmarkIntegrityError, Database
from backend.replay import ReplayCatalog, ReplayStore, replay_router
from backend.scheduler import ArmDeps, AsymmetricRaceError, EventBus, RaceEngine, race_spec
from workers.runners.modal_stats import (
    DEFAULT_POLL_INTERVAL_S,
    GPU_METRICS,
    STATS_SOURCE,
    ArmGpuTracker,
    FunctionStatsSnapshot,
    ModalStatsPoller,
    StatsFn,
    calls_in_flight_by_arm,
    function_for_hardware,
    latest_call_id,
)
from workers.targets.base import TargetSpec, get_target

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.environ.get("BUSY_MACHINES_DB", "data/busy_machines.db")
# Extra read-only sources of finished races to replay (.db files, .jsonl exports or folders of
# them), separated by os.pathsep (";" on Windows). The app's own DB is always a source.
REPLAY_SOURCES_ENV = "BUSY_MACHINES_REPLAY_SOURCES"
# Browser origins allowed to call the API (comma-separated). Unset = the Vite dev server; "" = none.
CORS_ORIGINS_ENV = "BUSY_MACHINES_CORS_ORIGINS"
DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")

MOCK_LLM_PATH = "backend.agent.mock_llm.MockLLM"
MUBIT_CLIENT_PATH = "backend.agent.mubit_client.MubitClient"
LOCAL_RUNNER_PATH = "workers.runners.local.LocalRunner"
MODAL_RUNNER_PATH = "workers.runners.modal_runner.ModalRunner"
SURROGATE_RUNNER_PATH = "workers.runners.surrogate.SurrogateRunner"

# Live Modal stats for demo races (docs/dashboard_modal.md). "0" turns polling off.
MODAL_STATS_ENV = "BUSY_MODAL_STATS"
# Every poll tick is appended here as JSON lines (race-<id>.jsonl), as evidence: the modal_stats
# events themselves are not stored in the race DB. Empty = no log.
MODAL_STATS_DIR = os.environ.get("BUSY_MODAL_STATS_DIR", "data/bench/modal_stats")
# Polling stops when the race is no longer live, and in any case this long after the race budget
# (pre-warm + scale-down allowance), so a race row stuck in 'running' cannot keep us polling.
MODAL_STATS_EXTRA_S = 900.0
MODAL_STATS_DISABLED_REASON = "live Modal stats are shown for demo races on per-arm Modal apps only"

ArmDepsFactory = Callable[[str, RaceConfig, TargetSpec, ArmKind], ArmDeps]


# =========================================================================== schemas


class RaceCreated(BaseModel):
    id: str
    config: RaceConfig


class FlagRequest(BaseModel):
    label: CorrectionLabel = CorrectionLabel.fake_improvement
    reason: str = ""
    source: CorrectionSource = CorrectionSource.human


class FlagResult(BaseModel):
    correction_id: str
    experiment_id: str
    applied_to_live_race: bool


class AdapterVersionIn(BaseModel):
    step_type: StepType
    version: str
    status: str = "candidate"
    trained_at: float | None = None
    notes: str = ""


# =========================================================================== default wiring


def _load(path: str) -> Any:
    mod, _, name = path.rpartition(".")
    return getattr(importlib.import_module(mod), name)


def _construct(cls: Any, **candidates: Any) -> Any:
    """Instantiate ``cls`` passing only the keyword arguments its signature accepts."""
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return cls()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return cls(**candidates)
    return cls(**{k: v for k, v in candidates.items() if k in params})


def _make_runner(cfg: RaceConfig, spec: TargetSpec, kind: ArmKind | None = None) -> Any:
    """A fresh runner per arm with exactly ``gpus_per_arm`` slots of the target's hardware.
    Local runs are laptop CPU at $0 (dev only: GPU $ shows 0, idle % is still real).
    Demo races with per-arm Modal apps (``cfg.modal_apps``, never a benchmark) pass the arm's
    app name; every other race builds the runner exactly as before (default app)."""
    n = cfg.gpus_per_arm
    if cfg.runner == "local":
        return _construct(_load(LOCAL_RUNNER_PATH), max_slots=n, hardware="CPU", price_per_second_usd=0.0)
    if cfg.runner == "surrogate":  # in-process, paced by predicted runtime; virtual GPU $ at the spec's price
        # A declared pacing (RaceConfig.surrogate_runtime_mult) is set process-wide first, so every
        # arm's runner reads the same BUSY_SURROGATE_RUNTIME_MULT (refused if the env disagrees).
        apply_surrogate_runtime_mult(cfg)
        return _construct(_load(SURROGATE_RUNNER_PATH), max_slots=n, hardware=spec.hardware, preload=[spec.name])
    app_name = cfg.modal_app_for(kind) if kind is not None else None
    if app_name is not None:
        if cfg.gpu_telemetry_s is not None:  # demo: in-container GPU sampling (same for every arm)
            return _construct(_load(MODAL_RUNNER_PATH), slots={spec.hardware: n}, app_name=app_name,
                              gpu_telemetry_s=cfg.gpu_telemetry_s)
        return _construct(_load(MODAL_RUNNER_PATH), slots={spec.hardware: n}, app_name=app_name)
    return _construct(_load(MODAL_RUNNER_PATH), slots={spec.hardware: n})


def default_arm_deps(race_id: str, cfg: RaceConfig, spec: TargetSpec, kind: ArmKind) -> ArmDeps:
    """One arm's researcher + runner. Every arm gets its own runner with the same slots, and
    every LLM arm the same researcher class, prompts and retry -> Opus-fallback policy.
    The recording mode is per arm (``cfg.record_mode_for(kind)``): ``off`` | ``metadata`` |
    ``full``, sent as ``x-middleware-record`` on every call of that arm except a fallback, which
    MubitClient always sends with ``off`` so it is a provider answer, not an adapter's."""
    runner = _make_runner(cfg, spec, kind)
    if kind == ArmKind.optuna:
        from backend.agent.optuna_arm import OptunaResearcher
        from backend.agent.study_worker import proposer_isolation

        # Benchmark races run the study in its own process (GIL isolation, same TPE and seed).
        return ArmDeps(kind=kind, researcher=OptunaResearcher(spec, cfg.seed, isolation=proposer_isolation(cfg.mode)),
                       runner=runner, llm_mode="none")

    from backend.agent.loop import LLMResearcher, load_prompts

    mode = cfg.record_mode_for(kind)
    # bool for off/full (the LLMClient protocol's ``record: bool``), the header value otherwise.
    record: bool | str = {"off": False, "full": True}.get(mode, mode)
    if cfg.llm == "mock":
        mock_kw: dict[str, Any] = {"latency_s": float(cfg.mock_latency_s.get(kind, 0.0))}
        if cfg.mock_trap_rate is not None:
            mock_kw["trap_rate"] = cfg.mock_trap_rate
        llm = _construct(_load(MOCK_LLM_PATH), spec=spec, seed=cfg.seed, **mock_kw)
        llm_mode, model = "mock", "mock-llm"
    else:
        if mode != "off":
            # Recording a draft prompt (or keeping its task fingerprint with "metadata") creates
            # a throwaway Mubit task and wastes paid example collection (CLAUDE.md "Freeze
            # prompts early"). Same gate as scripts/collect_examples.py.
            from backend.agent.prompts import is_frozen

            drafts = [s.value for s in StepType if not is_frozen(s)]
            if drafts:
                raise ValueError(f"record mode {mode!r} needs frozen prompts; still drafts: {', '.join(drafts)}")
        # Both LLM arms send the same provider model ID (claude-opus-5-5); Mubit decides
        # whether an adapter answers (Q1). Never a direct provider call. The ablation-only
        # "fast" arm sends claude-haiku-5-5 (cfg.model_for): a smaller faster model in the same
        # loop, not a Mubit adapter. Same retry -> Opus-fallback policy as every LLM arm.
        model = cfg.model_for(kind)
        llm = _construct(_load(MUBIT_CLIENT_PATH), arm_models={kind: model}, fallback_model=OPUS_MODEL)
        llm_mode = "mubit"
    researcher = LLMResearcher(kind=kind, llm=llm, spec=spec, session_id=session_id_for(race_id, kind, cfg.seed),
                               prompts=load_prompts(), fallback_model=OPUS_MODEL, record=record)
    # The arm's declared planner setup (D46 lean_planner feature, or a prompt-ablation arm):
    # propose prompt version + effort. Standard arms keep the frozen v1 at effort medium. The
    # mock LLM answers diff-form (v2) prompts too, so dry runs exercise the same path.
    variant = cfg.propose_variant_for(kind)
    if variant is not None:
        from backend.agent.loop import apply_propose_setup

        apply_propose_setup(researcher, *variant)
    return ArmDeps(kind=kind, researcher=researcher, runner=runner, model=model, llm_mode=llm_mode,
                   record_mode=mode)


# =========================================================================== app


class AppState:
    def __init__(self, db: Database, factory: ArmDepsFactory, tick_s: float) -> None:
        self.db = db
        self.bus = EventBus()
        self.factory = factory
        self.tick_s = tick_s
        self.engines: dict[str, RaceEngine] = {}
        self.tasks: dict[str, asyncio.Task[Any]] = {}
        self.modal_stats: ModalStatsHub | None = None   # set by create_app (demo races only)


def modal_stats_enabled(cfg: RaceConfig | None) -> bool:
    """Live Modal stats only for demo races on per-arm Modal apps (never benchmark / ablation)."""
    if cfg is None or os.environ.get(MODAL_STATS_ENV, "1").strip() == "0":
        return False
    return cfg.is_demo and bool(cfg.modal_apps) and cfg.runner == "modal" and not cfg.strict


def _config_of(race: dict[str, Any] | None) -> RaceConfig | None:
    if race is None:
        return None
    try:
        return RaceConfig.model_validate(race.get("config") or {})
    except Exception:  # noqa: BLE001 - unreadable config: no live Modal stats
        return None


class ModalStatsHub:
    """Live Modal stats for demo races (docs/dashboard_modal.md): one ``ModalStatsPoller`` and one
    ``ArmGpuTracker`` per race. It only reads (the app's DB, the event bus) and publishes
    ``modal_stats`` events with ``keep=False`` (not in the bus history, never stored); it never
    touches a race's engine. Races that are not demo races on per-arm Modal apps are never polled."""

    def __init__(self, state: AppState, *, stats_fn: StatsFn | None = None,
                 interval_s: float = DEFAULT_POLL_INTERVAL_S, log_dir: str | None = None) -> None:
        self.state = state
        self.stats_fn = stats_fn          # None: the real Modal call, created on the first demo poll
        self.interval_s = interval_s
        self.log_dir = log_dir or None
        self.pollers: dict[str, ModalStatsPoller] = {}
        self.trackers: dict[str, ArmGpuTracker] = {}
        state.bus.add_listener(self.on_event)

    # -- inputs -------------------------------------------------------------------

    def on_event(self, ev: Any) -> None:
        """Feed in-container GPU samples (metric points) into the race's tracker."""
        tr = self.trackers.get(ev.race_id)
        if tr is None:
            return
        if ev.type == "metric" and ev.name in GPU_METRICS:
            tr.observe(ev.arm.value, ev.experiment_id, ev.name, ev.value, ev.ts)
        elif ev.type == "experiment_finished":
            tr.finish(ev.arm.value, ev.experiment_id)

    def race_live(self, race_id: str) -> bool:
        eng = self.state.engines.get(race_id)
        if eng is not None:
            task = self.state.tasks.get(race_id)
            if task is not None and task.done():   # crashed, cancelled or pre-warm failed
                return False
            return eng.status in ("created", "running")
        race = self.state.db.get_race(race_id)    # a race run by another process (scripts/bench.py)
        return race is not None and race.get("status") in ("created", "running")

    def calls_by_arm(self, race_id: str) -> dict[str, list[dict[str, Any]]]:
        kinds = {str(a["id"]): str(a["kind"]) for a in self.state.db.get_arms(race_id)}
        return calls_in_flight_by_arm(self.state.db.list_experiments(race_id=race_id), kinds)

    # -- polling ------------------------------------------------------------------

    def publish(self, race_id: str, snaps: list[FunctionStatsSnapshot]) -> None:
        calls = self.calls_by_arm(race_id)
        tr = self.trackers.get(race_id)
        now_s = time.time()
        for s in snaps:
            gpu = tr.summary(s.arm, s.containers, s.calls_in_flight, now_s) if tr is not None and s.ok else {}
            arm_calls = calls.get(s.arm, [])
            self.state.bus.publish(ModalStatsEvent(
                race_id=race_id, ts=s.ts, arm=ArmKind(s.arm), app=s.app, function=s.function, ok=s.ok,
                error=s.error, containers=s.containers, calls_in_flight=s.calls_in_flight, backlog=s.backlog,
                input_headroom=s.input_headroom, idle_containers=s.idle_containers,
                call_ids=[c["modal_call_id"] for c in arm_calls if c["in_flight"]],
                latest_call_id=latest_call_id(arm_calls), poll_latency_s=s.poll_latency_s, **gpu), keep=False)

    def ensure(self, race_id: str, cfg: RaceConfig | None = None) -> ModalStatsPoller | None:
        """Start (or keep) polling a live demo race; a finished race keeps its last snapshot.
        None (and no Modal call) for every race that is not a demo race on per-arm apps."""
        if cfg is None:
            cfg = _config_of(self.state.db.get_race(race_id))
        if cfg is None or not modal_stats_enabled(cfg):
            return None
        self.trackers.setdefault(race_id, ArmGpuTracker())
        p = self.pollers.get(race_id)
        if (p is not None and p.running) or not self.race_live(race_id):
            return p
        try:
            fn = function_for_hardware(get_target(cfg.target).spec.hardware)
        except (KeyError, ImportError, ValueError) as e:
            log.warning("no Modal stats for race %s: %s", race_id, e)
            return p
        log_path = os.path.join(self.log_dir, f"race-{race_id}.jsonl") if self.log_dir else None
        if self.stats_fn is None:
            from workers.runners.modal_stats import modal_stats_fn

            self.stats_fn = modal_stats_fn()   # one cache of Modal function handles for every race
        p = ModalStatsPoller(
            {k.value: v for k, v in cfg.modal_apps.items()}, fn, stats_fn=self.stats_fn,
            interval_s=self.interval_s, keep_going=lambda: self.race_live(race_id),
            max_runtime_s=float(cfg.budget_s) + MODAL_STATS_EXTRA_S, log_path=log_path,
            on_snapshot=lambda snaps: self.publish(race_id, snaps),
        )
        self.pollers[race_id] = p
        p.start()
        return p

    async def stop_all(self) -> None:
        for p in list(self.pollers.values()):
            await p.stop()

    # -- view ---------------------------------------------------------------------

    def view(self, race_id: str, race: dict[str, Any], history: int = 0) -> dict[str, Any]:
        """``GET /api/races/{id}/modal``. Asking about a live demo race starts the poller if it
        is not running (e.g. the race runs in scripts/bench.py on the same DB)."""
        cfg = _config_of(race)
        if cfg is None or not modal_stats_enabled(cfg):
            return {"race_id": race_id, "enabled": False, "reason": MODAL_STATS_DISABLED_REASON}
        poller = self.ensure(race_id, cfg)
        calls = self.calls_by_arm(race_id)
        tr = self.trackers.get(race_id)
        now_s = time.time()
        arms: dict[str, Any] = {}
        for kind, app_name in cfg.modal_apps.items():
            a = kind.value
            snap = poller.latest.get(a) if poller is not None else None
            containers: int | None = None
            in_flight: int | None = None
            if snap is not None and snap.ok:
                containers, in_flight = snap.containers, snap.calls_in_flight
            arms[a] = {
                "app": app_name, "stats": snap.as_dict() if snap is not None else None,
                "gpu": tr.summary(a, containers, in_flight, now_s) if tr is not None else None,
                "calls_in_flight": [c["modal_call_id"] for c in calls.get(a, []) if c["in_flight"]],
                "latest_call_id": latest_call_id(calls.get(a, [])),
                "calls": calls.get(a, []),
            }
        pv = poller.view(history) if poller is not None else {}
        return {"race_id": race_id, "enabled": True, "race_status": race.get("status"),
                "polling": bool(poller is not None and poller.running), "source": STATS_SOURCE,
                "function": pv.get("function"), "interval_s": self.interval_s,
                "gpu_telemetry_s": cfg.gpu_telemetry_s, "ts": now_s, "arms": arms,
                "history": pv.get("history", [])}


def _arm_view(arm_row: dict[str, Any], live: tuple[ArmSummary, dict[str, Any]] | None) -> dict[str, Any]:
    out = {k: arm_row.get(k) for k in ("id", "kind", "model", "llm_mode", "is_dev", "n_slots", "hardware",
                                         "price_per_s", "status", "stop_reason", "started_at", "ended_at")}
    if live is not None:
        out["summary"] = live[0].model_dump(mode="json")
        out["report"] = live[1]
    else:
        out["summary"], out["report"] = arm_row.get("summary"), arm_row.get("report")
    return out


def create_app(
    db_path: str | None = None,
    *,
    arm_deps_factory: ArmDepsFactory | None = None,
    tick_s: float = 0.05,
    replay_sources: list[str] | None = None,
    modal_stats_fn: StatsFn | None = None,
    modal_stats_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    modal_stats_dir: str | None = MODAL_STATS_DIR,
) -> FastAPI:
    state = AppState(Database(db_path or DEFAULT_DB_PATH), arm_deps_factory or default_arm_deps, tick_s)
    hub = state.modal_stats = ModalStatsHub(state, stats_fn=modal_stats_fn, interval_s=modal_stats_interval_s,
                                            log_dir=modal_stats_dir)
    if replay_sources is None:
        replay_sources = [p for p in os.environ.get(REPLAY_SOURCES_ENV, "").split(os.pathsep) if p.strip()]
    replays = ReplayCatalog(state.db, replay_sources)

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
        yield
        for eng in state.engines.values():
            eng.stop("cancelled")
        for t in state.tasks.values():
            if not t.done():
                t.cancel()
        if state.tasks:
            await asyncio.gather(*state.tasks.values(), return_exceptions=True)
        await hub.stop_all()
        state.db.flush_metrics()

    app = FastAPI(title="Busy Machines", version="0.1.0", lifespan=lifespan)
    # CORS for the Vite dev frontend only (frontend/, http://localhost:5173). HTTP only: it changes
    # no route and no race behaviour. BUSY_MACHINES_CORS_ORIGINS="" turns it off; a comma list replaces it.
    cors_env = os.environ.get(CORS_ORIGINS_ENV)
    cors_origins = list(DEFAULT_CORS_ORIGINS) if cors_env is None else [o.strip() for o in cors_env.split(",") if o.strip()]
    if cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["GET", "POST"],
                           allow_headers=["*"])
    app.state.bm = state
    app.state.replays = ReplayStore()
    app.include_router(replay_router(replays, app.state.replays))
    from backend.dashboard import dashboard_router  # snapshot + dashboard WS streams (docs/dashboard_api.md)

    app.include_router(dashboard_router(state, replays, app.state.replays))
    from backend.demo import demo_router  # POST /api/demo/start (demo presets only, docs/dashboard_api.md)

    app.include_router(demo_router(state, check_keys=arm_deps_factory is None))

    def _race_or_404(race_id: str) -> dict[str, Any]:
        race = state.db.get_race(race_id)
        if race is None:
            raise HTTPException(404, f"race {race_id} not found")
        return race

    def _stored_config(race: dict[str, Any]) -> RaceConfig | None:
        try:
            return RaceConfig.model_validate(race.get("config") or {})
        except Exception:  # noqa: BLE001 - an unreadable config is treated as "refuse"
            return None

    def _corrections_allowed(race_id: str, eng: RaceEngine | None) -> bool:
        """Dev races always accept corrections; benchmark races only with benchmark_corrections="record"."""
        if eng is not None:
            return eng.corrections_allowed()
        race = state.db.get_race(race_id)
        if race is None:
            return True  # an experiment outside any race (nothing to protect)
        if race.get("is_dev", True):
            return True
        cfg = _stored_config(race)
        return cfg is not None and cfg.corrections_allowed

    def _record_stored_correction(race_id: str, cid: str, experiment_id: str, arm_id: str | None,
                                  body: FlagRequest) -> None:
        """A correction on a benchmark race that is not live in this process: list it in the
        stored fairness manifest too, so the headline blocks the race."""
        race = state.db.get_race(race_id)
        if race is None or race.get("is_dev", True):
            return
        fairness = dict(race.get("fairness") or {})
        fairness["corrections"] = list(fairness.get("corrections") or []) + [{
            "correction_id": cid, "experiment_id": experiment_id, "arm_id": arm_id, "label": body.label.value,
            "source": body.source.value, "reason": body.reason, "at": time.time(),
            "race_status": race.get("status"),
        }]
        state.db.update_race(race_id, fairness=fairness)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "live_races": [r for r, t in state.tasks.items() if not t.done()]}

    @app.post("/api/races", response_model=RaceCreated, status_code=201)
    async def create_race(cfg: RaceConfig) -> RaceCreated:
        try:
            get_target(cfg.target)
        except (KeyError, ImportError) as e:
            raise HTTPException(400, f"unknown target {cfg.target!r}: {e}") from e
        race_id = new_id()
        try:
            state.db.insert_race(race_id=race_id, target=cfg.target, budget_s=cfg.budget_s,
                                 gpus_per_arm=cfg.gpus_per_arm, seed=cfg.seed, config=cfg.model_dump(mode="json"),
                                 name=cfg.name, mode=cfg.mode, llm_mode=cfg.llm, runner=cfg.runner)
        except BenchmarkIntegrityError as e:
            raise HTTPException(400, str(e)) from e
        return RaceCreated(id=race_id, config=cfg)

    @app.get("/api/races")
    async def list_races(replayable: bool = False) -> list[dict[str, Any]]:
        """All races in the app DB; ``?replayable=true``: finished races from every replay
        source (app DB + ``replay_sources``), each with ``replayable`` and ``source``."""
        if replayable:
            return await asyncio.to_thread(replays.list_races, replayable_only=True)
        return state.db.list_races()

    @app.post("/api/races/{race_id}/start")
    async def start_race(race_id: str) -> dict[str, Any]:
        race = _race_or_404(race_id)
        if race["status"] != "created" or race_id in state.engines:
            raise HTTPException(409, f"race {race_id} is {race['status']}")
        cfg = RaceConfig.model_validate(race["config"])
        try:
            spec = race_spec(cfg)  # the race's goal (cfg.target_value) applied
            deps = [state.factory(race_id, cfg, spec, kind) for kind in cfg.arms]
            engine = RaceEngine(cfg, deps, spec=spec, race_id=race_id, db=state.db, bus=state.bus, tick_s=state.tick_s)
        except (AsymmetricRaceError, BenchmarkIntegrityError, ValueError, KeyError) as e:
            raise HTTPException(400, f"cannot start race: {e}") from e
        except (ImportError, AttributeError) as e:
            raise HTTPException(503, f"component not available yet: {e}") from e
        state.engines[race_id] = engine

        async def _run() -> None:
            try:
                await engine.run()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("race %s crashed", race_id)
                state.db.update_race(race_id, status="error")

        state.tasks[race_id] = asyncio.create_task(_run(), name=f"race-{race_id[:8]}")
        hub.ensure(race_id, cfg)   # demo races on per-arm Modal apps only; a no-op for every other race
        return {"id": race_id, "status": "running", "arms": [k.value for k in cfg.arms],
                "budget_s": cfg.budget_s, "gpus_per_arm": cfg.gpus_per_arm}

    @app.post("/api/races/{race_id}/stop")
    async def stop_race(race_id: str) -> dict[str, Any]:
        _race_or_404(race_id)
        eng = state.engines.get(race_id)
        if eng is None or eng.status != "running":
            raise HTTPException(409, "race is not running")
        eng.stop("cancelled")
        return {"id": race_id, "status": "stopping"}

    @app.get("/api/races/{race_id}")
    async def get_race(race_id: str) -> dict[str, Any]:
        race = _race_or_404(race_id)
        eng = state.engines.get(race_id)
        live: dict[str, tuple[ArmSummary, dict[str, Any]]] = {}
        if eng is not None and eng.status == "running":
            live = {k.value: a.live_summary() for k, a in eng.arms.items()}
        arms = [_arm_view(a, live.get(a["kind"])) for a in state.db.get_arms(race_id)]
        exps = state.db.list_experiments(race_id=race_id)
        for e in exps:
            e.pop("logs_tail", None)
        return {"race": race, "arms": arms, "experiments": exps,
                "corrections": state.db.list_corrections(race_id=race_id)}

    @app.get("/api/races/{race_id}/export.jsonl")
    async def export_race(race_id: str) -> StreamingResponse:
        _race_or_404(race_id)
        return StreamingResponse(state.db.export_race_jsonl(race_id), media_type="application/x-ndjson",
                                 headers={"Content-Disposition": f'attachment; filename="race-{race_id}.jsonl"'})

    @app.get("/api/races/{race_id}/modal")
    async def race_modal(race_id: str, history: int = Query(0, ge=0, le=600)) -> dict[str, Any]:
        """Live Modal view of a demo race (per arm: containers, calls in flight, backlog, idle
        containers, GPU % / memory from inside the containers, Modal call ids). Any other race:
        ``enabled: false`` and no Modal call (docs/dashboard_modal.md)."""
        return hub.view(race_id, _race_or_404(race_id), history)

    @app.post("/api/experiments/{experiment_id}/flag", response_model=FlagResult)
    async def flag_experiment(experiment_id: str, body: FlagRequest | None = None) -> FlagResult:
        body = body or FlagRequest()
        exp = state.db.get_experiment(experiment_id)
        if exp is None:
            raise HTTPException(404, f"experiment {experiment_id} not found")
        race_id = exp.get("race_id") or ""
        eng = state.engines.get(race_id)
        if not _corrections_allowed(race_id, eng):
            # A benchmark race refuses corrections (a human must not change its outcome) unless
            # its config says benchmark_corrections="record". Nothing is stored.
            raise HTTPException(409, f"race {race_id} is a benchmark race that does not accept corrections "
                                     f"(benchmark_corrections='refuse')")
        cid = state.db.insert_correction(experiment_id=experiment_id, source=body.source.value,
                                         label=body.label.value, reason=body.reason,
                                         race_id=exp.get("race_id"), arm_id=exp.get("arm_id"))
        applied = False
        if eng is not None:
            try:
                applied = eng.apply_correction(experiment_id, body.label, correction_id=cid, source=body.source,
                                               reason=body.reason) is not None and eng.status == "running"
            except BenchmarkIntegrityError as e:  # defence in depth: checked above
                raise HTTPException(409, str(e)) from e
        else:
            _record_stored_correction(race_id, cid, experiment_id, exp.get("arm_id"), body)
            from backend.core.events import CorrectionEvent

            arm = next((a for a in state.db.get_arms(exp["race_id"]) if a["id"] == exp["arm_id"]), None)
            state.bus.publish(CorrectionEvent(race_id=exp["race_id"], correction_id=cid, experiment_id=experiment_id,
                                              arm=ArmKind(arm["kind"]) if arm else None, source=body.source,
                                              label=body.label, reason=body.reason))
        return FlagResult(correction_id=cid, experiment_id=experiment_id, applied_to_live_race=applied)

    @app.get("/api/corrections")
    async def list_corrections() -> list[dict[str, Any]]:
        return state.db.list_corrections()

    @app.get("/api/adapters")
    async def list_adapters() -> dict[str, Any]:
        rows = state.db.list_adapter_versions()
        by_step: dict[str, list[dict[str, Any]]] = {s.value: [] for s in StepType}
        for r in rows:
            by_step.setdefault(r["step_type"], []).append(r)
        return {"adapters": rows, "by_step": by_step}

    @app.post("/api/adapters", status_code=201)
    async def upsert_adapter(body: AdapterVersionIn) -> dict[str, Any]:
        """Mirror an adapter version from the Mubit console (no API for it yet, Q9)."""
        aid = state.db.upsert_adapter_version(step_type=body.step_type.value, version=body.version,
                                              status=body.status, trained_at=body.trained_at, notes=body.notes)
        return {"id": aid}

    @app.websocket("/ws/races/{race_id}")
    async def race_events(websocket: WebSocket, race_id: str) -> None:
        await websocket.accept()
        race = state.db.get_race(race_id)
        if race is None:
            await websocket.close(code=4404)
            return
        q = state.bus.subscribe(race_id, replay=True)
        try:
            eng = state.engines.get(race_id)
            if eng is None and race["status"] not in ("created", "running") and not state.bus.history(race_id):
                # Finished before this process started: send the stored summary and close.
                sums = [ArmSummary.model_validate(a["summary"]) for a in state.db.get_arms(race_id) if a.get("summary")]
                ev = RaceFinishedEvent(race_id=race_id, summaries=sums, reason=race.get("finish_reason") or "budget")
                await websocket.send_text(ev.model_dump_json())
                await websocket.close()
                return
            hub.ensure(race_id)   # live demo race on per-arm Modal apps: modal_stats events every ~2 s
            while True:
                ev = await q.get()
                await websocket.send_text(ev.model_dump_json())
                if ev.type == "race_finished":
                    break
            await websocket.close()
        except WebSocketDisconnect:
            pass
        finally:
            state.bus.unsubscribe(race_id, q)

    return app


def __getattr__(name: str) -> Any:
    """``backend.app:app`` for uvicorn, created on first access (not at import)."""
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
