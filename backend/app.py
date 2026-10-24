"""FastAPI app: REST + WebSocket for races (architecture.md "API").

Run:  uv run uvicorn backend.app:app --reload        (DB at $BUSY_MACHINES_DB, default data/busy_machines.db)
  or: uv run uvicorn --factory backend.app:create_app

Dev mode (``llm="mock"``, ``runner="local"`` in the race body) wires the mock LLM and the
local runner, imported lazily by module path (``backend.agent.mock_llm.MockLLM``,
``workers.runners.local.LocalRunner``) so this file works before they exist. Real mode
(``llm="mubit"``) uses ``backend.agent.mubit_client.MubitClient``; ``runner="modal"`` uses
``workers.runners.modal_runner.ModalRunner``. Tests inject their own ``arm_deps_factory``.
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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.agent.llm_base import OPUS_MODEL, session_id_for
from backend.benchmark.configs import RaceConfig, apply_surrogate_runtime_mult
from backend.core.events import RaceFinishedEvent
from backend.core.types import ArmKind, ArmSummary, CorrectionLabel, CorrectionSource, StepType, new_id
from backend.db import BenchmarkIntegrityError, Database
from backend.replay import ReplayCatalog, ReplayStore, replay_router
from backend.scheduler import ArmDeps, AsymmetricRaceError, EventBus, RaceEngine, race_spec
from workers.targets.base import TargetSpec, get_target

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.environ.get("BUSY_MACHINES_DB", "data/busy_machines.db")
# Extra read-only sources of finished races to replay (.db files, .jsonl exports or folders of
# them), separated by os.pathsep (";" on Windows). The app's own DB is always a source.
REPLAY_SOURCES_ENV = "BUSY_MACHINES_REPLAY_SOURCES"

MOCK_LLM_PATH = "backend.agent.mock_llm.MockLLM"
MUBIT_CLIENT_PATH = "backend.agent.mubit_client.MubitClient"
LOCAL_RUNNER_PATH = "workers.runners.local.LocalRunner"
MODAL_RUNNER_PATH = "workers.runners.modal_runner.ModalRunner"
SURROGATE_RUNNER_PATH = "workers.runners.surrogate.SurrogateRunner"

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
        return _construct(_load(MODAL_RUNNER_PATH), slots={spec.hardware: n}, app_name=app_name)
    return _construct(_load(MODAL_RUNNER_PATH), slots={spec.hardware: n})


def default_arm_deps(race_id: str, cfg: RaceConfig, spec: TargetSpec, kind: ArmKind) -> ArmDeps:
    """One arm's researcher + runner. Every arm gets its own runner with the same slots, and
    every LLM arm the same researcher class, prompts and retry -> Opus-fallback policy.
    The recording mode is per arm (``cfg.record_mode_for(kind)``): ``off`` | ``metadata`` |
    ``full``, sent as ``x-middleware-record`` on every call of that arm (fallbacks included)."""
    runner = _make_runner(cfg, spec, kind)
    if kind == ArmKind.optuna:
        from backend.agent.optuna_arm import OptunaResearcher

        return ArmDeps(kind=kind, researcher=OptunaResearcher(spec, cfg.seed), runner=runner, llm_mode="none")

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
) -> FastAPI:
    state = AppState(Database(db_path or DEFAULT_DB_PATH), arm_deps_factory or default_arm_deps, tick_s)
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
        state.db.flush_metrics()

    app = FastAPI(title="Busy Machines", version="0.1.0", lifespan=lifespan)
    app.state.bm = state
    app.state.replays = ReplayStore()
    app.include_router(replay_router(replays, app.state.replays))

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
