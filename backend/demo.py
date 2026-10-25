"""One-click demo races for the stage screen: ``POST /api/demo/start?preset=...`` (docs/dashboard_api.md).

Only the demo presets in ``backend.benchmark.configs.DEMO_START_PRESETS`` can be started this way
(``demo-live-60s``: 60 s, LCBench surrogate, real Opus 5.5 through Mubit; ``demo-live-10s``: the
same race cut to 10 s, no baseline run, for the "Get started" page's live test; ``demo-cifar-modal-5m``:
5 min on Modal L4s, one Modal app per side). Every other preset is refused (400): benchmark and
ablation races run from ``scripts/bench.py``, never from a button. A demo race is a dev race with
``demo_label`` set, so it is labelled "demo run, not a benchmark result" in its name, header and
fairness manifest, and ``scripts/headline.py`` refuses it.

The race runs inside the API process (so ``/ws/races/{id}/dashboard`` streams it live) and starts
at once; the dashboard stream sends the race-so-far first, so connecting a moment later loses
nothing. One demo race at a time per process (409 while any race is live here), so a double click
never pays twice.

``dry_run=true`` rehearses the same preset at $0, like ``scripts/bench.py run --dry-run``: mock
LLM, the surrogate runner kept (or the laptop CPU runner instead of Modal), no pre-warm. Mock
numbers, never shown as a result.

``POST /api/mubit/check`` tests a Mubit key for the "Get started" page: with ``{"api_key": ...}``
one ``max_tokens=1`` Haiku call through the Mubit gateway (record off); without a key, only
whether the server's configured keys for the live test are present (no call).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request
from pydantic import BaseModel, Field

from backend.benchmark.configs import DEMO_START_PRESETS, PRESETS, RaceConfig
from backend.core.types import StepType, new_id
from backend.db import BenchmarkIntegrityError
from backend.scheduler import AsymmetricRaceError, RaceEngine, race_spec

log = logging.getLogger(__name__)

# The preset the "Get started" page's live test starts; the key check reports for it.
LIVE_TEST_PRESET = "demo-live-10s"
# Dry-run pacing of the live test (mock LLM seconds per call, per arm): R89's measured first-plan
# latencies (standard Opus plan 9.0-10.3 s, Busy Machines' lean plan 3.7-6.1 s). Mock, not a result.
LIVE_TEST_DRY_LATENCY_S = {"opus": 9.5, "opus_bm": 4.9}


class MubitCheckRequest(BaseModel):
    """Body of POST /api/mubit/check. ``api_key`` is used for one call and never stored or echoed."""

    api_key: str | None = Field(default=None, max_length=512)


DASHBOARD_WS = "/ws/races/{race_id}/dashboard?tick_s=1&raw=false"
RAW_WS = "/ws/races/{race_id}"


def demo_config(preset: str, *, dry_run: bool = False, seed: int = 0) -> RaceConfig:
    """The race config the endpoint starts: the preset, one seed, the ``[demo run]`` name tag that
    scripts/bench.py uses (plus ``[dry run]`` for a $0 rehearsal). ValueError for any preset
    outside ``DEMO_START_PRESETS``."""
    if preset not in DEMO_START_PRESETS:
        raise ValueError(f"preset {preset!r} cannot be started from the API; allowed: {', '.join(DEMO_START_PRESETS)}")
    base = PRESETS[preset]
    if not base.is_demo or base.strict:  # defence in depth: the allow-list holds demo presets only
        raise ValueError(f"preset {preset!r} is not a demo preset")
    upd: dict[str, Any] = {"seed": int(seed),
                           "name": f"{base.name} seed {int(seed)}" + (" [dry run]" if dry_run else "") + " [demo run]"}
    if dry_run:
        upd.update(mode="dev", llm="mock", runner="surrogate" if base.runner == "surrogate" else "local",
                   prewarm=False)
        if preset == LIVE_TEST_PRESET:
            upd["mock_latency_s"] = dict(LIVE_TEST_DRY_LATENCY_S)
    return RaceConfig.model_validate({**base.model_dump(), **upd})


def _mubit_key_problems(cfg: RaceConfig) -> list[str]:
    """Every Mubit project the race's LLM arms route a step to has a key (names env vars, never
    keys). Same rule as scripts/bench.py's live pre-flight, without starting anything."""
    from backend.agent.mubit_client import load_mubit_api_key, load_project_routing

    try:
        routing = load_project_routing()
    except Exception as e:  # noqa: BLE001
        return [f"Mubit project routing: {type(e).__name__}: {str(e)[:200]}"]
    projects = sorted({routing.project_for(a, s) for a in cfg.llm_arms for s in StepType})
    out = []
    for p in projects:
        var = routing.keys.get(p)
        if var is None:
            out.append(f"Mubit project {p} has no key entry in the routing")
            continue
        try:
            load_mubit_api_key(env_var=var)
        except Exception:  # noqa: BLE001 - the message must not carry a key
            out.append(f"{var} is not set (environment or .env)")
    return out


async def _prewarm(cfg: RaceConfig, deps: list[Any]) -> dict[str, Any] | None:
    """Modal demo races with ``prewarm``: pin and warm each arm's slots before t0 (the same steps as
    scripts/bench.py ``prewarm_pool`` / ``prewarm_pool_per_arm``). None when nothing to warm."""
    if not cfg.prewarm or cfg.runner != "modal" or not deps:
        return None
    if not all(hasattr(d.runner, "warm_up") and hasattr(d.runner, "prewarm") for d in deps):
        return None
    from workers.targets.base import get_target

    hw = get_target(cfg.target).spec.hardware
    if cfg.modal_apps:
        n = cfg.gpus_per_arm + cfg.prewarm_spare

        async def one(d: Any) -> dict[str, Any]:
            await d.runner.prewarm(hw, n)
            res = await d.runner.warm_up(hw, n, target=cfg.target)
            return {**res, "app": cfg.modal_app_for(d.kind)}

        results = await asyncio.gather(*(one(d) for d in deps))
        return {"hardware": hw, "wanted": n * len(deps), "warm": sum(int(r.get("warm") or 0) for r in results),
                "ok": all(bool(r.get("ok")) for r in results),
                "per_arm": {d.kind.value: r for d, r in zip(deps, results)},
                "note": "before t0: not in busy/idle/race $; one Modal app per arm (demo, started by the API)"}
    n = cfg.gpus_per_arm * len(cfg.arms) + cfg.prewarm_spare
    await deps[0].runner.prewarm(hw, n)
    res = await deps[0].runner.warm_up(hw, n, target=cfg.target)
    return {**res, "note": "before t0: not in busy/idle/race $ (demo, started by the API)"}


async def _cooldown(cfg: RaceConfig, deps: list[Any]) -> None:
    if not cfg.prewarm or cfg.runner != "modal":
        return
    from workers.targets.base import get_target

    hw = get_target(cfg.target).spec.hardware
    runners = [d.runner for d in deps] if cfg.modal_apps else [deps[0].runner]
    for r in runners:
        if hasattr(r, "cooldown"):
            try:
                await r.cooldown(hw, reset_scaledown=True)
            except Exception:  # noqa: BLE001 - must not hide the race result
                log.exception("demo race cooldown failed; unpin with ModalRunner.cooldown")
    for d in deps:
        drain = getattr(d.runner, "drain_background", None)
        if drain is not None:
            await drain()


async def _close_llms(deps: list[Any]) -> None:
    for d in deps:
        for attr in ("llm", "fallback_llm"):
            close = getattr(getattr(d.researcher, attr, None), "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # noqa: BLE001
                    pass


def demo_router(state: Any, *, check_keys: bool = True) -> APIRouter:
    """``state`` is ``backend.app.AppState``. ``check_keys``: refuse a real (Mubit) demo race when
    a routed project's key is missing (off when tests inject their own arm factory)."""
    from workers.runners.surrogate import ENV_MULTIPLIER

    router = APIRouter()
    env_mult_at_start = os.environ.get(ENV_MULTIPLIER)

    def _live() -> list[str]:
        return [r for r, t in state.tasks.items() if not t.done()]

    def _free_surrogate_pacing(cfg: RaceConfig) -> None:
        """The surrogate pacing is process-wide (apply_surrogate_runtime_mult refuses a second
        value). A value left by an EARLIER race of this process (none live) is cleared so the demo
        can set its own; a value that was in the environment when the API started is kept, and a
        mismatch is then refused as before."""
        if cfg.runner != "surrogate" or cfg.surrogate_runtime_mult is None or env_mult_at_start:
            return
        raw = os.environ.get(ENV_MULTIPLIER)
        if raw and not math.isclose(float(raw), float(cfg.surrogate_runtime_mult), rel_tol=0, abs_tol=1e-12):
            os.environ.pop(ENV_MULTIPLIER, None)

    @router.post("/api/demo/start", status_code=201)
    async def start_demo(request: Request, preset: str = Query(..., description=" | ".join(DEMO_START_PRESETS)),
                         dry_run: bool = False, seed: int = Query(0, ge=0)) -> dict[str, Any]:
        try:
            cfg = demo_config(preset, dry_run=dry_run, seed=seed)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, f"{e}") from e
        live = _live()
        if live:
            raise HTTPException(409, f"a race is already running in this process ({live[0][:8]}); "
                                     f"stop it first (POST /api/races/{live[0]}/stop)")
        if check_keys and cfg.llm == "mubit":
            problems = _mubit_key_problems(cfg)
            if problems:
                raise HTTPException(503, "cannot start a real demo race: " + "; ".join(problems))
        _free_surrogate_pacing(cfg)
        race_id = new_id()
        try:
            state.db.insert_race(race_id=race_id, target=cfg.target, budget_s=cfg.budget_s,
                                 gpus_per_arm=cfg.gpus_per_arm, seed=cfg.seed, config=cfg.model_dump(mode="json"),
                                 name=cfg.name, mode=cfg.mode, llm_mode=cfg.llm, runner=cfg.runner)
            spec = race_spec(cfg)
            deps = [state.factory(race_id, cfg, spec, kind) for kind in cfg.arms]
            engine = RaceEngine(cfg, deps, spec=spec, race_id=race_id, db=state.db, bus=state.bus, tick_s=state.tick_s)
        except (AsymmetricRaceError, BenchmarkIntegrityError, ValueError, KeyError) as e:
            if state.db.get_race(race_id) is not None:
                state.db.update_race(race_id, status="error")
            raise HTTPException(400, f"cannot start demo race: {e}") from e
        except (ImportError, AttributeError) as e:
            if state.db.get_race(race_id) is not None:
                state.db.update_race(race_id, status="error")
            raise HTTPException(503, f"component not available: {e}") from e
        state.engines[race_id] = engine
        warms = bool(cfg.prewarm and cfg.runner == "modal")

        async def _run() -> None:
            try:
                warm = await _prewarm(cfg, deps)
                if warm is not None:
                    engine.prewarm_info = warm
                    if not warm.get("ok"):
                        log.error("demo race %s: pre-warm reached %s of %s containers", race_id,
                                  warm.get("warm"), warm.get("wanted"))
                        state.db.update_race(race_id, status="error", finish_reason="prewarm_failed")
                        return
                await engine.run()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("demo race %s crashed", race_id)
                state.db.update_race(race_id, status="error")
            finally:
                await _cooldown(cfg, deps)
                await _close_llms(deps)

        state.tasks[race_id] = asyncio.create_task(_run(), name=f"demo-{race_id[:8]}")
        hub = getattr(state, "modal_stats", None)
        # demo-cifar-modal-5m (per-arm Modal apps): live Modal panel from the pre-warm on
        # (docs/dashboard_modal.md). None for every other preset and for dry runs (no Modal call).
        polling = hub.ensure(race_id, cfg) if hub is not None else None
        base_ws = str(request.base_url).replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
        ws = DASHBOARD_WS.format(race_id=race_id)
        return {
            "race_id": race_id, "preset": preset, "name": cfg.name, "demo_label": cfg.demo_label,
            "dry_run": dry_run, "status": "warming" if warms else "running",
            "ws": ws, "ws_url": base_ws + ws, "ws_raw": RAW_WS.format(race_id=race_id),
            "snapshot": f"/api/races/{race_id}/snapshot", "stop": f"/api/races/{race_id}/stop",
            "modal": f"/api/races/{race_id}/modal" if polling is not None else None,
            "arms": [k.value for k in cfg.arms], "budget_s": cfg.budget_s, "gpus_per_arm": cfg.gpus_per_arm,
            "llm": cfg.llm, "runner": cfg.runner, "cost_cap_usd": cfg.cost_cap_usd,
        }

    @router.post("/api/mubit/check")
    async def mubit_check(body: MubitCheckRequest | None = Body(default=None)) -> dict[str, Any]:
        """Test a Mubit key. With ``api_key``: one ``max_tokens=1`` call on the cheapest model through
        the Mubit gateway, record off (``check_mubit_key``). Without: whether the server's keys for
        the live test are configured (names env vars, never keys), with no call at all."""
        key = (body.api_key or "").strip() if body is not None else ""
        if key:
            from backend.agent.mubit_client import check_mubit_key

            res = await check_mubit_key(key)
            return {"mode": "call", **res}
        problems = _mubit_key_problems(demo_config(LIVE_TEST_PRESET))
        return {"mode": "presence", "ok": not problems, "latency_ms": None, "model": None, "request_id": None,
                "error": "; ".join(problems) or None, "preset": LIVE_TEST_PRESET}

    @router.get("/api/demo/presets")
    async def demo_presets() -> list[dict[str, Any]]:
        """The presets ``POST /api/demo/start`` accepts, with what each spends."""
        out = []
        for name in DEMO_START_PRESETS:
            c = PRESETS[name]
            out.append({"preset": name, "name": c.name, "demo_label": c.demo_label, "target": c.target,
                        "arms": [k.value for k in c.arms], "budget_s": c.budget_s, "gpus_per_arm": c.gpus_per_arm,
                        "llm": c.llm, "runner": c.runner, "cost_cap_usd": c.cost_cap_usd})
        return out

    return router
