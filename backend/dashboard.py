"""Dashboard API: what the two-column race screen needs, live and as replays (docs/dashboard_api.md).

A pure reducer (``DashboardState``) folds the race event stream (``backend.core.events``) into
one JSON snapshot per moment: race header (task, sides with plain labels, budget, goal, mode
tag, demo flag), and per arm the slot grid (busy / waiting / cancelled / off with timestamps),
AI bill (LLM $), GPU $, GPUs waiting %, experiments finished (by who proposed them), model
quality (best held-out score so far, with its step series and the goal time), checker verdicts
and "fake improvement caught" rows, and whether the AI planner is thinking.

Every number comes from events (the live engine's, or the replay's rebuilt ones); nothing is
estimated or typed in. At ``race_finished`` the headline numbers are replaced by the stored
``ArmSummary`` (the numbers the results log quotes), so the final frame matches the log.

Routes (mounted by ``backend.app.create_app``; they never touch a race's engine, so benchmark
race behaviour is unchanged):

- ``GET  /api/races/{id}/snapshot[?at_s=]``   one snapshot (live race: now; finished: at_s or end)
- ``GET  /api/dashboard/logged-races``        logged races the dashboard can replay (R78, R82, ...)
- ``WS   /ws/races/{id}/dashboard``           live: raw events + derived ``slot`` + periodic ``tick``
                                              (+ ``modal_stats`` every ~2 s for demo races on per-arm
                                              Modal apps, docs/dashboard_modal.md)
- ``WS   /ws/replays/{replay_id}/dashboard``  the same over a replay session, at its speed

The raw ``/ws/races/{id}`` and ``/ws/replays/{id}`` streams are unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from backend.benchmark.metrics import TRAP_CONTAMINATES_HELDOUT
from backend.core.events import (
    CorrectionEvent,
    Event,
    ExperimentFinishedEvent,
    ExperimentStartedEvent,
    GpuStateEvent,
    JudgeVerdictEvent,
    LLMCallEvent,
    NewBestEvent,
    PlannerFinishedEvent,
    PlannerStartedEvent,
    RaceFinishedEvent,
    SlotEvent,
    TickEvent,
)
from backend.core.types import CorrectionLabel, ExperimentStatus, SlotState

log = logging.getLogger(__name__)

SCHEMA = "busy-machines.dashboard/1"
SHOWCASE_PATH = Path(__file__).resolve().parents[1] / "data" / "frontend" / "showcase.json"

# ---------------------------------------------------------------------------- plain-language copy
# dashboard_spec.md section 3: left is always the standard assistant, right Busy Machines; the
# no-AI tuner is never on the main screen (D46: "secondary"). No Mubit claims in labels.
ARM_COLUMNS: dict[str, dict[str, str]] = {
    "opus": {"label": "Standard AI assistant", "sublabel": "Opus 5.5", "side": "left"},
    "opus_bm": {"label": "Busy Machines", "sublabel": "Opus 5.5", "side": "right"},
    "mubit": {"label": "Busy Machines", "sublabel": "Opus 5.5", "side": "right"},
    "optuna": {"label": "No-AI tuner", "sublabel": "no AI", "side": "secondary"},
    "mock": {"label": "Mock assistant", "sublabel": "dev only", "side": "left"},
    "fast": {"label": "Smaller-model assistant", "sublabel": "Haiku 5.5, ablation", "side": "secondary"},
    "lean": {"label": "Lean-planner assistant", "sublabel": "Opus 5.5, ablation", "side": "secondary"},
}
_SIDE_ORDER = {"left": 0, "right": 1, "secondary": 2}
TASK_LABELS: dict[str, str] = {
    "rl": "Game-playing AI",
    "lcbench": "Tabular model tuning",
    "cifar": "Image recognition",
    "lm": "Language model",
}
SOURCE_LABELS: dict[str, str] = {
    "llm": "planned by the AI",
    "sampler": "built-in tuner (no AI)",
    "baseline": "starting model",
    "optuna": "no-AI tuner",
}
LEAK_REASONS: dict[str, str] = {"contamination_or_eval_leak": "tested on data it had already seen"}
SCORE_DECIMALS: dict[str, int] = {"rl": 1}           # dashboard_spec.md 3.5; others 4
STOPPED = frozenset({ExperimentStatus.cancelled, ExperimentStatus.failed, ExperimentStatus.timed_out})
DEMO_TAG = "Demo run, not a benchmark result"
FEED_LIMIT = 200
FAKES_LIMIT = 100


def _minus(x: str) -> str:
    return x.replace("-", "−")


def fmt_score(v: float | None, decimals: int) -> str | None:
    if v is None or not math.isfinite(v):
        return None
    return _minus(f"{v:,.{decimals}f}")


def fmt_clock(s: float) -> str:
    s = max(0, int(s))   # rounded down (spec: race clock)
    return f"{s // 60:02d}:{s % 60:02d}"


def _short_clock(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 60}:{s % 60:02d}"


# ---------------------------------------------------------------------------- header


def logged_race_ref(race_id: str, path: Path = SHOWCASE_PATH) -> dict[str, Any] | None:
    """Which logged result a race belongs to ("R78", run 1 of 3), from the showcase file's
    ``sources.race_sets`` (generated from docs/results_log.md). None if it is not listed."""
    try:
        sets = json.loads(path.read_text(encoding="utf-8")).get("sources", {}).get("race_sets", {})
    except (OSError, ValueError):
        return None
    for ref, prefixes in sets.items():
        for i, p in enumerate(prefixes):
            if race_id.startswith(str(p)):
                return {"ref": ref, "run": i + 1, "runs": len(prefixes)}
    return None


def preset_of(name: str | None) -> str | None:
    """The preset a race was built from (scripts/bench.py names races "<preset name> seed N", plus
    tags such as " [dry run]" / " [demo run]")."""
    if not name:
        return None
    base = re.sub(r" seed \d+$", "", re.sub(r"( \[[^\]]+\])+$", "", name))
    try:
        from backend.benchmark.configs import PRESETS
    except Exception:  # noqa: BLE001 - header only
        return None
    for key, cfg in PRESETS.items():
        if cfg.name == base:
            return key
    return None


def _machines(n: int, hardware: str, runner: str) -> dict[str, Any]:
    hw = hardware or "?"
    simulated = runner == "surrogate"
    cpu = hw.upper() == "CPU"
    if runner == "local":
        label, caption = "Machines", f"{n} laptop CPU workers (dev)"
    elif simulated:
        label, caption = "Machines", f"{n} {'CPU workers' if cpu else 'GPUs'}, simulated machine time"
    elif cpu:
        label, caption = "Machines", f"{n} cloud CPU workers"
    else:
        label, caption = "GPUs", f"{n} {hw} GPUs on Modal" if runner == "modal" else f"{n} {hw} GPUs"
    return {"label": label, "caption": caption, "count": n, "hardware": hw, "simulated": simulated,
            "waiting_label": f"{label} waiting"}


def build_header(race: dict[str, Any], arms: Sequence[dict[str, Any]] = (), *, mode: str = "live",
                 replay: dict[str, Any] | None = None) -> dict[str, Any]:
    """Static race info for the header and the two columns (dashboard_spec.md sections 2-3).
    ``arms``: the race's arm rows (``n_slots``, ``hardware``, ``price_per_s``); empty before the
    race starts (then the config's arms and ``gpus_per_arm`` are used, price unknown)."""
    cfg = race.get("config") or {}
    target = str(race.get("target") or cfg.get("target") or "")
    budget_s = float(race.get("budget_s") or cfg.get("budget_s") or 0.0)
    gpus = int(race.get("gpus_per_arm") or cfg.get("gpus_per_arm") or 0)
    runner = str(race.get("runner") or cfg.get("runner") or "")
    cap = cfg.get("cost_cap_usd")
    rows = {str(a["kind"]): a for a in arms}
    kinds = [str(k) for k in (cfg.get("arms") or list(rows))]
    hw = next((str(a.get("hardware")) for a in arms if a.get("hardware")), "")
    if not hw:
        try:
            from workers.targets.base import get_target

            hw = get_target(target).spec.hardware
        except Exception:  # noqa: BLE001 - header only
            hw = ""
    feats = cfg.get("engine_features") or {}
    try:
        from backend.benchmark.configs import DEFAULT_ENGINE_FEATURES, ENGINE_FEATURE_LABELS
    except Exception:  # noqa: BLE001
        DEFAULT_ENGINE_FEATURES, ENGINE_FEATURE_LABELS = {}, {}  # type: ignore[assignment]
    arm_list: list[dict[str, Any]] = []
    for k in kinds:
        a = rows.get(k, {})
        col = ARM_COLUMNS.get(k, {"label": k, "sublabel": "", "side": "secondary"})
        f = list(feats.get(k) or [str(x) for x in DEFAULT_ENGINE_FEATURES.get(k, ())])  # type: ignore[call-overload]
        arm_list.append({
            "kind": k, **col, "uses_ai": k != "optuna",
            "n_slots": int(a.get("n_slots") or gpus), "hardware": a.get("hardware") or hw,
            "price_per_slot_s": float(a["price_per_s"]) if a.get("price_per_s") is not None else None,
            "model": a.get("model"), "features": f,
            "feature_labels": [ENGINE_FEATURE_LABELS.get(x, x) for x in f],
        })
    arm_list.sort(key=lambda x: _SIDE_ORDER.get(x["side"], 9))
    demo_label = cfg.get("demo_label")
    is_replay = mode == "replay"
    logged = logged_race_ref(str(race.get("id") or "")) if race.get("id") else None
    machines = _machines(gpus, hw, runner)
    speed = (replay or {}).get("speed")
    if is_replay:
        what = (f"Replay of logged race {logged['ref']}, run {logged['run']} of {logged['runs']}" if logged
                else "Replay of a logged race")
        mode_tag = what + (f" · {speed:g}× speed" if speed else "")
    elif demo_label:
        mode_tag = DEMO_TAG
    elif cfg.get("llm") == "mock" or race.get("is_dev"):
        mode_tag = "Dev run (mock AI), not a benchmark result" if cfg.get("llm") == "mock" else "Dev run, not a benchmark result"
    else:
        mode_tag = "Benchmark race"
    unit = f"{hw} GPUs" if machines["label"] == "GPUs" else "machines"
    budget_text = f"{budget_s / 60:g} min" + (f" · up to ${float(cap):.2f} per side" if cap else "") + \
        f" · {gpus} {unit} each"
    tv = cfg.get("target_value")
    decimals = SCORE_DECIMALS.get(target, 4)
    direction = None
    try:
        from workers.targets.base import get_target

        spec = get_target(target).spec
        direction = spec.direction
        if tv is None:
            tv = spec.target_value
    except Exception:  # noqa: BLE001
        pass
    return {
        "race_id": race.get("id"), "race_name": race.get("name") or cfg.get("name") or "",
        "title": "Busy Machines", "subtitle": "compute intelligence for training sweeps",
        "task": TASK_LABELS.get(target, target), "target": target,
        "budget_s": budget_s, "cost_cap_usd": cap, "gpus_per_arm": gpus, "budget_text": budget_text,
        "goal": tv, "goal_text": f"Goal: {_minus(format(float(tv), 'g'))}" if tv is not None else None,
        "direction": direction, "higher_is_better": direction != "min" if direction else None,
        "score_decimals": decimals,
        "machines": machines,
        "arms": arm_list,
        "mode": mode, "is_replay": is_replay, "replay": replay,
        "race_mode": race.get("mode") or cfg.get("mode"), "llm": race.get("llm_mode") or cfg.get("llm"),
        "runner": runner, "preset": preset_of(race.get("name") or cfg.get("name")),
        "demo_run": bool(demo_label), "demo_label": demo_label, "mode_tag": mode_tag,
        "logged_race": logged, "seed": race.get("seed", cfg.get("seed")),
        "started_at": race.get("started_at"), "stored_status": race.get("status"),
        "ai_cost_note": "AI $ at list price (estimate)",
        "gpu_cost_note": "simulated machine time" if runner == "surrogate" else
                         ("laptop CPU, $0" if runner == "local" else None),
    }


# ---------------------------------------------------------------------------- reducer


@dataclass
class _Slot:
    slot_id: str
    raw: str | None = None             # engine state: busy | idle (None = no event yet)
    opened_at: float | None = None
    closed_at: float | None = None
    changed_at: float = 0.0            # last raw change (accounting)
    since: float = 0.0                 # last display change
    busy_s: float = 0.0
    idle_s: float = 0.0
    experiment_id: str | None = None   # running run, or the last run (for "cancelled")
    stopped: bool = False
    display: str = "off"
    held: bool = False                 # a "busy" slot event waits for experiment_started (its source)

    def accrue(self, t: float) -> None:
        if self.raw is None or self.closed_at is not None:
            return
        d = max(0.0, t - self.changed_at)
        if self.raw == SlotState.busy.value:
            self.busy_s += d
        else:
            self.idle_s += d
        self.changed_at = max(self.changed_at, t)

    def paid(self, t: float) -> tuple[float, float]:
        """(busy_s, idle_s) up to ``t``, without changing state."""
        b, i = self.busy_s, self.idle_s
        if self.raw is not None and self.closed_at is None and t > self.changed_at:
            if self.raw == SlotState.busy.value:
                b += t - self.changed_at
            else:
                i += t - self.changed_at
        return b, i


@dataclass
class _Arm:
    kind: str
    price: float
    slots: dict[str, _Slot] = field(default_factory=dict)
    ai_bill: float = 0.0
    llm_calls: int = 0
    llm_invalid: int = 0
    started: dict[str, int] = field(default_factory=dict)     # by source
    finished: dict[str, int] = field(default_factory=dict)    # succeeded, by source
    failed: int = 0
    cancelled: int = 0
    counted: list[tuple[float, float, str]] = field(default_factory=list)   # (ts, quality, exp id)
    best: tuple[float, float, str] | None = None
    series: list[list[float | None]] = field(default_factory=list)  # [race_elapsed_s, best quality]
    baseline_quality: float | None = None
    goal: tuple[float, float, str] | None = None
    verdicts: dict[str, int] = field(default_factory=lambda: {"valid": 0, "invalid": 0, "unsure": 0, "none": 0})
    claims_rejected_real: int = 0                            # invalid on a run that was not a fake
    fakes: list[dict[str, Any]] = field(default_factory=list)
    fake_ids: set[str] = field(default_factory=set)
    planner_since: float | None = None
    planner_calls: int = 0
    planner_last_k: int | None = None
    planner_busy_s: float = 0.0
    reported_best: dict[str, Any] | None = None
    final: dict[str, Any] | None = None


class DashboardState:
    """Folds one race's events into dashboard snapshots. ``apply`` returns the derived ``slot``
    events the change caused (for streaming). Not thread-safe: call from one task."""

    def __init__(self, header: dict[str, Any], *, t0: float | None = None) -> None:
        self.header = header
        self.race_id = str(header.get("race_id") or "")
        self.t0 = t0 if t0 is not None else header.get("started_at")
        self.budget_s = float(header.get("budget_s") or 0.0)
        self.decimals = int(header.get("score_decimals") or 4)
        self.direction = header.get("direction") or "max"
        self.goal = header.get("goal")
        self.target = header.get("target")
        self.arms: dict[str, _Arm] = {}
        self.exps: dict[str, dict[str, Any]] = {}
        # No slot pays after the budget (the engine closes them at the deadline at the latest).
        self.feed: list[dict[str, Any]] = []
        self.finished_reason: str | None = None
        self.end_ts: float | None = None
        self.last_ts: float | None = None
        self.n_events = 0
        for a in header.get("arms") or []:
            self._arm(a["kind"], price=a.get("price_per_slot_s"))

    # ------------------------------------------------------------------ helpers

    def set_header(self, header: dict[str, Any]) -> None:
        """Refresh static info (e.g. arm prices once the race started)."""
        self.header = header
        if self.t0 is None:
            self.t0 = header.get("started_at")
        for a in header.get("arms") or []:
            arm = self._arm(a["kind"])
            if a.get("price_per_slot_s") is not None:
                arm.price = float(a["price_per_slot_s"])

    @property
    def header_complete(self) -> bool:
        return all(a.get("price_per_slot_s") is not None for a in self.header.get("arms") or [])

    def _arm(self, kind: str, price: float | None = None) -> _Arm:
        a = self.arms.get(kind)
        if a is None:
            a = self.arms[kind] = _Arm(kind=kind, price=float(price or 0.0))
        return a

    def _cap(self, t: float) -> float:
        if self.t0 is not None and self.budget_s:
            return min(t, self.t0 + self.budget_s)
        return t

    def _el(self, ts: float) -> float:
        return round(ts - self.t0, 3) if self.t0 is not None else 0.0

    def _better(self, a: float, b: float | None) -> bool:
        return b is None or (a < b if self.direction == "min" else a > b)

    def _reaches(self, q: float) -> bool:
        if self.goal is None:
            return False
        return q <= float(self.goal) if self.direction == "min" else q >= float(self.goal)

    def _set_display(self, arm: _Arm, s: _Slot, ts: float, out: list[SlotEvent]) -> None:
        if s.closed_at is not None or s.raw is None:
            d = "off"
        elif s.raw == SlotState.busy.value:
            d = "busy"
        else:
            d = "cancelled" if s.stopped else "waiting"
        if d != s.display:
            s.display, s.since = d, ts
            exp = s.experiment_id if d in ("busy", "cancelled") else None
            # The engine sends gpu_state busy just before experiment_started: hold the slot event
            # until then, so it carries who proposed the run.
            s.held = d == "busy" and exp is not None and exp not in self.exps
            if s.held:
                return
            out.append(SlotEvent(race_id=self.race_id, arm=arm.kind, slot_id=s.slot_id, state=d,  # type: ignore[arg-type]
                                 experiment_id=exp, source=(self.exps.get(exp) or {}).get("source") if exp else None,
                                 race_elapsed_s=self._el(ts), ts=ts))

    def _recompute_best(self, arm: _Arm, ts: float) -> None:
        best: tuple[float, float, str] | None = None
        goal: tuple[float, float, str] | None = None
        for c in arm.counted:
            if best is None or self._better(c[1], best[1]):
                best = c
            if goal is None and self._reaches(c[1]):
                goal = c
        if (best is None) != (arm.best is None) or (best and arm.best and best[1] != arm.best[1]):
            arm.series.append([self._el(ts), best[1] if best else None])
        arm.best, arm.goal = best, goal

    def _push_feed(self, item: dict[str, Any]) -> None:
        self.feed.append(item)
        if len(self.feed) > FEED_LIMIT:
            del self.feed[: len(self.feed) - FEED_LIMIT]

    def _add_fake(self, arm: _Arm, exp_id: str, ts: float, *, by: str, claimed: float | None,
                  quality: float | None, llm_reason: str | None, source: str | None) -> None:
        if exp_id in arm.fake_ids:
            return
        arm.fake_ids.add(exp_id)
        real = quality
        if real is not None and (self.target in TRAP_CONTAMINATES_HELDOUT
                                 or (claimed is not None and abs(real - claimed) <= 1e-9)):
            real = None   # the shortcut also spoils that score (lcbench), or nothing to compare
        why = LEAK_REASONS.get(llm_reason or "", "not a real gain")
        at = self._el(ts)
        text = f"{_short_clock(at)} · claimed {fmt_score(claimed, self.decimals) or '?'}"
        if real is not None:
            text += f", real score {fmt_score(real, self.decimals)}"
        text += f" ({why})"
        if by != "checker":
            text += " · flagged by a person" if by == "human" else f" · flagged by {by}"
        row = {"experiment_id": exp_id, "at_s": at, "ts": ts, "claimed": claimed, "real": real, "why": why,
               "by": by, "source": source, "text": text}
        arm.fakes.append(row)
        if len(arm.fakes) > FAKES_LIMIT:
            del arm.fakes[: len(arm.fakes) - FAKES_LIMIT]
        self._push_feed({"kind": "fake_caught", "arm": arm.kind, **row,
                         "headline": "Fake improvement caught"})

    # ------------------------------------------------------------------ apply

    def apply(self, ev: Event | Any) -> list[SlotEvent]:
        out: list[SlotEvent] = []
        ts = float(ev.ts)
        self.n_events += 1
        if ev.type in ("tick", "slot", "metric", "modal_stats"):
            return out
        self.last_ts = ts if self.last_ts is None else max(self.last_ts, ts)
        if self.t0 is None:
            self.t0 = ts
        if isinstance(ev, GpuStateEvent):
            arm = self._arm(ev.arm.value)
            s = arm.slots.get(ev.slot_id)
            if s is None:
                s = arm.slots[ev.slot_id] = _Slot(slot_id=ev.slot_id)
            st = ev.state.value
            if s.closed_at is not None:
                return out
            if s.raw is None:
                s.raw, s.opened_at, s.changed_at = st, ts, self._cap(ts)
            else:
                s.accrue(self._cap(ts))
                if st == SlotState.idle.value and s.raw == SlotState.idle.value:
                    s.closed_at = ts     # idle -> idle: the engine closed the slot (release / end)
                s.raw = st
            if st == SlotState.busy.value:
                s.experiment_id, s.stopped = ev.experiment_id, False
            self._set_display(arm, s, ts, out)
        elif isinstance(ev, ExperimentStartedEvent):
            arm = self._arm(ev.arm.value)
            src = ev.source or "unknown"
            self.exps[ev.experiment_id] = {"arm": arm.kind, "source": src, "slot_id": ev.slot_id,
                                           "started_ts": ts}
            arm.started[src] = arm.started.get(src, 0) + 1
            s = arm.slots.get(ev.slot_id)
            if s is not None and s.held and s.experiment_id == ev.experiment_id:
                s.held = False
                out.append(SlotEvent(race_id=self.race_id, arm=arm.kind, slot_id=s.slot_id, state="busy",
                                     experiment_id=ev.experiment_id, source=src, race_elapsed_s=self._el(s.since),
                                     ts=s.since))
        elif isinstance(ev, ExperimentFinishedEvent):
            arm = self._arm(ev.arm.value)
            x = self.exps.setdefault(ev.experiment_id, {"arm": arm.kind, "source": "unknown", "slot_id": ev.slot_id})
            x.update(status=ev.status.value, score=ev.score, quality=ev.quality, is_trap=ev.is_trap, ended_ts=ts)
            src = x.get("source") or "unknown"
            if ev.status == ExperimentStatus.succeeded:
                arm.finished[src] = arm.finished.get(src, 0) + 1
                if ev.quality is not None and not ev.is_trap and math.isfinite(ev.quality):
                    if src == "baseline" and arm.baseline_quality is None:
                        arm.baseline_quality = ev.quality
                    c = (ts, float(ev.quality), ev.experiment_id)
                    arm.counted.append(c)
                    if arm.best is None or self._better(c[1], arm.best[1]):
                        arm.best = c
                        arm.series.append([self._el(ts), c[1]])
                    if arm.goal is None and self._reaches(c[1]):
                        arm.goal = c
                        self._push_feed({"kind": "goal_reached", "arm": arm.kind, "at_s": self._el(ts), "ts": ts,
                                         "experiment_id": ev.experiment_id, "quality": c[1],
                                         "text": f"Goal reached at {_short_clock(self._el(ts))}"})
            elif ev.status in (ExperimentStatus.failed, ExperimentStatus.timed_out):
                arm.failed += 1
            elif ev.status == ExperimentStatus.cancelled:
                arm.cancelled += 1
            if ev.status in STOPPED:
                s = arm.slots.get(ev.slot_id)
                if s is not None and s.experiment_id == ev.experiment_id and s.raw == SlotState.idle.value:
                    s.stopped = True
                    self._set_display(arm, s, ts, out)
        elif isinstance(ev, JudgeVerdictEvent):
            arm = self._arm(ev.arm.value)
            v = ev.verdict.value if ev.verdict is not None else "none"
            arm.verdicts[v] = arm.verdicts.get(v, 0) + 1
            x = self.exps.setdefault(ev.experiment_id, {"arm": arm.kind, "source": ev.source or "unknown"})
            x["verdict"] = v
            if v == "invalid":
                if not ev.is_trap:
                    arm.claims_rejected_real += 1
                before = len(arm.counted)
                arm.counted = [c for c in arm.counted if c[2] != ev.experiment_id]
                if len(arm.counted) != before:
                    self._recompute_best(arm, ts)   # a rejected claim never counts (metrics.is_counted)
            if ev.fake_caught:
                self._add_fake(arm, ev.experiment_id, ts, by="checker", claimed=ev.claimed_score,
                               quality=ev.quality, llm_reason=ev.llm_reason, source=ev.source)
        elif isinstance(ev, CorrectionEvent):
            kind = ev.arm.value if ev.arm is not None else (self.exps.get(ev.experiment_id) or {}).get("arm")
            if kind and ev.label == CorrectionLabel.fake_improvement:
                x = self.exps.get(ev.experiment_id) or {}
                self._add_fake(self._arm(kind), ev.experiment_id, ts, by=ev.source.value, claimed=x.get("score"),
                               quality=x.get("quality"), llm_reason=None, source=x.get("source"))
            self._push_feed({"kind": "correction", "arm": kind, "at_s": self._el(ts), "ts": ts,
                             "experiment_id": ev.experiment_id, "label": ev.label.value, "by": ev.source.value,
                             "reason": ev.reason})
        elif isinstance(ev, LLMCallEvent):
            arm = self._arm(ev.call.arm.value)
            arm.ai_bill += float(ev.call.cost_usd or 0.0)
            arm.llm_calls += 1
            arm.llm_invalid += 0 if ev.call.valid else 1
        elif isinstance(ev, PlannerStartedEvent):
            arm = self._arm(ev.arm.value)
            arm.planner_since, arm.planner_last_k = ts, ev.k
            arm.planner_calls += 1
        elif isinstance(ev, PlannerFinishedEvent):
            arm = self._arm(ev.arm.value)
            arm.planner_busy_s += max(0.0, ts - (arm.planner_since if arm.planner_since is not None else ev.started_at))
            arm.planner_since = None
        elif isinstance(ev, NewBestEvent):
            arm = self._arm(ev.arm.value)
            arm.reported_best = {"experiment_id": ev.experiment_id, "score": ev.score, "at_s": self._el(ts),
                                 "source": (self.exps.get(ev.experiment_id) or {}).get("source")}
        elif isinstance(ev, RaceFinishedEvent):
            self.finished_reason, self.end_ts = ev.reason, ts
            for arm in self.arms.values():
                if arm.planner_since is not None:
                    arm.planner_busy_s += max(0.0, ts - arm.planner_since)
                    arm.planner_since = None
                for s in arm.slots.values():
                    if s.closed_at is None and s.raw is not None:
                        s.accrue(self._cap(ts))
                        s.closed_at = ts
                    self._set_display(arm, s, ts, out)
            for summ in ev.summaries:
                arm = self._arm(summ.arm.value)
                arm.final = summ.model_dump(mode="json")
        return out

    # ------------------------------------------------------------------ snapshot

    def snapshot(self, at: float | None = None) -> dict[str, Any]:
        """The dashboard state at race time ``at`` (default: the last event; a finished race:
        its end). Accrual between events (bills, waiting %) is computed up to ``at``."""
        if self.end_ts is not None:
            at = self.end_ts if at is None else min(at, self.end_ts)
        elif at is None:
            at = self.last_ts if self.last_ts is not None else (self.t0 or time.time())
        if self.t0 is not None and self.budget_s:
            at = min(max(at, self.t0), self.t0 + self.budget_s + 60.0)
        elapsed = max(0.0, at - self.t0) if self.t0 is not None else 0.0
        status = "finished" if self.end_ts is not None else ("running" if self.last_ts is not None else "not_started")
        arms_out: list[dict[str, Any]] = []
        order = {a["kind"]: i for i, a in enumerate(self.header.get("arms") or [])}
        hdr = {a["kind"]: a for a in self.header.get("arms") or []}
        for kind in sorted(self.arms, key=lambda k: order.get(k, 99)):
            arm = self.arms[kind]
            h = hdr.get(kind, {})
            n_slots = int(h.get("n_slots") or len(arm.slots))
            busy = idle = 0.0
            counts = {"busy": 0, "waiting": 0, "cancelled": 0, "off": 0}
            slots = []
            for sid in sorted(arm.slots, key=_natural_key):
                s = arm.slots[sid]
                b, i = s.paid(self._cap(at))
                busy += b
                idle += i
                counts[s.display] += 1
                exp = s.experiment_id if s.display in ("busy", "cancelled") else None
                slots.append({"slot_id": sid, "state": s.display, "since_s": self._el(s.since) if s.raw else None,
                              "experiment_id": exp,
                              "source": (self.exps.get(exp) or {}).get("source") if exp else None})
            counts["off"] += max(0, n_slots - len(arm.slots))
            paid = busy + idle
            gpu = paid * arm.price
            open_slots = counts["busy"] + counts["waiting"] + counts["cancelled"]
            fin = sum(arm.finished.values())
            planner_thinking = arm.planner_since is not None
            q = arm.best
            out = {
                "kind": kind, "label": h.get("label", kind), "sublabel": h.get("sublabel", ""),
                "side": h.get("side", "secondary"), "uses_ai": h.get("uses_ai", kind != "optuna"),
                "n_slots": n_slots, "slots": slots, "slot_counts": counts,
                "working_now": counts["busy"], "open_now": open_slots,
                "waiting_now_pct": round(100.0 * (open_slots - counts["busy"]) / open_slots, 2) if open_slots else None,
                "waiting_so_far_pct": round(100.0 * idle / paid, 3) if paid > 0 else 0.0,
                "busy_s": round(busy, 3), "idle_s": round(idle, 3),
                "ai_bill_usd": round(arm.ai_bill, 6), "gpu_bill_usd": round(gpu, 6),
                "total_usd": round(arm.ai_bill + gpu, 6),
                "llm_calls": arm.llm_calls, "llm_invalid_calls": arm.llm_invalid,
                "experiments_started": sum(arm.started.values()), "started_by_source": dict(arm.started),
                "experiments_finished": fin, "finished_by_source": dict(arm.finished),
                "experiments_failed": arm.failed, "experiments_cancelled": arm.cancelled,
                "model_quality": {
                    "best": q[1] if q else None, "best_text": fmt_score(q[1], self.decimals) if q else None,
                    "experiment_id": q[2] if q else None, "at_s": self._el(q[0]) if q else None,
                    "source": (self.exps.get(q[2]) or {}).get("source") if q else None,
                    "baseline": arm.baseline_quality,
                    "series": [p for p in arm.series if p[0] <= elapsed + 1e-6] if self.t0 is not None else arm.series,
                    "goal_reached_at_s": self._el(arm.goal[0]) if arm.goal else None,
                    "goal_reached_text": (f"Goal reached at {_short_clock(self._el(arm.goal[0]))}" if arm.goal else None),
                },
                "checks": {**arm.verdicts, "claims_rejected_not_fake": arm.claims_rejected_real},
                "fakes_caught": len(arm.fake_ids), "fakes": arm.fakes[-10:],
                "planner": {"thinking": planner_thinking,
                            "since_s": self._el(arm.planner_since) if planner_thinking else None,
                            "calls": arm.planner_calls, "last_k": arm.planner_last_k,
                            "thinking_s": round(arm.planner_busy_s + (max(0.0, self._cap(at) - arm.planner_since)
                                                                      if planner_thinking else 0.0), 3)},
                "reported_best": arm.reported_best,
                "final": arm.final,
            }
            if arm.final is not None:
                # The stored summary is what the results log quotes: the final frame shows exactly it.
                f = arm.final
                out.update(ai_bill_usd=f.get("llm_cost_usd", out["ai_bill_usd"]),
                           gpu_bill_usd=f.get("gpu_cost_usd", out["gpu_bill_usd"]),
                           total_usd=f.get("total_cost_usd", out["total_usd"]),
                           waiting_so_far_pct=f.get("gpu_idle_pct", out["waiting_so_far_pct"]),
                           experiments_finished=f.get("experiments_finished", fin),
                           experiments_failed=f.get("experiments_failed", arm.failed))
                mq = out["model_quality"]
                if f.get("best_score") is not None:
                    mq["best"], mq["best_text"] = f["best_score"], fmt_score(f["best_score"], self.decimals)
                    mq["experiment_id"] = f.get("best_experiment_id", mq["experiment_id"])
                if f.get("time_to_target_s") is not None:
                    mq["goal_reached_at_s"] = round(float(f["time_to_target_s"]), 3)
                    mq["goal_reached_text"] = f"Goal reached at {_short_clock(float(f['time_to_target_s']))}"
            arms_out.append(out)
        feed = [x for x in self.feed if x.get("ts", 0.0) <= at + 1e-6][-20:]
        return {
            "schema": SCHEMA, "race_id": self.race_id, "ts": at, "race_elapsed_s": round(elapsed, 3),
            "clock": f"{fmt_clock(elapsed)} / {fmt_clock(self.budget_s)}", "status": status,
            "finish_reason": self.finished_reason,
            "finished_text": (f"Finished · {fmt_clock(self.budget_s)}" if self.finished_reason == "budget"
                              else f"Stopped: {self.finished_reason}") if self.finished_reason else None,
            "header": self.header, "arms": arms_out, "feed": feed, "n_events": self.n_events,
        }


def _natural_key(s: str) -> tuple[Any, ...]:
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s))


def fold(events: Sequence[Event], header: dict[str, Any], *, t0: float | None = None,
         until: float | None = None) -> DashboardState:
    """A DashboardState with every event up to race time ``until`` (absolute ts) applied."""
    st = DashboardState(header, t0=t0)
    for ev in events:
        if until is not None and ev.ts > until:
            break
        st.apply(ev)
    return st


# ---------------------------------------------------------------------------- streaming


RAW_SKIP_DEFAULT = frozenset({"metric"})
_NOTABLE = frozenset({"judge_verdict", "new_best", "correction", "planner_started", "planner_finished",
                      "race_finished", "experiment_finished", "llm_call",
                      # live Modal panel (demo races on per-arm Modal apps only; not in the snapshot)
                      "modal_stats"})


def _wants(ev_type: str, *, raw: bool, metrics: bool) -> bool:
    if ev_type == "metric":
        return raw and metrics
    return raw or ev_type in _NOTABLE


async def play_dashboard(events: Sequence[Event], offsets: Sequence[float], send: Callable[[str], Awaitable[Any]],
                         dash: DashboardState, *, payload: Callable[[Any], dict[str, Any]], speed: float,
                         tick_s: float, raw: bool = True, metrics: bool = False, start_s: float = 0.0,
                         clock: Callable[[], float] = time.monotonic,
                         sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> int:
    """Replay ``events`` at their ``offsets`` (``backend.replay.schedule``) through ``dash``,
    sending each wanted raw event, the derived ``slot`` events and a ``tick`` snapshot every
    ``tick_s`` s of wall time (race time interpolated at ``speed``). ``start_s``: fold the first
    ``start_s`` seconds of race time silently, then play from there. Returns messages sent."""
    t0 = dash.t0 if dash.t0 is not None else (events[0].ts if events else 0.0)
    i, n, sent = 0, len(events), 0
    if start_s > 0:
        while i < n and events[i].ts - t0 <= start_s:
            dash.apply(events[i])
            i += 1
    last_ts = events[i - 1].ts if i > 0 else t0
    if start_s > 0:
        last_ts = max(last_ts, t0 + start_s)
    # wall 0 = race time last_ts: event j plays at offsets[j] - base
    base = (offsets[i] - max(0.0, events[i].ts - last_ts) / speed) if i < n else 0.0
    last_off = base

    async def out(ev: Any) -> None:
        nonlocal sent
        await send(json.dumps(payload(ev)))
        sent += 1

    def race_now(w: float) -> float:
        t = last_ts + max(0.0, w - (last_off - base)) * speed
        return min(t, events[i].ts) if i < n else last_ts

    async def tick(at: float) -> None:
        snap = dash.snapshot(at)
        await out(TickEvent(race_id=dash.race_id, ts=snap["ts"], race_elapsed_s=snap["race_elapsed_s"],
                            snapshot=snap))

    start = clock()
    next_tick = 0.0
    while i < n:
        w = clock() - start
        if w >= next_tick:
            await tick(race_now(w))
            next_tick = max(next_tick + tick_s, w)
            continue
        due = offsets[i] - base
        if due <= w:
            ev = events[i]
            derived = dash.apply(ev)
            last_ts, last_off = ev.ts, offsets[i]
            i += 1
            if _wants(ev.type, raw=raw, metrics=metrics):
                await out(ev)
            for d in derived:
                await out(d)
            continue
        await sleep(max(0.0, min(due, next_tick) - w))
    await tick(dash.end_ts if dash.end_ts is not None else last_ts)
    return sent


# ---------------------------------------------------------------------------- routes


class _TimelineCache:
    def __init__(self, size: int = 8) -> None:
        self.size = size
        self._d: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any:
        v = self._d.get(key)
        if v is not None:
            self._d.move_to_end(key)
        return v

    def put(self, key: str, v: Any) -> None:
        self._d[key] = v
        self._d.move_to_end(key)
        while len(self._d) > self.size:
            self._d.popitem(last=False)


def dashboard_router(state: Any, catalog: Any, store: Any) -> APIRouter:
    """``state``: the app's ``AppState`` (db, bus, engines); ``catalog`` / ``store``: the
    replay catalog and session store (``backend.replay``). Read-only for every race."""
    from backend.replay import ReplayError, rebuild_events, wire_payload

    router = APIRouter()
    live: dict[str, DashboardState] = {}       # races watched in this process (fed by the bus)
    timelines = _TimelineCache()

    def _on_event(ev: Any) -> None:
        d = live.get(ev.race_id)
        if d is not None:
            d.apply(ev)

    state.bus.add_listener(_on_event)

    def _resolve(key: str) -> str:
        if state.db.get_race(key) is not None:
            return key
        try:
            return catalog.resolve(key)
        except KeyError as e:
            raise HTTPException(404, f"race {key} not found") from e

    def _is_live(race_id: str) -> bool:
        """Run by this process (running, or finished with its event history still in memory)."""
        eng = state.engines.get(race_id)
        return eng is not None and (eng.status in ("created", "running") or bool(state.bus.history(race_id)))

    def _live_header(race_id: str) -> dict[str, Any]:
        race = state.db.get_race(race_id) or {"id": race_id}
        return build_header(race, state.db.get_arms(race_id), mode="live")

    def _live_state(race_id: str) -> DashboardState:
        d = live.get(race_id)
        if d is None:
            d = DashboardState(_live_header(race_id))
            for ev in state.bus.history(race_id):
                d.apply(ev)
            live[race_id] = d
        elif not d.header_complete:
            d.set_header(_live_header(race_id))
        return d

    def _timeline(race_id: str) -> Any:
        tl = timelines.get(race_id)
        if tl is None:
            rows, _src = catalog.load(race_id, metrics=False)
            tl = rebuild_events(rows)
            timelines.put(race_id, tl)
        return tl

    @router.get("/api/races/{race_id}/snapshot")
    async def race_snapshot(race_id: str,
                            at_s: float | None = Query(None, ge=0, description="race time (s) for a finished race")
                            ) -> dict[str, Any]:
        """The dashboard snapshot. A race running in this process: now. A finished race (app DB
        or a replay source): at ``at_s`` seconds of race time, default its end. A unique id
        prefix works too."""
        rid = _resolve(race_id)
        if _is_live(rid):
            d = _live_state(rid)
            if at_s is not None and d.t0 is not None:     # an earlier moment: fold the history up to it
                until = d.t0 + at_s
                return fold(state.bus.history(rid), d.header, t0=d.t0, until=until).snapshot(until)
            if d.end_ts is None and d.last_ts is not None:
                return d.snapshot(time.time())
            return d.snapshot()
        try:
            tl = await asyncio.to_thread(_timeline, rid)
        except KeyError as e:
            raise HTTPException(404, f"race {race_id} not found") from e
        except ReplayError as e:
            raise HTTPException(409, f"race {rid} is neither running in this process nor finished: {e}") from e
        header = build_header(tl.race, tl.arms, mode="replay")
        until = tl.t0 + at_s if at_s is not None else None
        d = fold(tl.events, header, t0=tl.t0, until=until)
        return d.snapshot(until)

    @router.get("/api/dashboard/logged-races")
    async def logged_races() -> list[dict[str, Any]]:
        """Races of logged results (``data/frontend/showcase.json`` race sets) that this server can
        replay, with their header: pick R78 run 1, R82 run 1, ... for the dashboard."""
        try:
            sets = json.loads(SHOWCASE_PATH.read_text(encoding="utf-8")).get("sources", {}).get("race_sets", {})
        except (OSError, ValueError):
            sets = {}
        races = await asyncio.to_thread(catalog.list_races, replayable_only=True)
        out: list[dict[str, Any]] = []
        for ref, prefixes in sets.items():
            for i, p in enumerate(prefixes):
                r = next((x for x in races if str(x["id"]).startswith(str(p))), None)
                if r is None:
                    continue
                h = build_header(r, (), mode="replay")
                out.append({"ref": ref, "run": i + 1, "runs": len(prefixes), "race_id": r["id"],
                            "name": r.get("name"), "target": r.get("target"), "task": h["task"],
                            "budget_text": h["budget_text"], "goal_text": h["goal_text"],
                            "arms": [a["kind"] for a in h["arms"]], "source": r.get("source")})
        return out

    @router.websocket("/ws/races/{race_id}/dashboard")
    async def race_dashboard(websocket: WebSocket, race_id: str, tick_s: float = 1.0, raw: bool = True,
                             metrics: bool = False) -> None:
        """Live race: every raw event (metric points only with ``metrics=true``; with
        ``raw=false`` only the notable ones), the derived ``slot`` events and a ``tick`` with the
        full snapshot every ``tick_s`` s. A race that already finished gets one final tick."""
        await websocket.accept()
        tick_s = min(max(float(tick_s), 0.05), 60.0)
        race = state.db.get_race(race_id)
        if race is None:
            await websocket.close(code=4404)
            return
        try:
            if not _is_live(race_id) and race["status"] not in ("created", "running"):
                try:
                    tl = await asyncio.to_thread(_timeline, race_id)
                except (KeyError, ReplayError):
                    await websocket.close(code=4409)
                    return
                d = fold(tl.events, build_header(tl.race, tl.arms, mode="replay"), t0=tl.t0)
                snap = d.snapshot()
                await websocket.send_text(TickEvent(race_id=race_id, ts=snap["ts"],
                                                    race_elapsed_s=snap["race_elapsed_s"],
                                                    snapshot=snap).model_dump_json())
                await websocket.close()
                return
            dash = DashboardState(_live_header(race_id))
            q = state.bus.subscribe(race_id, replay=True)
            hub = getattr(state, "modal_stats", None)
            if hub is not None:
                # A live demo race on per-arm Modal apps: modal_stats events every ~2 s on this
                # stream. No-op (no Modal call) for every other race.
                hub.ensure(race_id)
        except WebSocketDisconnect:
            return
        try:
            async def send_tick() -> None:
                if not dash.header_complete:
                    dash.set_header(_live_header(race_id))
                at = None if dash.end_ts is not None or dash.last_ts is None else time.time()
                snap = dash.snapshot(at)
                await websocket.send_text(TickEvent(race_id=race_id, ts=snap["ts"],
                                                    race_elapsed_s=snap["race_elapsed_s"],
                                                    snapshot=snap).model_dump_json())

            async def handle(ev: Any) -> bool:
                derived = dash.apply(ev)
                if _wants(ev.type, raw=raw, metrics=metrics):
                    await websocket.send_text(ev.model_dump_json())
                for x in derived:
                    await websocket.send_text(x.model_dump_json())
                return ev.type == "race_finished"

            done = False
            while not q.empty() and not done:      # the race so far (bus history)
                done = await handle(q.get_nowait())
            await send_tick()
            next_tick = time.monotonic() + tick_s
            while not done:
                wait = next_tick - time.monotonic()
                if wait <= 0:
                    await send_tick()
                    next_tick = time.monotonic() + tick_s
                    continue
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=wait)
                except TimeoutError:
                    continue
                done = await handle(ev)
            await send_tick()
            await websocket.close()
        except WebSocketDisconnect:
            pass
        finally:
            state.bus.unsubscribe(race_id, q)

    @router.websocket("/ws/replays/{replay_id}/dashboard")
    async def replay_dashboard(websocket: WebSocket, replay_id: str, tick_s: float = 0.25, raw: bool = True,
                               metrics: bool = False, start_s: float = 0.0) -> None:
        """A replay session (POST /api/races/{id}/replay) through the dashboard reducer, at the
        session's speed: raw events + ``slot`` + ``tick`` (every ``tick_s`` s of wall time). Each
        payload carries ``is_replay: true``, ``replay_id`` and ``race_elapsed_s``."""
        await websocket.accept()
        s = store.get(replay_id)
        if s is None:
            await websocket.close(code=4404)
            return
        tick_s = min(max(float(tick_s), 0.05), 60.0)
        tl = s.timeline
        header = build_header(tl.race, tl.arms, mode="replay",
                              replay={"replay_id": s.id, "speed": s.speed, "max_gap_s": s.max_gap_s})
        dash = DashboardState(header, t0=tl.t0)
        try:
            await play_dashboard(tl.events, s.offsets, websocket.send_text, dash,
                                 payload=lambda ev: wire_payload(ev, replay_id=s.id, t0=tl.t0), speed=s.speed,
                                 tick_s=tick_s, raw=raw, metrics=metrics, start_s=max(0.0, float(start_s)))
            await websocket.close()
        except WebSocketDisconnect:
            pass

    return router
