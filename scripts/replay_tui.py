"""Terminal race replay for the video: play a logged race back in a rich TUI.

Reads a finished race (from ``data/bench/bench.db`` opened read-only, or from a JSONL export),
rebuilds its event stream with ``backend.replay.rebuild_events`` (the same stream the web
dashboard gets on ``/ws/replays/{id}``) and draws it at N x real time:

- per arm, a grid of its compute slots: busy / waiting (paid, no experiment) / cancelled or
  failed (idle after a run was stopped) / off (slot closed);
- per arm: Model quality (best held-out score so far), AI bill (LLM $), GPUs waiting %
  (idle / paid slot time so far, the ``gpu_idle_pct`` definition), Experiments finished,
  Fakes caught and the goal line;
- a feed that flashes "Fake improvement caught" when an arm's checker rejects a fake (a
  ground-truth trap run whose final judge verdict is ``invalid``, shown when the judge call
  completes) or when a correction flags one.

Every number is read from the logged rows; nothing is estimated. Stored rows are only read.
No network, no Modal, no LLM calls.

Usage (PowerShell, from the repo root)::

    $env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"
    python -m uv run python scripts/replay_tui.py --list
    python -m uv run python scripts/replay_tui.py 4ca4ed27 --speed 30
    python -m uv run python scripts/replay_tui.py c1f4b1d9 --speed 30 --record
    python -m uv run python scripts/replay_tui.py 4ca4ed27 --at 450            # one frame, then exit
    python -m uv run python scripts/replay_tui.py 4ca4ed27 --at 900 --svg out.svg
    python -m uv run python scripts/replay_tui.py --jsonl data/bench/race-rl-seed0-4ca4ed27.jsonl --speed 30

``--record``: fixed 120x34 layout on the alternate screen, refreshed only by the replay loop
(no auto-refresh, no scrolling), for screen recording. Size the terminal to 120x34 first.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from backend.benchmark.export import parse_jsonl
from backend.benchmark.metrics import TRAP_CONTAMINATES_HELDOUT, is_counted, quality
from backend.core.events import (
    CorrectionEvent,
    Event,
    ExperimentFinishedEvent,
    GpuStateEvent,
    LLMCallEvent,
    RaceFinishedEvent,
)
from backend.core.types import CorrectionLabel, ExperimentStatus, SlotState
from backend.replay import ReadOnlyDB, Rows, Timeline, rebuild_events

RECORD_WIDTH, RECORD_HEIGHT = 120, 34
DEFAULT_DB = ROOT / "data" / "bench" / "bench.db"

ARM_LABELS: dict[str, str] = {
    "opus": "Standard Opus 5.5 agent",
    "opus_bm": "Busy Machines (Opus 5.5)",
    "mubit": "Busy Machines + Mubit",
    "optuna": "No-AI tuner (Optuna)",
    "fast": "Haiku 5.5 agent (ablation)",
    "lean": "Lean-prompt Opus (ablation)",
    "mock": "Mock agent (dev)",
}
TASK_LABELS: dict[str, str] = {
    "lcbench": "LCBench (tabular nets)",
    "rl": "RL agent",
    "cifar": "CIFAR-10 image model",
    "lm": "Language model",
}
LEAK_REASONS = {"contamination_or_eval_leak": "test data leaked in"}

# Slot display states and their colours (dark terminal).
BUSY, WAITING, CANCELLED, OFF = "busy", "waiting", "cancelled", "off"
SLOT_STYLE = {BUSY: "#3fb950", WAITING: "#d29922", CANCELLED: "#f85149", OFF: "#30363d"}
_STOPPED = frozenset({ExperimentStatus.cancelled, ExperimentStatus.failed, ExperimentStatus.timed_out})


# =========================================================================== loading


def _natural_key(s: str) -> tuple[Any, ...]:
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s))


def _rows_without_metrics(ro: ReadOnlyDB, race_id: str) -> Rows:
    """``ReadOnlyDB.race_rows`` minus the metrics table (the TUI draws no curves, and the RL
    races hold ~400k metric points). Same queries otherwise."""
    race = ro.get_race(race_id)
    if race is None:
        raise KeyError(race_id)
    q = ro._all  # read-only SELECTs
    return {
        "races": [race],
        "arms": q("SELECT * FROM arms WHERE race_id = ? ORDER BY kind", (race_id,)),
        "experiments": q("SELECT * FROM experiments WHERE race_id = ? ORDER BY proposed_at", (race_id,)),
        "metrics": [],
        "llm_calls": q("SELECT * FROM llm_calls WHERE race_id = ? ORDER BY completed_at", (race_id,)),
        "gpu_intervals": q('SELECT * FROM gpu_intervals WHERE race_id = ? ORDER BY arm_id, slot, "start"', (race_id,)),
        "corrections": q("SELECT * FROM corrections WHERE race_id = ? ORDER BY created_at", (race_id,)),
    }


def read_jsonl_without_metrics(path: str | Path) -> Rows:
    """A JSONL export (``backend.benchmark.export``) grouped by table, metric lines skipped."""
    with Path(path).open(encoding="utf-8") as f:
        lines = (ln for ln in f if '"table": "metrics"' not in ln[-48:])
        rows = parse_jsonl(lines)
    rows["metrics"] = []
    return rows


def resolve_race_id(ids: Iterable[str], key: str) -> str:
    ids = list(ids)
    if key in ids:
        return key
    hits = [i for i in ids if i.startswith(key)]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(f"race {key!r}: {'ambiguous prefix' if hits else 'not found'} ({len(hits)} matches)")


def load_rows(race: str | None, *, db: str | Path | None = None, jsonl: str | Path | None = None) -> tuple[Rows, str]:
    """Rows of one race (metrics left out) and where they came from."""
    if jsonl is not None:
        rows = read_jsonl_without_metrics(jsonl)
        rid = str((rows.get("races") or [{}])[0].get("id", ""))
        if race and not rid.startswith(race):
            raise SystemExit(f"{jsonl} holds race {rid}, not {race}")
        return rows, str(jsonl)
    if not race:
        raise SystemExit("give a race id (or prefix), or --jsonl")
    path = Path(db) if db is not None else DEFAULT_DB
    with ReadOnlyDB(path) as ro:
        rid = resolve_race_id((r["id"] for r in ro.list_races(limit=100_000)), race)
        return _rows_without_metrics(ro, rid), str(path)


# =========================================================================== model


@dataclass
class FeedItem:
    ts: float              # absolute race time
    at_s: float            # seconds since the race started
    kind: str              # "fake" | "goal"
    text: str


@dataclass
class ArmFrame:
    kind: str
    label: str
    slots: list[tuple[str, str]]          # (slot id, display state), in slot order
    ai_bill_usd: float
    waiting_pct: float
    experiments_finished: int
    best_quality: float | None
    quality_changed_recently: bool
    fakes_caught: int
    goal_at_s: float | None               # seconds since race start, None = not reached
    uses_ai: bool


@dataclass
class Frame:
    race_id: str
    task: str
    race_name: str
    arms_line: str
    speed: float
    elapsed_s: float
    duration_s: float
    target_value: float | None
    direction: str
    resource: str                         # "GPUs" or "Machines"
    arms: list[ArmFrame]
    note: str | None = field(default=None, kw_only=True)  # demo / dev / ablation label (race_note)
    feed: list[tuple[FeedItem, bool]]     # (item, still flashing)
    finished: bool


@dataclass
class _Exp:
    arm: str
    quality: float | None
    counted: bool
    is_trap: bool
    caught: bool                          # trap run the arm's checker rejected (final verdict invalid)
    score: float | None
    heldout: float | None
    llm_reason: str | None


@dataclass
class _ArmState:
    kind: str
    slot_busy: dict[str, bool] = field(default_factory=dict)
    slot_stopped: dict[str, bool] = field(default_factory=dict)
    ai_bill: float = 0.0
    finished: int = 0
    best: float | None = None
    best_changed_ts: float = -math.inf
    caught: int = 0
    goal_ts: float | None = None


def fmt_score(v: float | None) -> str:
    if v is None or not math.isfinite(v):
        return "-"
    a = abs(v)
    if a >= 100:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.3f}"
    return f"{v:.4f}"


def race_note(race: dict[str, Any]) -> str | None:
    """A label for races that are not benchmark results, shown on every frame: the demo label
    (``RaceConfig.demo_label``, e.g. demo-cifar-modal-5m), or the race's mode (dev / ablation)."""
    cfg = race.get("config") or {}
    if cfg.get("demo_label"):
        return str(cfg["demo_label"])
    mode = str(race.get("mode") or "")
    if mode == "ablation":
        return "ablation race, not a headline result"
    if mode and mode != "benchmark":
        return f"{mode} race, not a benchmark result"
    return None


def fmt_clock(s: float) -> str:
    s = max(0.0, s)
    return f"{int(s // 60):d}:{int(s % 60):02d}"


class ReplayModel:
    """Applies a rebuilt event stream up to a race time and builds plain ``Frame`` values.

    Pure and offline: no clock, no terminal. ``advance_to(t)`` only moves forward."""

    def __init__(self, timeline: Timeline, rows: Rows, *, speed: float = 30.0, flash_hold_s: float | None = None,
                 resource: str | None = None) -> None:
        if not speed > 0:
            raise ValueError("speed must be positive")
        self.tl = timeline
        self.events: list[Event] = timeline.events
        self.speed = float(speed)
        # How long (race seconds) a flash / a changed number stays highlighted: ~3 s on screen.
        self.flash_hold_s = float(flash_hold_s) if flash_hold_s is not None else 3.0 * self.speed
        race = timeline.race
        self.t0 = timeline.t0
        self.end = self.events[-1].ts if self.events else self.t0
        cfg = race.get("config") or {}
        arms = rows.get("arms") or []
        self.kind_of = {a["id"]: str(a["kind"]) for a in arms}
        order = [str(k) for k in (cfg.get("arms") or [])]
        kinds = sorted({str(a["kind"]) for a in arms}, key=lambda k: (order.index(k) if k in order else 99, k))
        self.kinds = kinds
        self.target = str(race.get("target") or "")

        hw = {str(a.get("hardware") or "") for a in arms} | {str(iv.get("hardware") or "")
                                                             for iv in rows.get("gpu_intervals") or []}
        hw.discard("")
        self.resource = resource or ("Machines" if hw and all("CPU" in h.upper() for h in hw) else "GPUs")

        # Slots per arm, and when each slot closes (its last interval end).
        slots: dict[str, set[str]] = defaultdict(set)
        self.slot_end: dict[tuple[str, str], float] = {}
        self.intervals: dict[str, list[tuple[float, float, bool]]] = defaultdict(list)
        for iv in rows.get("gpu_intervals") or []:
            k = self.kind_of.get(iv["arm_id"])
            if k is None:
                continue
            slots[k].add(str(iv["slot"]))
            s, e = float(iv["start"]), float(iv["end"])
            key = (k, str(iv["slot"]))
            self.slot_end[key] = max(self.slot_end.get(key, -math.inf), e)
            self.intervals[k].append((s, e, iv["state"] == "busy"))
        for e in rows.get("experiments") or []:
            k = self.kind_of.get(e.get("arm_id"))
            if k is not None and e.get("slot_id") and e.get("started_at") is not None:
                slots[k].add(str(e["slot_id"]))
        self.slots = {k: sorted(slots.get(k, set()), key=_natural_key) for k in kinds}
        for k in kinds:
            self.intervals[k].sort()

        # Per-experiment facts the event stream does not carry (held-out quality, verdicts).
        self.exps: dict[str, _Exp] = {}
        for e in rows.get("experiments") or []:
            k = self.kind_of.get(e.get("arm_id"))
            if k is None:
                continue
            trap = bool(e.get("is_trap"))
            pol = (e.get("summary") or {}).get("judge_policy") or {}
            h = (e.get("metrics") or {}).get("heldout_score")
            self.exps[e["id"]] = _Exp(arm=k, quality=quality(e), counted=is_counted(e), is_trap=trap,
                                      caught=trap and e.get("judge_verdict") == "invalid",
                                      score=e.get("score"), heldout=float(h) if h is not None else None,
                                      llm_reason=pol.get("llm_reason"))
        self.judged: set[str] = set()
        for ev in self.events:
            if isinstance(ev, LLMCallEvent) and ev.call.step_type.value == "judge_improvement":
                self.judged.update(ev.call.experiment_ids)
        self.uses_ai = {k: k != "optuna" for k in kinds}
        for ev in self.events:
            if isinstance(ev, LLMCallEvent):
                self.uses_ai[ev.call.arm.value] = True

        self.state = {k: _ArmState(kind=k, slot_busy={s: False for s in self.slots[k]},
                                   slot_stopped={s: False for s in self.slots[k]}) for k in kinds}
        self.feed: list[FeedItem] = []
        self._announced: set[str] = set()
        self.pos = 0
        self.now = self.t0

    # ------------------------------------------------------------------ events

    def _announce_fake(self, exp_id: str, ts: float, by: str = "its checker") -> None:
        if exp_id in self._announced:
            return
        x = self.exps.get(exp_id)
        if x is None:
            return
        self._announced.add(exp_id)
        st = self.state.get(x.arm)
        if st is not None:
            st.caught += 1
        why = LEAK_REASONS.get(x.llm_reason or "", "not a real gain")
        real = ""
        if (x.heldout is not None and x.score is not None and self.target not in TRAP_CONTAMINATES_HELDOUT
                and abs(x.heldout - float(x.score)) > 1e-9):
            real = f", real held-out {fmt_score(x.heldout)}"
        who = "" if by == "its checker" else f", flagged by {by}"
        self.feed.append(FeedItem(ts=ts, at_s=ts - self.t0, kind="fake", text=(
            f"Fake improvement caught  {ARM_LABELS.get(x.arm, x.arm)}: claimed {fmt_score(x.score)}{real} "
            f"({why}{who})")))

    def _apply(self, ev: Event) -> None:
        if isinstance(ev, GpuStateEvent):
            st = self.state.get(ev.arm.value)
            if st is not None:
                busy = ev.state == SlotState.busy
                st.slot_busy[ev.slot_id] = busy
                if busy:
                    st.slot_stopped[ev.slot_id] = False
        elif isinstance(ev, ExperimentFinishedEvent):
            st = self.state.get(ev.arm.value)
            if st is None:
                return
            if ev.status in _STOPPED:
                st.slot_stopped[ev.slot_id] = True
            if ev.status == ExperimentStatus.succeeded:
                st.finished += 1
                x = self.exps.get(ev.experiment_id)
                if x is not None and x.counted and x.quality is not None:
                    better = st.best is None or (x.quality < st.best if self.tl.direction == "min"
                                                 else x.quality > st.best)
                    if better:
                        st.best = x.quality
                        st.best_changed_ts = ev.ts
                    tv = self.tl.target_value
                    if st.goal_ts is None and tv is not None and (
                            x.quality <= tv if self.tl.direction == "min" else x.quality >= tv):
                        st.goal_ts = ev.ts
                        self.feed.append(FeedItem(ts=ev.ts, at_s=ev.ts - self.t0, kind="goal", text=(
                            f"Goal reached: {ARM_LABELS.get(st.kind, st.kind)} hit {fmt_score(x.quality)} "
                            f"at {fmt_clock(ev.ts - self.t0)}")))
                if x is not None and x.caught and ev.experiment_id not in self.judged:
                    self._announce_fake(ev.experiment_id, ev.ts)
        elif isinstance(ev, LLMCallEvent):
            st = self.state.get(ev.call.arm.value)
            if st is not None:
                st.ai_bill += float(ev.call.cost_usd or 0.0)
            if ev.call.step_type.value == "judge_improvement":
                for x_id in ev.call.experiment_ids:
                    x = self.exps.get(x_id)
                    if x is not None and x.caught:
                        self._announce_fake(x_id, ev.ts)
        elif isinstance(ev, CorrectionEvent):
            if ev.label == CorrectionLabel.fake_improvement:
                self._announce_fake(ev.experiment_id, ev.ts,
                                    by="the Sapling checker" if ev.source.value == "sapling" else "a human")
        elif isinstance(ev, RaceFinishedEvent):
            pass

    def advance_to(self, t_abs: float) -> None:
        t_abs = min(max(t_abs, self.now), self.end)
        while self.pos < len(self.events) and self.events[self.pos].ts <= t_abs:
            self._apply(self.events[self.pos])
            self.pos += 1
        self.now = t_abs

    def advance_by_elapsed(self, elapsed_s: float) -> None:
        self.advance_to(self.t0 + elapsed_s)

    @property
    def done(self) -> bool:
        return self.pos >= len(self.events)

    # ------------------------------------------------------------------ derived numbers

    def waiting_pct(self, kind: str, t_abs: float | None = None) -> float:
        """Idle / paid slot seconds up to ``t_abs`` x 100 (``gpu_idle_pct`` up to now)."""
        t = self.now if t_abs is None else t_abs
        busy = idle = 0.0
        for s, e, is_busy in self.intervals.get(kind, ()):
            if s >= t:
                break
            d = min(e, t) - s
            if d <= 0:
                continue
            if is_busy:
                busy += d
            else:
                idle += d
        paid = busy + idle
        return idle / paid * 100.0 if paid > 0 else 0.0

    def slot_state(self, kind: str, slot: str) -> str:
        st = self.state[kind]
        end = self.slot_end.get((kind, slot))
        if end is not None and self.now >= end and not st.slot_busy.get(slot):
            return OFF
        if st.slot_busy.get(slot):
            return BUSY
        return CANCELLED if st.slot_stopped.get(slot) else WAITING

    def frame(self) -> Frame:
        race = self.tl.race
        arms: list[ArmFrame] = []
        for k in self.kinds:
            st = self.state[k]
            arms.append(ArmFrame(
                kind=k, label=ARM_LABELS.get(k, k), slots=[(s, self.slot_state(k, s)) for s in self.slots[k]],
                ai_bill_usd=st.ai_bill, waiting_pct=self.waiting_pct(k), experiments_finished=st.finished,
                best_quality=st.best, quality_changed_recently=(self.now - st.best_changed_ts) <= self.flash_hold_s,
                fakes_caught=st.caught, goal_at_s=None if st.goal_ts is None else st.goal_ts - self.t0,
                uses_ai=self.uses_ai.get(k, True)))
        feed = [(f, (self.now - f.ts) <= self.flash_hold_s) for f in self.feed[-4:]]
        return Frame(
            race_id=self.tl.race_id, task=TASK_LABELS.get(self.target, self.target or "?"),
            race_name=str(race.get("name") or ""),
            arms_line=" vs ".join(ARM_LABELS.get(k, k) for k in self.kinds),
            speed=self.speed, elapsed_s=self.now - self.t0, duration_s=self.end - self.t0,
            target_value=self.tl.target_value, direction=self.tl.direction, resource=self.resource,
            arms=arms, feed=feed, finished=self.done, note=race_note(race))


def build_model(rows: Rows, *, speed: float = 30.0, resource: str | None = None) -> ReplayModel:
    rows = {**rows, "metrics": []}  # metric curves are not drawn; keeps the stream small
    return ReplayModel(rebuild_events(rows), rows, speed=speed, resource=resource)


# =========================================================================== rendering


def grid_lines(slots: Sequence[tuple[str, str]], width: int, max_lines: int) -> list[Text]:
    """The slot grid as text lines: the biggest cells that fit (5x3, 3x2, 2x1, else 1x1)."""
    n = len(slots)
    if n == 0:
        return [Text("no slots", style="dim")]
    for cw, ch in ((5, 3), (3, 2), (2, 1), (1, 1)):
        per_row = max(1, (width + 1) // (cw + 1))
        rows = math.ceil(n / per_row)
        lines_needed = rows * ch + (rows - 1 if ch > 1 else 0)
        if lines_needed <= max_lines or (cw, ch) == (1, 1):
            break
    out: list[Text] = []
    for r in range(rows):
        chunk = slots[r * per_row:(r + 1) * per_row]
        line = Text()
        for i, (_, state) in enumerate(chunk):
            if i:
                line.append(" ")
            line.append("█" * cw, style=SLOT_STYLE[state])
        for _ in range(ch):
            out.append(line.copy())
        if ch > 1 and r < rows - 1:
            out.append(Text(""))
    return out[:max_lines]


def _arm_panel(a: ArmFrame, f: Frame, width: int, height: int) -> Panel:
    inner_w = max(8, width - 4)
    stats = Table.grid(padding=(0, 1), expand=True)
    stats.add_column(style="#8b949e", no_wrap=True)
    stats.add_column(justify="right", no_wrap=True)
    q_style = "bold #ffffff on #1f6feb" if a.quality_changed_recently else "bold #ffffff"
    stats.add_row("Model quality", Text(fmt_score(a.best_quality), style=q_style))
    bill = f"${a.ai_bill_usd:,.2f}" if a.uses_ai else "$0.00 (no AI)"
    stats.add_row("AI bill", Text(bill, style="bold #e3b341"))
    w_style = "bold #f85149" if a.waiting_pct >= 20 else ("bold #d29922" if a.waiting_pct >= 5 else "bold #3fb950")
    stats.add_row(f"{f.resource} waiting", Text(f"{a.waiting_pct:4.1f}%", style=w_style))
    stats.add_row("Experiments finished", Text(f"{a.experiments_finished:,}", style="bold"))
    if a.uses_ai or a.fakes_caught:
        stats.add_row("Fakes caught", Text(str(a.fakes_caught), style="bold #f85149" if a.fakes_caught else "dim"))
    else:
        stats.add_row("Fakes caught", Text("no checker", style="dim"))
    if f.target_value is not None:
        goal = (Text(f"reached {fmt_clock(a.goal_at_s)}", style="bold #3fb950") if a.goal_at_s is not None
                else Text("not yet", style="dim"))
        stats.add_row(f"Goal {fmt_score(f.target_value)}", goal)
    stat_rows = 6 if f.target_value is not None else 5
    grid_max = max(1, height - 2 - stat_rows - 1)
    body = Group(*grid_lines(a.slots, inner_w, grid_max), Text(""), stats)
    border = "#1f6feb" if a.kind in ("opus_bm", "mubit") else "#30363d"
    return Panel(body, title=Text(a.label, style="bold"), title_align="left", border_style=border,
                 padding=(0, 1), height=height)


def _header(f: Frame, width: int) -> Group:
    top = Text()
    top.append(" BUSY MACHINES ", style="bold #0d1117 on #58a6ff")
    top.append(f"  replay of a logged race, x{f.speed:g}", style="bold #e3b341")
    clock = f"race time {fmt_clock(f.elapsed_s)} / {fmt_clock(f.duration_s)}"
    top.append(" " * max(1, width - len(top.plain) - len(clock) - 1))
    top.append(clock, style="bold")
    l2 = Text()
    if f.note:  # first on the line, so truncation never hides it
        l2.append(f"[{f.note}]  ", style="bold #f85149")
    l2.append("Race ", style="#8b949e")
    l2.append(f.race_id[:8], style="bold")
    l2.append("   Task ", style="#8b949e")
    l2.append(f.task, style="bold")
    if f.target_value is not None:
        l2.append("   Goal ", style="#8b949e")
        l2.append(f"{'<=' if f.direction == 'min' else '>='} {fmt_score(f.target_value)}", style="bold")
    l2.append("   " + f.race_name, style="#8b949e")
    l2.truncate(width, overflow="ellipsis")
    l3 = Text()
    l3.append("Arms ", style="#8b949e")
    l3.append(f.arms_line, style="bold")
    l3.append("   (same machines, time budget and seeds)", style="#8b949e")
    l3.truncate(width, overflow="ellipsis")
    for line in (top, l2, l3):  # one line each, whatever the console (never wrap into the next line)
        line.no_wrap, line.overflow = True, "ellipsis"
    frac = 0.0 if f.duration_s <= 0 else min(1.0, f.elapsed_s / f.duration_s)
    bar_w = max(10, width - 2)
    filled = round(frac * bar_w)
    bar = Text("━" * filled, style="#58a6ff")
    bar.append("━" * (bar_w - filled), style="#30363d")
    return Group(top, l2, l3, bar)


def _feed(f: Frame, height: int) -> Panel:
    lines: list[Text] = []
    for item, hot in f.feed:
        t = Text()
        t.append(f" {fmt_clock(item.at_s):>5} ", style="#8b949e")
        if item.kind == "fake":
            t.append(" ! ", style="bold #ffffff on #da3633" if hot else "#f85149")
            t.append(" " + item.text, style="bold #ffffff" if hot else "#8b949e")
        else:
            t.append(" * ", style="bold #0d1117 on #3fb950" if hot else "#3fb950")
            t.append(" " + item.text, style="bold #ffffff" if hot else "#8b949e")
        t.no_wrap = True
        t.overflow = "ellipsis"
        lines.append(t)
    if not lines:
        lines.append(Text(" No fake improvements so far. Each arm's claimed new bests are checked as they arrive.",
                          style="#8b949e"))
    if f.finished:
        lines.append(Text(" Race finished. Every number on this screen is from the logged race.", style="bold #58a6ff"))
    return Panel(Group(*lines[-(height - 2):]), title=Text("Checker", style="bold"), title_align="left",
                 border_style="#30363d", height=height, padding=(0, 0))


def _legend(f: Frame) -> Text:
    t = Text(" ")
    for state, label in ((BUSY, "busy"), (WAITING, "waiting (paid, idle)"), (CANCELLED, "cancelled / failed"),
                         (OFF, "off")):
        t.append("██", style=SLOT_STYLE[state])
        t.append(f" {label}   ", style="#8b949e")
    t.append("Model quality = best held-out score so far", style="#8b949e")
    return t


def render(f: Frame, width: int = RECORD_WIDTH, height: int = RECORD_HEIGHT) -> RenderableType:
    """The whole screen for one frame, sized to ``width`` x ``height``."""
    header_h, feed_h, legend_h = 4, 6, 1
    arms_h = max(10, height - header_h - feed_h - legend_h)
    layout = Layout()
    layout.split_column(Layout(name="header", size=header_h), Layout(name="arms", size=arms_h),
                        Layout(name="feed", size=feed_h), Layout(name="legend", size=legend_h))
    layout["header"].update(_header(f, width))
    n = max(1, len(f.arms))
    col_w = width // n
    cols = [Layout(_arm_panel(a, f, col_w, arms_h), name=f"arm{i}") for i, a in enumerate(f.arms)]
    if cols:
        layout["arms"].split_row(*cols)
    layout["feed"].update(_feed(f, feed_h))
    layout["legend"].update(_legend(f))
    return layout


def frame_text(f: Frame, width: int = RECORD_WIDTH, height: int = RECORD_HEIGHT) -> str:
    """The frame as plain text (no colour), exactly ``height`` lines: for tests and checks."""
    con = Console(width=width, height=height, record=True, color_system=None, file=_NullFile(),
                  legacy_windows=False)
    con.print(render(f, width, height))
    return con.export_text()


class _NullFile:
    def write(self, _s: str) -> int:
        return 0

    def flush(self) -> None:
        pass


# =========================================================================== playback


def play(model: ReplayModel, console: Console, *, fps: float, screen: bool, hold_start: float,
         hold_end: float, width: int | None, height: int | None) -> None:
    def view() -> RenderableType:
        w = width or console.size.width
        h = height or console.size.height
        return render(model.frame(), w, h)

    with Live(view(), console=console, screen=screen, auto_refresh=False, transient=False,
              redirect_stdout=False, redirect_stderr=False) as live:
        live.refresh()
        time.sleep(max(0.0, hold_start))
        start = time.monotonic()
        step = 1.0 / max(1.0, fps)
        while True:
            model.advance_by_elapsed((time.monotonic() - start) * model.speed)
            live.update(view(), refresh=True)
            if model.done:
                break
            time.sleep(step)
        time.sleep(max(0.0, hold_end))


def _print_list(db: Path) -> None:
    with ReadOnlyDB(db) as ro:
        for r in ro.list_races(limit=100_000):
            if r.get("status") not in ("finished", "cancelled", "error") or r.get("started_at") is None:
                continue
            print(f"{r['id']}  {r.get('target') or '':<8} {r.get('mode') or '':<9} {r.get('name') or ''}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scripts/replay_tui.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("race_id", nargs="?", help="race id or unique prefix (from --db)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--db", default=None, help=f"SQLite DB, opened read-only (default {DEFAULT_DB.relative_to(ROOT)})")
    src.add_argument("--jsonl", default=None, help="a race export (.jsonl) instead of the DB")
    ap.add_argument("--speed", type=float, default=30.0, help="playback speed, N x real time (default 30)")
    ap.add_argument("--fps", type=float, default=10.0, help="frames per second (default 10)")
    ap.add_argument("--record", action="store_true",
                    help="record-friendly: fixed 120x34 layout on the alternate screen, no flicker")
    ap.add_argument("--hold-start", type=float, default=None, help="seconds to show the first frame (record: 2)")
    ap.add_argument("--hold-end", type=float, default=None, help="seconds to hold the last frame (record: 5)")
    ap.add_argument("--at", type=float, default=None, help="print one frame at this race second and exit")
    ap.add_argument("--svg", default=None, help="with --at: also save that frame as an SVG")
    ap.add_argument("--resource-label", choices=("auto", "GPUs", "Machines"), default="auto",
                    help="label for slots: auto = 'Machines' when every slot is a CPU container, else 'GPUs'")
    ap.add_argument("--list", action="store_true", help="list replayable races in --db")
    args = ap.parse_args(argv)
    if not args.speed > 0 or not math.isfinite(args.speed):
        ap.error("--speed must be a positive number")

    if args.list:
        _print_list(Path(args.db) if args.db else DEFAULT_DB)
        return 0
    rows, _source = load_rows(args.race_id, db=args.db, jsonl=args.jsonl)
    model = build_model(rows, speed=args.speed,
                        resource=None if args.resource_label == "auto" else args.resource_label)

    if args.at is not None:
        model.advance_by_elapsed(args.at)
        # legacy_windows=False: these consoles always write VT codes; rich's legacy Windows path
        # otherwise wraps full-width lines (the header's second line pushed the arms line off).
        con = Console(width=RECORD_WIDTH, height=RECORD_HEIGHT, record=bool(args.svg), force_terminal=True,
                      color_system="truecolor", legacy_windows=False)
        con.print(render(model.frame()))
        if args.svg:
            con.save_svg(args.svg, title=f"Busy Machines replay {model.tl.race_id[:8]}")
        return 0

    if args.record:
        console = Console(width=RECORD_WIDTH, height=RECORD_HEIGHT, force_terminal=True, color_system="truecolor",
                          legacy_windows=False)
        size: tuple[int | None, int | None] = (RECORD_WIDTH, RECORD_HEIGHT)
    else:
        console = Console()
        size = (None, None)
    hold_start = args.hold_start if args.hold_start is not None else (2.0 if args.record else 0.0)
    hold_end = args.hold_end if args.hold_end is not None else (5.0 if args.record else 2.0)
    try:
        play(model, console, fps=args.fps, screen=True, hold_start=hold_start, hold_end=hold_end,
             width=size[0], height=size[1])
    except KeyboardInterrupt:
        pass
    console.print(render(model.frame(), *(s or d for s, d in zip(size, (console.size.width, console.size.height)))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
