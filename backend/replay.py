"""Race replay: rebuild a finished race's WebSocket event stream from storage (demo fallback).

The live stream on ``/ws/races/{id}`` is never stored as such, but every event is a view of
rows the engine does store. This module turns those rows (from the app's ``Database``, a
read-only SQLite file, or a JSONL export) back into the same ``backend.core.events`` models,
in timestamp order, and plays them at N x real time on ``/ws/replays/{replay_id}``.

How each event type is rebuilt (scheduler.py is the reference):

=====================  ===========================================================  =========
event                  source rows                                                  ts
=====================  ===========================================================  =========
gpu_state              ``gpu_intervals`` per slot: idle at the arm's t0, one event   exact
                       per interval start, a closing idle at the last interval end
experiment_started     ``experiments`` with ``started_at`` + ``slot_id``             exact
metric                 ``metrics`` (``t`` is seconds since the run started)          approx.
experiment_finished    ``experiments`` that ended inside the budget                 exact
new_best               ``experiments.is_best`` (+ bests later flagged fake), the     approx.
                       arm's best walked in order (``previous_best``)
llm_call               ``llm_calls`` (the stored ``LLMCallRecord``)                  exact
correction             ``corrections`` made before the race ended                    exact
race_finished          ``arms.summary`` + ``races.finish_reason``                    approx.
=====================  ===========================================================  =========

"approx." timestamps: a metric's live ``ts`` was its arrival time (estimated here as
``started_at + t``, clamped into the run); a new best was announced after the judge call
(its ``completed_at``) or once the agent loop was free; ``race_finished`` is put at the end.
Payloads are rebuilt field for field; ``tests/test_replay.py`` checks them against a live
event bus. Known gaps: a zero-length GPU interval is not stored, so an idle -> idle
duplicate the live stream sends on a deadline release is not reproduced; corrections made
after the race ended are not part of the race's stream (the live socket had closed).

On the wire every replayed event is the live JSON plus ``is_replay: true``, ``replay_id`` and
``race_elapsed_s`` (seconds since the race started). The event models ignore unknown keys,
so ``parse_event`` still parses it. Top-level ``is_replay`` (this is a replay of a recorded
race; demo.md section 4 "say so openly") is not ``llm_call.call.is_replay`` (an LLM reply
served from the record/replay cache). Stored rows are only ever read, never written.

CLI (prints a compact timeline, for checking)::

    python -m backend.replay <race_id|prefix> --speed 10 --db data/bench/bench.db
    python -m backend.replay --list --db data/bench/bench.db
    python -m backend.replay <race_id> --jsonl data/bench/dryrun/race-....jsonl --all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from backend.benchmark.metrics import is_counted, quality
from backend.core.events import (
    CorrectionEvent,
    Event,
    ExperimentFinishedEvent,
    ExperimentStartedEvent,
    GpuStateEvent,
    LLMCallEvent,
    MetricEvent,
    NewBestEvent,
    RaceFinishedEvent,
)
from backend.core.types import (
    ArmKind,
    ArmSummary,
    CorrectionLabel,
    CorrectionSource,
    ExperimentStatus,
    LLMCallRecord,
    SlotState,
    new_id,
)
from backend.db import Database, _decode_row

REPLAYABLE_STATUSES = frozenset({"finished", "cancelled", "error"})
MAX_SPEED = 1000.0
_END_ERRORS = ("stopped at end of budget",)  # set by the engine when a run ends after the budget
_DEADLINE_PREFIX = "deadline rule"           # buffered proposal dropped, never launched

# Tie-break for events with the same ts, in the order the engine emits them.
_RANK = {"metric": 0, "llm_call": 1, "gpu_idle": 2, "experiment_finished": 3, "new_best": 4,
         "correction": 5, "gpu_busy": 6, "experiment_started": 7, "race_finished": 9}

Rows = dict[str, list[dict[str, Any]]]


class ReplayError(ValueError):
    """The race cannot be replayed (not finished, never started, ...)."""


def is_replayable(race: dict[str, Any]) -> bool:
    return race.get("status") in REPLAYABLE_STATUSES and race.get("started_at") is not None


# =========================================================================== sources


class ReadOnlyDB:
    """A bench DB opened with ``mode=ro``: safe next to a live race writing the same file
    (WAL). Never creates tables or writes metadata (unlike ``Database``)."""

    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        self.path = str(p)
        self._conn = sqlite3.connect(f"{p.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ReadOnlyDB:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _all(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return [_decode_row(r) for r in self._conn.execute(sql, params).fetchall()]

    def get_race(self, race_id: str) -> dict[str, Any] | None:
        rows = self._all("SELECT * FROM races WHERE id = ?", (race_id,))
        return rows[0] if rows else None

    def list_races(self, limit: int = 500) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM races ORDER BY created_at DESC LIMIT ?", (limit,))

    def race_rows(self, race_id: str) -> Rows:
        """Same shape as ``Database.race_rows``."""
        race = self.get_race(race_id)
        if race is None:
            raise KeyError(race_id)
        exps = self._all("SELECT * FROM experiments WHERE race_id = ? ORDER BY proposed_at", (race_id,))
        metrics: list[dict[str, Any]] = []
        ids = [e["id"] for e in exps]
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            marks = ",".join("?" for _ in chunk)
            metrics.extend(self._all(f"SELECT * FROM metrics WHERE experiment_id IN ({marks}) "
                                     "ORDER BY experiment_id, step, t", tuple(chunk)))
        return {
            "races": [race],
            "arms": self._all("SELECT * FROM arms WHERE race_id = ? ORDER BY kind", (race_id,)),
            "experiments": exps,
            "metrics": metrics,
            "llm_calls": self._all("SELECT * FROM llm_calls WHERE race_id = ? ORDER BY completed_at", (race_id,)),
            "gpu_intervals": self._all('SELECT * FROM gpu_intervals WHERE race_id = ? ORDER BY arm_id, slot, "start"',
                                       (race_id,)),
            "corrections": self._all("SELECT * FROM corrections WHERE race_id = ? ORDER BY created_at", (race_id,)),
        }


def read_jsonl_rows(path: str | Path) -> Rows:
    """Rows of a JSONL export (``Database.export_race_jsonl``), grouped by table."""
    from backend.benchmark.export import read_export

    return read_export(path)


def _jsonl_race_row(path: Path) -> dict[str, Any] | None:
    """The race row of an export without reading the rest (races come first)."""
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    return obj["row"] if obj.get("table") == "races" else None
    except (OSError, ValueError, KeyError):
        return None
    return None


class ReplayCatalog:
    """Where replayable races come from: the app's own ``Database`` first, then extra
    sources opened read-only (``.db`` files, ``.jsonl`` exports, or folders of exports)."""

    def __init__(self, db: Database | None = None, sources: Sequence[str | Path] = ()) -> None:
        self.db = db
        self.sources = [Path(s) for s in sources if str(s).strip()]

    def _jsonl_files(self) -> list[Path]:
        out: list[Path] = []
        for s in self.sources:
            if s.is_dir():
                out.extend(sorted(s.glob("*.jsonl")))
            elif s.suffix == ".jsonl" and s.is_file():
                out.append(s)
        return out

    def _db_files(self) -> list[Path]:
        return [s for s in self.sources if s.is_file() and s.suffix != ".jsonl"]

    def list_races(self, *, replayable_only: bool = True) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []

        def add(race: dict[str, Any], source: str) -> None:
            if race["id"] in seen or (replayable_only and not is_replayable(race)):
                return
            seen.add(race["id"])
            out.append({**race, "replayable": is_replayable(race), "source": source})

        if self.db is not None:
            for r in self.db.list_races(limit=1000):
                add(r, "app")
        for p in self._db_files():
            try:
                with ReadOnlyDB(p) as ro:
                    for r in ro.list_races():
                        add(r, str(p))
            except (OSError, sqlite3.Error):
                continue
        for p in self._jsonl_files():
            r = _jsonl_race_row(p)
            if r is not None and "id" in r:
                add(r, str(p))
        out.sort(key=lambda r: r.get("created_at") or 0.0, reverse=True)
        return out

    def load(self, race_id: str) -> tuple[Rows, str]:
        """Every row of the race and where it came from. KeyError if no source has it."""
        if self.db is not None and self.db.get_race(race_id) is not None:
            return self.db.race_rows(race_id), "app"
        for p in self._db_files():
            try:
                with ReadOnlyDB(p) as ro:
                    if ro.get_race(race_id) is not None:
                        return ro.race_rows(race_id), str(p)
            except (OSError, sqlite3.Error):
                continue
        for p in self._jsonl_files():
            r = _jsonl_race_row(p)
            if r is not None and r.get("id") == race_id:
                return read_jsonl_rows(p), str(p)
        raise KeyError(race_id)


# =========================================================================== rebuild


@dataclass
class Timeline:
    """A race's rebuilt event stream plus what was left out."""

    race: dict[str, Any]
    events: list[Event]
    t0: float
    direction: str
    target_value: float | None
    dropped: Counter[str] = field(default_factory=Counter)  # e.g. late corrections

    @property
    def race_id(self) -> str:
        return str(self.race["id"])

    @property
    def duration_s(self) -> float:
        return (self.events[-1].ts - self.t0) if self.events else 0.0

    def counts(self) -> Counter[str]:
        return Counter(e.type for e in self.events)


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _target_spec(target: str) -> Any:
    try:
        from workers.targets.base import get_target

        return get_target(target).spec
    except Exception:  # unknown target or missing optional deps: fall back to the rows
        return None


def _direction(race: dict[str, Any], exps: list[dict[str, Any]], spec: Any) -> str:
    if spec is not None and getattr(spec, "direction", None) in ("min", "max"):
        return str(spec.direction)
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in exps:
        if e.get("is_best") and _f(e.get("score")) is not None:
            by_arm[e["arm_id"]].append(e)
    for bests in by_arm.values():
        bests.sort(key=lambda e: e.get("ended_at") or 0.0)
        if len(bests) >= 2:
            return "min" if bests[-1]["score"] < bests[0]["score"] else "max"
    return "max"


def _target_value(race: dict[str, Any], arms: list[dict[str, Any]], spec: Any) -> float | None:
    v = _f((race.get("config") or {}).get("target_value"))
    if v is not None:
        return v
    for a in arms:
        v = _f((a.get("report") or {}).get("target_value"))
        if v is not None:
            return v
    return _f(getattr(spec, "target_value", None))


def _better(a: float, b: float | None, direction: str) -> bool:
    return b is None or (a < b if direction == "min" else a > b)


def _reaches(q: float, target: float, direction: str) -> bool:
    return q <= target if direction == "min" else q >= target


def rebuild_events(rows: Rows, *, metric_stride: int = 1) -> Timeline:
    """Rebuild the race's WebSocket event stream from its stored rows, in ts order.

    ``metric_stride`` > 1 keeps every Nth metric point of each run (plus its last one), for
    a lighter replay; 1 (the default) keeps them all, like the live stream."""
    if not rows.get("races"):
        raise ReplayError("no race row")
    race = rows["races"][0]
    race_id = str(race["id"])
    if race.get("status") not in REPLAYABLE_STATUSES:
        raise ReplayError(f"race {race_id} is {race.get('status')!r}, not finished")
    arms = rows.get("arms") or []
    exps = rows.get("experiments") or []
    cfg = race.get("config") or {}
    kind_of = {a["id"]: ArmKind(a["kind"]) for a in arms}
    arm_t0 = {a["id"]: _f(a.get("started_at")) for a in arms}
    arm_end = {a["id"]: _f(a.get("ended_at")) for a in arms}
    t0 = _f(race.get("started_at")) or min((t for t in arm_t0.values() if t is not None), default=None)
    if t0 is None:
        raise ReplayError(f"race {race_id} never started")
    race_end = _f(race.get("ended_at")) or max((t for t in arm_end.values() if t is not None), default=t0)
    spec = _target_spec(str(race.get("target") or ""))
    direction = _direction(race, exps, spec)
    target_value = _target_value(race, arms, spec)

    out: list[tuple[float, int, int, Event]] = []

    def push(ev: Event, rank_key: str) -> None:
        out.append((ev.ts, _RANK[rank_key], len(out), ev))

    # --- gpu_state ------------------------------------------------------------------
    slots: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for iv in rows.get("gpu_intervals") or []:
        slots[(iv["arm_id"], iv["slot"])].append(iv)
    for (arm_id, slot), ivs in slots.items():
        kind = kind_of.get(arm_id)
        if kind is None:
            continue
        ivs.sort(key=lambda r: (r["start"], r["end"]))
        seq: list[tuple[float, SlotState, str | None]] = [(arm_t0.get(arm_id) or t0, SlotState.idle, None)]
        prev: dict[str, Any] | None = None
        for iv in ivs:
            st = SlotState(iv["state"])
            if prev is not None and prev["state"] == "busy" and st == SlotState.busy:
                seq.append((prev["end"], SlotState.idle, None))  # zero-length idle between two runs
            seq.append((iv["start"], st, iv.get("experiment_id") if st == SlotState.busy else None))
            prev = iv
        if prev is not None:
            seq.append((prev["end"], SlotState.idle, None))   # slot closed (end of budget / released)
        last: tuple[float, SlotState, str | None] | None = None
        for item in seq:
            if item == last:
                continue
            last = item
            ts, st, exp_id = item
            push(GpuStateEvent(race_id=race_id, arm=kind, slot_id=slot, state=st, experiment_id=exp_id, ts=ts),
                 "gpu_busy" if st == SlotState.busy else "gpu_idle")

    # --- experiments ------------------------------------------------------------------
    by_id = {e["id"]: e for e in exps}
    for e in exps:
        kind = kind_of.get(e.get("arm_id"))
        started = _f(e.get("started_at"))
        if kind is None or started is None or not e.get("slot_id"):
            continue
        push(ExperimentStartedEvent(race_id=race_id, arm=kind, experiment_id=e["id"], slot_id=e["slot_id"],
                                    config=e.get("config") or {}, hypothesis=e.get("hypothesis") or "",
                                    seed=int(e["seed"] if e.get("seed") is not None else cfg.get("seed", 0)),
                                    hardware=e.get("hardware") or "", ts=started), "experiment_started")
        ended = _f(e.get("ended_at"))
        status = e.get("status")
        err = e.get("error") or ""
        if ended is None or status in (ExperimentStatus.pending.value, ExperimentStatus.running.value):
            continue
        if status == ExperimentStatus.cancelled.value and (err in _END_ERRORS or err.startswith(_DEADLINE_PREFIX)):
            continue  # ended after the budget: the engine sends no event for it
        push(ExperimentFinishedEvent(race_id=race_id, arm=kind, experiment_id=e["id"], slot_id=e["slot_id"],
                                     status=ExperimentStatus(status), score=_f(e.get("score")),
                                     is_trap=bool(e.get("is_trap")), duration_s=ended - started,
                                     gpu_cost_usd=float(e.get("gpu_cost_usd") or 0.0), error=e.get("error"),
                                     ts=ended), "experiment_finished")

    # --- metrics ----------------------------------------------------------------------
    stride = max(1, int(metric_stride))
    per_exp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in rows.get("metrics") or []:
        per_exp[m["experiment_id"]].append(m)
    for exp_id, ms in per_exp.items():
        e = by_id.get(exp_id)
        kind = kind_of.get(e.get("arm_id")) if e else None
        started = _f(e.get("started_at")) if e else None
        if e is None or kind is None or started is None:
            continue
        hi = _f(e.get("ended_at")) or arm_end.get(e["arm_id"]) or race_end
        hi = max(hi, started + 1e-6)
        ms.sort(key=lambda m: (m.get("step") or 0, m.get("t") or 0.0))
        for i, m in enumerate(ms):
            if stride > 1 and i % stride and i != len(ms) - 1:
                continue
            value = _f(m.get("value"))
            if value is None:
                continue  # non-finite values are stored as NULL
            t = float(m.get("t") or 0.0)
            ts = min(max(started + t, started + 1e-6), hi)
            push(MetricEvent(race_id=race_id, arm=kind, experiment_id=exp_id, step=int(m.get("step") or 0), t=t,
                             name=str(m.get("name")), value=value, ts=ts), "metric")

    # --- llm calls --------------------------------------------------------------------
    dropped: Counter[str] = Counter()
    calls_by_arm: dict[str, list[tuple[float, float]]] = defaultdict(list)   # (start, end)
    judge_done: dict[str, float] = {}
    fields = set(LLMCallRecord.model_fields)
    for c in rows.get("llm_calls") or []:
        d = {k: v for k, v in c.items() if k in fields and v is not None}
        d["experiment_ids"] = list(c.get("experiment_ids") or [])
        try:
            rec = LLMCallRecord.model_validate(d)
        except ValueError:  # a row written by a newer schema (e.g. an unknown step type)
            dropped["llm_call_unparsed"] += 1
            continue
        ts = _f(c.get("completed_at")) or rec.created_at
        push(LLMCallEvent(race_id=race_id, call=rec, ts=ts), "llm_call")
        calls_by_arm[c.get("arm_id") or ""].append((ts - (rec.latency_ms or 0.0) / 1000.0, ts))
        if rec.step_type.value == "judge_improvement":
            for x in rec.experiment_ids:
                judge_done[x] = max(judge_done.get(x, ts), ts)

    # --- corrections ------------------------------------------------------------------
    corrections = sorted(rows.get("corrections") or [], key=lambda r: r.get("created_at") or 0.0)
    for c in corrections:
        ts = _f(c.get("created_at"))
        if ts is None or ts > race_end:
            dropped["correction_after_race"] += 1
            continue
        e = by_id.get(c["experiment_id"])
        arm_id = c.get("arm_id") or (e or {}).get("arm_id")
        push(CorrectionEvent(race_id=race_id, correction_id=c["id"], experiment_id=c["experiment_id"],
                             arm=kind_of.get(arm_id), source=CorrectionSource(c["source"]),
                             label=CorrectionLabel(c["label"]), reason=c.get("reason") or "", ts=ts), "correction")

    # --- new bests --------------------------------------------------------------------
    accept_unsure = bool(cfg.get("accept_unsure", False))
    for arm_id, kind in kind_of.items():
        mine = [e for e in exps if e.get("arm_id") == arm_id and e.get("status") == ExperimentStatus.succeeded.value
                and _f(e.get("score")) is not None and _f(e.get("ended_at")) is not None]
        flagged_ids = {c["experiment_id"] for c in corrections if c.get("label") == CorrectionLabel.fake_improvement.value}

        def verdict_ok(e: dict[str, Any]) -> bool:
            v = e.get("judge_verdict")
            return v != "invalid" and (v != "unsure" or accept_unsure)

        cands = [e for e in mine if e.get("is_best") or (e["id"] in flagged_ids and verdict_ok(e)
                                                          and (e.get("summary") or {}).get("claimed_new_best"))]
        spans = sorted(calls_by_arm.get(arm_id, []))

        def announce_ts(e: dict[str, Any], spans: list[tuple[float, float]] = spans) -> float:
            if e["id"] in judge_done:
                return judge_done[e["id"]]
            end = float(e["ended_at"])
            busy = [b for a, b in spans if a <= end < b]  # the agent loop was waiting on a call
            return max(busy) if busy else end

        processed_at = {e["id"]: announce_ts(e) for e in mine}  # when the agent loop saw the result
        steps: list[tuple[float, int, str, dict[str, Any]]] = [(processed_at[e["id"]], 0, "best", e) for e in cands]
        steps += [(float(c["created_at"]), 1, c["label"], by_id.get(c["experiment_id"]) or {"id": c["experiment_id"]})
                  for c in corrections if (_f(c.get("created_at")) or math.inf) <= race_end
                  and (c.get("arm_id") or (by_id.get(c["experiment_id"]) or {}).get("arm_id")) == arm_id]
        steps.sort(key=lambda s: (s[0], s[1]))
        flagged: set[str] = set()
        genuine: set[str] = set()
        best: dict[str, Any] | None = None
        last_ts = t0
        for ts, _, what, e in steps:
            if what == "best":
                score = float(e["score"])
                if not (e.get("is_best") or _better(score, _f(best["score"]) if best else None, direction)):
                    continue
                prev = _f(best["score"]) if best else None
                best = e
                ts = max(ts, last_ts)
                last_ts = ts
                q = quality(e)
                reached = bool(is_counted(e) and q is not None and target_value is not None
                               and _reaches(q, target_value, direction))
                push(NewBestEvent(race_id=race_id, arm=kind, experiment_id=e["id"], score=score, previous_best=prev,
                                  config=e.get("config") or {}, hypothesis=e.get("hypothesis") or "",
                                  elapsed_s=float(e["ended_at"]) - t0, reached_target=reached, ts=ts), "new_best")
                continue
            # A correction: the engine recomputes the arm's best over everything processed so far.
            if what == CorrectionLabel.fake_improvement.value:
                flagged.add(e["id"])
                genuine.discard(e["id"])
            else:
                genuine.add(e["id"])
                flagged.discard(e["id"])
            best = None
            for x in mine:
                if processed_at[x["id"]] > ts or x["id"] in flagged:
                    continue
                if x["id"] not in genuine and not verdict_ok(x):
                    continue
                if _better(float(x["score"]), _f(best["score"]) if best else None, direction):
                    best = x

    # --- race finished ----------------------------------------------------------------
    order = {k: i for i, k in enumerate(cfg.get("arms") or [])}
    sums = [ArmSummary.model_validate(a["summary"]) for a in sorted(arms, key=lambda a: order.get(a["kind"], 99))
            if a.get("summary")]
    status = race.get("status")
    reason = race.get("finish_reason") or ("budget" if status == "finished" else str(status))
    end_ts = max([race_end, *(x[0] for x in out)])
    push(RaceFinishedEvent(race_id=race_id, summaries=sums, reason=reason, ts=end_ts), "race_finished")

    out.sort(key=lambda x: (x[0], x[1], x[2]))
    return Timeline(race=race, events=[x[3] for x in out], t0=t0, direction=direction,
                    target_value=target_value, dropped=dropped)


def check_stream(events: Sequence[Event]) -> list[str]:
    """Consistency problems of an event stream (empty list = fine)."""
    problems: list[str] = []
    if not events:
        return ["no events"]
    race_ids = {e.race_id for e in events}
    if len(race_ids) != 1:
        problems.append(f"{len(race_ids)} race ids")
    if events[-1].type != "race_finished" or sum(e.type == "race_finished" for e in events) != 1:
        problems.append("race_finished must be the single last event")
    started: dict[str, str] = {}
    finished: set[str] = set()
    last_ts = -math.inf
    for i, e in enumerate(events):
        if e.ts < last_ts:
            problems.append(f"event {i} ({e.type}) goes back in time")
        last_ts = e.ts
        if isinstance(e, ExperimentStartedEvent):
            started[e.experiment_id] = e.slot_id
        elif isinstance(e, MetricEvent) and e.experiment_id not in started:
            problems.append(f"metric before experiment_started for {e.experiment_id}")
        elif isinstance(e, ExperimentFinishedEvent):
            if e.experiment_id not in started:
                problems.append(f"experiment_finished before experiment_started for {e.experiment_id}")
            finished.add(e.experiment_id)
        elif isinstance(e, NewBestEvent) and e.experiment_id not in finished:
            problems.append(f"new_best before experiment_finished for {e.experiment_id}")
        elif isinstance(e, GpuStateEvent) and e.state == SlotState.busy and not e.experiment_id:
            problems.append(f"busy gpu_state without experiment on {e.slot_id}")
    return problems[:50]


# =========================================================================== pacing


def schedule(events: Sequence[Event], speed: float, max_gap_s: float | None = None) -> list[float]:
    """Wall-clock offset (s from the replay start) of each event at ``speed`` x real time;
    ``max_gap_s`` caps any single wait between two events (skips dead time)."""
    if not speed > 0 or not math.isfinite(speed):
        raise ValueError("speed must be a positive number")
    offs: list[float] = []
    acc, prev = 0.0, None
    cap = max_gap_s if max_gap_s is not None and max_gap_s >= 0 else math.inf
    for e in events:
        if prev is not None:
            acc += min(max(0.0, e.ts - prev) / speed, cap)
        offs.append(acc)
        prev = e.ts
    return offs


def wire_payload(event: Event, *, replay_id: str, t0: float) -> dict[str, Any]:
    """The live JSON of the event plus the replay flag. The event itself is not changed."""
    d = event.model_dump(mode="json")
    d["is_replay"] = True
    d["replay_id"] = replay_id
    d["race_elapsed_s"] = round(event.ts - t0, 6)
    return d


async def play(events: Sequence[Event], offsets: Sequence[float], send: Callable[[str], Awaitable[Any]], *,
               replay_id: str, t0: float, clock: Callable[[], float] = time.monotonic,
               sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> int:
    """Send every event as JSON when its offset is due. Returns the number sent."""
    start = clock()
    sent = 0
    for ev, off in zip(events, offsets, strict=True):
        wait = off - (clock() - start)
        if wait > 0.002:
            await sleep(wait)
        await send(json.dumps(wire_payload(ev, replay_id=replay_id, t0=t0)))
        sent += 1
    return sent


# =========================================================================== sessions


@dataclass
class ReplaySession:
    id: str
    timeline: Timeline
    speed: float
    max_gap_s: float | None
    offsets: list[float]
    source: str
    created_at: float = field(default_factory=time.time)

    def info(self) -> dict[str, Any]:
        race = self.timeline.race
        return {
            "replay_id": self.id, "race_id": self.timeline.race_id, "race_name": race.get("name") or "",
            "target": race.get("target"), "source": self.source, "speed": self.speed, "max_gap_s": self.max_gap_s,
            "n_events": len(self.timeline.events), "counts": dict(self.timeline.counts()),
            "race_duration_s": round(self.timeline.duration_s, 3),
            "replay_duration_s": round(self.offsets[-1] if self.offsets else 0.0, 3),
            "dropped": dict(self.timeline.dropped), "ws": f"/ws/replays/{self.id}", "is_replay": True,
            "created_at": self.created_at,
        }


class ReplayStore:
    """In-memory replay sessions (bounded; oldest dropped first)."""

    def __init__(self, max_sessions: int = 16) -> None:
        self.max_sessions = max_sessions
        self._sessions: dict[str, ReplaySession] = {}

    def create(self, timeline: Timeline, *, speed: float, max_gap_s: float | None = None,
               source: str = "") -> ReplaySession:
        s = ReplaySession(id=new_id(), timeline=timeline, speed=speed, max_gap_s=max_gap_s,
                          offsets=schedule(timeline.events, speed, max_gap_s), source=source)
        self._sessions[s.id] = s
        while len(self._sessions) > self.max_sessions:
            self._sessions.pop(next(iter(self._sessions)))
        return s

    def get(self, replay_id: str) -> ReplaySession | None:
        return self._sessions.get(replay_id)

    def list(self) -> list[ReplaySession]:
        return list(self._sessions.values())


def replay_router(catalog: ReplayCatalog, store: ReplayStore) -> APIRouter:
    """REST + WebSocket routes for replays (mounted by ``backend.app.create_app``)."""
    router = APIRouter()

    def _build(race_id: str, metric_stride: int) -> tuple[Timeline, str]:
        rows, source = catalog.load(race_id)
        return rebuild_events(rows, metric_stride=metric_stride), source

    @router.post("/api/races/{race_id}/replay", status_code=201)
    async def create_replay(
        race_id: str,
        speed: float = Query(1.0, gt=0, le=MAX_SPEED, description="playback speed, N x real time"),
        max_gap_s: float | None = Query(None, ge=0, description="cap on any single wait (s, replay time)"),
        metric_stride: int = Query(1, ge=1, le=1000, description="keep every Nth metric point per run"),
    ) -> dict[str, Any]:
        try:
            timeline, source = await asyncio.to_thread(_build, race_id, metric_stride)
        except KeyError as e:
            raise HTTPException(404, f"race {race_id} not found") from e
        except ReplayError as e:
            raise HTTPException(409, str(e)) from e
        return store.create(timeline, speed=speed, max_gap_s=max_gap_s, source=source).info()

    @router.get("/api/replays")
    async def list_replays() -> list[dict[str, Any]]:
        return [s.info() for s in store.list()]

    @router.get("/api/replays/{replay_id}")
    async def get_replay(replay_id: str) -> dict[str, Any]:
        s = store.get(replay_id)
        if s is None:
            raise HTTPException(404, f"replay {replay_id} not found")
        return s.info()

    @router.websocket("/ws/replays/{replay_id}")
    async def replay_events(websocket: WebSocket, replay_id: str) -> None:
        await websocket.accept()
        s = store.get(replay_id)
        if s is None:
            await websocket.close(code=4404)
            return
        try:
            await play(s.timeline.events, s.offsets, websocket.send_text, replay_id=s.id, t0=s.timeline.t0)
            await websocket.close()
        except WebSocketDisconnect:
            pass

    return router


# =========================================================================== CLI


def _short(x: str | None) -> str:
    return (x or "-")[:8]


def _line(ev: Event, off: float, t0: float) -> str:
    head = f"[replay {off:8.2f}s | race {ev.ts - t0:8.2f}s]"
    arm = getattr(ev, "arm", None)
    arm_s = f"{(arm.value if arm else '-'):<6}"
    if isinstance(ev, GpuStateEvent):
        body = f"gpu      {ev.slot_id} {ev.state.value}" + (f" exp={_short(ev.experiment_id)}" if ev.experiment_id else "")
    elif isinstance(ev, ExperimentStartedEvent):
        body = f"started  exp={_short(ev.experiment_id)} slot={ev.slot_id} {ev.hypothesis[:50]!r}"
    elif isinstance(ev, MetricEvent):
        body = f"metric   exp={_short(ev.experiment_id)} step={ev.step} {ev.name}={ev.value:.4g}"
    elif isinstance(ev, ExperimentFinishedEvent):
        sc = "-" if ev.score is None else f"{ev.score:.4g}"
        body = (f"finished exp={_short(ev.experiment_id)} {ev.status.value} score={sc} {ev.duration_s:.1f}s "
                f"${ev.gpu_cost_usd:.4f}" + (" TRAP" if ev.is_trap else ""))
    elif isinstance(ev, NewBestEvent):
        prev = "-" if ev.previous_best is None else f"{ev.previous_best:.4g}"
        body = f"NEW BEST exp={_short(ev.experiment_id)} {prev} -> {ev.score:.4g}" + (" (goal)" if ev.reached_target else "")
    elif isinstance(ev, LLMCallEvent):
        c = ev.call
        arm_s = f"{c.arm.value:<6}"
        body = (f"llm      {c.step_type.value} {c.latency_ms / 1000:.1f}s ${c.cost_usd:.4f}"
                + ("" if c.valid else " INVALID") + (" retry" if c.is_retry else "") + (" fallback" if c.is_fallback else ""))
    elif isinstance(ev, CorrectionEvent):
        body = f"CORRECT  exp={_short(ev.experiment_id)} {ev.label.value} by {ev.source.value} {ev.reason[:40]!r}"
    elif isinstance(ev, RaceFinishedEvent):
        body = f"RACE FINISHED ({ev.reason})"
    else:  # pragma: no cover
        body = ev.type
    return f"{head} {arm_s} {body}"


def _resolve(races: Iterable[dict[str, Any]], key: str) -> str:
    ids = [r["id"] for r in races]
    if key in ids:
        return key
    hits = [i for i in ids if i.startswith(key)]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(f"race {key!r}: {'ambiguous prefix' if hits else 'not found'} ({len(hits)} matches)")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.replay",
                                 description="Rebuild a finished race's event stream and print a compact timeline.")
    ap.add_argument("race_id", nargs="?", help="race id or unique prefix (omit with --list)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--db", default=None, help="SQLite DB, opened read-only (default data/bench/bench.db)")
    src.add_argument("--jsonl", default=None, help="a race export (.jsonl) instead of a DB")
    ap.add_argument("--speed", type=float, default=10.0, help="playback speed for the replay clock (default 10)")
    ap.add_argument("--max-gap", type=float, default=None, help="cap any single wait at this many seconds")
    ap.add_argument("--metric-stride", type=int, default=1, help="keep every Nth metric point per run")
    ap.add_argument("--all", action="store_true", help="also print gpu_state and metric events")
    ap.add_argument("--limit", type=int, default=400, help="max timeline lines (0 = no limit)")
    ap.add_argument("--play", action="store_true", help="print in real time at --speed instead of all at once")
    ap.add_argument("--list", action="store_true", help="list races and whether they can be replayed")
    args = ap.parse_args(argv)

    source: Path = Path(args.jsonl or args.db or "data/bench/bench.db")
    catalog = ReplayCatalog(None, [source])
    if args.list or not args.race_id:
        for r in catalog.list_races(replayable_only=False):
            dur = (r["ended_at"] - r["started_at"]) if r.get("ended_at") and r.get("started_at") else None
            print(f"{r['id']}  {'replayable' if r['replayable'] else 'not-ready':<10} {r.get('status') or '':<9} "
                  f"{r.get('target') or '':<8} {r.get('mode') or '':<9} "
                  f"{'-' if dur is None else f'{dur:.0f}s':>6}  {r.get('name') or ''}")
        return 0
    race_id = _resolve(catalog.list_races(replayable_only=False), args.race_id)
    rows, where = catalog.load(race_id)
    try:
        tl = rebuild_events(rows, metric_stride=args.metric_stride)
    except ReplayError as e:
        print(f"cannot replay: {e}", file=sys.stderr)
        return 2
    offs = schedule(tl.events, args.speed, args.max_gap)
    race = tl.race
    print(f"race {tl.race_id}  {race.get('name') or ''}")
    print(f"  source {where} | target {race.get('target')} ({tl.direction}, goal {tl.target_value}) | "
          f"mode {race.get('mode')} | status {race.get('status')} ({race.get('finish_reason')})")
    print(f"  {len(tl.events)} events over {tl.duration_s:.1f}s race time -> {offs[-1]:.1f}s at {args.speed:g}x"
          + (f" (gaps capped at {args.max_gap:g}s)" if args.max_gap is not None else ""))
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(tl.counts().items())))
    if tl.dropped:
        print("  left out: " + ", ".join(f"{k}={v}" for k, v in tl.dropped.items()))
    print()
    shown = 0
    start = time.monotonic()
    for ev, off in zip(tl.events, offs, strict=True):
        if not args.all and ev.type in ("gpu_state", "metric"):
            continue
        if args.limit and shown >= args.limit and ev.type != "race_finished":
            continue
        if args.play:
            wait = off - (time.monotonic() - start)
            if wait > 0:
                time.sleep(wait)
        print(_line(ev, off, tl.t0))
        shown += 1
    hidden = len(tl.events) - shown
    if hidden:
        print(f"  ... {hidden} event(s) not shown" + ("" if args.all else " (gpu_state/metric hidden; --all shows them)"))
    last = tl.events[-1]
    if isinstance(last, RaceFinishedEvent):
        print()
        for s in last.summaries:
            best = "-" if s.best_score is None else f"{s.best_score:.4g}"
            ttt = "-" if s.time_to_target_s is None else f"{s.time_to_target_s:.0f}s"
            print(f"  {s.arm.value:<6} best={best:<10} finished={s.experiments_finished:<4} failed={s.experiments_failed:<4} "
                  f"waiting={s.gpu_idle_pct:5.1f}%  gpu=${s.gpu_cost_usd:.3f} llm=${s.llm_cost_usd:.3f} "
                  f"to-goal={ttt}")
    problems = check_stream(tl.events)
    print()
    print("checks: ok" if not problems else "checks: " + "; ".join(problems[:10]))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
