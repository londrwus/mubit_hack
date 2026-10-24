"""SQLite storage (stdlib sqlite3, WAL) for races, arms, experiments, metrics, LLM calls,
GPU intervals, corrections and adapter versions (architecture.md "Data model").

Design notes:
- One connection guarded by a ``threading.Lock`` (``check_same_thread=False``), so the
  FastAPI event loop and any worker thread can share it. Autocommit; WAL +
  ``synchronous=NORMAL`` keeps per-row writes cheap.
- Metrics are buffered and written in batches (``flush_metrics``); everything else is
  written immediately.
- **Benchmark integrity flag.** ``races.is_dev`` is 1 for any race that uses the mock LLM
  or record/replay. ``arms.llm_mode`` says which LLM an arm used (mubit | mock | replay |
  none). ``insert_llm_call`` refuses an ``is_replay`` call on a benchmark race
  (``BenchmarkIntegrityError``), and ``assert_benchmark_clean`` lets the analysis reject
  dev races outright.
- ``export_race_jsonl`` dumps every row of a race as ``{"table": ..., "row": {...}}`` lines.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from backend.core.types import LLMCallRecord, new_id, now

SCHEMA_VERSION = 3  # 2: llm_calls.mubit_project; 3: llm_calls.cost_source (added in place by _migrate)

# Columns added after a table was first shipped: (table, column, type). ``CREATE TABLE IF NOT
# EXISTS`` leaves an older file's table as it was, so ``_migrate`` adds any that are missing.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("llm_calls", "mubit_project", "TEXT"),
    ("llm_calls", "cost_source", "TEXT"),
)


class BenchmarkIntegrityError(RuntimeError):
    """A benchmark race touched mock/replay data (or vice versa)."""


_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS races (
    id TEXT PRIMARY KEY,
    name TEXT,
    target TEXT NOT NULL,
    budget_s REAL NOT NULL,
    gpus_per_arm INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    mode TEXT NOT NULL DEFAULT 'dev',          -- dev | benchmark
    is_dev INTEGER NOT NULL DEFAULT 1,         -- 1 => mock/replay allowed; benchmarks reject
    llm_mode TEXT,                             -- mubit | mock | replay
    runner TEXT,
    config_json TEXT,                          -- full RaceConfig
    fairness_json TEXT,                        -- the per-arm settings, proven identical
    status TEXT NOT NULL DEFAULT 'created',    -- created | running | finished | error | cancelled
    finish_reason TEXT,
    created_at REAL,
    started_at REAL,
    ended_at REAL
);
CREATE TABLE IF NOT EXISTS arms (
    id TEXT PRIMARY KEY,
    race_id TEXT NOT NULL REFERENCES races(id),
    kind TEXT NOT NULL,                        -- opus | mubit | optuna | mock
    model TEXT,
    llm_mode TEXT,                             -- mubit | mock | replay | none
    is_dev INTEGER NOT NULL DEFAULT 1,
    n_slots INTEGER,
    hardware TEXT,
    price_per_s REAL,
    status TEXT,
    stop_reason TEXT,
    started_at REAL,
    ended_at REAL,
    summary_json TEXT,                         -- ArmSummary
    report_json TEXT                           -- extra numbers (idle $, traps, retries ...)
);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    arm_id TEXT NOT NULL REFERENCES arms(id),
    race_id TEXT,
    slot_id TEXT,
    config_json TEXT,
    hypothesis TEXT,
    status TEXT,
    runner TEXT,
    run_ref TEXT,                              -- Modal call ID / local PID
    seed INTEGER,
    hardware TEXT,
    proposed_at REAL,
    started_at REAL,
    ended_at REAL,
    score REAL,
    is_best INTEGER NOT NULL DEFAULT 0,        -- accepted as a new best when it finished
    is_trap INTEGER NOT NULL DEFAULT 0,        -- ground truth: contamination trap enabled
    is_baseline INTEGER NOT NULL DEFAULT 0,    -- the starting-code config run by the scheduler
    judge_verdict TEXT,                        -- valid | invalid | unsure | NULL (not judged)
    judge_reason TEXT,
    diagnosis_json TEXT,
    summary_json TEXT,                         -- deterministic summarize_result output
    error TEXT,
    metrics_json TEXT,                         -- RunResult.metrics
    logs_tail_json TEXT,
    llm_call_id TEXT,                          -- the propose call that produced it
    mubit_request_id TEXT,
    gpu_cost_usd REAL
);
CREATE INDEX IF NOT EXISTS ix_experiments_arm ON experiments(arm_id);
CREATE INDEX IF NOT EXISTS ix_experiments_race ON experiments(race_id);
CREATE TABLE IF NOT EXISTS metrics (
    experiment_id TEXT NOT NULL,
    step INTEGER,
    t REAL,
    name TEXT,
    value REAL
);
CREATE INDEX IF NOT EXISTS ix_metrics_exp ON metrics(experiment_id);
CREATE TABLE IF NOT EXISTS llm_calls (
    id TEXT PRIMARY KEY,
    arm_id TEXT,
    race_id TEXT,
    arm TEXT,
    session_id TEXT,
    step_type TEXT,
    mubit_request_id TEXT,
    model TEXT,
    latency_ms REAL,
    input_tokens INTEGER,
    cache_creation_tokens INTEGER,
    cache_read_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    valid INTEGER,
    is_retry INTEGER,
    is_fallback INTEGER,
    is_replay INTEGER,
    recorded INTEGER,
    stop_reason TEXT,
    error TEXT,
    experiment_ids_json TEXT,
    created_at REAL,
    completed_at REAL,                         -- when the scheduler received the reply
    mubit_project TEXT,                        -- D38: Mubit project (A | B) that served it; NULL = mock/replay
    cost_source TEXT                           -- estimate@list | estimate@opus_list (mubit arm); NULL = mock/old
);
CREATE INDEX IF NOT EXISTS ix_llm_calls_arm ON llm_calls(arm_id);
CREATE INDEX IF NOT EXISTS ix_llm_calls_race ON llm_calls(race_id);
CREATE TABLE IF NOT EXISTS gpu_intervals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    arm_id TEXT NOT NULL,
    race_id TEXT,
    slot TEXT NOT NULL,
    hardware TEXT,
    state TEXT NOT NULL,                       -- busy | idle
    "start" REAL NOT NULL,
    "end" REAL NOT NULL,
    experiment_id TEXT,
    price_per_s REAL
);
CREATE INDEX IF NOT EXISTS ix_gpu_intervals_arm ON gpu_intervals(arm_id);
CREATE TABLE IF NOT EXISTS corrections (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    race_id TEXT,
    arm_id TEXT,
    source TEXT NOT NULL,                      -- human | sapling
    label TEXT NOT NULL,                       -- fake_improvement | genuine
    reason TEXT,
    created_at REAL,
    sent_to_mubit_at REAL
);
CREATE TABLE IF NOT EXISTS adapter_versions (
    id TEXT PRIMARY KEY,
    step_type TEXT NOT NULL,
    version TEXT NOT NULL,
    status TEXT,                               -- candidate | shadow | live | retired ...
    trained_at REAL,
    notes TEXT,
    created_at REAL,
    UNIQUE(step_type, version)
);
"""

# Columns stored as JSON text, decoded on read: column -> output key.
_JSON_COLS: dict[str, str] = {
    "config_json": "config",
    "fairness_json": "fairness",
    "summary_json": "summary",
    "report_json": "report",
    "diagnosis_json": "diagnosis",
    "metrics_json": "metrics",
    "logs_tail_json": "logs_tail",
    "experiment_ids_json": "experiment_ids",
}
_BOOL_COLS = {
    "is_dev", "is_best", "is_trap", "is_baseline", "valid", "is_retry", "is_fallback",
    "is_replay", "recorded",
}

TABLES_IN_EXPORT_ORDER = (
    "races", "arms", "experiments", "metrics", "llm_calls", "gpu_intervals", "corrections",
)


def _clean(v: Any) -> Any:
    """Make a value JSON-safe (non-finite floats -> None), recursively."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def _dumps(v: Any) -> str | None:
    if v is None:
        return None
    return json.dumps(_clean(v), sort_keys=True, default=str)


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in row.keys():
        v = row[k]
        if k in _JSON_COLS:
            out[_JSON_COLS[k]] = json.loads(v) if v is not None else None
        elif k in _BOOL_COLS and v is not None:
            out[k] = bool(v)
        else:
            out[k] = v
    return out


class Database:
    """Thin synchronous SQLite wrapper. All public methods are thread-safe."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._metric_buf: list[tuple[str, int, float, str, float]] = []
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=OFF")
            self._conn.executescript(_DDL)
            self._migrate()
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def _migrate(self) -> None:
        """Backwards-compatible: add columns missing from a file made by an older schema
        (nullable, no default, so old rows read as NULL). Never drops or rewrites anything."""
        for table, column, typ in _ADDED_COLUMNS:
            have = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
            if column not in have:
                self._conn.execute(f'ALTER TABLE {table} ADD COLUMN "{column}" {typ}')

    # ------------------------------------------------------------------ basics

    def close(self) -> None:
        with self._lock:
            self.flush_metrics()
            self._conn.close()

    def _exec(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def _all(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [_decode_row(r) for r in self._conn.execute(sql, params).fetchall()]

    def _one(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn.execute(sql, params).fetchone()
        return _decode_row(r) if r is not None else None

    def _insert(self, table: str, row: dict[str, Any], replace: bool = False) -> None:
        cols = list(row)
        quoted = ", ".join(f'"{c}"' for c in cols)
        marks = ", ".join("?" for _ in cols)
        verb = "INSERT OR REPLACE" if replace else "INSERT"
        self._exec(f"{verb} INTO {table} ({quoted}) VALUES ({marks})", tuple(row[c] for c in cols))

    def _update(self, table: str, key: str, key_value: Any, fields: dict[str, Any]) -> None:
        if not fields:
            return
        sets = ", ".join(f'"{c}" = ?' for c in fields)
        self._exec(f"UPDATE {table} SET {sets} WHERE {key} = ?", (*fields.values(), key_value))

    # ------------------------------------------------------------------ races

    def insert_race(
        self,
        *,
        race_id: str,
        target: str,
        budget_s: float,
        gpus_per_arm: int,
        seed: int,
        config: dict[str, Any],
        name: str = "",
        mode: str = "dev",
        llm_mode: str | None = None,
        runner: str | None = None,
    ) -> None:
        is_dev = mode != "benchmark"
        if not is_dev and llm_mode in ("mock", "replay"):
            raise BenchmarkIntegrityError(f"benchmark race cannot use llm_mode={llm_mode!r}")
        self._insert(
            "races",
            {
                "id": race_id,
                "name": name,
                "target": target,
                "budget_s": float(budget_s),
                "gpus_per_arm": int(gpus_per_arm),
                "seed": int(seed),
                "mode": mode,
                "is_dev": int(is_dev),
                "llm_mode": llm_mode,
                "runner": runner,
                "config_json": _dumps(config),
                "status": "created",
                "created_at": now(),
            },
        )

    def update_race(self, race_id: str, **fields: Any) -> None:
        if "fairness" in fields:
            fields["fairness_json"] = _dumps(fields.pop("fairness"))
        if "config" in fields:
            fields["config_json"] = _dumps(fields.pop("config"))
        self._update("races", "id", race_id, fields)

    def get_race(self, race_id: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM races WHERE id = ?", (race_id,))

    def list_races(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM races ORDER BY created_at DESC LIMIT ?", (limit,))

    def is_benchmark_race(self, race_id: str) -> bool:
        r = self._one("SELECT is_dev FROM races WHERE id = ?", (race_id,))
        return r is not None and not r["is_dev"]

    # ------------------------------------------------------------------ arms

    def insert_arm(
        self,
        *,
        arm_id: str,
        race_id: str,
        kind: str,
        model: str | None,
        llm_mode: str,
        n_slots: int,
        hardware: str,
        price_per_s: float,
    ) -> None:
        bench = self.is_benchmark_race(race_id)
        if bench and (llm_mode in ("mock", "replay") or kind == "mock"):
            raise BenchmarkIntegrityError(f"benchmark arm cannot be kind={kind} llm_mode={llm_mode}")
        self._insert(
            "arms",
            {
                "id": arm_id,
                "race_id": race_id,
                "kind": kind,
                "model": model,
                "llm_mode": llm_mode,
                "is_dev": int(not bench),
                "n_slots": n_slots,
                "hardware": hardware,
                "price_per_s": price_per_s,
                "status": "created",
            },
        )

    def update_arm(self, arm_id: str, **fields: Any) -> None:
        if "summary" in fields:
            fields["summary_json"] = _dumps(fields.pop("summary"))
        if "report" in fields:
            fields["report_json"] = _dumps(fields.pop("report"))
        self._update("arms", "id", arm_id, fields)

    def get_arms(self, race_id: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM arms WHERE race_id = ? ORDER BY kind", (race_id,))

    # ------------------------------------------------------------------ experiments

    def upsert_experiment(self, row: dict[str, Any]) -> None:
        """Insert or replace a full experiment row. JSON-ish keys (config, diagnosis,
        summary, metrics, logs_tail) are encoded here."""
        r = dict(row)
        for col, key in _JSON_COLS.items():
            if key in r:
                r[col] = _dumps(r.pop(key))
        for b in _BOOL_COLS:
            if b in r and r[b] is not None:
                r[b] = int(bool(r[b]))
        if isinstance(r.get("score"), float) and not math.isfinite(r["score"]):
            r["score"] = None
        self._insert("experiments", r, replace=True)

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM experiments WHERE id = ?", (experiment_id,))

    def list_experiments(self, *, race_id: str | None = None, arm_id: str | None = None) -> list[dict[str, Any]]:
        if arm_id is not None:
            return self._all("SELECT * FROM experiments WHERE arm_id = ? ORDER BY proposed_at", (arm_id,))
        return self._all("SELECT * FROM experiments WHERE race_id = ? ORDER BY proposed_at", (race_id,))

    # ------------------------------------------------------------------ metrics

    def add_metric(self, experiment_id: str, step: int, t: float, name: str, value: float) -> None:
        v = float(value)
        with self._lock:
            self._metric_buf.append((experiment_id, int(step), float(t), name, v if math.isfinite(v) else None))  # type: ignore[arg-type]
            if len(self._metric_buf) >= 500:
                self.flush_metrics()

    def flush_metrics(self) -> None:
        with self._lock:
            if not self._metric_buf:
                return
            buf, self._metric_buf = self._metric_buf, []
            self._conn.executemany(
                "INSERT INTO metrics (experiment_id, step, t, name, value) VALUES (?, ?, ?, ?, ?)", buf
            )

    def list_metrics(self, experiment_id: str) -> list[dict[str, Any]]:
        self.flush_metrics()
        return self._all("SELECT * FROM metrics WHERE experiment_id = ? ORDER BY step, t", (experiment_id,))

    # ------------------------------------------------------------------ llm calls

    def insert_llm_call(self, record: LLMCallRecord, *, race_id: str, completed_at: float | None = None) -> None:
        if record.is_replay and self.is_benchmark_race(race_id):
            raise BenchmarkIntegrityError("record/replay LLM call in a benchmark race")
        d = record.model_dump(mode="json")
        d["experiment_ids_json"] = _dumps(d.pop("experiment_ids"))
        for b in ("valid", "is_retry", "is_fallback", "is_replay", "recorded"):
            d[b] = int(bool(d[b]))
        d["race_id"] = race_id
        d["completed_at"] = completed_at if completed_at is not None else now()
        self._insert("llm_calls", d, replace=True)

    def list_llm_calls(self, *, race_id: str | None = None, arm_id: str | None = None) -> list[dict[str, Any]]:
        if arm_id is not None:
            return self._all("SELECT * FROM llm_calls WHERE arm_id = ? ORDER BY completed_at", (arm_id,))
        return self._all("SELECT * FROM llm_calls WHERE race_id = ? ORDER BY completed_at", (race_id,))

    # ------------------------------------------------------------------ gpu intervals

    def insert_gpu_interval(
        self,
        *,
        arm_id: str,
        race_id: str,
        slot: str,
        hardware: str,
        state: str,
        start: float,
        end: float,
        experiment_id: str | None,
        price_per_s: float,
    ) -> None:
        self._insert(
            "gpu_intervals",
            {
                "arm_id": arm_id,
                "race_id": race_id,
                "slot": slot,
                "hardware": hardware,
                "state": state,
                "start": start,
                "end": end,
                "experiment_id": experiment_id,
                "price_per_s": price_per_s,
            },
        )

    def list_gpu_intervals(self, *, race_id: str | None = None, arm_id: str | None = None) -> list[dict[str, Any]]:
        if arm_id is not None:
            return self._all('SELECT * FROM gpu_intervals WHERE arm_id = ? ORDER BY slot, "start"', (arm_id,))
        return self._all('SELECT * FROM gpu_intervals WHERE race_id = ? ORDER BY arm_id, slot, "start"', (race_id,))

    # ------------------------------------------------------------------ corrections

    def insert_correction(
        self,
        *,
        experiment_id: str,
        source: str,
        label: str,
        reason: str = "",
        correction_id: str | None = None,
        race_id: str | None = None,
        arm_id: str | None = None,
    ) -> str:
        cid = correction_id or new_id()
        self._insert(
            "corrections",
            {
                "id": cid,
                "experiment_id": experiment_id,
                "race_id": race_id,
                "arm_id": arm_id,
                "source": source,
                "label": label,
                "reason": reason,
                "created_at": now(),
                "sent_to_mubit_at": None,
            },
        )
        return cid

    def mark_correction_sent(self, correction_id: str, ts: float | None = None) -> None:
        self._update("corrections", "id", correction_id, {"sent_to_mubit_at": ts or now()})

    def list_corrections(self, *, race_id: str | None = None, experiment_id: str | None = None) -> list[dict[str, Any]]:
        if experiment_id is not None:
            return self._all("SELECT * FROM corrections WHERE experiment_id = ? ORDER BY created_at", (experiment_id,))
        if race_id is not None:
            return self._all("SELECT * FROM corrections WHERE race_id = ? ORDER BY created_at", (race_id,))
        return self._all("SELECT * FROM corrections ORDER BY created_at")

    # ------------------------------------------------------------------ adapter versions

    def upsert_adapter_version(
        self,
        *,
        step_type: str,
        version: str,
        status: str = "candidate",
        trained_at: float | None = None,
        notes: str = "",
    ) -> str:
        existing = self._one(
            "SELECT id FROM adapter_versions WHERE step_type = ? AND version = ?", (step_type, version)
        )
        aid = existing["id"] if existing else new_id()
        self._insert(
            "adapter_versions",
            {
                "id": aid,
                "step_type": step_type,
                "version": version,
                "status": status,
                "trained_at": trained_at,
                "notes": notes,
                "created_at": now(),
            },
            replace=True,
        )
        return aid

    def list_adapter_versions(self) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM adapter_versions ORDER BY step_type, created_at")

    # ------------------------------------------------------------------ integrity / export

    def assert_benchmark_clean(self, race_id: str) -> None:
        """Raise BenchmarkIntegrityError unless the race is a benchmark race with no mock or
        replay anywhere (race flag, arm flags, LLM calls)."""
        race = self.get_race(race_id)
        if race is None:
            raise KeyError(race_id)
        if race["is_dev"]:
            raise BenchmarkIntegrityError(f"race {race_id} is a dev race (mock/replay allowed)")
        for arm in self.get_arms(race_id):
            if arm["is_dev"] or arm["llm_mode"] in ("mock", "replay") or arm["kind"] == "mock":
                raise BenchmarkIntegrityError(f"arm {arm['id']} uses {arm['llm_mode']}")
        bad = self._one("SELECT COUNT(*) AS n FROM llm_calls WHERE race_id = ? AND is_replay = 1", (race_id,))
        if bad and bad["n"]:
            raise BenchmarkIntegrityError(f"race {race_id} has {bad['n']} replayed LLM calls")

    def race_rows(self, race_id: str) -> dict[str, list[dict[str, Any]]]:
        """Every row belonging to a race, grouped by table."""
        self.flush_metrics()
        race = self.get_race(race_id)
        if race is None:
            raise KeyError(race_id)
        exps = self.list_experiments(race_id=race_id)
        metrics: list[dict[str, Any]] = []
        if exps:
            ids = [e["id"] for e in exps]
            for i in range(0, len(ids), 500):
                chunk = ids[i : i + 500]
                marks = ",".join("?" for _ in chunk)
                metrics.extend(
                    self._all(
                        f"SELECT * FROM metrics WHERE experiment_id IN ({marks}) ORDER BY experiment_id, step, t",
                        tuple(chunk),
                    )
                )
        return {
            "races": [race],
            "arms": self.get_arms(race_id),
            "experiments": exps,
            "metrics": metrics,
            "llm_calls": self.list_llm_calls(race_id=race_id),
            "gpu_intervals": self.list_gpu_intervals(race_id=race_id),
            "corrections": self.list_corrections(race_id=race_id),
        }

    def export_race_jsonl(self, race_id: str) -> Iterator[str]:
        """Yield one JSON line per row: ``{"table": name, "row": {...}}`` (newline included).
        Tables in a fixed order (races first), so a reader can rebuild the race."""
        rows = self.race_rows(race_id)
        for table in TABLES_IN_EXPORT_ORDER:
            for row in rows[table]:
                yield json.dumps({"table": table, "row": _clean(row)}, sort_keys=True, default=str) + "\n"
