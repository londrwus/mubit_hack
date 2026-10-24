"""night/bench analysis: per-arm results over seeds, a markdown table for
docs/results_log.md, and plain-language PNG charts.

Reads races from the SQLite DB written by ``backend/db.py`` (opened read-only) and/or from
JSONL exports (``backend/benchmark/export.py``), groups them by target, and reports per arm
the mean +/- std over seeds of:

- Cost to reach the goal ($-to-target = GPU $ + LLM $ up to the first run that reaches the
  target threshold), Time to reach the goal (time-to-target)
- GPUs waiting % (idle / paid slot time)
- Experiments finished
- Model quality at the budget: the best held-out score, raw and normalised 0-100
  (0 = the starting config's held-out score, 100 = the best known result; demo.md)
- LLM $, p50 LLM latency per call (and per step type), malformed-reply rate, fallback rate

The per-race numbers come from ``backend.benchmark.metrics.summarize_arm``, the same code the
race engine uses, so the table matches the dashboard and anyone holding the JSONL export can
recompute it.

Excluded by default, and listed in the output: dev races (mock LLM / record-replay), races
with a mock arm or replayed calls, collection-probe races (name or id mentions probe /
collection), races that did not finish, LLM calls from probe sessions or with no arm.
``--allow-dev`` keeps dev races for a rehearsal; every table and chart is then stamped
"DEV / MOCK: not a benchmark result" and chart files get a ``-dev`` suffix.

Usage:
    python -m uv run python scripts/analyze_races.py --db data/busy_machines.db
    python -m uv run python scripts/analyze_races.py --db data/busy_machines.db --race ID1 --race ID2
    python -m uv run python scripts/analyze_races.py --jsonl race_s0.jsonl race_s1.jsonl race_s2.jsonl
    options: --target-value lm=2.1  --best-known lm=1.95  --start-score lm=2.6
             --charts-dir data/charts  --no-charts  --theme dark|light
             --out results.md  --json results.json  --allow-dev  --include-unfinished

No LLM calls, no compute: this only reads rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.benchmark.export import read_export  # noqa: E402
from backend.benchmark.metrics import is_counted, quality, summarize_arm  # noqa: E402

Rows = dict[str, list[dict[str, Any]]]

ARM_ORDER = ("mubit", "opus", "optuna", "mock")
ARM_LABEL = {
    "mubit": "Busy Machines (Mubit)",
    "opus": "Opus 5.5",
    "optuna": "Optuna (no AI)",
    "mock": "Mock LLM (dev)",
    # Ablation only (rl-planner-ablation), never a headline arm.
    "fast": "Haiku 5.5 in the same loop (smaller faster model, not a Mubit adapter)",
}
# Chart labels (demo.md "Dashboard labels"): the Opus arm is the "Standard AI assistant".
ARM_CHART_LABEL = {
    "mubit": "Busy Machines\n(powered by Mubit)",
    "opus": "Standard AI assistant\n(Opus 5.5)",
    "optuna": "Automatic search\n(no AI, Optuna)",
    "mock": "Mock assistant\n(dev only)",
    "fast": "Smaller faster model, same loop\n(not a Mubit adapter; ablation)",
}
TARGET_LABEL = {"lm": "Language model", "rl": "Game-playing AI", "cifar": "Image recognition"}

PROBE_MARKERS = ("probe", "collection")
DEV_LLM_MODES = ("mock", "replay")
DEV_STAMP = "DEV / MOCK: not a benchmark result"

# Tables a race needs (metrics are not used by any headline number).
RACE_TABLES = ("arms", "experiments", "llm_calls", "gpu_intervals", "corrections")


# --------------------------------------------------------------------------- loading


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    """Same decoding as backend/db.py: ``*_json`` columns -> parsed value under the bare name."""
    out: dict[str, Any] = {}
    for k in row.keys():
        v = row[k]
        if k.endswith("_json"):
            out[k[: -len("_json")]] = json.loads(v) if v is not None else None
        else:
            out[k] = v
    return out


def load_db(path: str | Path, race_ids: Sequence[str] | None = None) -> list[Rows]:
    """Every requested race (all races if ``race_ids`` is None) from the SQLite DB, opened
    read-only so a running server is never disturbed."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    conn = sqlite3.connect(f"file:{p.resolve().as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if race_ids:
            marks = ",".join("?" for _ in race_ids)
            races = conn.execute(f"SELECT * FROM races WHERE id IN ({marks}) ORDER BY created_at", tuple(race_ids))
        else:
            races = conn.execute("SELECT * FROM races ORDER BY created_at")
        race_rows = [_decode(r) for r in races.fetchall()]
        missing = set(race_ids or ()) - {r["id"] for r in race_rows}
        if missing:
            raise KeyError(f"race id(s) not in {p}: {sorted(missing)}")
        out: list[Rows] = []
        for race in race_rows:
            rows: Rows = {"races": [race]}
            for t in RACE_TABLES:
                rows[t] = [_decode(r) for r in conn.execute(f"SELECT * FROM {t} WHERE race_id = ?", (race["id"],))]
            out.append(rows)
        return out
    finally:
        conn.close()


def split_export(rows: Rows) -> list[Rows]:
    """A JSONL export normally holds one race; split it by race id in case it holds more."""
    out: list[Rows] = []
    for race in rows.get("races", []):
        rid = race["id"]
        r: Rows = {"races": [race]}
        arm_ids = {a["id"] for a in rows.get("arms", []) if a.get("race_id") == rid}
        for t in RACE_TABLES:
            r[t] = [x for x in rows.get(t, []) if x.get("race_id") == rid or (x.get("race_id") is None and x.get("arm_id") in arm_ids)]
        out.append(r)
    return out


def load_jsonl(paths: Iterable[str | Path]) -> list[Rows]:
    out: list[Rows] = []
    for p in paths:
        out.extend(split_export(read_export(p)))
    return out


# --------------------------------------------------------------------------- screening


@dataclass
class Exclusions:
    races: list[tuple[str, str]] = field(default_factory=list)  # (race id, reason)
    llm_calls: dict[str, int] = field(default_factory=dict)       # reason -> rows dropped
    experiments: dict[str, int] = field(default_factory=dict)
    duplicates: list[str] = field(default_factory=list)

    def add_rows(self, table: str, reason: str, n: int) -> None:
        if n:
            d = self.llm_calls if table == "llm_calls" else self.experiments
            d[reason] = d.get(reason, 0) + n


def _truthy(v: Any) -> bool:
    return bool(v) and v not in ("0", "false", "False")


def is_probe_race(race: Mapping[str, Any]) -> bool:
    text = f"{race.get('id', '')} {race.get('name') or ''}".lower()
    cfg = race.get("config") or {}
    purpose = str(cfg.get("purpose", "")).lower() if isinstance(cfg, dict) else ""
    return any(m in text for m in PROBE_MARKERS) or purpose in ("probe", "collection")


def dev_reasons(rows: Rows) -> list[str]:
    race = rows["races"][0]
    why: list[str] = []
    if _truthy(race.get("is_dev", True)) or race.get("mode") != "benchmark":
        why.append(f"dev race (mode={race.get('mode')})")
    if race.get("llm_mode") in DEV_LLM_MODES:
        why.append(f"llm_mode={race.get('llm_mode')}")
    for a in rows.get("arms", []):
        if a.get("kind") == "mock" or a.get("llm_mode") in DEV_LLM_MODES:
            why.append(f"arm {a.get('kind')} llm_mode={a.get('llm_mode')}")
        elif _truthy(a.get("is_dev")) and not _truthy(race.get("is_dev", True)):
            why.append(f"arm {a.get('kind')} flagged dev")
    n_replay = sum(1 for c in rows.get("llm_calls", []) if _truthy(c.get("is_replay")))
    if n_replay:
        why.append(f"{n_replay} replayed LLM calls")
    return why


def _is_probe_call(c: Mapping[str, Any]) -> bool:
    sid = str(c.get("session_id") or "").lower()
    return any(m in sid for m in PROBE_MARKERS)


def screen(
    races: list[Rows], *, allow_dev: bool = False, include_unfinished: bool = False
) -> tuple[list[tuple[Rows, bool]], Exclusions]:
    """Return ([(rows, is_dev)], exclusions). Rows are copies with probe / orphan rows removed."""
    ex = Exclusions()
    kept: list[tuple[Rows, bool]] = []
    seen: set[str] = set()
    for rows in races:
        race = rows["races"][0]
        rid = race["id"]
        if rid in seen:
            ex.duplicates.append(rid)
            continue
        seen.add(rid)
        if is_probe_race(race):
            ex.races.append((rid, "collection/probe race"))
            continue
        dev = dev_reasons(rows)
        if dev and not allow_dev:
            ex.races.append((rid, "; ".join(dict.fromkeys(dev))))
            continue
        if race.get("status") != "finished" and not include_unfinished:
            ex.races.append((rid, f"not finished (status={race.get('status')}, reason={race.get('finish_reason')})"))
            continue
        if race.get("started_at") is None:
            ex.races.append((rid, "never started"))
            continue
        if not rows.get("arms"):
            ex.races.append((rid, "no arms"))
            continue
        arm_ids = {a["id"] for a in rows["arms"]}
        calls = rows.get("llm_calls", [])
        probe_calls = [c for c in calls if _is_probe_call(c)]
        orphan_calls = [c for c in calls if not _is_probe_call(c) and c.get("arm_id") not in arm_ids]
        ex.add_rows("llm_calls", "probe session", len(probe_calls))
        ex.add_rows("llm_calls", "no arm", len(orphan_calls))
        exps = rows.get("experiments", [])
        orphan_exps = [e for e in exps if e.get("arm_id") not in arm_ids]
        ex.add_rows("experiments", "no arm", len(orphan_exps))
        clean = dict(rows)
        clean["llm_calls"] = [c for c in calls if not _is_probe_call(c) and c.get("arm_id") in arm_ids]
        clean["experiments"] = [e for e in exps if e.get("arm_id") in arm_ids]
        kept.append((clean, bool(dev)))
    return kept, ex


# --------------------------------------------------------------------------- per race, per arm


@dataclass
class ArmRun:
    race_id: str
    seed: int
    target: str
    arm: str
    is_dev: bool
    cost_to_target_usd: float | None
    time_to_target_s: float | None
    gpu_idle_pct: float
    experiments_finished: int
    experiments_failed: int
    best_quality: float | None
    best_reported_score: float | None
    shipped_quality: float | None
    start_quality: float | None        # held-out score of this arm's starting-config run(s)
    gpu_cost_usd: float
    idle_cost_usd: float
    llm_cost_usd: float
    total_cost_usd: float
    llm_calls: int
    llm_p50_latency_ms: float | None
    malformed_rate: float | None
    fallback_rate: float | None
    retry_rate: float | None
    p50_by_step: dict[str, float]
    p50_valid_by_step: dict[str, float]
    calls_by_step: dict[str, int]
    n_request_ids: int
    n_experiments: int
    n_run_refs: int
    n_corrections: int
    normalised_quality: float | None = None


def _p50(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def arm_runs_for_race(rows: Rows, *, direction: str, target_value: float, is_dev: bool) -> list[ArmRun]:
    race = rows["races"][0]
    out: list[ArmRun] = []
    for arm in rows["arms"]:
        aid = arm["id"]
        exps = [e for e in rows.get("experiments", []) if e["arm_id"] == aid]
        ivs = [i for i in rows.get("gpu_intervals", []) if i["arm_id"] == aid]
        calls = [c for c in rows.get("llm_calls", []) if c["arm_id"] == aid]
        summary, extras = summarize_arm(
            arm["kind"], direction=direction, target_value=float(target_value),
            race_start=float(race["started_at"]), experiments=exps, intervals=ivs, llm_calls=calls,
        )
        n = len(calls)
        by_step: dict[str, list[float]] = {}
        valid_by_step: dict[str, list[float]] = {}
        for c in calls:
            st = str(c.get("step_type") or "?")
            lat = float(c.get("latency_ms") or 0.0)
            by_step.setdefault(st, []).append(lat)
            if _truthy(c.get("valid")):
                valid_by_step.setdefault(st, []).append(lat)
        base_q = [q for e in exps if _truthy(e.get("is_baseline")) and is_counted(e) and (q := quality(e)) is not None]
        out.append(ArmRun(
            race_id=race["id"], seed=int(race.get("seed") or 0), target=race["target"], arm=arm["kind"], is_dev=is_dev,
            cost_to_target_usd=summary.cost_to_target_usd, time_to_target_s=summary.time_to_target_s,
            gpu_idle_pct=summary.gpu_idle_pct, experiments_finished=summary.experiments_finished,
            experiments_failed=summary.experiments_failed, best_quality=summary.best_score,
            best_reported_score=extras.get("best_reported_score"), shipped_quality=extras.get("shipped_quality"),
            start_quality=statistics.fmean(base_q) if base_q else None,
            gpu_cost_usd=summary.gpu_cost_usd, idle_cost_usd=float(extras.get("idle_cost_usd") or 0.0),
            llm_cost_usd=summary.llm_cost_usd, total_cost_usd=summary.gpu_cost_usd + summary.llm_cost_usd,
            llm_calls=n, llm_p50_latency_ms=summary.llm_p50_latency_ms,
            malformed_rate=(sum(1 for c in calls if not _truthy(c.get("valid"))) / n) if n else None,
            fallback_rate=(sum(1 for c in calls if _truthy(c.get("is_fallback"))) / n) if n else None,
            retry_rate=(sum(1 for c in calls if _truthy(c.get("is_retry"))) / n) if n else None,
            p50_by_step={k: statistics.median(v) for k, v in by_step.items()},
            p50_valid_by_step={k: statistics.median(v) for k, v in valid_by_step.items()},
            calls_by_step={k: len(v) for k, v in by_step.items()},
            n_request_ids=sum(1 for c in calls if c.get("mubit_request_id")),
            n_experiments=len(exps),
            n_run_refs=sum(1 for e in exps if e.get("run_ref")),
            n_corrections=sum(1 for c in rows.get("corrections", []) if c.get("arm_id") == aid),
        ))
    return out


# --------------------------------------------------------------------------- aggregation


@dataclass
class Stat:
    mean: float | None
    std: float | None
    n: int          # runs with a value
    n_total: int    # runs in the group (n_total - n = target never reached / no value)


def stat(values: Sequence[float | None]) -> Stat:
    got = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return Stat(
        mean=statistics.fmean(got) if got else None,
        std=statistics.stdev(got) if len(got) > 1 else (0.0 if got else None),
        n=len(got), n_total=len(values),
    )


AGG_FIELDS = (
    "cost_to_target_usd", "time_to_target_s", "gpu_idle_pct", "experiments_finished", "best_quality",
    "normalised_quality", "shipped_quality", "llm_cost_usd", "gpu_cost_usd", "idle_cost_usd", "total_cost_usd",
    "llm_p50_latency_ms", "malformed_rate", "fallback_rate", "retry_rate", "llm_calls",
)


@dataclass
class ArmAggregate:
    arm: str
    seeds: list[int]
    race_ids: list[str]
    stats: dict[str, Stat]
    p50_by_step: dict[str, Stat]
    p50_valid_by_step: dict[str, Stat]
    calls_by_step: dict[str, int]
    evidence: dict[str, int]


@dataclass
class TargetReport:
    target: str
    direction: str
    target_value: float
    target_value_source: str
    start_score: float | None
    start_source: str
    best_known: float | None
    best_source: str
    is_dev: bool
    race_ids: list[str]
    arms: dict[str, ArmAggregate]
    runs: list[ArmRun]
    warnings: list[str]


def normalise(q: float | None, start: float | None, best: float | None) -> float | None:
    """0 = starting config, 100 = best known. Direction-agnostic: (q - start) / (best - start)."""
    if q is None or start is None or best is None or best == start:
        return None
    return 100.0 * (q - start) / (best - start)


def _better(a: float, b: float, direction: str) -> bool:
    return a < b if direction == "min" else a > b


# --------------------------------------------------------------------------- fairness fingerprint

# Race-wide settings from the fairness manifest (scheduler ``ArmRunner.fairness``) that must
# match before two races are pooled into one mean (fairness review 05:55, D41 #2).
FP_RACE_KEYS = ("target", "target_value", "budget_s", "run_budget_s", "n_slots", "runner", "cost_cap_usd",
                "accept_unsure", "pipelining", "lookahead", "max_proposals_per_call", "run_baseline_first",
                "launch_cutoff_s", "median_stop_rule")
_MULT_RE = re.compile(r":x([0-9.]+)$")


def canon_kind(kind: Any) -> str:
    """Arm kind as stored ("opus-bm" and "opus_bm" are the same column)."""
    return str(kind).strip().lower().replace("-", "_").replace("+", "_").replace(" ", "_")


def parse_prompts(p: Any) -> dict[str, str]:
    """``"propose_experiments=v1:fdda8c5b0919,judge_improvement=v0:6233..."`` -> {step: version}."""
    if isinstance(p, Mapping):
        return {str(k): str(v) for k, v in p.items()}
    out: dict[str, str] = {}
    for part in str(p or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
        elif part.strip():
            out[part.strip()] = "?"
    return out


def _runtime_multiplier(rows: Rows, first: Mapping[str, Any]) -> Any:
    """The surrogate's pacing constant; races before it was in the manifest carry it in the run
    refs (``surrogate:<id>:x1``). Unknown if the refs disagree or say nothing."""
    if "runtime_multiplier" in first:
        return first["runtime_multiplier"]
    mults = {m.group(1) for e in rows.get("experiments", []) if (m := _MULT_RE.search(str(e.get("run_ref") or "")))}
    if len(mults) == 1:
        return float(next(iter(mults)))
    return None if not mults else "mixed"


def fairness_fingerprint(rows: Rows) -> dict[str, Any] | None:
    """What two races must share to be pooled, from ``races.fairness_json``: the arm set; per
    arm the researcher, model, LLM mode, record mode, engine features (+ their rule settings)
    and LLM policy (retries, fallback, the version of every prompt the race actually called);
    the race-wide settings in ``FP_RACE_KEYS``; hardware / price; the Mubit project routing
    (manifest ``mubit_projects``). ``None`` for a race without a manifest.

    A prompt listed in the policy but never called in the race (e.g. early_stop v2 after D40
    made early stopping an engine rule) cannot affect its numbers, so it is left out of the
    fingerprint and reported under ``unused_prompts`` instead."""
    race = rows["races"][0]
    f = race.get("fairness")
    if not isinstance(f, Mapping) or not f:
        return None
    per_arm = f.get("per_arm") or {}
    first: Mapping[str, Any] = next(iter(per_arm.values()), {}) if isinstance(per_arm, Mapping) else {}
    steps_used = {str(c.get("step_type")) for c in rows.get("llm_calls", []) if c.get("step_type")}
    differs = f.get("differs_only_in") or {}
    policies = f.get("llm_policy") or {}
    engine = f.get("engine") or {}
    feats_map = first.get("engine_features") or engine.get("features") or {}
    arms: dict[str, Any] = {}
    unused: dict[str, list[str]] = {}
    for a in sorted(rows.get("arms", []), key=lambda a: canon_kind(a["kind"])):
        raw = str(a["kind"])
        k = canon_kind(raw)
        d = differs.get(raw) or differs.get(k)
        rec_modes = first.get("record_modes") or {}
        entry: dict[str, Any] = {
            "researcher": d.get("researcher") if d else None,
            "model": d.get("model") if d else a.get("model"),
            "llm_mode": d.get("llm_mode") if d else a.get("llm_mode"),
            "record_mode": d.get("record_mode") if d else rec_modes.get(raw, rec_modes.get(k)),
            # Absent = a race from before the engine existed: no features, which is what ran.
            "engine_features": sorted((d or {}).get("engine_features", feats_map.get(raw, feats_map.get(k, [])))
                                      or []),
        }
        eng = (engine.get("per_arm") or {}).get(raw) or (engine.get("per_arm") or {}).get(k) or {}
        if eng.get("median_stop"):
            entry["median_stop"] = {x: eng["median_stop"].get(x)
                                    for x in ("fractions", "min_completed", "curve_metric", "time_scale")}
        if eng.get("judge_policy"):
            entry["judge_policy"] = eng["judge_policy"]
        p = policies.get(raw) or policies.get(k)
        if p:
            prompts = parse_prompts(p.get("prompts"))
            entry["llm_policy"] = {
                **{x: p.get(x) for x in ("researcher", "max_retries", "has_fallback", "fallback_model")},
                "prompts": {s: v for s, v in sorted(prompts.items()) if s in steps_used},
            }
            gone = [f"{s}={v}" for s, v in sorted(prompts.items()) if s not in steps_used]
            if gone:
                unused[k] = gone
        arms[k] = entry
    settings = {key: first.get(key) for key in FP_RACE_KEYS}
    settings["runtime_multiplier"] = _runtime_multiplier(rows, first)
    settings["hardware"] = sorted({str(h) for h in (first.get("hardware") or [])})
    settings["price_per_s"] = sorted({float(x) for x in (first.get("price_per_s") or [])})
    return {"arms": arms, "race": settings, "mubit_projects": f.get("mubit_projects"),
            "unused_prompts": unused}


def fingerprint_id(fp: Mapping[str, Any] | None) -> str | None:
    """Short stable hash of the fingerprint (``unused_prompts`` excluded); None without one."""
    if fp is None:
        return None
    core = {k: v for k, v in fp.items() if k != "unused_prompts"}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:10]


_ABSENT = object()


def _diff(a: Any, b: Any, path: str, out: list[str]) -> None:
    """Recurse while both sides are mappings; a key on one side only is reported once, whole."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for k in sorted(set(a) | set(b), key=str):
            _diff(a.get(k, _ABSENT), b.get(k, _ABSENT), f"{path}.{k}" if path else str(k), out)
    elif a != b:
        def show(v: Any) -> str:
            return "absent" if v is _ABSENT else str(v)[:60]
        out.append(f"{path}: {show(a)} vs {show(b)}")


def fingerprint_diff(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None, limit: int = 8) -> list[str]:
    """Human-readable differences between two fingerprints ("arm set", then flattened paths)."""
    if a is None or b is None:
        return ["one race has no fairness manifest"] if (a is None) != (b is None) else []
    out: list[str] = []
    sa, sb = sorted(a.get("arms", {})), sorted(b.get("arms", {}))
    if sa != sb:
        out.append(f"arm set {'/'.join(sa)} vs {'/'.join(sb)}")
    common = {k: a["arms"][k] for k in sa if k in sb}, {k: b["arms"][k] for k in sb if k in sa}
    _diff({"arms": common[0], "race": a.get("race"), "mubit_projects": a.get("mubit_projects")},
          {"arms": common[1], "race": b.get("race"), "mubit_projects": b.get("mubit_projects")}, "", out)
    if len(out) > limit:
        out = out[:limit] + [f"... {len(out) - limit} more"]
    return out


def fairness_warnings(group: list[Rows]) -> list[str]:
    w: list[str] = []
    fps = [(r["races"][0]["id"], fairness_fingerprint(r)) for r in group]
    ids = {fingerprint_id(fp) for _, fp in fps}
    if len(ids) > 1:
        ref_id, ref = fps[0]
        for rid, fp in fps[1:]:
            if fingerprint_id(fp) != fingerprint_id(ref):
                w.append(f"race {str(rid)[:8]} differs from race {str(ref_id)[:8]} in its fairness fingerprint "
                         f"({'; '.join(fingerprint_diff(ref, fp, limit=3))}): mean over unlike setups")
    for key in ("budget_s", "gpus_per_arm"):
        vals = {r["races"][0].get(key) for r in group}
        if len(vals) > 1:
            w.append(f"races differ in {key}: {sorted(vals, key=str)} (mean over unlike races)")
    arm_sets = {tuple(sorted(a["kind"] for a in r["arms"])) for r in group}
    if len(arm_sets) > 1:
        w.append(f"races have different arm sets: {sorted(arm_sets)}")
    seeds = [r["races"][0].get("seed") for r in group]
    if len(seeds) != len(set(seeds)):
        w.append(f"repeated seeds {sorted(seeds, key=str)}: rows are not independent seeds")
    for r in group:
        rid = r["races"][0]["id"]
        for key in ("n_slots", "hardware", "price_per_s"):
            vals = {a.get(key) for a in r["arms"]}
            if len(vals) > 1:
                w.append(f"race {rid}: arms differ in {key} {sorted(vals, key=str)} (unfair)")
    return w


def analyse(
    kept: list[tuple[Rows, bool]],
    *,
    target_values: Mapping[str, float] | None = None,
    best_known: Mapping[str, float] | None = None,
    start_scores: Mapping[str, float] | None = None,
) -> list[TargetReport]:
    target_values = target_values or {}
    best_known = best_known or {}
    start_scores = start_scores or {}
    by_target: dict[str, list[tuple[Rows, bool]]] = {}
    for rows, dev in kept:
        by_target.setdefault(rows["races"][0]["target"], []).append((rows, dev))

    reports: list[TargetReport] = []
    for target in sorted(by_target):
        group = by_target[target]
        warnings = fairness_warnings([g for g, _ in group])
        try:
            from workers.targets.base import get_target

            spec = get_target(target).spec
            direction, tv, tv_src = spec.direction, spec.target_value, "target spec"
        except Exception as e:  # unknown target: needs --target-value and a direction
            spec = None
            direction, tv, tv_src = "max", None, f"unknown target ({e!r})"
            warnings.append(f"target {target!r} not in the registry; direction assumed 'max'")
        if target in target_values:
            tv, tv_src = float(target_values[target]), "--target-value"
        else:
            # The goal each race actually ran with (bench.py stores it in the RaceConfig; the
            # agents saw it too). None => that race used the spec's value. Same rule as
            # backend.benchmark.analysis.summaries_from_rows, so both give the same numbers.
            race_goals: set[float | None] = set()
            provisional = from_config = False
            for g, _ in group:
                c = g["races"][0].get("config") or {}
                v = c.get("target_value")
                from_config = from_config or v is not None
                race_goals.add(float(v) if v is not None else (None if tv is None else float(tv)))
                provisional = provisional or bool(c.get("provisional"))
            if len(race_goals) > 1:
                warnings.append(f"races for {target!r} ran with different goals {sorted(race_goals, key=str)}: "
                                f"not aggregated (pass --target-value {target}=X to force one)")
                continue
            if from_config and race_goals:
                tv, tv_src = next(iter(race_goals)), "race config (bench.py goal)"
            if provisional:
                tv_src += ", PROVISIONAL"
        if tv is None:
            warnings.append(f"no target value for {target!r}: skipped (pass --target-value {target}=X)")
            continue

        runs: list[ArmRun] = []
        for rows, dev in sorted(group, key=lambda g: (g[0]["races"][0].get("seed") or 0, g[0]["races"][0]["id"])):
            runs.extend(arm_runs_for_race(rows, direction=direction, target_value=tv, is_dev=dev))

        if target in start_scores:
            start, start_src = float(start_scores[target]), "--start-score"
        else:
            sq = [r.start_quality for r in runs if r.start_quality is not None]
            start = statistics.fmean(sq) if sq else None
            start_src = f"mean held-out score of {len(sq)} starting-config runs" if sq else "none (no starting-config run)"
        if target in best_known:
            best, best_src = float(best_known[target]), "--best-known"
        else:
            qs = [r.best_quality for r in runs if r.best_quality is not None]
            best = None
            for q in qs:
                if best is None or _better(q, best, direction):
                    best = q
            best_src = "best held-out score seen in these races (pass --best-known to fix it)"
        for r in runs:
            r.normalised_quality = normalise(r.best_quality, start, best)

        arms: dict[str, ArmAggregate] = {}
        for kind in sorted({r.arm for r in runs}, key=lambda k: ARM_ORDER.index(k) if k in ARM_ORDER else 99):
            rs = [r for r in runs if r.arm == kind]
            steps = sorted({s for r in rs for s in r.p50_by_step})
            arms[kind] = ArmAggregate(
                arm=kind, seeds=[r.seed for r in rs], race_ids=[r.race_id for r in rs],
                stats={f: stat([getattr(r, f) for r in rs]) for f in AGG_FIELDS},
                p50_by_step={s: stat([r.p50_by_step.get(s) for r in rs]) for s in steps},
                p50_valid_by_step={s: stat([r.p50_valid_by_step.get(s) for r in rs]) for s in steps},
                calls_by_step={s: sum(r.calls_by_step.get(s, 0) for r in rs) for s in steps},
                evidence={
                    "llm_calls": sum(r.llm_calls for r in rs),
                    "request_ids": sum(r.n_request_ids for r in rs),
                    "experiments": sum(r.n_experiments for r in rs),
                    "run_ids": sum(r.n_run_refs for r in rs),
                    "corrections": sum(r.n_corrections for r in rs),
                },
            )
        reports.append(TargetReport(
            target=target, direction=direction, target_value=float(tv), target_value_source=tv_src,
            start_score=start, start_source=start_src, best_known=best, best_source=best_src,
            is_dev=any(d for _, d in group), race_ids=[g["races"][0]["id"] for g, _ in group],
            arms=arms, runs=runs, warnings=warnings,
        ))
    return reports


# --------------------------------------------------------------------------- markdown


def _fmt(s: Stat, scale: float = 1.0, digits: int = 2, unit: str = "", prefix: str = "", spec: str = "f") -> str:
    """mean ± std (std only with >1 value). spec="g" uses ``digits`` significant digits."""
    if s.mean is None:
        return "n/a"
    m = s.mean * scale
    sd = (s.std or 0.0) * scale
    return f"{prefix}{m:.{digits}{spec}}" + (f" ± {sd:.{digits}{spec}}" if s.n > 1 else "") + unit


def _fmt_reach(s: Stat, **kw: Any) -> str:
    if s.n == 0:
        return f"didn't reach it (0/{s.n_total})"
    body = _fmt(s, **kw)
    return body if s.n == s.n_total else f"{body} (reached {s.n}/{s.n_total})"


def _fmt_q(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4g}"


def _evidence(a: ArmAggregate) -> str:
    e = a.evidence
    return (f"{e['request_ids']}/{e['llm_calls']} calls with Mubit request ID · "
            f"{e['run_ids']}/{e['experiments']} runs with run ID")


def render_markdown(reports: list[TargetReport], ex: Exclusions) -> str:
    out: list[str] = []
    if not reports:
        out.append("_No races left to analyse after exclusions._\n")
    for rep in reports:
        label = TARGET_LABEL.get(rep.target, rep.target)
        n_seeds = max((len(a.seeds) for a in rep.arms.values()), default=0)
        stamp = f" **[{DEV_STAMP}]**" if rep.is_dev else ""
        goal = "≤" if rep.direction == "min" else "≥"
        out.append(f"### night/bench: {label} (`{rep.target}`), {n_seeds} seed(s){stamp}\n")
        out.append(f"- Goal: held-out score {goal} {rep.target_value:g} (source: {rep.target_value_source}).")
        out.append(f"- Model quality 0–100: 0 = {_fmt_q(rep.start_score)} ({rep.start_source}); "
                   f"100 = {_fmt_q(rep.best_known)} ({rep.best_source}).")
        out.append("- Mean ± std over seeds. Cost and time to the goal average only the runs that reached it.")
        out.append(f"- Races: {', '.join(f'`{r}`' for r in rep.race_ids)}.")
        for w in rep.warnings:
            out.append(f"- ⚠️ {w}")
        out.append("")
        out.append("| Arm | Seeds | Cost to reach the goal | Time to reach the goal | GPUs waiting | "
                   "Experiments finished | Model quality (0–100) | Best held-out score | Shipped model held-out | "
                   "LLM $ | GPU $ (of which waiting) | p50 LLM latency / call | Malformed replies | Fallbacks | Evidence |")
        out.append("|" + "---|" * 15)
        for kind, a in rep.arms.items():
            s = a.stats
            is_llm = s["llm_calls"].mean not in (None, 0.0)
            out.append("| " + " | ".join([
                ARM_LABEL.get(kind, kind),
                ",".join(str(x) for x in a.seeds),
                _fmt_reach(s["cost_to_target_usd"], prefix="$"),
                _fmt_reach(s["time_to_target_s"], scale=1 / 60, digits=1, unit=" min"),
                _fmt(s["gpu_idle_pct"], digits=1, unit="%"),
                _fmt(s["experiments_finished"], digits=1),
                _fmt(s["normalised_quality"], digits=0),
                _fmt(s["best_quality"], digits=4, spec="g"),
                _fmt(s["shipped_quality"], digits=4, spec="g"),
                _fmt(s["llm_cost_usd"], prefix="$", digits=3),
                f"{_fmt(s['gpu_cost_usd'], prefix='$')} ({_fmt(s['idle_cost_usd'], prefix='$')})",
                _fmt(s["llm_p50_latency_ms"], scale=1 / 1000, digits=2, unit=" s") if is_llm else "no LLM",
                _fmt(s["malformed_rate"], scale=100, digits=1, unit="%") if is_llm else "–",
                _fmt(s["fallback_rate"], scale=100, digits=1, unit="%") if is_llm else "–",
                _evidence(a),
            ]) + " |")
        out.append("")
        step_rows = [(k, a) for k, a in rep.arms.items() if a.p50_by_step]
        if step_rows:
            out.append("p50 LLM latency per step type (mean ± std of per-seed p50s). The last column counts valid "
                       "replies only, since malformed replies can be fast:\n")
            out.append("| Arm | Step | Calls | p50 latency (all) | p50 latency (valid only) |")
            out.append("|---|---|---|---|---|")
            for kind, a in step_rows:
                for st, ss in a.p50_by_step.items():
                    out.append(f"| {ARM_LABEL.get(kind, kind)} | {st} | {a.calls_by_step[st]} | "
                               f"{_fmt(ss, scale=1 / 1000, unit=' s')} | {_fmt(a.p50_valid_by_step[st], scale=1 / 1000, unit=' s')} |")
            out.append("")
        out.append("Rows for the \"Benchmark results\" table in docs/results_log.md:\n")
        out.append("| Target | Arm | Seeds | Time to goal | Cost to goal | GPUs waiting % | Model quality @ budget | Notes |")
        out.append("|---|---|---|---|---|---|---|---|")
        for kind, a in rep.arms.items():
            s = a.stats
            notes = (f"held-out {_fmt(s['best_quality'], digits=4, spec='g')}; LLM {_fmt(s['llm_cost_usd'], prefix='$', digits=3)}; "
                     f"{_evidence(a)}; races {', '.join(r[:8] for r in a.race_ids)}")
            if rep.is_dev:
                notes = f"{DEV_STAMP}; {notes}"
            out.append(f"| {label} | {ARM_LABEL.get(kind, kind)} | {len(a.seeds)} | "
                       f"{_fmt_reach(s['time_to_target_s'], scale=1 / 60, digits=1, unit=' min')} | "
                       f"{_fmt_reach(s['cost_to_target_usd'], prefix='$')} | {_fmt(s['gpu_idle_pct'], digits=1, unit='%')} | "
                       f"{_fmt(s['normalised_quality'], digits=0)} (0–100) | {notes} |")
        out.append("")
    out.append("#### Excluded from this analysis\n")
    if not (ex.races or ex.llm_calls or ex.experiments or ex.duplicates):
        out.append("- Nothing excluded.")
    for rid, why in ex.races:
        out.append(f"- race `{rid}`: {why}")
    for why, n in ex.llm_calls.items():
        out.append(f"- {n} LLM call row(s): {why}")
    for why, n in ex.experiments.items():
        out.append(f"- {n} experiment row(s): {why}")
    for rid in ex.duplicates:
        out.append(f"- race `{rid}`: duplicate (loaded twice; counted once)")
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- charts

# Reference palette (dataviz skill, references/palette.md): categorical slots 1-3 validate
# all-pairs in both modes. Colour follows the arm, never its rank.
THEMES = {
    "dark": {
        "surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#3a3a37",
        "arm": {"mubit": "#3987e5", "opus": "#d95926", "optuna": "#199e70", "mock": "#8a8984"},
    },
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
        "arm": {"mubit": "#2a78d6", "opus": "#eb6834", "optuna": "#1baf7a", "mock": "#8a8984"},
    },
}

# (file stem, stat field, title, how to read it, value label, axis tick format)
CHARTS = (
    ("gpus-waiting", "gpu_idle_pct", "GPUs waiting", "share of paid GPU time spent waiting · lower is better",
     lambda v: f"{v:.0f}%", "{x:.0f}%"),
    ("cost-to-goal", "cost_to_target_usd", "Cost to reach the goal", "GPU + assistant money until the goal is reached · lower is better",
     lambda v: f"${v:.2f}", "${x:.2f}"),
    ("model-quality", "normalised_quality", "Model quality", "0 = starting model, 100 = best known result · higher is better",
     lambda v: f"{v:.0f}", "{x:.0f}"),
    ("experiments-finished", "experiments_finished", "Experiments finished", "in the same time and budget · higher is better",
     lambda v: f"{v:.0f}", "{x:.0f}"),
)


def render_charts(reports: list[TargetReport], out_dir: str | Path, theme: str = "dark") -> list[Path]:
    """Four horizontal bar charts per target. Returns the written paths ([] without matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import StrMethodFormatter
    except ImportError:
        print("matplotlib not installed: charts skipped (python -m uv pip install matplotlib)", file=sys.stderr)
        return []
    th = THEMES[theme]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for rep in reports:
        label = TARGET_LABEL.get(rep.target, rep.target)
        kinds = list(rep.arms)
        for stem, fld, title, how, fmt, tick_fmt in CHARTS:
            fig_h = 2.2 + 1.05 * len(kinds)
            fig, ax = plt.subplots(figsize=(10, fig_h), dpi=150)
            fig.patch.set_facecolor(th["surface"])
            ax.set_facecolor(th["surface"])
            stats = [rep.arms[k].stats[fld] for k in kinds]
            vals = [s.mean for s in stats]
            finite = [v for v in vals if v is not None]
            hi = max([0.0] + [v + (s.std or 0.0) for v, s in zip(vals, stats) if v is not None])
            lo = min([0.0] + finite)
            span = (hi - lo) or 1.0
            ys = list(range(len(kinds)))[::-1]
            for y, k, s in zip(ys, kinds, stats):
                if s.mean is None:
                    msg = "didn't reach the goal" if fld == "cost_to_target_usd" else "no data"
                    if fld == "cost_to_target_usd" and s.n_total:
                        msg += f" (0 of {s.n_total} runs)"
                    ax.text(lo + 0.01 * span, y, msg, va="center", ha="left", color=th["muted"], fontsize=14)
                    continue
                ax.barh(y, s.mean, height=0.5, color=th["arm"].get(k, th["arm"]["mock"]), zorder=2)
                end = s.mean
                if s.n > 1 and s.std:
                    ax.plot([s.mean - s.std, s.mean + s.std], [y, y], color=th["muted"], lw=2, zorder=3,
                            solid_capstyle="butt")
                    end = max(s.mean, s.mean + s.std) if s.mean >= 0 else min(s.mean, s.mean - s.std)
                text = fmt(s.mean)
                if fld == "cost_to_target_usd" and s.n < s.n_total:
                    text += f"  (reached in {s.n} of {s.n_total} runs)"
                ha = "left" if s.mean >= 0 else "right"
                ax.text(end + (0.015 if s.mean >= 0 else -0.015) * span, y, text, va="center", ha=ha,
                        color=th["text"], fontsize=16, fontweight="bold")
            ax.set_yticks(ys)
            ax.set_yticklabels([ARM_CHART_LABEL.get(k, k) for k in kinds], color=th["text"], fontsize=14)
            if fld == "normalised_quality":  # a 0-100 bar; widen only if a run beats "best known"
                ax.set_xlim(min(lo, 0.0) - (5 if lo < 0 else 0), max(hi, 100.0) * 1.25)
                ax.set_xticks([t for t in (0, 25, 50, 75, 100) if t >= lo])
            else:
                ax.set_xlim(lo - (0.05 * span if lo < 0 else 0), hi + 0.35 * span)
            ax.xaxis.set_major_formatter(StrMethodFormatter(tick_fmt))
            ax.set_ylim(-0.6, len(kinds) - 0.4)
            ax.axvline(0, color=th["muted"], lw=1, zorder=1)
            ax.xaxis.grid(True, color=th["grid"], lw=0.8, zorder=0)
            ax.tick_params(axis="x", colors=th["muted"], labelsize=11)
            ax.tick_params(axis="y", length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            n_runs = max((s.n_total for s in stats), default=0)
            top = 1.0 - 0.12 / fig_h
            fig.text(0.02, top, f"{title} · {label}", ha="left", va="top", color=th["text"], fontsize=20,
                     fontweight="bold")
            sub = f"{how} · mean of {n_runs} run(s), grey line = ± 1 std"
            fig.text(0.02, top - 0.48 / fig_h, sub, ha="left", va="top", color=th["muted"], fontsize=12)
            if rep.is_dev:
                fig.text(0.02, top - 0.78 / fig_h, DEV_STAMP, ha="left", va="top", color=th["arm"]["opus"],
                         fontsize=12, fontweight="bold")
            fair = ("Same GPUs, same budget, same seeds." if not rep.warnings
                    else f"Fairness warnings ({len(rep.warnings)}): see the results table.")
            fig.text(0.02, 0.012, f"{fair} Source: night/bench races " + ", ".join(r[:8] for r in rep.race_ids),
                     color=th["muted"], fontsize=9)
            fig.tight_layout(rect=(0, 0.25 / fig_h, 1, 1.0 - 1.15 / fig_h))
            path = out / f"{rep.target}-{stem}{'-dev' if rep.is_dev else ''}.png"
            fig.savefig(path, facecolor=th["surface"])
            plt.close(fig)
            written.append(path)
    return written


# --------------------------------------------------------------------------- CLI


def _kv(items: Sequence[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for it in items or ():
        k, sep, v = it.partition("=")
        if not sep:
            raise SystemExit(f"expected TARGET=VALUE, got {it!r}")
        out[k.strip()] = float(v)
    return out


def to_json(reports: list[TargetReport], ex: Exclusions) -> dict[str, Any]:
    return {
        "reports": [asdict(r) for r in reports],
        "excluded": asdict(ex),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="night/bench analysis (see module docstring).")
    ap.add_argument("--db", help="SQLite DB written by backend/db.py")
    ap.add_argument("--race", action="append", help="race id to include (repeatable; default: every race in --db)")
    ap.add_argument("--jsonl", nargs="+", default=[], help="JSONL race exports")
    ap.add_argument("--target-value", action="append", help="TARGET=VALUE goal threshold (default: target spec)")
    ap.add_argument("--best-known", action="append", help="TARGET=VALUE for quality 100 (default: best seen)")
    ap.add_argument("--start-score", action="append", help="TARGET=VALUE for quality 0 (default: starting-config runs)")
    ap.add_argument("--allow-dev", action="store_true", help="keep mock/replay races (output stamped DEV)")
    ap.add_argument("--include-unfinished", action="store_true")
    ap.add_argument("--charts-dir", default=str(REPO_ROOT / "data" / "charts"))
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--theme", choices=sorted(THEMES), default="dark")
    ap.add_argument("--out", help="also write the markdown here")
    ap.add_argument("--json", help="also write every computed number here")
    args = ap.parse_args(argv)
    if not args.db and not args.jsonl:
        ap.error("give --db and/or --jsonl")

    races: list[Rows] = []
    if args.db:
        races.extend(load_db(args.db, args.race))
    if args.jsonl:
        races.extend(load_jsonl(args.jsonl))
    kept, ex = screen(races, allow_dev=args.allow_dev, include_unfinished=args.include_unfinished)
    reports = analyse(kept, target_values=_kv(args.target_value), best_known=_kv(args.best_known),
                      start_scores=_kv(args.start_score))
    md = render_markdown(reports, ex)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(to_json(reports, ex), indent=2, default=str), encoding="utf-8")
    if not args.no_charts and reports:
        for p in render_charts(reports, args.charts_dir, theme=args.theme):
            print(f"chart: {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
