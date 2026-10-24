"""Recompute the headline metrics from a JSONL export, independently of the engine, and
aggregate seeds (mean +/- std). Usage:

    python -m backend.benchmark.analysis race_seed0.jsonl race_seed1.jsonl ...

Dev races (mock LLM / replay) are rejected unless ``--allow-dev`` is given.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from typing import Any

from backend.benchmark.export import read_export
from backend.benchmark.metrics import summarize_arm
from backend.core.types import ArmSummary
from backend.db import BenchmarkIntegrityError
from workers.targets.base import get_target

HEADLINE_FIELDS = (
    "cost_to_target_usd", "time_to_target_s", "gpu_idle_pct", "best_score", "experiments_finished",
    "gpu_cost_usd", "llm_cost_usd", "total_cost_usd", "llm_p50_latency_ms",
)


def check_benchmark_rows(rows: dict[str, list[dict[str, Any]]]) -> None:
    race = rows["races"][0]
    if race.get("is_dev", True):
        raise BenchmarkIntegrityError(f"race {race['id']} is a dev race (mock/replay)")
    for arm in rows.get("arms", []):
        if arm.get("is_dev") or arm.get("llm_mode") in ("mock", "replay") or arm.get("kind") == "mock":
            raise BenchmarkIntegrityError(f"arm {arm['id']} uses {arm.get('llm_mode')}")
    if any(c.get("is_replay") for c in rows.get("llm_calls", [])):
        raise BenchmarkIntegrityError("export contains replayed LLM calls")


def summaries_from_rows(
    rows: dict[str, list[dict[str, Any]]],
    *,
    direction: str | None = None,
    target_value: float | None = None,
    allow_dev: bool = False,
) -> dict[str, tuple[ArmSummary, dict[str, Any]]]:
    """arm kind -> (ArmSummary, extras), recomputed from raw rows."""
    if not allow_dev:
        check_benchmark_rows(rows)
    race = rows["races"][0]
    if target_value is None:
        # The goal the race ran with (bench.py stores it in the RaceConfig), else the spec's.
        cfg_tv = (race.get("config") or {}).get("target_value")
        target_value = float(cfg_tv) if cfg_tv is not None else None
    if direction is None or target_value is None:
        spec = get_target(race["target"]).spec
        direction = direction or spec.direction
        target_value = spec.target_value if target_value is None else target_value
    out: dict[str, tuple[ArmSummary, dict[str, Any]]] = {}
    for arm in rows.get("arms", []):
        aid = arm["id"]
        out[arm["kind"]] = summarize_arm(
            arm["kind"], direction=direction, target_value=float(target_value),
            race_start=float(race["started_at"]),
            experiments=[e for e in rows.get("experiments", []) if e["arm_id"] == aid],
            intervals=[i for i in rows.get("gpu_intervals", []) if i["arm_id"] == aid],
            llm_calls=[c for c in rows.get("llm_calls", []) if c["arm_id"] == aid],
        )
    return out


def aggregate(per_seed: list[dict[str, ArmSummary]]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """arm -> field -> {mean, std, n, n_missing}. ``None`` values (target never reached)
    are counted in n_missing, never silently dropped."""
    arms = sorted({a for s in per_seed for a in s})
    out: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for arm in arms:
        out[arm] = {}
        for f in HEADLINE_FIELDS:
            vals = [getattr(s[arm], f) for s in per_seed if arm in s]
            got = [float(v) for v in vals if v is not None]
            out[arm][f] = {
                "mean": statistics.fmean(got) if got else None,
                "std": statistics.stdev(got) if len(got) > 1 else (0.0 if got else None),
                "n": len(got),
                "n_missing": len(vals) - len(got),
            }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("exports", nargs="+")
    ap.add_argument("--allow-dev", action="store_true")
    args = ap.parse_args(argv)
    per_seed = []
    for path in args.exports:
        rows = read_export(path)
        per_seed.append({k: v[0] for k, v in summaries_from_rows(rows, allow_dev=args.allow_dev).items()})
    agg = aggregate(per_seed)
    for arm, fields in agg.items():
        print(f"== {arm}")
        for f, st in fields.items():
            mean, std = st["mean"], st["std"]
            shown = "n/a" if mean is None else f"{mean:.4g} +/- {std:.2g}"
            print(f"  {f:22s} {shown}  (n={st['n']}, missing={st['n_missing']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
