"""JSONL export of a race (the verification dump) and its reader.

Format: one line per row, ``{"table": <name>, "row": {...}}``, tables in the order
races, arms, experiments, metrics, llm_calls, gpu_intervals, corrections.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from backend.db import Database


def export_race(db: Database, race_id: str, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for line in db.export_race_jsonl(race_id):
            f.write(line)
    return p


def parse_jsonl(lines: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        out[obj["table"]].append(obj["row"])
    return dict(out)


def read_export(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    with Path(path).open(encoding="utf-8") as f:
        return parse_jsonl(f)
