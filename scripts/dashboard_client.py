"""Small dashboard API client: start a mock dev race or replay a logged race over the dashboard
WebSocket, print a one-line summary per tick and save every message as JSON lines.

No LLM calls, no Modal: the dev race uses the mock LLM and the in-process surrogate runner ($0);
a replay only reads stored rows. Needs the API running, e.g. (PowerShell, repo root)::

    $env:BUSY_MACHINES_REPLAY_SOURCES="data/bench/bench.db"
    python -m uv run --extra ml --with onnxruntime==1.29.0 uvicorn --factory backend.app:create_app --port 8765

Then::

    python -m uv run python scripts/dashboard_client.py replay 4ca4ed27 --speed 60 --seconds 20
    python -m uv run python scripts/dashboard_client.py dev --budget 90 --seconds 100
    python -m uv run python scripts/dashboard_client.py snapshot c1f4b1d9 --at 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import websockets

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "dashboard_examples"

# A $0 dev race that looks like the demo: standard assistant vs Busy Machines (same mock AI,
# the same planner delay on both sides; Busy Machines' sampler fills the slots while it waits).
DEV_RACE: dict[str, Any] = {
    "name": "dashboard dev race (mock AI, simulated machines)",
    "target": "lcbench", "arms": ["opus", "opus_bm"], "budget_s": 90, "gpus_per_arm": 8, "seed": 0,
    "mode": "dev", "llm": "mock", "runner": "surrogate", "surrogate_runtime_mult": 0.1,
    "mock_latency_s": {"opus": 6.0, "opus_bm": 6.0}, "mock_trap_rate": 0.2,
    "max_proposals_per_call": 8,
    "engine_features": {"opus_bm": ["backfill", "judge_policy", "lean_planner", "planner_cadence"]},
    "feature_params": {"opus_bm": {"planner_cadence": {"min_propose_interval_s": 20}}},
}


def _line(snap: dict[str, Any]) -> str:
    parts = [f"{snap['clock']} {snap['status']:<8}"]
    for a in snap["arms"]:
        if a["side"] == "secondary":
            continue
        mq = a["model_quality"]["best_text"] or "-"
        think = " thinking" if a["planner"]["thinking"] else ""
        parts.append(f"{a['label'][:14]:<14} AI ${a['ai_bill_usd']:.2f} GPU ${a['gpu_bill_usd']:.3f} "
                     f"done {a['experiments_finished']:>4} q {mq:>8} wait {a['waiting_so_far_pct']:5.1f}% "
                     f"work {a['working_now']:>2}/{a['n_slots']} fakes {a['fakes_caught']}{think}")
    return " | ".join(parts)


async def stream(url: str, out: Path, seconds: float) -> Counter[str]:
    counts: Counter[str] = Counter()
    t_end = time.monotonic() + seconds
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        async with websockets.connect(url, max_size=None) as ws:
            while time.monotonic() < t_end:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=max(0.1, t_end - time.monotonic()))
                except (TimeoutError, websockets.ConnectionClosed):
                    break
                f.write(msg if isinstance(msg, str) else msg.decode())
                f.write("\n")
                ev = json.loads(msg)
                counts[ev["type"]] += 1
                if ev["type"] == "tick":
                    print(_line(ev["snapshot"]), flush=True)
                elif ev["type"] == "race_finished":
                    print("race_finished:", ev.get("reason"), flush=True)
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["replay", "dev", "snapshot"])
    ap.add_argument("race", nargs="?", help="race id or unique prefix (replay / snapshot)")
    ap.add_argument("--api", default="http://127.0.0.1:8765")
    ap.add_argument("--speed", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=20.0, help="how long to listen (wall s)")
    ap.add_argument("--tick", type=float, default=None, help="tick_s (default 0.5 replay, 1.0 live)")
    ap.add_argument("--budget", type=float, default=None, help="dev race budget_s")
    ap.add_argument("--at", type=float, default=None, help="snapshot: race time (s)")
    ap.add_argument("--raw", default="true", choices=["true", "false"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    ws_base = args.api.replace("http", "ws", 1)
    with httpx.Client(base_url=args.api, timeout=120) as c:
        if args.what == "snapshot":
            r = c.get(f"/api/races/{args.race}/snapshot", params={} if args.at is None else {"at_s": args.at})
            r.raise_for_status()
            snap = r.json()
            print(_line(snap))
            if args.out:
                Path(args.out).write_text(json.dumps(snap, indent=1), encoding="utf-8")
            return 0
        if args.what == "replay":
            r = c.post(f"/api/races/{args.race}/replay", params={"speed": args.speed, "metric_stride": 0})
            r.raise_for_status()
            info = r.json()
            print(f"replay {info['replay_id']} of race {info['race_id'][:8]}: {info['n_events']} events, "
                  f"{info['replay_duration_s']:.1f}s at {args.speed:g}x", flush=True)
            url = f"{ws_base}/ws/replays/{info['replay_id']}/dashboard?tick_s={args.tick or 0.5}&raw={args.raw}"
            out = Path(args.out) if args.out else OUT_DIR / f"replay-{info['race_id'][:8]}-x{args.speed:g}.jsonl"
        else:
            body = dict(DEV_RACE)
            if args.budget:
                body["budget_s"] = args.budget
            r = c.post("/api/races", json=body)
            r.raise_for_status()
            rid = r.json()["id"]
            url = f"{ws_base}/ws/races/{rid}/dashboard?tick_s={args.tick or 1.0}&raw={args.raw}"
            out = Path(args.out) if args.out else OUT_DIR / "live-dev-race.jsonl"
            print(f"dev race {rid}: connecting, then starting", flush=True)
    if args.what == "dev":
        async def run_dev() -> Counter[str]:
            task = asyncio.create_task(stream(url, out, args.seconds))
            await asyncio.sleep(0.5)
            async with httpx.AsyncClient(base_url=args.api, timeout=60) as ac:
                (await ac.post(f"/api/races/{rid}/start")).raise_for_status()
            return await task
        counts = asyncio.run(run_dev())
    else:
        counts = asyncio.run(stream(url, out, args.seconds))
    print(f"saved {sum(counts.values())} messages to {out}: {dict(counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
