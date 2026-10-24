"""End-to-end dev race through the real FastAPI app: mock LLM + local CPU runner. $0.

Starts uvicorn (unless --url is given), creates a race, starts it, watches the WebSocket
stream until ``race_finished``, then prints each arm's summary and saves the JSONL export.
No Modal, no real LLM. Every number it prints is a MOCK number (simulated LLM latency on
laptop CPU): never put it in results_log.md as a benchmark result.

    uv run --extra ml python scripts/mock_race.py
    uv run --extra ml python scripts/mock_race.py --target rl --budget 150 --run-budget 12 \
        --slots 2 --opus-latency 8 --mubit-latency 0.3
    uv run --extra ml python scripts/mock_race.py --url http://127.0.0.1:8000   # existing server
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import websockets

ROOT = Path(__file__).resolve().parents[1]


def start_server(port: int, db_path: Path, env_extra: dict[str, str]) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env.update({"BUSY_MACHINES_DB": str(db_path), "PYTHONUNBUFFERED": "1", **env_extra})
    log = open(db_path.with_suffix(".server.log"), "wb")  # noqa: SIM115 (lives as long as the server)
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", str(port),
         "--log-level", "warning"],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
    )


def wait_healthy(url: str, timeout_s: float = 60.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if httpx.get(f"{url}/api/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"server at {url} did not become healthy in {timeout_s}s")


async def watch(ws_url: str, verbose: bool) -> tuple[collections.Counter[str], dict[str, Any] | None, float]:
    counts: collections.Counter[str] = collections.Counter()
    finished: dict[str, Any] | None = None
    t0 = time.time()
    async with websockets.connect(ws_url, max_size=None, open_timeout=30) as ws:
        async for raw in ws:
            ev = json.loads(raw)
            counts[ev["type"]] += 1
            if verbose and ev["type"] in ("experiment_finished", "new_best", "race_finished"):
                arm = ev.get("arm", "")
                print(f"  t={time.time() - t0:6.1f}s {ev['type']:<20} {arm} "
                      f"{ev.get('status', '')} {ev.get('score', '')}")
            if ev["type"] == "race_finished":
                finished = ev
                break
    return counts, finished, time.time() - t0


def fmt(v: Any, nd: int = 1) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="use a running server instead of starting one")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--target", default="rl", help="rl (default) or cifar (synthetic tiny data on CPU)")
    ap.add_argument("--budget", type=float, default=150.0, help="race wall-clock per arm, s")
    ap.add_argument("--run-budget", type=int, default=12, help="per-run training budget, s")
    ap.add_argument("--slots", type=int, default=2, help="CPU slots per arm")
    ap.add_argument("--opus-latency", type=float, default=8.0, help="simulated s per call, Opus arm")
    ap.add_argument("--mubit-latency", type=float, default=0.3, help="simulated s per call, Mubit arm")
    ap.add_argument("--no-optuna", action="store_true", help="skip the Optuna baseline arm")
    ap.add_argument("--pipelining", action="store_true")
    ap.add_argument("--trap-rate", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="where to save the JSONL export")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="busy_mock_race_"))
    server = None
    url = args.url
    if url is None:
        env_extra = {"BUSY_CIFAR_SYNTHETIC": "1"} if args.target == "cifar" else {}
        server = start_server(args.port, work / "race.db", env_extra)
        url = f"http://127.0.0.1:{args.port}"
    try:
        wait_healthy(url)
        arms = ["opus", "mubit"] + ([] if args.no_optuna else ["optuna"])
        body: dict[str, Any] = {
            "name": f"mock race {args.target}", "target": args.target, "arms": arms,
            "budget_s": args.budget, "gpus_per_arm": args.slots, "seed": args.seed,
            "run_budget_s": args.run_budget, "pipelining": args.pipelining,
            "mode": "dev", "llm": "mock", "runner": "local",
            "mock_latency_s": {"opus": args.opus_latency, "mubit": args.mubit_latency},
        }
        if args.trap_rate is not None:
            body["mock_trap_rate"] = args.trap_rate
        r = httpx.post(f"{url}/api/races", json=body, timeout=30)
        r.raise_for_status()
        race_id = r.json()["id"]
        print(f"race {race_id} created ({args.target}, {args.slots} slots/arm, {args.budget:.0f}s, "
              f"mock latency opus {args.opus_latency}s / mubit {args.mubit_latency}s)")
        ws_url = url.replace("http", "ws", 1) + f"/ws/races/{race_id}"

        async def run() -> tuple[collections.Counter[str], dict[str, Any] | None, float]:
            task = asyncio.create_task(watch(ws_url, args.verbose))
            await asyncio.sleep(0.5)  # subscribe first (the bus also replays history)
            s = httpx.post(f"{url}/api/races/{race_id}/start", timeout=30)
            s.raise_for_status()
            return await task

        counts, finished, wall = asyncio.run(run())
        state = httpx.get(f"{url}/api/races/{race_id}", timeout=30).json()
        exp = httpx.get(f"{url}/api/races/{race_id}/export.jsonl", timeout=60)
        exp.raise_for_status()
        out = args.out or (work / f"race-{race_id}.jsonl")
        out.write_bytes(exp.content)
        tables = collections.Counter(json.loads(line)["table"] for line in exp.text.splitlines() if line.strip())

        print(f"\nWS events ({wall:.1f}s): {dict(counts)}")
        print(f"race status: {state['race']['status']}  finish: {finished and finished.get('reason')}")
        print("\nMOCK NUMBERS (simulated LLM latency, laptop CPU, $0) - not a benchmark result")
        hdr = f"{'arm':<7}{'idle %':>8}{'finished':>10}{'failed':>8}{'llm calls':>11}{'p50 ms':>9}" \
              f"{'best (heldout)':>16}{'t->target s':>13}"
        print(hdr)
        print("-" * len(hdr))
        for a in state["arms"]:
            s = a.get("summary") or {}
            print(f"{a['kind']:<7}{fmt(s.get('gpu_idle_pct')):>8}{fmt(s.get('experiments_finished')):>10}"
                  f"{fmt(s.get('experiments_failed')):>8}{fmt(s.get('llm_calls')):>11}"
                  f"{fmt(s.get('llm_p50_latency_ms'), 0):>9}{fmt(s.get('best_score'), 2):>16}"
                  f"{fmt(s.get('time_to_target_s')):>13}")
        print(f"\nexport: {out}  rows by table: {dict(tables)}")
        print(json.dumps({"race_id": race_id, "arms": [{"kind": a["kind"], "summary": a.get("summary"),
                                                        "stop_reason": a.get("stop_reason")} for a in state["arms"]]},
                         default=str))
        return 0 if state["race"]["status"] == "finished" else 1
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
