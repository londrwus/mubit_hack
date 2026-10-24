"""Micro-benchmark: TPE ask latency and event-loop lag, with and without a second TPE study
asking next to it, in-thread (the old layout) vs one child process per study (the fix).

What it reproduces (review finding, R67): the Optuna arm's study and a Busy Machines arm's
backfill sampler ask TPE in worker threads of the race process, while the event loop times
every arm's runs. Here:

- the "Optuna arm" asks ``--k-optuna`` configs per propose (``OptunaResearcher._ask_batch``),
  ``--repeats`` times, and is told a score for each;
- with contention, a "backfill sampler" (``BackfillSampler``) asks ``--k-sampler`` configs in a
  loop in another worker thread, the way ``ArmRunner._backfill_loop`` does;
- an asyncio ticker sleeps ``--tick-s`` and records how late it wakes (event-loop lag), like
  ``ArmRunner._monitor``.

Both studies start with ``--trials`` completed trials (random configs in the target's space,
added the way other runs are added), since TPE's ask cost grows with the study. The same seeds,
TPE settings and operations are used in every scenario; only the isolation and the contention
change. No LLM, no Modal, no network: CPU only.

CPU-heavy (a few cores for up to a minute): do NOT run it on the race host while races run, it
would perturb them. ``--quick`` is a smoke run (small study, 2 repeats).

Usage:
    python -m uv run python scripts/bench_proposer_contention.py                    # rl, 600 trials
    python -m uv run python scripts/bench_proposer_contention.py --target lcbench --trials 1200
    python -m uv run python scripts/bench_proposer_contention.py --quick --json out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agent.optuna_arm import BackfillSampler, OptunaResearcher  # noqa: E402
from backend.core.types import ExperimentStatus  # noqa: E402
from backend.scheduler import ExperimentSnapshot  # noqa: E402
from workers.targets.base import TargetSpec, get_target  # noqa: E402


def random_config(spec: TargetSpec, rng: random.Random) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, p in spec.config_space.items():
        if p.type == "bool":
            out[name] = rng.random() < 0.5
        elif p.type == "categorical":
            out[name] = rng.choice(list(p.choices or []))
        elif p.type == "int":
            lo, hi = int(p.low), int(p.high)  # type: ignore[arg-type]
            out[name] = int(round(math.exp(rng.uniform(math.log(lo), math.log(hi))))) if p.log else rng.randint(lo, hi)
            out[name] = min(hi, max(lo, out[name]))
        else:
            lo, hi = float(p.low), float(p.high)  # type: ignore[arg-type]
            out[name] = math.exp(rng.uniform(math.log(lo), math.log(hi))) if p.log else rng.uniform(lo, hi)
    return spec.validate_config(out)


def fake_score(spec: TargetSpec, config: dict[str, Any]) -> float:
    """Any deterministic function of the config (TPE only needs values to rank)."""
    return random.Random(json.dumps(config, sort_keys=True, default=str)).random()


def preload(spec: TargetSpec, optuna_r: OptunaResearcher, sampler: BackfillSampler, n: int, seed: int) -> None:
    rng = random.Random(seed)
    for i in range(n):
        c = random_config(spec, rng)
        optuna_r.observe(ExperimentSnapshot(id=f"pre-{i}", config=c, status=ExperimentStatus.succeeded,
                                            score=fake_score(spec, c)))
        sampler.tell(f"pre-{i}", c, fake_score(spec, c))
    sampler.apply_tells()
    optuna_r._ask_batch(1)   # flushes the queued adds in process mode; one ask in both modes (same state)


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


async def scenario(spec: TargetSpec, *, isolation: str, contended: bool, trials: int, k_optuna: int,
                   k_sampler: int, repeats: int, tick_s: float, seed: int) -> dict[str, Any]:
    optuna_r = OptunaResearcher(spec, seed, isolation=isolation)  # type: ignore[arg-type]
    sampler = BackfillSampler(spec, seed + 1, isolation=isolation)  # type: ignore[arg-type]
    try:
        t = time.perf_counter()
        optuna_r.wait_ready()
        sampler.wait_ready()
        start_s = time.perf_counter() - t
        preload(spec, optuna_r, sampler, trials, seed)
        stop = threading.Event()
        sampler_asks: list[float] = []
        lags: list[float] = []

        def sampler_loop() -> None:
            while not stop.is_set():
                t0 = time.perf_counter()
                batch = sampler.ask_batch(k_sampler)
                sampler_asks.append(time.perf_counter() - t0)
                for p in batch:
                    sampler.tell(p.experiment_id, p.config, fake_score(spec, p.config))

        async def ticker() -> None:
            while not stop.is_set():
                t0 = time.perf_counter()
                await asyncio.sleep(tick_s)
                lags.append(max(0.0, time.perf_counter() - t0 - tick_s))

        tick = asyncio.create_task(ticker())
        samp = asyncio.create_task(asyncio.to_thread(sampler_loop)) if contended else None
        if samp is not None:
            await asyncio.sleep(0.2)   # let the sampler get going
        asks: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            props = await asyncio.to_thread(optuna_r._ask_batch, k_optuna)
            asks.append(time.perf_counter() - t0)
            for p in props:
                optuna_r.observe(ExperimentSnapshot(id=p.experiment_id, config=p.config,
                                                    status=ExperimentStatus.succeeded, score=fake_score(spec, p.config)))
        stop.set()
        if samp is not None:
            await samp
        await tick
        return {
            "isolation": isolation, "contended": contended, "trials": trials, "child_start_s": round(start_s, 3),
            "optuna_ask_p50_s": statistics.median(asks), "optuna_ask_max_s": max(asks), "optuna_asks": len(asks),
            "sampler_asks": len(sampler_asks),
            "sampler_ask_p50_s": statistics.median(sampler_asks) if sampler_asks else None,
            "loop_lag_p50_ms": 1000 * (statistics.median(lags) if lags else 0.0),
            "loop_lag_p95_ms": 1000 * (_pct(lags, 0.95) or 0.0), "loop_lag_max_ms": 1000 * (max(lags) if lags else 0.0),
        }
    finally:
        optuna_r.close()
        sampler.close()


def render(rows: list[dict[str, Any]]) -> str:
    out = ["| Isolation | Sampler asking next to it | Optuna ask p50 | Optuna ask max | Sampler asks | "
           "Event-loop lag p50 / p95 / max |", "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['isolation']} | {'yes' if r['contended'] else 'no'} | {r['optuna_ask_p50_s']:.3f} s | "
                   f"{r['optuna_ask_max_s']:.3f} s | {r['sampler_asks']} | {r['loop_lag_p50_ms']:.1f} / "
                   f"{r['loop_lag_p95_ms']:.1f} / {r['loop_lag_max_ms']:.1f} ms |")
    base = {(r["isolation"], r["contended"]): r["optuna_ask_p50_s"] for r in rows}
    for iso in ("thread", "process"):
        if (iso, False) in base and (iso, True) in base and base[(iso, False)] > 0:
            out.append(f"\n{iso}: contention slows the Optuna ask {base[(iso, True)] / base[(iso, False)]:.2f}x (p50)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", default="rl", help="registered target whose config space is used (default rl)")
    ap.add_argument("--trials", type=int, default=600, help="completed trials in each study before timing")
    ap.add_argument("--k-optuna", type=int, default=8)
    ap.add_argument("--k-sampler", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--tick-s", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--isolation", choices=["thread", "process", "both"], default="both")
    ap.add_argument("--quick", action="store_true", help="smoke run: 60 trials, 2 repeats")
    ap.add_argument("--json", help="also write the rows here")
    args = ap.parse_args(argv)
    if args.quick:
        args.trials, args.repeats = min(args.trials, 60), min(args.repeats, 2)
    spec = get_target(args.target).spec
    modes = ["thread", "process"] if args.isolation == "both" else [args.isolation]
    rows = []
    for iso in modes:
        for contended in (False, True):
            r = asyncio.run(scenario(spec, isolation=iso, contended=contended, trials=args.trials,
                                     k_optuna=args.k_optuna, k_sampler=args.k_sampler, repeats=args.repeats,
                                     tick_s=args.tick_s, seed=args.seed))
            rows.append(r)
            print(f"done: {iso}, contended={contended}: Optuna ask p50 {r['optuna_ask_p50_s']:.3f} s", file=sys.stderr)
    print(f"target {args.target}, {args.trials} trials per study, Optuna asks {args.k_optuna}/propose, "
          f"sampler asks {args.k_sampler}/batch, {args.repeats} repeats\n")
    print(render(rows))
    if args.json:
        Path(args.json).write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
