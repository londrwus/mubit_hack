"""Project GPUs waiting (RL wide, C=1, no pipelining) for each lean-propose variant ($0, PROJECTION).

Feeds ``data/propose_lean/calls.jsonl`` (scripts/probe_propose_lean.py) into the unmodified R52
simulator (scripts/sim_planner_concurrency.py: R43 event model, 30 slots, 15 s runs, 900 s, K<=8,
judge calls as in R48). Per variant:
- latency(K) = a + b*K, OLS over the variant's measured K=8 and K=2 replay calls; noise by
  resampling the variant's own K=8 latencies scaled to K with that shape;
- "abs": those measured latencies as they are (provider speed at replay time);
- "rel": R48's measured K=8 latencies x (variant K=8 p50 / v1_med K=8 p50 at replay time), so a
  provider-speed difference between R48 and the replay cancels out (primary);
- LLM $ per proposal: the variant's measured output tokens per valid proposal (input as in R48).
Validation anchor: the simulator's own R48 latency model (R52: 67.5% vs 69.0% measured).

Usage: PYTHONUTF8=1 python -m uv run python scripts/sim_propose_lean.py [--seeds 20]
Writes data/propose_lean/sim.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
_s = importlib.util.spec_from_file_location("sim_pc", REPO / "scripts" / "sim_planner_concurrency.py")
sim = importlib.util.module_from_spec(_s)  # type: ignore[arg-type]
assert _s and _s.loader
sys.modules["sim_pc"] = sim
_s.loader.exec_module(sim)

CALLS = REPO / "data" / "propose_lean" / "calls.jsonl"
OUT = REPO / "data" / "propose_lean" / "sim.json"


def ols(x: list[float], y: list[float]) -> tuple[float, float]:
    mx, my = st.fmean(x), st.fmean(y)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    return my - b * mx, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    rows = [json.loads(line) for line in CALLS.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if "output" in r and r.get("n_valid", 0) > 0]
    d = sim.load_r48()
    pool = [r for r in sim.gate.load_regime("rl15", sim.gate.REGIMES["rl15"]) if r.candidate]
    r48_lat8 = d["lat8"]
    base_models = sim.latency_models

    res: dict[str, Any] = {"seeds": args.seeds, "setting": "C=1, pipelining off, share (today's scheduler)",
                           "r48_lat8_p50_s": st.median(r48_lat8), "r48_out_per_prop": d["out_per_prop"], "variants": {}}
    anchor = sim.run_setting(pool, d, args.seeds, C=1, pipelining=False, policy="share", lat="r48")
    res["anchor_r48_model"] = {k: anchor[k] for k in ("waiting", "waiting_sd", "experiments", "llm_usd")}
    print(f"anchor (sim's own R48 model): waiting {anchor['waiting']*100:.1f}% exps {anchor['experiments']:.0f} "
          f"LLM ${anchor['llm_usd']:.2f}")

    variants = sorted({r["variant"] for r in rows})
    p50_8 = {v: st.median(r["latency_ms"] / 1000 for r in rows if r["variant"] == v and r["k"] == 8) for v in variants}
    ref = p50_8.get("v1_med")
    for v in variants:
        vr = [r for r in rows if r["variant"] == v]
        a, b = ols([float(r["k"]) for r in vr], [r["latency_ms"] / 1000 for r in vr])
        lat8 = [r["latency_ms"] / 1000 for r in vr if r["k"] == 8]
        nv = sum(r["n_valid"] for r in vr if r["k"] == 8)
        out_pp = sum(r["output"] for r in vr if r["k"] == 8) / max(1, nv)
        ratio = p50_8[v] / ref if ref else float("nan")
        shape = lambda k, a=a, b=b: max(0.05, (a + b * k) / (a + b * 8))  # noqa: E731
        models = {
            "abs": lambda rng, k, lat8=lat8, shape=shape: rng.choice(lat8) * shape(k),
            "rel": lambda rng, k, ratio=ratio, shape=shape: rng.choice(r48_lat8) * ratio * shape(k),
        }
        dv = dict(d)
        dv["out_per_prop"] = out_pp
        out: dict[str, Any] = {"fit_a_s": a, "fit_b_s_per_k": b, "k8_p50_s": p50_8[v], "ratio_vs_v1_med": ratio,
                               "out_per_prop": out_pp}
        for name, fn in models.items():
            sim.latency_models = lambda _d, fn=fn: {**base_models(_d), "lean": fn}  # type: ignore[assignment]
            s = sim.run_setting(pool, dv, args.seeds, C=1, pipelining=False, policy="share", lat="lean")
            out[name] = {k: s[k] for k in ("waiting", "waiting_sd", "waiting_min", "waiting_max", "experiments",
                                            "llm_usd", "llm_usd_per_exp", "calls_propose")}
            print(f"{v:7s} {name}: lat(K)={a:.2f}+{b:.2f}K s, K8 p50 {p50_8[v]:.2f}s (x{ratio:.2f}), out/prop "
                  f"{out_pp:.0f}: waiting {s['waiting']*100:.1f}% ± {s['waiting_sd']*100:.1f}, exps "
                  f"{s['experiments']:.0f}, LLM ${s['llm_usd']:.2f}/race, {s['llm_usd_per_exp']*1000:.2f} m$/exp")
        sim.latency_models = base_models
        res["variants"][v] = out
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
