"""Fetch the YAHPO LCBench surrogate, scan tasks for headroom, and calibrate the lcbench goal.

Usage (repo root; onnxruntime is not in the lock file, so add it as an overlay):
    python -m uv run --with onnxruntime==1.29.0 python -m workers.targets.data.prep_lcbench fetch
    python -m uv run --with onnxruntime==1.29.0 python -m workers.targets.data.prep_lcbench scan
    python -m uv run --with onnxruntime==1.29.0 python -m workers.targets.data.prep_lcbench calibrate \
        --out data/calibration/lcbench_calibration.json

fetch      downloads the pinned files (``lcbench.YAHPO_DATA_COMMIT``) into an empty cache dir and
           checks their sha256 against ``lcbench.FILES``.
scan       per OpenML task: predicted test balanced accuracy of the YAHPO default config and of
           N random configs (52 epochs), Spearman(val, test), predicted runtime. Picks nothing;
           ``lcbench.TASK_ID`` is fixed in code.
calibrate  for the fixed task, through ``lcbench.evaluate`` with the real per-run budget:
           the starting (default) config, N random configs from the target's own config space
           (trap choice excluded) and the best of them ("best known"). The goal rule is fixed
           here, before any race: goal = default + 0.9 x (best_known - default), rounded DOWN
           to 0.01 (a held-out score 90% of the way from the starting code to the best of 500
           random configs).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import urllib.request
from pathlib import Path
from typing import Any

from workers.targets import lcbench
from workers.targets.lcbench import (
    FILES,
    YAHPO_DATA_COMMIT,
    YAHPO_DATA_REPO,
    Surrogate,
    check_files,
    data_dir,
    evaluate,
    sha256_file,
)

GOAL_FRACTION = 0.9
N_RANDOM = 500


def fetch(directory: Path) -> dict[str, bool]:
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        dest = directory / name
        if dest.exists() and sha256_file(dest) == digest:
            continue
        url = f"https://raw.githubusercontent.com/{YAHPO_DATA_REPO}/{YAHPO_DATA_COMMIT}/lcbench/{name}"
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:  # noqa: S310 (fixed https URL)
            f.write(r.read())
        if sha256_file(tmp) != digest:
            tmp.unlink()
            raise RuntimeError(f"{name}: sha256 mismatch (expected {digest})")
        tmp.replace(dest)
    return check_files(directory)


def _spearman(a: list[float], b: list[float]) -> float:
    def ranks(x: list[float]) -> list[float]:
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for k, i in enumerate(order):
            r[i] = float(k)
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else float("nan")


def _random_hp(rng: random.Random, epochs: int | None = None) -> dict[str, Any]:
    cfg = lcbench.SPEC.random_config(rng)
    while lcbench.SPEC.is_trap(cfg):
        cfg["checkpoint_selection"] = rng.choice(["last_epoch", "best_validation"])
    if epochs is not None:
        cfg["epochs"] = epochs
    return lcbench.SPEC.validate_config(cfg)


def scan(n: int, seed: int) -> list[dict[str, Any]]:
    enc = json.loads((data_dir() / "encoding.json").read_text(encoding="utf-8"))["OpenML_task_id"]
    tasks = json.loads((data_dir() / "config_space.json").read_text(encoding="utf-8"))["hyperparameters"][0]["choices"]
    rng = random.Random(seed)
    configs = [_random_hp(rng, epochs=lcbench.MAX_EPOCH) for _ in range(n)]
    default = {**lcbench.SPEC.default_config(), "epochs": lcbench.MAX_EPOCH}
    out = []
    for t in tasks:
        if t not in enc:
            continue
        s = Surrogate(task_id=t)
        ys = s.predict([{**{k: c[k] for k in lcbench.CONT_NAMES if k != "epoch"}, "epoch": c["epochs"]}
                        for c in [default, *configs]])
        val = [float(r[3]) for r in ys[1:]]
        test = [float(r[5]) for r in ys[1:]]
        tm = sorted(float(r[0]) for r in ys[1:])
        qs = statistics.quantiles(test, n=10)
        out.append({"task": t, "default_test_bacc": float(ys[0][5]), "p10": qs[0], "p50": statistics.median(test),
                    "p90": qs[-1], "max": max(test), "spearman_val_test": _spearman(val, test),
                    "median_time_s": tm[len(tm) // 2], "p90_time_s": tm[int(0.9 * len(tm))],
                    "headroom": max(test) - float(ys[0][5])})
    return sorted(out, key=lambda r: -r["headroom"])


def calibrate(n: int, seed: int) -> dict[str, Any]:
    budget = float(lcbench.SPEC.budget_s)
    default_cfg = lcbench.SPEC.default_config()
    d = evaluate(default_cfg, budget)
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        cfg = _random_hp(rng)
        try:
            r = evaluate(cfg, budget)
        except TimeoutError:
            rows.append({"config": cfg, "failed": "timeout"})
            continue
        rows.append({"config": cfg, "val": r["score"], "heldout": r["heldout_score"], "time_s": r["train_time_s"],
                     "budget_hit": r["budget_hit"]})
    ok = [r for r in rows if "heldout" in r]
    held = sorted(r["heldout"] for r in ok)
    best = max(ok, key=lambda r: r["val"])          # what an agent selecting on validation would keep
    best_h = max(ok, key=lambda r: r["heldout"])    # oracle best held-out among the 500
    best_known = best_h["heldout"]
    goal = math.floor((d["heldout_score"] + GOAL_FRACTION * (best_known - d["heldout_score"])) * 100) / 100
    qs = statistics.quantiles(held, n=100)
    return {
        "target": "lcbench", "task_id": lcbench.TASK_ID, "yahpo_data_commit": YAHPO_DATA_COMMIT,
        "budget_s": budget, "n_random": n, "seed": seed,
        "default": {"config": default_cfg, "val": d["score"], "heldout": d["heldout_score"],
                    "time_s": d["train_time_s"]},
        "random_heldout": {"p10": qs[9], "p25": qs[24], "p50": statistics.median(held), "p75": qs[74],
                           "p90": qs[89], "p95": qs[94], "p99": qs[98], "max": held[-1],
                           "n_ok": len(ok), "n_timeout": len(rows) - len(ok),
                           "share_at_or_above_goal": sum(h >= goal for h in held) / len(held),
                           "median_time_s": statistics.median(r["time_s"] for r in ok),
                           "share_budget_hit": sum(bool(r["budget_hit"]) for r in ok) / len(ok)},
        "best_known": {"rule": f"best held-out of {n} random configs (seed {seed})", "heldout": best_known,
                       "val": best_h["val"], "config": best_h["config"], "time_s": best_h["time_s"]},
        "best_by_val": {"heldout": best["heldout"], "val": best["val"], "config": best["config"]},
        "goal": {"rule": f"floor_0.01(default + {GOAL_FRACTION} x (best_known - default))", "value": goal},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--dir", type=Path, default=None)
    s = sub.add_parser("scan")
    s.add_argument("--n", type=int, default=N_RANDOM)
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("calibrate")
    c.add_argument("--n", type=int, default=N_RANDOM)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "fetch":
        print(json.dumps({"dir": str(a.dir or data_dir()), "ok": fetch(a.dir or data_dir())}, indent=2))
    elif a.cmd == "scan":
        rows = scan(a.n, a.seed)
        print(f"{'task':<8}{'dflt':>7}{'p10':>7}{'p50':>7}{'p90':>7}{'max':>7}{'spear':>7}{'med s':>8}{'p90 s':>8}")
        for r in rows:
            print(f"{r['task']:<8}{r['default_test_bacc']:7.3f}{r['p10']:7.3f}{r['p50']:7.3f}{r['p90']:7.3f}"
                  f"{r['max']:7.3f}{r['spearman_val_test']:7.3f}{r['median_time_s']:8.0f}{r['p90_time_s']:8.0f}")
    else:
        res = calibrate(a.n, a.seed)
        text = json.dumps(res, indent=2)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(text + "\n", encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
