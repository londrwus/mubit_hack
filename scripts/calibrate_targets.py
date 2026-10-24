"""Calibrate each target's goal (``target_value``) from the measured starting code.

benchmark.md: "Set the thresholds from the starting code's performance before the first
race, and don't move them afterwards." This script measures, on the deployed Modal app
(``busy-machines-runner``, never redeployed here):

- the **default config** (the starting code) x seeds 0,1,2 for each target;
- the **best config found so far** (best non-trap held-out score in the read-only collection
  DBs ``data/collection/collect_{target}.db``) x the same seeds, when one exists;

then proposes a goal per target: ``start_mean + frac * (best_mean - start_mean)`` (default
frac 0.6) on the **held-out** score (the graded number, ``backend.benchmark.metrics.quality``),
rounded to a sensible step. ``best_mean`` is the 3-seed replication of the best config, not
the single lucky collection run (that one is selected as a max over noisy runs).

Outputs (``--out-dir``, default ``data/calibration/``):
- ``calibration_v1.json``: every run (config, seed, Modal call id, status, reported and
  held-out scores, duration, est. $), per-group mean/std, the collection best, the rule used.
- ``target_values.json``: ``{"cifar": x, "lm": y, "rl": z}`` (proposed; frozen once used).

    PYTHONUTF8=1 python -m uv run --extra ml python scripts/calibrate_targets.py --yes
    python -m uv run --extra ml python scripts/calibrate_targets.py --plan     # print plan only
    python -m uv run --extra ml python scripts/calibrate_targets.py --propose-only  # recompute from JSON
    # more groups / seeds into the same file (e.g. a probe config, extra RL seeds)
    ... calibrate_targets.py --append --no-default --no-best --config-file groups.json --yes
    ... calibrate_targets.py --append --no-best --targets rl --seeds 3,4,5,6,7,8,9 --yes
    # record hand-chosen goals (with sanity numbers) next to the formula values
    ... calibrate_targets.py --propose-only --choose data/calibration/choices_v1.json

Rules it enforces:
- No LLM calls at all. Modal only through ``workers.runners.modal_runner.ModalRunner``
  (deployed=True: looks the functions up by name, never deploys).
- At most ``--max-l4`` (default 3) concurrent L4 runs; RL on CPU.
- A $ cap (``--usd-cap``, default 0.50): a run launches only if spent + in-flight
  worst-case + its own worst case stays under the cap.
- Collection DBs are opened ``mode=ro`` (URI); if locked/unavailable the best-config
  group is skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import math
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workers.targets.base import get_target  # noqa: E402

TARGETS: tuple[str, ...] = ("cifar", "lm", "rl")
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
COLLECTION_DIR = ROOT / "data" / "collection"
OUT_DIR = ROOT / "data" / "calibration"
# Rounding step of the proposed goal, in the target's metric units.
ROUND_STEP: dict[str, float] = {"cifar": 0.001, "lm": 0.01, "rl": 5.0}
# Worst-case seconds billed per run beyond budget_s (container start + eval + grace).
OVERHEAD_S = 90.0


# --------------------------------------------------------------------------- collection DB


def _ro_connect(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def collection_best(target: str, db_dir: Path = COLLECTION_DIR) -> dict[str, Any] | None:
    """Best non-trap succeeded experiment by held-out score, plus the time the collection
    race first reached each held-out level (for the sanity check). None if unavailable."""
    path = db_dir / f"collect_{target}.db"
    if not path.exists():
        return None
    spec = get_target(target).spec
    try:
        conn = _ro_connect(path)
        try:
            rows = conn.execute(
                "SELECT e.id, e.race_id, e.config_json, e.metrics_json, e.score, e.run_ref, e.seed, "
                "e.ended_at, e.is_trap FROM experiments e WHERE e.status = 'succeeded'"
            ).fetchall()
            race_start = {
                r["race_id"]: r["t0"] for r in conn.execute(
                    "SELECT race_id, MIN(proposed_at) AS t0 FROM experiments GROUP BY race_id")
            }
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f"[calibrate] {target}: collection DB unavailable ({exc}); skipping best config", flush=True)
        return None
    cands: list[dict[str, Any]] = []
    for r in rows:
        if r["is_trap"]:
            continue
        try:
            cfg = json.loads(r["config_json"] or "{}")
            metrics = json.loads(r["metrics_json"] or "{}")
        except json.JSONDecodeError:
            continue
        if spec.is_trap(cfg):
            continue
        h = metrics.get("heldout_score")
        if h is None or not math.isfinite(float(h)):
            continue
        t0 = race_start.get(r["race_id"])
        cands.append({
            "experiment_id": r["id"], "race_id": r["race_id"], "config": cfg, "seed": r["seed"],
            "heldout_score": float(h), "reported_score": r["score"], "run_ref": r["run_ref"],
            "minutes_into_race": (float(r["ended_at"]) - float(t0)) / 60 if (t0 and r["ended_at"]) else None,
        })
    if not cands:
        return None
    sign = 1.0 if spec.direction == "max" else -1.0
    best = max(cands, key=lambda c: sign * c["heldout_score"])
    return {"db": str(path.relative_to(ROOT)), "n_succeeded_non_trap": len(cands), "best": best,
            "curve": sorted(({"minutes": c["minutes_into_race"], "heldout": c["heldout_score"]}
                             for c in cands if c["minutes_into_race"] is not None),
                            key=lambda c: c["minutes"])}


def first_reach_minutes(curve: list[dict[str, Any]], value: float, direction: str) -> float | None:
    for c in curve:
        if (c["heldout"] >= value) if direction == "max" else (c["heldout"] <= value):
            return round(float(c["minutes"]), 2)
    return None


# --------------------------------------------------------------------------- stats / proposal


def stats(xs: list[float]) -> dict[str, Any]:
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not xs:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "values": []}
    return {"n": len(xs), "mean": statistics.fmean(xs), "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs), "values": xs}


def round_toward_start(value: float, step: float, direction: str) -> float:
    """Round the goal to ``step``, toward the starting score (i.e. never harder than the
    formula asks: floor for max metrics, ceil for min metrics)."""
    q = value / step
    q = math.floor(q + 1e-9) if direction == "max" else math.ceil(q - 1e-9)
    return round(q * step, 6)


def propose(target: str, groups: dict[str, dict[str, Any]], coll: dict[str, Any] | None,
            frac: float, min_runs: int = 3) -> dict[str, Any]:
    """Goal = start_mean + frac * (best_mean - start_mean), held-out units. best_mean is the
    best replicated (>= ``min_runs`` successful seeds) non-default group, never a single run."""
    spec = get_target(target).spec
    start = (groups.get("default") or {}).get("heldout")
    sign = 1.0 if spec.direction == "max" else -1.0
    out: dict[str, Any] = {"metric": spec.metric, "direction": spec.direction, "graded_on": "heldout_score",
                           "frac_of_gap": frac, "current_spec_value": spec.target_value}
    if coll is not None:
        out["collection_best_single"] = {k: coll["best"].get(k) for k in
                                         ("heldout_score", "reported_score", "experiment_id", "run_ref",
                                          "minutes_into_race")}
    if not start or start["mean"] is None:
        out.update(value=None, reason="no successful default-config runs")
        return out
    s_mean, s_std = start["mean"], start["std"]
    reps = {lab: g for lab, g in groups.items()
            if lab != "default" and g["heldout"]["n"] >= min_runs and g["heldout"]["mean"] is not None}
    out["replicated_groups"] = {lab: {"heldout_mean": g["heldout"]["mean"], "heldout_std": g["heldout"]["std"],
                                      "n": g["heldout"]["n"], "reported_mean": g["reported"]["mean"]}
                                for lab, g in reps.items()}
    out.update(start_mean=s_mean, start_std=s_std, start_n=start["n"])
    if not reps:
        out.update(value=None, reason="no replicated non-default config yet")
        return out
    best_lab = max(reps, key=lambda lab: sign * reps[lab]["heldout"]["mean"])
    best_ref = reps[best_lab]["heldout"]["mean"]
    raw = s_mean + frac * (best_ref - s_mean)
    value = round_toward_start(raw, ROUND_STEP[target], spec.direction)
    out.update(best_ref=best_ref, best_ref_label=best_lab, best_ref_std=reps[best_lab]["heldout"]["std"],
               gap=best_ref - s_mean, gap_in_start_stds=(abs(best_ref - s_mean) / s_std) if s_std else None,
               raw=raw, rounding=f"step {ROUND_STEP[target]}, toward the starting score")
    if sign * (value - s_mean) <= 0:
        out.update(value=None, rounded_candidate=value,
                   reason="no replicated config beats the starting code by more than the rounding step; the "
                          "formula would give a goal the starting code already reaches")
        return out
    out.update(value=value, goal_in_start_stds=(abs(value - s_mean) / s_std) if s_std else None)
    # What the agent-visible (reported) metric reads at the goal, using the best group's offset.
    out["reported_units_equivalent"] = value + ((reps[best_lab]["reported"]["mean"] or best_ref) - best_ref)
    if coll is not None:
        out["collection_first_reach_minutes"] = first_reach_minutes(coll["curve"], value, spec.direction)
        out["collection_note"] = ("minutes into the recorded Opus collection race at which a non-trap run first "
                                  "reached this held-out value (sanity check only; dev race, not a benchmark)")
    return out


def summarize(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for t in TARGETS:
        labels = sorted({r["label"] for r in runs if r["target"] == t}, key=lambda lab: (lab != "default", lab))
        for label in labels:
            rs = [r for r in runs if r["target"] == t and r["label"] == label]
            ok = [r for r in rs if r["status"] == "succeeded"]
            out.setdefault(t, {})[label] = {
                "runs": len(rs), "succeeded": len(ok), "config": rs[0]["config"],
                "heldout": stats([r["heldout_score"] for r in ok]),
                "reported": stats([r["reported_score"] for r in ok]),
                "duration_s": stats([r["duration_s"] for r in rs]),
                "est_usd": sum(r.get("est_usd") or 0.0 for r in rs),
                "run_refs": [r.get("run_ref") for r in rs],
            }
    return out


# --------------------------------------------------------------------------- runs


class Budget:
    def __init__(self, cap: float) -> None:
        self.cap = cap
        self.spent = 0.0
        self.reserved = 0.0

    def try_reserve(self, usd: float) -> bool:
        if self.spent + self.reserved + usd > self.cap + 1e-12:
            return False
        self.reserved += usd
        return True

    def settle(self, reserved: float, actual: float) -> None:
        self.reserved -= reserved
        self.spent += actual


async def run_one(runner: Any, sems: dict[str, asyncio.Semaphore], budget: Budget, target: str, label: str,
                  config: dict[str, Any], seed: int, record: list[dict[str, Any]], save: Any) -> None:
    spec = get_target(target).spec
    hw = spec.hardware
    worst = (spec.budget_s + OVERHEAD_S) * runner.price_per_second(hw)
    row: dict[str, Any] = {"target": target, "label": label, "seed": seed, "hardware": hw,
                           "budget_s": spec.budget_s, "config": config}
    async with sems[hw]:
        if not budget.try_reserve(worst):
            row.update(status="skipped_budget", est_usd=0.0, duration_s=0.0, metrics={}, heldout_score=None,
                       reported_score=None, run_ref=None)
            print(f"[calibrate] {target}/{label}/seed{seed}: SKIPPED (would exceed ${budget.cap:.2f})", flush=True)
            record.append(row)
            save()
            return
        t0 = time.time()
        actual = 0.0
        try:
            h = await runner.launch(target, config, seed, spec.budget_s, lambda _p: None)
            print(f"[calibrate] {target}/{label}/seed{seed}: launched {h.run_ref}", flush=True)
            r = await runner.wait(h)
            actual = r.duration_s * runner.price_per_second(hw)
            m = dict(r.metrics)
            row.update(
                run_ref=h.run_ref, status=r.status.value, error=r.error, duration_s=r.duration_s,
                est_usd=actual, metrics=m, reported_score=m.get("reported_score", r.score),
                heldout_score=m.get("heldout_score"), score=r.score, is_trap=r.is_trap,
                logs_tail=r.logs_tail[-6:], launched_at=h.started_at, ended_at=r.ended_at,
            )
        except Exception as exc:  # noqa: BLE001 (record and keep going)
            actual = (time.time() - t0) * runner.price_per_second(hw)
            row.update(status="error", error=f"{type(exc).__name__}: {exc}", est_usd=actual,
                       duration_s=time.time() - t0, metrics={}, heldout_score=None, reported_score=None,
                       run_ref=row.get("run_ref"))
        finally:
            budget.settle(worst, actual)
        print(f"[calibrate] {target}/{label}/seed{seed}: {row['status']} heldout={row.get('heldout_score')} "
              f"reported={row.get('reported_score')} {row['duration_s']:.1f}s ~${row['est_usd']:.4f} "
              f"(spent ${budget.spent:.3f}) {row.get('error') or ''}", flush=True)
        record.append(row)
        save()


def plan(targets: list[str], seeds: list[int], with_best: bool, price: dict[str, float]) -> float:
    total = 0.0
    for t in targets:
        spec = get_target(t).spec
        n = len(seeds) * (2 if with_best else 1)
        usd = n * (spec.budget_s + OVERHEAD_S) * price[spec.hardware]
        total += usd
        print(f"[plan] {t}: {n} runs x {spec.budget_s}s on {spec.hardware}, worst case ~${usd:.3f}")
    print(f"[plan] worst case total ~${total:.3f}")
    return total


async def main_async(args: argparse.Namespace) -> int:
    from workers.runners.modal_runner import ModalRunner

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_path = out_dir / "calibration_v1.json"
    targets = [t for t in args.targets.split(",") if t]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    extra_groups: list[dict[str, Any]] = (
        json.loads(Path(args.config_file).read_text(encoding="utf-8")) if args.config_file else [])
    runner = ModalRunner({"L4": args.max_l4, "CPU": args.max_cpu}, deployed=True)
    price = {hw: runner.price_per_second(hw) for hw in ("L4", "CPU")}
    worst = 0.0 if (args.no_default and args.no_best) else plan(targets, seeds, not args.no_best, price)
    for g in extra_groups:
        spec = get_target(g["target"]).spec
        usd = len(seeds) * (spec.budget_s + OVERHEAD_S) * price[spec.hardware]
        worst += usd
        print(f"[plan] {g['target']}/{g['label']}: {len(seeds)} runs, worst case ~${usd:.3f}")
    if args.plan:
        return 0
    if worst > args.usd_cap:
        print(f"[plan] worst case exceeds the cap ${args.usd_cap:.2f}; runs past the cap will be skipped")
    if not args.yes:
        print("refusing to spend Modal $ without --yes")
        return 2

    budget = Budget(args.usd_cap)
    runs: list[dict[str, Any]] = []
    prior: dict[str, Any] | None = None
    if args.append and cal_path.exists():
        # Extra seeds into the same file: keep earlier runs, reuse each target's "best" config
        # from them (the collection best may have moved since), and charge their $ to the cap.
        prior = json.loads(cal_path.read_text(encoding="utf-8"))
        runs.extend(prior.get("runs") or [])
        budget.spent = sum(r.get("est_usd") or 0.0 for r in runs)
        print(f"[calibrate] appending to {cal_path} ({len(runs)} runs, ${budget.spent:.3f} already spent)")
    doc: dict[str, Any] = {
        "version": "calibration_v1",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "modal_app": runner.app_name, "deployed": True, "cpu_cores_assumed": runner.cpu_cores,
        "price_per_second": price, "usd_cap": args.usd_cap, "seeds": seeds,
        "rule": f"target_value = start_mean + {args.frac} * (best_mean - start_mean) on held-out score, "
                "rounded toward the start; best_mean = best 3-seed held-out mean among replicated non-default configs "
                "(best non-trap collection configs; never a single run)",
        "runs": runs,
    }
    if prior is not None:
        doc["created_at"] = prior.get("created_at", doc["created_at"])
        doc["appended_at"] = (prior.get("appended_at") or []) + [_dt.datetime.now(_dt.timezone.utc).isoformat()]
        doc["seeds"] = sorted(set(prior.get("seeds") or []) | set(seeds))

    def save() -> None:
        doc["summary"] = summarize(runs)
        doc["spent_usd_est"] = sum(r.get("est_usd") or 0.0 for r in runs)
        cal_path.write_text(json.dumps(doc, indent=2, sort_keys=False, default=str), encoding="utf-8")

    sems = {"L4": asyncio.Semaphore(args.max_l4), "CPU": asyncio.Semaphore(args.max_cpu)}

    # Phase 1: starting code (default config), plus any --config-file groups.
    tasks = [] if args.no_default else [
        run_one(runner, sems, budget, t, "default", get_target(t).spec.default_config(), s, runs, save)
        for t in targets for s in seeds]
    for g in extra_groups:
        gspec = get_target(g["target"]).spec
        cfg = gspec.validate_config(g["config"])
        if gspec.is_trap(cfg):
            raise SystemExit(f"{g['label']}: a trap config is not a calibration group")
        tasks += [run_one(runner, sems, budget, g["target"], g["label"], cfg, s, runs, save) for s in seeds]
    await asyncio.gather(*tasks)

    # Phase 2: best config so far (read the collection DBs now, after phase 1, for freshness).
    colls: dict[str, Any] = {}
    if not args.no_best:
        tasks = []
        for t in targets:
            coll = collection_best(t)
            if prior is not None and (prior.get("collection") or {}).get(t):
                coll = dict(prior["collection"][t]) | {"curve": (coll or {}).get("curve", [])}
            colls[t] = coll
            if coll is None:
                print(f"[calibrate] {t}: no non-trap succeeded run in the collection DB; no best config", flush=True)
                continue
            cfg = get_target(t).spec.validate_config(coll["best"]["config"])
            if cfg == get_target(t).spec.default_config():
                continue
            print(f"[calibrate] {t}: best collection held-out {coll['best']['heldout_score']:.4f} "
                  f"({coll['best']['experiment_id']})", flush=True)
            tasks += [run_one(runner, sems, budget, t, "best", cfg, s, runs, save) for s in seeds]
        await asyncio.gather(*tasks)
    doc["collection"] = dict((prior or {}).get("collection") or {}) | {
        t: ({k: v for k, v in c.items() if k != "curve"} | {"curve_points": len(c["curve"])}) if c else None
        for t, c in colls.items()}
    colls = {t: c for t, c in colls.items() if c} | {
        t: (doc["collection"][t] | {"curve": (collection_best(t) or {}).get("curve", [])})
        for t in TARGETS if t not in colls and doc["collection"].get(t)}
    save()
    write_proposal(doc, colls, out_dir, args.frac, _frac_for(args), _choose(args))
    return 0


def _frac_for(args: argparse.Namespace) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in (args.frac_for or "").split(","):
        if item.strip():
            k, v = item.split("=")
            out[k.strip()] = float(v)
    return out


def check_choice(target: str, value: float, doc: dict[str, Any], coll: dict[str, Any] | None,
                 reason: str) -> dict[str, Any]:
    """Sanity numbers for a hand-chosen goal: where it sits vs the measured groups, and when
    the Opus collection race first reached it with a single non-trap run."""
    spec = get_target(target).spec
    hits = (lambda x: x >= value) if spec.direction == "max" else (lambda x: x <= value)
    groups = (doc.get("summary") or {}).get(target, {})
    start = (groups.get("default") or {}).get("heldout") or {}
    out: dict[str, Any] = {"value": value, "reason": reason, "graded_on": "heldout_score"}
    if start.get("mean") is not None:
        out["start_mean"], out["start_std"] = start["mean"], start["std"]
        out["goal_in_start_stds"] = abs(value - start["mean"]) / start["std"] if start["std"] else None
    out["single_run_hit_rate"] = {
        lab: f"{sum(hits(v) for v in g['heldout']['values'])}/{g['heldout']['n']}"
        for lab, g in groups.items()}
    if coll is not None:
        out["collection_first_reach_minutes"] = first_reach_minutes(coll.get("curve") or [], value, spec.direction)
        out["collection_best_single_heldout"] = coll["best"]["heldout_score"]
        out["collection_runs_reaching"] = (
            f"{sum(hits(c['heldout']) for c in coll.get('curve') or [])}/{len(coll.get('curve') or [])}")
    return out


def write_proposal(doc: dict[str, Any], colls: dict[str, Any], out_dir: Path, frac: float,
                   frac_for: dict[str, float] | None = None,
                   choose: dict[str, tuple[float, str]] | None = None) -> None:
    summary = doc.get("summary") or {}
    fracs = {t: (frac_for or {}).get(t, frac) for t in TARGETS}
    doc["fracs"] = fracs
    props = {t: propose(t, summary.get(t, {}), colls.get(t), fracs[t]) for t in TARGETS if t in summary}
    doc["proposal"] = props
    values = {t: p.get("value") for t, p in props.items()}
    if choose:
        doc["chosen"] = {t: check_choice(t, v, doc, colls.get(t), why) | {"formula_value": values.get(t)}
                         for t, (v, why) in choose.items()}
        values |= {t: v for t, (v, _why) in choose.items()}
    doc["target_values"] = values
    (out_dir / "calibration_v1.json").write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    (out_dir / "target_values.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"proposal": props, "chosen": doc.get("chosen"), "target_values": values},
                     indent=2, default=str))


def _choose(args: argparse.Namespace) -> dict[str, tuple[float, str]]:
    if not args.choose:
        return {}
    raw = json.loads(Path(args.choose).read_text(encoding="utf-8"))
    return {t: (float(c["value"]), str(c.get("reason", ""))) for t, c in raw.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--max-l4", type=int, default=3, help="max concurrent L4 runs (all targets)")
    ap.add_argument("--max-cpu", type=int, default=6)
    ap.add_argument("--usd-cap", type=float, default=0.50)
    ap.add_argument("--frac", type=float, default=0.6, help="fraction of the start->best gap the goal sits at")
    ap.add_argument("--frac-for", default="", help="per-target frac overrides, e.g. rl=0.9,cifar=0.6")
    ap.add_argument("--no-best", action="store_true", help="skip the best-collection-config group")
    ap.add_argument("--no-default", action="store_true", help="skip the default-config group (with --append)")
    ap.add_argument("--config-file", default="",
                    help='JSON list of {"target", "label", "config"} groups, each run over --seeds')
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--plan", action="store_true", help="print the plan and exit (no Modal calls)")
    ap.add_argument("--propose-only", action="store_true",
                    help="recompute the proposal from an existing calibration_v1.json (no Modal calls)")
    ap.add_argument("--append", action="store_true",
                    help="add runs (e.g. extra seeds) to an existing calibration_v1.json; the best group reuses "
                         "the best config recorded there")
    ap.add_argument("--choose", default="",
                    help='JSON file {"target": {"value": x, "reason": "..."}}: a goal chosen over the formula; '
                         "recorded with its sanity numbers next to the formula value")
    ap.add_argument("--yes", action="store_true", help="confirm Modal spend")
    args = ap.parse_args(argv)
    if args.propose_only:
        out_dir = Path(args.out_dir)
        doc = json.loads((out_dir / "calibration_v1.json").read_text(encoding="utf-8"))
        colls = {t: collection_best(t) for t in TARGETS}
        write_proposal(doc, colls, out_dir, args.frac, _frac_for(args), _choose(args))
        return 0
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
