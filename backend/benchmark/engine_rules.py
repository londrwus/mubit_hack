"""Busy Machines engine rules (D36, D40): deterministic code, no LLM call.

Two features, switched on per arm by ``RaceConfig.features_for`` (configs.py):

``judge_policy`` (SeedSpreadRule + ``combine``)
    The LLM judge (Opus, or the Mubit adapter) is consistent at spotting leakage and almost
    never confirms a real gain from race-format evidence (R34). So: an ``invalid`` from the
    LLM stays invalid; for every other claim (valid or unsure) the engine's calibrated
    seed-spread rule decides real vs noise. No usable LLM reply -> ``unsure`` (tag
    ``no_leakage_check``): the rule cannot see leakage, so it never confirms a gain alone. This is the R37 combination and the
    ``rule_calib`` arm of ``scripts/eval_noise_set.py``: the claimed gain (reported score,
    direction-aware, vs the previous best) must exceed ``2 x sigma``, where sigma is the
    run-to-run std of the held-out score at the previous best's score level, interpolated
    linearly between the calibrated starting code and the calibrated best config (clamped).
    real -> ``valid``; noise -> ``unsure``. Both the LLM verdict and the rule decision are kept
    per claim (experiment ``summary["judge_policy"]``).

    Sigma per target (``sigma_key``: the target, or ``target@Ns`` when the run budget is not
    the target's default, like ``bench.goal_key``):
    - ``rl``, ``cifar``, ``lm``: ``data/calibration/calibration_v1.json`` (labels default/best,
      exactly what R37 used);
    - ``rl@15s``: ``data/calibration/calibration_rl15_v1.json`` (default + the best group by
      mean; 15 s runs have a much smaller spread at the starting code, 19.9 vs 166.9);
    - ``lcbench``: 0. The YAHPO surrogate is deterministic per config (``seed`` is ignored;
      checked in tests/test_busy_machines.py), so seed spread is zero and any claimed gain is
      real *on the surrogate*. The val -> test transfer gap is not a seed spread and is not
      modelled here. A table whose stds are all 0 is ``degenerate``: its decisions are flagged
      (``RuleDecision.degenerate``) and excluded from real-gain statistics.

``median_stop`` (MedianStopRule)
    At each fraction ``f`` of the per-run budget (default 25% and 50%), a running run whose
    curve value (``CURVE_METRIC``: rl train_return, cifar val_acc, lm val_bpb, lcbench
    val_balanced_accuracy) is worse than the median of the completed runs' values at the same
    point is stopped, once at least ``min_completed`` completed runs reached that point. It is
    Optuna's MedianPruner rule, applied by the scheduler to every arm that has the feature.
    Time axis: ``MetricPoint.t`` (seconds since launch) for real runners; for the surrogate
    runner, predicted seconds x its runtime multiplier (the paced stream).
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.core.types import Verdict

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = REPO_ROOT / "data" / "calibration"

# The streamed metric the median rule reads, per target (mirrors
# backend.agent.prompts.build.EARLY_STOP_CURVE_METRIC; lcbench streams its validation metric).
CURVE_METRIC: dict[str, str] = {"rl": "train_return", "cifar": "val_acc", "lm": "val_bpb",
                                "lcbench": "val_balanced_accuracy"}

RULE_K_SIGMA = 2.0   # "real" = claimed gain > 2 x sigma (R34 / R37)


def curve_metric(target: str, default_metric: str) -> str:
    return CURVE_METRIC.get(target, default_metric)


def sigma_key(target: str, run_budget_s: int | float | None, default_budget_s: int | float) -> str:
    """``target`` or ``target@Ns`` (a spread is only valid at the run budget it was measured at)."""
    if run_budget_s is not None and int(run_budget_s) != int(default_budget_s):
        return f"{target}@{int(run_budget_s)}s"
    return target


# =========================================================================== judge policy


@dataclass(frozen=True)
class SigmaTable:
    """Held-out run-to-run std at two score levels (starting code, best config)."""

    key: str
    points: tuple[tuple[float, float], ...]   # (mean score, std), sorted by mean
    source: str

    @property
    def degenerate(self) -> bool:
        """Every calibrated std is 0 (lcbench: deterministic surrogate). The rule then calls any
        gain 'real', so it says nothing about real-vs-noise: decisions are flagged and kept out
        of real-gain statistics."""
        return all(float(sd) == 0.0 for _, sd in self.points)

    def sigma_at(self, prev_score: float | None) -> float:
        """Same interpolation as scripts/eval_noise_set.py ``calib_sigma``."""
        pts = self.points
        if len(pts) == 1 or prev_score is None or prev_score <= pts[0][0]:
            return float(pts[0][1])
        if prev_score >= pts[-1][0]:
            return float(pts[-1][1])
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        return float(y0 + (y1 - y0) * (prev_score - x0) / (x1 - x0))

    def manifest(self) -> dict[str, Any]:
        return {"key": self.key, "points": [list(p) for p in self.points], "source": self.source,
                "degenerate": self.degenerate}


def _points(groups: Mapping[str, Any], direction: str) -> list[tuple[float, float]]:
    """(mean, std) of the starting code and the best calibrated config. With a group named
    'best' (calibration_v1) that one is used, as R37 did; otherwise the group with the best
    held-out mean (n >= 3)."""
    def ok(g: Mapping[str, Any]) -> bool:
        h = g.get("heldout") or {}
        return (h.get("n") or 0) >= 3 and h.get("mean") is not None and h.get("std") is not None

    out: list[tuple[float, float]] = []
    if "default" in groups and ok(groups["default"]):
        out.append((float(groups["default"]["heldout"]["mean"]), float(groups["default"]["heldout"]["std"])))
    if "best" in groups and ok(groups["best"]):
        best = groups["best"]
    else:
        cands = [g for lab, g in groups.items() if lab != "default" and ok(g)]
        pick = max if direction == "max" else min
        best = pick(cands, key=lambda g: float(g["heldout"]["mean"])) if cands else None
    if best is not None:
        out.append((float(best["heldout"]["mean"]), float(best["heldout"]["std"])))
    return sorted(out, key=lambda p: p[0])


def load_sigma_tables(directory: Path | None = None,
                      directions: Mapping[str, str] | None = None) -> dict[str, SigmaTable]:
    """Every sigma table we have a calibration for (missing files are skipped)."""
    d = Path(directory) if directory is not None else CALIBRATION_DIR
    directions = dict(directions or {"rl": "max", "cifar": "max", "lm": "min"})
    out: dict[str, SigmaTable] = {}
    v1 = d / "calibration_v1.json"
    if v1.exists():
        summ = (json.loads(v1.read_text(encoding="utf-8")).get("summary") or {})
        for target, groups in summ.items():
            pts = _points(groups, directions.get(target, "max"))
            if pts:
                out[target] = SigmaTable(target, tuple(pts), f"{v1.name} summary.{target} (default, best)")
    rl15 = d / "calibration_rl15_v1.json"
    if rl15.exists():
        raw = json.loads(rl15.read_text(encoding="utf-8"))
        key = str(raw.get("goal_key") or "rl@15s")
        pts = _points(raw.get("summary") or {}, "max")
        if pts:
            out[key] = SigmaTable(key, tuple(pts), f"{rl15.name} summary (default, best mean)")
    if (d / "lcbench_calibration.json").exists():
        out["lcbench"] = SigmaTable("lcbench", ((0.0, 0.0),),
                                    "deterministic surrogate: same config, same result for every seed")
    return out


@dataclass(frozen=True)
class RuleDecision:
    decision: str            # real | noise | no_calibration
    reason: str
    gain: float | None
    sigma: float | None
    degenerate: bool = False  # sigma table is all zeros: 'real' here is not a real-vs-noise test


class SeedSpreadRule:
    """The calibrated seed-spread rule ('rule_calib', R34/R37) for one race's target."""

    def __init__(self, table: SigmaTable | None, k_sigma: float = RULE_K_SIGMA) -> None:
        self.table = table
        self.k_sigma = k_sigma

    @property
    def available(self) -> bool:
        return self.table is not None

    def decide(self, improvement: float | None, prev_score: float | None) -> RuleDecision:
        gain = math.inf if prev_score is None or improvement is None else float(improvement)
        if self.table is None:
            return RuleDecision("no_calibration", "no calibrated seed spread for this target", None, None)
        sigma = self.table.sigma_at(prev_score)
        deg = self.table.degenerate
        note = " (degenerate: sigma 0)" if deg else ""
        if gain > self.k_sigma * sigma:
            return RuleDecision("real", f"claimed gain {gain:.4g} > {self.k_sigma:g} x calibrated seed std {sigma:.4g}"
                                + note, gain, sigma, deg)
        return RuleDecision("noise", f"claimed gain {gain:.4g} <= {self.k_sigma:g} x calibrated seed std {sigma:.4g}"
                            + note, gain, sigma, deg)

    def manifest(self) -> dict[str, Any]:
        return {"k_sigma": self.k_sigma, "table": self.table.manifest() if self.table else None}


def combine(llm_verdict: Verdict | str | None, rule: RuleDecision, *, llm_replied: bool | None = None) -> Verdict | None:
    """D40: LLM 'invalid' -> invalid; otherwise the rule (real -> valid, noise -> unsure).
    No usable LLM reply (``llm_replied`` False, or no verdict) -> unsure: nothing checked for
    leakage, so the rule alone never confirms a gain. Without a calibration the LLM verdict
    stands (dev races only; benchmarks refuse)."""
    v = Verdict(llm_verdict) if llm_verdict is not None else None
    if v == Verdict.invalid:
        return Verdict.invalid
    if llm_replied is False or v is None:
        return Verdict.unsure
    if rule.decision == "real":
        return Verdict.valid
    if rule.decision == "noise":
        return Verdict.unsure
    return v


# =========================================================================== median stopping


def value_at(curve: Sequence[tuple[float, float]], t: float) -> float | None:
    """Last value at or before ``t`` (curve sorted by time); None if none yet."""
    v = None
    for ti, vi in curve:
        if ti <= t + 1e-9:
            v = vi
        else:
            break
    return v


@dataclass(frozen=True)
class MedianStopRule:
    fractions: tuple[float, ...] = (0.25, 0.5)
    min_completed: int = 5

    def checkpoints(self, run_budget_s: float, time_scale: float = 1.0) -> list[tuple[float, float]]:
        """[(fraction, checkpoint time on the metric time axis)]."""
        return [(f, f * float(run_budget_s) * float(time_scale)) for f in self.fractions]

    def completed_values(self, curve: Sequence[tuple[float, float]], run_budget_s: float,
                         time_scale: float = 1.0) -> dict[float, float]:
        """A completed run's value at each checkpoint it reached (its curve went past it)."""
        out: dict[float, float] = {}
        if not curve:
            return out
        last_t = curve[-1][0]
        for f, c in self.checkpoints(run_budget_s, time_scale):
            if last_t + 1e-9 >= c:
                v = value_at(curve, c)
                if v is not None:
                    out[f] = v
        return out

    def should_stop(self, value: float, reference: Iterable[float], direction: str) -> tuple[bool, float | None, int]:
        """(stop?, median, n). Stop iff n >= min_completed and value is worse than the median."""
        ref = [float(x) for x in reference if x is not None and math.isfinite(float(x))]
        if len(ref) < self.min_completed:
            return False, (float(statistics.median(ref)) if ref else None), len(ref)
        med = float(statistics.median(ref))
        worse = value > med if direction == "min" else value < med
        return worse, med, len(ref)

    def manifest(self) -> dict[str, Any]:
        return {"fractions": list(self.fractions), "min_completed": self.min_completed}


@dataclass
class EngineState:
    """Per-arm engine settings the scheduler resolves once (for the manifest)."""

    features: tuple[str, ...]
    judge_rule: SeedSpreadRule | None = None
    median_rule: MedianStopRule | None = None
    curve_metric: str | None = None
    time_scale: float = 1.0
    notes: list[str] = field(default_factory=list)


__all__ = [
    "CURVE_METRIC", "EngineState", "MedianStopRule", "RuleDecision", "SeedSpreadRule", "SigmaTable",
    "combine", "curve_metric", "load_sigma_tables", "sigma_key", "value_at",
]
