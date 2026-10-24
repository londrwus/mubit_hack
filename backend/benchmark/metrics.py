"""Headline metrics, as pure functions over plain rows.

The race engine and the export analysis both call ``summarize_arm`` with rows in the same
shape as the SQLite tables, so anyone holding the JSONL export can recompute every number
on the results slide (benchmark.md section 5, "What makes it verifiable").

Definitions (benchmark.md "Metrics"):
- **Paid time** of a slot = every second between the arm's start and end (busy + idle).
- **GPU idle %** = idle seconds / paid seconds x 100. LLM wait time is idle time.
- **GPU $** = sum over intervals of (end - start) x price_per_s (busy and idle alike).
- **Quality** of a run = its held-out score, measured on a split no decision ever used:
  ``metrics["heldout_score"]`` when the target reports it (lm, rl, cifar all do). The
  agent only ever sees the *reported* ``score``, which the contamination trap inflates.
  For a target without ``heldout_score`` the reported score is used, and a ground-truth
  trap run has no quality at all (its score was measured on contaminated data).
- **Counted experiment** = finished with a score (status succeeded) AND not judged invalid
  by the arm's own judge (the agent threw it away) AND not a ground-truth trap run (same
  rule for every arm, judge or not) AND has a quality.
- **Best score** = best counted quality at the end of the budget ("Model quality").
- **Time-to-target** = seconds from the race start to the end of the first counted
  experiment whose quality reaches ``target_value``.
- **$-to-target** = GPU $ accrued in [race start, that moment] + LLM $ of calls completed
  by then.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from backend.core.types import ArmKind, ArmSummary, ExperimentStatus


def _get(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    v = row.get(key, default)
    return default if v is None else v


def quality(exp: Mapping[str, Any]) -> float | None:
    """Held-out score if reported, else the reported score unless the run is a trap."""
    h = (exp.get("metrics") or {}).get("heldout_score")
    if h is not None and math.isfinite(float(h)):
        return float(h)
    if bool(_get(exp, "is_trap", False)):
        return None
    s = exp.get("score")
    return float(s) if s is not None and math.isfinite(float(s)) else None


def is_counted(exp: Mapping[str, Any]) -> bool:
    """A ground-truth trap run never counts, even with an honest held-out score: its pipeline
    trained or tuned on evaluation data (lm adds the val shard, cifar fine-tunes on test), so
    its held-out gain partly comes from contamination. Counting it would reward an arm for
    accepting fake improvements (Optuna has no judge) and penalise an arm whose judge is right."""
    return (
        _get(exp, "status") == ExperimentStatus.succeeded.value
        and exp.get("score") is not None
        and _get(exp, "judge_verdict") != "invalid"
        and not bool(_get(exp, "is_trap", False))
        and quality(exp) is not None
    )


def _better(a: float, b: float | None, direction: str) -> bool:
    if b is None:
        return True
    return a < b if direction == "min" else a > b


def _reaches(score: float, target_value: float, direction: str) -> bool:
    return score <= target_value if direction == "min" else score >= target_value


def gpu_cost_until(intervals: Iterable[Mapping[str, Any]], t: float) -> float:
    """GPU $ accrued by time ``t`` (intervals clipped at t)."""
    total = 0.0
    for iv in intervals:
        s, e = float(iv["start"]), float(iv["end"])
        if s >= t:
            continue
        total += (min(e, t) - s) * float(_get(iv, "price_per_s", 0.0))
    return total


def interval_totals(intervals: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    busy = idle = busy_cost = idle_cost = 0.0
    for iv in intervals:
        d = max(0.0, float(iv["end"]) - float(iv["start"]))
        c = d * float(_get(iv, "price_per_s", 0.0))
        if iv["state"] == "busy":
            busy += d
            busy_cost += c
        else:
            idle += d
            idle_cost += c
    return {
        "busy_s": busy,
        "idle_s": idle,
        "paid_s": busy + idle,
        "busy_cost_usd": busy_cost,
        "idle_cost_usd": idle_cost,
        "gpu_cost_usd": busy_cost + idle_cost,
    }


def summarize_arm(
    arm: ArmKind | str,
    *,
    direction: Literal["min", "max"] | str,
    target_value: float,
    race_start: float,
    experiments: list[Mapping[str, Any]],
    intervals: list[Mapping[str, Any]],
    llm_calls: list[Mapping[str, Any]],
) -> tuple[ArmSummary, dict[str, Any]]:
    """Return (ArmSummary, extras). ``extras`` holds the supporting numbers (busy/idle
    seconds, idle $, trap counts, retries, ...) shown on hover / in the export."""
    tot = interval_totals(intervals)

    by_end = sorted(experiments, key=lambda r: float(_get(r, "ended_at", 0.0)))
    counted = [e for e in by_end if is_counted(e)]
    best: Mapping[str, Any] | None = None
    for e in counted:
        if best is None or _better(quality(e), quality(best), direction):  # type: ignore[arg-type]
            best = e

    hit: Mapping[str, Any] | None = None
    for e in counted:
        if _reaches(quality(e), target_value, direction):  # type: ignore[arg-type]
            hit = e
            break
    # The model the agent would ship: its last accepted best (reported score), and how good
    # it really is (quality). Shows the cost of accepting a fake improvement.
    shipped = next((e for e in reversed(by_end) if bool(_get(e, "is_best", False))
                    and _get(e, "status") == ExperimentStatus.succeeded.value), None)

    time_to_target = cost_to_target = None
    if hit is not None:
        t_hit = float(hit["ended_at"])
        time_to_target = max(0.0, t_hit - race_start)
        llm_by_then = sum(
            float(_get(c, "cost_usd", 0.0))
            for c in llm_calls
            if float(_get(c, "completed_at", _get(c, "created_at", 0.0))) <= t_hit
        )
        cost_to_target = gpu_cost_until(intervals, t_hit) + llm_by_then

    statuses = [_get(e, "status") for e in experiments]
    finished = sum(1 for s in statuses if s == ExperimentStatus.succeeded.value)
    failed = sum(1 for s in statuses if s in (ExperimentStatus.failed.value, ExperimentStatus.timed_out.value))
    cancelled = sum(1 for s in statuses if s == ExperimentStatus.cancelled.value)

    latencies = [float(_get(c, "latency_ms", 0.0)) for c in llm_calls]
    llm_cost = sum(float(_get(c, "cost_usd", 0.0)) for c in llm_calls)

    summary = ArmSummary(
        arm=ArmKind(arm),
        best_score=quality(best) if best is not None else None,
        best_experiment_id=str(best["id"]) if best is not None else None,
        experiments_finished=finished,
        experiments_failed=failed,
        gpu_cost_usd=tot["gpu_cost_usd"],
        llm_cost_usd=llm_cost,
        gpu_idle_pct=(tot["idle_s"] / tot["paid_s"] * 100.0) if tot["paid_s"] > 0 else 0.0,
        time_to_target_s=time_to_target,
        cost_to_target_usd=cost_to_target,
        llm_calls=len(llm_calls),
        llm_invalid_calls=sum(1 for c in llm_calls if not bool(_get(c, "valid", False))),
        llm_p50_latency_ms=statistics.median(latencies) if latencies else None,
    )

    trap_runs = [e for e in experiments if bool(_get(e, "is_trap", False)) and _get(e, "status") == "succeeded"]
    by_step: dict[str, int] = {}
    for c in llm_calls:
        st = str(_get(c, "step_type", "?"))
        by_step[st] = by_step.get(st, 0) + 1
    paid_h = tot["paid_s"] / 3600.0
    extras: dict[str, Any] = {
        **tot,
        "quality_source": "heldout_score" if any((e.get("metrics") or {}).get("heldout_score") is not None
                                                  for e in experiments) else "score",
        "best_reported_score": float(best["score"]) if best is not None else None,
        "shipped_experiment_id": shipped["id"] if shipped is not None else None,
        "shipped_reported_score": float(shipped["score"]) if shipped is not None and shipped.get("score") is not None else None,
        "shipped_quality": quality(shipped) if shipped is not None else None,
        "experiments_cancelled": cancelled,
        "experiments_total": len(experiments),
        "experiments_per_gpu_hour": (finished / paid_h) if paid_h > 0 else 0.0,
        "trap_runs": len(trap_runs),
        "fake_best_accepted": sum(1 for e in trap_runs if bool(_get(e, "is_best", False))),
        "fake_caught_by_judge": sum(1 for e in trap_runs if _get(e, "judge_verdict") == "invalid"),
        "genuine_rejected_by_judge": sum(
            1
            for e in experiments
            if _get(e, "status") == "succeeded"
            and not bool(_get(e, "is_trap", False))
            and _get(e, "judge_verdict") == "invalid"
        ),
        "llm_calls_by_step": by_step,
        "llm_retries": sum(1 for c in llm_calls if bool(_get(c, "is_retry", False))),
        "llm_fallbacks": sum(1 for c in llm_calls if bool(_get(c, "is_fallback", False))),
        "llm_replayed": sum(1 for c in llm_calls if bool(_get(c, "is_replay", False))),
        "malformed_reply_rate": (summary.llm_invalid_calls / len(llm_calls)) if llm_calls else 0.0,
        "llm_cost_per_experiment_usd": (llm_cost / finished) if finished else None,
        **engine_extras(experiments),
    }
    return summary, extras


def engine_extras(experiments: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Busy Machines engine records (scheduler: ``summary["early_stop"]`` per stopped run,
    ``summary["judge_policy"]`` per judged claim), recomputable from the export."""
    es = [(e.get("summary") or {}).get("early_stop") for e in experiments]
    es = [x for x in es if x]
    stopped = [x for x in es if x.get("outcome") == "stopped"]
    jp = [(e.get("summary") or {}).get("judge_policy") for e in experiments]
    jp = [x for x in jp if x]

    def count(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for x in jp:
            k = str(x.get(key))
            out[k] = out.get(k, 0) + 1
        return out

    lat = [float(x["stop_latency_s"]) for x in stopped if x.get("stop_latency_s") is not None]
    clag = [float(x["runner_cancel_lag_s"]) for x in stopped if x.get("runner_cancel_lag_s") is not None]
    # Slot time freed: older races stored it as "saved_s" / "saved_usd". It is slot time made
    # available, not money saved, so the $ number is only an equivalent (never a $ framing).
    freed = sum(float(x.get("slot_s_freed", x.get("saved_s")) or 0.0) for x in stopped)
    freed_usd = sum(float(x.get("freed_slot_usd_equiv", x.get("saved_usd")) or 0.0) for x in stopped)
    # Degenerate rule decisions (sigma 0, lcbench) are not a real-vs-noise test: counted apart.
    real_jp = [x for x in jp if not x.get("degenerate")]

    def rule_counts() -> dict[str, int]:
        out: dict[str, int] = {}
        for x in jp:
            k = "degenerate" if x.get("degenerate") else str(x.get("rule"))
            out[k] = out.get(k, 0) + 1
        return out

    return {
        "early_stopped": len(stopped),
        "early_stop_too_late": sum(1 for x in es if x.get("outcome") == "too_late"),
        "early_stop_failed_first": sum(1 for x in es if str(x.get("outcome") or "").endswith("_before_stop")),
        "early_stop_by_fraction": {str(f): sum(1 for x in stopped if x.get("fraction") == f)
                                   for f in sorted({x.get("fraction") for x in stopped})},
        "early_stop_slot_s_freed": freed,
        "early_stop_freed_slot_usd_equiv": freed_usd,
        "early_stop_saved_s": freed,   # deprecated alias of early_stop_slot_s_freed (bench.py table)
        "early_stop_latency_p50_s": statistics.median(lat) if lat else None,
        "early_stop_runner_cancel_lag_p50_s": statistics.median(clag) if clag else None,
        "early_stop_cancel_unconfirmed": sum(1 for x in stopped if x.get("cancel_confirmed") is False),
        "judge_policy_claims": len(jp),
        "judge_policy_llm_verdicts": count("llm_verdict"),
        "judge_policy_rule": rule_counts(),
        "judge_policy_final": count("final"),
        "judge_policy_degenerate": len(jp) - len(real_jp),
        "judge_policy_no_leakage_check": sum(1 for x in jp if "no_leakage_check" in (x.get("tags") or [])),
        "judge_policy_rule_changed_llm": sum(1 for x in real_jp if x.get("final") != x.get("llm_verdict")),
    }
