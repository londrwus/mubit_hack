"""User-message builders and wire schemas for the three research steps. DRAFT v0 - NOT frozen.

Layout of every user message (budget.md, lever 2):

    TARGET SPEC block      <- static per race; mubit_client puts cache breakpoint 2 here
    CACHE_BREAK marker     <- split point; never sent to the model
    dynamic digest         <- per call: state / run / claim

All rendering is deterministic (fixed key order, numbers at 4 significant figures, no
timestamps, no UUIDs), so identical states give byte-identical messages.

The spec block never reveals which parameter is the contamination trap: the trap appears as
an ordinary parameter with its own (innocent) description.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from backend.agent.prompts import CURRENT_VERSION
from backend.core.types import (
    DiagnoseOutput,
    FailureClass,
    JudgeOutput,
    JudgeReasonCode,
    ProposeOutput,
    StepType,
    StopDecision,
    Verdict,
)
from workers.targets.base import TargetSpec

log = logging.getLogger(__name__)

CACHE_BREAK = "\n<<<CACHE_BREAK>>>\n"
"""Separates the cached spec block from the dynamic digest inside a ``user`` string."""

FREE_SLOTS_LABEL = "FREE SLOTS (K):"
LEADERBOARD_SIZE = 5
RECENT_SIZE = 8
LOG_LINES = 50


def split_user(user: str) -> tuple[str | None, str]:
    """(cached prefix, dynamic rest). Prefix is None when the marker is absent."""
    if CACHE_BREAK in user:
        prefix, rest = user.split(CACHE_BREAK, 1)
        return prefix, rest
    return None, user


def strip_cache_break(user: str) -> str:
    """The text the model actually sees (marker removed)."""
    prefix, rest = split_user(user)
    return rest if prefix is None else f"{prefix}\n{rest}"


# --------------------------------------------------------------------------- formatting


def fmt_num(x: float | int | None) -> str:
    """4 significant figures; ints stay ints; None -> 'n/a'."""
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    return f"{x:.4g}"


def fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return fmt_num(v)
    if v is None:
        return "null"
    return json.dumps(v) if not isinstance(v, str) else v


def fmt_config(config: dict[str, Any], order: Sequence[str] | None = None) -> str:
    """Compact one-line config in config-space order: ``lr=0.003 n_layer=4 ...``."""
    keys = list(order) if order is not None else sorted(config)
    keys += [k for k in sorted(config) if k not in keys]
    return " ".join(f"{k}={fmt_value(config[k])}" for k in keys if k in config)


# --------------------------------------------------------------------------- target spec block


def target_spec_block(spec: TargetSpec) -> str:
    """Static per-race block (cache breakpoint 2). No trap fields, no IDs, no timestamps."""
    better = "lower is better" if spec.direction == "min" else "higher is better"
    lines = [
        "TARGET SPEC",
        f"name: {spec.name}",
        f"task: {spec.description.strip()}",
        f"metric: {spec.metric} ({better})",
        f"goal: {spec.metric} {'<=' if spec.direction == 'min' else '>='} {fmt_num(spec.target_value)}",
        f"hardware per run: {spec.hardware}",
        f"training budget per run: {spec.budget_s} s wall-clock",
        "config space (every config must set every parameter):",
    ]
    for name, p in spec.config_space.items():
        if p.type in ("float", "int"):
            dom = f"{p.type} in [{fmt_num(p.low if p.type == 'float' else int(p.low or 0))}, " \
                  f"{fmt_num(p.high if p.type == 'float' else int(p.high or 0))}]"
            if p.log:
                dom += " log-scale"
        elif p.type == "categorical":
            dom = "one of [" + ", ".join(fmt_value(c) for c in (p.choices or [])) + "]"
        else:
            dom = "bool"
        line = f"- {name}: {dom}; default {fmt_value(p.default)}"
        if p.description:
            line += f"; {p.description.strip()}"
        lines.append(line)
    lines.append(f"default config: {fmt_config(spec.default_config(), list(spec.config_space))}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- digest rows


class ResultRow(BaseModel):
    """One finished (or failed) experiment as the LLM sees it. ``label`` is a short,
    race-local name like "#12" (not a UUID)."""

    label: str
    config: dict[str, Any]
    status: str = "succeeded"          # ExperimentStatus value
    score: float | None = None
    note: str | None = None            # e.g. error class, "diverged at step 120"
    hypothesis: str | None = None


def _row_line(r: ResultRow, order: Sequence[str]) -> str:
    s = f"{r.label} {r.status}"
    s += f" score={fmt_num(r.score)}" if r.score is not None else ""
    s += f" | {fmt_config(r.config, order)}"
    if r.note:
        s += f" | note: {r.note.strip()}"
    return s


def leaderboard(spec: TargetSpec, history: Sequence[ResultRow], n: int = LEADERBOARD_SIZE) -> list[ResultRow]:
    scored = [r for r in history if r.score is not None and math.isfinite(r.score)]
    return sorted(scored, key=lambda r: r.score if spec.direction == "min" else -r.score)[:n]  # type: ignore[operator, return-value]


# --------------------------------------------------------------------------- builders


def build_propose_user(
    spec: TargetSpec,
    history: Sequence[ResultRow],
    free_slots: int,
    running: Sequence[dict[str, Any]] = (),
    time_left_s: float | None = None,
) -> str:
    """propose_experiments user message: spec + top-5 + last 8 + best config + in-flight + K.

    ``history`` is in completion order (oldest first)."""
    if free_slots < 1:
        raise ValueError("free_slots must be >= 1")
    order = list(spec.config_space)
    top = leaderboard(spec, history)
    recent = list(history)[-RECENT_SIZE:]
    lines = ["STATE", f"{FREE_SLOTS_LABEL} {free_slots}", f"experiments finished so far: {len(history)}"]
    if time_left_s is not None:
        lines.append(f"race time left: {int(time_left_s)} s")
    lines.append(f"leaderboard (top {LEADERBOARD_SIZE}, best first):")
    lines += [f"  {_row_line(r, order)}" for r in top] or ["  (no scored runs yet)"]
    lines.append(f"recent results (last {RECENT_SIZE}, newest last):")
    lines += [f"  {_row_line(r, order)}" for r in recent] or ["  (none yet)"]
    best = top[0] if top else None
    lines.append("current best config: " + (fmt_config(best.config, order) if best else "none yet (defaults are the starting code)"))
    lines.append("still running:")
    lines += [f"  {fmt_config(c, order)}" for c in running] or ["  (none)"]
    lines.append(f"Propose up to {free_slots} new experiment(s), one per free slot.")
    return target_spec_block(spec) + CACHE_BREAK + "\n".join(lines)


def build_diagnose_user(
    spec: TargetSpec,
    config: dict[str, Any],
    status: str,
    error: str | None,
    logs_tail: Sequence[str],
    metrics: dict[str, float] | None = None,
) -> str:
    """diagnose_run user message: spec + the failed run (config, status, error, metrics, last 50 log lines)."""
    order = list(spec.config_space)
    lines = [
        "RUN",
        f"config: {fmt_config(config, order)}",
        f"status: {status}",
        f"error: {error.strip() if error else 'none'}",
        "summary metrics: " + (" ".join(f"{k}={fmt_num(v)}" for k, v in sorted((metrics or {}).items())) or "none"),
        f"last log lines (up to {LOG_LINES}):",
    ]
    tail = [ln.rstrip() for ln in list(logs_tail)[-LOG_LINES:]]
    lines += [f"  {ln}" for ln in tail] or ["  (no logs captured)"]
    return target_spec_block(spec) + CACHE_BREAK + "\n".join(lines)


class JudgeClaim(BaseModel):
    """A claimed new best, with the evidence the judge needs."""

    prev_best_score: float | None
    prev_best_config: dict[str, Any] | None = None
    new_score: float
    new_config: dict[str, Any]
    # Free-form evidence: split hashes, overlap counts, eval config, selection split, eval seeds,
    # seed spread... Rendered as sorted ``key: value`` lines.
    evidence: dict[str, Any] = Field(default_factory=dict)


def build_judge_user(spec: TargetSpec, claim: JudgeClaim) -> str:
    """judge_improvement user message: spec + claimed gain + config diff + evidence."""
    order = list(spec.config_space)
    gain = None if claim.prev_best_score is None else claim.new_score - claim.prev_best_score
    prev_cfg = claim.prev_best_config or spec.default_config()
    changed = [k for k in order if k in claim.new_config and claim.new_config.get(k) != prev_cfg.get(k)]
    lines = [
        "CLAIM",
        f"previous best {spec.metric}: {fmt_num(claim.prev_best_score)}",
        f"previous best config: {fmt_config(prev_cfg, order)}",
        f"new {spec.metric}: {fmt_num(claim.new_score)}",
        f"new config: {fmt_config(claim.new_config, order)}",
        f"claimed gain: {fmt_num(gain)} ({'lower is better' if spec.direction == 'min' else 'higher is better'})",
        "changed parameters: "
        + (", ".join(f"{k} {fmt_value(prev_cfg.get(k))} -> {fmt_value(claim.new_config[k])}" for k in changed) or "none"),
        "EVIDENCE",
    ]
    lines += [f"  {k}: {fmt_value(v)}" for k, v in sorted(claim.evidence.items())] or ["  (none provided)"]
    return target_spec_block(spec) + CACHE_BREAK + "\n".join(lines)


# --------------------------------------------------------------------------- early_stop (D34)
# Separate Mubit task with its own frozen prompt; nothing above this line changes for it.

EARLY_STOP_CURVE_POINTS = 12
"""At most this many curve points are shown (evenly spaced, first and latest always kept)."""

EARLY_STOP_CURVE_METRIC: dict[str, str] = {"rl": "train_return", "cifar": "val_acc", "lm": "val_bpb"}
"""The streamed metric the early_stop step reads, per target. RL streams only its training
return; CIFAR and LM stream a validation metric. Same direction as the target's metric."""


def early_stop_curve_metric(spec: TargetSpec) -> str:
    return EARLY_STOP_CURVE_METRIC.get(spec.name, spec.metric)


class CurvePoint(BaseModel):
    step: int
    t: float      # seconds of TRAINING elapsed (from training start, not from launch / cold start)
    value: float


class EarlyStopState(BaseModel):
    """What the early_stop step sees at one checkpoint. Only agent-visible numbers: the run's
    own streamed curve and the curves of runs that had already completed in the same race
    (judge-invalid runs excluded). Never held-out scores or trap ground truth."""

    config: dict[str, Any]
    curve_metric: str
    elapsed_s: float              # training seconds used so far
    budget_s: float               # per-run training budget
    curve: list[CurvePoint]       # oldest first, t <= elapsed_s
    n_completed: int
    best_final: float | None      # best final curve value among completed runs
    median_final: float | None    # median final curve value among completed runs
    median_at_point: float | None = None    # completed runs' median value at the same training time
    best_run_at_point: float | None = None  # the best completed run's value at the same training time


def downsample_curve(points: Sequence[CurvePoint], n: int = EARLY_STOP_CURVE_POINTS) -> list[CurvePoint]:
    """Deterministic, evenly spaced subset of at most ``n`` points (first and latest kept)."""
    pts = list(points)
    if len(pts) <= n:
        return pts
    idx = sorted({round(i * (len(pts) - 1) / (n - 1)) for i in range(n)})
    return [pts[i] for i in idx]


def build_early_stop_user(spec: TargetSpec, state: EarlyStopState) -> str:
    """early_stop user message: spec + the running run (config, progress, compact curve) +
    the completed runs' final and same-time summary."""
    if not state.curve:
        raise ValueError("early_stop needs at least one curve point")
    order = list(spec.config_space)
    m = state.curve_metric
    better = "lower is better" if spec.direction == "min" else "higher is better"
    pct = 100.0 * state.elapsed_s / state.budget_s if state.budget_s > 0 else 0.0
    lines = [
        "RUN IN PROGRESS",
        f"config: {fmt_config(state.config, order)}",
        f"training time used: {int(round(state.elapsed_s))} s of {int(round(state.budget_s))} s budget ({int(round(pct))}%)",
        f"curve: {m} ({better}); columns: step, training seconds, value; oldest first",
    ]
    lines += [f"  {p.step} {fmt_num(float(p.t))} {fmt_num(float(p.value))}" for p in downsample_curve(state.curve)]
    lines += [
        f"latest {m}: {fmt_num(float(state.curve[-1].value))}",
        "COMPLETED RUNS IN THIS RACE",
        f"completed runs: {state.n_completed}",
        f"best final {m}: {fmt_num(state.best_final)}",
        f"median final {m}: {fmt_num(state.median_final)}",
        f"at the same training time, completed runs had: median {fmt_num(state.median_at_point)}; "
        f"the best run had {fmt_num(state.best_run_at_point)}",
        "Decide: continue or stop this run.",
    ]
    return target_spec_block(spec) + CACHE_BREAK + "\n".join(lines)


def early_stop_wire_schema() -> dict[str, Any]:
    """Fully closed, decision only (prompt v2, D39 gate): an adapter can match Opus exactly."""
    return {
        "type": "object",
        "properties": {"decision": {"type": "string", "enum": [d.value for d in StopDecision]}},
        "required": ["decision"],
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------- wire schemas
# Closed JSON schemas for output_config.format. No numeric/length/array-size constraints
# (not supported by structured outputs); those are enforced by Pydantic + validate_config.
# The schemas do NOT depend on K, so the output format (and the Mubit task) is stable.


# Target-agnostic settings (D24): the same schema for every target, so propose_experiments is
# ONE Mubit task. A compact "name=value, ..." string keeps output tokens near the per-target
# object form (R17: a [{name, value}] list doubled them). Types and ranges are checked by
# spec.validate_config.
SETTINGS_SCHEMA: dict[str, Any] = {
    "type": "string",
    "description": "Every parameter once, as name=value pairs separated by commas.",
}


def propose_wire_schema(spec: TargetSpec | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "experiments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "config": SETTINGS_SCHEMA,
                        "hypothesis": {"type": "string", "description": "At most 20 words."},
                    },
                    "required": ["config", "hypothesis"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["experiments"],
        "additionalProperties": False,
    }


def diagnose_wire_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "failure_class": {"type": "string", "enum": [c.value for c in FailureClass]},
            "cause": {"type": "string"},
            "fix": {"type": "string"},
        },
        "required": ["failure_class", "cause", "fix"],
        "additionalProperties": False,
    }


def judge_wire_schema(version: str | None = None) -> dict[str, Any]:
    """By prompt version (default: the current one). v0 (D27): verdict + free-text reason.
    v1 (D37): fully closed, verdict + reason_code enums, exactly the schema the D39 gate
    measured (``scripts/gate_judge_v1.py wire_schema("v1b")``)."""
    version = version or CURRENT_VERSION[StepType.judge_improvement]
    if version == "v0":
        return {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": [v.value for v in Verdict]},
                "reason": {"type": "string"},
            },
            "required": ["verdict", "reason"],
            "additionalProperties": False,
        }
    if version == "v1":
        return {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": [v.value for v in Verdict]},
                "reason_code": {"type": "string", "enum": [c.value for c in JudgeReasonCode]},
            },
            "required": ["verdict", "reason_code"],
            "additionalProperties": False,
        }
    raise ValueError(f"no judge wire schema for prompt version {version!r}")


def wire_schema_for(step: StepType | str, spec: TargetSpec | None = None) -> dict[str, Any]:
    step = StepType(step)
    if step is StepType.propose_experiments:
        if spec is None:
            raise ValueError("propose_experiments needs the target spec for its config schema")
        return propose_wire_schema(spec)
    if step is StepType.early_stop:
        return early_stop_wire_schema()
    return diagnose_wire_schema() if step is StepType.diagnose_run else judge_wire_schema()


def propose_validator(spec: TargetSpec, max_experiments: int | None = None) -> Callable[[ProposeOutput], ProposeOutput]:
    """Post-Pydantic check for propose replies: each config must pass ``spec.validate_config``
    and is replaced by its coerced form. Invalid experiments are dropped (logged) and the rest
    kept; the reply is invalid only if none survive. Extras beyond K are dropped. Same rule on
    every arm."""

    def _validate(out: ProposeOutput) -> ProposeOutput:
        exps = out.experiments[:max_experiments] if max_experiments else out.experiments
        kept, errors = [], []
        for i, e in enumerate(exps):
            try:
                e.config = spec.validate_config(e.config)
                kept.append(e)
            except ValueError as err:
                errors.append(f"experiments[{i}].config: {err}")
        if errors:
            log.warning("propose reply: dropped %d of %d experiments: %s", len(errors), len(exps), "; ".join(errors))
        if not kept:
            raise ValueError("; ".join(errors))
        return ProposeOutput(experiments=kept)

    return _validate


__all__ = [
    "CACHE_BREAK",
    "FREE_SLOTS_LABEL",
    "DiagnoseOutput",
    "JudgeClaim",
    "JudgeOutput",
    "ProposeOutput",
    "ResultRow",
    "build_diagnose_user",
    "build_judge_user",
    "build_propose_user",
    "CurvePoint",
    "EarlyStopState",
    "build_early_stop_user",
    "downsample_curve",
    "early_stop_curve_metric",
    "early_stop_wire_schema",
    "diagnose_wire_schema",
    "fmt_config",
    "fmt_num",
    "judge_wire_schema",
    "leaderboard",
    "propose_validator",
    "propose_wire_schema",
    "split_user",
    "strip_cache_break",
    "target_spec_block",
    "wire_schema_for",
]
