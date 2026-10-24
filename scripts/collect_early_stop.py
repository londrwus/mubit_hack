"""Example collection and evaluation for the 4th recurring step, ``early_stop`` (D34).

No GPU: every case is a checkpoint of a run that already completed in the overnight
collection races (``data/collection/collect_{rl,cifar,lm}.db``, opened read-only), so the
ground truth (did the run go on to beat the best?) is known.

Case = one completed, non-probe, non-trap run at ~25% or ~50% of its training budget:
- the run's config, its streamed curve up to that training time (``EARLY_STOP_CURVE_METRIC``:
  rl train_return, cifar val_acc, lm val_bpb), and the training time used;
- context = the runs of the same race that had COMPLETED by that moment (wall clock of the
  checkpoint; judge-invalid and trap runs excluded): count, best and median final curve value,
  and their median / best-run value at the same training time;
- labels (never shown to the model): ``beats_val`` = the run's final curve value beat the
  best-at-the-time by more than ``MARGIN_VAL``; ``beats_heldout`` = its held-out score beat the
  best held-out score at the time by more than ``MARGIN_HELDOUT``. ``stop_correct`` = neither.

Phases (each resumable; spend from every calls file counts toward ``LLM_CAP_USD``)::

    python -m uv run --extra ml python scripts/collect_early_stop.py cases
    python -m uv run --extra ml python scripts/collect_early_stop.py draft --n 20 [--version vN] [--model ...]  # record off
    python -m uv run --extra ml python scripts/collect_early_stop.py gate --pass opus1|opus2|haiku              # D39, record off
    python -m uv run --extra ml python scripts/collect_early_stop.py gate-report
    python -m uv run --extra ml python scripts/collect_early_stop.py record --yes --project b   # Opus, record full; refuses
                                                                  # unless gate_summary_<fingerprint>.json passed (D38/D39)
    python -m uv run --extra ml python scripts/collect_early_stop.py eval --yes        # Opus, record off, held-out runs
    python -m uv run --extra ml python scripts/collect_early_stop.py report

Status and results: data/early_stop/README.md.

Every call goes through ``MubitClient`` (the Mubit gateway). The draft phase may run before
``StepType.early_stop`` exists; it then sends with the judge_improvement step config (same
effort "low" and max_tokens 4000), record off, Haiku: Mubit groups tasks by model + system
prompt + schema, not by our step label, so nothing is recorded or mixed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agent import prompts as prompts_pkg  # noqa: E402
from backend.agent.llm_base import DEV_MODEL, OPUS_MODEL  # noqa: E402
from backend.agent.prompts.build import (  # noqa: E402
    CurvePoint,
    EarlyStopState,
    build_early_stop_user,
    early_stop_curve_metric,
    early_stop_wire_schema,
)
from backend.core.types import ArmKind, EarlyStopOutput, StepType  # noqa: E402

# Prompt v1 (frozen, never recorded) also returned a reason_code; kept here only to analyse the
# v1 gate replies in gate_calls.jsonl.
V1_REASON_DECISION: dict[str, str] = {
    "on_best_pace": "continue", "closing_gap": "continue", "too_early_to_tell": "continue",
    "behind_median": "stop", "plateau_below_best": "stop", "diverging": "stop",
}
from workers.targets.base import TargetSpec, get_target  # noqa: E402

DATE = "2026-10-24"
COLLECTION_DIR = REPO_ROOT / "data" / "collection"
OUT_DIR = REPO_ROOT / "data" / "early_stop"
CASES_PATH = OUT_DIR / "cases.jsonl"
CALL_FILES = {
    "draft": OUT_DIR / "draft_calls.jsonl",
    "record": OUT_DIR / "record_calls.jsonl",
    "eval": OUT_DIR / "eval_calls.jsonl",
    "gate": OUT_DIR / "gate_calls.jsonl",
}
SUMMARY_PATH = OUT_DIR / "summary.json"
SESSION_PREFIX = {
    "draft": f"draft-es-{DATE}",
    "record": f"collect-es-{DATE}",
    "eval": f"eval-es-{DATE}",
    "gate": f"gate-es-{DATE}",
}
# D39 gate: 40 diverse cases, Opus twice + Haiku once, record off (project A).
GATE_PASSES: dict[str, str] = {"opus1": OPUS_MODEL, "opus2": OPUS_MODEL, "haiku": DEV_MODEL}
GATE_QUOTA: dict[str, int] = {"rl": 14, "cifar": 14, "lm": 12}
# Project B (D38): recorded calls only. Provider key added 03:05 (normal concurrency); 429 backoff kept.
PROJECT_KEY_ENV = {"a": "MUBIT_API_KEY", "b": "MUBIT_API_KEY_B"}
MAX_CONCURRENCY_B = 4
BACKOFF_429_S = (2.0, 4.0, 8.0, 16.0, 32.0, 60.0)
TARGETS = ("rl", "cifar", "lm")
FRACTIONS = (0.25, 0.5)
MIN_COMPLETED = 3          # below this the step is not called (scheduler: continue)

# "Did not beat the best by more than a small margin" (labels only; the model never sees these).
# Roughly one seed-spread of a good config (R23: rl 6.2, cifar 0.0018-0.0028 held-out, lm val 0.0066).
MARGIN_VAL: dict[str, float] = {"rl": 10.0, "cifar": 0.002, "lm": 0.005}
MARGIN_HELDOUT: dict[str, float] = {"rl": 10.0, "cifar": 0.002, "lm": 0.01}

# Split by RUN (both checkpoints of a run land in the same split).
EVAL_RUNS: dict[str, int] = {"rl": 36, "cifar": 26, "lm": 14}          # ~150 cases, natural label mix
RECORD_QUOTA: dict[str, int] = {"rl": 110, "cifar": 90, "lm": 50}        # ~250 recorded calls
RECORD_CONTINUE_SHARE = 0.4    # oversample continue-correct cases in the recorded set
SPLIT_SEED = 20261024

LLM_CAP_USD = 4.00
EST_CALL_USD = {OPUS_MODEL: 0.02, DEV_MODEL: 0.002}
CONCURRENCY = 4

TRAIN_TIME_KEY = {"rl": "train_s", "cifar": "train_time_s", "lm": "train_time_s"}
POST_TRAIN_METRICS = {"test_acc"}   # evaluated after training ends; not a training-time point


def early_stop_step() -> StepType | None:
    return getattr(StepType, "early_stop", None)


# =========================================================================== cases


@dataclass
class Run:
    id: str
    race_id: str
    started_at: float
    ended_at: float
    config: dict[str, Any]
    curve: list[CurvePoint]        # training-time axis
    heldout: float | None
    excluded_from_context: bool    # trap or judge-invalid (the scheduler drops judge-invalid runs)
    is_case: bool                  # non-probe, non-trap
    train_time: float
    t_start: float                 # MetricPoint.t of training start (launch-relative)

    @property
    def final(self) -> float:
        return self.curve[-1].value


def _better(spec: TargetSpec, a: float, b: float) -> bool:
    return a < b if spec.direction == "min" else a > b


def _gain(spec: TargetSpec, a: float, b: float) -> float:
    """How much a beats b (positive = better), in metric units."""
    return (b - a) if spec.direction == "min" else (a - b)


def value_at(curve: list[CurvePoint], t: float) -> float | None:
    v = None
    for p in curve:
        if p.t <= t + 1e-9:
            v = p.value
        else:
            break
    return v


def load_runs(target: str) -> tuple[TargetSpec, dict[str, list[Run]]]:
    db = COLLECTION_DIR / f"collect_{target}.db"
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    exps = con.execute(
        "SELECT e.id, e.race_id, e.started_at, e.ended_at, e.config_json, e.metrics_json, e.is_trap, "
        "e.judge_verdict, p.kind AS probe_kind FROM experiments e LEFT JOIN collection_probes p "
        "ON p.experiment_id = e.id WHERE e.status = 'succeeded' AND e.score IS NOT NULL").fetchall()
    pts: dict[str, list[tuple[int, float, str, float]]] = defaultdict(list)
    for r in con.execute("SELECT experiment_id, step, t, name, value FROM metrics ORDER BY experiment_id, t, step"):
        pts[r[0]].append((int(r[1] or 0), float(r[2]), str(r[3]), float(r[4])))
    con.close()
    base = get_target(target)
    spec = getattr(base, "inner", base).spec
    cm = early_stop_curve_metric(spec)
    by_race: dict[str, list[Run]] = defaultdict(list)
    for e in exps:
        m = json.loads(e["metrics_json"] or "{}")
        tt = m.get(TRAIN_TIME_KEY[target])
        p = [x for x in pts.get(e["id"], []) if x[2] not in POST_TRAIN_METRICS]
        if tt is None or not p:
            continue
        t_end = max(x[1] for x in p)
        t0 = t_end - float(tt)
        curve = [CurvePoint(step=s, t=max(0.0, t - t0), value=v) for s, t, n, v in p
                 if n == cm and math.isfinite(v)]
        if not curve:
            continue
        ho = m.get("heldout_score")
        trap = bool(e["is_trap"])
        by_race[e["race_id"]].append(Run(
            id=e["id"], race_id=e["race_id"], started_at=float(e["started_at"]), ended_at=float(e["ended_at"]),
            config=json.loads(e["config_json"]), curve=curve,
            heldout=float(ho) if ho is not None and math.isfinite(float(ho)) else None,
            excluded_from_context=trap or e["judge_verdict"] == "invalid",
            is_case=(e["probe_kind"] is None and not trap), train_time=float(tt), t_start=t0))
    return spec, by_race


def _median(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def build_cases(target: str) -> list[dict[str, Any]]:
    spec, by_race = load_runs(target)
    cm = early_stop_curve_metric(spec)
    pick_best = min if spec.direction == "min" else max
    out: list[dict[str, Any]] = []
    for race_id, runs in sorted(by_race.items()):
        for run in sorted(runs, key=lambda r: r.started_at):
            if not run.is_case:
                continue
            for f in FRACTIONS:
                elapsed = f * spec.budget_s
                if run.train_time < elapsed:      # finished before this checkpoint: nothing to decide
                    continue
                curve = [p for p in run.curve if p.t <= elapsed + 1e-9]
                if not curve:
                    continue
                wall = run.started_at + run.t_start + elapsed
                ctx = [r for r in runs if r.id != run.id and not r.excluded_from_context and r.ended_at <= wall]
                if len(ctx) < MIN_COMPLETED:
                    continue
                best_run = pick_best(ctx, key=lambda r: r.final)
                at_pt = [v for v in (value_at(r.curve, elapsed) for r in ctx) if v is not None]
                state = EarlyStopState(
                    config=run.config, curve_metric=cm, elapsed_s=elapsed, budget_s=float(spec.budget_s),
                    curve=curve, n_completed=len(ctx), best_final=best_run.final,
                    median_final=_median([r.final for r in ctx]), median_at_point=_median(at_pt),
                    best_run_at_point=value_at(best_run.curve, elapsed))
                ho_ctx = [r.heldout for r in ctx if r.heldout is not None]
                best_ho = pick_best(ho_ctx) if ho_ctx else None
                gain_val = _gain(spec, run.final, best_run.final)
                gain_ho = _gain(spec, run.heldout, best_ho) if (run.heldout is not None and best_ho is not None) else None
                beats_val = gain_val > MARGIN_VAL[target]
                beats_ho = gain_ho is not None and gain_ho > MARGIN_HELDOUT[target]
                latest = curve[-1].value
                median_rule_stop = state.median_at_point is not None and _better(spec, state.median_at_point, latest)
                out.append({
                    "key": f"{target}:{run.id}:{int(f * 100)}",
                    "target": target, "race_id": race_id, "experiment_id": run.id, "fraction": f,
                    "state": state.model_dump(mode="json"),
                    "labels": {
                        "final_val": run.final, "best_val_at_time": best_run.final, "gain_val": gain_val,
                        "beats_val": beats_val, "heldout": run.heldout, "best_heldout_at_time": best_ho,
                        "gain_heldout": gain_ho, "beats_heldout": beats_ho,
                        "stop_correct": not (beats_val or beats_ho),
                    },
                    "median_rule": "stop" if median_rule_stop else "continue",
                    "train_time_s": run.train_time,
                })
    return out


def _h(s: str) -> int:
    return int(hashlib.sha256(f"{SPLIT_SEED}|{s}".encode()).hexdigest()[:12], 16)


def assign_splits(cases: list[dict[str, Any]]) -> None:
    """eval = whole runs (natural label mix); record = quota per target from the other runs,
    continue-correct oversampled to RECORD_CONTINUE_SHARE; the rest unused."""
    rng = random.Random(SPLIT_SEED)
    for t in TARGETS:
        tc = [c for c in cases if c["target"] == t]
        winners = sorted({c["experiment_id"] for c in tc if not c["labels"]["stop_correct"]}, key=_h)
        others = sorted({c["experiment_id"] for c in tc} - set(winners), key=_h)
        # Winner runs are rare (~3% of cases), so half of them (rounded up) go to eval on purpose;
        # the report also gives budget saved reweighted to the natural winner-run share.
        n_win = (len(winners) + 1) // 2
        eval_runs = set(winners[:n_win]) | set(others[: max(0, EVAL_RUNS[t] - n_win)])
        pool = []
        for c in tc:
            if c["experiment_id"] in eval_runs:
                c["split"] = "eval"
            else:
                c["split"] = "unused"
                pool.append(c)
        cont = sorted([c for c in pool if not c["labels"]["stop_correct"]], key=lambda c: _h(c["key"]))
        stop = sorted([c for c in pool if c["labels"]["stop_correct"]], key=lambda c: _h(c["key"]))
        q = RECORD_QUOTA[t]
        n_cont = min(len(cont), round(q * RECORD_CONTINUE_SHARE))
        n_stop = min(len(stop), q - n_cont)
        n_cont = min(len(cont), q - n_stop)
        chosen = cont[:n_cont] + stop[:n_stop]
        rng.shuffle(chosen)
        for i, c in enumerate(chosen):
            c["split"] = "record"
            c["record_order"] = i


def cmd_cases(_: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for t in TARGETS:
        cases += build_cases(t)
    assign_splits(cases)
    with CASES_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, sort_keys=True) + "\n")
    print(f"[cases] {len(cases)} candidate cases -> {CASES_PATH.relative_to(REPO_ROOT)}")
    for t in TARGETS:
        tc = [c for c in cases if c["target"] == t]
        for split in ("record", "eval", "unused"):
            sc = [c for c in tc if c["split"] == split]
            n_stop = sum(c["labels"]["stop_correct"] for c in sc)
            by_f = Counter(c["fraction"] for c in sc)
            mr_acc = sum((c["median_rule"] == "stop") == c["labels"]["stop_correct"] for c in sc) / max(1, len(sc))
            print(f"  {t:5s} {split:6s} n={len(sc):4d} stop_correct={n_stop:4d} continue_correct={len(sc) - n_stop:4d} "
                  f"by_fraction={dict(by_f)} median_rule_acc={mr_acc:.3f}")
        agree = sum(c["labels"]["beats_val"] == c["labels"]["beats_heldout"] for c in tc)
        print(f"  {t:5s} val/heldout label agreement {agree}/{len(tc)}")


def load_cases() -> list[dict[str, Any]]:
    if not CASES_PATH.exists():
        raise SystemExit("no cases yet: run `cases` first")
    return [json.loads(ln) for ln in CASES_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]


# =========================================================================== calls


_HEADER_RE = re.compile(r"\A(?:\s*<!--.*?-->\s*)+", re.DOTALL)


def system_prompt(version: str | None = None) -> tuple[str, str]:
    """(text, fingerprint) of ``version`` (default: CURRENT_VERSION). Uses the registry once the
    step exists; before that (first draft phase) reads the v0 file with the same header
    stripping and normalisation."""
    step = early_stop_step()
    if step is not None:
        return prompts_pkg.load_system_prompt(step, version), prompts_pkg.prompt_fingerprint(step, version)
    raw = (prompts_pkg.PROMPTS_DIR / "early_stop.v0.md").read_text(encoding="utf-8")
    text = _HEADER_RE.sub("", raw, count=1).strip().replace("\r\n", "\n") + "\n"
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def check_frozen() -> list[str]:
    step = early_stop_step()
    if step is None:
        return ["StepType.early_stop does not exist yet"]
    if prompts_pkg.FROZEN.get(step) is None or prompts_pkg.FROZEN.get(step) != prompts_pkg.CURRENT_VERSION.get(step):
        return [f"early_stop prompt {prompts_pkg.CURRENT_VERSION.get(step)} is not frozen"]
    return []


def load_calls(phase: str) -> list[dict[str, Any]]:
    p = CALL_FILES[phase]
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def total_spent() -> float:
    return sum(float(c.get("cost_usd") or 0.0) for ph in CALL_FILES for c in load_calls(ph))


def spec_for(target: str, config: dict[str, Any]) -> TargetSpec:
    """Current spec restricted to the parameters the run actually had (same rule as
    collect_diagnose.spec_view: never list a parameter the run did not have)."""
    base = get_target(target)
    spec = getattr(base, "inner", base).spec
    keys = set(config)
    if set(spec.config_space) <= keys:
        return spec
    return spec.model_copy(update={"config_space": {k: p for k, p in spec.config_space.items() if k in keys}})


def user_for(case: dict[str, Any]) -> str:
    st = EarlyStopState.model_validate(case["state"])
    return build_early_stop_user(spec_for(case["target"], st.config), st)


def select(phase: str, cases: list[dict[str, Any]], n: int | None) -> list[dict[str, Any]]:
    if phase == "record":
        sel = sorted([c for c in cases if c["split"] == "record"], key=lambda c: c["record_order"])
    elif phase == "eval":
        sel = sorted([c for c in cases if c["split"] == "eval"], key=lambda c: _h(c["key"]))
    elif phase == "gate":
        sel = gate_cases(cases)
    else:  # draft: from non-eval runs, balanced over targets and labels
        sel = []
        per = {"rl": 7, "cifar": 7, "lm": 6}
        for t in TARGETS:
            pool = sorted([c for c in cases if c["target"] == t and c["split"] != "eval"], key=lambda c: _h("d" + c["key"]))
            cont = [c for c in pool if not c["labels"]["stop_correct"]][: per[t] // 2]
            stop = [c for c in pool if c["labels"]["stop_correct"]][: per[t] - len(cont)]
            sel += cont + stop
    return sel[: n] if n else sel


def gate_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """40 diverse non-eval cases: per target, winners up to half the quota, the rest
    round-robin over (fraction, median-rule decision) strata. Deterministic."""
    out: list[dict[str, Any]] = []
    for t in TARGETS:
        pool = sorted([c for c in cases if c["target"] == t and c["split"] != "eval"], key=lambda c: _h("g" + c["key"]))
        q = GATE_QUOTA[t]
        pick = [c for c in pool if not c["labels"]["stop_correct"]][: q // 2]
        picked = {c["key"] for c in pick}
        strata: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
        for c in pool:
            if c["key"] not in picked:
                strata[(c["fraction"], c["median_rule"])].append(c)
        keys = sorted(strata)
        while len(pick) < q and any(strata[k] for k in keys):
            for k in keys:
                if strata[k] and len(pick) < q:
                    pick.append(strata[k].pop(0))
        out += pick
    return out


def _is_429(r: Any) -> bool:
    return bool(r.error) and str(r.error).startswith("http 429")


async def send_with_backoff(client: Any, step: StepType, system: str, user: str, sid: str, record: bool,
                            model: str) -> tuple[EarlyStopOutput | None, list[Any]]:
    """call_with_fallback's policy (try, retry once, then fall back to the same model), except
    that HTTP 429 (project B starter credit busy) waits and repeats the same attempt."""
    recs: list[Any] = []
    for extra in ({}, {"is_retry": True}, {"is_fallback": True, "model": model}):
        for wait in (*BACKOFF_429_S, None):
            out, rec = await client.call(step, system, user, EarlyStopOutput, ArmKind.opus, sid, record,
                                         wire_schema=early_stop_wire_schema(), **extra)
            recs.append(rec)
            if out is not None:
                return out, recs
            if not _is_429(rec) or wait is None:
                break
            await asyncio.sleep(wait + random.random())
    return None, recs


def use_project(project: str) -> None:
    """Point this process (only) at a Mubit project's key. The key is never printed."""
    if project == "a":
        return
    import os

    from dotenv import dotenv_values

    key = (dotenv_values(REPO_ROOT / ".env").get(PROJECT_KEY_ENV[project]) or "").strip()
    if not key:
        raise SystemExit(f"{PROJECT_KEY_ENV[project]} is not set in .env")
    os.environ["MUBIT_API_KEY"] = key


async def run_phase(phase: str, n: int | None, yes: bool, dry: bool, concurrency: int,
                    draft_model: str = DEV_MODEL, draft_version: str | None = None,
                    gate_pass: str | None = None, project: str = "a") -> None:
    from backend.agent.mubit_client import MubitClient  # the ONLY LLM client (Mubit gateway)

    cases = load_cases()
    todo_all = select(phase, cases, n)
    prior = load_calls(phase)
    if phase == "gate":
        if gate_pass not in GATE_PASSES:
            raise SystemExit(f"gate needs --pass one of {sorted(GATE_PASSES)}")
        prior = [c for c in prior if c.get("gate_pass") == gate_pass]
    if phase not in ("draft", "gate"):
        draft_version = None   # record / eval always send the frozen CURRENT_VERSION
    system, fp = system_prompt(draft_version)
    model = (draft_model if phase == "draft" else
             GATE_PASSES[gate_pass] if phase == "gate" else OPUS_MODEL)   # draft / gate / eval: record off
    if project != "a" and phase != "record":
        raise SystemExit("project B takes recorded calls only (D38)")
    if project == "b":
        concurrency = min(concurrency, MAX_CONCURRENCY_B)
    prior = [c for c in prior if c.get("project", "a") == project]
    done = {c["case_key"] for c in prior if c.get("final") and c.get("valid") and c.get("prompt_fingerprint") == fp
            and c.get("model") == model}
    todo = [c for c in todo_all if c["key"] not in done]
    step = early_stop_step()
    record = phase == "record"
    if phase == "record":
        gate = OUT_DIR / f"gate_summary_{fp}.json"
        ok = gate.exists() and bool(json.loads(gate.read_text(encoding="utf-8")).get("recordable_variants"))
        if not ok:
            raise SystemExit(f"refusing record: no passing D39 gate for prompt {fp} ({gate.name})")
    if phase in ("record", "eval"):
        problems = check_frozen()
        if problems:
            raise SystemExit(f"refusing {phase}: " + "; ".join(problems))
        if not yes and not dry:
            raise SystemExit(f"{phase} sends paid Opus calls; pass --yes")
    if step is None:
        if phase != "draft":
            raise SystemExit("draft only before StepType.early_stop exists")
        step = StepType.judge_improvement   # same StepConfig (low, 4000); record off; see module doc
    spent = total_spent()
    print(f"[{phase}{'/' + gate_pass if gate_pass else ''}] project={project} model={model} "
          f"record={'full' if record else 'off'} concurrency={concurrency} fingerprint={fp} "
          f"todo={len(todo)} of {len(todo_all)} spent so far=${spent:.4f} cap=${LLM_CAP_USD}")
    if dry:
        for c in todo[:2]:
            print("=" * 80, c["key"], c["labels"])
            print(user_for(c))
        print(f"[{phase}] system prompt ~{len(system) // 4} tokens")
        return
    n_by_target: Counter[str] = Counter()
    for c in prior:
        n_by_target[c["target"]] = max(n_by_target[c["target"]], int(c["session_id"].rsplit("/", 1)[1]))
    use_project(project)
    client = MubitClient(arm_models={ArmKind.opus: model}, fallback_model=model)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    state = {"spent": spent, "stopped": False}

    async def one(c: dict[str, Any]) -> None:
        async with sem:
            async with lock:
                if state["stopped"] or state["spent"] + EST_CALL_USD[model] > LLM_CAP_USD:
                    state["stopped"] = True
                    return
                state["spent"] += EST_CALL_USD[model]   # reserve; corrected below
                n_by_target[c["target"]] += 1
                prefix = SESSION_PREFIX[phase] + (f"/{gate_pass}" if gate_pass else "")
                sid = f"{prefix}/{c['target']}/{n_by_target[c['target']]}"
            out, recs = await send_with_backoff(client, step, system, user_for(c), sid, record, model)
            async with lock:
                state["spent"] += sum(float(r.cost_usd or 0.0) for r in recs) - EST_CALL_USD[model]
                with CALL_FILES[phase].open("a", encoding="utf-8", newline="\n") as f:
                    for r in recs:
                        last = r is recs[-1]
                        f.write(json.dumps({
                            "case_key": c["key"], "target": c["target"], "fraction": c["fraction"],
                            "split": c["split"], "session_id": sid, "step_sent_as": step.value,
                            "mubit_request_id": r.mubit_request_id, "model": r.model, "recorded": r.recorded,
                            "valid": r.valid, "is_retry": r.is_retry, "is_fallback": r.is_fallback, "error": r.error,
                            "latency_ms": r.latency_ms, "input_tokens": r.input_tokens,
                            "cache_creation_tokens": r.cache_creation_tokens, "cache_read_tokens": r.cache_read_tokens,
                            "output_tokens": r.output_tokens, "cost_usd": r.cost_usd, "stop_reason": r.stop_reason,
                            "final": last,
                            "reply": out.model_dump(mode="json") if (out is not None and last) else None,
                            "stop_correct": c["labels"]["stop_correct"], "median_rule": c["median_rule"],
                            "prompt_fingerprint": fp, "project": project, "gate_pass": gate_pass,
                            "ts": time.time()}, default=str) + "\n")
            d = out.decision.value if out is not None else "INVALID"
            gt = "stop" if c["labels"]["stop_correct"] else "continue"
            print(f"[{phase}] {sid:36s} f={c['fraction']:.2f} gt={gt:8s} -> {d:8s} "
                  f"{recs[-1].latency_ms:6.0f}ms ${sum(float(r.cost_usd or 0) for r in recs):.4f} "
                  f"{recs[-1].mubit_request_id} | {('' if out else (recs[-1].error or ''))[:80]}")

    try:
        if todo:
            await one(todo[0])   # warm the system-prompt cache before going parallel (budget.md lever 2)
        await asyncio.gather(*(one(c) for c in todo[1:]))
    finally:
        await client.aclose()
    if state["stopped"]:
        print(f"[{phase}] stopped at the LLM cap")
    print(f"[{phase}] total spent ${total_spent():.4f}")


# =========================================================================== report


def _final_calls(phase: str, fp: str | None = None) -> dict[str, dict[str, Any]]:
    """Last final attempt per case (only calls sent with prompt fingerprint ``fp`` if given)."""
    out: dict[str, dict[str, Any]] = {}
    for c in load_calls(phase):
        if c.get("final") and (fp is None or c.get("prompt_fingerprint") == fp):
            out[c["case_key"]] = c
    return out


def _p50(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def evaluate(cases: list[dict[str, Any]], decide: dict[str, str]) -> dict[str, Any]:
    """Case-level confusion + run-level policy simulation (stop at the first checkpoint whose
    decision is stop). Budget saved = training seconds not spent / training seconds of the runs."""
    cs = [c for c in cases if c["key"] in decide]
    n = len(cs)
    tp = sum(decide[c["key"]] == "stop" and c["labels"]["stop_correct"] for c in cs)
    fp = sum(decide[c["key"]] == "stop" and not c["labels"]["stop_correct"] for c in cs)
    tn = sum(decide[c["key"]] == "continue" and not c["labels"]["stop_correct"] for c in cs)
    fn = sum(decide[c["key"]] == "continue" and c["labels"]["stop_correct"] for c in cs)
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cs:
        by_run[c["experiment_id"]].append(c)
    used = full = 0.0
    stopped_runs = winners_stopped = winners = 0
    grp: dict[tuple[str, bool], list[float]] = defaultdict(lambda: [0.0, 0.0])   # (target, winner) -> [saved, total]
    for _, rc in by_run.items():
        rc.sort(key=lambda c: c["fraction"])
        tt = float(rc[0]["train_time_s"])
        full += tt
        stop_at = next((c for c in rc if decide[c["key"]] == "stop"), None)
        is_winner = any(not c["labels"]["stop_correct"] for c in rc)
        winners += is_winner
        u = tt if stop_at is None else stop_at["fraction"] * float(stop_at["state"]["budget_s"])
        used += u
        g = grp[(rc[0]["target"], is_winner)]
        g[0] += tt - u
        g[1] += tt
        if stop_at is not None:
            stopped_runs += 1
            winners_stopped += not stop_at["labels"]["stop_correct"]
    # Reweight to the natural share of winner runs per target (eval oversamples winners), then
    # pool targets by their training seconds in this set.
    nat_saved = nat_total = 0.0
    for t in {k[0] for k in grp}:
        share = NATURAL_WINNER_SHARE.get(t)
        w, lo = grp.get((t, True), [0.0, 0.0]), grp.get((t, False), [0.0, 0.0])
        tot = w[1] + lo[1]
        if share is None or tot == 0:
            continue
        fw = w[0] / w[1] if w[1] else 0.0
        fl = lo[0] / lo[1] if lo[1] else 0.0
        nat_saved += tot * (share * fw + (1 - share) * fl)
        nat_total += tot
    return {
        "cases": n, "accuracy": (tp + tn) / n if n else None,
        "stop_and_right": tp, "stop_but_winner": fp, "continue_and_winner": tn, "continue_but_loser": fn,
        "winner_cases": tn + fp, "winner_cases_wrongly_stopped": fp,
        "runs": len(by_run), "runs_stopped": stopped_runs, "winner_runs": winners,
        "winner_runs_wrongly_stopped": winners_stopped,
        "training_s_saved": full - used, "training_s_total": full,
        "budget_saved_pct": 100.0 * (full - used) / full if full else None,
        "budget_saved_pct_natural_mix": 100.0 * nat_saved / nat_total if nat_total else None,
    }


NATURAL_WINNER_SHARE: dict[str, float] = {}


def natural_winner_share(cases: list[dict[str, Any]]) -> dict[str, float]:
    out = {}
    for t in TARGETS:
        runs: dict[str, bool] = {}
        for c in cases:
            if c["target"] == t:
                runs[c["experiment_id"]] = runs.get(c["experiment_id"], False) or not c["labels"]["stop_correct"]
        if runs:
            out[t] = sum(runs.values()) / len(runs)
    return out


def cmd_report(_: argparse.Namespace) -> None:
    cases = load_cases()
    NATURAL_WINNER_SHARE.update(natural_winner_share(cases))
    by_key = {c["key"]: c for c in cases}
    summary: dict[str, Any] = {"prompt_fingerprint": system_prompt()[1], "n_candidate_cases": len(cases),
                               "margins": {"val": MARGIN_VAL, "heldout": MARGIN_HELDOUT},
                               "splits": dict(Counter(c["split"] for c in cases)),
                               "natural_winner_run_share": dict(NATURAL_WINNER_SHARE)}
    for phase in ("draft", "record", "eval"):
        fp = system_prompt()[1]
        calls = [c for c in load_calls(phase) if c.get("prompt_fingerprint") == fp]
        if not calls:
            continue
        fin = _final_calls(phase, fp)
        dec = {k: v["reply"]["decision"] for k, v in fin.items() if v.get("reply")}
        pc = [by_key[k] for k in fin if k in by_key]
        lat = [float(c["latency_ms"]) for c in calls if c.get("valid")]
        s = {
            "calls": len(calls), "cases": len(fin), "valid_final": sum(1 for v in fin.values() if v.get("valid")),
            "retries": sum(1 for c in calls if c.get("is_retry")), "fallbacks": sum(1 for c in calls if c.get("is_fallback")),
            "cost_usd": sum(float(c.get("cost_usd") or 0) for c in calls),
            "p50_latency_ms": _p50(lat),
            "p90_latency_ms": (float(statistics.quantiles(lat, n=10)[-1]) if len(lat) >= 10 else None),
            "first_request_id": next((c["mubit_request_id"] for c in calls if c.get("mubit_request_id")), None),
            "models": dict(Counter(c["model"] for c in calls)),
            "recorded": dict(Counter(bool(c.get("recorded")) for c in calls)),
            "by_target": dict(Counter(by_key[k]["target"] for k in fin if k in by_key)),
            "labels": dict(Counter("stop_correct" if by_key[k]["labels"]["stop_correct"] else "continue_correct"
                                   for k in fin if k in by_key)),
            "decisions": dict(Counter(dec.values())),
            "llm": evaluate(pc, dec),
            "median_rule": evaluate(pc, {c["key"]: c["median_rule"] for c in pc}),
            "always_continue": evaluate(pc, {c["key"]: "continue" for c in pc}),
            "always_stop_at_25pct": evaluate(pc, {c["key"]: "stop" for c in pc}),
        }
        for t in TARGETS:
            tc = [c for c in pc if c["target"] == t]
            s[f"llm_{t}"] = evaluate(tc, dec)
            s[f"median_rule_{t}"] = evaluate(tc, {c["key"]: c["median_rule"] for c in tc})
        summary[phase] = s
        print(f"== {phase}: {s['cases']} cases, {s['calls']} calls, valid {s['valid_final']}, ${s['cost_usd']:.4f}, "
              f"p50 {s['p50_latency_ms']} ms, first {s['first_request_id']}")
        for name in ("llm", "median_rule", "always_continue", "always_stop_at_25pct"):
            e = s[name]
            print(f"   {name:16s} acc={e['accuracy']} stop_but_winner={e['stop_but_winner']}/{e['winner_cases']} "
                  f"winner_runs_stopped={e['winner_runs_wrongly_stopped']}/{e['winner_runs']} "
                  f"saved={e['budget_saved_pct']}% runs_stopped={e['runs_stopped']}/{e['runs']}")
    # The rule on the full candidate set (no LLM, free)
    summary["median_rule_all_cases"] = evaluate(cases, {c["key"]: c["median_rule"] for c in cases})
    summary["gate_summaries"] = sorted(p.name for p in OUT_DIR.glob("gate_summary_*.json"))
    summary["total_llm_usd"] = total_spent()
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"[report] -> {SUMMARY_PATH.relative_to(REPO_ROOT)}; total LLM ${summary['total_llm_usd']:.4f}")


# Candidate output formats for the D39 gate, as maps from the v1 reply to the graded answer.
# Merges are measured post hoc on the same replies (an indication: a prompt that only offers the
# merged codes may behave differently and would need its own gate run).
GATE_VARIANTS: dict[str, Any] = {
    "full_output": lambda r: tuple(sorted(r.items())),
    "merged_4": lambda r: (r["decision"], {"on_best_pace": "promising", "closing_gap": "promising",
                                           "too_early_to_tell": "too_early", "behind_median": "behind",
                                           "plateau_below_best": "behind", "diverging": "broken"}[r["reason_code"]]),
    "merged_2_stop_detail": lambda r: (r["decision"], "broken" if r["reason_code"] == "diverging" else "-"),
    "decision_only": lambda r: (r["decision"],),
}
GATE_CEILING = 0.97      # Opus self-agreement needed on the graded output (exam: >= 95% vs stored answer)
GATE_MAX_CLASS = 0.70
GATE_MIN_INPUTS = 150


def cmd_gate_report(a: argparse.Namespace) -> None:
    cases = load_cases()
    fp = system_prompt(getattr(a, "version", None))[1]
    calls = [c for c in load_calls("gate") if c.get("prompt_fingerprint") == fp]
    by_pass: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for c in calls:
        if c.get("final") and c.get("reply"):
            by_pass[c["gate_pass"]][c["case_key"]] = c["reply"]
    o1, o2, hk = by_pass.get("opus1", {}), by_pass.get("opus2", {}), by_pass.get("haiku", {})
    common = sorted(set(o1) & set(o2))
    common_h = sorted(set(o1) & set(hk))
    res: dict[str, Any] = {"prompt_fingerprint": fp, "n_opus_pairs": len(common), "n_haiku_pairs": len(common_h),
                           "calls": len(calls), "cost_usd": sum(float(c.get("cost_usd") or 0) for c in calls),
                           "first_request_id": next((c["mubit_request_id"] for c in calls if c.get("mubit_request_id")), None),
                           "recorded_calls": sum(1 for c in calls if c.get("recorded")), "variants": {}}
    for name, f in GATE_VARIANTS.items():
        if name.startswith("merged") and not all("reason_code" in o1[k] for k in common):
            continue   # only meaningful for v1 replies
        a = sum(f(o1[k]) == f(o2[k]) for k in common) / max(1, len(common))
        h = sum(f(o1[k]) == f(hk[k]) for k in common_h) / max(1, len(common_h))
        cls = Counter(f(o1[k]) for k in common) + Counter(f(o2[k]) for k in common)
        top = max(cls.values()) / max(1, sum(cls.values())) if cls else None
        res["variants"][name] = {"opus_self_agreement": a, "haiku_vs_opus1": h,
                                 "classes_opus": {"|".join(map(str, k)): v for k, v in cls.items()},
                                 "largest_class_share": top, "passes_ceiling": a >= GATE_CEILING,
                                 "passes_balance": top is not None and top <= GATE_MAX_CLASS}
    res["disagreements_full"] = [{"case": k, "opus1": o1[k], "opus2": o2[k]} for k in common
                                 if GATE_VARIANTS["full_output"](o1[k]) != GATE_VARIANTS["full_output"](o2[k])]
    res["inconsistent_pairs_opus"] = sum(
        1 for d in (o1, o2) for r in d.values()
        if "reason_code" in r and V1_REASON_DECISION[r["reason_code"]] != r["decision"])
    users = {hashlib.sha256(user_for(c).encode("utf-8")).hexdigest() for c in cases}
    res["distinct_candidate_inputs"] = len(users)
    res["passes_min_inputs"] = len(users) >= GATE_MIN_INPUTS
    # Only the prompt's own full output is graded by Mubit; other variants are indications for a
    # new version, which needs its own gate run.
    v = res["variants"]["full_output"]
    res["recordable_variants"] = (["full_output"] if v["passes_ceiling"] and v["passes_balance"]
                                  and res["passes_min_inputs"] else [])
    res["indicative_passing_variants"] = [k for k, x in res["variants"].items() if x["passes_ceiling"] and x["passes_balance"]]
    out = OUT_DIR / f"gate_summary_{fp}.json"
    out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[gate] fingerprint {fp}: opus pairs {len(common)}, haiku pairs {len(common_h)}, "
          f"{len(calls)} calls (recorded {res['recorded_calls']}), ${res['cost_usd']:.4f}, first {res['first_request_id']}")
    for n, v in res["variants"].items():
        print(f"  {n:22s} opus self-agreement {v['opus_self_agreement']:.3f}  haiku vs opus1 {v['haiku_vs_opus1']:.3f}  "
              f"largest class {v['largest_class_share']:.2f}  ceiling {'PASS' if v['passes_ceiling'] else 'FAIL'}  "
              f"balance {'PASS' if v['passes_balance'] else 'FAIL'}  classes {v['classes_opus']}")
    print(f"  distinct candidate inputs {len(users)} (need >= {GATE_MIN_INPUTS}); inconsistent code/decision pairs "
          f"{res['inconsistent_pairs_opus']}; recordable: {res['recordable_variants'] or 'NONE'}")
    for d in res["disagreements_full"]:
        print(f"  disagree {d['case']}: {d['opus1']} vs {d['opus2']}")
    print(f"[gate] -> {out.relative_to(REPO_ROOT)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cases", help="build candidate cases from the collection DBs (read-only)")
    for ph in ("draft", "record", "eval", "gate"):
        p = sub.add_parser(ph)
        p.add_argument("--n", type=int, default=None)
        p.add_argument("--yes", action="store_true")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--concurrency", type=int, default=CONCURRENCY)
        if ph == "gate":
            p.add_argument("--pass", dest="gate_pass", choices=sorted(GATE_PASSES), required=True)
        if ph == "record":
            p.add_argument("--project", choices=sorted(PROJECT_KEY_ENV), default="a",
                           help="Mubit project whose key the recorded calls use (D38: b)")
        if ph in ("draft", "gate"):
            p.add_argument("--model", default=DEV_MODEL, choices=[DEV_MODEL, OPUS_MODEL],
                           help="draft model (record is always off in draft)")
            p.add_argument("--version", default=None, help="prompt version file to draft (default CURRENT_VERSION)")
    sub.add_parser("report")
    g = sub.add_parser("gate-report", help="D39 gate: Opus self-agreement, Haiku agreement, balance, distinct inputs")
    g.add_argument("--version", default=None, help="prompt version to report (default CURRENT_VERSION)")
    a = ap.parse_args(argv)
    if a.cmd == "cases":
        cmd_cases(a)
    elif a.cmd == "report":
        cmd_report(a)
    elif a.cmd == "gate-report":
        cmd_gate_report(a)
    else:
        asyncio.run(run_phase(a.cmd, a.n, a.yes, a.dry_run, a.concurrency, getattr(a, "model", DEV_MODEL),
                              getattr(a, "version", None), getattr(a, "gate_pass", None),
                              getattr(a, "project", "a")))


if __name__ == "__main__":
    main()
