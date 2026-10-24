"""Targeted example collection for the frozen ``diagnose_run`` Mubit task.

Overnight collection (scripts/collect_examples.py) produced 0 diagnose_run examples: none of
~620 runs failed (results_log R20). This script fills that gap with a deliberate mix of

- **real failures**: real runs on the DEPLOYED Modal app (``ModalRunner(deployed=True)``)
  with settings inside each target's config space that genuinely fail (aggressive lr with
  no warmup, a per-run budget too short for one PPO update, ...). Only what the runner
  returned is used: real status, real error, real log tail. Nothing is fabricated.
- **healthy runs** (``no_failure`` candidates): succeeded runs from the overnight collection
  DBs (``data/collection/collect_{rl,cifar,lm}.db``, opened read-only), including trap runs
  and poorly scoring runs, plus succeeded probe runs from the step above. An adapter trained
  only on failures invents problems (R9), so healthy runs are part of the mix on purpose.

Every diagnose call is built exactly like the race loop builds it: an ``_Exp`` (scheduler
state) -> ``.snapshot()`` (ground-truth redaction: ``agent_visible_metrics`` /
``agent_visible_logs``) -> ``LLMResearcher.diagnose`` (``RepoPrompts.diagnose_user`` ->
``prompts.build.build_diagnose_user``; frozen system prompt ``load_system_prompt``; frozen
wire schema ``wire_schema_for(diagnose_run)``; ``MubitClient`` default step config, i.e.
effort "low" as in every race; same retry/fallback policy), with ``record=True`` (full) and
the Mubit session id ``collect-diag-2026-10-24/{target}/{n}``.

Spec block: each message shows the config space the run actually ran with. Overnight DB runs
(and runs on a deployed image older than the local source) predate some parameters; for
those the spec is the current local spec restricted to the run's parameters
(``spec_view``), so the message never lists a parameter the run did not have.

    # 1. real runs on the deployed Modal app (GPU cap $0.60, 2 L4 + 4 CPU at once)
    python -m uv run --extra ml python scripts/collect_diagnose.py runs --plan explore
    python -m uv run --extra ml python scripts/collect_diagnose.py runs --plan main

    # 2. what would be sent (no network)
    python -m uv run --extra ml python scripts/collect_diagnose.py diagnose --dry-run

    # 3. the recorded Opus calls (LLM cap $2), resumable
    python -m uv run --extra ml python scripts/collect_diagnose.py diagnose --yes

Outputs (all new files, git-ignored data dir):
- ``data/collection/diag_runs-2026-10-24.jsonl``: one line per real run (config, seed,
  budget, Modal call id, status, error, metrics, logs tail, GPU $ estimate).
- ``data/collection/collect-diag-2026-10-24.jsonl``: one line per diagnose call (case key,
  source, Mubit request id, session id, reply, tokens, latency, $ estimate).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import sqlite3
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agent import prompts as prompts_pkg  # noqa: E402
from backend.agent.llm_base import OPUS_MODEL  # noqa: E402
from backend.agent.loop import LLMResearcher, load_prompts  # noqa: E402
from backend.core.types import ArmKind, ExperimentStatus, LLMCallRecord, RunResult, StepType  # noqa: E402
from backend.scheduler import ExperimentSnapshot, _Exp  # noqa: E402
from workers.targets.base import TargetSpec, get_target  # noqa: E402

DATE = "2026-10-24"
SESSION_PREFIX = f"collect-diag-{DATE}"
DATA_DIR = REPO_ROOT / "data" / "collection"
RUNS_PATH = DATA_DIR / f"diag_runs-{DATE}.jsonl"
CALLS_PATH = DATA_DIR / f"collect-diag-{DATE}.jsonl"
TARGETS = ("cifar", "lm", "rl")

GPU_CAP_USD = 0.60
LLM_CAP_USD = 2.00
MAX_L4 = 2
MAX_CPU = 4
# Conservative $ estimate of one diagnose call (Opus, ~2.5k input incl. cache write, ~400 out):
# a call is not started unless the cap still covers it.
EST_CALL_USD = 0.05

STALE_RE = re.compile(r"unknown params: \[([^\]]*)\]")


# =========================================================================== spec views


def spec_view(spec: TargetSpec, keys: set[str] | None) -> TargetSpec:
    """``spec`` restricted to the parameters ``keys`` (config-space order kept). Used when a
    run ran with fewer parameters than the local spec has (old DB runs, stale deployment)."""
    if keys is None or set(spec.config_space) <= keys:
        return spec
    space = {k: p for k, p in spec.config_space.items() if k in keys}
    return spec.model_copy(update={"config_space": space})


@dataclass
class _TargetView:
    """Stand-in registry entry with a restricted spec (this process only). ModalRunner.launch
    validates against ``get_target(name).spec`` and sends the validated config, so this makes
    it send only the parameters the deployed image knows."""

    spec: TargetSpec
    inner: Any

    def run(self, *a: Any, **kw: Any) -> Any:  # never called on the host
        return self.inner.run(*a, **kw)


def install_view(target: str, keys: set[str]) -> TargetSpec:
    from workers.targets import base as tb

    real = get_target(target)
    inner = getattr(real, "inner", real)
    view = spec_view(inner.spec, keys)
    tb.TARGETS[target] = _TargetView(spec=view, inner=inner)  # type: ignore[assignment]
    return view


# =========================================================================== snapshots


def make_snapshot(*, config: dict[str, Any], seed: int, status: ExperimentStatus, score: float | None,
                  error: str | None, metrics: dict[str, float], logs_tail: list[str],
                  started_at: float | None, ended_at: float | None) -> ExperimentSnapshot:
    """Same state the scheduler keeps for a finished run (``ArmRunner._on_run_end``) and the
    same redacted view it hands the researcher (``_Exp.snapshot``)."""
    e = _Exp(id=str(uuid.uuid4()), arm_id="collect-diag", config=dict(config), hypothesis="", seed=int(seed),
             proposed_at=float(started_at or 0.0))
    if status == ExperimentStatus.succeeded and (score is None or not math.isfinite(float(score))):
        status, score, error = ExperimentStatus.failed, None, error or f"non-finite score {score!r}"
    if status != ExperimentStatus.succeeded:
        score = None
    e.status, e.score, e.error = status, (float(score) if score is not None else None), error
    e.metrics = dict(metrics)
    e.logs_tail = list(logs_tail[-50:])
    e.started_at, e.ended_at = started_at, ended_at
    e.processed = True
    return e.snapshot()


# =========================================================================== real runs


@dataclass(frozen=True)
class Probe:
    target: str
    name: str
    overrides: dict[str, Any]
    budget_s: int
    seeds: tuple[int, ...] = (101,)
    expect: str = "fail"   # fail | ok (an informative run that may well succeed)


def _plan_explore() -> list[Probe]:
    """Cheap first pass: which in-space settings really fail on the deployed code?"""
    return [
        # RL (CPU): budget shorter than JIT compile -> TimeoutError; extreme PPO settings -> NaN?
        Probe("rl", "rl_budget1_default", {}, 1),
        Probe("rl", "rl_budget2_big", {"num_envs": 128, "num_steps": 512, "hidden_size": 256,
                                        "update_epochs": 10, "num_minibatches": 32}, 2),
        Probe("rl", "rl_lr_max_unscaled", {"lr": 0.01, "reward_scale": 1.0, "clip_eps": 0.5, "update_epochs": 10,
                                            "num_minibatches": 32, "ent_coef": 0.05, "num_envs": 4,
                                            "num_steps": 16, "gamma": 0.9999, "gae_lambda": 1.0}, 30),
        Probe("rl", "rl_lr_max_big", {"lr": 0.01, "reward_scale": 1.0, "clip_eps": 0.5, "update_epochs": 10,
                                       "num_minibatches": 1, "hidden_size": 256, "num_envs": 128,
                                       "num_steps": 512}, 30),
        # CIFAR (L4): lr 4x default, max momentum, no warmup, largest batch (sum-reduced loss).
        Probe("cifar", "cifar_lr40_m95_bs2048_nowarm", {"lr": 40.0, "momentum": 0.95, "warmup_frac": 0.0,
                                                         "batch_size": 2048, "label_smoothing": 0.0,
                                                         "weight_decay": 0.001, "epochs": 4.0}, 20),
        Probe("cifar", "cifar_lr40_m95_bs2048_w2", {"lr": 40.0, "momentum": 0.95, "warmup_frac": 0.0,
                                                     "batch_size": 2048, "width": 2.0, "label_smoothing": 0.0,
                                                     "epochs": 4.0}, 20),
        # LM (L4): max lr, no warmup, largest model / context / batch.
        Probe("lm", "lm_lr01_nowarm_max", {"lr": 0.01, "warmup_frac": 0.0, "n_layer": 8, "n_embd": 512,
                                           "n_head": 8, "block_size": 512, "batch_size": 128,
                                           "weight_decay": 0.0}, 20),
        Probe("lm", "lm_lr01_nowarm_deep_narrow", {"lr": 0.01, "warmup_frac": 0.0, "n_layer": 8, "n_embd": 64,
                                                   "n_head": 1, "block_size": 64, "batch_size": 8,
                                                   "weight_decay": 0.0}, 20),
    ]


def _plan_main() -> list[Probe]:
    """After the explore pass (diag_runs-2026-10-24.jsonl, plan=explore), on the deployed image:
    - rl at lr=0.01 / reward_scale=1 / clip 0.5 / 10 epochs x 32 minibatches -> NaN
      (FloatingPointError at update ~257); budget_s=1 < JIT compile (~1.6 s) -> TimeoutError.
    - cifar at lr=40 / momentum 0.95 / no warmup / bs 2048 collapses to chance (test_acc 0.10)
      but never raises; lm at lr=0.01 / no warmup only trains slowly (grad clip 1.0). Neither GPU
      target fails with an exception inside its config space, so their probes are succeeded runs
      with broken or poor training (expect="ok"), diagnosed as such.
    """
    nan = {"lr": 0.01, "reward_scale": 1.0, "clip_eps": 0.5, "update_epochs": 10, "num_minibatches": 32,
           "ent_coef": 0.05, "num_envs": 4, "num_steps": 16, "gamma": 0.9999, "gae_lambda": 1.0}
    return [
        # ---- rl: NaN divergence family (vary the config so the logs and configs differ)
        Probe("rl", "rl_nan_base", nan, 30, seeds=(102, 103)),
        Probe("rl", "rl_nan_ent0", {**nan, "ent_coef": 0.0}, 30),
        Probe("rl", "rl_nan_env8_steps32", {**nan, "num_envs": 8, "num_steps": 32}, 30),
        Probe("rl", "rl_nan_h256", {**nan, "hidden_size": 256}, 30),
        Probe("rl", "rl_nan_lr005", {**nan, "lr": 0.005}, 30),
        Probe("rl", "rl_nan_clip02_ep4", {**nan, "clip_eps": 0.2, "update_epochs": 4}, 30),
        Probe("rl", "rl_nan_gamma99", {**nan, "gamma": 0.99, "gae_lambda": 0.95}, 30),
        Probe("rl", "rl_nan_anneal", {**nan, "anneal_lr": True}, 30),
        Probe("rl", "rl_nan_rs01", {**nan, "reward_scale": 0.1}, 30, expect="ok"),
        Probe("rl", "rl_nan_mb8_h16", {**nan, "num_minibatches": 8, "hidden_size": 16}, 30),
        Probe("rl", "rl_nan_env16_steps64_lr003", {**nan, "num_envs": 16, "num_steps": 64, "lr": 0.003}, 30),
        # ---- rl: per-run budget shorter than setup + JIT compile -> TimeoutError
        Probe("rl", "rl_budget1_default", {}, 1, seeds=(102,)),
        Probe("rl", "rl_budget1_huge", {"num_envs": 128, "num_steps": 512, "hidden_size": 256,
                                         "update_epochs": 10, "num_minibatches": 32}, 1),
        Probe("rl", "rl_budget1_tuned", {"lr": 0.001, "anneal_lr": True, "num_envs": 16, "num_steps": 256,
                                          "gamma": 0.95, "gae_lambda": 0.9, "update_epochs": 10,
                                          "num_minibatches": 8, "reward_scale": 0.1}, 1),
        Probe("rl", "rl_budget1_small", {"num_envs": 4, "num_steps": 16, "hidden_size": 16}, 1),
        Probe("rl", "rl_budget1_lr3e3_env32", {"lr": 0.003, "num_envs": 32, "num_steps": 128,
                                                "hidden_size": 128}, 1),
        Probe("rl", "rl_budget2_ep10_mb32", {"update_epochs": 10, "num_minibatches": 32, "hidden_size": 128}, 2),
        Probe("rl", "rl_budget2_env64", {"num_envs": 64, "num_steps": 256, "hidden_size": 128,
                                          "update_epochs": 8, "num_minibatches": 16}, 2),
        # ---- rl: healthy-but-poor (tiny lr) for the no_failure / slow side
        Probe("rl", "rl_lr1e5", {"lr": 1e-5}, 30, expect="ok"),
        # ---- cifar: collapse to chance at extreme lr (succeeded runs), truncation by budget
        Probe("cifar", "cifar_collapse_bs1024", {"lr": 40.0, "momentum": 0.95, "warmup_frac": 0.0,
                                                  "batch_size": 1024, "label_smoothing": 0.0, "epochs": 4.0}, 20,
              expect="ok"),
        Probe("cifar", "cifar_collapse_m90_ls02", {"lr": 40.0, "momentum": 0.9, "warmup_frac": 0.0,
                                                    "batch_size": 2048, "epochs": 4.0}, 20, expect="ok"),
        Probe("cifar", "cifar_collapse_lr25", {"lr": 25.0, "momentum": 0.95, "warmup_frac": 0.0,
                                                "batch_size": 2048, "epochs": 4.0}, 20, expect="ok"),
        Probe("cifar", "cifar_collapse_bs512_w05", {"lr": 40.0, "momentum": 0.95, "warmup_frac": 0.0,
                                                     "batch_size": 512, "width": 0.5, "epochs": 6.0}, 20,
              expect="ok"),
        Probe("cifar", "cifar_budget3_default", {}, 3, expect="ok"),
        Probe("cifar", "cifar_trunc_ep30_w2_bs256", {"epochs": 30.0, "width": 2.0, "batch_size": 256}, 20,
              expect="ok"),
        # ---- lm: max lr without warmup (slow / unstable), tiny lr (stall), heavy dropout
        Probe("lm", "lm_lr01_nowarm_mid_h1", {"lr": 0.01, "warmup_frac": 0.0, "n_layer": 6, "n_embd": 384,
                                              "n_head": 1, "block_size": 256, "batch_size": 64,
                                              "weight_decay": 0.0}, 30, expect="ok"),
        Probe("lm", "lm_lr01_nowarm_max_30s", {"lr": 0.01, "warmup_frac": 0.0, "n_layer": 8, "n_embd": 512,
                                               "n_head": 8, "block_size": 512, "batch_size": 128,
                                               "weight_decay": 0.0}, 30, seeds=(102,), expect="ok"),
        Probe("lm", "lm_lr1e4_tiny", {"lr": 0.0001, "n_layer": 1, "n_embd": 64, "n_head": 1, "block_size": 64,
                                      "batch_size": 8}, 20, expect="ok"),
        Probe("lm", "lm_dropout05_big", {"dropout": 0.5, "n_layer": 8, "n_embd": 512, "block_size": 512,
                                         "batch_size": 128, "lr": 0.003}, 20, expect="ok"),
        Probe("lm", "lm_lr01_nowarm_wd03", {"lr": 0.01, "warmup_frac": 0.0, "n_layer": 4, "n_embd": 256,
                                            "n_head": 1, "block_size": 128, "batch_size": 128,
                                            "weight_decay": 0.3}, 20, expect="ok"),
    ]


def _plan_more() -> list[Probe]:
    """Second rl batch (main gave 13 real failures). budget_s=1 only fails when setup + compile
    exceed 1 s (main: small/default configs sometimes compile in <1 s and succeed), so the
    timeout probes use large rollouts / many epochs x minibatches."""
    nan = {"lr": 0.01, "reward_scale": 1.0, "clip_eps": 0.5, "update_epochs": 10, "num_minibatches": 32,
           "ent_coef": 0.05, "num_envs": 4, "num_steps": 16, "gamma": 0.9999, "gae_lambda": 1.0}
    big = {"num_envs": 128, "num_steps": 512, "update_epochs": 10, "num_minibatches": 32}
    return [
        Probe("rl", "rl_nan_h32_ep6", {**nan, "hidden_size": 32, "update_epochs": 6}, 30),
        Probe("rl", "rl_nan_h128_g95", {**nan, "hidden_size": 128, "gamma": 0.95, "gae_lambda": 0.9}, 30),
        Probe("rl", "rl_nan_steps64_mb16", {**nan, "num_steps": 64, "num_minibatches": 16}, 30),
        Probe("rl", "rl_nan_lr007_ent002", {**nan, "lr": 0.007, "ent_coef": 0.02}, 30),
        Probe("rl", "rl_nan_clip03_rs03", {**nan, "clip_eps": 0.3, "reward_scale": 0.3}, 30),
        Probe("rl", "rl_nan_env32_steps16", {**nan, "num_envs": 32, "num_steps": 16}, 30),
        Probe("rl", "rl_nan_ep8_gae09", {**nan, "update_epochs": 8, "gae_lambda": 0.9}, 30, seeds=(104,)),
        Probe("rl", "rl_nan_base_s105", nan, 30, seeds=(105,)),
        Probe("rl", "rl_budget1_big_h256", {**big, "hidden_size": 256}, 1),
        Probe("rl", "rl_budget1_big_h64_lr1e3", {**big, "hidden_size": 64, "lr": 0.001, "anneal_lr": True}, 1),
        Probe("rl", "rl_budget1_big_g999", {**big, "hidden_size": 128, "gamma": 0.999, "ent_coef": 0.01}, 1),
        Probe("rl", "rl_budget1_env64_ep10", {"num_envs": 64, "num_steps": 256, "update_epochs": 10,
                                               "num_minibatches": 32, "hidden_size": 128}, 1),
        Probe("rl", "rl_budget1_env128_steps128", {"num_envs": 128, "num_steps": 128, "update_epochs": 8,
                                                    "num_minibatches": 16, "hidden_size": 256,
                                                    "reward_scale": 0.1}, 1),
        Probe("rl", "rl_budget2_big_h256", {**big, "hidden_size": 256, "lr": 0.0003}, 2),
    ]


PLANS = {"explore": _plan_explore, "main": _plan_main, "more": _plan_more}


class RunStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: list[dict[str, Any]] = []
        if path.exists():
            self.rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def gpu_usd(self) -> float:
        return sum(float(r.get("gpu_usd") or 0.0) for r in self.rows if r.get("hardware") != "CPU")

    def cpu_usd(self) -> float:
        return sum(float(r.get("gpu_usd") or 0.0) for r in self.rows if r.get("hardware") == "CPU")


def is_artefact(row: dict[str, Any]) -> str | None:
    """Why a run is not a genuine training outcome (never diagnosed), or None."""
    err = row.get("error") or ""
    if row.get("infra"):
        return "infra"
    if STALE_RE.search(err):
        return "stale deployment (unknown params)"
    if "validate_config" in "\n".join(row.get("logs_tail") or []) and err.startswith("ValueError"):
        return "deployed spec rejected the config"
    if row.get("status") == "cancelled":
        return "cancelled"
    return None


async def detect_deployed_keys(target: str) -> set[str] | None:
    """Parameters of the local spec the DEPLOYED image does not know, found cheaply: run the
    target's config validation in a CPU container (validate_config runs before any training;
    GPU targets then fail at the torch import, which costs a CPU container for seconds).
    Returns the deployed parameter set, or None if the deployment knows every local key."""
    from workers.runners.modal_runner import ModalRunner

    spec = get_target(target).spec
    if spec.hardware == "CPU":
        runner = ModalRunner({"CPU": 1}, deployed=True)
    else:
        runner = ModalRunner({"CPU": 1}, deployed=True, hardware_for={target: "CPU"})
    h = await runner.launch(target, spec.default_config(), 0, 1, lambda _p: None)
    res = await runner.wait(h)
    m = STALE_RE.search(res.error or "")
    print(f"[detect] {target}: status={res.status.value} error={(res.error or '')[:160]!r}")
    if not m:
        return None
    unknown = {s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()}
    return set(spec.config_space) - unknown


async def run_plan(plan: str, *, only: set[str] | None, dry: bool) -> None:
    from workers.runners.modal_runner import ModalRunner

    store = RunStore(RUNS_PATH)
    probes = [p for p in PLANS[plan]() if not only or p.name in only or p.target in only]
    jobs = [(p, s) for p in probes for s in p.seeds]
    print(f"[runs] plan={plan} jobs={len(jobs)} GPU $ so far={store.gpu_usd():.4f} (cap {GPU_CAP_USD})")
    if dry:
        for p, s in jobs:
            print("  ", p.target, p.name, p.budget_s, s, p.overrides)
        return

    views: dict[str, TargetSpec] = {}
    for t in sorted({p.target for p, _ in jobs}):
        keys = await detect_deployed_keys(t)
        views[t] = install_view(t, keys) if keys is not None else get_target(t).spec
        dropped = sorted(set(getattr(get_target(t), "inner", get_target(t)).spec.config_space) - set(views[t].config_space))
        print(f"[runs] {t}: deployed spec {'= local' if not dropped else 'lacks ' + str(dropped)}")

    runner = ModalRunner({"L4": MAX_L4, "CPU": MAX_CPU}, deployed=True)
    sems = {"L4": asyncio.Semaphore(MAX_L4), "CPU": asyncio.Semaphore(MAX_CPU)}
    in_flight_est = {"usd": 0.0}
    lock = asyncio.Lock()

    async def one(p: Probe, seed: int) -> None:
        spec = views[p.target]
        hw = spec.hardware
        cfg = {k: v for k, v in {**spec.default_config(), **p.overrides}.items() if k in spec.config_space}
        cfg = spec.validate_config(cfg)
        est = (p.budget_s + 90) * runner.price_per_second(hw)  # worst case incl. startup + eval
        async with sems[hw]:
            async with lock:
                if hw != "CPU" and store.gpu_usd() + in_flight_est["usd"] + est > GPU_CAP_USD:
                    print(f"[runs] skip {p.name} seed={seed}: GPU cap")
                    return
                if hw != "CPU":
                    in_flight_est["usd"] += est
            row: dict[str, Any] = {"plan": plan, "probe": p.name, "expect": p.expect, "target": p.target,
                                   "config": cfg, "seed": seed, "budget_s": p.budget_s, "hardware": hw,
                                   "spec_keys": list(spec.config_space)}
            t0 = time.time()
            try:
                h = await runner.launch(p.target, cfg, seed, p.budget_s, lambda _pt: None)
                row["run_ref"] = h.run_ref
                res: RunResult = await runner.wait(h)
            except Exception as e:  # noqa: BLE001 - infra, never diagnosed
                row.update(status="failed", error=f"infra: {e!r}", infra=True, started_at=t0, ended_at=time.time(),
                           metrics={}, logs_tail=[], score=None, gpu_usd=(time.time() - t0) * runner.price_per_second(hw))
            else:
                row.update(status=res.status.value, score=res.score, error=res.error, metrics=res.metrics,
                           logs_tail=res.logs_tail, is_trap=bool(res.is_trap), started_at=res.started_at,
                           ended_at=res.ended_at,
                           gpu_usd=max(0.0, (res.ended_at or time.time()) - (res.started_at or t0))
                           * runner.price_per_second(hw))
            finally:
                if hw != "CPU":
                    in_flight_est["usd"] -= est
            row["artefact"] = is_artefact(row)
            store.add(row)
            last = (row.get("logs_tail") or [""])[-1][:140]
            print(f"[runs] {p.target:5s} {p.name:34s} seed={seed} -> {row['status']:9s} "
                  f"err={(row.get('error') or '')[:90]!r} score={row.get('score')} "
                  f"dur={(row['ended_at'] or 0) - (row['started_at'] or 0):.0f}s ${row['gpu_usd']:.4f} | {last}")

    await asyncio.gather(*(one(p, s) for p, s in jobs))
    print(f"[runs] done. GPU $ total={store.gpu_usd():.4f}  CPU $ total={store.cpu_usd():.4f}")


# =========================================================================== cases


@dataclass
class Case:
    key: str
    source: str          # failure | probe_ok | healthy
    target: str
    spec: TargetSpec
    snap: ExperimentSnapshot
    info: dict[str, Any] = field(default_factory=dict)   # ground truth etc., never sent


def failure_cases(max_per_probe: int, probe_ok_per_target: int) -> list[Case]:
    """Real runs from RUNS_PATH: failed / timed_out ones (``failure``, the only kind a race
    diagnoses) and succeeded probe runs (``probe_ok``: collapsed, stalled or truncated
    training), at most ``probe_ok_per_target`` of those per target."""
    store = RunStore(RUNS_PATH)
    out: list[Case] = []
    per_probe: Counter[Any] = Counter()
    ok_per_target: Counter[str] = Counter()
    for i, r in enumerate(store.rows):
        if is_artefact(r):
            continue
        status = ExperimentStatus(r["status"])
        failed = status in (ExperimentStatus.failed, ExperimentStatus.timed_out)
        source = "failure" if failed else "probe_ok"
        if per_probe[(r["probe"], source)] >= max_per_probe:
            continue
        if source == "probe_ok":
            if ok_per_target[r["target"]] >= probe_ok_per_target:
                continue
            ok_per_target[r["target"]] += 1
        per_probe[(r["probe"], source)] += 1
        base = get_target(r["target"])
        spec = spec_view(getattr(base, "inner", base).spec, set(r["spec_keys"]))
        snap = make_snapshot(config=r["config"], seed=r["seed"], status=status, score=r.get("score"),
                             error=r.get("error"), metrics=r.get("metrics") or {}, logs_tail=r.get("logs_tail") or [],
                             started_at=r.get("started_at"), ended_at=r.get("ended_at"))
        out.append(Case(key=f"run:{r.get('run_ref') or i}", source=source, target=r["target"], spec=spec, snap=snap,
                        info={"probe": r["probe"], "run_ref": r.get("run_ref"), "status": r["status"],
                              "error": (r.get("error") or "")[:200], "is_trap": r.get("is_trap"),
                              "budget_s": r["budget_s"]}))
    return out


def healthy_cases(per_target: int, seed: int = 20261024) -> list[Case]:
    """Succeeded overnight runs: some trap runs, some probe runs (diverge/overfit configs that
    did not fail), the worst scores, the rest random. Ground truth redacted by the snapshot."""
    rng = random.Random(seed)
    out: list[Case] = []
    for t in TARGETS:
        db = DATA_DIR / f"collect_{t}.db"
        if not db.exists():
            continue
        con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT e.*, p.kind AS probe_kind FROM experiments e LEFT JOIN collection_probes p "
            "ON p.experiment_id = e.id WHERE e.status = 'succeeded' AND e.score IS NOT NULL").fetchall()
        con.close()
        base = get_target(t)
        spec_full = getattr(base, "inner", base).spec
        worst_first = sorted(rows, key=lambda r: r["score"] if spec_full.direction == "max" else -r["score"])
        n_trap, n_probe, n_worst = max(1, per_target // 4), max(1, per_target // 6), max(1, per_target // 4)
        picked: dict[str, tuple[sqlite3.Row, str]] = {}

        def take(cands: list[sqlite3.Row], n: int, why: str) -> None:
            for r in cands:
                if n <= 0:
                    return
                if r["id"] not in picked:
                    picked[r["id"]] = (r, why)
                    n -= 1

        traps = [r for r in rows if r["is_trap"]]
        rng.shuffle(traps)
        take(traps, n_trap, "trap")
        probes = [r for r in rows if r["probe_kind"] in ("diverge", "overfit") and not r["is_trap"]]
        rng.shuffle(probes)
        take(probes, n_probe, "probe")
        take([r for r in worst_first if not r["is_trap"]], n_worst, "worst")
        rest = [r for r in rows if not r["is_trap"]]
        rng.shuffle(rest)
        take(rest, per_target - len(picked), "random")
        for r, why in picked.values():
            cfg = json.loads(r["config_json"])
            spec = spec_view(spec_full, set(cfg))
            try:
                cfg = spec.validate_config(cfg)
            except ValueError:
                continue
            snap = make_snapshot(config=cfg, seed=r["seed"], status=ExperimentStatus.succeeded, score=r["score"],
                                 error=r["error"], metrics=json.loads(r["metrics_json"] or "{}"),
                                 logs_tail=json.loads(r["logs_tail_json"] or "[]"),
                                 started_at=r["started_at"], ended_at=r["ended_at"])
            out.append(Case(key=f"db:{t}:{r['id']}", source="healthy", target=t, spec=spec, snap=snap,
                            info={"why": why, "is_trap": bool(r["is_trap"]), "probe_kind": r["probe_kind"],
                                  "score": r["score"], "experiment_id": r["id"]}))
    return out


def interleave(groups: list[list[Case]]) -> list[Case]:
    """Round-robin across groups so a cap hit mid-way still leaves a balanced mix."""
    out: list[Case] = []
    its = [list(g) for g in groups if g]
    while its:
        for g in list(its):
            out.append(g.pop(0))
            if not g:
                its.remove(g)
    return out


# =========================================================================== diagnose calls


def check_frozen() -> list[str]:
    s = StepType.diagnose_run
    if prompts_pkg.FROZEN.get(s) != prompts_pkg.CURRENT_VERSION[s]:
        return [f"diagnose_run prompt {prompts_pkg.CURRENT_VERSION[s]} is not frozen"]
    return []


def load_calls() -> list[dict[str, Any]]:
    if not CALLS_PATH.exists():
        return []
    return [json.loads(ln) for ln in CALLS_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]


async def run_diagnose(*, per_target_healthy: int, max_per_probe: int, probe_ok_per_target: int, cap_usd: float,
                       dry: bool, limit: int | None, show: int) -> None:
    problems = check_frozen()
    if problems and not dry:
        raise SystemExit("refusing to record: " + "; ".join(problems))
    fails = failure_cases(max_per_probe, probe_ok_per_target)
    healthy = healthy_cases(per_target_healthy)
    by_group: dict[tuple[str, str], list[Case]] = defaultdict(list)
    for c in fails + healthy:
        by_group[(c.source, c.target)].append(c)
    cases = interleave([by_group[k] for k in sorted(by_group)])
    done = {c["case_key"] for c in load_calls() if c.get("valid")}
    prior = load_calls()
    spent = sum(float(c.get("cost_usd") or 0.0) for c in prior)
    n_by_target: Counter[str] = Counter()
    for c in prior:
        n_by_target[c["target"]] = max(n_by_target[c["target"]], int(c["session_id"].rsplit("/", 1)[1]))
    todo = [c for c in cases if c.key not in done][: limit or None]
    print(f"[diag] cases: {Counter((c.source, c.target) for c in cases)}")
    print(f"[diag] todo={len(todo)} already valid={len(done)} spent so far=${spent:.4f} cap=${cap_usd}")
    prompts = load_prompts()
    if dry:
        for c in todo[:show]:
            print("=" * 80, c.key, c.source, c.info)
            print(prompts.diagnose_user(c.spec, c.snap))
        print(f"[diag] system prompt fingerprint {prompts_pkg.prompt_fingerprint(StepType.diagnose_run)}")
        return

    from backend.agent.mubit_client import MubitClient  # the ONLY LLM client (Mubit gateway)

    client = MubitClient(arm_models={ArmKind.opus: OPUS_MODEL}, fallback_model=OPUS_MODEL)
    try:
        for c in todo:
            if spent + EST_CALL_USD > cap_usd:
                print(f"[diag] stop: LLM cap (${spent:.4f} spent)")
                break
            n_by_target[c.target] += 1
            sid = f"{SESSION_PREFIX}/{c.target}/{n_by_target[c.target]}"
            recs: list[LLMCallRecord] = []
            r = LLMResearcher(kind=ArmKind.opus, llm=client, spec=c.spec, session_id=sid, prompts=prompts,
                              fallback_model=OPUS_MODEL, record=True)
            r.on_call = recs.append
            out = await r.diagnose(c.snap)
            for rec in recs:
                spent += float(rec.cost_usd or 0.0)
                row = {"case_key": c.key, "source": c.source, "target": c.target, "info": c.info,
                       "session_id": sid, "mubit_request_id": rec.mubit_request_id, "model": rec.model,
                       "recorded": rec.recorded, "valid": rec.valid, "is_retry": rec.is_retry,
                       "is_fallback": rec.is_fallback, "error": rec.error, "latency_ms": rec.latency_ms,
                       "input_tokens": rec.input_tokens, "cache_creation_tokens": rec.cache_creation_tokens,
                       "cache_read_tokens": rec.cache_read_tokens, "output_tokens": rec.output_tokens,
                       "cost_usd": rec.cost_usd, "stop_reason": rec.stop_reason,
                       "reply": out.model_dump(mode="json") if (out is not None and rec is recs[-1]) else None,
                       "prompt_fingerprint": prompts_pkg.prompt_fingerprint(StepType.diagnose_run),
                       "ts": time.time()}
                with CALLS_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, default=str) + "\n")
            cls = out.failure_class.value if out is not None else "INVALID"
            print(f"[diag] {sid:40s} {c.source:8s} {c.info.get('probe') or c.info.get('why')!s:28s} -> {cls:14s} "
                  f"${sum(float(x.cost_usd or 0) for x in recs):.4f} {recs[-1].latency_ms or 0:.0f}ms "
                  f"{recs[-1].mubit_request_id} | {(out.cause if out else recs[-1].error or '')[:90]}")
    finally:
        await client.aclose()
    summarize()


# =========================================================================== summary


def summarize() -> None:
    calls = load_calls()
    if not calls:
        print("[summary] no calls yet")
        return
    final = {}
    for c in calls:  # last record per case = the attempt whose reply counts
        final[c["case_key"]] = c
    lat = [c["latency_ms"] for c in calls if c.get("latency_ms")]
    print("=" * 80)
    print(f"diagnose_run collection ({SESSION_PREFIX}), model {calls[0]['model']}, record=full")
    print(f"HTTP calls: {len(calls)} (cases {len(final)}); valid {sum(1 for c in calls if c['valid'])}/{len(calls)} "
          f"= {sum(1 for c in calls if c['valid']) / len(calls):.0%}; retries {sum(1 for c in calls if c['is_retry'])}, "
          f"fallbacks {sum(1 for c in calls if c['is_fallback'])}")
    print(f"latency p50 {statistics.median(lat) / 1000:.1f}s, p90 "
          f"{sorted(lat)[int(0.9 * (len(lat) - 1))] / 1000:.1f}s; est. LLM $ {sum(float(c['cost_usd'] or 0) for c in calls):.4f}")
    print("calls per target:", dict(Counter(c["target"] for c in calls)))
    cls = Counter((c["source"], (c.get("reply") or {}).get("failure_class", "INVALID")) for c in final.values())
    print("returned failure_class by source:")
    for src in ("failure", "probe_ok", "healthy"):
        row = {k[1]: v for k, v in cls.items() if k[0] == src}
        if row:
            print(f"  {src:9s} {dict(sorted(row.items(), key=lambda kv: -kv[1]))}")
    print("returned failure_class overall:",
          dict(Counter((c.get("reply") or {}).get("failure_class", "INVALID") for c in final.values())))
    gt = Counter((c["info"].get("probe") or c["info"].get("why"), c["info"].get("status", "succeeded"),
                  (c.get("reply") or {}).get("failure_class", "INVALID")) for c in final.values())
    print("by probe/why, run status -> class:")
    for (k, st, fc), n in sorted(gt.items()):
        print(f"  {k!s:34s} {st:9s} -> {fc:14s} x{n}")
    print("first request ids:", [c["mubit_request_id"] for c in calls[:5]])
    runs = RunStore(RUNS_PATH)
    if runs.rows:
        print(f"real runs: {len(runs.rows)} (GPU est. ${runs.gpu_usd():.4f}, CPU est. ${runs.cpu_usd():.4f}); "
              f"status {dict(Counter(r['status'] for r in runs.rows))}; artefacts "
              f"{dict(Counter(r.get('artefact') for r in runs.rows if r.get('artefact')))}")


# =========================================================================== CLI


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("runs", help="launch real runs on the deployed Modal app")
    r.add_argument("--plan", choices=sorted(PLANS), required=True)
    r.add_argument("--only", default="", help="comma list of probe names or targets")
    r.add_argument("--dry-run", action="store_true")
    d = sub.add_parser("diagnose", help="recorded Opus diagnose_run calls through Mubit")
    d.add_argument("--healthy-per-target", type=int, default=12)
    d.add_argument("--max-per-probe", type=int, default=3)
    d.add_argument("--probe-ok-per-target", type=int, default=8)
    d.add_argument("--show", type=int, default=3, help="dry run: messages to print")
    d.add_argument("--cap-usd", type=float, default=LLM_CAP_USD)
    d.add_argument("--limit", type=int, default=None)
    d.add_argument("--dry-run", action="store_true")
    d.add_argument("--yes", action="store_true", help="required for paid calls")
    sub.add_parser("summary")
    a = ap.parse_args(argv)
    if a.cmd == "runs":
        asyncio.run(run_plan(a.plan, only={s for s in a.only.split(",") if s} or None, dry=a.dry_run))
    elif a.cmd == "diagnose":
        if not a.dry_run and not a.yes:
            raise SystemExit("paid run: pass --yes (or --dry-run)")
        asyncio.run(run_diagnose(per_target_healthy=a.healthy_per_target, max_per_probe=a.max_per_probe,
                                 probe_ok_per_target=a.probe_ok_per_target, show=a.show, cap_usd=min(a.cap_usd, LLM_CAP_USD), dry=a.dry_run, limit=a.limit))
    else:
        summarize()


if __name__ == "__main__":
    main()
