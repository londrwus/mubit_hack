"""Opus vs Haiku latency probe through Mubit (plan.md step 3), plus schema smoke checks.

Sends the same realistic propose_experiments prompt (DRAFT v0 system prompt + an LM-speedrun
style target spec + a 12-run digest, K=4) to claude-haiku-5-5 and claude-opus-5-5 through
the Mubit gateway, N calls each, sequentially, with ``x-middleware-record: off``. Then runs
2 Haiku calls each for diagnose_run and judge_improvement to check those schemas parse.

Every call: latency, the four token meters, estimated $, Mubit request ID, validity.
Hard cost guard: stops before any call once the estimated total passes --max-usd.

Usage:  python -m uv run python scripts/probe_latency.py [--n 3] [--max-usd 0.30] [--out probe.jsonl]
Auth:   MUBIT_API_KEY only (env or .env). Never ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.agent.llm_base import DEV_MODEL, OPUS_MODEL, session_id_for  # noqa: E402
from backend.agent.mubit_client import MubitClient  # noqa: E402
from backend.agent.prompts import load_system_prompt, prompt_fingerprint  # noqa: E402
from backend.agent.prompts.build import (  # noqa: E402
    JudgeClaim,
    ResultRow,
    build_diagnose_user,
    build_judge_user,
    build_propose_user,
    diagnose_wire_schema,
    judge_wire_schema,
    propose_validator,
    propose_wire_schema,
)
from backend.core.types import ArmKind, DiagnoseOutput, JudgeOutput, LLMCallRecord, ProposeOutput, StepType  # noqa: E402
from workers.targets.base import ParamSpec, TargetSpec  # noqa: E402

K = 4

SPEC = TargetSpec(
    name="lm",
    description=(
        "Train a small GPT-style language model from scratch on a fixed web-text shard for a fixed "
        "wall-clock budget, then report validation bits per byte on a held-out shard."
    ),
    metric="val_bpb",
    direction="min",
    target_value=1.55,
    hardware="L4",
    budget_s=60,
    config_space={
        "lr": ParamSpec(type="float", low=1e-4, high=3e-2, log=True, default=1e-3, description="peak AdamW learning rate"),
        "warmup_steps": ParamSpec(type="int", low=0, high=500, default=50, description="linear warmup steps"),
        "batch_size": ParamSpec(type="categorical", choices=[16, 32, 64, 128], default=32, description="sequences per step"),
        "n_layer": ParamSpec(type="int", low=2, high=12, default=6, description="transformer blocks"),
        "n_embd": ParamSpec(type="categorical", choices=[128, 256, 384, 512], default=256, description="model width"),
        "weight_decay": ParamSpec(type="float", low=0.0, high=0.3, default=0.1, description="AdamW weight decay"),
        "schedule": ParamSpec(type="categorical", choices=["cosine", "linear", "constant"], default="cosine",
                              description="learning-rate decay after warmup"),
        "grad_clip": ParamSpec(type="float", low=0.0, high=2.0, default=1.0, description="max grad norm (0 = off)"),
        "val_from_train_pool": ParamSpec(type="bool", default=False,
                                         description="sample validation sequences from the training shard pool for speed"),
    },
    trap_option="val_from_train_pool",
)


def history() -> list[ResultRow]:
    d = SPEC.default_config()
    runs: list[tuple[dict[str, Any], str, float | None, str | None]] = [
        ({}, "succeeded", 1.742, None),
        ({"lr": 3e-3}, "succeeded", 1.688, None),
        ({"lr": 1e-2}, "failed", None, "divergence: loss=nan at step 140"),
        ({"lr": 3e-3, "warmup_steps": 150}, "succeeded", 1.671, None),
        ({"lr": 3e-3, "n_layer": 10, "n_embd": 512}, "timed_out", None, "only 310 of 900 steps in budget"),
        ({"lr": 3e-3, "batch_size": 64}, "succeeded", 1.659, None),
        ({"lr": 5e-3, "warmup_steps": 200, "batch_size": 64}, "succeeded", 1.647, None),
        ({"lr": 5e-3, "warmup_steps": 200, "batch_size": 64, "schedule": "linear"}, "succeeded", 1.655, None),
        ({"lr": 2e-3, "n_layer": 4, "n_embd": 384}, "succeeded", 1.701, None),
        ({"lr": 5e-3, "warmup_steps": 200, "batch_size": 128}, "failed", None, "oom: CUDA out of memory at step 3"),
        ({"lr": 5e-3, "warmup_steps": 200, "batch_size": 64, "weight_decay": 0.05}, "succeeded", 1.641, None),
        ({"lr": 4e-3, "warmup_steps": 250, "batch_size": 64, "grad_clip": 0.5}, "succeeded", 1.644, None),
    ]
    return [ResultRow(label=f"#{i + 1}", config=d | c, status=s, score=sc, note=n) for i, (c, s, sc, n) in enumerate(runs)]


RUNNING = [SPEC.default_config() | {"lr": 6e-3, "warmup_steps": 300, "batch_size": 64, "weight_decay": 0.05}]

DIAG_CASES = [
    ("failed", "FloatingPointError: loss is nan",
     [f"step {s:4d} | loss {l} | lr {lr} | grad_norm {g}" for s, l, lr, g in [
         (100, "3.912", "8.0e-03", "1.9"), (120, "3.870", "9.6e-03", "4.7"), (130, "4.410", "1.0e-02", "38.2"),
         (135, "7.950", "1.0e-02", "911.4"), (140, "nan", "1.0e-02", "nan")]] + ["Traceback (most recent call last):",
                                                                                 "FloatingPointError: loss is nan"]),
    ("succeeded", None,
     [f"step {s:4d} | loss {l:.3f} | lr 3.0e-03" for s, l in [(100, 4.1), (300, 3.2), (500, 2.9), (700, 2.75), (900, 2.68)]]
     + ["eval | val_bpb 1.671", "done in 59.8 s"]),
]

JUDGE_CASES = [
    JudgeClaim(prev_best_score=1.641, prev_best_config=history()[10].config, new_score=1.412,
               new_config=history()[10].config | {"val_from_train_pool": True},
               evidence={"train_shard_hashes": "a91f,c22e,7b0d", "val_shard_hash": "c22e", "val_train_overlap_seqs": 4096,
                         "selection_split": "val", "eval_examples": 4096, "seed_spread_bpb": 0.006}),
    JudgeClaim(prev_best_score=1.641, prev_best_config=history()[10].config, new_score=1.618,
               new_config=history()[10].config | {"warmup_steps": 300, "lr": 6e-3},
               evidence={"train_shard_hashes": "a91f,c22e,7b0d", "val_shard_hash": "e5d3", "val_train_overlap_seqs": 0,
                         "selection_split": "val", "eval_examples": 4096, "seed_spread_bpb": 0.006}),
]


def row(label: str, r: LLMCallRecord) -> dict[str, Any]:
    return {
        "label": label, "step": r.step_type.value, "model": r.model, "latency_ms": round(r.latency_ms),
        "input": r.input_tokens, "cache_write": r.cache_creation_tokens, "cache_read": r.cache_read_tokens,
        "output": r.output_tokens, "cost_usd": round(r.cost_usd, 5), "valid": r.valid, "stop": r.stop_reason,
        "request_id": r.mubit_request_id, "error": r.error, "recorded": r.recorded,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="propose calls per model")
    ap.add_argument("--max-usd", type=float, default=0.30)
    ap.add_argument("--out", type=Path, default=None, help="write one JSON line per call")
    ap.add_argument("--skip-opus", action="store_true")
    args = ap.parse_args()

    client = MubitClient()
    session = session_id_for("probe-latency", ArmKind.opus, 0)
    rows: list[dict[str, Any]] = []
    spent = 0.0
    last_cost: dict[str, float] = {}

    async def one(label: str, step: StepType, user: str, schema: Any, wire: dict[str, Any], model: str,
                  validator: Any = None) -> LLMCallRecord | None:
        nonlocal spent
        reserve = last_cost.get(model, 0.0)  # expect the next call to cost about what the last one did
        if spent + reserve >= args.max_usd:
            print(f"!! cost guard: ${spent:.4f} + ~${reserve:.4f} >= ${args.max_usd}; skipping {label}")
            return None
        parsed, rec = await client.call(step, load_system_prompt(step), user, schema, ArmKind.opus, session,
                                        record=False, wire_schema=wire, validator=validator, model=model)
        spent += rec.cost_usd
        last_cost[model] = max(last_cost.get(model, 0.0), rec.cost_usd)
        r = row(label, rec)
        if parsed is not None and step is StepType.propose_experiments:
            r["n_experiments"] = len(parsed.experiments)
            r["hypotheses"] = [e.hypothesis for e in parsed.experiments]
            r["traps_proposed"] = sum(SPEC.is_trap(e.config) for e in parsed.experiments)
        elif parsed is not None:
            r["reply"] = parsed.model_dump(mode="json")
        rows.append(r)
        print(json.dumps(r))
        return rec

    propose_user = build_propose_user(SPEC, history(), free_slots=K, running=RUNNING)
    pwire, pval = propose_wire_schema(SPEC), propose_validator(SPEC, K)
    print(f"prompt fingerprints: " + ", ".join(f"{s.value}={prompt_fingerprint(s)}" for s in StepType))

    plan = [("haiku", DEV_MODEL)] + ([] if args.skip_opus else [("opus", OPUS_MODEL)])
    for name, model in plan:
        for i in range(args.n):
            rec = await one(f"propose/{name}/{i + 1}", StepType.propose_experiments, propose_user, ProposeOutput,
                            pwire, model, pval)
            if rec is not None and rec.error and rec.error.startswith("http 4"):
                print("!! client error; stopping before spending more")
                await client.aclose()
                return 1

    for i, (status, err, logs) in enumerate(DIAG_CASES):
        await one(f"diagnose/haiku/{i + 1}", StepType.diagnose_run,
                  build_diagnose_user(SPEC, history()[2].config if i == 0 else history()[3].config, status, err, logs,
                                      {"final_loss": float("nan") if i == 0 else 2.68}),
                  DiagnoseOutput, diagnose_wire_schema(), DEV_MODEL)
    for i, claim in enumerate(JUDGE_CASES):
        await one(f"judge/haiku/{i + 1}", StepType.judge_improvement, build_judge_user(SPEC, claim), JudgeOutput,
                  judge_wire_schema(), DEV_MODEL)
    await client.aclose()

    print("\nsummary (propose_experiments):")
    for name, model in plan:
        lat = [r["latency_ms"] for r in rows if r["model"] == model and r["step"] == "propose_experiments"]
        if lat:
            print(f"  {name}: latency ms {lat} median {statistics.median(lat):.0f}")
    print(f"total estimated cost: ${spent:.4f}")
    if args.out:
        args.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
