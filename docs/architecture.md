# Architecture

Status (24 Oct, ~05:00 BST): **backend built, benchmark races running.** Built: FastAPI app (REST, WebSocket, race replay), race engine with the Busy Machines engine rules (judge policy, median stopping), deadline rule and pre-warm, agent loop with frozen prompts, Mubit client with per-arm record modes, mock LLM, Optuna arm, benchmark package (presets, metrics, export, analysis, engine rules), `scripts/bench.py` and `scripts/headline.py`, four targets (`lm`, `cifar`, `rl`, `lcbench`), three runners (`local`, `modal`, `surrogate`), and Sapling on a Modal H100 (offline). Races so far: LCBench 3 seeds Opus vs Optuna (R38), RL wide seed 0 (R36, not valid for GPUs waiting). **Not built yet:** frontend, sending corrections to Mubit, routing a race's steps to Mubit project B, Sapling inside the live race. **Not measured yet:** any race with the `opus_bm` or `mubit` arm, and any adapter serving live (none has passed its exam: R28, R30). Update this doc when the code diverges from it.

**Stack (D8):** Python 3.12+ / FastAPI (REST + WebSocket) / SQLite · Vite + React + shadcn/ui + Tailwind (+ Recharts) · Pen.dev for design · pluggable runners (`local`, `modal`, `surrogate`) · every LLM call through the Mubit gateway · Banyan Sapling as the offline verifier.

## Overview
```
                 ┌──────────────────────────── FastAPI backend ────────────────────────────┐
 React dashboard │  RaceEngine ──► ArmRunner × 2–4: opus · opus_bm · mubit · optuna         │
 (not built)     │      │            researcher: propose → diagnose? → judge? → engine rules│
   ▲  WebSocket  │      │                  │ LLM calls                │ runs               │
   └─────────────┤  Event bus ◄────────────┤                          ▼                    │
     REST        │      │          Mubit gateway (projects A, B)   Runners: local · modal  │
     + replays   │   SQLite                                          · surrogate           │
                 └─────────────────────────────────────────────────────────────────────────┘
 Offline: Sapling on Modal H100 · Mubit's hosted Sapling · headline.py / analysis.py on JSONL exports
```

## 1. Agent loop (`backend/agent/`)
One **arm session** = one researcher working on one target with N slots for a fixed wall-clock budget.
There are **four recurring LLM steps**. Each is a **stateless single-turn call** with a frozen system prompt and a structured-output JSON schema, so each prompt version is exactly one Mubit task (cost design in `budget.md`):

| Step | Frozen prompt (fingerprint) | When | Output (JSON) | In races |
|---|---|---|---|---|
| `propose_experiments` | v1 `fdda8c5b0919` (D27) | slots free up | `{experiments: [{config, hypothesis}] × ≤K}`; config is a compact settings string, one schema for every target (D24) | every LLM arm |
| `diagnose_run` | v0 `22ecb45368e3` (D27) | **only on failures** | `{failure_class, cause, fix}` | every LLM arm; fires almost only on RL, because CIFAR and LM can't fail inside their config space (R26) |
| `judge_improvement` | v1 `5c1d3cfd2941`, closed output (D37, R39); v0 `623366123052` kept as history | **only on a claimed new best** | `{verdict, reason_code}`, both from fixed lists (5 codes); no free text | every LLM arm. Default since R39; races R35–R38 used v0 |
| `early_stop` | v2 `f1f3bf21aa79`, decision only (D34) | — | `{decision: continue/stop}` | **not called.** It failed the D39 gate (Opus self-agreement 92.5% full output, 95% decision only, gate 97%; R31), so nothing was recorded. The engine's median rule does this job (D40) |

`summarize_result` is **deterministic code** (delta vs best, keep/drop), not an LLM call.

**Every reply is validated against its Pydantic schema.** Adapters don't reliably keep the format (R9: 3 of 7 toy replies were malformed). An invalid reply is logged (`llm_calls.valid = false`) and retried once, then the call falls back to Opus through Mubit. Retries and fallbacks count toward the arm's time and $, and a malformed reply never counts as a fast answer.

As built (`loop.py`, `mubit_client.py`):
- `LLMResearcher` runs the policy (attempt → retry once → Opus fallback) on **every** LLM arm, so the rule is the same for all. Each attempt is its own `llm_calls` row.
- A propose reply is invalid if no config passes `validate_config`; single bad configs are dropped. Hypotheses: the prompt asks for ≤20 words, code trims anything over 25 (D26).
- K per propose call is capped by `max_proposals_per_call` (8 in the RL wide and LCBench presets), the same for every arm. Opus propose latency grows with K (R27, R36).
- The researcher is one sequential coroutine: diagnose and judge run before the next propose, so their wait also counts as GPU idle.
- Ground-truth keys (`heldout_score`, `train_levels_score`, `is_trap`, anything with "heldout", ...) are stripped before a result reaches the LLM. The planning prompt is neutral about trap settings (D25).
- Effort per step: propose medium, diagnose / judge / early_stop low. `MubitClient` refuses any base URL that isn't Mubit and turns off SDK retries (one call = one request).
- `MockLLM` (deterministic, `benchmark_safe = False`) stands in for dev. The DB and the engine refuse mock or replayed calls in a benchmark race.

Prompts live in `backend/agent/prompts/`, one file per version. A frozen file is never edited: a change means a new file, a new Mubit task and a new example collection. The prompt fingerprint goes into every race's policy string.

**Example coverage matters more than count.** The toy adapter failed exactly the cases its examples lacked (R9). **Gate before recording a new task (D39):** Opus self-agreement ≥97% on the full output, ≥150 distinct inputs, no class over 70%, no free text. Judge v1 passed (68/68, R32); early_stop failed (R31).

## 2. Arms (D36)
| Arm (`ArmKind`) | On screen | LLM | Engine features | Record mode (bench presets) |
|---|---|---|---|---|
| `opus` | Standard AI assistant | Opus 5.5 on every step, through Mubit | none (the standard agent) | `off` |
| `opus_bm` | Busy Machines with Opus | Opus 5.5 on every step, through Mubit | `judge_policy` + `backfill` (median stop off, D43) | `off` |
| `mubit` | Mubit + Busy Machines | same model ID as the Opus arm; Mubit's small model answers where an adapter passed its exam and has a route, the provider answers the rest | `judge_policy` + `backfill` (median stop off, D43) | `metadata` |
| `optuna` | Automatic search (no AI) | none | `median_stop` (Optuna's own MedianPruner practice) | n/a |

- `opus_bm` splits the gain into "our engine" (`opus_bm` vs `opus`) and "Mubit's models" (`mubit` vs `opus_bm`).
- The engine refuses a race if the two Busy Machines arms get different features, if Optuna's median rule doesn't match theirs, if a benchmark Opus arm has a feature, or if the judge policy has no calibration for the target. Features are in the fairness manifest.
- Today no adapter serves (R28: propose 14.3%, R30: judge v0 42.9%, both vs a 95% target; judge v1 adapter: not measured yet). So the RL wide and LCBench presets run `opus`, `opus_bm` and `optuna`; `mubit` joins once an adapter serves.

## 3. Engine rules (`backend/benchmark/engine_rules.py`, D40)
Deterministic code, no LLM call, no prompt change, so no Mubit task changes.

**Judge policy** (`judge_policy`): the LLM judge decides leakage, the seed-spread rule decides real vs noise.
- LLM says `invalid` → invalid.
- Anything else (valid, unsure, no usable reply) → the claimed gain must exceed 2 × the calibrated run-to-run spread of the held-out score at the previous best's level. Real → `valid`; noise → `unsure`.
- The spread comes from starting-code and best-config calibration runs: `data/calibration/calibration_v1.json` (rl, cifar, lm), `calibration_rl15_v1.json` (`rl@15s`). LCBench uses 0, because the surrogate gives the same result for every seed.
- Both the LLM verdict and the rule decision are stored per claim (`experiments.summary_json.judge_policy`).
- Why (offline, on stored verdicts, 75 cases): Opus alone confirms 1/7 real gains; the rule alone lets 24/32 fakes through (R34); the combination confirms 7/7 real gains, accepts 1/36 noise and lets 0/32 fakes through (R37). Small set (7 real), and not yet run live in a race; the adapter version is not measured yet.
- D36 also named verification re-runs of claimed bests. Not built: the calibrated spread replaces per-claim re-runs.

**Median stopping** (`median_stop`): at 25% and 50% of a run's budget, a running run whose curve value is worse than the median of the arm's completed runs at the same point (once ≥5 have reached it) is cancelled and its slot freed. Curve metric: rl `train_return`, cifar `val_acc`, lm `val_bpb`, lcbench `val_balanced_accuracy`. Stored as `summary_json.early_stop` (checkpoint, value, median, estimated time saved). Effect on cost or quality: not measured yet.

**Backfill** (`backfill`, "keeps GPUs busy with a no-AI sampler while the AI thinks"; offline gate in `data/backfill/`, projections only until a race measures it): the arm holds an Optuna TPE sampler (`backend/agent/optuna_arm.py` `BackfillSampler`, seeded with the race seed).
- The LLM's demand K counts only its own runs and buffered proposals. Its proposals launch first; every slot still free gets a sampler config at once, also while the planner call is in flight. TPE asks run in a worker thread (dedicated backfill task in the scheduler).
- Sampler runs go through the same processing as LLM runs: a claimed new best is judged by the same LLM judge + judge policy, and the run renders in the planner digest exactly like an LLM run.
- Feedback is checker-gated: after the verdict, the sampler is told every processed run of the arm (LLM, baseline, its own). A claim the LLM judge rejected is told the worst score seen so far, never its inflated score. The sampler is never told which option is the trap.
- Each experiment stores `source` (`baseline` | `llm` | `optuna` | `sampler`); reports and exports carry `runs_by_source`, sampler trap runs / claims rejected / fakes shipped, and `shipped_source`. The fairness manifest (`engine.per_arm.<arm>.backfill`) records the sampler settings.
- Presets: `rl-wide-bm-15m` (92 containers incl. 2 spare), `cifar-bm-15m` (9 L4), `lcbench-bm-15m`, each `opus` vs `opus_bm` vs `optuna` with the headline preset's settings. The byte-identical planner test (D44) runs with backfill off; with it on, the judge policy's labels still never reach the planner (`tests/test_backfill.py`).

## 4. Mubit integration
### Rules
- `backend/agent/mubit_client.py` is the **only** place LLM clients are constructed. Base URL `https://api.mubit.ai/anthropic`, auth `MUBIT_API_KEY`.
- Per-call variable content (history, metrics, configs) goes in the **user message**, never the system prompt, so each step stays one task (and caches).
- Session per researcher run: `metadata.user_id` = `race-{id}/{arm}/seed-{n}`.
- Capture `x-middleware-request-id` from every response into `llm_calls.mubit_request_id`, linked to the experiments it produced.
- Races, collection and probes all bill to our Anthropic provider key (D14, D38).

### Two projects (D38)
| Project | Key | Tasks | Used by |
|---|---|---|---|
| A | `MUBIT_API_KEY` | propose v1, judge v0, diagnose v0 | every race and the overnight collection |
| B | `MUBIT_API_KEY_B` | judge v1 (177 recorded answers, R39); early_stop (nothing recorded, R31) | `scripts/record_judge_v1.py`, `scripts/collect_early_stop.py` |

The client reads the key per process; the B scripts set `MUBIT_API_KEY` to B's key for their own process only. **Routing one step of a race (e.g. judge v1) to project B is not built yet.**

### Record modes (`x-middleware-record`, per arm)
| Mode | Effect (our reading of Mubit's docs) | Used for |
|---|---|---|
| `full` | recorded; can become a kept example | example collection and recording scripts |
| `metadata` | keeps the task fingerprint (routable to an adapter), drops the content | the `mubit` arm in benchmark presets |
| `off` | drops content and fingerprint, so it can't be routed: always the provider | `opus` and `opus_bm` arms, probes, gates |

Set by `RaceConfig.record_mode` (`BENCH_RECORD_MODES`); fallback calls inherit their arm's mode. It is the one listed difference besides the client (`differs_only_in` in the fairness manifest). Pending Mubit's confirmation (`mubit_office_hours.md` item 2).

### Capability map (D13: use every Mubit capability)
Status: ✅ documented or verified · 📣 marketing/launch claim, verify at office hours · ❓ unknown.

| Mubit capability | Status | How Busy Machines uses it |
|---|---|---|
| **Gateway** | ✅ | Every LLM call: agent steps, every LLM arm, data labelling |
| **Tasks** | ✅ | One task per frozen step prompt: three in project A, judge v1 and early_stop in project B |
| **Sessions** | ✅ | One session = one researcher run; in the console it's the agent's **procedural log** (Anil's phrase) |
| **Request IDs** | ✅ | On every experiment row, so each GPU result links to the decision that caused it |
| **Recording controls** | ✅ | Per-arm modes keep the Opus arms pure; the answer to "labs won't send their traces" |
| **Adapters + versions + grading** | ✅ | Trained so far: propose v1 14.3% (R28), judge v0 42.9% (R30), both below the 95% target. Judge v1 adapter: not measured yet |
| **Try models** | ✅ | Demo moment: adapter vs Opus on the contaminated claim |
| **Rate reply Good/Bad** | ✅ UI / ❓ effect | Correction channel if there's no API (Q9) |
| **OpenAI-compatible services** | ✅ | Open-model arm and fallback if adapters can't serve: `https://api.mubit.ai/compat/{groq,together,openrouter,mistral}/v1`. Not built |
| **Import past calls** | ✅ | Pitch only: day-0 onboarding for labs |
| **Coding-tool integration** | ✅ | Zero-build adoption: researchers' Claude Code/Codex sessions also flow through Mubit |
| **Outcome learning** | 📣 | The verified training metric *is* the outcome, attached by request ID |
| **Routing to the dedicated model only when better** | 📣 (launch) | The `mubit` arm uses adapters automatically once they beat the provider (Q1) |
| **Change control / rollback** | 📣 | Roll an adapter back if verifier accuracy drops |
| **Experience reuse** | 📣 | Lessons carry across targets |
| **Banyan Sapling** | ✅ | Offline verifier, self-hosted and Mubit's hosted endpoint (§7) |
| **latte/bench** | ✅ | `night/bench` is its research counterpart |
| **Mubit's metrics** | 📣 | Per research step, plus GPU $ |

## 5. Scheduler (`backend/scheduler.py`)
- Each arm has N slots. While the LLM is thinking, **the slot is idle and the idle clock runs**. GPU idle % = idle time / paid time.
- At t0 every slot is idle. With `run_baseline_first`, slot 0 starts the default config at t0 in every arm.
- One batched propose call asks for K = N + lookahead − running − buffered configs (capped by `max_proposals_per_call`). Lookahead is N with pipelining and 0 without, the same for every arm.
- A slot is busy from `launch()` until `wait()` returns, and idle otherwise.
- **Deadline rule** (`launch_cutoff_s`, D32, same for every arm): when less than one run's wall time is left, no run is launched and no proposal is asked for; buffered proposals are dropped and each slot is released as soon as it is idle. A released slot stops paying. Without the rule, runs in flight at the deadline are cancelled and don't count.
- **Pre-warm** (`prewarm`, Modal only, D32): `bench.py` pins and warms one container per slot of every arm (plus spares) and takes t0 only when all are warm. Pre-warm time is outside the race, so it's never busy, idle or race $.
- Per-arm $ cap (GPU $ + LLM $) stops the arm when hit.
- Optuna's asks run in a worker thread, off the shared event loop (fixes the R35/R36 caveat).
- `RaceEngine` refuses to start if arms would get different hardware, prices, slots or budgets, if engine features break the rules in §2, if a target needs H100 without `allow_h100`, or if the GPU $ estimate is above `max_estimated_gpu_usd`. It stores a fairness manifest (settings, engine features, record modes, pre-warm info) in the race row.
- Every experiment uses the race seed; variation comes from running several race seeds.

## 6. Compute: pluggable runners (`workers/`)
**The product is compute-agnostic (D17). Modal is only our test/demo compute.** Targets never import a compute SDK; `modal` is imported only in `workers/runners/modal_*.py` (D28).

```python
class Runner(Protocol):
    name: str                                   # "local" | "modal" | "surrogate"
    def slots(self) -> list[Slot]: ...
    async def launch(self, target: str, config: dict, seed: int,
                     budget_s: int, on_metric: Callable) -> RunHandle: ...
    async def wait(self, handle: RunHandle) -> RunResult: ...   # never raises for training failures
    async def cancel(self, handle: RunHandle) -> None: ...      # idempotent
    def price_per_second(self, hardware: str) -> float: ...
```

| Runner | Use | Notes |
|---|---|---|
| `local` | Dev and tests on the laptop (mock LLM, tiny CPU runs) | One subprocess per run, 2 slots by default, priced $0. **No local GPUs** |
| `modal` | GPU and CPU benchmark races, the demo | Modal Functions (not Sandboxes), deployed app `busy-machines-runner` |
| `surrogate` | LCBench races | In-process, no compute rented; see below |
| `slurm` / `k8s` / `ray` | Customer deployment (pitch only) | Same interface; not built |

**`modal` as built:** `run_l4` (L4: `lm`, `cifar`) and `run_cpu` (`rl`), picked from `TargetSpec.hardware`. Two images: torch for L4, jax + optax for CPU. Data in the Volume `busy-machines-data` at `/data`; metrics on the Queue `busy-machines-metrics`, one partition per run. `run_ref` is the FunctionCall ID. Caps (10 GPUs / 100 containers) are enforced on the host, so a race never queues silently.
- Pre-warm: `prewarm(hw, n)` pins n containers, `warm_up(hw, n)` waits until n distinct containers answered a warm-up job, `cooldown(hw)` unpins them (also on errors and Ctrl-C).
- Hang fix (R36): at 64 concurrent runs some `wait()` calls never returned. Every host-side Modal call now has a 10 s cap and is abandoned, not awaited, on timeout; the metrics poller uses short non-blocking reads. Stress-tested at 64 concurrent calls afterwards (R36).
- Smoke test: `python -m workers.runners.modal_runner smoke [cifar|lm|rl]`.

**`surrogate` as built:** the LCBench answer arrives in milliseconds, but the run it stands for took minutes, so the runner holds the slot busy for the run's predicted training time (× `BUSY_SURROGATE_RUNTIME_MULT`, default 1) and replays its metrics at their predicted times. Slots are priced at the L4 rate, so **GPU $ is virtual**: real spend in LCBench races is the LLM only (R35, R38). The multiplier is written into every `run_ref`.

Targets implement a common interface (`workers/targets/base.py`: `get_target`, `list_targets`):
```python
class Target(Protocol):
    name: str                 # "lm" | "rl" | "cifar" | "lcbench"
    metric: str               # what the agent sees
    direction: Literal["min", "max"]
    target_value: float       # placeholder; races use the frozen goal file
    hardware: str             # "L4" | "CPU"
    budget_s: int             # default per-run training budget
    config_space: dict        # for proposals and for Optuna
```
- **`score` is the reported number** (what the agent's own pipeline reports, inflated by the trap). The ground truth is `metrics.heldout_score`, evaluated on a split the agent never sees. Model quality, time-to-target and $-to-target use `heldout_score`.
- **Goals** live in `data/calibration/target_values.json`, keyed by target, or `target@Ns` when the run budget isn't the default (e.g. `rl@15s`). `bench.py` refuses a target without one unless `--target-value` is given, which marks the race provisional. The `rl@15s` goal is too easy at 32 slots (R36).

| Target | What trains | Metric (agent sees / held out) | Hardware, run budget | Trap option |
|---|---|---|---|---|
| `lm` | char-level nanoGPT on tiny-shakespeare | `val_bpb` ↓ / test bpb | L4, 60 s | `train_on_all_shards` |
| `cifar` | airbench-style CIFAR-10 net | `test_acc` ↑ / held-out acc | L4, 45 s | `tune_on_test` |
| `rl` | single-file JAX PPO on Pendulum-v1 with randomised physics levels | `mean_return` ↑ / held-out levels | CPU, 30 s (15 s in `rl-wide-15m`) | `eval_on_train_levels` |
| `lcbench` | YAHPO LCBench surrogate of real MLP runs, OpenML task 167168 | val balanced acc ↑ / test balanced acc | virtual L4, 300 predicted s | `checkpoint_selection=best_test` |

Targets offer real research choices, not only numeric settings (D29). LM is demoted (no config beat the starting code, R23; D32). LCBench caveats: Opus may know LCBench from training (R35, R38). The surrogate's val and test predictions may rank configs almost identically, which would make held-out a weaker check there; that is an observation, not a logged measurement (not in `results_log.md`).

## 7. Verifier (`workers/verifier/`, `workers/runners/modal_sapling.py`)
- **Self-hosted Banyan Sapling on a Modal H100** (separate app `busy-machines-sapling`, pinned revision, every weight file checked by SHA-256 once). One H100 pass extracts pooled features and stock logits; `sapling-research` retrains only the verification head on CPU, backbone frozen (D23), with ML-research-only calibration (D28).
- Result (R33): a big gain over stock Sapling, far below Opus as a judge. Trap set: stock 10/50, ours 14/50 (α 0.01) / 18/50 (α 0.05), Opus 5.5 50/50. Its edge is speed and price (73 ms and ~$0.00015 vs ~2.2 s and ~$0.0046 per case). Wording rules: `sapling_results.md` §5.
- **Mubit's hosted Sapling endpoint (D33)**: the official stock baseline in the three-column table and a parity check against our self-hosted stock run (`scripts/sapling_stage_v2.py hosted-parity`, ≤2 requests at a time, synthetic data only). Parity result: not measured yet.
- Sapling is **not in the live race loop** (D28 (6)); it verifies offline and in the benchmark table. It isn't an LLM call, so it doesn't go through the gateway; D33 mostly settles Q5 for the stock model.

## 8. Contamination trap
One option per target silently **contaminates the evaluation**; the ground truth is known (table in §6). The judge sees run metrics but never the held-out score. **Trap runs never count for any arm** (`metrics.is_counted`), even with an honest held-out score, so an arm without a judge (Optuna) isn't rewarded for accepting them. How often each arm accepts a fake as its best is reported separately (R36, R38).

## 9. Corrections → model update
UI flag or Sapling flag → `corrections` row → sent to Mubit (mechanism: **Q2/Q9**; console Good/Bad rating as fallback) → a new adapter version is trained; the version history is shown in the console during the demo.

As built: `POST /api/experiments/{id}/flag` stores the correction with `sent_to_mubit_at = NULL`; nothing sends it yet (there is no corrections API, `demo.md` §0). On a live race a `fake_improvement` flag demotes that experiment from the arm's best.

## 10. Benchmark tooling
- **`scripts/bench.py`** (`list`, `check`, `run`): runs a preset's arms at the same t0 on the deployed Modal app or the surrogate, seeds one after another, into `data/bench/bench.db` plus a JSONL export per race. Paid runs need `--yes`; `--dry-run` uses the mock LLM and the local runner ($0, never a result).
- **Presets** (`backend/benchmark/configs.py`): `rl-wide-15m` (the headline regime, D32: 30 CPU slots per arm in the current preset, 15 s runs, 15 min, deadline rule, pre-warm; the logged R36 race ran 32 slots per arm, so the two are not directly comparable), `lcbench-15m` (16 virtual L4 slots, 15 min), `cifar-30m`, `lm-30m`, `rl-cpu-30m`, `demo-3m`, `dev-local`.
- **`backend/benchmark/analysis.py`** recomputes every summary from the JSONL export alone and rejects dev, mock and replayed races.
- **`scripts/headline.py`** (read-only): the D31/D36 table (Opus · BM with Opus · Mubit + BM · Optuna, mean ± std over seeds), the `benchmark.md` §6b framings for each pair (a worse result is printed as worse), the same-quality and same-money views (D30), and the evidence behind each number (request IDs, Modal call IDs). A framing prints only if every race is finished, in benchmark mode, with a frozen goal, identical slots and ≥3 seeds; otherwise "not measured" and why. `--json` for the slides.
- **`scripts/analyze_races.py`**: per-arm tables for `results_log.md` and plain-language charts.

## Data model (SQLite, WAL mode)
```
meta           key, value
races          id, name, target, budget_s, gpus_per_arm, seed, mode (dev|benchmark), is_dev,
               llm_mode, runner, config_json, fairness_json, status, finish_reason,
               created_at, started_at, ended_at
arms           id, race_id, kind (opus|opus_bm|mubit|optuna|mock), model, llm_mode, is_dev,
               n_slots, hardware, price_per_s, status, stop_reason, started_at, ended_at,
               summary_json, report_json
experiments    id, arm_id, race_id, slot_id, config_json, hypothesis, status, runner, run_ref,
               seed, hardware, proposed_at, started_at, ended_at, score, is_best, is_trap,
               is_baseline, judge_verdict, judge_reason, diagnosis_json, summary_json, error,
               metrics_json, logs_tail_json, llm_call_id, mubit_request_id, gpu_cost_usd
metrics        experiment_id, step, t, name, value              -- streamed curves
llm_calls      id, arm_id, race_id, arm, session_id, step_type, mubit_request_id, model,
               latency_ms, input_tokens, cache_creation_tokens, cache_read_tokens,
               output_tokens, cost_usd, valid, is_retry, is_fallback, is_replay, recorded,
               stop_reason, error, experiment_ids_json, created_at, completed_at
gpu_intervals  id, arm_id, race_id, slot, hardware, state (busy|idle), start, end,
               experiment_id, price_per_s
corrections    id, experiment_id, race_id, arm_id, source (human|sapling), label, reason,
               created_at, sent_to_mubit_at
adapter_versions  id, step_type, version, status, trained_at, notes, created_at   -- mirrored from console
```
- `run_ref` holds the Modal call ID, the local PID or `surrogate:<id>:x<mult>`.
- `experiments.summary_json` carries the engine decisions: `judge_policy` (LLM verdict + rule decision) and `early_stop` (median stop).
- `arms.report_json` carries engine counters (slots released early, median checks, stops requested, traps, retries).
- `races.fairness_json` carries the manifest: shared settings, engine features per arm, record modes (`differs_only_in`), pre-warm info.
- Export: `GET /api/races/{id}/export.jsonl`, one `{"table", "row"}` line per row.

## Cost accounting
- **GPU $** = (busy + idle seconds) × the runner's rate, until the slot closes or is released. Modal list prices: L4 $0.000222/s, H100 $0.001097/s, CPU core $0.0000131/s; a customer enters their own. The surrogate's GPU $ is virtual (L4 rate, nothing rented).
- **LLM $** = the four token meters × provider price.
- **$-to-target** = GPU $ accrued (busy + idle) + LLM $ of calls completed, up to the first counted run that reaches the goal.
- A run counts if it succeeded, its arm's judge did not reject it, it isn't a trap run, and it has a held-out score.

## API (FastAPI)
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/races` | create a race (`RaceConfig`; dev races also take `mock_latency_s`, `mock_trap_rate`) |
| POST | `/api/races/{id}/start` | start all arms at the same t0 |
| POST | `/api/races/{id}/stop` | stop a race |
| GET | `/api/races` | list races; `?replayable=true` lists finished races from every replay source |
| GET | `/api/races/{id}` | state and summary stats |
| GET | `/api/races/{id}/export.jsonl` | the verification dump |
| POST | `/api/races/{id}/replay` | build a replay (`speed` up to 1000×, `max_gap_s`, `metric_stride`) |
| GET | `/api/replays`, `/api/replays/{id}` | replay sessions |
| POST | `/api/experiments/{id}/flag` | human correction (fake improvement) |
| GET | `/api/corrections` | corrections |
| GET / POST | `/api/adapters` | adapter versions per step (mirrored from the console) |
| GET | `/api/health` | health |
| WS | `/ws/races/{id}` | live event stream |
| WS | `/ws/replays/{id}` | replayed event stream |

**WebSocket events:** `gpu_state`, `experiment_started`, `metric`, `experiment_finished`, `new_best`, `llm_call`, `correction`, `race_finished`. A late subscriber gets the race's history replayed.

**Replay** (`backend/replay.py`, the demo fallback): rebuilds a finished race's event stream from stored rows (app DB, read-only `.db` files or JSONL exports; extra sources in `BUSY_MACHINES_REPLAY_SOURCES`) and plays it at N× real time. Each event is the live JSON plus `is_replay: true`, `replay_id` and `race_elapsed_s`, so the screen can say it's a replay (`demo.md` §4). Stored rows are only read. Metric and new-best timestamps are approximate; corrections made after the race aren't included. CLI: `python -m backend.replay <race_id> --speed 10 --db data/bench/bench.db`.

## Frontend (`frontend/`, not built)
The "Night Shift Race" split screen (`demo.md`): GPU tiles, live curves, leaderboard ticker, thought stream, cost panel, flag button, and a results page with the 3-run charts. Designed in Pen.dev first, built with shadcn/ui + Tailwind. Dark theme, large type, plain-language labels. Fed by the live WebSocket or a replay.

## Directory layout
```
backend/
  app.py              FastAPI app, routes, WebSocket
  db.py               SQLite access
  scheduler.py        RaceEngine, ArmRunner, deadline rule, engine rules wiring
  replay.py           race replay (REST + WebSocket + CLI)
  core/               shared contracts: types.py, events.py (see core/README.md)
  agent/
    loop.py           LLMResearcher, retry/fallback policy, prompt wiring
    optuna_arm.py     non-LLM baseline
    mock_llm.py       deterministic dev LLM (never in a benchmark)
    llm_base.py       LLMClient protocol, Mubit constants, cost estimate
    mubit_client.py   the only place LLM clients are constructed
    prompts/          versioned prompt files + build.py message builders
  benchmark/          configs.py (RaceConfig, presets), engine_rules.py, metrics.py, export.py, analysis.py
workers/
  targets/            base.py (spec, registry), lm.py, rl.py, cifar.py, lcbench.py, data/ (no compute SDK imports)
  runners/            base.py, local.py, modal_runner.py, surrogate.py, modal_sapling.py
  verifier/           sapling.py (pins, rows, calibration), sapling_runtime.py (torch side)
scripts/
  bench.py, headline.py, analyze_races.py           benchmark
  collect_examples.py, collect_diagnose.py,         example collection and recording
  collect_early_stop.py, record_judge_v1.py
  gate_judge_v1.py, eval_trap_set.py, eval_noise_set.py, make_trap_set.py, make_noise_set.py
  calibrate_targets.py, probe_latency.py
  sapling_stage.py, sapling_stage_v2.py, make_ml_claims.py, make_ml_claims_extra.py
  dev_server.py, mock_race.py
data/                 bench/, calibration/, collection/, gates/, noise_set/, trap_set/, trap_eval/, sapling/, ml_claims/, early_stop/
tests/                pytest, one file per module
frontend/             Vite React app (not built)
design/               Pen.dev files
docs/
```
