# Architecture (planned)

Status: **design only, no code yet.** Update this doc when the implementation diverges from it.

## Overview
```
                 ┌──────────────────────── FastAPI backend ─────────────────────────┐
 React dashboard │  Race orchestrator ──► Arm session (×2–3)                         │
 (Vite/shadcn)   │        │                  │  agent loop: propose → run → judge    │
   ▲  WebSocket  │        │                  │      │ LLM calls      │ train jobs     │
   └─────────────┤  Event bus ◄──────────────┤      ▼                ▼                │
     REST        │        │            Mubit gateway        Runners: local · Modal   │
                 │     SQLite          (all LLM calls)      nanochat · RL · CIFAR     │
                 │                           │                     │ metrics stream   │
                 │                     Sapling verifier (Modal) ◄──┘                  │
                 └───────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Agent loop (`backend/agent/`)
One **arm session** = one researcher working on one target with N GPU slots for a fixed wall-clock budget.
The researcher makes **three recurring LLM calls** (cost design in `18_llm_cost_optimization.md`). Each one is a **stateless single-turn call** with a frozen system prompt and a structured-output JSON schema, so each forms exactly one Mubit task:

| Step | When | Input (user message) | Output (JSON) |
|---|---|---|---|
| `propose_experiments` | a GPU slot frees up | target spec (cached) + top-5 leaderboard + last 8 results + K free slots | `{experiments: [{config, hypothesis}] × ≤K}` |
| `diagnose_run` | **only on failures** | last 50 log lines + metrics | `{failure_class, cause, fix}` |
| `judge_improvement` | **only on a claimed new best** | claimed gain + evidence (split hashes, eval config) | `{verdict: valid/invalid/unsure, reason}` |

`summarize_result` is **deterministic code** (delta vs best, keep/drop), not an LLM call.

- **Arm A (Opus):** every call goes to `claude-opus-5-5` *through Mubit*. These calls also produce the training examples for the adapters.
- **Arm B (Mubit):** the same calls, served by adapters. ⚠️ How we force or observe adapter routing depends on **Q1** in `decisions.md`.
- **Baseline:** Optuna / random search over the same config space. It makes no LLM calls.

Prompts live in `backend/agent/prompts/` and are **versioned and frozen**. Changing one creates a new Mubit task.

**Mubit integration rules** (full map in `16_full_mubit_integration.md`):
- Per-call variable content (history, metrics, configs) goes in the **user message**, never the system prompt, so each step stays one task.
- Set the session per researcher run: `metadata.user_id` (Anthropic) or `metadata.session_id` (OpenAI) = `race-{id}/{arm}/seed-{n}`.
- Capture `x-middleware-request-id` from every response into `llm_calls.mubit_request_id`, and link it to the experiment it produced.
- Use `x-middleware-record: metadata` for steps flagged as sensitive (the enterprise data-control story).
- **Open-model arm / fallback:** `https://api.mubit.ai/compat/{groq|together|openrouter|mistral}/v1`.
- Use our own provider key for races (the starter credit has a concurrency limit and returns `429 starter_credit_busy`).

### 2. Scheduler (`backend/scheduler.py`)
- Each arm has N GPU slots. When a slot frees up, the arm needs a proposal; while the LLM is thinking, **the slot is idle and the idle clock runs**.
- Proposals may be pipelined (ask for the next proposal while runs are in flight). Both arms use the same scheduler logic for fairness.
- It records `gpu_interval(slot, state=busy|idle, start, end)`. **GPU idle % = idle time / paid time.**

### 3. Compute: pluggable runners (`workers/`)
**The product is compute-agnostic. Modal is only the compute we use to test and demo.** An AI lab would run it on its own cluster. Targets never import a compute SDK; only runners do.

```python
class Runner(Protocol):
    name: str                                   # "local" | "modal" | later "slurm" | "k8s" | "ray"
    def slots(self) -> list[Slot]: ...           # available GPU/CPU slots
    async def launch(self, target: str, config: dict, seed: int,
                     budget_s: int, on_metric: Callable) -> RunHandle: ...
    async def cancel(self, handle: RunHandle) -> None: ...
    def price_per_second(self, hardware: str) -> float: ...   # from config, for cost accounting
```

| Runner | Use | Notes |
|---|---|---|
| `local` | Development and tests on the laptop (mock LLM, tiny CPU runs) | Subprocess per run; $0. **We have no local GPUs**: every GPU run goes through `modal` |
| `modal` | The demo and benchmark at scale | Modal Functions (not Sandboxes); pre-warm containers before the demo. Starter limits: **10 GPUs + 100 containers at once**. The runner enforces slot caps so a race never queues silently (a queued slot would count as fake idle time) |
| `slurm` / `k8s` / `ray` | **Customer deployment (pitch only)** | Same interface; mentioned in the sales story, not built |

- Metrics stream back through a callback (`on_metric`), independent of the runner, so the dashboard doesn't care where training runs.
- Targets implement a common interface:
  ```python
  class Target(Protocol):
      name: str                 # "nanochat" | "rl" | "cifar"
      metric: str               # "val_bpb" | "mean_return" | "test_acc"
      direction: Literal["min", "max"]
      target_value: float       # for $-to-target / time-to-target
      hardware: str             # "L4" | "CPU"
      budget_s: int             # per-run training budget (e.g. 300)
      config_space: dict        # for proposals and for Optuna
  ```
- Held-out evaluation happens inside the target on a split the agent never sees.

### 4. Verifier (`workers/verifier/`)
- Banyan Sapling, served on whatever GPU runner is available (Modal for the demo: it needs an A100-80GB/H100). It checks claims of the form "run X improved metric by Y" against evidence: data split hashes, eval config, train/eval overlap.
- Its flags become **corrections** (along with human flags from the UI). ⚠️ Whether it must sit behind Mubit is **Q5**.

### 5. Contamination trap
- One option per target silently **contaminates the evaluation**, the classic AI-lab failure. The ground truth is known.
  - nanochat: the validation shard overlaps the training shards
  - CIFAR: the test set is used for checkpoint selection or augmentation tuning
  - RL: evaluation on the training seeds/levels
- It is used for the live correction moment and for the contamination-trap accuracy table (Opus vs adapter v1 vs v2).

### 6. Corrections → model update
UI flag or Sapling flag → `corrections` row → sent to Mubit (mechanism: **Q2**) → a new adapter version is trained, and the version history is shown in the Mubit console during the demo.

## Data model (SQLite)
```
races          id, name, target, budget_s, gpus_per_arm, seed, started_at, ended_at
arms           id, race_id, kind (opus|mubit|optuna), model
experiments    id, arm_id, config_json, hypothesis, status, runner, run_ref,
               started_at, ended_at, score, is_best, is_trap
metrics        experiment_id, step, t, name, value              -- streamed curves
llm_calls      id, arm_id, step_type, mubit_request_id, model, latency_ms,
               input_tokens, output_tokens, cost_usd, created_at
gpu_intervals  arm_id, slot, state (busy|idle), start, end
corrections    id, experiment_id, source (human|sapling), label, reason,
               sent_to_mubit_at
adapter_versions  step_type, version, status, trained_at, notes   -- mirrored from console
```
Export: `GET /api/races/{id}/export.jsonl` dumps everything for verification.

## Cost accounting
- **GPU $** = busy + idle seconds × the runner's configured rate. The demo uses Modal's list prices (L4 $0.000222/s, H100 $0.001097/s, CPU core $0.0000131/s); a customer enters their own $/GPU-hour (reserved cluster, cloud contract).
- **LLM $** = tokens × the provider price (or the cost Mubit reports).
- **$-to-target** = GPU $ + LLM $ until the first experiment reaching `target_value`.

## API (FastAPI)
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/races` | create a race (target, arms, budget, gpus_per_arm, seed) |
| POST | `/api/races/{id}/start` | start all arms simultaneously |
| GET | `/api/races/{id}` | state and summary stats |
| GET | `/api/races/{id}/export.jsonl` | the verification dump |
| POST | `/api/experiments/{id}/flag` | human correction (e.g. leakage) |
| GET | `/api/adapters` | adapter versions per step |
| WS | `/ws/races/{id}` | live event stream |

**WebSocket events:** `gpu_state`, `experiment_started`, `metric`, `experiment_finished`, `new_best`, `llm_call`, `correction`, `race_finished`.

## Frontend (`frontend/`)
The "Night Shift Race" split screen (see `09_demo_showcase.md`):
- GPU slot tiles (busy/idle), live loss curves, a leaderboard ticker, a thought stream, a cost breakdown (GPU $, idle $, LLM $), and the flag button.
- A results page with the 1-hour × 3-seed charts and the H100 replication table.
- Designed in Pen.dev first, then built with shadcn/ui + Tailwind. Dark theme with large type.

## Planned directory layout
```
backend/
  app.py              FastAPI app, routes, WebSocket
  db.py               SQLite access
  scheduler.py
  agent/
    loop.py
    prompts/          frozen, versioned prompt files
    mubit_client.py   the only place LLM clients are constructed
  benchmark/          race configs, export, analysis
workers/
  targets/            nanochat.py, rl.py, cifar.py (pure Python, no compute SDK imports)
  runners/            base.py (Runner protocol), local.py, modal_runner.py
  verifier/           Sapling serving
frontend/             Vite React app
design/               Pen.dev files
docs/
```
