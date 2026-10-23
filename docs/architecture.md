# Architecture (planned)

Status: **design only, no code yet.** Update this doc when the implementation diverges from it.

**Stack (D8):** Python 3.12+ / FastAPI (REST + WebSocket) / SQLite · Vite + React + shadcn/ui + Tailwind (+ Recharts) · Pen.dev for design · pluggable runners (`local`, `modal`) · every LLM call through the Mubit gateway · Banyan Sapling as the verifier.

## Overview
```
                 ┌──────────────────────── FastAPI backend ─────────────────────────┐
 React dashboard │  Race orchestrator ──► Arm session (×2–3)                         │
 (Vite/shadcn)   │        │                  │  agent loop: propose → run → judge    │
   ▲  WebSocket  │        │                  │      │ LLM calls      │ train jobs     │
   └─────────────┤  Event bus ◄──────────────┤      ▼                ▼                │
     REST        │        │            Mubit gateway        Runners: local · Modal   │
                 │     SQLite          (all LLM calls)      LM · RL · CIFAR targets  │
                 │                           │                     │ metrics stream   │
                 │                     Sapling verifier (Modal) ◄──┘                  │
                 └───────────────────────────────────────────────────────────────────┘
```

## 1. Agent loop (`backend/agent/`)
One **arm session** = one researcher working on one target with N slots for a fixed wall-clock budget.
The researcher makes **three recurring LLM calls**. Each is a **stateless single-turn call** with a frozen system prompt and a structured-output JSON schema, so each forms exactly one Mubit task (cost design in `budget.md`):

| Step | When | Input (user message) | Output (JSON) |
|---|---|---|---|
| `propose_experiments` | slots free up | target spec (cached) + top-5 leaderboard + last 8 results + K free slots | `{experiments: [{config, hypothesis}] × ≤K}` |
| `diagnose_run` | **only on failures** | last 50 log lines + metrics | `{failure_class, cause, fix}` |
| `judge_improvement` | **only on a claimed new best** | claimed gain + evidence (split hashes, eval config) | `{verdict: valid/invalid/unsure, reason}` |

`summarize_result` is **deterministic code** (delta vs best, keep/drop), not an LLM call.

- **Arm A (Opus):** every call goes to `claude-opus-5-5` *through Mubit*. These calls also produce the training examples for the adapters.
- **Arm B (Mubit):** the same calls, served by adapters once Mubit routes to them. ⚠️ How we force or observe routing is **Q1**; the API response doesn't say, the console's "Served by" field does.
- **Baseline:** Optuna / random search over the same config space. No LLM calls.

**Every reply is validated against its Pydantic schema.** Adapters don't reliably keep the format (R9: 3 of 7 toy replies were malformed). An invalid reply is logged (`llm_calls.valid = false`) and retried once, then the call falls back to Opus through Mubit. Retries and fallbacks count toward the arm's time and $, and a malformed reply never counts as a fast answer.

**Example coverage matters more than count.** The toy adapter failed exactly the cases its examples lacked (healthy, overfit and contaminated runs). Collected examples for each step must cover the full mix, including "nothing wrong" and contamination cases.

Prompts live in `backend/agent/prompts/` and are **versioned and frozen**. Changing one creates a new Mubit task and resets example collection.

## 2. Mubit integration
### Rules
- `backend/agent/mubit_client.py` is the **only** place LLM clients are constructed. Base URL `https://api.mubit.ai/anthropic`, auth `MUBIT_API_KEY`.
- Per-call variable content (history, metrics, configs) goes in the **user message**, never the system prompt, so each step stays one task (and caches).
- Session per researcher run: `metadata.user_id` (Anthropic) or `metadata.session_id` (OpenAI) = `race-{id}/{arm}/seed-{n}`.
- Capture `x-middleware-request-id` from every response into `llm_calls.mubit_request_id`, linked to the experiment it produced.
- `x-middleware-record: off` on dev/smoke/warm-up calls; `metadata` for steps flagged as sensitive (the data-control story).
- Races run on our Anthropic provider key (no starter-credit limits).
- **Open-model arm / fallback:** `https://api.mubit.ai/compat/{groq|together|openrouter|mistral}/v1`.

### Capability map (D13: use every Mubit capability)
Status: ✅ documented or verified · 📣 marketing/launch claim, verify at office hours · ❓ unknown.

| Mubit capability | Status | How Busy Machines uses it |
|---|---|---|
| **Gateway** | ✅ | Every LLM call: agent steps, the Opus arm, data labelling |
| **Tasks** | ✅ | The three research steps are three clean tasks |
| **Sessions** | ✅ | One session = one researcher run; in the console it's the agent's **procedural log** (Anil's phrase) |
| **Request IDs** | ✅ | On every experiment row, so each GPU result links to the decision that caused it: outcome attribution and a verifiable benchmark |
| **Recording controls** | ✅ | The answer to "labs won't send their traces" |
| **Adapters + versions + grading** | ✅ | Each step gets its own small model; `judge_improvement` v1 → v2 is the correction demo |
| **Try models** | ✅ | Demo moment: v1 vs v2 (or adapter vs Opus) on the contaminated claim |
| **Rate reply Good/Bad** | ✅ UI / ❓ effect | Correction channel if there's no API (Q9) |
| **OpenAI-compatible services** | ✅ | Open-model arm ("own your intelligence layer") and the fallback if adapters can't serve live traffic |
| **Import past calls** | ✅ | Pitch only: day-0 onboarding for labs. Can't demo it (one project per workspace, imports only before the first live call) |
| **Coding-tool integration** | ✅ | Zero-build adoption: researchers' Claude Code/Codex sessions also flow through Mubit |
| **Outcome learning** | 📣 | The verified training metric *is* the outcome, attached by request ID. How adapters could beat the teacher, not just copy it |
| **Routing to the dedicated model only when better** | 📣 (launch) | The Mubit arm runs on adapters automatically once they beat the provider (Q1) |
| **Change control / rollback** | 📣 | Roll an adapter back if verifier accuracy drops; show version history |
| **Experience reuse** | 📣 | Lessons carry across targets (e.g. warm-up findings from LM to CIFAR) |
| **Prompt / skill refinement** | 📣 | Post-event only: applying a suggestion creates a new task |
| **Banyan Sapling** | ✅ (HF) | Independent verifier (`sapling_research.md`) |
| **latte/bench** | ✅ | `night/bench` is its research counterpart |
| **Mubit's metrics** (task success, repeated failures, latency, cost per successful task) | 📣 | The dashboard shows these per research step, plus GPU $ |

## 3. Scheduler (`backend/scheduler.py`)
- Each arm has N slots. When a slot frees up, the arm needs a proposal; while the LLM is thinking, **the slot is idle and the idle clock runs**.
- Proposals may be pipelined (ask for the next batch while runs are in flight). Both arms use the same scheduler logic.
- Records `gpu_interval(slot, state=busy|idle, start, end)`. **GPU idle % = idle time / paid time.**
- Per-arm $ cap: stops the arm when the running cost hits the cap (logged).

## 4. Compute: pluggable runners (`workers/`)
**The product is compute-agnostic (D17). Modal is only our test/demo compute.** A lab would run it on its own cluster. Targets never import a compute SDK; only runners do, and `modal` is imported only in `workers/runners/modal_runner.py`.

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
| `local` | Dev and tests on the laptop (mock LLM, tiny CPU runs) | Subprocess per run; $0. **No local GPUs**: every GPU run goes through `modal` |
| `modal` | The demo and benchmark | Modal Functions (not Sandboxes); metrics back via `modal.Queue` polled by the backend; pre-warm before the demo. Enforces Starter caps (**10 GPUs + 100 containers**) so a race never queues silently (queued time would count as fake idle) |
| `slurm` / `k8s` / `ray` | **Customer deployment (pitch only)** | Same interface; not built |

- Metrics stream through `on_metric`, independent of the runner.
- Targets implement a common interface:
  ```python
  class Target(Protocol):
      name: str                 # "lm" | "rl" | "cifar"
      metric: str               # "val_bpb" | "mean_return" | "test_acc"
      direction: Literal["min", "max"]
      target_value: float       # for $-to-target / time-to-target
      hardware: str             # "L4" | "CPU"
      budget_s: int             # per-run training budget (20–90 s, e.g. 60 for lm)
      config_space: dict        # for proposals and for Optuna
  ```
- Held-out evaluation happens inside the target on a split the agent never sees.

## 5. Verifier (`workers/verifier/`)
- Banyan Sapling (or our `sapling-research`) on an 80 GB GPU runner (Modal H100 for the demo). It checks claims like "run X improved the metric by Y" against evidence: split hashes, eval config, train/eval overlap.
- Its flags become **corrections**, alongside human flags from the UI. ⚠️ Whether it must sit behind Mubit is **Q5**.

## 6. Contamination trap
One option per target silently **contaminates the evaluation**; the ground truth is known:
- LM: the validation shard overlaps the training shards
- CIFAR: the test set is used for checkpoint selection or augmentation tuning
- RL: evaluation on the training seeds/levels

Used for the live correction moment and the contamination-trap accuracy table (Opus vs adapter v1 vs v2).

## 7. Corrections → model update
UI flag or Sapling flag → `corrections` row → sent to Mubit (mechanism: **Q2/Q9**; console Good/Bad rating as fallback) → a new adapter version is trained; the version history is shown in the console during the demo.

## Data model (SQLite)
```
races          id, name, target, budget_s, gpus_per_arm, seed, started_at, ended_at
arms           id, race_id, kind (opus|mubit|optuna), model
experiments    id, arm_id, config_json, hypothesis, status, runner, run_ref,
               started_at, ended_at, score, is_best, is_trap
metrics        experiment_id, step, t, name, value              -- streamed curves
llm_calls      id, arm_id, step_type, mubit_request_id, model, latency_ms,
               input_tokens, cache_creation_tokens, cache_read_tokens,
               output_tokens, cost_usd, valid, is_retry, is_fallback,
               is_replay, created_at
gpu_intervals  arm_id, slot, state (busy|idle), start, end
corrections    id, experiment_id, source (human|sapling), label, reason,
               sent_to_mubit_at
adapter_versions  step_type, version, status, trained_at, notes   -- mirrored from console
```
`run_ref` holds the Modal call ID. Export: `GET /api/races/{id}/export.jsonl` dumps everything for verification.

## Cost accounting
- **GPU $** = (busy + idle seconds) × the runner's configured rate. The demo uses Modal list prices (L4 $0.000222/s, H100 $0.001097/s, CPU core $0.0000131/s); a customer enters their own $/GPU-hour.
- **LLM $** = the four token meters × provider price (or the cost Mubit reports).
- **$-to-target** = GPU $ + LLM $ until the first experiment reaching `target_value`.

## API (FastAPI)
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/races` | create a race (target, arms, budget, gpus_per_arm, seed) |
| POST | `/api/races/{id}/start` | start all arms simultaneously |
| GET | `/api/races/{id}` | state and summary stats |
| GET | `/api/races/{id}/export.jsonl` | the verification dump |
| POST | `/api/experiments/{id}/flag` | human correction (fake improvement) |
| GET | `/api/adapters` | adapter versions per step |
| WS | `/ws/races/{id}` | live event stream |

**WebSocket events:** `gpu_state`, `experiment_started`, `metric`, `experiment_finished`, `new_best`, `llm_call`, `correction`, `race_finished`.

## Frontend (`frontend/`)
The "Night Shift Race" split screen (`demo.md`): GPU tiles, live curves, leaderboard ticker, thought stream, cost panel, flag button, and a results page with the 3-run charts. Designed in Pen.dev first, built with shadcn/ui + Tailwind. Dark theme, large type, plain-language labels.

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
  targets/            lm.py, rl.py, cifar.py (pure Python, no compute SDK imports)
  runners/            base.py (Runner protocol), local.py, modal_runner.py
  verifier/           Sapling serving
frontend/             Vite React app
design/               Pen.dev files
docs/
```
