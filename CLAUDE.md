# CLAUDE.md

Guidance for AI coding agents (Claude Code, Codex, etc.) working in this repo. Read this first, then `docs/README.md`.

## What this project is
Our entry for **Night Shift London**, the $65K Mubit hackathon (23–25 Oct 2026).

**Project name: Busy Machines.** The name is the pitch: GPUs that are always busy, never idle. It is an ML research agent that improves any ML/DL/RL model by proposing, running and judging experiments on GPUs. All of its LLM calls go through **Mubit**, which distils the agent's recurring steps into small owned models ("adapters"). The result is that the agent waits less, keeps the GPUs busy and costs less per discovery.

**Customer: AI labs only** (research, RL/post-training and eval teams). Don't add bank/fraud/enterprise framing (D18).
**Pitch:** *"Same GPUs, same budget. The Mubit-powered researcher gets more science out of it."*
**Headline metrics:** **$-to-target**, **time-to-target**, **GPU idle %**, and **best score at a fixed budget** vs an Opus 5.5 agent.
**Benchmark working name:** `night/bench`, which complements Mubit's own `latte/bench` (theirs measures note-based learning; ours measures weight learning on real tasks, costed in GPU $).

## Priorities (D20)
1. **MUST:** a fair benchmark vs Opus 5.5 that proves **train faster, better model, less money** (`docs/benchmark.md`, `docs/architecture.md`).
2. **OPTIONAL:** `sapling-research`, our fine-tune of Mubit's Banyan Sapling for spotting fake improvements in AI-research results (`docs/sapling_research.md`). It never takes time from #1. Hard stop at Sat 18:00.

## Hard rules (breaking these disqualifies us)
1. **Every LLM call must go through Mubit.** Never call OpenAI/Anthropic/Gemini directly.
   - OpenAI: `https://api.mubit.ai/openai/v1`
   - Anthropic: `https://api.mubit.ai/anthropic` (no `/v1`)
   - Gemini: `https://api.mubit.ai/gemini` (no `/v1beta`)
   - Auth: the `MUBIT_API_KEY` env var. Keep provider model IDs unchanged (e.g. `claude-opus-5-5`).
   - **Claude Code itself is a dev tool, not the product's LLM.** Never use the coding session to stand in for runtime calls (the Opus arm, example collection, labelling, verification). Those must be API calls through Mubit. See D15 in `docs/decisions.md`.
2. **All project code is written during the event** (from Fri 23 Oct, 19:30). Open-source libraries are fine.
3. **The demo must show a correction turning into a model update** (judging criterion #1).
4. **Code freeze: Sun 25 Oct, 06:00 London time: the LAST moment to push code and submit** (moved from 10:00 by the user on Sat 24 Oct, D53). Clocks go back at 02:00 on Sunday.

## Judging (equal weight)
Use of Mubit (correction → model update) · Usefulness (real recurring task) · Technical execution (works, holds up under questions) · Demo (clear, live, under the time limit). Details: `docs/context.md`.

## Tech stack (decided; don't propose alternatives unless asked)
| Layer | Choice |
|---|---|
| Backend | Python 3.12+, **FastAPI** (REST + WebSocket) |
| DB | **SQLite** (move to PostgreSQL only if there's a real need) |
| Frontend | **Vite + React**, **shadcn/ui**, **Tailwind CSS** |
| Design | **Pen.dev**. `.pen` files are encrypted, so access them only through the pencil MCP tools, never Read/Grep |
| Compute | **Compute-agnostic product.** A pluggable `Runner` interface (`local`, `modal`). **Modal is only our test/demo compute** (Starter: $30 credit, max 10 concurrent GPUs; use Functions, not Sandboxes). Never import `modal` outside `workers/runners/modal_*.py` (`modal_runner.py` for training, `modal_sapling.py` for the verifier, D28) |
| LLM routing | Mubit gateway (see the hard rules) |
| Verifier | Banyan Sapling (`MubitLabs/banyan-sapling`, Apache 2.0) as the independent claim checker |

## Repo layout
```
CLAUDE.md          ← you are here (agent guidance)
AGENTS.md          ← pointer for non-Claude agents
README.md          ← human-facing overview
docs/              ← all planning, research and design docs (index: docs/README.md)
backend/           ← (planned) FastAPI app, agent loop, benchmark harness
frontend/          ← (planned) Vite React race dashboard
workers/           ← (planned) targets (LM speedrun, RL, CIFAR speedrun), runners (local, modal), verifier
design/            ← (planned) Pen.dev design files
```
The planned architecture, data model and API are in `docs/architecture.md`.

## Conventions
- **Results log:** every measurement goes into `docs/results_log.md`, **including results where we're worse**. Never delete, hide, or cherry-pick re-runs. Slides and the README may quote only logged numbers and must not omit a worse result on the same comparison.
- **Claims discipline:** never put an unmeasured number in the UI, README or slides. Placeholders in the docs are labelled as placeholders. Don't claim "better than Opus 5.5" per call. The claim is **better at a fixed GPU/time/$ budget** (see `docs/benchmark.md`).
- **Fairness in benchmarks:** both arms get the same GPUs, concurrency, wall-clock, $ cap, seeds and starting code. Only the LLM in the loop differs. Always include a non-LLM baseline (Optuna/random search).
- **Freeze prompts early.** Mubit groups calls into a task by an *identical* model, instructions, tools, tool choice and output format. Any prompt drift splits the task and resets example collection (25 kept examples are needed before the first training).
- **Log everything traceable:** store the Mubit request ID, Modal call ID, seed and config for every experiment in SQLite, and export to JSONL for verification.
- **Secrets:** the Anthropic key is stored only in Mubit → Provider keys and a git-ignored `.env`. The app uses `MUBIT_API_KEY` only. Never commit keys.
- **Budgets:** LLM ≈ $287 Anthropic + $25 Mubit (see `docs/budget.md`); GPU = Modal $30, which is the real bottleneck. **No local GPUs**: the laptop runs mock/tiny CPU tests only, and everything else runs on Modal (10 GPUs / 100 containers at once).
- **LLM cost rules** (`docs/budget.md`):
  - stateless calls, laid out as system → target spec → dynamic digest, cached
  - structured outputs
  - batch K proposals per call; diagnose only on failures, judge only on a new best
  - `x-middleware-record: off` on non-example calls
  - develop on Haiku via the Mubit credit or the mock LLM; Opus only for frozen prompts
  - both arms get the same optimisations
- **Budget guard:** check estimated GPU $ before launching sweeps. Never launch H100s without a reason (the L4 at $0.80/h is the default).
- **Python:** type hints, Pydantic models for API schemas, `uv` for dependency management.
- **Frontend:** dark theme, large type (it is shown on a stage screen), shadcn components before custom ones.
- **UI copy is plain language** (`docs/demo.md`, `docs/pitch.md`): "GPUs waiting", "Cost to reach the goal", "Model quality", "Fake improvement caught". Never show jargon (val_bpb, adapter, RL, idle %) as a main label; raw metrics only on hover or in the export.
- **Git:** commit or push only when the user asks. End commit messages with the Co-Authored-By line from the session's attribution reminder.

## Mubit facts agents get wrong
- Adapters are **graded against the provider model's answers** ("Passed" means close enough to the provider), not against ground truth.
- Limits: **3 adapter starts per org per UTC day**, a daily training limit, a pause between trainings, and cold starts of up to about 20 minutes. Plan the training schedule; don't trigger trainings casually.
- Mubit routes to a dedicated model only when it performs better (per the 23 Oct launch).
- Pipeline timing (toy task): calls are grouped ~10 minutes after they arrive, reviewed, and trained in ~5 minutes, so roughly 20–25 minutes from calls to adapter. A trained adapter serves nothing until it passes its exam (85% quality target vs the provider), goes through shadow, and has a route pointing to it.
- Mubit launched on 23 Oct 2026: the **platform**, **Banyan Sapling** (open verifier model) and **latte/bench**. See `docs/context.md`.

## Open questions
These are tracked in `docs/decisions.md`. Resolve them with the Mubit team at office hours before building anything that depends on them.
