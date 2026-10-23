# Busy Machines

> **Spend less on GPUs. Get a better model. Train faster.**
> Your GPUs should never sit waiting for your AI assistant.

Built at **Night Shift London**, the Mubit hackathon (23–25 Oct 2026).

**Built for AI labs:** research, RL/post-training and eval teams whose GPU budgets are spent on thousands of short experiments.

## The problem
AI agents now run ML experiments: they propose changes, launch training, read results and iterate. But **every time the agent stops to think, the GPUs sit idle**, and you pay for them anyway. Frontier-model reasoning is slow and expensive, and the agent's thinking becomes the bottleneck on research throughput.

## What we built
An ML research agent that improves **any ML/DL/RL model** whose training can be scripted. Its LLM calls go through **[Mubit](https://mubit.ai)**:

- The agent's **recurring steps** (propose the next experiment, diagnose a run, summarise a result, judge an improvement) are distilled by Mubit into **small task-specific models** that answer in a fraction of the time and cost.
- An **independent verifier** ([Banyan Sapling](https://huggingface.co/MubitLabs/banyan-sapling)) checks whether each claimed improvement is supported by the evidence, catching **eval contamination** (test/val overlap with training data) and overfitting.
- **Every human or verifier correction becomes a model update.** The agent gets better each night.

## The benchmark: `night/bench`
Two AI researchers race on **identical GPUs with identical budgets**:

| Arm | LLM in the loop |
|---|---|
| A | Opus 5.5 for every step |
| B | Mubit adapters for the recurring steps |
| Baseline | Optuna / random search (no LLM) |

**Tasks (the work AI labs run thousands of times):** LM pretraining speedrun (nanochat-style), RL (gymnax / Craftax), vision speedrun (CIFAR-10).
**Metrics:** $-to-target · time-to-target · GPU idle % · experiments per GPU-hour · best held-out score at a fixed budget. At least 3 seeds per arm, and every number traces back to a Mubit request ID and a Modal run ID.

> Results: _to be filled with measured numbers only._

## Stack
FastAPI · SQLite · Vite + React · shadcn/ui · Tailwind CSS · Modal · Mubit · Pen.dev

## Repo
- [`docs/`](docs/README.md): research, plans, architecture and decisions
- [`CLAUDE.md`](CLAUDE.md): guidance for AI coding agents
- `backend/`, `frontend/`, `workers/`: _coming during the hackathon_

Busy Machines is **compute-agnostic**: training runs go through a pluggable runner (local machine, Modal, and in production a lab's own cluster). We use Modal GPUs for the demo.

## Running it
_Set-up instructions will be added once the code exists._ Required env vars (planned): `MUBIT_API_KEY`, plus Modal auth (`modal token new`).
