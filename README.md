# Busy Machines

> **Spend less on GPUs. Get a better model. Train faster.**
> Your GPUs should never sit waiting for your AI assistant.

Built at **Night Shift London**, the Mubit hackathon (23–25 Oct 2026).

**Built for AI labs:** research, RL/post-training and eval teams whose GPU budgets are spent on thousands of short experiments.

## The problem
AI agents now run ML experiments: they propose changes, launch training, read results and iterate. But **every time the agent stops to think, the GPUs sit idle**, and you pay for them anyway. Frontier-model reasoning is slow and expensive, and the agent's thinking becomes the bottleneck on research throughput.

## What we built
An ML research agent that improves **any ML/DL/RL model** whose training can be scripted. Its LLM calls go through **[Mubit](https://mubit.ai)**:

- The agent's **recurring steps** (propose the next experiment, diagnose a failed run, judge a claimed improvement; summarising a result is plain code) are recorded by Mubit so it can distil them into **small task-specific models**, which are meant to answer faster and cheaper than the frontier model. **Not measured yet:** none of our small models has passed its exam or answered a call in a race [R7, R28, R30].
- A **checker** decides whether a claimed new best is real: in races, the LLM judge looks for **eval contamination** (test/val overlap with training data) and a seed-spread rule looks for noise (D40). Offline, our fine-tuned **[Banyan Sapling](https://huggingface.co/MubitLabs/banyan-sapling)** verifier catches some of the fakes the stock model leaves unresolved [R33]. Sapling is not in the live race loop yet (D28).
- **Every human or verifier correction is stored against its Mubit request ID**, so it can become training for the next version of the small model. Correction → model update is not shown end to end yet: Mubit has no corrections API (Q2, Q9).

## The benchmark: `night/bench`
Two AI researchers race on **identical GPUs with identical budgets**:

| Arm | LLM in the loop |
|---|---|
| A | Opus 5.5 for every step (the standard assistant) |
| B | Busy Machines with Opus: our engine rules, Opus 5.5 for every step (D36) |
| C | Busy Machines + Mubit: our engine rules; Mubit's small model answers a step only where it passed its exam, the provider answers the rest |
| Baseline | Optuna / random search (no LLM) |

**Tasks (the work AI labs run thousands of times):** RL (Pendulum, JAX PPO, on CPU), vision speedrun (CIFAR-10, airbench-style, on L4), tabular model tuning (LCBench surrogate, simulated GPU time). The LM pretraining speedrun (char-level nanoGPT) is demoted: no setting beat its starting code [R23, D32].
**Metrics:** $-to-target · time-to-target · GPU idle % · experiments per GPU-hour · best held-out score at a fixed budget. At least 3 seeds per arm for a headline number, and every number traces back to a Mubit request ID and a runner ID (a Modal call ID, or the surrogate's ID for LCBench).

## Results so far (measured)
As of Sat 24 Oct, ~05:40 BST (results log up to R40). Every number comes from [`docs/results_log.md`](docs/results_log.md) (ids in brackets), and the worse result on the same comparison is listed beside it. "3 runs" means 3 seeds, mean ± std.

**Not measured yet: the headline.** We have no result yet for Busy Machines with Mubit's small models vs an Opus 5.5 agent (cost to reach the goal, time to reach it, GPUs waiting, model quality at a fixed budget). Mubit's first adapters scored below their exam targets (propose 14.3% vs 95%, judge 42.9% vs 95%, toy task 70% vs 85%), and none has a route, so none of them has answered a call in our races [R7, R28, R30]. Everything below uses Opus 5.5 as the LLM, or no LLM at all.

| Comparison | Result | Worse side / caveats |
|---|---|---|
| Opus 5.5 research agent vs Optuna (no AI), LCBench tuning task, 3 runs, simulated GPU time [R38] | Best model 0.856 ± 0.010 vs 0.799 ± 0.022 held-out balanced accuracy: **28% fewer errors**. Optuna never reached Opus's final quality (0/3). Fakes accepted as a new best: **Opus 0/33, Optuna 8/62** | Opus ran **0.3× the experiments per $** (22.7 vs 80.9); GPUs waited 5.0% vs 0.3%; in seed 2 **Optuna reached the goal faster and cheaper** (108 s / $0.38 vs 193 s / $0.96). GPU $ is simulated; Opus may know LCBench from training. Cost and time to the goal are **not measured** as a 3-run comparison (Optuna missed the goal in seed 0). In seed 0, Optuna's asks briefly froze the shared event loop, slightly against the Opus arm (R35, since fixed) |
| Busy Machines judge (Opus checks for leakage, a seed-spread rule checks for noise) vs Opus alone, 75 stored claims with 10-seed ground truth [R37, R34] | Real gains confirmed **7/7 vs 1/7**; fakes let through **0/32 for both** | Noise accepted 1/36 vs Opus's 0/36. Only 7 real gains, offline (not yet live in a race), and the rule and the ground truth both rest on seed spread. The rule alone lets 24/32 fakes through. With the closed-output judge v1 now in use, test split only: 4/4 real, 1/20 noise, 0/16 fakes, and Opus v1 alone confirms 0/4 [R40] |
| Research agent's best settings vs the starting code [R23] | Image recognition (CIFAR-10, 3 runs each): 0.9277 → 0.9371, **13% fewer errors**. Game-playing AI (Pendulum; start 10 runs, best 3 runs): −966 → −164.4, **5.9× smaller penalty** | Language model: **no gain** (2.5222 → 2.5219). An earlier 18% CIFAR figure was one lucky run (R22, superseded). Opus only, no Optuna comparison |
| Optuna with no checker, game-playing race, 1 run [R36] | 643 of its 677 runs used the trap setting; its reported best is fake (held-out −201.2 vs Opus's best −163.8) | Optuna reached the (too easy) goal far sooner and cheaper: 18.8 s / $0.008 vs 79 s / $0.256. The race is **invalid for GPUs waiting and throughput** (a runner bug, since fixed); re-run pending |
| Opus planning time vs batch size [R36] | Typical call: 3.5 s for 1 proposal, 16.0 s for 8 | 1 race. How long GPUs wait at 32 workers is **not measured yet**; at 2–8 workers with 30–70 s runs, GPUs waited only 7.6–16.8% [R21] |
| `sapling-research` (verification head fine-tuned, backbone frozen) vs stock Banyan Sapling, 500 held-out synthetic research claims [R33] | Raw accuracy 22.0% → 58.2%; fakes caught **0 → 21** (α 0.01) / 43 (α 0.05) of 157; fakes approved 0/157 for both | Opus 5.5 is a far better judge on the trap set (50/50 vs 14–18/50), and at α 0.05 ours lets 2/25 trap-set fakes through; 82–94% of claims still come back "unsure"; real gains accepted only 0–1 of 132 |

## Stack
FastAPI · SQLite · Vite + React · shadcn/ui · Tailwind CSS · Modal · Mubit · Pen.dev

## Repo
- [`docs/`](docs/README.md): research, plans, architecture and decisions
- [`CLAUDE.md`](CLAUDE.md): guidance for AI coding agents
- `backend/`: FastAPI app, race engine, SQLite, agent loop, Mubit client, benchmark package
- `workers/`: targets (`lm`, `cifar` on L4, `rl` on CPU) and runners (`local`, `modal`)
- `scripts/`: dev server, mock race, latency probe, benchmark pre-flight
- `frontend/`: _later_

Busy Machines is **compute-agnostic**: training runs go through a pluggable runner (local machine, Modal, and in production a lab's own cluster). We use Modal GPUs for the demo.

## Running it
Python 3.12 and [uv](https://docs.astral.sh/uv/). Every command runs from the repo root. `make <target>` wraps the same commands where `make` exists.

```bash
uv sync --extra ml                                   # backend + CPU torch + jax/optax + dev tools (add --extra rlcheck for the gymnax cross-check test)
uv run --extra ml pytest -q                          # full test suite (no network, no GPU)
uv run --extra ml python scripts/dev_server.py       # API on http://127.0.0.1:8000 (auto-reload)
uv run --extra ml python scripts/mock_race.py -v     # end-to-end race through the API: mock LLM, laptop CPU, $0
```

`scripts/mock_race.py` starts its own server, creates and starts a race, follows the WebSocket stream, then prints each arm's summary and saves the JSONL export. Options: `--target rl|cifar`, `--budget`, `--run-budget`, `--slots`, `--opus-latency`, `--mubit-latency`, `--no-optuna`, `--url` (use a running server). Its numbers come from a simulated LLM, so they are not benchmark results.

**Real LLM calls** go through Mubit only. Put `MUBIT_API_KEY` in `.env` (copy `.env.example`). The app never reads a provider key.
```bash
uv run python scripts/probe_latency.py --n 3 --max-usd 0.30   # Opus vs Haiku latency, record off (~$0.07)
```

**Modal** (test/demo compute): run `modal token new` once, then:
```bash
uv run python -m workers.runners.modal_runner smoke rl      # one RL run on a CPU container (~$0.001)
uv run python -m workers.runners.modal_runner smoke cifar   # one CIFAR run on one L4 (~$0.01)
uv run modal deploy workers/runners/modal_runner.py         # needed before races with runner="modal"
```

**Benchmark** (`night/bench`, stub for now):
```bash
uv run python scripts/bench.py list            # presets with their GPU $ upper bound
uv run python scripts/bench.py check lm-30m    # pre-flight: frozen prompts, frozen target_value, $ guard
```
`bench.py run` refuses until the prompts and thresholds are frozen.

Env vars: `MUBIT_API_KEY` (real LLM calls), `BUSY_MACHINES_DB` (SQLite path, default `data/busy_machines.db`), `BUSY_MODAL_CPU_CORES` (cores per CPU container, default 1), `BUSY_MODAL_SCALEDOWN_S` (warm time per container, default 60 s, billed).
