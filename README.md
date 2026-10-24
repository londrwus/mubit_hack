# Busy Machines

> **Built to get a better model for less money, from the AI research assistant you already use.**
> Measured so far, on two tasks, 3 runs each: **the same model as a standard Opus 5.5 agent for a quarter of the AI bill** (AI $ is a list-price estimate). Game-playing: a tie, not a better one, 4.0× less AI money (R78). Tabular tuning: within 0.0013, with the standard agent slightly ahead in every run, 4.0× less AI money (R82, pre-registered, success line met). R82's setting was chosen after our first pre-registered tabular race **failed**: 8.2× less AI money, but 7% more errors (R75). How: the machines stop waiting while the assistant thinks, so the same budget runs more experiments. See "Results so far".

Built at **Night Shift London**, the Mubit hackathon (23–25 Oct 2026).

**Built for AI labs:** research, RL/post-training and eval teams whose GPU budgets are spent on thousands of short experiments.

## The problem
AI agents now run ML experiments: they propose changes, launch training, read results and iterate. But **every time the agent stops to think, the GPUs sit idle**: with 30 workers and 15-second runs, a standard Opus 5.5 agent left them waiting 69% of the time [R48]. Frontier-model reasoning is slow and expensive: on that task 90.6% of the agent's spend was its own thinking, not GPUs [R54]. So the agent's thinking is both the bottleneck on research throughput and most of the bill.

## What we built
An ML research agent that improves **any ML/DL/RL model** whose training can be scripted. Its LLM calls go through **[Mubit](https://mubit.ai)**:

- The agent's **recurring steps** (propose the next experiment, diagnose a failed run, judge a claimed improvement; summarising a result is plain code) are recorded by Mubit so it can distil them into **small task-specific models**, which are meant to answer faster and cheaper than the frontier model. **Not measured yet:** none of our small models has passed its exam or answered a call in a race [R7, R28, R30, R70].
- A **checker** decides whether a claimed new best is real: in races, the LLM judge looks for **eval contamination** (test/val overlap with training data) and a seed-spread rule looks for noise (D40). Offline, our fine-tuned **[Banyan Sapling](https://huggingface.co/MubitLabs/banyan-sapling)** verifier catches some of the fakes the stock model leaves unresolved [R42]. Sapling is not in the live race loop yet (D28).
- **Every human or verifier correction is stored against its Mubit request ID**, so it can become training for the next version of the small model. Correction → model update is not shown end to end yet: Mubit has no corrections API (Q2, Q9).

## The benchmark: `night/bench`
Two AI researchers race on **identical GPUs with identical budgets**:

| Arm | LLM in the loop |
|---|---|
| A | Opus 5.5 for every step (the standard assistant) |
| B | Busy Machines with Opus: Opus 5.5 for every step, plus our engine: the lean planner prompt, backfill (a no-AI sampler fills idle slots while Opus thinks, fed only checker-approved results), a planner cadence (Opus called at most every 60 s in R78, 90 s in R75, 30 s in the D49 follow-up R82), and the checker (D46, D47). Each add-on is disclosed in the race's fairness manifest |
| C | Mubit + Busy Machines: the same engine; Mubit's small model answers a step only where it passed its exam, the provider answers the rest. Raced only once an adapter serves (none has yet) |
| Baseline | Optuna / random search (no LLM) |

**Tasks (the work AI labs run thousands of times):** RL (Pendulum, JAX PPO, on CPU), vision speedrun (CIFAR-10, airbench-style, on L4), tabular model tuning (LCBench surrogate, simulated GPU time). The LM pretraining speedrun (char-level nanoGPT) is demoted: no setting beat its starting code [R23, D32].
**Metrics (D48 order):** best held-out score at a fixed budget · total $ to reach a quality target · time to reach it · GPUs waiting (%) · experiments per GPU-hour. At least 3 seeds per arm for a headline number, and every number traces back to a Mubit request ID and a runner ID (a Modal call ID, or the surrogate's ID for LCBench).

## Results so far (measured)
As of Sat 24 Oct, evening (results log up to R82). Every number comes from [`docs/results_log.md`](docs/results_log.md) (ids in brackets), and the worse result on the same comparison is listed beside it. "3 runs" means 3 seeds, mean ± std. Order (D48): model quality and money first; GPUs waiting is the mechanism. **LLM $ is a list-price token estimate** (it excludes any Mubit fee, R72). Numbers for Optuna from races that include a Busy Machines arm (R65, R67, R75, R78, R82) carry the **R74 caveat**: Optuna shared one Python process with Busy Machines' sampler, so its GPUs-waiting and experiment counts there are not isolated (Opus vs Busy Machines is essentially unaffected).

**The headline comparison is the standard Opus 5.5 agent vs Busy Machines with Opus (arm B, D46).** Three 3-run races are logged, two of them headline-eligible with **the same model for a quarter of the AI bill**: the game-playing race (R78: a tie, 4.0× less AI money, 3.1× less in total) and the pre-registered tabular follow-up with the planner every ≥30 s (R82, D49, [`docs/preregistration_lcbench_wide_c30.md`](docs/preregistration_lcbench_wide_c30.md): **success line met**, within 0.0013 but the standard agent ahead in every run, 4.0× less AI money, 35% less in total with virtual GPU $). R82's setting was **chosen after** the first pre-registered tabular race (R75) **failed its success line** (8.2× less AI money, but a 7% worse model), so the two are always reported together.

**Not measured: Mubit + Busy Machines (arm C).** No Mubit adapter has answered a call. The cards show propose 14.3% vs 95%, judge v0 42.9% vs 95%, toy 70% vs 85%, and the closed-output judge v1 **11.4% on 35 test cases vs 95% after ~7 h of training** (workspace C's copy, same data: 58.3% on 24 cases, R79). The diagnose model, retried by Mubit, scored 8.3% on 12 cases vs 95% (R80), and no task has a route [R7, R28, R30, R46, R70, R79, R80]. So every row below uses Opus 5.5 as the LLM, or no LLM at all, and none is a "Mubit beats Opus" result.

| Comparison | Result | Worse side / caveats |
|---|---|---|
| **Busy Machines with Opus (lean planner + backfill + checker + planner every ≥60 s) vs the standard Opus agent, game-playing AI (Pendulum, 30 CPU workers per arm, 15 s runs, 15 min), 3 runs, headline-eligible [R78]** | **Same model, a tie within noise:** best held-out **−154.3 ± 0.5 vs −154.5 ± 0.6** (same at Opus's $3.65 spend). **LLM $ per race $0.83 ± 0.23 vs $3.30 ± 0.05: 4.0× less (75% less)**; total $ **$1.18 vs $3.65 (3.1× less)**. Same quality (reach Opus's final best): $0.46 vs $3.10, in 2 of 3 seeds. GPUs waiting **1.5% vs 68.8%**; experiments 1,587 vs 500, **10× more per $** (1,376 vs 137). Fakes accepted as best: **0 of 847 trap runs** (Opus 0 of 10). Goal −170: $0.022 / 0.3 min vs $0.20 / 1.0 min. 0 malformed replies, retries or fallbacks on both AI arms | **Not a better model** (per seed +0.14, −0.19, +0.51). Same quality reached in **2 of 3 seeds only** (seed 1 missed by 0.19). **Seed 2 judge burst:** the sampler found the trap (759 trap runs); the checker rejected 128 claims and shipped none, but 143 judge calls cost $0.56, taking that seed to 0.34× (per seed 0.21×, 0.21×, 0.34×). ~1,466 of BM's experiments came from the sampler (120 from Opus), and the shipped model came from the sampler in all 3 runs (R84). Total $ counts in-race GPU: Modal metered ~$2.5 per race vs $1.05 in-race GPU (pre-warm/idle containers, same for all arms). **Optuna: cheapest to the easy goal** ($0.009 / 0.4 min), total $0.35, 4,511 experiments per $, but best −157.3 ± 1.4 and **53 of 4,648 trap runs accepted as best** (R74 caveat). Planner interval chosen by an offline gate before the race (projected 0.22× LLM, 3.6% waiting; measured 0.25×, 1.5%) |
| **Busy Machines with Opus (lean planner + backfill + checker + planner every ≥90 s) vs the standard Opus agent, pre-registered LCBench wide race, 3 runs [R75, D47]: FAILED its success line** | **LLM $ per race $0.347 vs $2.831 (8.2× less, 87.7% less)**; total $3.54 vs $6.03 (−41%, GPU $ virtual). GPUs waiting 2.5% vs 29.4%. Both reached 0.87 in 3/3 seeds. Fakes shipped 0 vs 0 | **Worse model in all 3 seeds: 0.8816 ± 0.0038 vs 0.8892 ± 0.0017 (7% more errors).** Never reached Opus's final best (0/3); to 0.86 it was slower and dearer ($1.57 / 6.5 min vs $1.07 / 2.8 min). Success line: Opus waiting ≥40% ❌ (29.4%), BM waiting ≤2% ❌ (2.5%), BM LLM $ ≤0.8× ✅, BM best ≥ Opus ❌, BM ≥ Optuna ✅. The 90 s cadence kept only 80 AI proposals; the shipped model came from the AI planner 3/3. GPU time simulated (surrogate at 0.1× its runtimes). Optuna 0.8493, 19 fakes of 618 trap runs (R74 caveat). Logged as a fail, no re-runs |
| **Pre-registered follow-up: same as R75 with the planner every ≥30 s, 3 runs, headline-eligible [R82, D49]: success line MET (4 of 4)** | **Practically the same model:** best **0.8884 ± 0.0017 vs 0.8897 ± 0.0013** (same at Opus's spend). **LLM $ per race $0.698 vs $2.774: 0.25×, 4.0× less (74.8% less)** (per seed 0.22×, 0.28×, 0.26×); planner calls alone $0.552 vs $2.632. Total $3.89 vs $5.97 (−35%, GPU $ virtual). GPUs waiting 1.1% vs 30.3%; experiments 886 (240 from the AI) vs 422. Both reached 0.87 in 3/3 seeds. Fakes shipped **0 (of 85 trap runs; 16 claims rejected)**; shipped model from the AI planner 3/3. Success line: BM best ≥ Opus − 0.003 ✅ (0.8884 ≥ 0.8867), BM LLM $ ≤ 0.4× ✅, BM ≥ Optuna ✅, 0 BM fakes shipped ✅. Projection was BM LLM $0.85; measured $0.70 | **Not a better model: Opus ahead in every seed** (BM − Opus −0.0001, −0.0008, −0.0030; about 1% more errors). The pass is on 3-seed means; seed 2 alone sits at the margin (−0.0030). Never reached Opus's final best (0/3); to 0.86 it was slower and dearer ($1.14 / 4.2 min vs $1.08 / 2.9 min). **Chosen after R75 failed**, pre-registered before the race, the last cadence setting tried; always reported with R75. GPU time simulated. Optuna: lowest total ($3.20) but 0.8576 ± 0.018, accepted 15 of 236 trap runs as best and shipped a fake in seed 2 (R74 caveat) |
| Busy Machines with Opus (lean planner + backfill + checker, **no cadence**) vs standard Opus, game-playing AI (30 CPU workers per arm, 15 s runs, 15 min), **2 runs** [R67] | Best model **−153.9 ± 0.6 vs −156.2 ± 0.4**; goal reached for **$0.11 vs $0.20**, in **0.5 vs 1.1 min**; GPUs waiting 2.0% vs 66.0%; 1,378 vs 457 experiments | **LLM $ per race $6.50 vs $3.07 (2.1× more)**: without a cadence the faster planner calls Opus ~2.7× as often (the cadence in R78 is the fix). Stopped before seed 2, so not headline-eligible. Optuna: best −159.7, shipped −216.1 ± 79 (it shipped fakes); its 6.4% waiting here carries the R74 caveat |
| Busy Machines with Opus (backfill + checker, v1 prompt) vs standard Opus, LCBench, **2 runs** [R65] | Goal reached in **1.2 vs 2.5 min**, for **$0.36 vs $0.74**; GPUs waiting 0.0% vs 4.0%; 1.49× the experiments | Best model slightly **lower**: 0.8590 ± 0.0069 vs 0.8612 ± 0.0050 (within the 0.01 noise allowance); LLM $ 1.19× higher. Not headline-eligible (n=2) |
| Lean planner prompt (v2, effort low) vs the v1 prompt, both on Opus, LCBench, 3 runs, dev race [R63; probe R62] | Best model 0.866 ± 0.006 vs 0.849 ± 0.010; LLM $ per experiment −10.5%; output tokens per proposal 137 → 68. Probe: −71% to −73% output tokens and −46% to −48% $ per proposal (R62) | Dev race (arms use different prompts). Pooled over 9 earlier v1 runs the standard agent scores 0.854 ± 0.008, so "at least as good", not a proven gain |
| Arm B, **earlier engine version** (judge policy + median stopping, since removed, D43) vs plain Opus, LCBench, 3 runs [R44] | Best model 0.843 ± 0.009 vs 0.856 ± 0.005: **Busy Machines with Opus had 9% MORE errors**, was 5% more expensive and 9% slower to the goal | Its only edge: 1.1× the experiments per $. The equal $ cap stopped arm B 27–86 s early in every seed (it spent more on LLM). The judge policy was degenerate on LCBench (σ = 0). The mechanism first suspected could not have acted (R51); the cause is unclear |
| *Second comparison: the standard Opus agent vs Optuna, a standard no-AI tuning tool (the required no-LLM baseline, D46)* | | |
| Opus 5.5 research agent vs Optuna (no AI), LCBench tuning task, 3 runs, simulated GPU time [R38; repeated with judge v1 in R44] | Best model 0.856 ± 0.010 vs 0.799 ± 0.022 held-out balanced accuracy: **28% fewer errors** (R44: 0.856 ± 0.005 vs 0.800 ± 0.021, 28% again). Optuna never reached Opus's final quality (0/3). Fakes accepted as a new best: **Opus 0/33, Optuna 8/62** | Opus ran **0.3× the experiments per $** (22.7 vs 80.9); GPUs waited 5.0% vs 0.3% (R44: 4.6% vs 0.1%); in seed 2 **Optuna reached the goal faster and cheaper** (108 s / $0.38 vs 193 s / $0.96). GPU $ is simulated; Opus may know LCBench from training. Cost and time to the goal are **not measured** as a 3-run comparison (Optuna missed the goal in seed 0). In seed 0, Optuna's asks briefly froze the shared event loop, slightly against the Opus arm (R35, since fixed). R38 and R44 use the same seeds on a deterministic surrogate, so they are not 6 independent runs |
| Opus 5.5 vs Optuna, game-playing AI (Pendulum, 30 CPU workers per arm, 15 s runs, 15 min), 3 runs [R48] | **Shipped model −156.9 vs −177.1 ± 8.8** (Optuna's misses the −170 goal on held-out data, because ~97% of its runs used the trap setting). Fakes accepted as a new best: **Opus 0 of 15 trap runs, Optuna 36 of 4,661** | **Optuna wins every search number:** best clean model −155.2 ± 0.7 vs −156.9 ± 2.9 (a 1% lower penalty, within noise), goal reached for $0.007 / 0.3 min vs $0.222 / 1.1 min (the goal is too easy at 30 workers), 1,605 vs 504 experiments, 4,585 vs 137 experiments per $, total spend $0.35 vs $3.69. **Opus left GPUs waiting 69.0 ± 0.4%** of the time vs 0.9 ± 0.1% (R41: 70.0%, 1 run). Supersedes R36 |
| Opus 5.5 vs Optuna, image recognition (CIFAR-10, 3 L4s per arm, 15 min), 3 runs [R53, seed 0 = R49] | Best model 0.9448 ± 0.0014 vs 0.9294 ± 0.0022 held-out accuracy: **22% fewer errors**. Reached the 0.94 goal: **Opus 3/3** ($0.229 ± 0.077, 2.0 ± 0.6 min), **Optuna 0/3**. Shipped model 0.9431 vs 0.8955 ± 0.016. Fakes accepted as best: Opus 0 (it ran 0 trap runs), Optuna 15 of 118 trap runs | Opus costs **$1.65 vs $0.60** per race and runs **0.5× the experiments per $** (38.7 vs 81.2); GPUs waited 10.9 ± 1.6% vs 0.0%. Opus's checker was not tested here (no trap runs). LLM $ is an estimate at list price; in seed 2, 8.1% of Opus replies were malformed (retried) |
| Busy Machines judge (Opus checks for leakage, a seed-spread rule checks for noise) vs Opus alone, 75 stored claims with 10-seed ground truth [R37, R34] | Real gains confirmed **7/7 vs 1/7**; fakes let through **0/32 for both** | Noise accepted 1/36 vs Opus's 0/36. Only 7 real gains, offline (not yet live in a race), and the rule and the ground truth both rest on seed spread. The rule alone lets 24/32 fakes through. With the closed-output judge v1 now in use, test split only: 4/4 real, 1/20 noise, 0/16 fakes, and Opus v1 alone confirms 0/4 [R40] |
| Research agent's best settings vs the starting code [R23; R53 for the race] | Image recognition (CIFAR-10, 3 runs each): 0.9277 → 0.9371, **13% fewer errors**; in the 3-run race, **22% fewer** (0.9288 → 0.9448). Game-playing AI (Pendulum; start 10 runs, best 3 runs): −966 → −164.4, **5.9× smaller penalty**. Tabular: 74% fewer errors [R38, R44] | Language model: **no gain** (2.5222 → 2.5219). An earlier 18% CIFAR figure was one lucky run (R22, superseded). Optuna: 64% fewer errors on tabular, 1% on image recognition |
| Opus planning time vs batch size [R49, R48, R50] | Typical call: 3.15 s for 1 proposal (R49), 13.1 s for 8 (R48, 192 of 192 calls); fit ~1.62 s + 1.62 s per proposal (R50) | A smaller model in the same loop (Haiku 5.5, **not a Mubit adapter**, 1 run) cut GPUs waiting from 71.0% to 56.9% for $0.16 vs $3.27 of LLM spend, but was only ~28% faster per call and reached the goal later (86 s vs 44 s) [R50]. Planner-speed and concurrency gains beyond that are **projections** (R43, R52) |
| `sapling-research` v2 (verification head fine-tuned, backbone frozen) vs stock Banyan Sapling, 500 held-out synthetic research claims [R42, supersedes R33] | Raw accuracy 22.0% → 66.0%; fakes caught **0 → 52** (α 0.01) / 118 (α 0.05) of 157; fakes approved 0/157 in every column; ContractNLI 77.3% → 84.3% (no regression) | Opus 5.5 is a far better judge on the trap set (50/50 vs 18/50 at α 0.01; 36/50 at α 0.05 with 1 fake through). At α 0.05 the answers are less often right (75.5% vs v1's 88.9%), coverage on new templates breaks (77.8% vs 95%), and 43 genuine gains are rejected. Mubit's hosted endpoint takes ~1.1 s per call vs 75 ms on our H100 (R42); the released package measured 99.4 ms median per claim on an H100 (R76) |

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

**Benchmark** (`night/bench`):
```bash
uv run python scripts/bench.py list            # presets with their GPU $ upper bound
uv run python scripts/bench.py check lm-30m    # pre-flight: frozen prompts, frozen target_value, $ guard
```
`bench.py run` refuses until the prompts and thresholds are frozen.

Env vars: `MUBIT_API_KEY` (real LLM calls), `BUSY_MACHINES_DB` (SQLite path, default `data/busy_machines.db`), `BUSY_MODAL_CPU_CORES` (cores per CPU container, default 1), `BUSY_MODAL_SCALEDOWN_S` (warm time per container, default 60 s, billed).
