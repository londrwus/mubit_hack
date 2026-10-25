# Busy Machines: full write-up

For ML and RL teams at AI labs who run training sweeps every day: Busy Machines plans the sweep with Opus 5.5 less often, keeps every machine working, and blocks hallucinated wins.

**A typical user:** an RL post-training engineer who launches a hyperparameter sweep on a shared compute pool every day; the recurring steps Busy Machines handles are planning the next experiments, checking each claimed improvement, and keeping every machine busy.

**What the demo shows** (one user flow, ≤ 2-minute video)
- The landing page (`/`), then **Get started** (`/app`): add your Mubit key, connect your compute, start a sweep.
- **Compare** (`/app/compare`): a replay of logged race R78, standard Opus 5.5 agent vs Busy Machines, ending on the logged numbers: AI bill $3.30 vs $0.70, model quality −155.0 vs −154.9 (run 1, R90).
- **Mubit console:** a correction rated Bad on its exact request, then the task's model version history: recorded examples become training data, and a new version trains after 25 new examples (R58).
- Our open model on Hugging Face.

**Results** ([`docs/results_log.md`](docs/results_log.md))
- Same model quality for a quarter of the AI bill (4.0× less) on two tasks, 3 runs each: game-playing sweeps (R78) and tabular tuning, pre-registered, matched within its margin (R82).
- Built for sweeps of many short runs (hyperparameter and RL sweeps). Every race we ran, on every task, is in the results log.
- Our open model catches 57 of 109 hallucinated wins in real race claims and approves none; Mubit's published Sapling caught 0 (R81).

**How Mubit is used**
- Every LLM call goes to **`https://api.mubit.ai/anthropic`** with `MUBIT_API_KEY`: set in [`backend/agent/llm_base.py`](backend/agent/llm_base.py) line 29; [`backend/agent/mubit_client.py`](backend/agent/mubit_client.py) rejects any other host (lines 354–357), reads the key from `MUBIT_API_KEY`, never a provider key (lines 120–136) and builds the only SDK client (lines 379–386).
- Calls are recorded into Mubit tasks, and Mubit trains small models on them (best so far: 58.3% match with Opus, R79); a small model takes over a step once it passes Mubit's exam ([details](docs/qa_mubit_constraints.md)).
- Our fine-tune of Mubit's Banyan Sapling: https://huggingface.co/busy-machines/sapling-research

**Run it** (PowerShell, repo root; a replay makes no LLM or Modal call)
```powershell
$env:PYTHONUTF8="1"; $env:BUSY_MACHINES_REPLAY_SOURCES="data/bench"
python -m uv run --extra ml --with onnxruntime==1.29.0 uvicorn backend.app:app --port 8765
cd frontend; npm install; npm run dev    # http://localhost:5173
```
`data/bench/` keeps the race exports (`.jsonl`, ~970 MB) as evidence and replay source. More: [`frontend/README.md`](frontend/README.md).

---

Built at **Night Shift London**, the Mubit hackathon (23–25 Oct 2026). The sections below are the full write-up.

## The problem
Labs now let an AI plan their training sweeps: it proposes settings, launches training, reads results and iterates. But **every time the planner stops to think, the machines sit idle**: with 30 cloud CPU workers and 15-second runs, a standard Opus 5.5 sweep left them waiting 69% of the time [R48]. Frontier-model reasoning is slow and expensive: on that task 90.6% of the sweep's spend was the planner's own thinking, not compute [R54]. So the planner is both the bottleneck on sweep throughput and most of the bill, and nothing in the sweep decides when it is worth calling.

## What it is
**Compute intelligence for training sweeps.** Busy Machines sits under the AI planner of a sweep (Opus 5.5 in our races, or a lab's own research agent; the planner is not the product) and runs the sweep on a compute pool, for **any ML/DL/RL model** whose training can be scripted:

- **When to call the planner:** a pace limit and a shorter planner prompt, so the sweep makes the AI calls it needs and no more (AI bill 0.25× on R78 and R82).
- **Every machine busy:** a no-AI sampler fills idle machines between planner calls, fed only checked results (machines waiting 1.5% and 1.1%, R78, R82).
- **Fake improvements blocked:** a checker rejects claimed bests that come from test-set tuning or noise (0 of 847 and 0 of 85 trap runs accepted, R78, R82).

Every LLM call goes through **[Mubit](https://mubit.ai)**, the cheap-AI layer next to ours. Our measured savings are the Busy Machines layer alone; Mubit's small models would make each remaining call cheaper on top:

- The planner's **recurring steps** (propose the next experiment, diagnose a failed run, judge a claimed improvement; summarising a result is plain code) are recorded by Mubit so it can distil them into **small task-specific models**, which are meant to answer faster and cheaper than the frontier model. They take over a step once they pass Mubit's exam; our races so far ran with Opus answering through Mubit [R7, R28, R30, R70].
- A **checker** decides whether a claimed new best is real: in races, the LLM judge looks for **eval contamination** (test/val overlap with training data) and a seed-spread rule looks for noise (D40). Offline, our fine-tuned **[Banyan Sapling](https://huggingface.co/MubitLabs/banyan-sapling)** verifier catches some of the fakes the stock model leaves unresolved [R42, R81]. Sapling runs as an offline verifier (D28). Our fine-tune (v2) and its data are **published** at https://huggingface.co/busy-machines/sapling-research and https://huggingface.co/datasets/busy-machines/ml-claims (public since R92; a community fine-tune, not an official MubitLabs release).
- **Every human or verifier correction is stored against its Mubit request ID**, ready to become training data for the next version of the small model (stored locally with the request ID; Q2, Q9).

## The benchmark: `night/bench`
Two sweeps race on **identical machines (GPUs or CPUs) with identical budgets**:

| Arm | LLM in the loop |
|---|---|
| A | Opus 5.5 for every step (the standard assistant) |
| B | Busy Machines with Opus: Opus 5.5 for every step, plus our engine: the lean planner prompt, backfill (a no-AI sampler fills idle slots while Opus thinks, fed only checker-approved results), a planner cadence (Opus called at most every 60 s in R78, 90 s in R75, 30 s in the D49 follow-up R82), and the checker (D46, D47). Each add-on is disclosed in the race's fairness manifest |
| C | Mubit + Busy Machines: the same engine; Mubit's small model answers a step only where it passed its exam, the provider answers the rest. Raced once an adapter passes Mubit's exam |
| Baseline | Optuna / random search (no LLM) |

**Tasks (the work AI labs run thousands of times):** RL (Pendulum, JAX PPO, on CPU), vision speedrun (CIFAR-10, airbench-style, on L4), tabular model tuning (LCBench surrogate, simulated GPU time). The LM pretraining speedrun (char-level nanoGPT) is demoted: no setting beat its starting code [R23, D32].
**Metrics (D48 order):** best held-out score at a fixed budget · total $ to reach a quality target · time to reach it · machines waiting (%) · experiments per dollar. At least 3 seeds per arm for a headline number, and every number traces back to a Mubit request ID and a runner ID (a Modal call ID, or the surrogate's ID for LCBench).

## Results so far (measured)
As of Sun 25 Oct, ~00:50 (results log up to R92; our code freeze Sun 25 Oct 06:00 London, D53; submissions close 08:00 and `main` is judged as it is then, D55). Every number comes from [`docs/results_log.md`](docs/results_log.md) (ids in brackets), and the worse result on the same comparison is listed beside it. "3 runs" means 3 seeds, mean ± std. Order (D48): model quality and money first; machines waiting is the mechanism. **LLM $ is a list-price token estimate** (it excludes any Mubit fee, R72). Numbers for Optuna from races that include a Busy Machines arm (R65, R67, R75, R78, R82, R85) carry the **R74 caveat**: Optuna shared one Python process with Busy Machines' sampler, so its machines-waiting and experiment counts there are not isolated (Opus vs Busy Machines is essentially unaffected). From R88 on (patch 09), future races run each sampler in its own process. That cuts the contention from 2.52× / 2.49× to 1.40× / 1.46× slower (R88) but does not remove it, and the logged races keep the caveat.

**The headline comparison is the standard Opus 5.5 agent vs Busy Machines with Opus (arm B, D46).** Three 3-run races are logged, two of them headline-eligible with **the same model for a quarter of the AI bill**: the game-playing race (R78: a tie, 4.0× less AI money, 3.1× less in total) and the pre-registered tabular follow-up with the planner every ≥30 s (R82, D49, [`docs/preregistration_lcbench_wide_c30.md`](docs/preregistration_lcbench_wide_c30.md): **success line met**, within 0.0013 but the standard agent ahead in every run, 4.0× less AI money, 35% less in total with virtual GPU $). R82's setting was **chosen after** the first pre-registered tabular race (R75) **failed its success line** (8.2× less AI money, but a 7% worse model), so the two are always reported together.

**Not measured: Mubit + Busy Machines (arm C).** No Mubit adapter has answered a call. The cards show propose 14.3% vs 95%, judge v0 42.9% vs 95%, toy 70% vs 85%, and the closed-output judge v1 **11.4% on 35 test cases vs 95% after ~7 h of training** (workspace C's copy, same data: 58.3% on 24 cases, R79). The diagnose model, retried by Mubit, scored 8.3% on 12 cases vs 95% (R80), and no task has a route [R7, R28, R30, R46, R70, R79, R80]. So every row below uses Opus 5.5 as the LLM, or no LLM at all, and none is a "Mubit beats Opus" result.

| Comparison | Result | Worse side / caveats |
|---|---|---|
| **Busy Machines with Opus (lean planner + backfill + checker + planner every ≥60 s) vs the standard Opus agent, game-playing AI (Pendulum, 30 CPU workers per arm, 15 s runs, 15 min), 3 runs, headline-eligible [R78]** | **Same model, a tie within noise:** best held-out **−154.3 ± 0.5 vs −154.5 ± 0.6** (same at Opus's $3.65 spend). **LLM $ per race $0.83 ± 0.23 vs $3.30 ± 0.05: 4.0× less (75% less)**; total $ **$1.18 vs $3.65 (3.1× less)**. Same quality (reach Opus's final best): $0.46 vs $3.10, in 2 of 3 seeds. Machines waiting **1.5% vs 68.8%**; experiments 1,587 vs 500, **10× more per $** (1,376 vs 137). Fakes accepted as best: **0 of 847 trap runs** (Opus 0 of 10). Goal −170: $0.022 / 0.3 min vs $0.20 / 1.0 min, i.e. 89% cheaper and 72% faster [R87]. 0 malformed replies, retries or fallbacks on both AI arms | **Not a better model** (per seed +0.14, −0.19, +0.51). Same quality reached in **2 of 3 seeds only** (seed 1 missed by 0.19). **Seed 2 judge burst:** the sampler found the trap (759 trap runs); the checker rejected 128 claims and shipped none, but 143 judge calls cost $0.56, taking that seed to 0.34× (per seed 0.21×, 0.21×, 0.34×). ~1,466 of BM's experiments came from the sampler (120 from Opus), and the shipped model came from the sampler in all 3 runs (R84). Total $ counts in-race GPU: Modal metered ~$2.5 per race vs $1.05 in-race GPU (pre-warm/idle containers, same for all arms). **Optuna: cheapest to the easy goal** ($0.009 / 0.4 min, so R87's goal framing is quoted only with this caveat), total $0.35, 4,511 experiments per $, but best −157.3 ± 1.4, **53 of 4,648 trap runs accepted as best**, and its shipped model was a fake in 3 of 3 runs (R84; its −157.7 shipped score is still a real held-out number; R74 caveat). Planner interval chosen by an offline gate before the race (projected 0.22× LLM, 3.6% waiting; measured 0.25×, 1.5%) |
| **Busy Machines with Opus (lean planner + backfill + checker + planner every ≥90 s) vs the standard Opus agent, pre-registered LCBench wide race, 3 runs [R75, D47]: FAILED its success line** | **LLM $ per race $0.347 vs $2.831 (8.2× less, 87.7% less)**; total $3.54 vs $6.03 (−41%, GPU $ virtual). Machines waiting 2.5% vs 29.4%. Both reached 0.87 in 3/3 seeds. Fakes shipped 0 vs 0 | **Worse model in all 3 seeds: 0.8816 ± 0.0038 vs 0.8892 ± 0.0017 (7% more errors).** Never reached Opus's final best (0/3); to 0.86 it was slower and dearer ($1.57 / 6.5 min vs $1.07 / 2.8 min). Success line: Opus waiting ≥40% ❌ (29.4%), BM waiting ≤2% ❌ (2.5%), BM LLM $ ≤0.8× ✅, BM best ≥ Opus ❌, BM ≥ Optuna ✅. The 90 s cadence kept only 80 AI proposals; the shipped model came from the AI planner 3/3. GPU time simulated (surrogate at 0.1× its runtimes). Optuna 0.8493, accepted 19 of 618 trap runs as best at some point and shipped a fake in 1 of 3 runs (R84; R74 caveat). Logged as a fail, no re-runs |
| **Pre-registered follow-up: same as R75 with the planner every ≥30 s, 3 runs, headline-eligible [R82, D49]: success line MET (4 of 4)** | **Practically the same model:** best **0.8884 ± 0.0017 vs 0.8897 ± 0.0013** (same at Opus's spend). **LLM $ per race $0.698 vs $2.774: 0.25×, 4.0× less (74.8% less)** (per seed 0.22×, 0.28×, 0.26×); planner calls alone $0.552 vs $2.632. Total $3.89 vs $5.97 (−35%, GPU $ virtual). Machines waiting 1.1% vs 30.3%; experiments 886 (240 from the AI) vs 422. Both reached 0.87 in 3/3 seeds. Fakes shipped **0 (of 85 trap runs; 16 claims rejected)**; shipped model from the AI planner 3/3. Success line: BM best ≥ Opus − 0.003 ✅ (0.8884 ≥ 0.8867), BM LLM $ ≤ 0.4× ✅, BM ≥ Optuna ✅, 0 BM fakes shipped ✅. Projection was BM LLM $0.85; measured $0.70 | **Not a better model: Opus ahead in every seed** (BM − Opus −0.0001, −0.0008, −0.0030; about 1% more errors). The pass is on 3-seed means; seed 2 alone sits at the margin (−0.0030). Never reached Opus's final best (0/3); to 0.86 it was slower and dearer ($1.14 / 4.2 min vs $1.08 / 2.9 min). **Chosen after R75 failed**, pre-registered before the race, the last cadence setting tried; always reported with R75. GPU time simulated. Optuna: lowest total ($3.20) but 0.8576 ± 0.018, accepted 15 of 236 trap runs as best and shipped a fake in seed 2 (R74 caveat) |
| **Busy Machines with Opus (lean planner + backfill + checker + planner every ≥60 s) vs the standard Opus agent, image recognition on real L4 GPUs (3 per side, 15 min), 3 runs, headline-eligible [R85]: quality line FAILED** | AI bill **$0.233 vs $1.056 per race (4.5× less)**, total $0.83 vs $1.66 (2× less); machines waiting 0.0% vs 12.0%; 0 of 12 fake attempts accepted | **The model got worse: 0.9424 vs 0.9451 (5% more errors)**, behind in every run; never reached the standard agent's final quality; slower and dearer to the 0.94 goal ($0.317 / 5.4 min vs $0.273 / 2.5 min); shipped model 0.9395, below the goal. On long runs with few machines the no-AI sampler adds nothing, so planning less often costs quality. Our only Busy Machines race on real GPUs. AI $ at list price. Optuna 0.9266, never reached the goal, shipped a fake in 3 of 3 (R74 caveat) |
| Busy Machines with Opus (lean planner + backfill + checker, **no cadence**) vs standard Opus, game-playing AI (30 CPU workers per arm, 15 s runs, 15 min), **2 runs** [R67] | Best model **−153.9 ± 0.6 vs −156.2 ± 0.4**; goal reached for **$0.11 vs $0.20**, in **0.5 vs 1.1 min**; machines waiting 2.0% vs 66.0%; 1,378 vs 457 experiments | **LLM $ per race $6.50 vs $3.07 (2.1× more)**: without a cadence the faster planner calls Opus ~2.7× as often (the cadence in R78 is the fix). Stopped before seed 2, so not headline-eligible. Optuna: best −159.7, shipped −216.1 ± 79 (it shipped fakes); its 6.4% waiting here carries the R74 caveat |
| Busy Machines with Opus (backfill + checker, v1 prompt) vs standard Opus, LCBench, **2 runs** [R65] | Goal reached in **1.2 vs 2.5 min**, for **$0.36 vs $0.74**; machines waiting 0.0% vs 4.0% (simulated GPU time); 1.49× the experiments | Best model slightly **lower**: 0.8590 ± 0.0069 vs 0.8612 ± 0.0050 (within the 0.01 noise allowance); LLM $ 1.19× higher. Not headline-eligible (n=2) |
| Lean planner prompt (v2, effort low) vs the v1 prompt, both on Opus, LCBench, 3 runs, dev race [R63; probe R62] | Best model 0.866 ± 0.006 vs 0.849 ± 0.010; LLM $ per experiment −10.5%; output tokens per proposal 137 → 68. Probe: −71% to −73% output tokens and −46% to −48% $ per proposal (R62) | Dev race (arms use different prompts). Pooled over 9 earlier v1 runs the standard agent scores 0.854 ± 0.008, so "at least as good", not a proven gain |
| Arm B, **earlier engine version** (judge policy + median stopping, since removed, D43) vs plain Opus, LCBench, 3 runs [R44] | Best model 0.843 ± 0.009 vs 0.856 ± 0.005: **Busy Machines with Opus had 9% MORE errors**, was 5% more expensive and 9% slower to the goal | Its only edge: 1.1× the experiments per $. The equal $ cap stopped arm B 27–86 s early in every seed (it spent more on LLM). The judge policy was degenerate on LCBench (σ = 0). The mechanism first suspected could not have acted (R51); the cause is unclear |
| *Second comparison: the standard Opus agent vs Optuna, a standard no-AI tuning tool (the required no-LLM baseline, D46)* | | |
| Opus 5.5 research agent vs Optuna (no AI), LCBench tuning task, 3 runs, simulated GPU time [R38; repeated with judge v1 in R44] | Best model 0.856 ± 0.010 vs 0.799 ± 0.022 held-out balanced accuracy: **28% fewer errors** (R44: 0.856 ± 0.005 vs 0.800 ± 0.021, 28% again). Optuna never reached Opus's final quality (0/3). Fakes accepted as a new best: **Opus 0/33, Optuna 8/62** | Opus ran **0.3× the experiments per $** (22.7 vs 80.9); machines waited 5.0% vs 0.3% (simulated GPU time) (R44: 4.6% vs 0.1%); in seed 2 **Optuna reached the goal faster and cheaper** (108 s / $0.38 vs 193 s / $0.96). GPU $ is simulated; Opus may know LCBench from training. Cost and time to the goal are **not measured** as a 3-run comparison (Optuna missed the goal in seed 0). In seed 0, Optuna's asks briefly froze the shared event loop, slightly against the Opus arm (R35, since fixed). R38 and R44 use the same seeds on a deterministic surrogate, so they are not 6 independent runs |
| Opus 5.5 vs Optuna, game-playing AI (Pendulum, 30 CPU workers per arm, 15 s runs, 15 min), 3 runs [R48] | **Shipped model −156.9 vs −177.1 ± 8.8** (Optuna's misses the −170 goal on held-out data, because ~97% of its runs used the trap setting). Fakes accepted as a new best: **Opus 0 of 15 trap runs, Optuna 36 of 4,661** | **Optuna wins every search number:** best clean model −155.2 ± 0.7 vs −156.9 ± 2.9 (a 1% lower penalty, within noise), goal reached for $0.007 / 0.3 min vs $0.222 / 1.1 min (the goal is too easy at 30 workers), 1,605 vs 504 experiments, 4,585 vs 137 experiments per $, total spend $0.35 vs $3.69. **Opus left the CPU workers waiting 69.0 ± 0.4%** of the time vs 0.9 ± 0.1% (R41: 70.0%, 1 run). Supersedes R36 |
| Opus 5.5 vs Optuna, image recognition (CIFAR-10, 3 L4s per arm, 15 min), 3 runs [R53, seed 0 = R49] | Best model 0.9448 ± 0.0014 vs 0.9294 ± 0.0022 held-out accuracy: **22% fewer errors**. Reached the 0.94 goal: **Opus 3/3** ($0.229 ± 0.077, 2.0 ± 0.6 min), **Optuna 0/3**. Shipped model 0.9431 vs 0.8955 ± 0.016. Fakes accepted as best: Opus 0 (it ran 0 trap runs), Optuna 15 of 118 trap runs | Opus costs **$1.65 vs $0.60** per race and runs **0.5× the experiments per $** (38.7 vs 81.2); GPUs waited 10.9 ± 1.6% vs 0.0%. Opus's checker was not tested here (no trap runs). LLM $ is an estimate at list price; in seed 2, 8.1% of Opus replies were malformed (retried) |
| Busy Machines judge (Opus checks for leakage, a seed-spread rule checks for noise) vs Opus alone, 75 stored claims with 10-seed ground truth [R37, R34] | Real gains confirmed **7/7 vs 1/7**; fakes let through **0/32 for both** | Noise accepted 1/36 vs Opus's 0/36. Only 7 real gains, offline (not yet live in a race), and the rule and the ground truth both rest on seed spread. The rule alone lets 24/32 fakes through. With the closed-output judge v1 now in use, test split only: 4/4 real, 1/20 noise, 0/16 fakes, and Opus v1 alone confirms 0/4 [R40] |
| Research agent's best settings vs the starting code [R23; R53 for the race] | Image recognition (CIFAR-10, 3 runs each): 0.9277 → 0.9371, **13% fewer errors**; in the 3-run race, **22% fewer** (0.9288 → 0.9448). Game-playing AI (Pendulum; start 10 runs, best 3 runs): −966 → −164.4, **5.9× smaller penalty**. Tabular: 74% fewer errors [R38, R44] | Language model: **no gain** (2.5222 → 2.5219). An earlier 18% CIFAR figure was one lucky run (R22, superseded). Optuna: 64% fewer errors on tabular, 1% on image recognition |
| Opus planning time vs batch size [R49, R48, R50] | Typical call: 3.15 s for 1 proposal (R49), 13.1 s for 8 (R48, 192 of 192 calls); fit ~1.62 s + 1.62 s per proposal (R50) | A smaller model in the same loop (Haiku 5.5, **not a Mubit adapter**, 1 run) cut machines waiting from 71.0% to 56.9% for $0.16 vs $3.27 of LLM spend, but was only ~28% faster per call and reached the goal later (86 s vs 44 s) [R50]. Planner-speed and concurrency gains beyond that are **projections** (R43, R52) |
| `sapling-research` v2 (verification head fine-tuned, backbone frozen) vs stock Banyan Sapling, 500 held-out synthetic research claims [R42, supersedes R33] | Raw accuracy 22.0% → 66.0%; fakes caught **0 → 52** (α 0.01) / 118 (α 0.05) of 157; fakes approved 0/157 in every column; ContractNLI 77.3% → 84.3% on our own 400-row layout (no regression; not a reproduction of the card's 86.32% on its 731 cases, R81). **On 152 real race claims with known answers [R81]:** fakes caught published 0 / ours 8 (α 0.01) / 57 (α 0.05) / Opus 109, of 109; fakes approved 0 for all. Published at https://huggingface.co/busy-machines/sapling-research (R92) | Opus 5.5 is a far better judge (and on real claims: 109 vs 57; real gains confirmed 0 of 7 for all four, R81). Later versions v3 and v4 failed their pre-declared gates, so v2 stays the release (R77, R83) on the trap set (50/50 vs 18/50 at α 0.01; 36/50 at α 0.05 with 1 fake through). At α 0.05 the answers are less often right (75.5% vs v1's 88.9%), coverage on new templates breaks (77.8% vs 95%), and 43 genuine gains are rejected. Mubit's hosted endpoint takes ~1.1 s per call vs 75 ms on our H100 (R42); the released package measured 99.4 ms median per claim on an H100 (R76) |
| *Checks, not results: backend and demo tooling* | | |
| Backend end to end and freeze check [R86, R88, R91] | API + live events + replay smoke test passed (R86). Review patches 06–09 applied, dashboard API for live races and replays, full suite 817 passed, 3 skipped, 0 failed; secrets / provider-URL / `import modal` scans clean. The dashboard's end-of-race values equal the stored summaries for R78 and R82 (R88; R78 run 1 values on the design frames: R90). After live Modal metrics and a timer-flake fix: **full suite 840 passed, 3 skipped, 0 failed**; the frontend builds, and its Compare replay of R78 run 1 ends exactly on the R90 values (R91) | `GET /api/races/{id}` returned null fields in the smoke test, still to check (R86). Patch 09 reduces the Optuna-side contention (2.52× / 2.49× → 1.40× / 1.46× slower) but doesn't remove it. A known timing-flaky test passed in the final run |
| 60-second live demo, one test run [R89] | **The demo tool works; this is not a result.** Machines waiting 3.5% vs 82.1%, experiments 445 vs 25 | **The standard agent was slightly ahead on quality (0.8090 vs 0.8064) and AI bill ($0.133 vs $0.142) at 60 s**, and it reached the goal sooner and cheaper. 60 s is too short for the pace limit and the sampler to pay off, so our demo uses the R78 replay instead |

**Positioning (D54):** Busy Machines is compute intelligence for training sweeps, and it **runs on Mubit**. The 4× above comes from the Busy Machines layer alone, with Opus answering through Mubit. It is not "powered by Mubit's models", and Mubit didn't cut the bill: no Mubit model has served.

**Where these numbers appear in the designs** (Pen, `mubit.pen`; exports and number check in `design/exports/final/README.md`):
- **Landing page `pOKAY`:** hero `KekD1`, engine `QYavo` (R78), our model `MgmgS` (R81), benchmarks `oT6Ad` (R78, R82, R87), call to action `c50WCZ`.
- **Platform:** Overview `bxWh3` (R78, R82, R90) and Sweep detail (live compare) `eGVso` (R78 run 1, R90).
- **Before they are shown:** `KekD1` must drop "GPUs" alone (D54; the built hero still says "Your GPUs should never wait"), `MgmgS`'s "99.9% vs 88.7%" field note is R54's one-run figure (the no-AI tool's shipped image model self-reported 0.9994, held-out 0.8874), so it is shown only labelled that way, and `oT6Ad` needs R75, R85 and R87's caveat beside it.
- **Frontend (built, R91):** `frontend/` (Vite + React + Tailwind): `/` landing, `/app` Get started, `/app/compare` live replay of logged races. See [`frontend/README.md`](frontend/README.md).

## Stack
FastAPI · SQLite · Vite + React · shadcn/ui · Tailwind CSS · Modal · Mubit · Pen.dev

## Repo
- [`docs/`](docs/README.md): research, plans, architecture and decisions
- [`CLAUDE.md`](CLAUDE.md): guidance for AI coding agents
- `backend/`: FastAPI app, race engine, SQLite, agent loop, Mubit client, benchmark package
- `workers/`: targets (`lm`, `cifar` on L4, `rl` on CPU) and runners (`local`, `modal`)
- `scripts/`: dev server, mock race, latency probe, benchmark pre-flight
- `frontend/`: Vite + React dashboard (landing, Get started, live compare replay; R91)

Busy Machines is **compute-agnostic**: training runs go through a pluggable runner (local machine, Modal, and in production a lab's own cluster). Our races used Modal: cloud CPU workers for game-playing (R78), L4 GPUs for image recognition (R85); tabular tuning used simulated GPU time (R82).

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
