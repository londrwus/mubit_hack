# Opportunity benchmarks: where Busy Machines should win, fairly

**Status:** desk research (24 Oct 2026). No runs and no LLM or Modal calls were made for this doc. Numbers from papers are the papers' own and are linked. Our own numbers are quoted from `results_log.md` (R#) or `optimisation_analysis.md`. **Cost and effort figures are estimates, not measurements**, and none of this can go on a slide until it's measured and logged.

The question: in which fair regimes does a loop of small distilled models (Mubit adapters) plus an independent verifier (Sapling) beat an Opus 5.5 loop on the same GPUs, and why?

Short answer: we have **two structural advantages**, and each regime below uses one or both.
1. **Speed and $ per decision.** It matters only when decisions are frequent compared with run length (short runs, many slots, cheap evaluations). R20: LLM calls were **74% of total spend**. R21: GPUs waited 8–17% for Opus.
2. **Ground truth beats teacher imitation on narrow judgements.** An adapter corrected with ground-truth labels (v2), or a fine-tuned verifier head, can learn a lab's rule ("a gain under 2σ is noise") that Opus doesn't apply on its own. R25 and `optimisation_analysis.md` §3: Opus confirmed **0 of 10** clearly real gains (up to 6.6σ) and said "unsure" on all 46 claims.

Two dependencies apply to every regime:
- **Q1/R8:** adapter latency in live routing hasn't been measured. In Try it, the toy adapter took 1.6–1.8 s, no faster than Haiku.
- **Q2:** a correction must be able to override the teacher's label. Mubit grades adapters against the provider, so v1 can at best tie Opus on accuracy (R7).

---

## Part A: what the literature says

### A1. Small fine-tuned models on narrow judge, verify and classify tasks
| Evidence | Finding | Link |
|---|---|---|
| MiniCheck (Tang, Laban, Durrett, EMNLP 2024) | A 770M fact-checker reaches GPT-4 accuracy on LLM-AggreFact at **~400× lower cost** | https://arxiv.org/abs/2404.10774 |
| JudgeLM (Zhu et al., 2023) | Fine-tuned 7B–33B judges agree with their GPT-4 teacher >90%, above human–human agreement. The 7B judges 5K samples in ~3 min on 8 A100s | https://arxiv.org/abs/2310.17631 |
| Prometheus 2 (Kim et al., EMNLP 2024) | An open 7B/8×7B evaluator has the highest agreement with humans and GPT-4 among open judges, and roughly halves the pairwise gap to GPT-4 | https://arxiv.org/abs/2405.01535 |
| LoRA Land (Predibase, 2024) | 224 of 310 LoRA-tuned 7B models beat GPT-4 on narrow tasks. GPT-4 keeps its lead on broad ones (coding, MMLU) | https://arxiv.org/abs/2405.00732 |
| Bucher & Martini (2024) | Fine-tuned small models beat zero-shot GPT-4 and Claude Opus on every text-classification task they tested | https://arxiv.org/abs/2406.08660 |
| Distilling step-by-step (Hsieh et al., ACL 2023) | A 770M T5 beats few-shot 540B PaLM on a benchmark, using 80% of the data | https://arxiv.org/abs/2305.02301 |
| Fisher-R1 / P-Bench (Miao, Mu, Chen, Zou, Aug 2026) | A 14B model trained for hypothesis testing **matches or beats GPT-5.4 on 3 of 4 strict metrics**. GPT-5.4 gets the right decision 58% of the time but the right p-value only 30.5% (hard split) | https://arxiv.org/abs/2608.07437 |
| Ferreira, Hutter, Zela et al. (2026), "Can LLMs beat classical HPO?" | A **0.8B LLM** steering CMA-ES ("Centaur") beat every classical method and every pure-LLM agent, Opus 4.6 included | https://arxiv.org/abs/2603.24647 |
| Caveat: cross-evaluation of fine-tuned judges (2024) | Fine-tuned judges win in-domain but overfit to their own evaluation scheme | https://arxiv.org/abs/2403.02839 |

**Takeaway:** "small, tuned, narrow" matching or beating frontier models is well supported, provided the task is narrow and stays in-distribution. Out of distribution it breaks, as our own R9 shows: the toy adapter failed on overfit, contamination and healthy-run cases.

### A2. Frontier agents make statistical and evaluation errors in ML experiments
| Evidence | Finding | Link |
|---|---|---|
| Luo, Kasirzadeh, Shah (NeurIPS 2025), "Hidden pitfalls of AI scientist systems" | **Post-hoc selection bias:** when test scores were inverted, AI Scientist v2 still picked the *worst* candidate 49% of the time (0% in the control) and Agent Laboratory 10% (vs 1%). Both preferred easy benchmarks, and both subsampled or synthesised datasets without saying so. A detector reading only the paper scored 55% accuracy; with logs and code, 82% | https://arxiv.org/abs/2509.08713 |
| MLR-Bench (Chen et al., NeurIPS 2025 D&B) | Coding agents produced fabricated or invalid experimental results in "e.g. 80%" of cases | https://arxiv.org/abs/2505.19955 |
| METR (Jun 2025), "Recent frontier models are reward hacking" | On RE-Bench, o3 hacked the scorer instead of doing the task, almost every time on some tasks | https://metr.org/blog/2025-06-05-recent-reward-hacking/ |
| SPOT (2025) | The best model (o3) found published-paper errors with **21% recall and 6% precision** | https://arxiv.org/abs/2505.11855 |
| Beel, Kan, Baumgart (2025) on Sakana's AI Scientist | 42% of experiments failed; some finished runs gave logically flawed or misleading results; some papers had hallucinated numbers | https://arxiv.org/abs/2502.14297 |
| "LLM hacking" (arXiv 2509.08825, 2025) | Across 2,361 hypotheses, LLM-annotated analyses reached the wrong conclusion on ~31% (state-of-the-art models) | https://arxiv.org/abs/2509.08825 |
| Xiong et al. (ICLR 2024) | LLMs are overconfident when they state their confidence. Calibration improves with scale but stays far from ideal | https://arxiv.org/abs/2306.13063 |
| Bouthillier et al. (MLSys 2021) | Seed, data-split and HPO variance can reverse published comparisons. Single-run comparisons mislead | https://arxiv.org/abs/2103.03098 |
| airbench paper (Jordan, 2024) | CIFAR-10 run-to-run std is **0.12–0.16 points** (10,000 runs). Confirming a +0.02-point gain needs ~133 runs | https://arxiv.org/abs/2404.00498 |

**Our own data points the other way on blatant traps.** Opus caught 92 of 92 trap runs in races (R25) and scored 50/50 on the trap set (R24). Opus is **not** weak at spotting obvious leakage. Its weak spots in our data are **noise against real gains**: it never confirms a real gain (§3 of `optimisation_analysis.md`), and the agent loop itself quoted seed luck as gains (R22, corrected by R23). The literature supports **selection bias and subtle, undisclosed data changes**, not obvious leaks.

### A3. Cheap, very short ML benchmarks
| Benchmark | Run length / cost | Licence | Notes | Link |
|---|---|---|---|---|
| airbench (CIFAR-10) | 94% in **2.6 s** (A100, muon), 95% in 10 s, 96% in 27 s | MIT | Our CIFAR target is airbench-style. On an L4, a 45 s run costs ~$0.01, and the first epoch (warmup) takes ~11 s (R12, R16) | https://github.com/KellerJordan/cifar10-airbench |
| gymnax + purejaxrl | CartPole/Pendulum: 1M steps in 0.05–0.07 s (A100, 2,000 envs). PureJaxRL claims ">1000× faster than PyTorch RL" | Apache-2.0 (both) | Our RL target: Pendulum PPO, ~30 s on 1 CPU core, ~$0.0005 per run (R16) | https://github.com/RobertTLange/gymnax, https://github.com/luchris429/purejaxrl |
| modded-nanogpt | Records are on 8×H100 (minutes), **too expensive for us**. nanoGPT char-level is the cheap variant (our LM target: 60 s on L4, ~$0.016) | MIT | The LM target has no replicable gain (R23), so it's a weak race target | https://github.com/KellerJordan/modded-nanogpt |
| **YAHPO Gym, LCBench scenario** | Surrogate (2.1 MB ONNX): **milliseconds per evaluation on CPU**. 34 OpenML datasets, 7 settings plus an epoch fidelity. Predicts **val and test** accuracy and **runtime** | Apache-2.0 (code and LCBench data) | `yahpo_gym` pins Python ≤3.11.9 and numpy<2. Either run it in its own 3.11 env or call `onnxruntime` directly on 3.12 | https://github.com/slds-lmu/yahpo_gym, https://github.com/slds-lmu/yahpo_data |
| YAHPO rbv2_* / iaml_* | Surrogates for SVM, XGBoost, ranger and glmnet over ~100 datasets each | Apache-2.0 | Classical ML, less "AI lab" | same |
| HPO-B | 176 meta-datasets of OpenML results, XGBoost surrogates for continuous spaces | MIT | Tabular and surrogate, CPU | https://github.com/machinelearningnuremberg/HPO-B |
| JAHS-Bench-201 | Joint architecture and HPO surrogates (CIFAR-10, Fashion-MNIST, colorectal histology) | MIT | Closer to "research choices" (D29) | https://github.com/automl/jahs_bench_201 |
| HPOBench | Tabular, surrogate and raw benchmarks | Apache-2.0 | Many need containers, so more setup | https://github.com/automl/HPOBench |
| mf-prior-bench | One API over YAHPO LCBench, JAHS and PD1 | Apache-2.0 | Python ≥3.10 | https://github.com/automl/mf-prior-bench |

### A4. How much of an agent loop's cost and time goes to the LLM
- **RE-Bench (METR, 2024):** ~29M input and 0.5M output tokens per 8-hour run, **≈ $123 of tokens**, GPUs excluded. https://arxiv.org/abs/2411.15114
- **Sakana AI Scientist:** "≈ $15 per paper" (self-reported). Beel et al. confirm under $15 in API fees and ~3.5 h per paper. https://sakana.ai/ai-scientist/
- **LLAMBO (ICLR 2024)** has a higher clock time per iteration than GP-based BO, "predominantly" from API latency. On Optuna Hub, a 30-trial run costs $0.05–0.10 with gpt-4o-mini. https://arxiv.org/abs/2402.03921, https://hub.optuna.org/samplers/llambo/
- **ASAP (Jun 2026)** calls end-to-end wall-clock the neglected metric in LLM-HPO: LLM inference runs serially after each evaluation. Its fixes are KV-cache reuse and speculative execution that hides LLM latency. https://arxiv.org/abs/2606.25207
- **Ferreira et al. 2026** gave each method 24 GPU-hours **excluding LLM inference time**. The standard methodology simply leaves out the cost we measure.
- **Ours:** LLM spend was 74% of the total (R20) and 92–94% of the $ to reach the goal on RL (`optimisation_analysis.md` §4). Opus `propose` p50 was 3.1–4.2 s in the loop and 6–9.5 s in probes (R10, R18, R21).

### A5. LLM against Optuna and BO on HPO (setting honest expectations)
- **Zhang et al. (2023):** GPT-4 Turbo beat random search on 81% of HPOBench tasks at **10 evaluations**, better than BO (50–56%). The edge **shrinks by 100 evaluations** (88% against BO's 92%). Single seed. https://arxiv.org/abs/2312.04528
- **Ferreira et al. (2026, autoresearch, 3 seeds, ~300 trials):** in a fixed search space, **TPE (0.9768) beat or tied every pure-LLM agent**, Opus 4.6 code-editing included (0.9770 ± 0.0027). LLM agents "struggle to track optimisation state": their OOM rates were 48–61%, like random search (56%), while TPE's was 11%. The hybrid won. https://arxiv.org/abs/2603.24647
- **AgentHPO (Liu et al., 2024)** reports beating random search and often matching human-best. Self-reported. https://arxiv.org/abs/2402.01881
- **"LLMs for BO in science: are we there yet?" (2025):** LLMs don't use experimental feedback well in context. https://arxiv.org/abs/2509.21403

**Expectation for our Optuna arm:** on numeric-only spaces with more than ~30 evaluations, **Optuna probably ties or wins on quality, and it is always cheaper**. LLM arms have an edge only in the first 10–30 evaluations and on research choices (D29). Say this on the slide before a judge does.

---

## Part B: ranked regimes for night/bench

Shared rules for all regimes (on top of `benchmark.md` §4):
- **Pre-register:** write the metric, thresholds, seeds and arms into `results_log.md` *before* the first run.
- **Reference columns that kill the "rigged" objection:** (a) **Opus + rule**: the same Opus with our rule or extra evidence in the *user message* (the frozen instructions stay unchanged, as in D29). (b) **Rule only**: a non-LLM baseline (Welch t-test, Optuna/TPE, median stopping). If the rule alone ties us, we say so.
- Report everything, including the regimes where we lose.

### 1. Real gain or noise? A judge with many-seed ground truth ★ best accuracy story
- **What:** a set of "claimed new best" cases, each graded by **re-running both configs over 10+ seeds** on the held-out split. Real = gain > 2σ, noise ≤ 1σ (the `optimisation_analysis.md` §3 definition, fixed now). Seed from the 46 R25 claims plus new ones from races. Split dev/test, stratified by target.
- **Why Opus is weak here:** in our data it confirmed 0 of 10 real gains (R25, §3), and our own loop reported seed luck as gains (R22 → R23). Literature: selection bias (Luo et al.), overconfidence (Xiong), wrong p-values even when the decision is right (P-Bench), single-run comparisons mislead (Bouthillier). The catch: Opus says "unsure" partly *because the evidence lacks seed spread* (§3).
- **Why we should be strong:** v2 is corrected with *ground-truth* labels, so it learns the lab's threshold. Sapling's head is fine-tuned with noise mapped to `insufficient_information` (D28). Supported by Fisher-R1 (a 14B model trained on test correctness ≥ GPT-5.4) and MiniCheck (GPT-4 level at 1/400 of the cost).
- **Fair metric:** on the test split, **real gains confirmed (%) at noise let through ≤ 5%**, plus balanced accuracy. Also $ and p50 latency per verdict.
- **Columns:** Opus · Opus + σ in the evidence and "gain > 2σ = real" in the message · Welch t-test on the same seeds (rule only) · v1 · v2 · stock Sapling · sapling-research.
- **Cost (estimate):** RL reruns ~40 configs × 10 seeds × ~$0.0005–0.001 ≈ **$0.5** CPU. CIFAR ~20 configs × 8 seeds × ~$0.01 ≈ **$1.6** L4. LLM: ~100 test cases × 7 judge columns × ≤ $0.005 ≈ $3. **Setup: 3–4 h** (`calibrate_targets.py`, `eval_trap_set.py` and `make_ml_claims.py` exist).
- **Rigging risk: high if done naively** ("you hid σ from Opus"). Pre-empt with the Opus + σ column. If Opus + σ ties v2, the honest claim becomes "same verdicts at X% of the cost and Y× faster", which is still a win, but a cost win. If the t-test ties everything, say the learned judge is worth it only because it **also** handles leakage in the same verdict, and show that on the mixed set (regime 5). **Depends on Q2.**

### 2. Wide-parallel, short-run race (RL on CPU, 32 slots, ~15 s runs) ★ biggest throughput story
- **What:** the D21 race, with run length cut to ~15 s and 32 CPU slots per arm. Arms: Opus · Mubit stock · Mubit + ours · Optuna.
- **Why Opus is weak here:** a sequential researcher makes at most ~0.9 proposals/s at Opus's latency. 32 slots of 15 s runs need ~2.1/s. Projection (`optimisation_analysis.md` §2, a model, not a measurement): **Opus leaves 33–67% of slots waiting; a 1 s adapter, 10–15%.** Literature: LLM latency is serial and usually left out of the clock (ASAP, Ferreira et al.). RE-Bench's token bill is the same order as its GPUs.
- **Why we should be strong:** cheaper and faster calls mean more experiments per $ and per minute. Throughput then decides quality at a fixed budget (Claim 1). gymnax/purejaxrl make 15 s runs realistic.
- **Fair metric:** GPUs waiting %, experiments per $, held-out best at a fixed wall-clock and $ cap, $ and time to target (−165 on RL), 3 seeds, mean ± std.
- **Cost (estimate):** ~1 core per slot at ~$0.05/h. 32 slots × 4 arms × 3 seeds × 0.5 h ≈ 190 slot-hours ≈ **$8–10** CPU, or **~$4** with 15-minute races. Opus LLM ≈ 600 calls × ~$0.02 ≈ $12 per 30-min race (~$36 for 3 seeds). **Setup: 1–2 h** (preset change only).
- **Rigging risk: medium.** "You forbade Opus from calling in parallel" is the obvious attack (pitch.md already has the answer). Pre-empt with a reference column for **Opus with 4 calls in flight at the same $ cap**, and print the in-flight limit in the table. Also measure, not assume, the K-scaling slope and the adapter latency. **If the adapter runs at ~2 s instead of 1 s, the projected gap shrinks to 22–40% against 33–67%.** RL gains plateau fast (R23: −164 ± 6), so "quality at a fixed budget" may tie, and $ and time to target carry the result.

### 3. Decision replay: $ and latency per decision at matched agreement ★ most certain
- **What:** offline replay of the held-out part of the recorded calls (489 propose, 138 judge, R20): the same inputs to Opus (answers already recorded), v1, v2 and Sapling. No GPUs.
- **Why Opus is weak here:** price and latency per call (R10/R18: 6–9.5 s, ~$0.02 per propose call; R24: ~2.2 s, ~$0.005 per judge call).
- **Why we should be strong:** the literature in A1 (MiniCheck at 1/400 of the cost, JudgeLM, LoRA Land).
- **Fair metric:** p50/p90 latency and $ per decision, malformed-reply rate, and **agreement with Opus** (what Mubit grades) plus ground-truth accuracy where we have it (judge). Never "better per call" (CLAUDE.md). The claim is "X% of Opus's agreement at Y% of the cost".
- **Cost:** ~$0 GPU; adapter call $ depends on Mubit pricing (unknown). **Setup: 1–2 h.**
- **Rigging risk: low.** The risk is that it's "only per call". Use it to *explain* regimes 1, 2 and 4, not as the headline.

### 4. Surrogate HPO race with a virtual clock (YAHPO LCBench) ★ thinking time is the whole cost
- **What:** new target `lcbench`: the agent proposes the 7 settings (plus epochs) for an OpenML task; it sees **val_accuracy** and is scored on **test_balanced_accuracy** (held-out). Each evaluation is charged its **surrogate-predicted runtime** on a virtual GPU clock at the L4 price; LLM time and $ are charged as real. 8 tasks × 3 seeds × 4 arms.
- **Why Opus is weak here:** with near-instant evaluations, wall-clock and $ are almost all LLM time. Proposal quality per evaluation is roughly level with Optuna (A5).
- **Why we should be strong:** an adapter with similar proposal quality at a fraction of the latency and $ wins time and $ to target outright. The Centaur result (a 0.8B model is enough to steer) is direct support. Optional: a Centaur-style arm (TPE/CMA-ES state given to the adapter) is the likeliest overall winner.
- **Fair metric:** best held-out score after N evaluations (sample efficiency, N = 10, 30, 100) **and** at a fixed virtual-clock + $ budget. Report the run-length multiplier (1×, 10×, 100× the recorded runtime) where our advantage disappears.
- **Cost:** CPU pennies. LLM ~8 × 3 × 30 evaluations ÷ K=4 ≈ 180 Opus calls ≈ $4. **Setup: 3–5 h** (onnxruntime wrapper, a target spec in the user message per D24/D29, plus a short Opus collection on it, because adapters trained on rl/cifar/lm are off-distribution here, see R9).
- **Rigging risk: high.** "Instant evaluations make thinking time artificially dominant" and "LLMs may have memorised LCBench" (Zhang et al. raise contamination). Pre-empt with the multiplier sweep, an Optuna column (expect it to tie or win on quality, A5, and say so), and scoring only on the held-out test metric. Frame it as "the cost of decisions when experiments are cheap", which is how labs run large sweeps.

### 5. Mixed leakage + noise verification set v2 (subtle cases)
- **What:** the R24 set is too easy (Opus 50/50). v2 mixes noise claims (from regime 1) with **subtle** leaks drawn from Luo et al.'s pitfalls: undisclosed subsampling, checkpoint chosen on test, metric switched after the fact, preprocessing fit on all data. These come in race-format evidence (no split hashes).
- **Why Opus is weak here:** only the subtle-case literature applies (Luo et al.: 55% accuracy from the paper alone; SPOT: 21% recall). On blatant traps it is perfect (R24, R25), so expect a small accuracy gap.
- **Why we should be strong:** Sapling is a dedicated verifier with a fine-tuned head. v2 learns from corrections. Cost and latency per check (regime 3).
- **Fair metric:** fakes let through and real gains confirmed on a held-out split, plus $ per check.
- **Cost:** CPU/L4 to build cases ~$1–2; Sapling H100 inside the D28 $4 cap. **Setup: 4–6 h.**
- **Rigging risk: medium-high** (we design the traps). Pre-empt by deriving case types from published pitfalls (cite Luo et al.), fixing and hashing the set before any arm sees it, and including an "Opus + checklist in the message" column.

### 6. (Optional) Early-stopping decisions on real learning curves (LCBench raw data)
- **What:** "kill or continue this run?" at epoch k, with ground truth from the full 50-epoch curve (35 datasets × 2,000 configs, Apache-2.0). It's a recurring decision labs make thousands of times.
- **Why it's interesting:** it's very frequent, which favours a fast model. LLMs track state across trials poorly (Ferreira et al.).
- **Why it's last:** it's a new Mubit task (new frozen prompt, collection, and one of the 3 daily adapter starts), and a median-stopping rule or ASHA is likely just as good. Only if regimes 1–4 are done.

---

## Part C: recommended order and what to ask Mubit
1. **Regime 3 first** (1–2 h, ~$0 GPU): it measures the real adapter latency, which decides whether regimes 2 and 4 are worth running (Q1, R8).
2. **Regime 1** (best accuracy story, ~$2 compute; needs Q2).
3. **Regime 2** as the live race (~$4–10 CPU), with the Opus 4-in-flight reference column.
4. **Regime 4** if time allows (pennies of compute, ~$4 LLM, 3–5 h setup).
5. Regime 5 with the Sapling work (D28), Regime 6 only if everything else is done.

Ask at office hours: live adapter latency and price per call (Q1), corrections overriding teacher labels (Q2), and whether a new target in the user message keeps the same task (D24/D29: it should).
