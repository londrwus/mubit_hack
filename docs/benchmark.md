# night/bench: the benchmark vs Opus 5.5 (the MUST, D20)

Goal: prove, fairly and verifiably, that the Mubit-powered researcher **trains faster, gets a better model, and spends less** than an Opus 5.5 researcher on the same GPUs. It complements Mubit's `latte/bench` (note-based learning) by measuring **weight learning on real tasks, costed in GPU $**.

## 1. The claims we can win
A small distilled model will **not** beat Opus 5.5 call-for-call on general reasoning, and Mubit grades adapters on *matching* the provider. Claiming that would be false and judges would catch it (D3). Two claims are true and verifiable:

### Claim 1 (headline): same budget, a better result
> **"Same GPUs, same clock, same $ budget: the Mubit-powered researcher reaches the goal sooner and cheaper, and ends with a better model than an Opus 5.5 researcher."**

The outcome is **decided by experiment throughput**. If adapters answer much faster and cheaper than Opus, our arm runs more experiments in the same budget, and more search finds a better model. Anil's "agents wait less, do more", as a number. "Better model" is not guaranteed: report whichever way it lands, per target.

### Claim 2 (Mubit-native): after corrections, the adapter beats its teacher on one narrow task
`judge_improvement` asks whether a gain is real or contaminated/overfit. We grade it against **ground truth** (we plant the contamination), not against Opus. Opus will sometimes accept contaminated gains. After corrections, adapter v2 learns *our* rule, so **v2 can score higher than Opus on this task**: the correction → model update moment, with a number.
⚠️ Depends on corrections being able to override the teacher label (**Q2**). If they can't, keep Claim 1 only.

## 2. Design
| Item | Choice |
|---|---|
| **Arms** | (A) Opus 5.5 for every LLM step · (B) Mubit adapters for the recurring steps (Opus as fallback, via Mubit routing) · (C) **Optuna / random search, no LLM** (the "why not just Optuna?" answer) |
| **LLM steps** | Three recurring calls: `propose_experiments`, `diagnose_run` (failures only), `judge_improvement` (claimed new bests only). Summarising is code (`architecture.md`) |
| **Targets** | LM speedrun, RL, vision speedrun (section 3) |
| **Budget** | Fixed wall-clock **and** a fixed $ cap per arm. Report both |
| **Score** | The target's own **held-out** metric; the eval split is never seen by the agent |
| **Seeds** | ≥3 per arm, mean ± std. Without this, any win is noise |
| **Method** | RE-Bench's fixed-time-budget method on public speedrun task families |
| **Contamination-trap set** | 50 planted "improvements", half contaminated and half genuine. Accuracy of Opus vs adapter v1 vs v2 (Claim 2) |

### Metrics
| Metric | Definition | Plain label (`demo.md`) |
|---|---|---|
| **$-to-target** | GPU $ + LLM $ until the first experiment reaching `target_value` | Cost to reach the goal |
| **Time-to-target** | wall-clock minutes to the same point | Time to reach the goal |
| **GPU idle %** | idle time / paid time (`gpu_intervals`) | GPUs waiting |
| **Experiments per GPU-hour** | throughput; explains *why* we win | Experiments finished |
| **Best score at a fixed budget** | best held-out score at the end of the budget | Model quality |
| Efficiency | p50 LLM latency per step, tokens and $ per experiment | (hover / export) |

Target examples: LM val_bpb ≤ a threshold, CIFAR-10 test accuracy ≥ 94%, RL return ≥ X. Set the thresholds from the starting code's performance before the first race, and don't move them afterwards.

## 3. Targets: the work labs repeat thousands of times (D18, D21)
| Target | What | Metric | Hardware | Run length | Contamination trap |
|---|---|---|---|---|---|
| **LM speedrun** | nanoGPT-style pretraining on a small text dataset (Shakespeare-char style); priml's nanochat baseline (Apache 2.0) is the alternative | val bits-per-byte ↓ | L4, 4 per arm | **60 s** budget | the validation shard overlaps the training shards |
| **RL** | gymnax classic environments, single-file PPO (purejaxrl-style) | mean return over held-out eval seeds ↑ | **CPU, 32 workers per arm** | ~30 s | evaluation on the training seeds/levels |
| **Vision speedrun** | CIFAR-10 speedrun (airbench-style), ~8 exposed settings | test accuracy ↑ | L4, 4 per arm | seconds | the test set is used for checkpoint selection or augmentation tuning |
| *Stretch:* tiny-LM post-training | SFT/DPO on a small open model | eval loss / win rate | L4/A10 | ~90 s | eval prompts present in the SFT data |

Check each repo's licence and pin versions on Friday night. All code we write is written during the event.

### Why runs must be short (20–90 s)
Run length decides whether the "GPUs waiting" effect is real. With 5-minute runs and 20 s of thinking, the Opus arm's GPUs are idle only ~6% of the time, which is a weak result. Short runs are also where labs run thousands of sweeps, and it's honest to say so.

### Why the RL target runs on CPU with 32 workers
Modal Starter allows 10 GPUs but 100 containers at once. With 32 workers per arm, a slow-thinking assistant can't keep them all fed while a fast one can. That's the most dramatic version of the race, it runs every seed in parallel, and it costs ~$3. Plain line: *"With 32 machines, the standard assistant keeps most of them waiting. Ours keeps them all working."*

### Seed and time plan (fits the $30, see `budget.md`)
- GPU seeds run **one after another**. Only **one GPU target** gets the full 3 seeds × 2 arms × 1 h (≈ 24 L4-hours ≈ $19, cut to 30 min ≈ $10 if Sapling needs the H100); the other GPU target runs 30-minute races.
- The CPU RL target runs all seeds in parallel.
- The Optuna arm runs in a separate time slot on the same slot count if GPUs are short.

## 4. Fairness rules
- Both arms get **identical** GPUs/CPU slots, concurrency, wall-clock, $ cap, seeds and starting code. **Only the LLM in the loop changes.**
- Both arms use the same scheduler, the same batching (K proposals per call) and the same LLM cost optimisations (`budget.md`). We must beat an *optimised* Opus agent, not a strawman.
- Malformed replies, retries and Opus fallbacks count against the arm that produced them (time and $). Report the malformed-reply rate per arm.
- The runner enforces slot caps so a race never queues silently (queued time would show up as fake idle time).
- Record/replay and the mock LLM are dev-only; a benchmark run never uses them (flagged in the DB).

## 5. What makes it verifiable
- One command (`make bench`) with fixed seeds and fixed data splits.
- Every LLM call stores its Mubit request ID; every run its Modal call ID, seed and config. `GET /api/races/{id}/export.jsonl` dumps it all so judges can cross-check in the console.
- Every measurement goes into `results_log.md`, **including where we lose** (e.g. "on RL the gap is small"). Slides quote only logged numbers.
- Optional, budget permitting: re-run each arm's best config × 3 seeds on an H100 as an independent replication.

## 6. Results slide (layout only)
```
                                    Standard AI assistant   Busy Machines     No-AI search
                                    (Opus 5.5)              (Mubit)           (Optuna)
Cost to reach the goal              $X                      $Y                $Z
Time to reach the goal              X min                   Y min             Z min
GPUs waiting                        X%                      Y%                Z%
Model quality @ same budget         X                       Y                 Z      (3 runs each)
Fake improvements caught            Opus X%   v1 Y%   v2 (after corrections) Z%
```
Placeholders: only measured values from `results_log.md` go on stage.

## 7. Risks
- **Adapters serving live traffic (Q1)** is the biggest external risk. If routing can't be observed or doesn't kick in, the fallback is an open-model arm through Mubit's `/compat/{groq,together}` routes, which changes the story slightly (`plan.md`).
- **Adapters must pass the exam before they serve.** The toy adapter scored 70% vs an 85% target (R7), and routing goes through shadow before live. If our step adapters stay below target, Arm B runs on Opus fallback and there's no race. Collect plenty of examples, keep each step's output simple, and ask Mubit how to create a route (Q1, Q14).
- **Training timing:** 3 adapter starts per UTC day (resets 01:00 BST). Training itself is quick (~5 min). Collect examples Friday night; train Saturday morning.
- **Noise:** 60 s LM runs are noisy; compare with 3 seeds and say so.
