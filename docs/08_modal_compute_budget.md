# Modal compute plan ($30 budget)

## ⭐ UPDATE (D21): concurrency limits shape the benchmark
**Starter limits: 10 GPUs at once, 100 containers at once (CPU).** Plan races around them.

### GPU slots (10)
| Use | GPUs | Notes |
|---|---|---|
| Opus arm | 4 × L4 | LM speedrun / CIFAR speedrun |
| Mubit arm | 4 × L4 | identical |
| Sapling verifier | 1 × H100 (80 GB) | only while the verifier is in use; otherwise give it to the baseline |
| Spare / Optuna baseline | 1 | run the baseline arm separately if needed (same 4 GPUs, a separate time slot) |
- GPU seeds run **one after another** (3 seeds × 2 arms × 1 h ≈ 24 L4-hours ≈ $19 per GPU target). **Only one GPU target gets the full 3 × 1 h treatment** within $30; the second GPU target runs 30-minute races.

### CPU containers (100): the cheap scale story
- **Move the RL target to CPU** (gymnax classic environments run fast in JAX on CPU). CPU costs ~$0.05/core-hour, so it's almost free.
- **Scale race:** each arm gets **32 CPU workers** (2 arms = 64 containers, leaving room for the Optuna baseline at 32). Every seed can run in parallel at once.
- This is where the **"GPUs waiting" effect is biggest**: with 32 workers, a slow-thinking assistant can't keep them all fed, while the fast one can. It's the most dramatic version of the demo, and it's cheap: 64 cores × 1 h ≈ $3 (plus memory).
- Plain-language line: *"With 32 machines, the standard assistant keeps most of them waiting. Ours keeps them all working."*
- CPU containers also run the **dataset generator** for `ml-claims` (doc 21) and any analysis jobs.

### Fairness (unchanged)
Both arms get identical slots and both batch K proposals per call. The difference is only how fast and how cheaply the assistant decides.

### Revised $30 split
| Bucket | Cost |
|---|---|
| Dev / smoke tests (T4/L4 + CPU) | ~$3 |
| GPU target A (LM speedrun): 3 seeds × 2 arms × 1 h × 4 L4 | ~$19 → **reduce to 30 min ($10)** if Sapling needs H100 time |
| GPU target B (CIFAR speedrun): 30-minute races | ~$5 |
| CPU target (RL, 32 workers per arm, all seeds in parallel) | ~$3 |
| Sapling stages 1–2 on H100 | ~$5 |
| Live demo | ~$1 |
| **Total** | ~$30 (tight; ask about GPU credits, Q6) |

---
*(Original plan below, kept for reference.)*

## Modal prices (Starter plan, checked 23 Oct 2026)
| Resource | $/hour | 5-min run costs |
|---|---|---|
| H100 | $3.95 | $0.33 |
| A100 80GB | $2.50 | $0.21 |
| L40S | $1.95 | $0.16 |
| A10 | $1.10 | $0.09 |
| **L4** | **$0.80** | **$0.067** |
| T4 | $0.59 | $0.05 |
| CPU core | $0.047 | ~free |
| Memory GiB | $0.008 | ~free |

Sandboxes cost **3× on CPU/memory** (GPU price is the same). Use Functions, not Sandboxes, for training.
**Starter limits: 10 concurrent GPUs, 100 containers, $30 free credit per month.**

## Key insight: the budget *is* the benchmark
$30 is tight. Instead of treating that as a weakness, make it the headline:

> **"Give two AI researchers the same $X of GPU and the same hour. The Mubit-powered one gets more science out of it."**

Every lab and every bank is GPU-budget-bound, so this is the most relatable framing available. The metric becomes **best score per GPU-dollar**, plus **GPU idle %** (the Opus arm's GPUs sit idle while Opus thinks).

## Allocation
| Bucket | Hardware | Plan | Cost |
|---|---|---|---|
| Dev and smoke tests | T4/L4 + CPU | Harness, 5-minute runs, debugging | ~$4 |
| **Main benchmark** | **L4 × 4 parallel** | 2 arms (Opus agent, Mubit agent) × 3 seeds × 1 hour × 4 GPUs | 6 × 4 × $0.80 = **~$19** |
| Champion verification | H100 | Re-run each arm's best config × 3 seeds × 5 min on H100 (an "independent replication" the judges see) | ~$2 |
| Fraud / tabular arm | CPU only | XGBoost/LightGBM, many seeds | <$1 |
| Buffer | | reruns, the demo-day live run | ~$4 |

Total ≈ $30. **Set a hard spend cap in the Modal dashboard.**

## Which GPU task
- **Main:** a small nanoGPT/nanochat-style language model with a **5-minute L4 budget**, scored on validation bits-per-byte. priml's nanochat baseline is a ready-made starting point (Apache-2.0). It's cheap, standard, and lab-relevant.
- **RL:** Craftax-Classic or a gymnax environment in JAX. It runs millions of steps per minute on one L4, so it's also cheap.
- **Bank:** fraud PR-AUC on CPU.

## Fairness and verifiability
- Same GPUs, same concurrency (4), same 1-hour wall-clock, same seeds, same starting code. Only the LLM in the loop changes.
- Log Modal run IDs and Mubit request IDs into a results JSONL file, so every number traces back to a run.

## Other money to use
- The hackathon gives $35k in OpenAI and other AI credits. **Ask whether Modal or GPU credits are among them.** If so, scale to 10 GPUs per arm (the Starter cap) for a stronger parallelism story.
- The LLM cost of the Opus arm is *not* Modal money. Track it separately; it's part of the result (the $ per experiment for each agent).

## Live demo trick
Run a mini race on stage: both agents, 2 L4s each, 3 minutes. This costs about $0.20. Show a live dashboard of GPU utilisation for each arm, with the Opus side visibly idle while it waits on Opus. Show the full 1-hour, 3-seed results on the slide.
