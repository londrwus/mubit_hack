# AI-labs focus (D18)

Busy Machines is built for **AI labs**. Banks, fraud and generic enterprise ML are out of scope for the pitch, the targets and the demo.

## Who we're talking to
| Persona | Their day | Their pain | What we show them |
|---|---|---|---|
| **Research engineer** (pretraining / architecture) | Agent-driven sweeps and ablations overnight | GPUs idle while the agent thinks; slow iteration | GPU idle % → ~0, more experiments per GPU-hour, LM speedrun result |
| **RL / post-training lead** | Reward models, LLM judges, rollout filtering | Judge cost and latency; judges get gamed and go stale | Distilled judges that refresh from corrections (Mubit adapters v1 → v2) |
| **Eval lead** | Benchmarks, leaderboards, release gates | **Contaminated or overfit "gains"** reaching papers and launches | An independent verifier (Sapling) catching the contamination trap |
| **Head of compute / research ops** | GPU budget and allocation | Paying for idle GPUs; no "cost per result" visibility | The $-to-target dashboard: cost per research result, not per GPU-hour |

## Targets (benchmark + demo): the work labs repeat thousands of times
| Target | What | Metric | Hardware | Run length | Contamination trap |
|---|---|---|---|---|---|
| **LM speedrun** | nanochat / nanoGPT-style pretraining (priml's nanochat baseline is an Apache-2.0 option) | val bits-per-byte ↓ | L4 | 60 s budget | the validation shard overlaps the training shards |
| **RL** | gymnax classic environments (JAX) | mean return ↑ | **CPU** (32 workers per arm, D21) | ~30 s | evaluation on training seeds/levels |
| **Vision speedrun** | CIFAR-10 speedrun (airbench-style) | test accuracy ↑ | L4 | seconds | the test set is used for checkpoint selection or augmentation tuning |
| *Stretch:* **tiny-LM post-training** | SFT/DPO on a small open model | eval loss / win rate | L4/A10 | ~90 s | eval prompts present in the SFT data |

Check each repo's licence and pin versions on Friday night. All code we write is written during the event (rule).

## Why "contamination" is the right trap
- It is a **real, current AI-lab problem**: benchmark contamination and test-set leakage affect papers, leaderboards and launch decisions.
- It is **the verifier-bias story Anil wrote about**: an agent judging its own result with shared context approves fake gains, and an independent verifier doesn't.
- It gives **ground truth** (we plant it), so the "correction → model update" moment is measurable, not a matter of opinion.

## Pitch lines (AI-lab version)
- **"Same GPUs. Less money. Better models."**
- "Most of a lab's GPU bill is experiments. That's exactly where we work." (*verify the Epoch AI figure before quoting*)
- "Your research agent shouldn't keep your GPUs waiting, and it shouldn't believe contaminated results."

## What changed vs earlier docs
Fraud and bank material in docs 03–08, 10 and 14 is **superseded**: decisions.md D18 wins. The CPU fraud target is replaced by the CIFAR-10 speedrun. The leakage trap is now the contamination trap.
