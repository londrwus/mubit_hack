# A verifiable benchmark where we beat Opus 5.5

## Choose the claim we can actually win
A small distilled model will **not** beat Opus 5.5 call-for-call on general reasoning. Mubit even grades adapters on *matching* the provider. Claiming that would be false, and judges would catch it.

There are two claims that are true and verifiable:

### Claim 1 (headline): fixed time or budget, a better final model
> **"Same 60 minutes, same $ budget: our Mubit-powered agent finds a better fraud model than an Opus 5.5 agent."**

Why we win: the outcome is **decided by experiment throughput**. If adapters answer in about 0.3 s instead of Opus's 5–15 s, and cost far less, our agent runs many more experiments in the same budget. More search finds a better model. This is Anil's "agents wait less, do more" argument turned into a number.

### Claim 2 (Mubit-native): corrections let the adapter beat its teacher on one narrow task
The `judge_improvement` step asks whether a gain is real or caused by leakage or overfitting. We grade it against **ground truth** (we plant the leakage, so we know the answer), not against Opus. Opus will sometimes accept leaked gains. After researcher corrections, adapter v2 learns *our* rule. **v2 can score higher than Opus on this task.** That is the "correction → model update" moment, with a number attached.
⚠️ This depends on corrections being able to override the teacher label. **Confirm with Mubit at office hours on Saturday morning.** If they can't, keep Claim 1 only.

## Benchmark design ("NightBench")
| Item | Choice |
|---|---|
| Tasks | 3 targets: credit-card fraud (Kaggle, tabular), IEEE-CIS fraud subset, CartPole/LunarLander (RL). All CPU-trainable in seconds to minutes. |
| Arms | (A) the agent with Opus 5.5 for every step, (B) the agent with Mubit adapters for the 4 recurring steps (Opus only as fallback), (C) a random/grid-search baseline as a sanity floor. |
| Budget | Fixed wall-clock (e.g. 30 or 60 min) **and** a fixed $ cap. Report both. |
| Metric | **Held-out test** PR-AUC (fraud) and mean return over 100 eval episodes (RL). The test set is never seen by the agent. |
| Seeds | ≥3 seeds per arm. Report mean ± std. Without this, any win is noise. |
| Leakage trap set | 50 synthetic "improvements", half leaked and half genuine, for Claim 2. Accuracy of Opus vs adapter v1 vs v2. |
| Efficiency | p50 latency per step, tokens and $ per experiment, experiments per hour. |

## What makes it verifiable (judges trust this)
- Public repo with a single `make bench` command, plus fixed seeds and fixed data splits.
- Results log every run as JSONL, including the Mubit request IDs, so judges can cross-check them in the console.
- Show the losing cases honestly, e.g. "on CartPole the gap is small". This builds credibility for the "holds up under questions" criterion.

## Demo slide
```
Fraud PR-AUC after 60 min        Opus agent 0.81   Mubit agent 0.86   (3 seeds)
Experiments run                  14                 90
$ per experiment                 $0.40              $0.03
Leakage-trap accuracy            Opus 78%   v1 76%   v2 (after corrections) 94%
```
(These numbers are illustrative placeholders. Only the real measured values go on stage.)

## Risks
- If adapters aren't callable for live inference outside Try models (there's no Mubit model ID), Arm B can't run. **This is the first question to ask Mubit on Friday night.** The fallback is to run the small base model ourselves, but the rules say every LLM call must go through Mubit, so check with them first.
- Training slots are limited (3 per day). Collect examples on Friday night and train on Saturday morning.
- Arm A costs real Opus money. Use the provided credits and cap it.
