# Pitch: "Cost per discovery"

## One-liner
> **"We make AI research cheaper per GPU-hour. Your ML/DL/RL experiments reach the same result in less time and fewer dollars, because the agent driving them runs on models Mubit trains from its own work."**

## Headline metrics
The core metric is **time-to-target** and **$-to-target**: how long and how much money it takes to reach a fixed quality bar. For example, nanochat val_bpb ≤ 1.10, fraud PR-AUC ≥ 0.85, or Craftax return ≥ X.

| Metric | Definition | Why judges care |
|---|---|---|
| **$-to-target** | GPU $ + LLM $ until the target is hit | The bank and lab buyer's language |
| **Time-to-target** | wall-clock minutes | Anil's "agents wait less" |
| **GPU utilisation %** | busy time / paid time | Visible waste in the Opus arm |
| **Experiments per GPU-hour** | throughput | Explains *why* we win |
| **Best score at fixed budget** | at 60 min / $X | The "beats Opus 5.5" headline |

A combined cost breakdown, shown live on the dashboard:
```
                    Opus agent     Mubit agent
GPU $ (paid)        $3.20          $3.20
  ↳ idle GPU $      $2.40 (75%)    $0.20 (6%)
LLM $               $6.10          $0.40
Time to target      did not reach  38 min
Cost to target      —              $2.90
```
(These are placeholders. Only measured numbers go on stage.)

## Why this framing works
- **Simple:** everyone understands "you're paying for GPUs that sit idle."
- **Broad and true:** it applies to any ML/DL/RL training loop driven by an agent.
- **Mubit-native:** the speed-up comes from Mubit adapters, and the correction moment improves them further. Each fix lowers the cost per discovery the next night.
- **Buyers:** AI labs ("more discoveries per GPU budget") and banks ("retrain fraud models faster and cheaper as fraud patterns drift"), which also fits the Fast refresh track.
- **Not what Mubit already pitches:** they talk about agent cost per task. We extend it to **cost per research outcome on GPUs**, which is a new market for them.

## Honest caveat for Q&A
The savings come mostly from **LLM wait time and LLM cost**, not from faster GPUs. When a training run is long (hours), the LLM wait matters less. Our answer: "That's why we win most in fast-iteration loops: hyperparameter sweeps, small-model research, RL tuning, and fraud retraining. These are exactly where labs run thousands of short experiments."
