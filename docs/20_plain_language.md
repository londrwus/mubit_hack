# Plain-language pitch and demo (D19)

**Rule:** everything said on stage and shown on screen must make sense to someone who has never trained a model. Technical depth is for Q&A and the docs only.

## The three promises (the whole pitch)
> ### 💸 Spend less on GPUs · 🏆 Get a better model · ⚡ Train faster

Every slide, every screen and every sentence supports one of these three. If it doesn't, cut it.

## Why "better model" follows from "faster and cheaper"
> **"Because the GPUs never wait, more of your budget goes into actual training, and more training means a better model."**

Faster thinking leads to more experiments on the same GPUs, which means more chances to find better settings and a better final model. This is the honest causal chain, and the race **measures** it: same budget and same time, compare model quality. Use the line on stage; the number on the slide must be the measured one. If one target doesn't show a quality gain, say "same quality, faster and cheaper" for that target.

## 20-second pitch
> "AI companies spend a fortune on GPUs to train their models. Today, an AI assistant runs those experiments, but every time it stops to think, **the GPUs sit there doing nothing, and you still pay for them.**
>
> Busy Machines keeps the GPUs busy. **You spend less, you train faster, and you end up with a better model.**"

## 1-minute pitch
> "Training AI models is mostly trial and error: try a setting, train, check the result, try again. Companies now let an AI assistant do that loop. The problem: every time the assistant thinks about what to try next, **the expensive GPUs wait. You're paying for machines that do nothing.**
>
> Busy Machines fixes that. With Mubit, the assistant learns from its own work and answers in a split second, so **the GPUs never wait.**
>
> Same GPUs, same money: **we finish faster, spend less, and get a better model.** And when it makes a mistake, you correct it once, and it learns. It won't make that mistake again."

## Demo narration (plain words)
| Moment | Say | Don't say |
|---|---|---|
| Opening | "Two AI assistants. Same GPUs. Same budget. One hour. Let's race." | "Two agent arms with N GPU slots each…" |
| Grey tiles | "Grey means the GPU is waiting. You're paying for every grey second." | "GPU idle %, utilisation" |
| Counters | "This one has finished 17 experiments. That one, 3." | "experiments per GPU-hour" |
| Leaderboard | "Higher is better. Ours is ahead." | "val_bpb went from 1.14 to 1.09" |
| Cost | "That side has wasted $2.40 on waiting. Ours, 20 cents." | "idle GPU $, $-to-target" |
| The mistake | "Look, it thinks it found a huge improvement. But the improvement is fake: the model was tested on data it had already seen." | "eval contamination, train/val overlap" |
| The correction | "We flag it once… Mubit retrains it… now it spots fake improvements on its own." | "adapter v1 → v2, corrections as training examples" |
| Close | "Spend less on GPUs. Get a better model. Train faster." | "cost per discovery at fixed budget" |

## On-screen labels (dashboard copy)
| Technical name | What the screen says |
|---|---|
| GPU idle % | **GPUs waiting** |
| GPU busy | **GPUs working** |
| $-to-target | **Cost to reach the goal** |
| time-to-target | **Time to reach the goal** |
| best score at fixed budget | **Model quality** (show as a 0–100 bar, higher = better) |
| experiments per GPU-hour | **Experiments finished** |
| idle GPU $ | **Money wasted on waiting** |
| val_bpb / test_acc / mean return | **Model quality** (the raw metric appears only on hover) |
| contamination trap | **Fake improvement caught** |
| adapter v1 → v2 | **"Learned from your correction"** |
| Opus 5.5 arm | **Standard AI assistant** (subtitle: Opus 5.5) |
| Mubit arm | **Busy Machines** (subtitle: powered by Mubit) |
| nanochat / RL / CIFAR | **Language model · Game-playing AI · Image recognition** |

The model quality bar is normalised per target: 0 = the starting model, 100 = the best known result. The raw number is on hover and in the export, so the claim stays verifiable.

## Words to avoid on stage → say instead
| Avoid | Say |
|---|---|
| ML / DL / RL | "training AI models" |
| hyperparameters, configs | "settings" |
| agent | "AI assistant" |
| adapter, distillation | "a small, fast model trained on its own work" |
| inference, latency | "how fast it answers" |
| eval contamination, leakage | "a fake improvement" (tested on data it had already seen) |
| benchmark | "a fair race" |
| seeds | "we ran it three times" |

## Where the technical depth goes
- **Q&A** (judges will ask): $-to-target, seeds, Optuna baseline, Mubit task grouping, contamination types. Prepare one-sentence answers (docs 06, 10, 12).
- **Results page / export:** raw metrics, request IDs, run IDs.
- **Docs and README:** full technical detail.
- **One exception on stage:** say "**Mubit**" by name every time the learning moment happens. The judges need to hear it.
