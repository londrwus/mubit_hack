# Pitch options: Busy Machines

> **Stage wording lives in `20_plain_language.md`** (D19): spend less on GPUs, get a better model, train faster. This file holds the reasoning and the Q&A-level versions.

## ⭐ PRIMARY PITCH: "Cost and efficiency" for AI labs' ML/DL/RL research

Built on Anil's own frame: *"after all the shiny aspects… only 2 things matter: cost and efficiency."*

### Two pillars
| | **COST** | **EFFICIENCY** |
|---|---|---|
| Promise | Labs spend **less money on GPUs** | Labs get **better models** from the GPUs they have |
| How | No idle GPU time while the agent thinks; Mubit adapters answer fast and cheap | More experiments per GPU-hour, every gain verified as real |
| Proof metric | **$-to-target**, idle GPU $ | **Best score at a fixed budget**, experiments per GPU-hour |

### Scope (say this precisely; it makes the claim defensible)
We cut cost and improve results in the **experimentation layer** of AI labs: hyperparameter sweeps, ablations, architecture search, RL environment and reward tuning, and small-scale research runs. That is the work that decides what the big run looks like.
We do **not** claim to speed up a single giant pretraining run.
> Hook to verify before using: Epoch AI estimated that most of OpenAI's 2024 R&D compute went to **experimental and research runs, not the final training runs**. If confirmed, that's the line: *"Most of a lab's GPU bill is experiments. That's exactly where we work."*

### Primary pitch (30 seconds)
> "Anil says only two things matter: **cost and efficiency.** For AI labs, both are decided in the experiment loop. That's where agents now run sweeps, ablations and RL tuning across thousands of short runs.
>
> Today that loop is wasteful. Every time the agent stops to think, **the GPUs sit idle and the lab pays for it.**
>
> Busy Machines runs the agent's thinking through Mubit. Its recurring decisions become small, fast models trained on its own work.
>
> **Cost:** labs spend less on GPUs, because the idle time disappears.
> **Efficiency:** they get better models from the same compute, because they run more experiments and an independent verifier makes sure every gain is real.
>
> Same GPUs. Less money. Better models."

### Closing line options
- **"Same GPUs. Less money. Better models."**
- **"Cost down. Quality up. That's the whole pitch."**
- **"Cost and efficiency, measured on real GPUs."**

---

The four benefits we sell, each tied to a **measured metric**. Never claim a benefit we didn't measure.

| Benefit | Metric that proves it |
|---|---|
| 💸 **Less money on GPUs** | $-to-target, idle GPU $ |
| ⚡ **Hit the target faster** | time-to-target |
| 🔥 **GPUs always busy** | GPU idle % |
| 🏆 **Better results** | best held-out score at a fixed budget |

## Taglines
1. **"Spend less. Wait less. Discover more."** (money / speed + busy / results in three beats)
2. **"Your GPUs should never wait for your agent."**
3. **"Cheaper, faster, busier, better: the same GPUs, a smarter night shift."**
4. **"Every GPU-second spent on science, not on waiting."**
5. **"Same budget. More discovery."**
6. **"Your agent thinks in milliseconds, so your GPUs never sleep."**

## One-sentence pitches
- **A. Problem-first:** "AI agents now run ML experiments, but every time they stop to think, your GPUs sit idle and you pay for it. Busy Machines keeps them busy, so you hit your target faster, spend less, and end up with a better model."
- **B. Result-first:** "On the same GPUs and the same budget, our researcher reaches the target X× faster, for Y% less, and finishes with a better model than an Opus 5.5 agent."
- **C. Mechanism-first (for Mubit judges):** "Busy Machines turns an ML agent's recurring decisions into small models via Mubit, so its thinking stops being the bottleneck: GPUs busier, results sooner, budget further, and every correction makes tomorrow's run better."

## 30-second pitch
> "AI agents now run ML experiments. They propose a change, launch training, read the result and repeat. But every time the agent stops to think, **your GPUs sit idle, and you're paying for them.**
>
> Busy Machines routes the agent's thinking through Mubit, which turns its recurring decisions into small, fast models trained on its own work.
>
> The result: **your GPUs stay busy**, you **reach your target faster**, you **spend less per discovery**, and on the same budget you **end up with a better model.** And every correction a researcher makes becomes a model update, so tonight's run beats last night's."

## Four-pillar slide
```
 💸 LESS MONEY        ⚡ FASTER            🔥 BUSIER            🏆 BETTER
 $-to-target          time-to-target       GPU idle %           best score @ same budget
 $X vs $Y             X min vs Y min       6% vs 75%            0.86 vs 0.81
```
(These numbers are placeholders. Only measured results go on stage.)

## Audience variants
- **AI labs:** "More experiments per GPU-hour means more discoveries per quarter, on the compute you already have."
- **RL / post-training teams:** "Your reward and eval loops run millions of judge calls; we make them fast, owned and refreshed from your corrections."
- **Eval teams:** "An independent verifier that catches contaminated gains before they reach a paper or a launch."
- **Mubit judges:** "Agents wait less and do more, measured on real GPUs. Every correction becomes a model update."

## Demo opening line options
- "Look at these GPUs. Grey means idle. You're paying for every grey second."
- "This is an Opus 5.5 researcher. It's brilliant. And 75% of the time, its GPUs are doing nothing."
- "Two AI researchers. Same GPUs, same budget, same hour. One change."

## Lead pitch: better quality AND less money (built on Anil's own words)

### The core claim: one curve, two wins
Busy Machines moves the whole **quality-vs-spend curve**:
- **Fix the budget → better quality.** Same $, a better model (best held-out score at the same budget).
- **Fix the quality target → less money.** Same model, fewer $ (the $-to-target).

> **"A better model for the same money, or the same model for less money. You choose."**

This is honest (both are measured on the same runs) and it avoids Anil's objection that "you're not saving money, you'll burn those tokens on something else." We agree: the saved GPU money **becomes more experiments, which becomes better quality.**

### Anil's problems → our answer → the proof
| Anil said | The problem | Busy Machines answer | Metric |
|---|---|---|---|
| "Token efficiency… makes your agents **wait less and do more**" | Agents waste time waiting on slow LLM reasoning | The agent's recurring decisions run on fast Mubit adapters | LLM latency per step, experiments/hour |
| "Combine this with **parallelism and cloud agents**, you have the **full utilisation cycle**" | GPUs sit idle while the agent thinks | The agent keeps N GPUs busy in parallel | **GPU idle %** |
| "After all the shiny aspects… **only 2 things matter: cost and efficiency**" | Research budgets are GPU-bound | Less idle spend; savings reinvested in experiments | **$-to-target**, idle GPU $ |
| "Avoid layers that could **add bias**… verifier with shared context" | Agents fool themselves with leaked or overfit "gains" | **Independent verifier** (Banyan Sapling) with no shared context checks every claimed gain | Leakage-trap accuracy |
| "A herculean task to manage a **farm of models which continuously evolve** from your pointers… **low latency**" | Keeping task models fresh and fast | Each correction → a new adapter version through Mubit; the agent improves every night | Adapter v1 → v2 quality |
| "**Time to visibility**" | Value is unclear for weeks | A dashboard shows $ saved and quality gained **after one night** | The live race |

So **quality is better** for two reasons, and both are measured:
1. **More experiments in the same budget**, so the search finds better models.
2. **Verified results**: the independent verifier rejects fake gains, so the "better" number is real.

### Lead pitch (30 seconds)
> "Anil says only two things matter: **cost and efficiency.** In ML research both are broken. Your agent stops to think, your GPUs sit idle, and you pay for it.
>
> Busy Machines runs the agent's thinking through Mubit, so it **waits less and does more**. Your GPUs stay busy and the money you were wasting becomes more experiments.
>
> That gives you **a better model for the same money, or the same model for less money.** An independent verifier makes sure every gain is real, and every correction becomes a model update."

### Taglines for this angle
- **"Better models. Smaller GPU bills."**
- **"Same money, better model. Same model, less money."**
- **"Stop paying for idle GPUs. Start paying for results."**

## Q&A guardrails
- "Better results" means **at a fixed budget**, not that a small model is smarter than Opus per call.
- Savings come mostly from **idle GPU time and throughput** in short-iteration loops. Long single runs benefit less; say so.
