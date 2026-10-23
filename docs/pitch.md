# Pitch: who we sell to, what we say, how we answer

**Stage rule (D19):** everything said on stage and shown on screen must make sense to someone who has never trained a model. Technical depth is for Q&A, the results export and the docs. Dashboard copy and demo narration are in `demo.md`.

## 1. The customer: AI labs only (D18)
Banks, fraud and generic enterprise ML are out of scope for the pitch, the targets and the demo.

### Personas
| Persona | Their day | Their pain | What we show them |
|---|---|---|---|
| **Research engineer** (pretraining / architecture) | Agent-driven sweeps and ablations overnight | GPUs idle while the agent thinks; slow iteration | GPUs waiting → ~0, more experiments per GPU-hour, the LM speedrun result |
| **RL / post-training lead** | Reward models, LLM judges, rollout filtering | Judge cost and latency; judges get gamed and go stale | Distilled judges that refresh from corrections (adapter v1 → v2) |
| **Eval lead** | Benchmarks, leaderboards, release gates | **Contaminated or overfit "gains"** reaching papers and launches | An independent verifier (Sapling) catching the contamination trap |
| **Head of compute / research ops** | GPU budget and allocation | Paying for idle GPUs; no "cost per result" visibility | The cost-to-goal dashboard: cost per research result, not per GPU-hour |

### Segments, ranked by how much Mubit should care
Mubit's business is usage through the gateway → owned adapters per recurring task → retention (the models are trained on *your* traces). It wants **high-volume, recurring, verifiable LLM calls**. AI-lab research loops have that shape.

| # | Segment | Recurring LLM volume | Willingness to pay | Verdict |
|---|---|---|---|---|
| 1 | **RL / post-training and eval teams** (LLM-as-judge, reward grading, rollout filtering) at mid-size labs, fine-tuning companies and RL-environment vendors | **Enormous**: millions of judge calls per run, fixed rubric and output | High: judge cost and latency slow training | ⭐⭐⭐⭐⭐ The highest-volume wedge: distilled judges plus a Sapling-style independent verifier |
| 2 | **Research teams at mid-size labs and neolabs** running agent-driven sweeps, ablations, architecture search, speedruns | High, nightly | High: compute is their biggest cost line | ⭐⭐⭐⭐⭐ The core Busy Machines buyer: $-to-target and GPU idle % speak to their budget |
| 3 | **AI-scientist startups** (Rekursiv-type) | High per customer, few customers | Medium | ⭐⭐⭐ Design partners and logos |
| 4 | **GPU clouds / neoclouds** (Modal, Lambda, CoreWeave) as a *channel* | n/a | Partnership | ⭐⭐⭐ They want customers who use GPUs productively; a marketplace integration |
| 5 | **Frontier labs** | Huge | Low: they build in-house and keep traces internal | ⭐⭐ **Credibility, not revenue**: open night/bench results, collaborations, owned-model options later |

### Why contamination is the right trap
- It's a **real, current AI-lab problem**: benchmark contamination and test-set leakage affect papers, leaderboards and launch decisions.
- It's **the verifier-bias story Anil wrote about**: an agent judging its own result with shared context approves fake gains; an independent verifier doesn't.
- It gives **ground truth** (we plant it), so the correction moment is measurable, not a matter of opinion.

## 2. The message
### The three promises (the whole pitch)
> ### 💸 Spend less on GPUs · 🏆 Get a better model · ⚡ Train faster

Every slide, screen and sentence supports one of these. If it doesn't, cut it. Each is tied to a measured metric, and we never claim one we didn't measure:

| Promise | Metric that proves it |
|---|---|
| 💸 Less money on GPUs | $-to-target, idle GPU $ |
| ⚡ Hit the target faster | time-to-target |
| 🔥 GPUs always busy | GPU idle % |
| 🏆 Better results | best held-out score at a fixed budget |

### Why "better model" follows from "faster and cheaper"
> **"Because the GPUs never wait, more of your budget goes into actual training, and more training means a better model."**

Faster thinking → more experiments on the same GPUs → more chances to find better settings → a better final model. The race **measures** this chain (same budget, same time, compare quality). If one target shows no quality gain, say "same quality, faster and cheaper" for that target.

Quality is better for two measured reasons: **more experiments in the same budget**, and **verified results** (the independent verifier rejects fake gains, so the "better" number is real).

### One curve, two wins
- **Fix the budget → better quality** (best held-out score at the same budget).
- **Fix the quality target → less money** ($-to-target).

> **"A better model for the same money, or the same model for less money. You choose."**

This answers Anil's objection ("you're not saving money, you'll burn those tokens on something else"). We agree: the saved GPU money **becomes more experiments, which becomes better quality.**

### Scope (say it precisely; it makes the claim defensible)
We cut cost and improve results in the **experimentation layer**: hyperparameter sweeps, ablations, architecture search, RL environment and reward tuning, small-scale research runs. That's the work that decides what the big run looks like. We do **not** claim to speed up a single giant pretraining run.
> Hook, **verify before quoting:** Epoch AI estimated most of OpenAI's 2024 R&D compute went to experimental and research runs, not final training runs. If confirmed: *"Most of a lab's GPU bill is experiments. That's exactly where we work."*

## 3. Scripts
### 20 seconds (plain, for anyone)
> "AI companies spend a fortune on GPUs to train their models. Today, an AI assistant runs those experiments, but every time it stops to think, **the GPUs sit there doing nothing, and you still pay for them.**
>
> Busy Machines keeps the GPUs busy. **You spend less, you train faster, and you end up with a better model.**"

### 1 minute (plain)
> "Training AI models is mostly trial and error: try a setting, train, check the result, try again. Companies now let an AI assistant do that loop. The problem: every time the assistant thinks about what to try next, **the expensive GPUs wait. You're paying for machines that do nothing.**
>
> Busy Machines fixes that. With Mubit, the assistant learns from its own work and answers in a split second, so **the GPUs never wait.**
>
> Same GPUs, same money: **we finish faster, spend less, and get a better model.** And when it makes a mistake, you correct it once, and it learns. It won't make that mistake again."

### 30 seconds (for Mubit judges: cost and efficiency, in Anil's words)
> "Anil says only two things matter: **cost and efficiency.** For AI labs, both are decided in the experiment loop, where agents now run sweeps, ablations and RL tuning across thousands of short runs.
>
> Today that loop is wasteful. Every time the agent stops to think, **the GPUs sit idle and the lab pays for it.**
>
> Busy Machines runs the agent's thinking through Mubit, so it **waits less and does more**. Its recurring decisions become small, fast models trained on its own work.
>
> **Cost:** less money on GPUs, because the idle time disappears. **Efficiency:** better models from the same compute, because it runs more experiments and an independent verifier makes sure every gain is real. And every correction becomes a model update, so tonight's run beats last night's.
>
> Same GPUs. Less money. Better models."

### One-sentence versions
- **Problem-first:** "AI agents now run ML experiments, but every time they stop to think, your GPUs sit idle and you pay for it. Busy Machines keeps them busy, so you hit your target faster, spend less, and end up with a better model."
- **Result-first** (only with measured numbers): "On the same GPUs and the same budget, our researcher reaches the target X× faster, for Y% less, and finishes with a better model than an Opus 5.5 agent."
- **Mechanism-first (Mubit judges):** "Busy Machines turns an ML agent's recurring decisions into small models via Mubit, so its thinking stops being the bottleneck: GPUs busier, results sooner, budget further, and every correction makes tomorrow's run better."

### Lines
- **Closers:** "Same GPUs. Less money. Better models." · "Cost down. Quality up. That's the whole pitch." · "Cost and efficiency, measured on real GPUs."
- **Taglines:** "Spend less. Wait less. Discover more." · "Your GPUs should never wait for your agent." · "Every GPU-second spent on science, not on waiting." · "Same budget. More discovery." · "Better models. Smaller GPU bills." · "Stop paying for idle GPUs. Start paying for results."
- **Openers:** "Look at these GPUs. Grey means idle. You're paying for every grey second." · "This is an Opus 5.5 researcher. It's brilliant. And most of the time, its GPUs are doing nothing." (quote the measured idle % only) · "Two AI researchers. Same GPUs, same budget, same clock. One change."
- **Audience variants:**
  - Research teams: "More experiments per GPU-hour means more discoveries per quarter, on the compute you already have."
  - RL / post-training: "Your reward and eval loops run millions of judge calls; we make them fast, owned and refreshed from your corrections."
  - Eval teams: "An independent verifier that catches contaminated gains before they reach a paper or a launch."
  - Mubit judges: "Agents wait less and do more, measured on real GPUs. Every correction becomes a model update."
- **latte/bench line:** "latte/bench asks whether agents learn. We measure what that learning is worth: dollars and hours to a real research result."

### Anil's problems → our answer → the proof
| Anil said | The problem | Busy Machines answer | Metric |
|---|---|---|---|
| "Token efficiency… makes your agents **wait less and do more**" | Agents waste time waiting on slow LLM reasoning | Recurring decisions run on fast Mubit adapters | LLM latency per step, experiments/hour |
| "Combine this with **parallelism and cloud agents**, you have the **full utilisation cycle**" | GPUs sit idle while the agent thinks | The agent keeps N GPUs busy in parallel | **GPU idle %** |
| "After all the shiny aspects… **only 2 things matter: cost and efficiency**" | Research budgets are GPU-bound | Less idle spend; savings reinvested in experiments | **$-to-target**, idle GPU $ |
| "Avoid layers that could **add bias**… verifier with shared context" | Agents fool themselves with contaminated or overfit "gains" | **Independent verifier** (Banyan Sapling), no shared context | Contamination-trap accuracy |
| "A herculean task to manage a **farm of models which continuously evolve**… **low latency**" | Keeping task models fresh and fast | Each correction → a new adapter version through Mubit | Adapter v1 → v2 quality |
| "**Time to visibility**" | Value is unclear for weeks | Dashboard shows $ saved and quality gained **after one night** | The live race |

## 4. Plain language on stage
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

- **One exception:** say "**Mubit**" by name every time the learning moment happens. The judges need to hear it.
- Technical depth goes to **Q&A** (section 6), the **results export** (raw metrics, request IDs, run IDs), and the **docs/README**.

## 5. The commercial story: "Mubit for Research" (D13)
Busy Machines is positioned as the **reference solution for "Mubit for Research"**, a vertical Mubit could take to AI labs during launch week. Every layer runs on a Mubit capability (map in `architecture.md`).
> **"We built the AI-lab product you can sell on Monday. Same base URL, three research tasks, and a GPU bill that shrinks every night."**

We'd also be the first team to use all three of their launches together (platform, Sapling, latte/bench counterpart), in a market they haven't addressed, on launch weekend.

### What the customer gets
| Component | What it is |
|---|---|
| **Research Task Pack** | Prompt templates for the three recurring steps, shaped to form clean Mubit tasks (stable instructions, variable content in the user message, JSON output) |
| **Outcome connector** | Training metrics (W&B / MLflow / custom) attached to Mubit request IDs |
| **Verifier** | Banyan Sapling calibrated for ML-research claims (contamination, overfit, noise) |
| **GPU efficiency dashboard** | $-to-target, GPU idle %, best score at a fixed budget, cost per successful experiment |
| **Onboarding** | Import Langfuse/Braintrust/Logfire traces; swap the base URL in Claude Code/Codex |
| **Data control** | Per-step recording levels (`full / metadata / off`), so proprietary architectures aren't stored |

### Customer journey
| When | What happens | What the customer sees |
|---|---|---|
| Day 0 | Base-URL swap + trace import | Nothing changes in their workflow |
| Night 1 | The agent runs on the frontier model through Mubit; tasks fill up | Three research tasks appear in the console |
| Day 2 | Adapters pass grading; routing begins | Step latency drops, GPU idle % falls |
| Week 1 | Corrections and outcomes produce v2, v3… | **$-to-target report: less GPU money, better models** |
| Ongoing | Weekly report: GPU $ saved, quality gained | Expansion to more teams and tasks |

This is Anil's **time to visibility**: value on Day 2, no forward-deployed engineers.

### Go-to-market
- **The wedge already exists:** lab researchers already drive experiments with Claude Code or Codex, which Mubit supports. *"Point your research agent at Mubit. Tonight's sweep runs on models trained from last night's."*
- **Pricing:** usage-based per gateway call (their model), plus a research tier priced against **GPU $ saved** (a share of the idle spend eliminated). Compute is the line item labs care about more than tokens.
- **Trust:** recording controls, owned adapters, open models (Sapling is Apache 2.0), and an independent verifier that catches contamination before a result reaches a paper or a launch.

### Commercial slide
```
Who pays        AI labs: RL/post-training & eval teams · research teams at mid-size labs and neolabs · AI-scientist startups
Why now         Agents now run the experiments; their GPUs sit idle while they think
Wedge           One base-URL change in the research agent labs already use (Claude Code / Codex)
Metric sold     $-to-target and GPU idle %, visible after one night
Moat for Mubit  Adapters trained on the lab's own research traces + an independent contamination verifier
```

## 6. Q&A prep (questions we expect, and our answers)
| Question | Answer |
|---|---|
| "ML models don't make LLM calls. How does Mubit improve them?" | It doesn't, directly. It improves **the agent that builds and tunes them**. Say "works on any model whose training you can script", never "improves every model". |
| "Is your small model better than Opus 5.5?" | Not per call: adapters are graded on *matching* the provider. The claim is **better at a fixed GPU/time/$ budget** (D3). |
| "Research calls aren't recurring." | The **steps** are: propose, diagnose, judge, with frozen prompts and fixed JSON. RL judging is the most recurring call in ML. Recurring steps power non-repeating discovery. |
| "Why an LLM agent at all? Optuna / Ray Tune / W&B Sweeps are free." | The biggest threat. Agents change *code, architecture, data mixtures and recipes*, not just numbers. **The benchmark includes an Optuna baseline**, and we report it even where it does well. |
| "LLM cost is small next to the GPU bill." | True for long runs. Our value is **idle GPU time and throughput** in short-iteration loops: sweeps, ablations, speedruns, RL tuning. That's why runs are 20–90 s. |
| "Long training runs barely wait on the LLM." | Correct; we say so. We win most in fast-iteration loops, where labs run thousands of short experiments. |
| "Why not just call Opus in parallel?" | Parallel calls cut idle time but cost scales linearly with calls, and each proposal depends on the latest results, so the loop is inherently sequential. Both arms get **the same concurrency and batching** (K proposals per call). |
| "Did you handicap the Opus arm?" | No: it gets every optimisation ours gets (caching, batching, structured outputs, effort tuning). "We even made the baseline cheaper and still won." |
| "Labs won't send traces." | Recording controls (`metadata`/`off`), owned adapters, open models. That's the "own your intelligence layer" track. |
| "Market size?" | Compute is the largest cost line in every AI lab, mostly spent on experiments (Epoch AI, **verify before quoting**). RL judge volume grows with every post-training run. |
| "How many seeds? On what benchmark?" | ≥3 seeds per arm, mean ± std, public speedrun task families with their own held-out metrics, every number traceable to a Mubit request ID and a Modal run ID (`benchmark.md`). |
| "Which layers of Sapling did you train?" | Exactly what we did, no more (wording table in `sapling_research.md`). |
| "Isn't this what Rekursiv does?" | They compete on better research. We make *any* research agent cheaper per discovery, via the agent's own calls. |
