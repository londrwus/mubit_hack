# Pitch: who we sell to, what we say, how we answer

**Stage rule (D19):** everything said on stage and shown on screen must make sense to someone who has never trained a model. Technical depth is for Q&A, the results export and the docs. Dashboard copy and demo narration are in `demo.md`.

## 1. The customer: AI labs only (D18)
Banks, fraud and generic enterprise ML are out of scope for the pitch, the targets and the demo.

### Personas
| Persona | Their day | Their pain | What we show them |
|---|---|---|---|
| **Research engineer** (pretraining / architecture) | Agent-driven sweeps and ablations overnight | GPUs idle while the agent thinks; slow iteration | The same model for a quarter of the AI bill on two tasks, 3 runs each: game-playing (R78, a tie; GPUs waiting 1.5% vs 68.8% as the reason) and tabular tuning (R82, within 0.0013, the standard agent slightly ahead in every run), always beside the first tabular race, which failed (R75: 8.2× lower AI bill, 7% more errors; R82's setting was chosen after it); the image-recognition and game-playing gains (R23, R53; the language model showed no gain) |
| **RL / post-training lead** | Reward models, LLM judges, rollout filtering | Judge cost and latency; judges get gamed and go stale | Distilled judges that refresh from corrections (adapter v1 → v2; no v2 trained yet, and every judge adapter is below target: v0 42.9%, v1 11.4% in B and 58.3% in workspace C, vs 95%, R30, R70, R79) |
| **Eval lead** | Benchmarks, leaderboards, release gates | **Contaminated or overfit "gains"** reaching papers and launches | A checker that let 0 of 32 planted fakes through (R37), and Sapling as an offline filter (R42) |
| **Head of compute / research ops** | GPU budget and allocation | Paying for idle GPUs; no "cost per result" visibility | The cost-to-goal dashboard: cost per research result, not per GPU-hour |

### Segments, ranked by how much Mubit should care
Mubit's business is usage through the gateway → owned adapters per recurring task → retention (the models are trained on *your* traces). It wants **high-volume, recurring, verifiable LLM calls**. AI-lab research loops have that shape.

| # | Segment | Recurring LLM volume | Willingness to pay | Verdict |
|---|---|---|---|---|
| 1 | **RL / post-training and eval teams** (LLM-as-judge, reward grading, rollout filtering) at mid-size labs, fine-tuning companies and RL-environment vendors | **Enormous**: judge calls on every rollout batch (industry estimate, verify before quoting a number), fixed rubric and output | High: judge cost and latency slow training | ⭐⭐⭐⭐⭐ The highest-volume wedge: distilled judges plus a Sapling-style independent verifier |
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
> ### 🏆 Get a better model · 💸 Spend less · ⚡ Train faster

Order per D48: a better model and less money lead; GPUs waiting is *why it works*, not a promise of its own. "Spend less" means the **total** bill (GPU + the assistant's own AI calls): on short runs the AI calls are most of it (90.6% on the game-playing task, R54), and the GPU $ lost to waiting is small next to them (R54), so never sell it as "spend less on GPUs".

Every slide, screen and sentence supports one of these. If it doesn't, cut it. Each is tied to a measured metric, and we never claim one we didn't measure:

| Promise | Metric that proves it |
|---|---|
| 🏆 Better model | best held-out score at the standard assistant's spend (same money) |
| 💸 Less money | total $ (GPU + AI) to reach the standard assistant's quality (same quality); quality per $ |
| ⚡ Hit the target faster | time-to-target |
| Why it works (not a promise) | GPUs waiting % |

### Why "better model" follows from "faster and cheaper"
> **"Because the GPUs wait less, more of your budget goes into actual training, and more training means a better model."**

Faster thinking → more experiments on the same GPUs → more chances to find better settings → a better final model. This is the chain we expect; the race **measures** it (same budget, same time, compare quality). **What the races show so far (Busy Machines vs the standard Opus agent):**
- **Game-playing AI, 3 runs, headline-eligible (R78):** the chain held for cost, not for quality. GPUs waiting 1.5% vs 68.8%, 10× the experiments per $, and **the same model (a tie, −154.3 vs −154.5) for 4.0× less AI money** ($0.83 vs $3.30 per race) and 3.1× less in total ($1.18 vs $3.65). Not a better model.
- **Tabular tuning, pre-registered follow-up, 3 runs, headline-eligible (R82, D49): success line MET (4 of 4).** Practically **the same model: 0.8884 ± 0.0017 vs 0.8897 ± 0.0013**, within 0.0013 (about 1% more errors), but **the standard agent was ahead in every run** (−0.0001, −0.0008, −0.0030). **AI bill 0.25×** ($0.698 vs $2.774 per race, 4.0× less) and 35% less in total ($3.89 vs $5.97, GPU $ virtual). Never reached the standard agent's final best (0/3), and to 0.86 it was slower and dearer ($1.14 / 4.2 min vs $1.08 / 2.9 min). Its planner pace (every ≥30 s) was **chosen after R75 failed** and pre-registered before the race ran.
- **Tabular tuning, the first pre-registered race, 3 runs (R75): FAILED its success line.** Planner every ≥90 s: AI bill 8.2× lower, but **7% more errors** (0.8816 vs 0.8892) in every run. Always said beside R82.
- **Without a pace limit (R67, 2 runs):** a better model (−153.9 vs −156.2), but 2.1× the AI bill.

So on stage: "same model for a quarter of the AI bill" on two tasks (R78 a tie, R82 within 0.0013 with the standard agent ahead in every run), always with R75 beside R82 and the fact that R82's setting was chosen after R75 failed. "A better model" stays a promise until a logged race shows it without a higher bill.

We expect better quality for two reasons, and the race measures both: **more experiments in the same budget**, and **verified results** (the checker rejects fake gains, so the "better" number is real). Note that more experiments aren't the whole story: on LCBench the Opus agent beat Optuna on quality while running 0.3× the experiments per dollar (R38, repeated in R44), and on image recognition at 0.5× (R53).

### One curve, two wins
- **Fix the budget → better quality** (best held-out score at the same budget).
- **Fix the quality target → less money** ($-to-target).

> **"A better model for the same money, or the same model for less money. You choose."**

This answers Anil's objection ("you're not saving money, you'll burn those tokens on something else"). We agree: the saved GPU money **becomes more experiments, which becomes better quality.**

### Scope (say it precisely; it makes the claim defensible)
We cut cost and improve results in the **experimentation layer**: hyperparameter sweeps, ablations, architecture search, RL environment and reward tuning, small-scale research runs. That's the work that decides what the big run looks like. We do **not** claim to speed up a single giant pretraining run.
> Hook, **verify before quoting:** Epoch AI estimated most of OpenAI's 2024 R&D compute went to experimental and research runs, not final training runs. If confirmed: *"Most of a lab's GPU bill is experiments. That's exactly where we work."*

## 3. Scripts
**Claims rule for every script.** The three promises are what Busy Machines is built to do. The measured result is narrower: **the same model for a quarter of the AI bill on two tasks**, game-playing (R78, a tie) and tabular (R82, within 0.0013, the standard assistant ahead in every run), plus a **failed** first pre-registered tabular race (R75) that R82 was chosen after. Say the promises as promises, say R78 and R82 with their tasks and their worse sides, say R75 in the same breath as R82 (with "chosen after"), and take every number from §7 only. Never say the small model is answering, or that a correction has already changed a model (`demo.md` §2).

### 20 seconds (plain, for anyone)
> "AI companies want better models for less money. Today an AI assistant runs their training experiments, and it's slow and expensive to think: while it plans, the machines wait, and its own thinking can be most of the bill.
>
> Busy Machines is built to get **a better model for less money** from the same assistant, by keeping the machines busy while it thinks."

### 1 minute (plain)
> "Training AI models is mostly trial and error: try a setting, train, check the result, try again. Companies now let an AI assistant do that loop. The problem: every time the assistant thinks about what to try next, **the expensive GPUs wait. You're paying for machines that do nothing.**
>
> Busy Machines is built to fix that, with the same assistant: shorter plans, a simple no-AI tuner that keeps the machines busy while it thinks, and a checker that throws out fake improvements. Every call goes through Mubit, so each recurring decision can become a small, fast model trained on its own work once it passes Mubit's exam (none has yet).
>
> Same model, same machines, same money: **the race shows who gets the better model and who spends less.** On two tasks, three runs each, Busy Machines got the same model for a quarter of the AI bill: a tie on game-playing, and on tabular tuning within a fraction of a percent, though the standard assistant was slightly ahead every time. Our first tabular race failed: 8 times cheaper on AI, but a worse model. We changed the planner's pace after seeing that, pre-registered the retry, and it passed. We report both. And when it makes a mistake, you flag it once: the correction is stored against that exact Mubit call."
>
> (R78: $0.83 vs $3.30 AI $ per race, best −154.3 vs −154.5, a tie; R82: $0.698 vs $2.774, best 0.8884 vs 0.8897, Opus ahead in all 3 runs; R75: $0.347 vs $2.831, 0.8816 vs 0.8892. AI $ are list-price estimates; tabular GPU $ is virtual.)

### 30 seconds (for Mubit judges: cost and efficiency, in Anil's words)
> "Anil says only two things matter: **cost and efficiency.** On our short-run game-playing task, 90% of the standard assistant's bill was its own thinking.
>
> Busy Machines runs the same Opus 5.5 through Mubit, with our engine: Opus plans less often, a no-AI tuner keeps the machines busy in between, and a checker blocks fake results.
>
> Two tasks, three runs each: **the same model as the standard Opus assistant, for a quarter of the AI bill.** On game-playing, a tie. On tabular tuning, within 0.0013, though the standard assistant was slightly ahead in every run.
>
> That tabular result was our second pre-registered try: we changed the pace after the first one failed, eight times cheaper on AI but 7% more errors. We report both.
>
> Same model. Same machines. A quarter of the AI bill."

(~125 words, about 45–50 s at a normal pace; if it must be 30 s, cut the second paragraph to "Busy Machines runs the same Opus 5.5 with our engine, through Mubit." Never cut the first tabular fail (R75), "changed the pace after" or "slightly ahead in every run". Optional mechanism line if there is time: "On game-playing its machines waited 1.5% of the time, not 68.8%" (R78).)

Numbers in this script (D48 order: same quality for less money first, waiting as the mechanism): **R54** (on the game-playing task 90.6% of the standard assistant's spend was its own thinking; so the script names that task; it is 63.6% on image recognition and 30.8% on tabular with simulated GPU $ (R54; in R82 $2.774 of $5.97, also simulated GPU $); on long runs we expect the GPU share to dominate, not measured), **R78** (game-playing, 30 machines, 15 s runs, 3 runs, headline-eligible: best −154.3 ± 0.5 vs −154.5 ± 0.6, a tie within noise; AI $ per race $0.83 vs $3.30, 0.25×, 75% less; GPUs waiting 1.5% vs 68.8%; Opus plans less often: 120 AI proposals vs 500 experiments on the standard side; 0 of 847 trap runs accepted as best), **R82** (tabular, pre-registered follow-up with the planner every ≥30 s, success line MET, headline-eligible: best 0.8884 ± 0.0017 vs 0.8897 ± 0.0013, within 0.0013 but Opus ahead in every run, per run −0.0001, −0.0008, −0.0030; AI $ per race $0.698 vs $2.774, 0.25×; total $3.89 vs $5.97, −35%, GPU $ virtual; chosen after R75 failed), **R75** (tabular, the first pre-registered race, FAILED: AI $ $0.347 vs $2.831, 8.2× lower; best 0.8816 vs 0.8892, 7% more errors). AI $ is a list-price token estimate. Backup numbers if there is time: total $1.18 vs $3.65 (3.1× less, in-race GPU only); 10× the experiments per $ (1,376 vs 137). **Say beside them when asked, the first one first:** in 1 of 3 game-playing runs a burst of 143 checks cost $0.56 and the AI bill was 0.34×, not 0.21×; BM reached the standard assistant's final quality in only 2 of 3 runs ($0.46 vs $3.10); the no-AI tuning tool reached the easy −170 goal cheapest ($0.009 vs BM $0.022 vs Opus $0.20) but had a worse model (−157.3) and accepted 53 fakes as its best, and its numbers in BM races carry the R74 caveat (shared process); total $ counts in-race GPU time only (Modal metered ~$2.5 per race vs $1.05 in-race GPU, pre-warm and idle containers, same for all arms, R78); the tabular races' GPU $ is virtual (simulated, R38, D47); in R82 Busy Machines never reached the standard assistant's final best (0/3) and was slower and dearer to 0.86 ($1.14 / 4.2 min vs $1.08 / 2.9 min), its pass rests on 3-run means with run 3 at the margin (−0.0030 vs the −0.003 line), and its 30 s pace was chosen after seeing R75 (pre-registered before it ran, the last pace setting tried, D49); the no-AI tool had the lowest total in R82 ($3.20) but a worse model (0.8576) and shipped a fake in 1 of 3 runs (R74 caveat). Earlier pieces: lean plans alone R63; backfill pilot R65; no pace limit R67 (better model, 2.1× the AI bill). The Mubit sentence was cut from this version for time; if added back it is a promise ("each step can become a small, fast model once it passes Mubit's exam; none has yet", R70). Never say "Mubit beats Opus" (no adapter has served, R46, R70) and never "a better model" for R78 (a tie) or R82 (Opus ahead in every run). The no-AI baseline (Optuna) is "a standard no-AI tuning tool", second slide or Q&A (D46). Full stage script: `demo_script.md`.

### One-sentence versions
- **Problem-first:** "AI agents now run ML experiments, but every time they stop to think, your GPUs sit idle and you pay for it. Busy Machines keeps them busy, so you hit your target faster, spend less, and end up with a better model."
- **Result-first** (measured, R78 and R82 with R75 beside them): "On the same machines with the same Opus 5.5, Busy Machines got the standard assistant's model for a quarter of the AI bill on two tasks, three runs each: a tie on game-playing, and within 0.0013 on tabular, where the standard assistant was slightly ahead every time. Our first tabular race, before we changed the pace, failed: eight times cheaper on AI but 7% worse, and we report that too." (Not "a better model": R78 is a tie, R82 is slightly behind in every run.)
- **Mechanism-first (Mubit judges):** "Busy Machines keeps an ML agent's machines busy while it thinks, on the same model, and routes every call through Mubit so its recurring decisions can become small models once they pass: results sooner, budget further, and every check we record becomes training for tomorrow's model." (No adapter has passed yet, R70; a correction is stored against its request ID, but we don't know whether it changes training, Q2/Q9.)

### Lines
- **Closers:** measured (R78, R82): "Same model. Same machines. A quarter of the AI bill." · "Cost and efficiency, measured and logged, including the race we lost (R75)." Promises only (no logged race shows a better model without a higher bill): "Same GPUs. Less money. Better models." · "Cost down. Quality up." (R78 ran on real Modal CPU workers; R75's and R82's GPU time is simulated, D47, D49.)
- **Taglines:** "Spend less. Wait less. Discover more." · "Your GPUs should never wait for your agent." · "Every GPU-second spent on science, not on waiting." · "Same budget. More discovery." · "Better models. Smaller GPU bills." · "Stop paying for idle GPUs. Start paying for results."
- **Openers:** "Look at these GPUs. Grey means idle. You're paying for every grey second." · "This is an Opus 5.5 researcher. It's brilliant. And every time it stops to think, its GPUs wait." (quote the measured GPUs-waiting % only, with its setting: 69.0 ± 0.4% with 30 workers and 15 s runs, R48; 10.9 ± 1.6% on image recognition with 3 L4s, R53; 4.6–5.0% on the tuning task, R38, R44. "Most of the time" is supported only for the 30-worker, 15 s setting) · "Two AI researchers. Same GPUs, same budget, same clock. One change."
- **Audience variants:**
  - Research teams: "More experiments per GPU-hour means more discoveries per quarter, on the compute you already have."
  - RL / post-training: "Your reward and eval loops run millions of judge calls; we're built to make them fast, owned and refreshed from new examples." (Promise; "millions" is an industry estimate, verify before quoting.)
  - Eval teams: "An independent verifier that catches contaminated gains before they reach a paper or a launch."
  - Mubit judges: "Agents wait less and do more, measured and logged. Every check we record becomes training for the next model."
- **latte/bench line:** "latte/bench asks whether agents learn. We measure what that learning is worth: dollars and hours to a real research result."

### Anil's problems → our answer → the proof
| Anil said | The problem | Busy Machines answer | Metric |
|---|---|---|---|
| "Token efficiency… makes your agents **wait less and do more**" | Agents waste time waiting on slow LLM reasoning | Our engine on the same Opus (lean planner, backfill, cadence) cuts the waiting today: 68.8% → 1.5% on the game-playing task, 3 runs (R78, with a tied model, not a better one); on tabular 30.3% → 1.1% with the 30 s pace (R82, the standard agent's model slightly better in every run) and 29.4% → 2.5% with the 90 s pace (R75, 7% more errors, the race failed); recurring decisions move to Mubit adapters once they pass the exam (none has yet: R28, R30, R70) | LLM latency per step, experiments/hour |
| "Combine this with **parallelism and cloud agents**, you have the **full utilisation cycle**" | GPUs sit idle while the agent thinks | The agent keeps N GPUs busy in parallel | **GPU idle %** |
| "After all the shiny aspects… **only 2 things matter: cost and efficiency**" | Research budgets are GPU-bound | Less idle spend; savings reinvested in experiments | **$-to-target**, idle GPU $ |
| "Avoid layers that could **add bias**… verifier with shared context" | Agents fool themselves with contaminated or overfit "gains" | **Independent verifier**, no shared context: in races, Opus checks for leaks and a seed-spread rule checks for noise (R37); Banyan Sapling offline only (R42, D28) | Contamination-trap accuracy |
| "A herculean task to manage a **farm of models which continuously evolve**… **low latency**" | Keeping task models fresh and fast | Each correction is stored against its request ID, for the next adapter version through Mubit (no corrections API yet, Q2/Q9) | Adapter v1 → v2 quality (not measured yet) |
| "**Time to visibility**" | Value is unclear for weeks | Dashboard shows cost, GPUs waiting and model quality **live, from the first night** | The live race |

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
> **"We built the AI-lab product you can sell on Monday. Same base URL, three research tasks, and a GPU-efficiency report from the first night."**

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

This is Anil's **time to visibility**: value on Day 2, no forward-deployed engineers. It is the intended journey, not our measured one: in our own run no adapter passed its exam by Day 2 (R28, R30), so the provider still answered every call.

### Go-to-market
- **The wedge already exists:** lab researchers already drive experiments with Claude Code or Codex, which Mubit supports. *"Point your research agent at Mubit. Tonight's sweep runs on models trained from last night's."*
- **Pricing:** usage-based per gateway call (their model), plus a research tier priced against **GPU $ saved** (a share of the idle spend eliminated). Compute is the line item labs care about more than tokens.
- **Trust:** recording controls, owned adapters, open models (Sapling is Apache 2.0), and an independent check for contamination before a result reaches a paper or a launch.

### Commercial slide
```
Who pays        AI labs: RL/post-training & eval teams · research teams at mid-size labs and neolabs · AI-scientist startups
Why now         Agents now run the experiments; their GPUs sit idle while they think
Wedge           One base-URL change in the research agent labs already use (Claude Code / Codex)
Metric sold     Model quality per $ and cost to reach the goal (GPUs waiting as the why), visible after one night
Moat for Mubit  Adapters trained on the lab's own research traces + an independent contamination verifier
```

## 6. Q&A prep (questions we expect, and our answers)
| Question | Answer |
|---|---|
| "ML models don't make LLM calls. How does Mubit improve them?" | It doesn't, directly. It improves **the agent that builds and tunes them**. Say "works on any model whose training you can script", never "improves every model". |
| "Is your small model better than Opus 5.5?" | Not per call: adapters are graded on *matching* the provider. The claim is **better at a fixed GPU/time/$ budget** (D3). |
| "Research calls aren't recurring." | The **steps** are: propose, diagnose, judge, with frozen prompts and fixed JSON. RL judging is among the most recurring calls in ML (our reading; no number to quote). Recurring steps power non-repeating discovery. |
| "Why an LLM agent at all? Optuna / Ray Tune / W&B Sweeps are free." | The biggest threat. Agents change *code, architecture, data mixtures and recipes*, not just numbers. **The benchmark includes an Optuna baseline**, and we report it even where it does well. |
| "LLM cost is small next to the GPU bill." | True for long runs. In our short-run races it's the other way round: the Opus assistant's game-playing race cost $3.69, of which $3.27–3.40 was its own thinking (R48; R78: $3.30 of $3.65). So our value is **throughput and a cheaper planner**, not the GPU money lost to waiting, in short-iteration loops: sweeps, ablations, speedruns, RL tuning. That's why runs are 15–90 s. It's also where R78's saving comes from: Busy Machines' in-race GPU $ is the same as Opus's (~$0.35 each, by arithmetic: total $ minus AI $); the difference is the AI bill ($0.83 vs $3.30, list-price estimate). |
| "Long training runs barely wait on the LLM." | Correct; we say so. We win most in fast-iteration loops, where labs run thousands of short experiments. |
| "Why not just call Opus in parallel?" | Parallel calls cut idle time but cost scales linearly with calls, and each proposal depends on the latest results, so the loop is inherently sequential. Both arms get **the same concurrency and batching** (K proposals per call). Our simulator projects that 3 Opus calls at once would cut waiting to 6.0% but cost 3.55× the LLM bill per race (R52: a projection, not built or raced). |
| "Did you handicap the Opus arm?" | No. It gets the same caching, batching (K proposals per call) and structured outputs, the same model, machines, clock, $ cap, seeds and judge prompt. It is the plain v1 loop a lab would run. What differs is our engine (lean prompt at effort low, backfill, a planner cadence: ≥60 s in R78, chosen by an offline gate before the race; ≥90 s in R75, pre-registered; ≥30 s in R82, chosen after R75 failed and pre-registered before it ran; the checker policy), BM-only by design and disclosed per race (D46); each part's own effect is quoted from its own test (R63, R65, R67). Never say "and still won": R78 is a tie on quality, in R82 the standard arm was ahead in every run, and R75 lost on quality. Disclose if asked: an earlier engine version on plain Opus did **worse** than plain Opus on tabular tuning, 9% more errors (R44; the cause is unclear, R51), and without a cadence our engine doubled the AI bill on the game-playing task (R67). |
| "Labs won't send traces." | Recording controls (`metadata`/`off`), owned adapters, open models. That's the "own your intelligence layer" track. |
| "Market size?" | Compute is the largest cost line in every AI lab, mostly spent on experiments (Epoch AI, **verify before quoting**). RL judge volume grows with every post-training run. |
| "How many seeds? On what benchmark?" | ≥3 seeds per arm for any headline number, mean ± std, public task families with their own held-out metrics, every number traceable to a Mubit request ID and a runner ID (Modal call ID; surrogate ID for LCBench, where GPU time is simulated) (`benchmark.md`). Opus vs Optuna has 3 seeds on all three tasks: game-playing AI (R48), image recognition (R53), tabular tuning (R38, repeated in R44; same seeds on a deterministic surrogate, so never pool them as 6). Busy Machines with Opus vs Opus: 3 runs on game-playing (R78, headline-eligible; the 3 seeds pool under one fingerprint after D50, which drops the failure-only diagnose step from the pooling key), 3 runs on tabular (R75, pre-registered, failed), 3 runs of the pre-registered 30 s tabular follow-up (R82, headline-eligible, met its line; chosen after R75), and 2-run pilots (R65, R67). Busy Machines with Mubit's models: 0 races, because no small model serves yet (R46, R70). |
| "Why did the tabular race fail?" | "We fixed the success line before the race and missed it: Busy Machines' AI bill was 8.2× lower, but its model had 7% more errors, in all 3 runs (R75). Measured: its 90-second planner pace left only 80 AI proposals per race, and on that task the shipped model came from the AI planner in 3 of 3 runs, so the AI's ideas were what mattered and it had too few; the standard assistant also waited only 29.4%, not the 45.8% our offline gate projected (R66, flagged as fragile; the success line was ≥40%), so there was less idle time to fill. So we ran one follow-up, a 30-second pace, chosen after seeing the fail, pre-registered before it ran and reported beside it (R82): 240 AI proposals per race instead of 80, and it met its line: within 0.0013 of the standard assistant (0.8884 vs 0.8897), though the standard assistant was still ahead in every run, for 0.25× the AI bill ($0.698 vs $2.774)." GPU $ there is virtual. |
| "Did you pick the setting after seeing results?" | "For the tabular follow-up, yes, and we say so. The 30-second pace was chosen after the first pre-registered race failed (R75). We froze its setup and success line before it ran (`preregistration_lcbench_wide_c30.md`, D49), said in advance it was the last pace we'd try, and report R75 next to it every time. It met the line (R82). The game-playing pace (60 s) was chosen by an offline gate before its race (R78)." Say beside it: R82's pass is on 3-run means, and run 3 sat at the margin (−0.0030 vs the −0.003 line). |
| "Is it just Optuna doing the work?" | Say what is measured and what isn't. **Measured:** in R78 the no-AI tuner inside Busy Machines ran ~1,466 of 1,587 experiments (Opus proposed 120), and the shipped model came from the tuner in all 3 runs (R84); in R75 and R82 the shipped model came from the AI planner in 3 of 3. In all three races Busy Machines beat the plain no-AI tool on model quality (R78: −154.3 vs −157.3; R75: 0.8816 vs 0.8493; R82: 0.8884 vs 0.8576) and shipped 0 fakes where the unchecked tool let fakes through (R78: 53 accepted as its best, of 4,648 trap runs; R75: 19 of 618 trap runs; R82: 15 of 236 accepted as best, a fake shipped in 1 of 3 runs, vs BM 0 of 85); the tool's side in these races carries the R74 caveat (it shared a process with BM's tuner). Fairness note: in an earlier race without a BM arm, the no-AI tool alone got −155.2 ± 0.7 on the same game-playing setup (R48, a different race, not pooled), close to BM's −154.3. **Not measured:** a "tuner + checker, no AI" arm, so how much the AI's proposals steer the tuner is our design ("the AI picks the regions, the tuner fills the GPUs, the checker blocks the fakes"), not a measured split. Without the checker, the tuner finds the shortcut and floods the results (R64 replay: 97.7% trap share; R78 seed 2: 759 trap runs, 128 claims rejected, 0 shipped). |
| "Which layers of Sapling did you train?" | Exactly what we did, no more: "verification head fine-tuned, backbone frozen" (D23; wording tables in `sapling_research.md` and `sapling_results.md` §5). |
| "Did you just reuse Mubit's model?" | "No. We fine-tuned Banyan Sapling's verification head, backbone frozen, on AI-research claims, and refit its calibration for that domain. It is our community fine-tune, not an official MubitLabs model, and MubitLabs hasn't reviewed it. Against the published model on the same 500 research claims it had never seen: answers given went from 0% to 19% at the strict setting (64% at the looser one), fakes caught from 0 to 52 of 157 (118 looser), and fakes approved stayed at 0. On ContractNLI, Sapling's own domain, it went from 77.3% to 84.3% on our 400 cases, so there was no regression (R42)." **Add (R81):** "On 226 real claims from our races, the published model answered none; ours caught 57 of 109 fakes at the looser setting and approved none." **Say beside it:** Opus is still the better judge (50/50 on the trap set vs 18/50 strict and 36/50 looser, R42, R24; on real claims 109 vs 57). It confirms 0 of 132 genuine gains at the strict setting. At the looser setting its 95% guarantee falls to 77.8% on new claim types. The claims are synthetic. A v3 trained on race claims failed its gate, so we kept v2 (R77). Our ContractNLI set is not the card's: the card reports 86.32% on its own 731 cases. In races, Sapling is offline only: the live checker is Opus plus the seed-spread rule (D28, R37). Details: `sapling_vs_published.md`. |
| "Isn't this what Rekursiv does?" | They compete on better research. We make *any* research agent cheaper per discovery, via the agent's own calls. |

## 7. Measured lines we can say today
Lines 1–9 written Sat 24 Oct ~10:00 BST (log up to R53); the intro, the H lines and the "never say" list were updated ~18:00 BST (log up to R78) and again after R82 was logged (log up to R82: the D49 tabular follow-up added as H1b, R75 kept beside it). Under D46 lines 1–3 and 6 (vs Optuna) go on the second results slide or in Q&A only, as "a standard no-AI tuning tool"; the Busy Machines vs Opus lines (H1–H7 below) are also in `demo_script.md` §3 and `slides_outline.md` slides 4–5. Each line uses the fixed framing from `benchmark.md` §6b, cites its results-log entry, and comes with the worse result we say **in the same breath**. If a line is used on a slide, its "say beside it" goes on the same slide. "Three runs" means 3 seeds, mean ± std.

**First, what we can't say yet.** The headline (Busy Machines with Mubit's small models vs an Opus 5.5 assistant: cost to reach the goal, time to reach it, GPUs waiting, model quality at the same budget) is **not measured yet**. Mubit's small models all scored below their exam targets (planning 14.3% vs 95%, checking 42.9% vs 95%, toy task 70% vs 85%, and the closed-answer checker 11.4% on 35 test cases vs 95% after ~7 h of training, with workspace C's copy on the same data at 58.3% on 24 cases), the diagnose model failed its first training and scored 8.3% on 12 cases after Mubit retried it, and none has answered a live call [R7, R28, R30, R46, R70, R79, R80]. Our engine on plain Opus (Busy Machines with Opus, D46) **is** measured, as the headline comparison: a 3-run game-playing race (R78: same model, a tie, 4.0× less AI money), a pre-registered 3-run tabular race that **failed** (R75: 8.2× less AI money, 7% more errors), and its pre-registered follow-up, chosen after R75 (R82: within 0.0013 with Opus ahead in every run, 4.0× less AI money, success line met). Earlier: R44 (earlier engine, tabular) made the model **worse**, 9% more errors, 5% more expensive and 9% slower to the goal [the cause is unclear, R51]; R65 (tabular, 2 runs) reached the goal faster and cheaper with a slightly lower best model and a 1.19× AI bill; R67 (game-playing, 2 runs, no pace limit) found a better model faster but at **2.1× the AI bill**. Every line below uses Opus 5.5 as the assistant's brain, or no AI at all, so none of them is a "Mubit beats Opus" line.

**Busy Machines vs the standard Opus 5.5 assistant (the headline, D46/D48 order).** AI $ = list-price token estimate (R72). On game-playing, both sides run on real Modal CPU workers; on tabular, GPU $ is virtual. Any number for the no-AI tool from these races carries the R74 caveat (it shared a Python process with Busy Machines' tuner, so its waiting and experiment counts there are not isolated; its waiting may be inflated).

| # | Say on stage | Source | Say beside it (worse result, caveat) |
|---|---|---|---|
| H1 | "On a game-playing task, three runs: Busy Machines got **the same model** as the standard Opus 5.5 assistant, a tie within noise, for **a quarter of the AI bill**: **$0.83 vs $3.30** per race, 75% less, and **$1.18 vs $3.65** in total." | R78 (−154.3 ± 0.5 vs −154.5 ± 0.6; per run +0.14, −0.19, +0.51; same money at Opus's $3.65: −154.3 vs −154.5) | "A tie, not a better model. In one run a burst of 143 checks cost $0.56 and its bill was 0.34× Opus's, not 0.21×. It reached the standard assistant's final quality in only 2 of 3 runs ($0.46 vs $3.10); in the third it missed by 0.19. Total $ counts GPU time inside the race: Modal's meter showed ~$2.5 per race vs $1.05 in-race GPU (pre-warm and idle containers, the same for all arms)." |
| H1b | "On tabular model tuning, three runs, pre-registered: practically **the same model**, within 0.0013 (0.8884 vs 0.8897), for **a quarter of the AI bill**: **$0.70 vs $2.77** per race, and **$3.89 vs $5.97** in total." | R82 (headline-eligible; success line met 4 of 4: best ≥ Opus − 0.003, AI $ ≤ 0.4×, ≥ the no-AI tool, 0 fakes shipped) | "Not a better model: the standard assistant was ahead in **every** run (by 0.0001, 0.0008, 0.0030), about 1% more errors on our side, and run 3 sat at the margin of the line. It never reached the standard assistant's final best (0/3), and to 0.86 it was slower and dearer ($1.14 / 4.2 min vs $1.08 / 2.9 min). **This was our second try:** the first pre-registered tabular race failed (H4, R75), and we chose this 30 s pace after seeing that, then froze it before the race. GPU $ is virtual (simulated GPU time)." |
| H2 | "Why it works: its machines waited **1.5%** of the time; the standard assistant's, **68.8%**. So it ran **10× the experiments per dollar**: 1,376 vs 137." | R78 (1,587 vs 500 experiments) | "Most of them came from the no-AI tuner inside Busy Machines (~1,466; Opus proposed 120), and the shipped model came from the tuner in all 3 runs (R84). The plain no-AI tool ran even more per dollar (4,511) and reached the easy −170 goal cheapest ($0.009 vs $0.022 vs $0.20), but finished with a worse model (−157.3; R74 caveat)." |
| H3 | "Busy Machines' side ran a fake shortcut **847 times**, mostly from its tuner; the checker let **0** through. The plain no-AI tool, unchecked, accepted fakes as its best **53 times**." | R78 (0 of 847 trap runs accepted as best; seed 2 alone 759 sampler trap runs, 128 claims rejected; Opus 0 of 10; tool 53 of 4,648) | "Checked vs unchecked, not one checker vs another. Catching them cost money: one run needed 143 checks ($0.56), because checks aren't paced yet." |
| H4 | "We also pre-registered a tabular race, and **it failed**: Busy Machines' AI bill was **8.2× lower** ($0.347 vs $2.831 per race), but its model had **7% more errors** (0.8816 vs 0.8892), in all 3 runs. We report it." | R75 | "Also failed: the standard assistant waited 29.4% (line: ≥40%) and Busy Machines 2.5% (line: ≤2%). Passed: AI bill ≤0.8×, and better than the no-AI tool (0.8493). Total $6.03 vs $3.54 (−41%), GPU $ virtual. It never reached the standard assistant's final quality (0/3). One follow-up with a 30 s planner pace, chosen after seeing this and pre-registered before it ran, met its line: within 0.0013 at 0.25× the AI bill, the standard assistant still ahead in every run (H1b, R82)." |
| H5 | "Without a pace limit on Opus, the same engine found a better model on the game-playing task (−153.9 vs −156.2) but **doubled the AI bill** ($6.50 vs $3.07)." | R67 (2 runs) | "That's why the pace limit exists. With it (R78) the bill fell to 0.25×, and the model became a tie rather than better. Different races, not pooled; R67 is not headline-eligible." |
| H6 | "Shorter plans alone, on tabular tuning: a model at least as good (0.866 vs 0.849), at 10.5% less AI money per experiment and half the output per proposal." | R63 (3 runs; probe R62: −71 to −73% output tokens) | "Dev race (the arms use different prompts). Pooled over 9 earlier runs the standard assistant scores 0.854, so 'at least as good', not a proven gain." |
| H7 | "The no-AI tuner filling idle machines, on tabular tuning: the goal in 1.2 vs 2.5 minutes, for $0.36 vs $0.74." | R65 (2 runs) | "Its best model was slightly worse (0.859 vs 0.861, within noise) and its AI bill 1.19× higher (no pace limit). 2 runs, not headline." |

**Second results slide or Q&A: the standard assistant vs a standard no-AI tuning tool, and other measured lines.**

| # | Say on stage | Source | Say beside it (worse result, caveat) |
|---|---|---|---|
| 1 | "An AI research assistant running on Opus 5.5 (the standard assistant in our race, not yet Busy Machines) beat Optuna, the standard no-AI tuning tool, on model quality in 2 of 3 tasks and tied on the third. Run three times each. **Tabular model tuning: 28% fewer errors. Image recognition: 22% fewer errors. Game-playing AI: a tie.**" | R38 (repeated in R44), R53, R48 | "Optuna ran far more experiments per dollar on every task (80.9 vs 22.7 on tabular, 81.2 vs 38.7 on images, 4,585 vs 137 on the game). On the game-playing AI its best model is slightly better (−155.2 vs −156.9 penalty, within noise), and it reached the goal far sooner and cheaper ($0.007 / 0.3 min vs $0.222 / 1.1 min) because the goal was too easy. GPU time on the tabular task was simulated, and Opus may have seen that task while it was trained." |
| 2 | "Left unchecked, the no-AI tuner accepted **fake improvements** as its best on every task: **36** times on the game-playing AI, **15** on image recognition, **8** on tabular tuning. The Opus 5.5 assistant: **0**." | R48 (36 of 4,661 shortcut runs vs 0 of 15), R53 (15 of 118 vs 0), R38 (8 of 62 vs 0 of 33) | "Optuna has no checker by design, so this is checked vs unchecked, not one checker vs another (that test is line 5). On image recognition the assistant never tried the shortcut (0 runs), so its checker wasn't tested there." |
| 3 | "On image recognition, the model the no-AI tuner handed over scored **89.6%** on fresh images; the assistant's scored **94.3%**. On the game-playing AI, the tuner's final model **missed the goal on fresh data**; the assistant's didn't." | R53 (0.8955 ± 0.016 vs 0.9431), R48 (−177.1 ± 8.8 vs −156.9; goal −170) | "On the game-playing AI, Optuna's best clean model (−155.2) is slightly better than the assistant's; it shipped a worse one because ~97% of its runs used the shortcut. Same checked-vs-unchecked caveat as line 2." |
| 4 | "With 30 machines and 15-second runs, the standard AI assistant kept its GPUs **waiting 69% of the time**. The no-AI tool: under 1%." | R48 (69.0 ± 0.4% vs 0.9 ± 0.1%; R41: 70.0%, 1 run) | "With longer runs and fewer machines the waiting is small: 10.9% on image recognition (R53), 4.6–5.0% on tabular tuning (R38, R44). The GPU money lost to waiting is small next to the assistant's own bill (R48: $3.69 per race, of which $3.27–3.40 is thinking). Busy Machines cut the waiting to 1.5% vs 68.8% on the same setup, 3 runs, at 0.25× the AI bill, with a tied model (R78; H1–H2)." Say it as the problem and the mechanism, never as our headline. |
| 5 | "Our checker confirmed **all 7 real improvements**. Opus on its own confirmed **1**. Neither let a single fake through: **0 out of 32**." | R37 (cases from R34) | "Only 7 real cases, so we quote counts. Tested offline on stored verdicts, not yet live in a race. Ours accepted 1 of 36 lucky results as real; Opus accepted 0. The run-to-run check alone would let 24 of 32 fakes through, which is why Opus still does the fake check." |
| 6 | "On image recognition, the assistant hit the 94% goal in **all 3 runs**, in about 2 minutes for about 23 cents. The no-AI tuner never hit it." | R53 ($0.229 ± 0.077, 2.0 ± 0.6 min, 3/3 vs 0/3) | "The assistant's race cost $1.65 vs the tuner's $0.60, it ran half the experiments per dollar, and its GPUs waited 10.9% vs 0.0%. Some of its replies were malformed and retried (8.1% in one run)." |
| 7 | "The assistant improves real models. Image recognition: **22% fewer errors** than the starting code. Tabular tuning: **74% fewer**. The game-playing AI's penalty: **5.9× smaller**." | R53, R38 / R44, R23 | "On the language model it found **no real gain** (R23). The no-AI tool also improved the tabular model (64% fewer errors) but barely moved image recognition (1%). An earlier, bigger image number (18%) was one lucky run, so we dropped it." |
| 8 | "Ask Opus for 1 experiment and it plans in about **3 seconds**. Ask for 8 and it takes about **13 seconds**. The more GPUs it has to feed, the longer they wait." | R49 (p50 3.15 s at 1), R48 (p50 13.1 s at 8, 192 of 192 calls) | "Measured fit: about 1.6 s plus 1.6 s per experiment (R50). A smaller model in the same loop (Haiku, not a Mubit model) was only ~28% faster per call, because it writes longer replies (R50)." |
| 9 | "We fine-tuned the verification head of Mubit's open fact-checker, Banyan Sapling, for AI research, with the backbone frozen. On 500 research claims it had never seen, it **catches 52 of 157 fake improvements the original caught none of, and approves none of them**." | R42 (α 0.01; supersedes R33) | "Opus is still a far better judge: 50 of 50 on our trap set, vs 18 of 50. At the strict setting it confirms none of the 132 genuine gains. At the looser setting it catches 118 and confirms 17 genuine gains, but lets 1 trap-set fake through, wrongly rejects 43 genuine gains, and its 95% guarantee falls to 77.8% on new claim types. The claims are synthetic, from our own generator. Mubit's hosted version takes ~1.1 s per check; 75 ms is on our own H100 (R42); the released package measured 99.4 ms median per claim on an H100 (R76)." In Q&A, say exactly "verification head fine-tuned, backbone frozen" (D23). |

**Q&A only (never on a slide):**
- "A smaller model in the same loop, **not a Mubit model**, left GPUs waiting less: 56.9% vs 71.0%, for $0.16 vs $3.27 of LLM spend, best −161.4 vs −165.1." [R50, 1 run] Say beside it: Opus reached the goal first (44 s vs 86 s), and Haiku was only ~28% faster per call.
- Projections, labelled as projections: a 2-second planner → 15.2% waiting at 30 machines (R43); 3 Opus calls at once → 6.0% waiting at 3.55× the LLM bill (R52, failed its cost gate, not built).

**Never say (yet):**
- "Mubit beats Opus", or any Mubit-model number as cheaper or faster than Opus: no adapter has served (R46, R70; §6c columns are empty). R78's saving is our engine on plain Opus, never Mubit's models.
- "Busy Machines gets a better model" as a measured result: R78 is a tie, R82 is within 0.0013 but the standard assistant wins every run, R75 is 7% worse, and the only better model (R67) came with 2.1× the AI bill on 2 runs.
- "4× cheaper" (or "3.1×", "75% less", "35% less") without the task and the fail: the 4× is the game-playing task (R78) and the 30 s tabular follow-up (R82); the first pre-registered tabular race failed (R75), and R82 was chosen after it. Always "AI bill", never "GPU bill": in-race GPU $ was the same on both sides in R78, and tabular GPU $ is virtual.
- R82 as "reached the standard assistant's quality" (0 of 3 runs) or "faster to 0.86" (it was slower and dearer: $1.14 / 4.2 min vs $1.08 / 2.9 min); R82's −35% total without "GPU $ virtual".
- R78's total $ as the full Modal bill (it counts in-race GPU only; metered ~$2.5 per race vs $1.05 in-race, R78).
- The R78 "same quality for $0.46 vs $3.10" without "in 2 of 3 runs".
- The no-AI tool's waiting or experiment counts from BM races (R65, R67, R75, R78, R82) without the R74 caveat.
- R82 without R75 beside it and without "chosen after R75 failed", or R75 without saying the follow-up was chosen after seeing it.
- R65, R67 (2-run pilots) as headlines; R44 (earlier engine) was worse than plain Opus. R41's Busy Machines arm is invalid (broken stopping rule).
- "Most of the time" (GPUs waiting) without naming 30 machines and 15-second runs (R48).
- "Spend less on GPUs", "Stop paying for idle GPUs" or "Every grey second is paid for" as a measured result: on our races the GPU money lost to waiting is small next to the assistant's own LLM bill (R48, R54), and in R78 the in-race GPU $ was the same on both sides; the saving was the AI bill. They are promises (§3).
- Opus's "0 fakes" on image recognition as a checker win: it ran 0 shortcut runs there (R53).
- R38 and R44's Optuna runs pooled as "6 runs": same seeds on a deterministic surrogate.
- The R36 numbers (superseded by R48), anything from R22 (superseded by R23), R33 (superseded by R42), R35 (folded into R38), or the R27/R43/R52/R64/R66 projections or R78's offline-gate projection as results.
- Haiku's result (R50) as Mubit's model or as Busy Machines.
- "Opus rejects real wins". Fair form: "Opus on its own confirmed 1 of 7 real improvements" (R34).
- For Sapling: "answers instead of unsure" or "beats Opus" (R33, R42, `sapling_results.md` §5). Also never "matches Opus" or "confirms real gains", never call our fine-tune "Mubit's model" or official, and quote the 226-row real-claims numbers only as logged in R81 (`sapling_vs_published.md`).
