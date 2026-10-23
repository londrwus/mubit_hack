# Keeping the broad pitch, and making it technically true

## The problem with "we improve every ML/DL/RL model by taking their LLM calls"
ML, DL and RL models **don't make LLM calls**. A fraud XGBoost model, a CNN or a PPO policy has no LLM traffic for Mubit to route. A technical judge (criterion: "holds up under questions") will spot this in the first question. As worded, the claim fails.

**What *does* make LLM calls:** the **AI agent that builds and tunes those models**, such as AutoML agents, "AI scientist" loops, and Claude Code or Codex doing ML engineering. That is where the broad claim becomes true.

## The reframed pitch (still simple, still broad)
> **"Night Scientist: an ML research agent that improves any model you give it, and gets faster and cheaper at it every run, because its own thinking is distilled by Mubit."**

The pitch has two parts:
1. **The agent improves *your* model.** This works for any ML, DL or RL model. It proposes experiments, runs them, reads results, diagnoses failures and iterates. This is the broad, impressive part you want.
2. **Mubit improves *the agent*.** Its inner loop is full of **recurring, fixed-format calls**, and these are exactly the units Mubit learns from:
   - `propose_next_experiment` → JSON config
   - `diagnose_run` (logs → `{failure_class, cause, fix}`)
   - `summarize_result` → structured finding
   - `judge_improvement` (did this actually help, or is it overfitting or leakage?)

   Each one becomes an adapter. Each researcher override (for example, "no, that's data leakage, not a real gain") is a **correction → model update**, which is the mandatory demo beat.

## Why this beats both earlier versions
- **Broad:** "any model, any lab, any bank" stays true.
- **Honest:** Mubit is used exactly as designed.
- **Anil's thesis:** agents waiting less means more experiments per night. Discovery loops are throughput-bound, so the token-efficiency argument fits directly. The `judge_improvement` verifier is kept context-isolated from the proposer, which is his bias point.
- **Novel:** Mubit pitches enterprise agents. AI-for-ML research loops are a buyer they haven't addressed.

## Demo plan (3 minutes)
1. **Target:** a bank fraud model on a public dataset (Kaggle credit-card fraud or IEEE-CIS). This keeps your bank story and trains in seconds on CPU. As a second target, show a small RL environment (CartPole) to prove "any model".
2. **Overnight run:** the agent ran N experiments on Friday night through Saturday. Show the fraud model's PR-AUC climbing.
3. **Correction live:** the agent claims a big gain that is actually leakage (for example, it used a future-timestamp feature). A researcher flags it, and `judge_improvement` v2 (already trained) now rejects it. Show both versions in the Mubit console.
4. **The chart:** frontier-only agent vs Mubit-adapted agent, comparing experiments per hour, $ per experiment, and final model quality. **Headline: "same discoveries, X× more experiments per hour."**

## Claims to avoid
- "Better than Opus 5.5." Say instead: "matches frontier on these steps, at a fraction of the latency and cost."
- "Improves every model." Say instead: "works on any model whose training you can script, shown here on fraud and RL."

## 42-hour plan
| When | What |
|---|---|
| Fri 19:30–23:00 | Agent loop (4 fixed prompts), fraud training harness, everything routed through Mubit. Freeze the prompts. |
| Fri 23:00–Sat 08:00 | Run the agent overnight with the frontier model to generate ≥25 kept examples per task. Trigger the adapter v1 trainings (3 starts per day). |
| Sat | Confirm with Mubit office hours how corrections enter training. Build the leakage trap and collect corrections, then train v2. Add CartPole. |
| Sat night | Benchmark runs: frontier vs adapters. Build the chart. |
| Sun 06:00–10:00 | Demo script, rehearse, write a fallback recording. |
