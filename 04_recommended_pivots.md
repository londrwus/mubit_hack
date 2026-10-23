# Pivots that keep your thesis and fit the rubric

## A (recommended): "Refresh-able reward judge": a distilled, independent verifier for RL / eval loops
**Track:** Fast refresh, which also scores on "Own your intelligence layer".

**The problem:** RL post-training and eval pipelines call a frontier LLM-as-judge millions of times. That is slow and expensive, and the policy learns to **reward-hack** the judge. The judge's blind spots shift as the policy improves, so a fixed judge goes stale. That is a changing workload by definition.

**The build (fits in 38 hours):**
1. Pick a small RL-style environment, for example code-fix or math answers with a rubric. The "policy" is a cheap model that generates rollouts.
2. The **judge call** goes through Mubit: fixed instructions and a fixed JSON output (`{score, verdict, reason}`), so it forms **one clean recurring task**. The frontier model labels the first ≥25 examples, and Mubit trains the adapter v1.
3. **Hacking appears:** the policy finds an exploit, such as hard-coding test outputs or a confident wrong format, and judge v1 approves it.
4. **Correction:** a *separate* verifier catches it. That can be a human, unit-test execution, or a context-isolated checker, which is Anil's "don't share context" point. Corrections are fed back as new examples, and Mubit trains **judge v2**.
5. **Live demo:** the same exploit rollout gets approved by v1 and rejected by v2. Show the version history in the Mubit console.

**Metrics chart:** agreement with ground truth (v1, v2, frontier), p50 latency per judgment, and $ per 1k judgments. Optionally add a reward-hacking rate over policy iterations.

**Pitch line:** "Every lab runs a judge farm. We turn it into a model per rubric that refreshes as fast as the policy learns to cheat it."

**Risks:** Training slots are limited (3 starts per day), so **trigger v1 on Friday night and v2 by Saturday afternoon**. Keep the exploit simple and reproducible.

## B (bank-friendly): fraud alert triage copilot
**Track:** Recurring tasks + Fast refresh.
An LLM reads a flagged alert (transaction context plus customer notes) and outputs `{disposition, risk_reasons, next_action}`. Analyst overrides act as corrections, and a new scam pattern (for example APP fraud with a new script) arrives mid-demo; the adapter updates.
Strengths: a clear buyer, a regulated domain where owning the model matters (data residency), and a clean recurring task. Weakness: it is less novel, because triage is literally one of their example tasks.

## C: ML experiment-run triage for AI labs (closest to your original sandbox)
Classify each failed training run's logs into `{failure_class, likely_cause, suggested_fix}` (NaN loss, OOM, data bug, LR divergence). Researcher corrections update the adapter. It is recurring and useful, but harder to make look exciting in a demo.

## Recommendation
Go with **A**. It keeps your instinct (AI labs, RL, discovery loops), uses Anil's verifier-bias thesis, and makes "correction → model update" the core of the product rather than an extra. Keep B as a slide: "the same engine works for bank fraud alert triage".

## Saturday morning office-hours questions for Mubit
1. How do corrections enter training? Can we submit corrected outputs, or only new calls?
2. Can we use an adapter for live inference outside Try models (since there is no Mubit model ID)?
3. Can they raise our daily training or adapter-start limits for the demo?
4. Which base models are served?
