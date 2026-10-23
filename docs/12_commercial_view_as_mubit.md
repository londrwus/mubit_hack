# Thinking as Mubit: would we sell this, and to whom?

Mubit's business: usage through the gateway → owned adapters per recurring task → retention, because the models are trained on *your* traces. Mubit therefore wants customers with **high-volume, recurring, verifiable LLM calls**. The question is whether ML/RL research is one of them.

## Segments, ranked by how much Mubit should care
| # | Segment | Recurring LLM volume | Willingness to pay | Fit | Verdict |
|---|---|---|---|---|---|
| 1 | **RL post-training and eval teams** (LLM-as-judge, reward grading, rollout filtering): mid-size labs, fine-tuning startups, **RL-environment vendors** (a fast-growing 2025–26 market) | **Enormous**: millions of judge calls per run, fixed rubric, fixed output | High: judge cost and latency directly slow their training | ⭐⭐⭐⭐⭐ | The **highest-volume** wedge. Distilled judges plus Sapling-style verification. |
| 2 | **Enterprise ML teams** (banks: fraud/credit; Spotify-type recsys; ad-tech) using coding agents for retraining, sweeps and model-risk documentation | Medium–high, recurring weekly | High, with existing budgets | ⭐⭐⭐⭐ | Easiest to sell. Mubit already targets enterprise, and the "fast refresh" angle fits fraud drift. |
| 3 | **AI-scientist / autonomous-research startups** (Rekursiv-type, FutureHouse-type, materials and bio discovery) | High per customer, few customers | Medium (startups) | ⭐⭐⭐ | Great design partners and logos, a small market today. |
| 4 | **Neoclouds / GPU platforms** (Modal, Lambda, CoreWeave) as a *channel* | n/a | Partnership | ⭐⭐⭐ | They want customers who use GPUs productively. Offer as a marketplace integration. |
| 5 | **Frontier labs** (OpenAI, Anthropic, DeepMind) | Huge | Low: they build in-house, own the models, and won't send traces out | ⭐ | Don't pitch as a customer. Use as *benchmark credibility* only. |

**Honest correction:** the "AI labs" pitch should mean **mid-size labs, RL vendors and enterprise ML teams**, not frontier labs.

## Go-to-market Mubit would actually use
- **The wedge already exists:** ML researchers *already* drive experiments with Claude Code or Codex, and Mubit already supports Claude Code through the gateway. The pitch: **"Point your research agent at Mubit. Tonight's sweep runs on models trained from last night's."** No new product, only a new use case for the existing one. This matches Anil's "product-first, no forward-deployed engineers" and "time to visibility".
- **Time to visibility:** the **$-to-target / GPU idle %** dashboard *is* the visibility metric. A customer sees value after the first night. This is the strongest commercial argument in our project.
- **Pricing ideas:** usage-based per gateway call (their existing model), plus a "research tier" priced against **GPU $ saved** (e.g. a % of idle GPU spend eliminated). That is easier to justify than token savings, which Anil himself says enterprises don't buy for.
- **Banks:** a regulated angle. **Owned, on-prem adapters plus Banyan Sapling as a claim-vs-evidence verifier** fits **model risk management** (validating that a model improvement is supported by evidence). This is a real compliance pain with budget behind it.

## Attack on our own idea (what a Mubit PM would ask)
1. **"Why an LLM agent at all? Optuna / Ray Tune / W&B Sweeps do hyperparameter search for free in milliseconds."**
   → This is the biggest threat. Answer: agents change *code, architecture, data and features*, not just numbers. **We must include an Optuna baseline in the benchmark**, or judges will ask about it.
2. **"LLM cost is small next to the GPU bill."** True for long runs. Our value is **idle GPU time and throughput** in short-iteration loops, not LLM dollars. Keep the pitch on utilisation.
3. **"Research calls aren't recurring."** The *steps* are (propose / diagnose / judge), and the RL **judge** is the most recurring call in all of ML. This is the strongest technical fit.
4. **"Data sensitivity: labs won't send traces."** Answer: owned adapters, open models (Sapling is Apache 2.0) and on-prem options. This is exactly the "own your intelligence layer" track.
5. **"Market size?"** RL-judge volume plus enterprise retraining is large. Pure AI-scientist startups are small. Lead with the former in the commercial slide.

## Recommendation for the pitch's commercial slide
```
Who pays        RL & eval teams (judge volume) · Enterprise ML (fraud, recsys) · AI-research startups
Why now         Agents now run the experiments; their GPUs sit idle while they think
Wedge           One base-URL change in the research agent you already use (Claude Code / Codex)
Metric sold     $-to-target and GPU idle % — visible after one night
Moat for Mubit  Adapters trained on the customer's own research traces + Sapling verifier
```
This keeps the broad "ML/DL/RL" story and the bank example, and leads commercially with the segment that has the volume Mubit's business model needs.
