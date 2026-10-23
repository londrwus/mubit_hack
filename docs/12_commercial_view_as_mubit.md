# Thinking as Mubit: would we sell this, and to whom? (AI labs only, per D18)

Mubit's business: usage through the gateway → owned adapters per recurring task → retention, because the models are trained on *your* traces. Mubit therefore wants customers with **high-volume, recurring, verifiable LLM calls**. AI-lab research loops have exactly that shape.

## Segments inside AI labs, ranked by how much Mubit should care
| # | Segment | Recurring LLM volume | Willingness to pay | Fit | Verdict |
|---|---|---|---|---|---|
| 1 | **RL / post-training and eval teams** (LLM-as-judge, reward grading, rollout filtering, eval pipelines) at mid-size labs, fine-tuning companies and **RL-environment vendors** | **Enormous**: millions of judge calls per run, fixed rubric, fixed output | High: judge cost and latency directly slow training | ⭐⭐⭐⭐⭐ | The **highest-volume** wedge: distilled judges plus a Sapling-style independent verifier |
| 2 | **Research teams at mid-size labs and neolabs** running agent-driven sweeps, ablations, architecture search and speedruns | High, nightly | High: compute is their biggest cost line | ⭐⭐⭐⭐⭐ | The core Busy Machines buyer: **$-to-target and GPU idle %** speak directly to their compute budget |
| 3 | **AI-scientist / autonomous-research startups** (Rekursiv-type) | High per customer, few customers | Medium | ⭐⭐⭐ | Design partners and logos; a small market today |
| 4 | **GPU clouds / neoclouds** (Modal, Lambda, CoreWeave) as a *channel* | n/a | Partnership | ⭐⭐⭐ | They want AI-lab customers who use GPUs productively. Offer as a marketplace integration |
| 5 | **Frontier labs** (OpenAI, Anthropic, DeepMind) | Huge | Low as customers: they build in-house and keep traces internal | ⭐⭐ | **Credibility, not revenue**: open benchmark results (night/bench), research collaborations, and on-prem/owned-model options later |

## Go-to-market Mubit would actually use
- **The wedge already exists:** AI-lab researchers *already* drive experiments with Claude Code or Codex, and Mubit already supports Claude Code through the gateway. The pitch: **"Point your research agent at Mubit. Tonight's sweep runs on models trained from last night's."** No new product, only a new use case.
- **Time to visibility (Anil's metric):** the **$-to-target / GPU idle %** dashboard shows value after the first night.
- **Pricing:** usage-based per gateway call (their existing model), plus a "research tier" priced against **GPU $ saved** (a share of the idle GPU spend eliminated). Compute is the line item labs care about, more than tokens (Anil: token cost isn't the motivation, efficiency is).
- **Trust features labs need:**
  - per-step recording controls (`x-middleware-record`), so proprietary architectures aren't stored
  - owned adapters
  - an independent verifier that catches **eval contamination** before a result reaches a paper or a launch

## Attack on our own idea (what a Mubit PM would ask)
1. **"Why an LLM agent at all? Optuna / Ray Tune / W&B Sweeps do hyperparameter search for free."**
   → The biggest threat. Agents change *code, architecture, data mixtures and training recipes*, not just numbers. **The benchmark includes an Optuna baseline.**
2. **"LLM cost is small next to the GPU bill."** True for long runs. Our value is **idle GPU time and throughput** in short-iteration loops: sweeps, ablations, speedruns and RL tuning. Keep the pitch on utilisation.
3. **"Research calls aren't recurring."** The *steps* are (propose / diagnose / judge), and RL **judging** is the most recurring call in ML.
4. **"Labs won't send traces."** Recording controls, owned adapters and open models (Sapling is Apache 2.0). This is the "own your intelligence layer" track.
5. **"Market size?"** Compute is the largest cost line in every AI lab, and most of it goes to experiments rather than final runs (Epoch AI's OpenAI estimate; **verify before quoting**). RL judge volume grows with every post-training run.

## Commercial slide
```
Who pays        AI labs: RL/post-training & eval teams · research teams at mid-size labs and neolabs · AI-scientist startups
Why now         Agents now run the experiments; their GPUs sit idle while they think
Wedge           One base-URL change in the research agent labs already use (Claude Code / Codex)
Metric sold     $-to-target and GPU idle %, visible after one night
Moat for Mubit  Adapters trained on the lab's own research traces + an independent contamination verifier
```
