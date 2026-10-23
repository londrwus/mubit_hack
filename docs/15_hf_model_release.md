# Can we release a model on Hugging Face, like Mubit did with Banyan Sapling?

**Short answer: yes, but not as "a Mubit product", and the cheapest strong version is a *dataset + calibration + benchmark* that extends Sapling to a new domain, not a full retrain.**

## Facts about Banyan Sapling (from the model card, `MubitLabs/banyan-sapling`)
| | |
|---|---|
| Base | Qwen3-30B-A3B-Instruct-2507, pruned MoE (96 experts/layer) + a "TriVector" verification head |
| Size | **23B params, ~46 GB BF16.** The reference setup is an H200. Needs an A100-80GB/H100-class GPU |
| Input → output | `(policy, evidence, claim)` → `supported / violation / insufficient_information / inconsistent_context`, plus calibrated probabilities and an `unsure` status |
| Training | Attention LoRA + head, 1 epoch, 50k examples (42.8k synthetic + 7.2k ContractNLI). Temperature scaling + conformal calibration |
| Results | 94.22% raw accuracy (1,731 cases); **only 45% coverage**. Synthetic 100% / 77% coverage. **ContractNLI 0.82% coverage. Insurance claims: all `unsure`** |
| Their own stated gap | "Domain expansion requires additional data and calibration." Custom loader needed |
| Latency | ~97 ms median per case |
| License | Weights Apache 2.0, code MIT |

**What this means for us:** Sapling will probably return `unsure` on most ML-research claims out of the box (a new domain). That is a risk for our verifier step, and also **an opening: Mubit explicitly says domain expansion needs data and calibration. We can provide exactly that for ML research.**

## Options
| Option | What we release | Cost (Modal) | Risk | Fit |
|---|---|---|---|---|
| **A. Dataset + recalibration + benchmark** ⭐ | `busy-machines/ml-claims`: (policy, evidence, claim) triples about ML experiments (leakage, overfit, eval/train overlap, seed noise, real gains), labelled by **ground truth** from our planted traps. Plus a recalibrated Sapling config (new temperature + conformal thresholds) and a model card showing coverage before → after | ~$2–5 (Sapling inference on an H100 for ~2k cases at ~0.1 s each, plus cold start) | Low | High: it's literally the "additional data and calibration" they asked for |
| **B. Fine-tune Sapling (LoRA + head) → `sapling-ml`** | A domain-adapted checkpoint | $10–25+ on an H100/H200, plus custom-loader engineering | **High** (23B MoE, custom head, $30 budget) | High, but a stretch goal only (with extra credits) |
| **C. Export our Mubit adapters** (the agent's `judge_improvement` v1 → v2) | Small task models trained *by Mubit* from our corrections, with a model card and version history | ~$0 | Depends on whether Mubit allows export (**Q8**) | **Highest:** the model *is* the Mubit loop |
| **D. Train our own small verifier** (e.g. a Qwen3-1.7B LoRA classifier) | A tiny, fast ML-claims verifier | ~$2–5 on an L4/A10 | Low | Medium: it bypasses Mubit's product and is weaker on the "Use of Mubit" criterion |

**Recommendation: A, plus C if Mubit allows export.** Do B only if we get extra GPU credits *and* the core demo is done by Saturday evening.

## Benchmark for the release
Held-out split of `ml-claims` (planted traps with known answers). Report:
| Model | Raw accuracy | Coverage (answered, not `unsure`) | False `supported` on leaked gains | Latency |
|---|---|---|---|---|
| Opus 5.5 (via Mubit) | | n/a | | |
| Sapling (stock calibration) | | | | |
| Sapling (our recalibration) | | | | |
| Mubit adapter v1 / v2 (if C) | | | | |
(Fill in measured numbers only.)

The key headline metric: **false "supported" on leaked gains**, i.e. how often a fake improvement gets approved. That is the verifier-bias problem Anil wrote about.

## How to present it (important)
- ❌ **Do not present it as a Mubit release** or publish under a Mubit-looking name or org. That impersonates the company, and the judges *are* the company.
- ✅ Publish under our own HF org: **"Built on Banyan Sapling by MubitLabs, trained/calibrated through Mubit."** Keep the Qwen + Mubit attribution (the Apache 2.0 terms require it).
- ✅ **Pitch it as a contribution to their ecosystem:** "Sapling's card says domain expansion needs data and calibration. We brought ML research as a new domain: here's the dataset, the calibration and the benchmark." On their launch weekend, that's flattering and useful.
- ✅ **Ask Mubit at office hours** whether they'd like to co-release it or link it from the Sapling card. Only call it "evolving Sapling" with their permission.

## Rules check
- Training and calibration runs aren't LLM calls, so they're fine. Generating or labelling claim data with an LLM **must go through Mubit**.
- Sapling inference: see **Q5** (does a locally hosted model count as an LLM call that must route through Mubit?).
- Code written during the event: ✅. Dataset generation happens during the event.

## Time cost
Option A ≈ 4–6 hours for one person, and most of the data already falls out of our benchmark's leakage traps. Schedule: Saturday afternoon, after the core race works.
