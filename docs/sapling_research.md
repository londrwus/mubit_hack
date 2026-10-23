# OPTIONAL: `sapling-research`, our fine-tune of Banyan Sapling

Status: **OPTIONAL / STRETCH** (D20). The benchmark vs Opus 5.5 comes first, always. Go/no-go at **Sat 14:00**, hard stop at **Sat 18:00**.

## The idea in one line
Mubit's new open model **Banyan Sapling** checks whether a claim is backed by evidence. Its own model card says it struggles outside its training domain and that **"domain expansion requires additional data and calibration."** We build that expansion for AI labs: **a version of Sapling that spots fake improvements in AI research results** (contaminated tests, overfit gains, noise), published with a benchmark so Mubit can ship it, sell it, or link it from the Sapling card.

Stage line (plain, D19): *"We taught Mubit's new model to spot fake improvements in AI research, in minutes, for under five dollars."*
Judge-level line: *"We didn't just use Mubit. We extended Mubit's brand-new model for AI labs, and it beats the original at spotting fake improvements."* (Only if measured.)

### Why it belongs in the demo
- It **is** the independent verifier in the race: it catches the fake improvement on stage.
- Two Mubit-native models on stage: **Mubit adapters run the assistant**, **our Sapling judges the results**.
- It answers Anil's verifier-bias point with a real model, not a slide.

## Facts about Banyan Sapling (model card, `MubitLabs/banyan-sapling`)
| | |
|---|---|
| Base | Qwen3-30B-A3B-Instruct-2507, pruned MoE (96 experts/layer, ~3B active) + a "TriVector" verification head |
| Size | **23B params, 46.2 GB BF16** in one `model.safetensors`; custom `inference.py` loader; small `calibration.json` (temperature + conformal thresholds) |
| Hardware | **One 80 GB GPU** (A100-80GB / H100 / H200); reference setup H200. ~97 ms median per case |
| Input → output | `(policy, evidence, claim)` → `supported / violation / insufficient_information / inconsistent_context`, with calibrated probabilities and an `unsure` status |
| Training | Attention LoRA + head (routers frozen), 1 epoch, 50k examples (42.8k synthetic + 7.2k ContractNLI); temperature scaling + conformal calibration |
| Results | 94.22% raw accuracy (1,731 cases) but **only 45% coverage**. Synthetic 100% / 77% coverage. **ContractNLI 0.82% coverage. Insurance claims: all `unsure`** |
| Licence | Weights Apache 2.0, code MIT |

**What this means:** out of the box, Sapling will probably answer `unsure` on most ML-research claims. That's a risk for our verifier step and exactly the gap we fill.

## Staged plan (each stage is a shippable result)
| Stage | What | GPU cost (H100 @ $3.95/h) | Risk | Shippable result |
|---|---|---|---|---|
| **0. Dataset** | `ml-claims` generator: (policy, evidence, claim) triples about research results. Labels from **planted ground truth**: real gains, contaminated test, eval on training seeds, too-small difference (noise), missing evidence. Paraphrase variety from Haiku **via Mubit**. ~3–5k examples + a held-out test split. Most of it falls out of the benchmark's contamination traps | $0 (CPU + ~$2 Haiku) | Low | An HF dataset |
| **1. Baseline** | Stock Sapling on the held-out split: accuracy, **coverage (not `unsure`)**, **fake improvements approved**, latency | ~$1–2 | Low | "Stock Sapling vs research claims" numbers |
| **2. Head fine-tune + recalibration** | Freeze the 23B backbone. Run it **once** over the dataset and cache pooled policy/evidence/claim features. Retrain only the verification head (CPU or any GPU), then refit temperature + conformal thresholds | ~$2–4 | Medium (needs hooks into `inference.py`) | **`sapling-research` v1**: coverage up, fake improvements approved down |

LoRA / full fine-tuning is **dropped (D23)**: high risk, $10–25 we don't have, little pitch value.

### Gates
- **Fri night:** read `inference.py` and the repo docs. **Can we get pooled features out?** If not, stage 2 becomes calibration-only.
- **Sat 14:00:** no stage 2 → stop. Ship dataset + baseline + calibration, with the matching wording below.
- **Sat 18:00:** hard stop for all model work. Everything after goes to the benchmark and the demo.

## Describe it precisely (D23)
Stage 2 **is real fine-tuning**: we train the verification head on our data and refit calibration, with the backbone frozen. Present the frozen backbone as a **deliberate design choice** (cheap, fast, keeps the original knowledge), which is the cost-and-efficiency story Mubit itself tells.

| What we actually did | ✅ Say | ❌ Never say |
|---|---|---|
| Stage 2 (head trained + recalibrated) | "We **fine-tuned** Banyan Sapling for AI research: we froze the 23B backbone and retrained its verification head on our dataset, for under $5." | "We fully fine-tuned / LoRA-trained the 23B model." |
| Only calibration worked | "We **recalibrated** Sapling for AI-research claims with our new dataset." | "fine-tuned" |
| Only stage 1 | "We built the first AI-research benchmark for Sapling and measured where it fails." | "trained" / "fine-tuned" |

**Why precision matters:** Mubit *built* Sapling and knows its architecture better than anyone; "which layers did you train?" is a likely question. A precise answer (the head, backbone frozen, $X, Y minutes, plus the no-regression check) impresses; a vague "we fine-tuned it" falls apart on the first follow-up.

## The benchmark we publish
| Model | Accuracy | Answers given (not `unsure`) | **Fake improvements approved** ↓ | Speed |
|---|---|---|---|---|
| Opus 5.5 (via Mubit) | | n/a | | |
| Stock Banyan Sapling | | | | |
| **Our sapling-research** | | | | |
| *(If Mubit allows)* `judge_improvement` adapter v1 / v2 | | | | |

- The key metric is **fake improvements approved**: how often a contaminated gain gets through. That's the verifier-bias problem Anil wrote about.
- Plus a **no-regression check** on Sapling's original domain (e.g. public ContractNLI).
- Plain headline: **"It spots fake improvements the original misses, and it answers instead of saying 'unsure'."**
- Measured numbers only, logged in `results_log.md`.

## Publishing and credit (non-negotiable)
- Publish under **our** HF org: `busy-machines/sapling-research` (model) + `busy-machines/ml-claims` (dataset), Apache 2.0, keeping the Qwen + MubitLabs attribution.
- Model card: **"Community fine-tune of MubitLabs/banyan-sapling for AI-research claim verification."** Never present it as an official Mubit release or under a Mubit-looking name: that would impersonate the company, and the judges *are* the company.
- Pitch it as a contribution to their ecosystem: "Sapling's card says domain expansion needs data and calibration. We brought AI research as a new domain."
- **Ask Mubit on Saturday morning (Q8)** whether they want to co-release it or link it from the Sapling card. Only if yes: "Mubit can ship this on Monday."

## Compute
- Needs an 80 GB GPU; **Modal is our only compute** (D22). Stages 1–2 (~$5) fit in the $30 (`budget.md`).
- Download the 46 GB weights **once into a Modal Volume** and keep the H100 container warm only while needed.

## Rules check
- Fine-tuning and feature extraction aren't LLM calls. Paraphrasing / labelling calls go through Mubit ✅.
- Verifier inference during the race: **Q5** (does a self-hosted model need to sit behind Mubit?).
- All code and data generation happen during the event ✅.
