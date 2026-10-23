# Optional "wow" piece: our own model built on Mubit's Banyan Sapling

Status: **OPTIONAL / STRETCH** (D20). The benchmark vs Opus 5.5 comes first, always. This replaces options B and C in doc 15 with a staged, de-risked plan.

## The idea in one line
Mubit's new open model **Banyan Sapling** checks whether a claim is backed by evidence. Its own model card says it struggles outside its training domain and that "domain expansion requires additional data and calibration."
**We build that expansion for AI labs: a version of Sapling that spots *fake improvements in AI research results*** (contaminated tests, overfit gains). We publish it with a benchmark, so Mubit can ship it, sell it, or link it from the Sapling card.

Stage line: *"We didn't just use Mubit. We extended Mubit's brand-new model for AI labs, and it beats the original at spotting fake improvements."*

## Why it fits the demo (not a side project)
- It **is** the independent verifier in the race: it catches the fake improvement on stage.
- It gives two Mubit-native models on stage: **small fast models (Mubit adapters) run the assistant**, and **our fine-tuned Sapling judges the results**.
- It answers Anil's verifier-bias point with a real model, not a slide.

## Facts that shape the plan (from the HF repo)
- 23B params (Qwen3-30B-A3B pruned MoE, ~3B active), **46.2 GB** BF16 in one `model.safetensors`, a custom `inference.py` loader, and a small `calibration.json` (temperature + conformal thresholds).
- Needs **one 80 GB GPU** (A100-80GB / H100 / H200) to load. ~97 ms per case.
- Input `(policy, evidence, claim)` → `supported / violation / insufficient_information / inconsistent_context`, or `unsure`.
- Out of domain it mostly answers `unsure` (ContractNLI 0.8% coverage; insurance: all `unsure`).

## Staged plan: each stage is a shippable result
| Stage | What | GPU cost (H100 @ $3.95/h) | Risk | Shippable result |
|---|---|---|---|---|
| **0. Dataset** | `ml-claims` generator: (policy, evidence, claim) triples about research results. Labels come from **planted ground truth**: real gains, contaminated test, eval on training seeds, too-small difference (noise), missing evidence. Paraphrase variety from Haiku **via Mubit**. Target ~3–5k examples + a held-out test split | $0 (CPU + ~$2 Haiku) | Low | An HF dataset |
| **1. Baseline** | Stock Sapling on the held-out test: accuracy, **coverage (not `unsure`)**, **fake improvements approved**, latency | ~$1–2 | Low | "Stock Sapling vs research claims" numbers |
| **2. Recalibrate + head-only fine-tune** | Freeze the 23B backbone. Run it **once** over the dataset and cache the pooled policy/evidence/claim features. Retrain only the small verification head on CPU/any GPU, then refit temperature + conformal thresholds (`calibration.json`) | ~$2–4 (one forward pass + loading) | Medium (needs hooks into `inference.py`) | **`sapling-research` v1**: coverage up, fake improvements approved down |
| ~~**3. LoRA fine-tune**~~ | **Dropped (D23).** High risk, $10–25 we don't have, and it adds little to the pitch | n/a | n/a | n/a |

**Go/no-go gates:**
- Fri night: read `inference.py` and the `docs/`. **Can we get pooled features out?** If not, stage 2 becomes calibration-only.
- Sat 14:00: no stage 2 → stop. Ship the dataset + baseline + calibration only, and use the matching wording below.
- Sat 18:00: hard stop for all model work. Everything after that goes to the benchmark and the demo.

## How to describe it honestly (D23)
Stage 2 **is real fine-tuning**: we train the model's verification head on our data and refit its calibration, with the 23B backbone frozen. Present the frozen backbone as a **deliberate design choice**, not a shortcut: it's cheap, fast, and keeps the original model's knowledge intact. That is the cost-and-efficiency story Mubit itself tells.

| What we actually did | ✅ Say | ❌ Never say |
|---|---|---|
| Stage 2 done (head trained + recalibrated) | "We **fine-tuned** Banyan Sapling for AI research: we froze the 23B backbone and retrained its verification head on our dataset, for under $5." | "We fully fine-tuned / LoRA-trained the 23B model." |
| Only calibration worked (no feature hooks) | "We **recalibrated** Sapling for AI-research claims with our new dataset." | "fine-tuned" |
| Only stage 1 | "We built the first AI-research benchmark for Sapling and measured where it fails." | "trained" / "fine-tuned" |

**Why precision matters here:** Mubit *built* Sapling (attention LoRA + head, routers frozen). The judges know its architecture better than anyone, and "which layers did you train?" is a likely question. The precise answer ("the head, backbone frozen, $X, Y minutes, and here's the no-regression check") **impresses them**. A vague "we fine-tuned it" falls apart on the first follow-up, and "holds up under questions" is a judging criterion.

Plain-language stage line (D19): *"We taught Mubit's new model to spot fake improvements in AI research, in minutes, for under five dollars."*

## The model's own benchmark (what we publish)
| Model | Accuracy | Answers given (not "unsure") | **Fake improvements approved** ↓ | Speed |
|---|---|---|---|---|
| Opus 5.5 (via Mubit) | | n/a | | |
| Stock Banyan Sapling | | | | |
| **Our sapling-research** | | | | |
- Plus a **no-regression check** on Sapling's original domain (e.g. public ContractNLI), so we can show we didn't break it.
- Plain-language headline (D19): **"It spots fake improvements the original misses, and it answers instead of saying 'unsure'."**
(Measured numbers only.)

## Publishing and credit (non-negotiable)
- Publish under **our** HF org: `busy-machines/sapling-research` (model) + `busy-machines/ml-claims` (dataset), with an **Apache 2.0** licence and the Qwen + MubitLabs attribution kept.
- Model card: **"Community fine-tune of MubitLabs/banyan-sapling for AI-research claim verification."** Never present it as an official Mubit release.
- **Ask Mubit on Saturday morning** whether they want to co-release it or link it from the Sapling card. If they say yes, the pitch line becomes "Mubit can ship this on Monday."

## Compute
- An 80 GB GPU is required. **We have no local GPUs; Modal is the only compute** (D22). Options: Modal H100 within the $30, or **hackathon GPU credits (Q6)**.
- The Modal $30 is mostly reserved for the benchmark (doc 08). Stages 1–2 (~$5) fit; **stage 3 only happens if we get extra GPU credits.** Otherwise the plan ends at stage 2.
- Downloading the 46 GB weights costs container time on every cold start. **Cache them in a Modal Volume once** (1 TiB free) and keep the H100 container warm only while it's needed.

## Rules check
- Fine-tuning and feature extraction aren't LLM calls. Paraphrasing calls go through Mubit ✅.
- Verifier inference during the race: **Q5** (does a self-hosted model need to sit behind Mubit?).
- All code is written during the event ✅.
