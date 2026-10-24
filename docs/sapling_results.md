# Sapling stages 1–2: results (`sapling-research` v1, DRAFT)

Status: **measured 2026-10-24, 02:07–03:37 BST.** These are the first Banyan Sapling numbers on AI-research claims. Plan: `sapling_research.md`; method: `sapling_gate.md`; decisions: D23, D28. Logged in `results_log.md` as **R33**, including the rows where we lose. **Nothing is published** (the human decides on Hugging Face).

**What we did, in the D23 wording:** *we fine-tuned Banyan Sapling's verification head on our ml-claims dataset and refitted its calibration for AI-research claims, with the 23B backbone frozen.* We never trained the backbone, LoRA or routers. One H100 pass cached the features; head training ran on the laptop CPU.

## TL;DR

- **Stock Sapling can't judge AI-research claims.** On 500 held-out research claims it answers `unsure` on **100%** of them. Its raw best guess is right **22%** of the time, below the 34.2% you'd get by always guessing the most common training label (`supported`). It calls almost every claim `insufficient_information`, including all 132 real gains.
- **sapling-research** (head fine-tuned, backbone frozen, our ML-research calibration):
  - raw accuracy **22% → 58%**; **27% → 61%** on the 3 tasks that never appear in training
  - it starts answering: **6%** of claims at alpha 0.01, **18%** at alpha 0.05, with **96.7%** and **88.9%** of those answers right
  - **fake improvements approved: 0 of 157 → 0 of 157**; fakes it calls out as fake: **0 → 21** (alpha 0.01) or **43** (alpha 0.05)
- **What it still can't do:** confirm real gains. It accepts **0 of 132** at alpha 0.01 and **1 of 132** at alpha 0.05; most still come back `unsure`. With calibration turned off, its raw guess would approve **19 of 157 fakes (12%)**, against 1 for stock. The calibration is what holds that back.
- **No regression on Sapling's original domain.** On 400 ContractNLI test cases, raw accuracy went **77.3% → 83.8%**.
- **Opus 5.5 is far better on the 50-case trap set:** 50/50, against 14/50 (alpha 0.01) and 18/50 (alpha 0.05) for sapling-research, and 10/50 for stock. Sapling's edges are speed and price: **73 ms vs about 2.2 s per case**, and about **$0.00015 vs $0.0046** per case.
- **Cost:** one H100 pass of **7.0 minutes** over 4,350 cases, about **$0.64 billed** (H100 $0.55). Head training took 80 minutes for all 5 settings tried and 13 minutes for the selected one, on the laptop CPU at **$0 GPU**.

## 1. Headline table: held-out ml-claims test (500 cases, new template families)

There are 289 "is this a genuine improvement?" claims (157 fake, 132 genuine) and 211 claims about held-out data or leaks. "Calibrated" is the status Sapling serves: a label, or `unsure`.

| | Stock Sapling | Stock head + our calibration (α 0.01) | **sapling-research** (head fine-tuned, backbone frozen), α 0.01 | **sapling-research**, α 0.05 |
|---|---:|---:|---:|---:|
| Raw accuracy (best guess, no `unsure`) | 22.0% | 22.0% | **58.2%** | **58.2%** |
| Answers given (not `unsure`) | 0.0% (0/500) | 0.0% | 6.0% (30/500) | 18.0% (90/500) |
| Accuracy of the answers given | n/a | n/a | 96.7% (29/30) | 88.9% (80/90) |
| **Fake improvements approved** ↓ | **0 / 157** | 0 / 157 | **0 / 157** | **0 / 157** |
| Fake improvements caught (called fake) | 0 / 157 | 0 / 157 | 21 / 157 | 43 / 157 |
| Genuine gains accepted | 0 / 132 | 0 / 132 | 0 / 132 | 1 / 132 |
| Genuine gains rejected (called fake) | 0 / 132 | 0 / 132 | 2 / 132 | 12 / 132 |
| *Raw (uncalibrated) fakes approved* | *1 / 157* | *1 / 157* | *19 / 157* | *19 / 157* |
| *Raw genuine rejected* | *132 / 132* | *132 / 132* | *82 / 132* | *82 / 132* |
| Model latency (H100, batch 1, median / p90) | 75 / 81 ms | same | same forward pass | same |

- **Tasks never seen in training** (LibriSpeech, HumanEval, MS MARCO; 106 test rows): raw accuracy 27.4% → **61.3%**. Answers given 0% → 10.4% (α 0.01) or 23.6% (α 0.05), with 100% and 92% of those right. Fakes approved 0/34 in every column.
- **Recalibration alone does nothing for the stock head** (column 2). Its features put the ML-research claims in the wrong class with too much confidence, so a refitted T = 15.8 still leaves every set at 3–4 labels. The gain comes from the head fine-tune. Calibration then decides how much of it gets served.
- **Which fakes it catches** (α 0.05, test): test data in training 14/25, tune on test 7/18, contaminated corpus 6/18, noise within the seed spread 6/19, changed evaluation set 5/18, select on test 3/19, evaluation on training levels 1/19, val shard in training 1/7, lucky seed 0/11, eval on training seeds 0/3. None were approved; the rest are `unsure`.
- **Does the conformal set contain the true label** (it should, about 1 − α of the time)? Fitted rows (conformal fold, train families): 99.5% at α 0.01, 95.3% at α 0.05. **Held-out test (new families): 98.2% and 91.2%**, slightly under the nominal 99% and 95%. The guarantee holds only on the fitted distribution; on new template families it is measured, not guaranteed (`data/ml_claims/README.md`).

## 2. The 50-case trap set (same cases Opus was graded on)

`data/ml_claims/trap50.jsonl` holds the 50 `trap_set_v1` cases turned into Sapling triples. Each system is graded with the trap set's own `acceptable_verdicts`, exactly as `scripts/eval_trap_set.py` grades the LLM judge: `supported` → valid, any other label → invalid, `unsure` → unsure. Unsure counts as correct on noise, lucky-seed and changed-eval fakes, and as wrong on genuine gains. **Same planted cases, different input format:** Opus reads the judge prompt (target spec plus a config and evidence JSON); Sapling reads (policy, evidence, claim). We never tuned anything on trap50 (dev or test).

| | Stock Sapling | **sapling-research** α 0.01 | **sapling-research** α 0.05 | Opus 5.5 via Mubit (R24) |
|---|---:|---:|---:|---:|
| Correct under the trap-set grading | 10 / 50 | 14 / 50 | 18 / 50 | **50 / 50** |
| dev / test split | 6/30 · 4/20 | 9/30 · 5/20 | 11/30 · 7/20 | 30/30 · 20/20 |
| Answers given (not `unsure`) | 0 / 50 | 4 / 50 | 12 / 50 | 45 / 50 |
| **Fake improvements approved** ↓ | 0 / 25 | 0 / 25 | **2 / 25** | 0 / 25 |
| Genuine gains accepted | 0 / 25 | 1 / 25 | 6 / 25 | 25 / 25 |
| Genuine gains rejected (called fake) | 0 / 25 | 0 / 25 | 0 / 25 | 0 / 25 |
| 4-class raw accuracy (Sapling labels) | 28% | 72% | 72% | n/a |
| Latency per case | 73 ms (H100) | same | same | p50 2.2–2.3 s |
| Cost per case | ~$0.00015 | same | same | ~$0.0046 ($0.229 for 50) |

**Read this honestly:** Opus 5.5 wins this set outright. The 10/50 for stock Sapling is only the 10 fakes where `unsure` is an acceptable verdict. At α 0.05, sapling-research **lets 2 of 25 fakes through**; at α 0.01 it lets none through but answers only 4 of 50 cases. Sapling cost per case = the extraction run's billed $0.64 ÷ 4,350 cases, including the cold start.

## 3. No-regression check: ContractNLI (Sapling's original domain)

- **Cases:** 400 ContractNLI **test** cases (Task A, the contract excerpt relevant to the hypothesis), from HF `kiddothe2b/contract-nli`; parquet sha256 `14a21c54…`.
  - stratified: 150 entailment → `supported`, 100 contradiction → `violation`, 150 not mentioned → `insufficient_information`
  - seed 20261024; duplicate (premise, hypothesis) pairs dropped
  - Sapling trained on ContractNLI *train*, never on test
- **Layout:** Sapling's own ContractNLI input layout is not published, so we built two:
  - `ev`: contract excerpt as evidence, with a short fixed policy
  - `pol`: contract excerpt as the policy
  - We kept the one where the **stock** head scores higher (`ev`: 77.3% vs 75.0%) and decided this before looking at ours.
- **Format caveat:** stock Sapling scores 86.32% raw on its own 731 ContractNLI cases, so our layout is not identical to theirs. The comparison that matters is stock vs ours on the same rows.

| ContractNLI test, 400 cases (`ev` layout) | Stock Sapling | **sapling-research** |
|---|---:|---:|
| **Raw accuracy** | 77.3% | **83.8%** |
| Raw false `supported` | 7 | 8 |
| Answers given, stock calibration | 0.0% | 0.0% |
| Answers given, multidomain refit (α 0.01; 133 held-out cases) | 7.5% (90% right) | 23.3% (96.8% right) |
| Answers given, our ML-only calibration (**not valid for contracts**) | 0.0% | 81.8% (87.5% right, 4 false `supported`) |
| Other layout (`pol`), raw accuracy | 75.0% | 77.7% |

**No regression:** raw accuracy on contracts went up, not down, and raw false approvals stayed flat (7 → 8). The rise is a measured fact. We have no explanation for it yet; one hypothesis is that our claims also teach "the evidence does not establish this".

**Why the release ships an ML-only calibration (D28 (3)):**
- The multidomain refit (ML-research + ContractNLI, thresholds = the max over domains, the same method as Sapling's own `fit_multidomain`) is shown above for transparency.
- With it, the ContractNLI threshold (q_hat 0.981) dominates, and ML-research coverage drops back to **0%**. That is the same trade-off the stock package makes.
- The ML-only calibration is for ML-research claims only and is stated as such. Its contract numbers above are out of scope and shown only so nothing is hidden.

## 4. Method

**Stage 1 (one combined H100 pass, `busy-machines-sapling` on Modal):**
- **Inputs:** 4,350 rows in one cold start:
  - ml-claims train 3,000 + test 500 + trap50 50
  - ContractNLI 400 cases × 2 layouts
- **Same pass:** stock logits and stock calibrated status, plus the pooled policy/evidence/claim features (forward pre-hook on `model.head`, [n, 3, 2048] fp32)
- **Checks:**
  - the package's example matched the published H200 logits exactly (max diff 0.0)
  - the stock head on captured features matched the stock logits exactly on the GPU in all 9 chunks (0.0) and to ≤ 4.8e-6 batched on CPU
- **Speed:** model load 36.3 s; **0.072–0.077 s per case** wall; model median 73.4 ms (p90 77.6 ms); median input 617 tokens (test)
- **Wall time:** 422.7 s (7.0 min) client wall
- **Spend:** billed in the 01:00–02:00 UTC hour: H100 $0.549, CPU $0.053, memory $0.036. Total Sapling app spend so far today is **$1.03**, of which H100 is **$0.71** (this includes the earlier $0.39 setup and smoke). That is well under the $3 cap for this step and the $4 cap in D28. The billing report may still lag slightly. No Sapling containers are left running.

**Stage 2 (laptop CPU, cached features, backbone frozen):**
- **Parity first:** the stock head (from `data/sapling/stock_head/head.safetensors`) on the cached features reproduces the H100 stock logits on all 4,350 rows. Max abs diff 5.7e-6, argmax 100% identical, calibrated status 100% identical. Head-only cost on CPU: 1.5 ms per case, batched.
- **Splits, all grouped by scenario** (a scenario's paraphrases never cross a split):
  - ml-claims train `fold=head` (2,219) → head-train 1,874 / head-val 345 (15% of groups, hashed)
  - `fold=temperature` (397) → T
  - `fold=conformal` (384) → q_hat
  - Test (500, held-out template families and 3 unseen tasks), trap50 and ContractNLI are evaluation only.
- **Training:**
  - warm start from the stock head
  - AdamW with no weight decay toward 0, batch 32, at most 25 epochs, early stopping after 3 epochs with no gain in validation cross-entropy
  - 5 settings decided in advance: lr {1e-5, 3e-5, 1e-4}, plus an L2 pull toward the original weights (0.01) at lr {3e-5, 1e-4}
  - **Selected (lowest val CE): lr 1e-4, no L2 pull, best epoch 14**: val CE 0.767, val acc 70.4%
  - The L2 pull gave worse validation scores (0.864–0.882 CE), so it was not chosen
  - Weights moved 24% (relative L2) from the stock head
- **Calibration:** Sapling's own `ConformalCalibrator` (the package file, sha256 checked against its manifest):
  - `fit(conformal, temperature_logits=…)` on the ML-research folds only
  - **α 0.01: T = 1.008, q_hat = 0.9766**; α 0.05: T = 1.008, q_hat = 0.9141
  - stock: T = 5.17, q_hat = 0.9955

## 5. What we can and can't say

| ✅ Supported by these numbers | ❌ Not supported |
|---|---|
| "We fine-tuned Banyan Sapling's verification head for AI-research claims, backbone frozen, with about $0.64 of H100 and 13 minutes of laptop CPU." | "We fine-tuned / LoRA-trained the 23B model." |
| "Stock Sapling says `unsure` on every AI-research claim we tried; ours starts answering and stays right 89–97% of the time when it does." | "It answers instead of saying unsure." (82–94% still `unsure`) |
| "It catches 21–43 of 157 held-out fake improvements that the original left unresolved, and approves none of them." | "It beats Opus at spotting fake improvements." (Opus 50/50 vs 14–18/50) |
| "Raw accuracy on held-out research claims 22% → 58%, and no regression on ContractNLI (77% → 84%)." | "It confirms real gains." (0–1 of 132 accepted) |
| "Sapling judges a claim in ~75 ms for ~$0.00015 of GPU; Opus takes ~2.2 s and ~$0.0046." | Any coverage guarantee on new template families or real lab logs |

Plain stage line, if used: *"Mubit's new verifier said 'unsure' to every AI-research claim we tried. We retrained its verification head in minutes, for 64 cents of GPU. Now it catches fake improvements the original missed, approves none of them, and still holds up on contracts."* "None of them" means the 157 held-out fakes; at α 0.05 it let 2 of 25 trap-set fakes through (§2), so if the trap set is on the slide, say that too. The honest follow-up: it is not yet a replacement for an LLM judge on real gains.

## 6. Feedback for Mubit (the "we improved Mubit" story)

1. **Domain expansion works at the head level:** a frozen backbone plus a retrained TriVector head lifts held-out raw accuracy from 22% to 58% on a new domain for under $1 of GPU. A **head-only delta** (235 MB fp32, `data/sapling/release/`) is a cheap way to ship domain packs.
2. **Coverage is limited by the shared multidomain threshold:** ContractNLI's q_hat (0.981–0.995) sets the threshold for every domain. **Per-domain calibration** (choosing the threshold by the request's domain) would let each domain keep its own coverage.
3. **Packaging drift:** `requirements.txt` at revision `437cf0b` does not match the package's own `distribution_manifest.json`, so `verify_files` / `inference.load_model` fail on a clean download. `handler.py` is not in the manifest at all (`workers/verifier/sapling.py`, `MANIFEST_DRIFT_SHA256`, `UNLISTED_PINNED_SHA256`).
4. **The ContractNLI input layout is not published.** With our layout the stock model scores 77% raw against the 86% reported, so publishing it would make outside regression checks comparable.

## 7. Files and how to reproduce

| File | What |
|---|---|
| `scripts/sapling_stage.py` | `prep-cnli` (ContractNLI rows), `stage1` (stock baseline), `stage2` (head fine-tune, calibration, evaluation, release files) |
| `data/sapling/contractnli/rows.jsonl`, `meta.json` | 400 ContractNLI test cases × 2 layouts; source, sha256, seed (CC BY-NC-SA 4.0, evaluation only, not redistributed) |
| `data/sapling/sapling_v1/records.jsonl`, `report.json`, `extract.log` | H100 pass: stock logits, status, latency per row; per-chunk timings and parity |
| `data/sapling/sapling_v1/volume/` | cached features and logits (`chunk_*_features.npy`, 106 MB) |
| `data/sapling/sapling_v1/stage1.json`, `stage2.json`, `stage2.log` | all metrics above, the grid and training curves |
| `data/sapling/release/` | **draft release:** `head.safetensors` (sha256 `79459531…`), `calibration.json` (α 0.01), `calibration_alpha0.05.json`, `load_sapling_research.py`, `training_config.json`, `metrics.json`, `README.md` (model card) |

```
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python -m uv run --with pyarrow python scripts/sapling_stage.py prep-cnli
python -m uv run python -m workers.runners.modal_sapling extract data/ml_claims/train.jsonl data/ml_claims/test.jsonl data/ml_claims/trap50.jsonl data/sapling/contractnli/rows.jsonl sapling_v1 --chunk 500 --max-usd 3
python -m uv run python -m workers.runners.modal_sapling fetch outputs/sapling_v1 data/sapling/sapling_v1/volume
python -m uv run python scripts/sapling_stage.py stage1
python -m uv run --extra ml --with safetensors==0.8.0 python scripts/sapling_stage.py stage2
```
`stage2` needs `data/sapling/stock_head/` (dumped by the H100 smoke) and `data/sapling/pkg/runtime/` (Sapling's runtime code, fetched from the Volume and checked against the manifest).

## 8. Caveats

- **The dataset is synthetic and templated** (`data/ml_claims/README.md`). Test uses held-out template families and 3 unseen tasks, but it is still our generator. Real lab logs are untested.
- **The trap-set comparison with Opus uses different input formats** for the same planted cases (see §2).
- **Our ContractNLI layout is our own guess** at the input format (see §3).
- **One seed, one run.** Head-val is in the train template families, so its 70% overstates held-out accuracy (58% on test).
- **Latency for sapling-research was not measured separately on the GPU.** The head has the same shape and FLOPs as the stock head, so the 73 ms figure carries over by construction, not by measurement.
- **Sapling stays out of the live race loop** until Q5 is answered (D28 (6)). Nothing here is an LLM call, so nothing goes through Mubit.
