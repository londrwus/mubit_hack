# sapling-research v2: results (one timeboxed iteration, DRAFT)

Status: **measured 2026-10-24, 03:45–05:56 BST.** This follows v1 (`sapling_results.md`, R33). Logged in `results_log.md` as R42 (Hugging Face package check: R69), including the rows where v2 loses. **Nothing is published.** Wording rules from D23 still apply: *we fine-tuned Banyan Sapling's verification head, backbone frozen*. v2 also uses a seed ensemble of five heads (§5).

**Goal of this iteration:** more answers (fewer `unsure`) and more genuine gains accepted, with **fakes approved kept at about 0** and **no ContractNLI regression**.

## TL;DR

- **More answers, still no fakes approved.** On the same 500 held-out research claims and with Sapling's own calibration method (LAC):

  | | v1 | v2 |
  |---|---:|---:|
  | Raw accuracy | 58.2% | **66.0%** |
  | Answers given, α 0.01 | 6.0% | **19.0%** (94.7% of them right) |
  | Answers given, α 0.05 | 18.0% | **63.8%** (75.5% of them right) |
  | Fakes approved | 0 / 157 | **0 / 157** at both α |
  | Fakes caught, α 0.01 | 21 | **52** |
  | Fakes caught, α 0.05 | 43 | **118** |

- **Genuine gains are still the weak spot.**
  - Accepted: **0 / 132** at α 0.01 (v1: 0) and **17 / 132** at α 0.05 (v1: 1).
  - v2 also **rejects more real gains**: 9 vs 2 at α 0.01, and **43 vs 12** at α 0.05.
  - Raw (uncalibrated) genuine rejected went up, 82 → 91 of 132, while raw fakes approved went down, 19 → 4 of 157. v2 leans further toward "not established".
- **Accuracy of the answers fell at α 0.05** (88.9% → 75.5%), and **conformal coverage breaks under template shift**:
  - at α 0.05 the set holds the true label on only **77.8%** of test rows (95% nominal; v1 had 91.2%)
  - at α 0.01 it is 96.2% (99% nominal)
  - on the fitted fold it holds (95.3% / 99.5%)
- **No ContractNLI regression** for the selected model: raw accuracy stock **77.3%** → v1 **83.8%** → v2 **84.3%** (309 / 335 / 337 of 400), raw false `supported` 7 / 8 / 8. *A first v2 without an anchor to the stock head fell to 52.0%* (§4). We fixed that with an L2 pull toward the stock head, selected on a ContractNLI dev third, and we report the untouched two thirds separately.
- **Trap set (50):**
  - stock 10/50; v1 14/50 (α 0.01) and 18/50 (α 0.05)
  - **v2 18/50 (α 0.01) and 36/50 (α 0.05)**, with 0 and 1 fakes let through
  - **Opus 5.5 is still 50/50.**
- **Data was the bottleneck.** The learning curve was still rising steeply, so we ran one extra H100 extraction on 3,000 new rows: **274 s wall, $0.43 billed** (H100 $0.373).
  - With the extra rows, validation cross-entropy fell from 0.909 to 0.653 and validation accuracy rose from 66% to 77%.
  - Every other lever we tried moved validation far less (§3).
- **Parity with Mubit's hosted Sapling (D33):**
  - **30/30 rows agree** with our H100 stock run on raw label, status (all `unsure`) and token count
  - logits are **not bit-identical**: max |Δ| 0.87, median per-row max 0.24 (§6)

## 1. Held-out ml-claims test (500 cases, new template families)

All columns use the same rows, metric code and grading as R33. v1 is re-scored from its saved logits with its calibration refitted the same way, and it reproduces R33 exactly. "Calibrated" means the status the system serves: a label or `unsure`.

| | Stock Sapling | v1 α 0.01 | v1 α 0.05 | **v2 α 0.01** | **v2 α 0.05** |
|---|---:|---:|---:|---:|---:|
| Raw accuracy (best guess, no `unsure`) | 22.0% | 58.2% | 58.2% | **66.0%** | **66.0%** |
| Answers given (not `unsure`) | 0.0% (0/500) | 6.0% (30/500) | 18.0% (90/500) | **19.0% (95/500)** | **63.8% (319/500)** |
| Accuracy of the answers given | n/a | 96.7% | 88.9% | 94.7% | **75.5%** ↓ |
| **Fake improvements approved** ↓ | **0 / 157** | **0 / 157** | **0 / 157** | **0 / 157** | **0 / 157** |
| Fake improvements caught | 0 / 157 | 21 / 157 | 43 / 157 | **52 / 157** | **118 / 157** |
| Genuine gains accepted | 0 / 132 | 0 / 132 | 1 / 132 | 0 / 132 | **17 / 132** |
| Genuine gains rejected (called fake) ↓ | 0 / 132 | 2 / 132 | 12 / 132 | 9 / 132 ↑ | **43 / 132** ↑ |
| False `supported`, all claim kinds | 0 | 0 | 1 | 0 | 4 |
| *Raw fakes approved* | *1 / 157* | *19 / 157* | *19 / 157* | *4 / 157* | *4 / 157* |
| *Raw genuine rejected* | *132 / 132* | *82 / 132* | *82 / 132* | *91 / 132* ↑ | *91 / 132* ↑ |
| Set contains the true label (nominal 99% / 95%) | 100% | 98.2% | 91.2% | 96.2% | **77.8%** ↓ |
| Calibration (T, q_hat) | 5.17, 0.9955 | 1.008, 0.9766 | 1.008, 0.9141 | 0.793, 0.9853 | 0.793, 0.8158 |

**Tasks never seen in training** (LibriSpeech, HumanEval, MS MARCO; 106 test rows):

| | Stock | v1 α 0.01 | v1 α 0.05 | **v2 α 0.01** | **v2 α 0.05** |
|---|---:|---:|---:|---:|---:|
| Raw accuracy | 27.4% | 61.3% | 61.3% | **67.0%** | **67.0%** |
| Answers given | 0% | 10.4% (100% right) | 23.6% (92.0% right) | 17.9% (89.5% right) | **63.2% (79.1% right)** |
| Fakes approved | 0 / 34 | 0 / 34 | 0 / 34 | 0 / 34 | 0 / 34 |
| Fakes caught | 0 / 34 | 9 / 34 | 15 / 34 | 9 / 34 | **27 / 34** |
| Genuine accepted | 0 / 26 | 0 / 26 | 0 / 26 | 0 / 26 | 3 / 26 |
| Genuine rejected | 0 / 26 | 1 / 26 | 2 / 26 | 3 / 26 | 11 / 26 |
| Set contains true label | 100% | 99.1% | 94.3% | 96.2% | 81.1% |

**Which fakes v2 catches** (α 0.05; v1 in brackets): noise within the seed spread 17/19 (6), test data in training 20/25 (14), tune on test 15/18 (7), select on test 15/19 (3), contaminated corpus 15/18 (6), changed evaluation set 12/18 (5), evaluation on training levels 11/19 (1), lucky seed 6/11 (0), val shard in training 5/7 (1), eval on training seeds 2/3 (0). It approved none of them.

## 2. Trap set (50 cases, graded exactly as the LLM judge)

Same grading as R33 §2 (`acceptable_verdicts`; `unsure` counts as correct only on noise, lucky-seed and changed-eval fakes). We never tuned anything on trap50.

| | Stock | v1 α 0.01 | v1 α 0.05 | **v2 α 0.01** | **v2 α 0.05** | Opus 5.5 via Mubit (R24) |
|---|---:|---:|---:|---:|---:|---:|
| Correct | 10 / 50 | 14 / 50 | 18 / 50 | **18 / 50** | **36 / 50** | **50 / 50** |
| dev / test | 6/30 · 4/20 | 9/30 · 5/20 | 11/30 · 7/20 | 12/30 · 6/20 | 21/30 · 15/20 | 30/30 · 20/20 |
| Answers given | 0 / 50 | 4 / 50 | 12 / 50 | 8 / 50 | 31 / 50 | 45 / 50 |
| **Fakes approved** ↓ | 0 / 25 | 0 / 25 | 2 / 25 | **0 / 25** | **1 / 25** | 0 / 25 |
| Genuine accepted | 0 / 25 | 1 / 25 | 6 / 25 | 2 / 25 | 15 / 25 | 25 / 25 |
| Genuine rejected | 0 / 25 | 0 / 25 | 0 / 25 | 0 / 25 | 0 / 25 | 0 / 25 |
| 4-class raw accuracy | 28% | 72% | 72% | **88%** | **88%** | n/a |

**Read this honestly:** Opus 5.5 still wins outright. At α 0.05, v2 lets 1 of 25 fakes through (v1: 2).

## 3. What we tried, and what the validation split said (no test used for any choice)

**Validation split (new in v2): template-held-out within the train families.** A scenario group goes to validation (`val_tf`) if its policy family is `terse` or its evidence family is `table`. That gives 986 rows; 1,233 head-fold rows stay for training. Test families (`qa`/`memo`, `log`/`sections`, `casual`/`headline`) are never used. v1's validation set was a random 15% of groups from the same families, so it overstated held-out accuracy (70% val vs 58% test). val_tf is the closer proxy: 66–77% val here vs 66% test.

**Fast trainer.** We cache the stock head's first-layer output once. The update is low-rank on W1 (LoRA) plus full updates to b1, W2 and b2, and it is merged back into the stock head's exact shape. That is about 2 s per epoch on the laptop CPU instead of about 40 s. The parity check (stock path through the cache vs the H100 logits) gives a max diff of 5.7e-6.

**Validation results** (3 seeds each; "ens" = mean logits of the 3 seeds; CNLI dev = the ContractNLI third `case % 3 == 0`, 134 rows; stock = 73.9%):

| Config | Extra rows | Mean val CE | **Ens val CE** | Ens val acc | CNLI dev (ens) | Note |
|---|---|---:|---:|---:|---:|---|
| LoRA r64, lr 3e-4 | no | 0.953 | 0.909 | 66.3% | — | |
| LoRA r64, lr 1e-3 | no | 1.028 | 0.923 | 64.5% | — | |
| LoRA r64, lr 1e-4 | no | 0.993 | 0.985 | 59.6% | — | |
| LoRA r256, lr 3e-4 | no | 0.969 | 0.949 | 62.1% | — | |
| + class-balanced loss | no | 1.012 | 0.955 | 60.6% | — | worse |
| + balanced + cost-sensitive (penalty on P(sup) for non-supported) | no | 1.206 | 1.061 | 58.7% | — | worse; LAC α 0.05 accepts 0/253 real gains |
| + cost-sensitive only | no | 1.062 | 1.037 | 57.4% | — | worse; same |
| + dropout 0.3 | no | 0.983 | 0.959 | 62.6% | — | |
| + weight decay 1e-2 | no | 0.952 | 0.924 | 65.3% | — | |
| v1 recipe (full W1, lr 1e-4, batch 32), 1 seed | no | 0.893 | — | 68.5% | — | best single seed without extra data, ~11 min |
| LoRA r64, lr 3e-4 | **yes** | 0.761 | 0.697 | 76.6% | **52.2%** ✗ | |
| LoRA r256, lr 3e-4 | **yes** | 0.677 | **0.653** | 76.7% | **66.4%** ✗ | became the first, unanchored final (§4) |
| + L2 pull to stock 1e-4 | yes | 0.689 | 0.647 | 76.0% | 73.1% ✗ | just below stock |
| **+ L2 pull to stock 1e-3** | yes | 0.695 | **0.664** | 75.5% | **81.3%** ✓ | **selected**: best val CE among ✓ |
| + L2 pull to stock 1e-2 | yes | 0.801 | 0.773 | 71.9% | 82.1% ✓ | |
| + L2 pull to stock 1e-1 | yes | 1.027 | 1.012 | 59.9% | 82.1% ✓ | |
| WiSE-FT interpolation β 0.25 / 0.5 / 0.75 (of the unanchored r256) | yes | — | 3.59 / 2.05 / 1.03 | 33–64% | 72–69% ✗ | interpolating LoRA deltas fails |

- **Learning curve** (LoRA r64, val_tf, 3 seeds; mean val CE / accuracy by head-train rows):

  | Head-train rows | Val CE | Val accuracy |
  |---:|---:|---:|
  | 319 | 1.247 | 45.8% |
  | 633 | 1.103 | 54.7% |
  | 932 | 1.071 | 57.9% |
  | 1,233 | 0.953 | 64.4% |

  It was still rising steeply, so **data was the bottleneck**. That justified the one extra H100 run, and the gain from it is larger than every other lever combined.
- **Seed ensembles help, weight soups don't.** Averaging logits lowered val CE by 0.02–0.10 in every config. Averaging LoRA weights across seeds (soups) was worse than any single seed, because each seed's random A start lands in a different basin. v2 therefore ships **5 heads as compact deltas** (§5).
- **Calibration variants** (the same v2 head; all on test, all reported, **none used for selection**):

  | Calibration | Answers given | Fakes approved | Fakes caught | Genuine accepted | Genuine rejected | Trap50 (fakes through) | Set has true label |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | LAC α 0.01 (**primary**) | 19.0% | 0 | 52 | 0 | 9 | 18 (0) | 96.2% |
  | LAC α 0.05 (**primary**) | 63.8% | 0 | 118 | 17 | 43 | 36 (1) | 77.8% |
  | Mondrian (per-class) α 0.01 | 0% | 0 | 0 | 0 | 0 | 10 (0) | — |
  | Mondrian α 0.05 | 16.6% | 0 | 35 | 0 | 23 | 11 (0) | 78.4% |
  | One-sided on P(supported), α 0.01 | 64.0% | 0 | 131 | 1 | 54 | 29 (1) | 91.4% |
  | One-sided, α 0.05 | 91.0% | **1** | 149 | **33** | **78** | 44 (2) | 83.0% |
  | Asymmetric (approve at α ≈ 0.004, reject at α), α 0.01 | 61.8% | 0 | 131 | 0 | 54 | 26 (0) | 91.8% |
  | Asymmetric, α 0.05 | 76.0% | 0 | 149 | 0 | 78 | 27 (0) | 85.2% |

  - **Mondrian is degenerate at α 0.01.** The conformal fold has too few `inconsistent_context` rows for the rank, so that label is always in the set and nothing gets answered.
  - **The one-sided rules raise coverage mostly by rejecting more.** Approve iff P(sup) > the (1−α) conformal quantile over non-supported rows; reject iff below the mirror quantile over supported rows. Genuine rejected rises to 54–78 of 132.
  - **The primary calibration was fixed in advance:** Sapling's own LAC method at α 0.01 and 0.05, the same as v1, so v1 vs v2 is a head-only comparison. The others are reported, not selected.

## 4. Two finals, and the ContractNLI fix

**Selection rule, as applied:**
1. The ContractNLI dev third (134 rows) must score ≥ the stock head on the same rows (73.9%).
2. Among those configs, take the lowest ensemble val_tf CE.
3. Epochs = the median best epoch.
4. Refit on the full head fold plus the extra rows (5,219 rows), 5 seeds, no early stopping.

| ContractNLI test, `ev` layout | Stock | v1 | **v2 (selected, L2 pull 1e-3)** | v2 first final (no anchor) |
|---|---:|---:|---:|---:|
| Raw accuracy, all 400 | 77.3% | 83.8% | **84.3%** | **52.0%** ✗ |
| Raw accuracy, untouched two thirds (266) | 78.9% | 85.3% | **85.3%** | n/a |
| Raw accuracy, dev third (134, used for selection) | 73.9% | 80.6% | 82.1% | n/a |
| Raw false `supported`, all 400 | 7 | 8 | 8 | 0 |
| Answers given, ML-only calibration α 0.01 (**not valid for contracts**) | 0% | 81.8% (87.5% right) | 82.5% (88.2% right, 5 false `supported`) | n/a |
| Other layout (`pol`), raw accuracy | 75.0% | 77.8% | 79.3% (317/400) | n/a |

- **What went wrong first:** the unanchored head drifted **121%** from the stock weights (relative L2; v1 drifted 24%). It stopped predicting `supported` on contracts at all: 149 of 150 entailments became `insufficient_information`. On research claims it scored slightly higher (test raw 68.2%; α 0.01: 17.2% answered, 0/157 fakes approved; α 0.05: 66.2% answered, 20/132 genuine accepted, 0 fakes approved).
- **The fix:** an L2 pull toward the stock head (λ 1e-3) holds the drift at **17–18%**.
- **Honesty note:** we added the anchor after seeing the ContractNLI collapse on all 400 rows, which include the dev third, and after the first final had printed its test numbers. No v2 setting was chosen from ml-claims test numbers. The anchor strength was chosen on val_tf plus the dev third only. For a clean no-regression claim, quote the **untouched 266** (78.9% → 85.3% → 85.3%).

## 5. Method and cost

- **Data:**
  - ml-claims v1 train `fold=head` (2,219 rows)
  - **plus 3,000 new rows** from `scripts/make_ml_claims_extra.py`. It reuses `scripts/make_ml_claims.py` unchanged with seed 2026102402. Only train template families and train targets are used; rows whose text matches any existing train, test or trap50 row are dropped (0 dropped). No scenario (target, category, planted numbers) also appears in test (0 overlap); 14 overlap with v1 train. All new rows are head-fold only.
  - **Temperature and conformal folds are v1's, unchanged** (397 / 384 rows).
- **H100 (one run, `busy-machines-sapling`, `sapling_v2x`):**
  - 3,000 rows in 6 chunks, one cold start, model load 25.1 s
  - **0.070 s per case, model median 69 ms**
  - reference example max diff 0.0 vs the published H200 logits; stock head on captured features 0.0 on the GPU and 3.8e-6 batched on the CPU
  - **274 s client wall**
  - billed in the 03:00 UTC hour: **H100 $0.373 + CPU $0.036 + memory $0.024 = $0.43** (the report may still lag)
  - no Sapling containers left running (checked; the last one was already stopped)
  - Sapling H100 spend today is now **$1.08**: $0.16 setup/smoke + $0.55 v1 + $0.37 v2. That is under the $4 D28 cap and inside this step's $1.50.
- **Head training (laptop CPU, $0 GPU):**
  - LoRA rank 256 on W1 (α/r scale 1), plus full b1, W2 and b2
  - L2 pull to the stock head λ 1e-3 (on ‖BA‖², ΔW2, Δb1, Δb2)
  - AdamW lr 3e-4, batch 64, dropout 0.1, 18 epochs
  - 5 seeds (20261024–28): 23 min total; the exploration grid took about 1.5 h of CPU
- **Served model:** the mean of the 5 heads' logits, then Sapling's `ConformalCalibrator` LAC fit on the ML-research folds: α 0.01 T 0.793, q_hat 0.9853; α 0.05 q_hat 0.8158.
- **Release draft** (`data/sapling/sapling_v2/release/`, not published):
  - `heads_ensemble_delta.safetensors`: **94.8 MB** for all 5 heads, sha256 `4b61a35c…`. Per seed it stores `lora_A [256, 14336]`, `lora_B [4096, 256]`, `db1`, `W2`, `b2` on top of the stock head. v1's single head is 235 MB.
  - `calibration_a01.json` and `calibration_a05.json` (LAC), plus the binary and asymmetric variants
  - **Parity:** each seed, merged into Sapling's own `TriVectorInteractionHead`, reproduces the trainer's logits to 6.7e-6.
- **Latency:** the head ensemble was **not measured on the GPU**. Five heads run 5× the head FLOPs, which is small next to the 23B backbone. The backbone forward pass (75 ms median, test rows) is unchanged.

## 6. Parity with Mubit's hosted stock Sapling (D33)

We sent **30 ml-claims rows** to `https://6adbee0669a00298cd4774c9.endpoints.huggingface.cloud`:
- 27 test and 3 trap50 rows; 9 supported, 9 violation, 8 insufficient_information, 4 inconsistent_context (seeded sample)
- **one request at a time**, 0.5 s apart, no auth, synthetic data only

We compared the replies with the stock logits from our H100 pass on the same rows.

| | Result |
|---|---|
| Rows answered | 30 / 30 (no errors, no retries) |
| Raw label agrees | **30 / 30** |
| Calibrated status agrees | **30 / 30** (all `unsure`, as in R33) |
| Input token count agrees | 30 / 30 |
| Max abs logit diff (worst row) | 0.874 |
| Median of per-row max abs diff | 0.242 |
| Mean abs logit diff | 0.173 |
| Smallest raw-label margin on these rows (H100) | 1.66 logits, so no near-ties were tested |
| Hosted model time, median | **1,099 ms** (our H100: 74.8 ms on the same rows); HTTP wall 1.29 s |
| Package example on hosted | [-10.375, 18.163, -8.862, -18.499] vs published H200 [-10.318, 18.048, -8.825, -18.418]: max diff 0.115 (our H100 matched H200 exactly) |

**Reading:**
- Mubit's deployment gives the same decisions as our self-hosted stock run on every row we tried.
- The logits differ by up to about 0.9, which looks like different hardware or kernels: the same package hashes are reported (`weights_sha256 103a67f7…`, `source_release_sha256 245970b0…`).
- So "same logits" holds only to within about 1 logit, not bit for bit. Decisions near a threshold could flip; we didn't test any.
- The hosted endpoint is about 15× slower per call than our H100. Its hardware is not stated.

## 7. What we can and can't say

| ✅ Supported | ❌ Not supported |
|---|---|
| "v2 answers about 3–3.5× as many held-out research claims as v1 (19% vs 6% at α 0.01; 64% vs 18% at α 0.05) and still approves none of 157 fakes." | "It confirms real gains." (0/132 at α 0.01; 17/132 at α 0.05, while 43 are wrongly called fake) |
| "Raw accuracy on held-out research claims 22% (stock) → 58% (v1) → 66% (v2); no regression on ContractNLI (untouched 266 cases: 78.9% → 85.3% → 85.3%)." | "Answers are as reliable as v1's." (answered accuracy at α 0.05 fell 88.9% → 75.5%) |
| "One extra $0.43 H100 pass over 3,000 new synthetic claims was the lever that mattered; loss and calibration tweaks moved little." | "Its conformal coverage holds on new templates." (77.8% at nominal 95%) |
| "Mubit's hosted Sapling and our self-hosted run agree on all 30 sampled decisions." | "Bit-identical to Mubit's endpoint." (logits differ by up to 0.87) |
| "Trap set: 36/50 at α 0.05 with 1 fake let through (v1: 18/50, 2 through)." | "Close to Opus." (Opus 50/50) |

## 8. Caveats

- **Synthetic, templated data** (as in v1). The extra rows come from the same generator and the same train families, so they add scenarios, not new phrasing. Real lab logs are untested.
- **val_tf is only partly held out:** a val row has a new policy *or* evidence family (only rows that have both are fully new), and the claim families are all seen.
- **CPU training is not bit-reproducible across thread counts:** the same config re-run with 4 threads gave ensemble val CE 0.663 vs 0.653 with 8.
- **We saw test numbers twice during development** without them driving a choice:
  1. A 1-epoch smoke run of `final` printed test metrics for the v1 calibration variants before the v2 setting was frozen.
  2. The unanchored final printed test numbers before the anchor search (§4).
  Both are disclosed here; the selection rules (§3, §4) use only val_tf and the ContractNLI dev third.
- **Ensemble, not one head:** the release is 5 heads, so the loader must average logits. Sapling's stock runtime loads one head, so this needs a small loader (not written yet).
- **Sapling stays out of the live race loop** (D28 (6)). Nothing here is an LLM call. The hosted endpoint calls are Mubit's own verifier, used per D33.

## 9. Files and how to reproduce

| File | What |
|---|---|
| `scripts/sapling_stage_v2.py` | `hosted-parity`, `explore` (val_tf grid, ensembles, soups, learning curve, WiSE, ContractNLI dev), `final` (refit, calibrations, evaluation of stock/v1/v2, release) |
| `scripts/make_ml_claims_extra.py` | 3,000 extra train-family rows → `data/sapling/sapling_v2/extra_train.jsonl` |
| `data/sapling/sapling_v2/x_extract/` | records, report, fetched features of the extra H100 pass; `x_extract.log` |
| `data/sapling/sapling_v2/explore*.json/.log` | every validation run in §3 |
| `data/sapling/sapling_v2/final.json`, `final.log` | the selected v2: all tables above, all calibration variants, per-category fakes |
| `data/sapling/sapling_v2/final_unanchored.*`, `release_final_unanchored/` | the first final (ContractNLI 52%); kept as a loss, not a candidate |
| `data/sapling/sapling_v2/release/` | v2 draft release (5-head delta + calibrations), not published |
| `data/sapling/sapling_v2/hosted_parity.json` | the 30 hosted rows, both logit vectors, statuses, timings |

```
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
R="python -m uv run --extra ml --with safetensors==0.8.0 python scripts/sapling_stage_v2.py"
$R hosted-parity
$R explore --seeds 3 --curve lora64_lr3e-4
python -m uv run python scripts/make_ml_claims_extra.py --n 3000
python -m uv run python -m workers.runners.modal_sapling extract data/sapling/sapling_v2/extra_train.jsonl sapling_v2x --chunk 500 --max-usd 1.5 --out data/sapling/sapling_v2/x_extract
python -m uv run python -m workers.runners.modal_sapling fetch outputs/sapling_v2x data/sapling/sapling_v2/x_extract/volume
$R explore --seeds 3 --extra sapling_v2/x_extract --only lora256_lr3e-4,lora256_lr3e-4_l2sp1e-3,lora256_lr3e-4_l2sp1e-4
$R final --seeds 5 --extra sapling_v2/x_extract --config '{"name":"sapling-research-v2","mode":"lora","rank":256,"lr":3e-4,"l2sp":1e-3,"epochs":18,"aggregate":"ensemble","calibration":"lac"}'
```
