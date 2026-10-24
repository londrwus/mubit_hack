# Published Banyan Sapling vs our fine-tune vs Opus 5.5

**The question this answers:** did we just reuse Mubit's model, or did we improve it? We fine-tuned Banyan Sapling's verification head, backbone frozen (D23), and refit its calibration for ML-research claims. The result is `sapling-research` v2, a community fine-tune. It is **not** an official MubitLabs model, and MubitLabs has not reviewed it. In races the checker is still Opus plus a seed-spread rule. Sapling runs offline only (D28).

**Status of the numbers (24 Oct):**
- Rows marked **NEW, to be logged** come from `data/sapling/compare_published/compare_published.json` and `per_row.jsonl` (`scripts/sapling_compare_published.py`). They are not in `results_log.md` yet. Don't put them on a slide or say them on stage until their R-row exists.
- Every other number cites its R-id.

## The three verifiers
| | What it is | How it decides |
|---|---|---|
| **Published Sapling** | `MubitLabs/banyan-sapling` as released: stock head and its own calibration | Gives one of 4 labels, or `unsure` when it isn't confident enough |
| **Our v2** | The same frozen backbone with our 5-head verification-head ensemble and a calibration for ML-research claims | The same, at two strictness settings: strict (α 0.01) and looser (α 0.05) |
| **Opus 5.5** | Our frozen judge prompt v1 through the Mubit gateway | Says valid, invalid or unsure. It never abstains |

## 1. Real claims from our own races (NEW, to be logged)
There are 226 claims in the live judge's format. They come from race traps, collection runs and the noise set. Ground truth exists for 152 of them: 109 fakes, 36 noise and 7 real gains.
- v2 and published Sapling never saw this format.
- 186 of the rows trained v3, which failed its gate (R77).
- 77 rows carry labels taught by Opus, so Opus is graded on the 152 ground-truth rows only.

| On the 152 rows with ground truth | Published Sapling | v2 strict | v2 looser | Opus 5.5 |
|---|---:|---:|---:|---:|
| Fakes caught | 0 / 109 | 8 / 109 | **57 / 109** | **109 / 109** |
| Fakes approved | 0 | 0 | 0 | 0 |
| Noise accepted as real | 0 / 36 | 0 / 36 | 0 / 36 | 0 / 36 |
| Real gains confirmed | 0 / 7 | 0 / 7 | 0 / 7 | 0 / 7 |
| No answer (`unsure`) | 152 / 152 | 105 / 152 | 24 / 152 | none (no abstain option) |
| Answer allowed by ground truth | 0 / 152 | 39 / 152 | 95 / 152 | 152 / 152 |
| Agrees with Opus, all 226 rows | 0 / 226 | 81 / 226 | 163 / 226 | n/a |

- **Fakes caught by v2 looser, by source:** collection traps 36/60, race traps 7/17, planted noise-set fakes 14/32.
- **Raw top label, no calibration (secondary):** published Sapling catches 4/109 fakes and v2 catches 69/109. v2 calls 1 of the 7 real gains `supported`.
- **Checks:** the stock head reproduces the H100 package logits to 3.8e-6, the published calibration reproduces the package statuses on 226/226 rows, and the v2 loader reproduces the logged logits to 2.4e-6.

## 2. Synthetic research claims (logged)
These are our own `ml-claims` sets from our generator, not Mubit's data.

| | Published Sapling | v2 | Opus 5.5 | Source |
|---|---:|---:|---:|---|
| 500 held-out claims: raw accuracy | 22.0% | 66.0% | not run | R42 |
| Answers given (strict / looser) | 0% | 19.0% / 63.8% | n/a | R42 |
| Fakes caught, of 157 (strict / looser) | 0 | 52 / 118 | not run | R42 |
| Fakes approved, of 157 | 0 | 0 | not run | R42 |
| Tasks never seen in training: raw accuracy | 27.4% | 67.0% | not run | R42 |
| Trap set, 50 cases (strict / looser) | 10 / 50 | 18 / 50 (0 fakes through) / 36 / 50 (1 fake through) | **50 / 50** | R42, R24 |
| ContractNLI, 400 of our cases: raw accuracy | 77.3% | 84.3% | not run | R42 |
| ContractNLI, the 266 untouched cases | 78.9% | 85.3% | not run | R77 (v2), `sapling_results_v2.md` (stock) |
| Speed per check on an H100 | 95.7 ms | 99.4 ms | p50 1.65–1.70 s (judge v1) | R76, R40 |
| Cost per check | ~$0.00015 | ~$0.00015 | ~$0.0046 (judge v0) | R33 |

**NEW, to be logged (speed and cost):**
- Published Sapling: median 105.2 ms on the 226 rows (p90 115.8 ms).
- Opus judge v1 on those rows: p50 1,655.7 ms and p90 2,307.5 ms over 405 calls, at $0.00338 mean and $0.00306 median per call.
- v2 at steady state: about $0.00011 per check. This is an estimate (0.0994 s × $3.95/h) that leaves out the 44 s model load.

## Where we lose (say these beside the wins)
- **Opus is far better.** It caught 109 of 109 fakes on real claims, against 57 for v2 (NEW). It scored 50/50 on the trap set, against 18 or 36 for v2 (R42, R24). Sapling's edge is speed and price, not quality.
- **No verifier confirms real gains on its own.** All three confirm 0 of 7 (NEW). On synthetic claims, v2 confirms 0/132 at the strict setting and 17/132 at the looser one (R42). Our race checker gets 7/7 only by adding the seed-spread rule to Opus (R37).
- **The looser setting is less trustworthy.** Its 95% guarantee falls to 77.8% on new claim types (R42). Its answers are right less often than v1's (75.5% vs 88.9%). It wrongly rejects more genuine gains (12 → 43 of 132).
- **The synthetic tables are synthetic.** Every number in them comes from claims our own generator planted. The real-claims table is small: only 7 real gains, so quote counts, not rates.
- **v3 failed.** Training on these judge-format rows made the model worse on 5 of 7 gate checks. We kept v2 (R77).
- **The ContractNLI number is not a reproduction of Mubit's card.** The card reports 86.32% on its own 731 ContractNLI cases. Our 400 cases are a stratified draw from the `contractnli_a` test split, in our own input layout, because Sapling's layout isn't published. Published Sapling scores 77.3% on them. "77.3% → 84.3%" compares published Sapling and v2 on the same rows (R33, R42).
- **The unanchored first v2 collapsed** to 52.0% on ContractNLI. We added the anchor after seeing that result (R42).

## The pitch sentence
**Once the 226-row numbers are logged (exact wording):**
> "We fine-tuned Banyan Sapling's verification head, backbone frozen, for AI research. On 226 real claims from our races, the published model answered none; ours, which had never seen that format, caught 57 of 109 fake improvements and approved none, in about a tenth of a second per check. Opus 5.5 is still the better judge: it caught all 109."

**Until then (logged numbers only, R42):**
> "We fine-tuned Banyan Sapling's verification head, backbone frozen, for AI research. On 500 research claims it had never seen, the published model answered none; ours catches 52 of 157 fakes at its strict setting and approves none. Opus 5.5 is still the better judge: 50 of 50 on our trap set, against our 18."

**Never say:**
- "Mubit's model", "Mubit's official fact-checker" or "MubitLabs released" about our fine-tune.
- "Matches Opus" or "beats Opus".
- "Confirms real gains".
- "Answers instead of unsure" (D23, `sapling_results.md` §5).
- "Reproduces Sapling's ContractNLI score".
- "Full fine-tune" or "LoRA fine-tune".

**Rerun (CPU only, from cached features, no Modal and no LLM):**
`python -m uv run --extra ml --with safetensors==0.8.0 python scripts/sapling_compare_published.py`
