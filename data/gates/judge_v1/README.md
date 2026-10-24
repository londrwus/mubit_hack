# D39 gate: closed-output judge_improvement v1 (D37)

Run 2026-10-24 with `scripts/gate_judge_v1.py`. Every call went through `MubitClient` (project A, `MUBIT_API_KEY`) as step judge_improvement (effort low, max_tokens 4000). Recording was off (`x-middleware-record: off`): 0 of 408 calls were recorded, so neither draft created or fed a Mubit task. The raw numbers are in `results.json`, and every call (with its Mubit request id) is in `calls/*.jsonl`.

## Files
| file | what |
|---|---|
| `judge_improvement.v1.draft.md` | draft v1 (fingerprint `5fea39984f30`, 4,859 chars sent) |
| `judge_improvement.v1b.draft.md` | **recommended**: v1 plus one sentence in case 4 (fingerprint `5c1d3cfd2941`, 5,180 chars sent, about 1.1k tokens) |
| `results.json` | the full report: agreement, disagreements, flip pairs, class balance, accuracy by category, cost, latency, request ids |
| `calls/{variant}.{model}.pass{n}.jsonl` | one row per call |

Both drafts keep v0's instructions and decision rules. They replace the free-text `reason` with a closed output, `{"verdict": valid|invalid|unsure, "reason_code": <enum>}`, built from 5 ordered cases where the first case that applies fixes both fields:

| # | reason_code | verdict |
|---|---|---|
| 1 | `contamination_or_eval_leak` | invalid |
| 2 | `eval_setup_changed` | invalid |
| 3 | `gain_within_noise` (spread shown, gain not above it, or a lucky best seed) | unsure |
| 4 | `insufficient_evidence` (nothing that rules out leakage; run metrics alone don't) | unsure |
| 5 | `clean_gain` | valid |

The wire schema is closed: two string enums, `additionalProperties: false`, no free text.

v1b adds one sentence to case 4: distinct train and held-out hashes, with no leaking option enabled, rule leakage out even when overlap counts or selection details are not listed. I tuned it on 2 trap-set **dev** cases (`trap_v1_030`, `trap_v1_042`), which v1 judged `insufficient_evidence` on both passes. v0 had judged them valid. No test-split case was read.

## Cases (68)
- All 50 cases from `trap_set_v1` (25 valid, 25 invalid; planted, with known `acceptable_verdicts`).
- 18 cases from `noise_set_v1` in race format (run metrics only): 7 real (all of them), 7 noise, 4 fake. Ground truth comes from 10-seed reruns. Accepted verdicts: real → valid, noise → unsure or invalid, fake → invalid (the same convention as `scripts/eval_noise_set.py`).
- By target: rl 26, cifar 25, lm 17. The noise set has no lm cases.

## Results
| | v1 | **v1b** |
|---|---|---|
| Opus self-agreement, full output (pass 1 vs 2) | **68/68 = 100%** | **68/68 = 100%** |
| Opus self-agreement, verdict only | 68/68 = 100% | 68/68 = 100% |
| Haiku 5.5 vs Opus pass 1, full output | 64/68 = 94.1% | 65/68 = 95.6% |
| Haiku vs Opus, verdict only | 94.1% | 95.6% |
| Opus verdict balance | unsure 23 / valid 23 / invalid 22 (max 33.8%) | valid 25 / invalid 22 / unsure 21 (max 36.8%) |
| Opus reason_code balance | clean 23, leak 19, insufficient 16, noise 7, eval_changed 3 | clean 25, leak 19, insufficient 14, noise 7, eval_changed 3 |
| Opus accuracy, trap set (both passes) | 48/50 (valid 23/25, invalid 25/25) | **50/50** (dev 30/30, test 20/20) |
| Opus accuracy, noise set | 11/18 (fake 4/4, noise 7/7, real 0/7) | 11/18 (the same) |
| Haiku accuracy, trap set | 50/50 | 49/50 |
| Code/verdict pairing violations | 0 | 0 |
| Opus p50 latency, mean output tokens | 1.7 s, 33 | 1.7 s, 30 |
| Gate (≥97% full self-agreement, no class >70%) | pass | **pass** |

- **Opus never flips.** Neither variant changed a single code between passes, so there are no flip pairs to report. I did not run a merged-code or verdict-only variant because none was needed (`v1m` is defined in the script but has no prompt file and was never run).
- **Haiku's disagreements with Opus** (v1b): 2× `unsure/insufficient_evidence` ↔ `valid/clean_gain` (`ns_v1_013`, `trap_v1_014`) and 1× `invalid/eval_setup_changed` ↔ `unsure/insufficient_evidence` (`ns_v1_052`). In v1, all 4 disagreements were the insufficient ↔ clean pair. That boundary (case 4 vs case 5) is the hardest one for a small model.
- **First request ids:**
  - v1: Opus pass 1 `mreq_01a16bab7d967d6383806b5e80f97b5a`, pass 2 `mreq_01a16baca2287231a20e7271c088b517`; Haiku `mreq_01a16bad4d6c7e1295ab498f5ea685e3`.
  - v1b: Opus pass 1 `mreq_01a16baf582f7450b13ae8b879b9cdc5`, pass 2 `mreq_01a16bb0001d7bb3b1bd649d9a72cfd1`; Haiku `mreq_01a16bb0aca3723391170ffb0f2e0b43`.
- **Cost** (estimated from the usage meters): v1 $0.551, v1b $0.569, **total $1.120** against the $1.50 cap.

## Recommendation
**Freeze v1b** as `judge_improvement.v1`: copy `judge_improvement.v1b.draft.md` unchanged, so the sent text keeps fingerprint `5c1d3cfd2941`. Its wire schema is the 2-enum schema in `wire_schema("v1b")`. It passes criteria (1) and (4) of the D39 gate on this set. It also matches v0's 50/50 accuracy on the trap set, whereas v1 drops 2 clean cases.

Caveats and the remaining D39 items:
- 68/68 has a one-sided 95% lower bound of about 95.7% (rule of three). It is strong evidence for ≥97%, not proof. Re-check self-agreement on the first recorded batch.
- D39 (3) still needs **≥150 distinct inputs** in the recording set. This gate used 68.
- `eval_setup_changed` is rare (3/68, 4%). Unless the recording set adds more changed-eval claims, the adapter will see few examples of it.
- In race-format evidence (no split hashes), Opus answers `insufficient_evidence` on every clean claim: 0/7 real gains were confirmed. This is v0's behaviour carried over (R25), not a regression. In live races the judge's classes will be mostly invalid or unsure unless race evidence starts to carry split hashes.
- Haiku zero-shot agreement with Opus (95.6%) is a lower bound for an adapter. It sits right at the exam's 95% target, so the trained adapter should clear it, but this is not yet measured.
