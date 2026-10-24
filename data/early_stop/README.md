# early_stop (D34): cases, gate and evaluation

The 4th recurring research step: at fixed checkpoints of a running experiment, decide `continue` or `stop` from its curve so far. Built by `scripts/collect_early_stop.py`; prompt `backend/agent/prompts/early_stop.v2.md`; builder and schema in `backend/agent/prompts/build.py`; output type `EarlyStopOutput` in `backend/core/types.py`.

**Status (2026-10-24 ~03:20 BST): no recorded calls in either Mubit project.** The D39 gate failed (Opus self-agreement 95.0% < 97% on the frozen decision-only prompt), so `record` refuses. All 509 calls so far (489 logged + 20 in the deleted first draft) were sent with `x-middleware-record: off` on project A.

## Files
| File | What |
|---|---|
| `cases.jsonl` | 1,121 candidate cases (rl 848, cifar 209, lm 64), with split (`record` 237 / `eval` 149 / `unused` 735), state, labels and the median-rule decision |
| `draft_calls.jsonl` | Draft iterations (Haiku and Opus, record off), keyed by prompt fingerprint |
| `gate_calls.jsonl`, `gate_summary_<fp>.json` | D39 gate: the same 40 cases sent to Opus twice and to Haiku once, record off |
| `eval_calls.jsonl` | Opus on the 149 held-out eval cases, record off |
| `summary.json` | `report` output: all metrics below |

## Cases
- Source: `data/collection/collect_{rl,cifar,lm}.db` (opened read-only, `mode=ro`). Cases are succeeded, non-probe, non-trap runs.
- Checkpoints: 25% and 50% of the target's per-run training budget (rl 30 s, cifar 45 s, lm 60 s), measured in **training** seconds. Training start = last training-time metric t minus the run's own train-time (`train_s` / `train_time_s`), so cold start and JIT are excluded. A checkpoint is skipped if the run finished before reaching it.
- Curve metric (`EARLY_STOP_CURVE_METRIC`): rl `train_return` (the only signal RL streams), cifar `val_acc`, lm `val_bpb`. The direction is the same as the target metric. At most 12 points are shown, evenly spaced, with the first and latest always kept, at 4 significant figures.
- Context: runs of the **same race** that had completed by the checkpoint's wall-clock time (`started_at + t_train0 + elapsed`). The brief said "finished before that run started"; I used the checkpoint time because that is what a live scheduler knows. Judge-invalid and trap runs are excluded; probe runs count, because they were real history. At least 3 completed runs are required (`MIN_COMPLETED`), otherwise the step is not called.
- The model sees: the run's config, the training time used out of the budget, the curve, the completed count, the best final value, the median final value, and the completed runs' **median and best-run value at the same training time**. I added the last two so the LLM gets at least what the median rule uses. It never sees held-out scores, labels or trap flags; `test_early_stop.py` checks this.
- Labels (never shown to the model): `beats_val` means the run's final curve value beat the best-at-the-time by more than `MARGIN_VAL` (rl 10, cifar 0.002, lm 0.005). `beats_heldout` is the same test on the held-out score (margins rl 10, cifar 0.002, lm 0.01). **`stop_correct` = neither.** Val and held-out labels agree on 842/848 rl, 194/209 cifar and 54/64 lm cases.
- **Winners are rare:** 35 of 1,121 cases (3.1%) and 18 of 566 runs. The natural winner-run share is rl 1.2%, cifar 7.3%, lm 15.2%.
- Splits are by run. **Eval**: half of each target's winner runs (rounded up) plus random other runs, 149 cases from 76 runs. Winners are oversampled on purpose, so budget saved is also reported reweighted to the natural winner share. **Record**: up to rl 110 / cifar 90 / lm 50 cases from the other runs, with winners oversampled toward 40%. Only 15 winner cases exist outside eval, so the record set is 237 cases, 15 of them `continue_correct`.

## Prompt history (no version was ever recorded)
| Version | Output | Fingerprint | What happened |
|---|---|---|---|
| v0 draft 1 | decision + free-text reason (≤15 words) | `38d570702e38` | Haiku, 20 cases: 20/20 valid, **all `continue`**, $0.0021, first `mreq_01a16b9e0b167101b3e3b6fec3f71df9`. I deleted these call rows before switching to per-fingerprint logging; the numbers are from the console |
| v0 draft 2 | same | `1472bb36961c` | Added "winners are rare". Haiku 9/20 correct, 2 stops |
| v0 (frozen) | same | `91deaf91052d` | Added "still improving / near the median is not a reason". Haiku 9/20 (2 stops); **Opus 12/20** (9 stops), $0.0865, p50 2.1 s. Frozen, then superseded before any recording by the coordinator's closed-output instruction |
| v1 (frozen) | decision + `reason_code` (6-value enum) | `a470db926989` | Haiku 9/20, Opus 11/20 on the draft cases. **D39 gate failed** (below). Never recorded |
| **v2 (frozen, current)** | **decision only** | **`f1f3bf21aa79`** | **D39 gate failed** (below). Never recorded. Recording is blocked in code until a gate summary for this fingerprint passes |

The draft cases are 20 non-eval cases with balanced labels, so their accuracies are not comparable with the eval numbers.

## D39 gate (40 diverse non-eval cases; per target, winners up to half, the rest spread over fraction × median-rule strata)
Ceiling needed: Opus self-agreement ≥ 97% on the graded output. Mubit's exam needs ≥ 95% match with the stored answer.

| Prompt | Graded output | Opus run 1 vs run 2 | Haiku 5.5 vs Opus run 1 | Largest class | Verdict |
|---|---|---|---|---|---|
| v1 `a470db926989` | decision + reason_code | **92.5%** (37/40) | 67.5% | 56% (`stop/behind_median` 21, `continue/on_best_pace` 45 … of 80) | FAIL |
| v1 | 4 merged codes (post hoc) | 92.5% | 70.0% | 59% | FAIL |
| v1 | decision only (post hoc) | 97.5% (39/40) | 77.5% | 69% `continue` | indication only |
| **v2 `f1f3bf21aa79`** | **decision only** | **95.0%** (38/40) | **70.0%** | 65% `continue` (52/80) | **FAIL** |

- Pooled decision agreement over both rounds is 77/80 = 96.25%. One lm boundary case (`lm:114a2bf3…:50`) flipped in both rounds. The ceiling is just out of reach: Opus is stochastic on cases near the median/best boundary.
- Class balance passes (≤ 70%). Distinct candidate inputs: 1,121 (≥ 150, passes).
- Gate cost: v1 $0.2434, v2 $0.2479, **$0.49 total**. First request ids: v1 `mreq_01a16ba9506b7893959be62437c41c32`, v2 `mreq_01a16bac56ad75f1896fe52fec24fbf9`.

## Evaluation: Opus (v2, record off) vs ground truth on the 149 held-out cases (76 runs, 10 winner runs)
Policy simulation per run: stop at the first checkpoint where the decision is `stop`. Budget saved = training seconds not spent ÷ the runs' training seconds.

| Policy | Case accuracy | Winner cases stopped | **Winner runs wrongly stopped** | Runs stopped | **Budget saved** (raw / natural mix) |
|---|---|---|---|---|---|
| **Opus 5.5 (early_stop v2)** | **29.5%** | 11/20 | **7/10** | 36/76 | **34.5% / 33.4%** |
| Median-stopping rule (Optuna-style) | 43.6% | 13/20 | 8/10 | 52/76 | 47.0% / 46.7% |
| Always continue | 13.4% | 0/20 | 0/10 | 0/76 | 0% |
| Always stop at 25% | 86.6% | 20/20 | 10/10 | 76/76 | 71.2% |

Per target (Opus vs median rule): rl accuracy 29.2% vs 34.7%, winner runs stopped 1/3 vs 1/3, saved 30.8% vs 36.2%. cifar 22.0% vs 52.0%, 3/4 vs 4/4, 24.7% vs 45.0%. lm 44.4% vs 51.9%, 3/3 vs 3/3, 50.0% vs 62.5%.
Median rule on all 1,121 cases: accuracy 40.6%, winner runs stopped 12/18, budget saved 38.2%.

Opus: 149/149 valid, 0 retries, **p50 1.51 s**, p90 3.00 s, **$0.3927** ($0.0026 per call: about 2.4k cached input, 360 uncached input and 36 output tokens). First request `mreq_01a16baef22d70c086bc24bfcbf25dcd`.

**Reading:** on these curves, Opus as early-stopper is **worse than the free median rule**: it saves less budget (34.5% vs 47.0%) and is less accurate. It wrongly stops one winner run fewer (7 vs 8 of 10), but that difference is noise at n = 10. Neither policy can see winners coming: about 70–80% of eventual winners look ordinary at 25–50% of their budget, especially on RL, where the training-return curve saturates within seconds. Accuracy is low for both because 87% of eval cases are `stop_correct` and both policies `continue` many ordinary runs. These numbers are not in `docs/results_log.md` (not edited here, by instruction); log them as a new entry, including that Opus lost to the rule.

## Spend
LLM total **$1.069**, under the $4 cap: drafts $0.185 + gate $0.491 + eval $0.393 (+$0.0021 from the deleted first draft). **Recorded calls: 0 in project A, 0 in project B.** The first `record` attempt on project A was stopped before any reply was written (empty log, no `record_calls.jsonl`); at most the single warm-up request may have reached Mubit, without a reply. No GPU.

## Commands
```
python -m uv run --extra ml python scripts/collect_early_stop.py cases
python -m uv run --extra ml python scripts/collect_early_stop.py draft --n 20 [--version vN] [--model claude-opus-5-5]   # record off
python -m uv run --extra ml python scripts/collect_early_stop.py gate --pass opus1|opus2|haiku [--version vN]            # record off, project A
python -m uv run --extra ml python scripts/collect_early_stop.py gate-report [--version vN]
python -m uv run --extra ml python scripts/collect_early_stop.py record --yes --project b   # refuses until gate_summary_<fp>.json passes (D38: project B)
python -m uv run --extra ml python scripts/collect_early_stop.py eval --yes                 # record off, project A
python -m uv run --extra ml python scripts/collect_early_stop.py report
```
`--project b` reads `MUBIT_API_KEY_B` from `.env` with `dotenv_values` into this process's `MUBIT_API_KEY` only (the key is never printed). It retries HTTP 429 with backoff, and concurrency is ≤ 4.

## Scheduler integration spec (not implemented: `backend/scheduler.py` belongs to another agent)
1. **When.** For each running experiment on an LLM arm, at **training** time `elapsed = 0.25 × spec.budget_s` and `0.5 × spec.budget_s`. Use a timer from training start, and use the curve points received up to then. Call only if (a) the arm has ≥ 3 completed runs (succeeded, not judge-invalid), (b) the run has ≥ 1 point of `early_stop_curve_metric(spec)`, and (c) the run has not finished. One call per checkpoint per run.
2. **Training start.** `MetricPoint.t` counts from launch, cold start included. Best: the targets emit a `train_start` point (step 0) when training begins; this needs a change in `workers/targets`. Until then, estimate it as `t_first − step_first × (t2 − t1)/(s2 − s1)` from the first two curve points: accurate on rl (~0 s) and lm (~1 s), but about 10 s late on cifar because the first epoch includes compile. The offline cases used the exact start, so use the marker for parity.
3. **State.** Build `EarlyStopState` exactly as `scripts/collect_early_stop.py::build_cases` does: CurvePoints on the training-time axis; `best_final` / `median_final` over the completed runs' last curve value; `median_at_point` = median over completed runs of their value at the same elapsed time (the last point ≤ elapsed); `best_run_at_point` = the best-final run's value at that time. Then call `build_early_stop_user(spec, state)`, system `load_system_prompt(StepType.early_stop)`, schema `EarlyStopOutput`, `wire_schema_for(StepType.early_stop)`, effort low (`DEFAULT_STEP_CONFIGS`), the arm's session id and the arm's record policy (off in benchmark races). Use the same `call_with_fallback` policy as the other steps.
4. **Non-blocking.** The run keeps training while the call is in flight; the call never blocks `propose_experiments`. If there is no valid reply after the retry/fallback, or no reply within 10 s, **continue** (safe default). If the run ends before the reply, ignore the reply.
5. **Cancel semantics.** On `stop`: `runner.cancel(handle)`. The experiment becomes `cancelled`, with `summary_json.early_stopped = {fraction, elapsed_s, decision_call_id}`. It has no score, does not count on the leaderboard, and is never judged or diagnosed. The slot is free at once and counts toward K in the next `propose_experiments` call. Its partial curve stays visible. In the UI, the explanation is generated **from the numbers in code** (for example "behind the typical run at this point"), not from model text.
6. **Budget accounting.** GPU $ accrue until the cancel is acknowledged (`gpu_intervals` busy up to then). Log per experiment `early_stop_saved_s = budget_s − elapsed_at_cancel` (and × price per second), and per arm the totals: runs stopped, training seconds and $ saved, early_stop calls, LLM $ and p50 latency. The early_stop LLM $ go into the arm's LLM $. The time the stop decision takes is also wasted run-time and must not be hidden.
7. **Identical per arm.** Same checkpoints (0.25, 0.5), `MIN_COMPLETED = 3`, state builder, timeout and default on every LLM arm (the Opus arm and the Mubit arm send the same model ID; Mubit routes if an adapter is ever live). The **Optuna arm uses the median rule on the same checkpoints and the same state**: stop iff `median_at_point` exists and the latest value is strictly worse than it, which is the `median_rule` field in `cases.jsonl`. No LLM. Record the policy per arm in the race's `fairness_json`.
8. **Recommendation from the measurements.** The gate failed, so there is no adapter path for this step today. The eval also shows Opus is worse than the median rule here. The fair, cheaper option is to **run the median rule on every arm as deterministic infrastructure** (like `summarize_result`) and keep the LLM early_stop step out of the benchmark races, unless a v3 prompt passes both the D39 gate and a re-run eval that beats the rule. A larger gate (≥ 100 cases) would also narrow the 95–97.5% uncertainty.
