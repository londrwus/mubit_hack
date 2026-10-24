# Pre-registration: LCBench wide setup (frozen 2026-10-24 ~14:00 BST, before any race)

Written by the offline gate (workflow `wvceak9k9`, R66) and frozen here before the code or the race exists. No change to the setup, cadence, seeds or success line after seeing results; every seed is logged, including fails.

```
PRE-REGISTRATION: LCBench wide regime, Busy Machines vs the standard Opus agent vs Optuna. Frozen before any race. Report every seed whatever happens, including worse results, in docs/results_log.md as a new R-row. Everything below is PROJECTION/REPLAY until the race runs.

Task: lcbench (YAHPO LCBench surrogate, OpenML task 167168, unchanged; per-run budget 300 predicted s). No task switch. Task 167168 was chosen on 24 Oct for headroom, before this question came up, and its ceiling (~0.89) already allows 0.87.

Compute: runner=surrogate, BUSY_SURROGATE_RUNTIME_MULT=0.1. The env var is process-wide, so all 3 arms get it; run_ref shows x0.1. Each run holds its slot for 0.1 x its predicted LCBench training time (about 9 s median for Opus's configs). Story: a lab's GPU pool running many short tabular/HPO jobs. GPU $ is virtual.

Slots: 16 virtual L4 per arm. Budget 900 s wall clock per arm, all arms start together. max_proposals_per_call 8. run_baseline_first on. No deadline rule (late runs are cancelled at the deadline). Pipelining off. accept_unsure on.

$ cap: 9.20 per arm, same for every arm (virtual GPU 16 x 900 x 0.000222 = $3.20, plus $6 LLM headroom, which is at least 1.5x the projected Opus LLM $ of $3.08, or $4.03 if Opus latency is 0.8x). The old $7.2 cap could stop the Opus arm early. max_estimated_gpu_usd >= 3 x 3.20 = 9.6 (virtual). Mode benchmark, llm=mubit, project A, record off on every LLM arm.

Seeds: 0, 1, 2.

Arms:
(1) opus: the standard agent. Plain v1 loop, propose v1 at effort medium, plain Opus judge v1, no engine features.
(2) opus_bm (Busy Machines with Opus, D46):
- the same model, claude-opus-5-5;
- the lean planner propose v2 at effort low;
- backfill: Optuna TPE, seed = race seed, n_startup 10, constant_liar, LLM proposals launch first, checker-gated feedback 'worst_on_rejected';
- judge v1 + judge policy (D40);
- CADENCE POLICY 'int90': at least 90 s between the starts of the BM planner's propose calls, K = min(8, LLM demand), where LLM demand counts only LLM runs, as scheduler._demand does with backfill. Judge calls are not throttled. Expect ~10 propose calls and ~80 Opus proposals per race.
(3) optuna: the required no-AI baseline, unchanged.
Disclose in the fairness manifest: BM's prompt (v2/low), backfill and the 90 s cadence.

Goal: keep 0.78 frozen (data/calibration/target_values.json) for time-to-goal and cost-to-goal. Secondary stretch line, pre-registered now: held-out balanced accuracy >= 0.87 (count of seeds per arm). It is reported, not used as the goal.

Metrics, per arm, mean ± sd over n=3:
- GPUs waiting %
- LLM $ per race (estimate at list price)
- best held-out balanced accuracy (counted runs: non-trap, finished)
- shipped held-out
- fakes accepted / shipped
- trap claims and judge calls
- experiments (LLM vs sampler)
- time and $ to 0.78
- seeds reaching 0.87
- propose calls, propose p50 latency, sampler ask time

SUCCESS LINE (all must hold on the 3-seed means):
- Opus waiting >= 40%
- BM waiting <= 2%
- BM LLM $/race <= 0.8 x Opus
- BM best held-out >= Opus best held-out
- BM best held-out >= Optuna best held-out
If any part fails, log it as a fail. No re-runs, no swapping seeds, no changing the cadence or the regime after seeing results.

Prerequisites not built yet (no code was changed here): the cadence knob (e.g. RaceConfig min_propose_interval_s=90 applied to opus_bm only); lean v2/low on opus_bm (D46); a new preset, e.g. 'lcbench-wide-bm-15m', with the settings above. headline.py must not pool this race with R38/R44/R65 (different fingerprint).
```

## Addendum (2026-10-24 ~14:10 BST, still before any race): primary endpoints re-ordered by the user
The pitch problem is **better models and less money for that quality**; GPUs waiting is a mechanism, not the headline. So the race report leads with these fixed-formula endpoints (all already computed by `headline.py`), per arm, mean ± sd over seeds 0–2:
1. **Same quality, less money:** total $ (GPU + LLM) and time to reach the standard Opus arm's final best held-out score (paired per race), and to reach 0.86.
2. **Same money, better model:** best held-out score at the standard Opus arm's total spend.
3. **Quality per dollar:** best held-out score and total $ per race.
The frozen success line above is unchanged and still reported line by line; nothing is dropped.
