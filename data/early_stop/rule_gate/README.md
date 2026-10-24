# D42 gate: early-stopping engine rules (offline replay + race simulation)

Built by `scripts/gate_early_stop_rules.py` (read-only on every DB, no LLM, no Modal). Full numbers for all 27 rules × 5 regimes are in `results.json`. Run it with:
```
python -m uv run --extra ml python scripts/gate_early_stop_rules.py --seeds 20
```

**Verdict: no rule passes on any target. Recommendation: `opus_bm` / `mubit` run with no early stopping on rl (30 s and 15 s), cifar, lm and lcbench (judge policy only, per D42).** The Optuna arm must then also run without the median rule, because the scheduler requires Optuna's median rule to match the BM arms.

Not logged in `docs/results_log.md` (not edited here, by instruction). It should be logged as a new entry.

## Data
| Regime | Source | Runs (candidates) | Notes |
|---|---|---|---|
| rl30 | `collect_rl.db`, 3 races × 8 slots, 30 s runs | 526 (444) | Opus collection races, never stopped |
| rl15 | `bench.db` race `f2bee077…` (R41), **opus arm only** | 465 (461) | The opus_bm and optuna runs were stopped, so their curves are censored and not used |
| cifar | `collect_cifar.db`, 2 races × 3 slots, 45 s runs | 153 (114) | |
| lm | `collect_lm.db`, 2 races × 2 slots, 60 s runs | 56 (38) | Only 4 top-10% runs, so even 1 wrong stop is 25% |
| lcbench | `bench.db`, 3 surrogate races, opus + optuna arms | 1,079 (977) | Sim pool = opus runs only (266) |

Candidates are succeeded runs that are not baselines, traps or collection probes. A reference run also has to not be judged invalid. "Score" is the run's reported score, which is what the leaderboard ranks by. On rl it equals the held-out-levels score for non-trap runs, so the two top-10% columns are identical there. `results.json` also has top-10% by final curve value.

## Method (what is faithful to `backend/scheduler.py` + `MedianStopRule`)
- **Time axis:** checkpoints are at `f × run_budget_s` on `MetricPoint.t` (seconds since launch), as the scheduler does it. This differs from `cases.jsonl`, which used training time. For lcbench: 300 s × multiplier 1, so most runs end before 75 s and are never checked.
- **When a run is checked:** at the first curve point with `t ≥ c`, using the last value at or before `c`. Each checkpoint is checked once. A skip for too few references is final, and the first stop wins.
- **Threshold:** the `p`-th percentile (linear interpolation) of the reference values, direction-aware. `p = 0.5` is exactly the current median rule.
- **Closed loop:** the reference is the same arm's runs that *completed* before the decision on the wall clock. **Runs this rule stopped never enter the reference.**
- **Stop latency:** a stop lands 1.87 s later (the measured R41 median over 2,115 stops), or 0 s on the lcbench surrogate. If the run would have ended first, it completes normally.
- **Event simulation:**
  - Setup: 30 slots, a 900 s window, and a launch cutoff equal to the p90 run duration.
  - Planner: one sequential planner with `K = min(8, free slots)`. Latency per call is either Opus at `2.48 + 1.65·K` s (R36), or a flat 1 s or 2 s.
  - Runs: bootstrap draws from the regime's pool, with common random numbers across rules over 20 seeds. Judge and diagnose calls are not modelled.
  - **Validation:** the sim of rl15 with no stopping and Opus latency gives **71.2% GPUs waiting vs 70.0% measured in R41**. Replaying the raced rule (median @ 25%+50%, n≥5) on the rl15 opus curves stops **85.9%** of runs (R41 measured 92% on opus_bm and 74% on optuna).
- **Gate (D42):**
  - (a) top-10% wrong-stop ≤ 5%, by score AND by held-out
  - (b) slot time freed ≥ 20%
  - (c) GPUs waiting in the Opus-latency sim ≤ no stopping + 0.5 pp

## Results (compact; the full grid is in `results.json`)
For each regime the table shows the default rule, the rule that frees the most slot time while keeping (a), and the rule with the least harm to top-10% runs while keeping (b).

| Regime | Rule | Runs stopped | Top-10% stopped (score / held-out) | Slot time freed | Waiting, Opus sim (Δ) | Waiting 2 s | Waiting 1 s | Best score, Opus sim (shipped held-out) | Gate |
|---|---|---|---|---|---|---|---|---|---|
| rl30 | no stopping | 0% | 0 | 0% | 48.1% | 8.3% | 4.4% | -158.7 (-158.7) | baseline |
| rl30 | median @ 25%+50%, n≥5 (current default) | 65% | 33/45 / 33/45 | 41.0% | 77.6% (+29.5) | 17.2% | 10.6% | -159.6 (-159.6) | FAIL a,c |
| rl30 | p10 @ 25%+50%, n≥10 | 7% | 0/45 / 0/45 | 4.4% | 60.8% (+12.7) | 11.5% | 6.0% | -158.8 (-158.8) | FAIL b,c |
| rl30 | p25 @ 25%+50%, n≥5 | 33% | 15/45 / 15/45 | 21.2% | 70.4% (+22.3) | 14.2% | 8.2% | -159.0 (-159.0) | FAIL a,c |
| rl15 | no stopping | 0% | 0 | 0% | 71.2% | 15.2% | 7.9% | -158.0 (-158.0) | baseline |
| rl15 | median @ 25%+50%, n≥5 (as raced in R41) | 86% | 44/47 / 44/47 | 53.8% | 87.9% (+16.7) | 28.2% | 16.8% | -160.1 (-160.1) | FAIL a,c |
| rl15 | p10 @ 75%, n≥5 | 12% | 2/47 / 2/47 | 2.6% | 73.3% (+2.2) | 15.7% | 8.5% | -158.0 (-158.0) | FAIL b,c |
| rl15 | median @ 50%, n≥20 | 63% | 30/47 / 30/47 | 27.7% | 80.7% (+9.5) | 19.2% | 12.0% | -158.3 (-158.3) | FAIL a,c |
| cifar | no stopping | 0% | 0 | 0% | 35.0% | 6.8% | 3.3% | 0.9498 (0.9407) | baseline |
| cifar | median @ 25%+50%, n≥5 (current default) | 82% | 11/12 / 11/12 | 46.1% | 66.3% (+31.2) | 12.5% | 6.8% | 0.9491 (0.9408) | FAIL a,c |
| cifar | p10 @ 50%, n≥20 | 12% | 0/12 / 0/12 | 6.2% | 40.5% (+5.5) | 7.6% | 3.7% | 0.9497 (0.9396) | FAIL b,c |
| cifar | p10 @ 25%+50%, n≥10 | 36% | 6/12 / 5/12 | 21.3% | 49.7% (+14.7) | 8.9% | 4.4% | 0.9493 (0.9397) | FAIL a,c |
| lm | no stopping | 0% | 0 | 0% | 19.7% | 4.5% | 2.2% | 2.124 (2.495) | baseline |
| lm | median @ 25%+50%, n≥5 (current default) | 55% | 4/4 / 3/4 | 34.7% | 49.9% (+30.2) | 9.1% | 4.5% | 2.135 (2.533) | FAIL a,c |
| lm | p10 @ 75%, n≥10 | 8% | 0/4 / 0/4 | 1.0% | 21.3% (+1.6) | 4.7% | 2.3% | 2.125 (2.498) | FAIL b,c |
| lm | median @ 25%+50%, n≥10 | 39% | 3/4 / 2/4 | 24.7% | 46.9% (+27.2) | 9.1% | 4.5% | 2.128 (2.512) | FAIL a,c |
| lcbench | no stopping | 0% | 0 | 0% | 9.3% | 2.3% | 1.1% | 0.8644 (0.8617) | baseline |
| lcbench | median @ 25%+50%, n≥5 (current default) | 7% | 46/98 / 46/98 | 8.8% | 14.9% (+5.5) | 3.3% | 1.6% | 0.8620 (0.8593) | FAIL a,b,c |
| lcbench | p25 @ 50%, n≥5 | 0% | 0/98 / 0/98 | 0.2% | 9.8% (+0.4) | 2.4% | 1.1% | 0.8645 (0.8617) | FAIL b |

How many of the 27 rules pass each criterion:

| Regime | (a) | (b) | (a)+(b) | All three |
|---|---|---|---|---|
| rl30 | 9 | 8 | **0** | 0 |
| rl15 | 3 | 9 | **0** | 0 |
| cifar | 2 | 8 | **0** | 0 |
| lm | 7 | 4 | **0** | 0 |
| lcbench | 16 | 0 | **0** | 0 |

## Why
1. **Early curve values do not predict the final result, so (a) and (b) cannot both hold.** Every rule that frees ≥ 20% of slot time stops a third or more of the eventual top-10% runs.

   Spearman correlation between the curve value at the checkpoint and the final score (`predictiveness` in `results.json`):

   | Regime | 25% | 50% | 75% |
   |---|---|---|---|
   | rl30 | 0.15 | 0.40 | 0.38 |
   | rl15 | −0.04 | 0.29 | 0.58 |
   | cifar | −0.11 | −0.04 | −0.02 |
   | lm | −0.71 | −0.59 | −0.19 |
   | lcbench | 0.46 (n=272) | 0.83 (n=83) | 0.99 (n=12) |

   On lm the correlation is *negative*: runs that overfit (e.g. batch 64) look best at 10–20 s and end worst. On lcbench the late checkpoints correlate well, but they almost never fire because most runs end before 75 s.
2. **Closed-loop ratchet.** The runs that get stopped never join the reference set, so the threshold keeps rising. In the sim, the median rule stops ~92% of rl15 runs with the 1 s planner (3,011 stopped, 192 completed), which is the R41 failure mode.
3. **With Opus's latency every race is planner-bound** (no-stopping waiting is 48% on rl30, 71% on rl15, 35% on cifar and 20% on lm). Freed slots only add idle time, so (c) fails for every rule that stops anything material. Even with a 1–2 s planner, waiting % rises with stopping, because busy periods get shorter while the per-launch planner gap stays the same. No rule improved the best score reached in the window either: it is equal or worse in every regime.

## Caveats
- **Sample sizes:** lm has 38 candidates and 4 top-10% runs; cifar has 12. A 5% bar there means zero wrong stops.
- **Sim draws:** the sim samples runs i.i.d. from the recorded pool, so it does not model the planner learning over the race. "Best score" mostly measures how many runs completed.
- **Slots and timing:** the replay uses the collection races' own slot counts and timing (8/3/2 slots). The sim uses 30 slots.
- **Cancel latency:** measured on Modal for rl only, and assumed to be the same for cifar and lm.
- **Not modelled:** cold-start variation and judge/diagnose calls, which block the planner.
