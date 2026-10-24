# Target calibration (goal values)

**Status: PROPOSED, not yet frozen.** Once a benchmark race uses these values they are frozen:
never move them afterwards (benchmark.md: "Set the thresholds from the starting code's
performance before the first race, and don't move them afterwards"). A change means a new
file version (`calibration_v2.json`) and new races, never an edit of this one.

Made by `scripts/calibrate_targets.py` on 24 Oct 2026, ~00:58–01:20 BST. It used the deployed
Modal app `busy-machines-runner` (workspace flagformgaze51). cifar and lm ran on L4, rl on
CPU (1 core). No LLM calls. Estimated GPU spend: **$0.22**.

| File | What |
|---|---|
| `calibration_v1.json` | every run (config, seed, Modal call id, status, reported and held-out score, duration, est. $), per-group mean/std, the formula proposal, the chosen values and their sanity checks |
| `choices_v1.json` | the chosen value per target and why it differs from the formula |
| `target_values.json` | `{"cifar": 0.94, "lm": 2.48, "rl": -165.0}` (held-out units) |

## What was measured (held-out score: the graded number, `metrics.quality`)

| Target | Starting code (default config) | Best replicated config (3 seeds) | Formula goal | **Proposed goal** |
|---|---|---|---|---|
| cifar (test_acc, higher is better) | 0.9277 ± 0.0018 (n=3); reported 0.9319 | 0.9371 ± 0.0028 (18 epochs) | 0.933 | **0.940** |
| lm (test_bpb, lower is better) | 2.5222 ± 0.0231 (n=3); reported val_bpb 2.1415 ± 0.0066 | none better (2.5219, 2.5404, 2.5812) | none | **2.48** (unvalidated) |
| rl (mean_return, higher is better) | −966.3 ± 166.9 (n=10) | −163.6 ± 6.0 and −164.4 ± 6.2 | −485 | **−165** |

Formula: `start_mean + 0.6 × (best_replicated_mean − start_mean)`, rounded toward the start.
"Best" is always a 3-seed replication of the best non-trap collection config, never one lucky
run.

## Why the proposed values differ from the formula
- **cifar 0.940:** the recorded Opus collection race passed 0.933 after 3.3 min, so 0.933 is not
  hard. No 3-seed mean reaches 0.940, but single non-trap collection runs did: at 7.7 min, and
  0.9434 by 11.6 min. So it is reachable in 30 min. It is also the airbench94 bar and the value
  already in the target spec, so the prompt bytes stay the same.
- **rl −165:** the gap is a cliff, not a slope. The recorded Opus collection race passed −485
  after 0.7 min. −165 sits on the best-config plateau: 4 of 6 replicated single runs reach it,
  and the Opus collection race first reached it at 2.9 min. A harder alternative is −160
  (2 of 6 runs; 7.1 min).
- **lm 2.48: NOT VALIDATED.** No config beats the starting code once it is replicated. The one
  collection run under 2.48 (2.4757) was a lucky seed of a config whose mean is 2.522. The
  held-out spread across seeds (0.023) is 3.5× the validation spread (0.0066), because the test
  shard is the end of the play. On this target a single-run goal can be hit by seed luck.

The collection-race times are a sanity check only. Those are dev races (`is_dev = 1`) with
probes mixed in, not benchmark results.

## Caveats
- The rl CPU score depends on `BUSY_MODAL_CPU_CORES`. These numbers assume 1 core (the
  runner's default), which was not checked against the deployed app.
- The 3 cifar runs of the newest collection best (width 1.75) failed with
  `unknown params: convs_per_group, cutout, lr_schedule, norm, optimizer`. The local
  `workers/targets/*.py` had gained new parameters after the deploy. All other runs used the
  deployed code. Re-check the starting-code numbers after the next deploy.

## rl at 15 s runs: goal key `rl@15s` (REGIME #2, D32)

A separate goal for the `rl-wide-15m` preset (15 s runs). `bench.py` looks it up as
`"rl@15s"` (`goal_key`: `target@Ns` whenever the per-run budget is not the target's
default), so the 30 s goal `rl` = −165 is untouched. Measured 24 Oct 2026 ~02:29 BST on the
redeployed `busy-machines-runner` (1 CPU core per container, verified with `nproc` in a live
container), all 19 runs launched at once on a pre-warmed pool (the race path). No LLM calls.
Spend ≈ **$0.004**.

| File | What |
|---|---|
| `calibration_rl15_v1.json` | every run (config, seed, Modal call id, held-out score, wall time, startup, compile), per-group stats, warm-pool record |
| `choices_rl15_v1.json` | the chosen goal, formula value, hit rates, alternatives and why |

| Group (15 s runs) | Held-out mean ± std (n) | Single runs reaching −170 |
|---|---|---|
| default (starting code) | −1028.6 ± 19.9 (10) | 0/10 |
| A: collection `f2835d4f` (hidden 32, reward_scale 0.003) | **−163.9 ± 6.4** (3) | 2/3 |
| B: collection `5aef031e` (hidden 64, 32 envs) | −204.1 ± 33.8 (3) | 0/3 |
| C: calibration_v1 best (hidden 128) | −190.2 ± 42.4 (3) | 1/3 |

**Proposed goal: −170** (formula −510 sits on the cliff, like the 30 s case). Warm run wall
time for a 15 s budget: p50 15.99 s, p90 16.11 s, max 16.36 s (n = 19), which sets the
preset's deadline rule `launch_cutoff_s = 17`.
