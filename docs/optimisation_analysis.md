# Optimisation analysis: where the paid GPU time and the $ go

**Status:** analysis of existing data only. No new runs, no Modal or LLM calls. Source: the R20 Opus collection races (`data/collection/collect_{rl,cifar,lm}.db`, opened read-only) and `data/calibration/calibration_v1.json`. These are **dev races** (`is_dev = 1`): Opus only, recording on, probes mixed in. They are not benchmark results. Projections (section 2) and what-ifs (sections 1 and 4) are model outputs, not measurements, so slides can't quote them as results. These numbers still have to be logged in `results_log.md` (not done here: this task did not edit existing files).

Races: rl has 3 races × 8 CPU slots × 12–15 min, 30 s run budget. cifar has 2 × 3 L4 × 20 min, 45 s budget. lm has 2 × 2 L4 × 17.5 min, 60 s budget. Pipelining is off, as it is in every benchmark preset.

---

## 1. GPU-second overhead per run

**How the time was split.** A slot's paid time is busy time plus idle time (`gpu_intervals`). For each succeeded run, busy time = `ended_at − started_at`. It was split as follows:

| Part | Where it comes from |
|---|---|
| Launch + return | `modal_startup_s` (host launch → container code starts) + the time from the end of the run to the host receiving the result. The two are summed because the container clock runs about 0.4 s ahead of the host clock, which makes the return part slightly negative on its own. **We can't split queue time from container boot**: we only record launch and container-start timestamps |
| Cold-start excess | Launch + return above the warm-run median (0.55–0.8 s) |
| Setup in the container | imports, Volume reload, data to device, model build: first streamed metric − startup − training time up to that metric |
| Warmup / compile | rl: `compile_s` (JIT, counted inside the 30 s budget). cifar and lm: `train_time − steps / steady-state rate`, where the steady rate is the median steps/s between later streamed metrics. For cifar this mostly captures cudnn.benchmark autotuning |
| Steady training | training time − warmup |
| Eval | rl `eval_s`, cifar `eval_time_s` (per-epoch val + final TTA), lm `wall_time_s − train_time_s` |

### Share of paid slot-seconds (all races together)

| Part | rl (20,015 s) | cifar (7,175 s) | lm (4,186 s) |
|---|---|---|---|
| Idle: waiting for the LLM (section 2) | 16.8% | 9.7% | 7.6% |
| Busy, but the run was cancelled at the race deadline | 1.7% | 1.6% | 3.2% |
| Launch + return | 1.8% | 8.9% | 10.1% |
| … of which cold start at race start | 0.15% | **7.6%** | **8.7%** |
| Setup in the container | 0.2% | 2.4% | 0.7% |
| Warmup / JIT compile | 3.3% | 4.5% ¹ | 0.4% |
| Eval | 0.8% | 4.6% ¹ | 0.3% |
| **Steady training** | **75.4%** | **68.2%** | **77.8%** |
| **Not training** | **24.6%** | **31.8%** | **22.2%** |

¹ cifar trap runs inflate these numbers: their fine-tune on the test split counts as `train_time` but not as `steps`, and they run extra evals. On clean runs only, warmup is 1.7 s per run (p90 6.1 s, max ~17 s) and eval is 1.75 s per run.

### Per run (warm container, mean)

| | rl | cifar | lm |
|---|---|---|---|
| Wall time (busy) | 31.0 s | 38.2 s | 60.3 s |
| Launch + return | 0.6 s | 0.6 s | 1.0 s |
| Setup | 0.06 s | 0.96 s (data to GPU 0.43 s) | 0.18 s |
| Warmup / compile | 1.25 s | 1.9 s | 0.34 s |
| Eval | 0.30 s | 2.15 s | 0.20 s |
| Cold runs (startup > 10 s) | 1 of 526 | 7 of 153 | 5 of 56 |
| Startup of a cold run | 31 s | 79 s mean (up to 160 s) | 74 s mean (up to 159 s) |

Every cold start came at the start of a race. After that, every run landed on a warm container: the 60 s `scaledown_window` is already long enough, and no idle gap was longer than that. The R12/R16 smoke tests showed a 9–12 s container start when the image was cached, so most of the ~75 s at race start is probably queueing or image distribution for many containers at once. That is an inference; our timestamps can't confirm it.

### What each change would save

Savings are given as the share of paid slot-time that would become training time. In our accounting the paid time is fixed (the slots × the race clock), so a saving shows up as more experiments, not a smaller bill. On Modal's real bill, a pre-warm moves cost outside the race window rather than removing it.

| Change | rl | cifar | lm | Code to change |
|---|---|---|---|---|
| (a) Pre-warm the containers before t0 (`ModalRunner.prewarm`, `min_containers` = slots, for both arms) | ~0.2% | **~7.6%** | **~8.7%** | **Runner / race setup** only. Warm reuse between runs already works (scaledown 60 s). Removing the remaining ~0.6–1 s per run of launch round-trip (1–2%) would need a long-lived worker that pulls jobs from a queue: runner code |
| (b) Keep data resident in the container between runs (a module-level cache of tensors already on the device) | 0.2% | ≤2.4% | ≤0.7% | **Target code** (`cifar.py`, `lm.py`). The cold-start setup (4–5 s) disappears with (a) |
| (c) Keep cudnn/JIT caches warm | ≤3.3% (JIT is 4% of each 30 s budget; a persistent JAX cache on the Volume only helps for repeated shapes) | ≤3% (pre-autotune the common width/batch shapes at container start, or set `cudnn.benchmark = False`) | ~0.4% | **Target code** (plus an env/cache dir in the image). In both rl and cifar this time counts against the run budget, so it is a **fairness constant**: freeze it before the races |
| (d) Shorter eval (fewer per-epoch val passes, one TTA pass) | 0.8% max | ~2% (about half of 4.6%) | 0.3% max | **Target code**. It changes the score, so it would be a new target version |
| (e) Don't launch runs that can't finish before the deadline | 1.7% | 1.6% | 3.2% | **Scheduler** (the same rule for every arm) |

**Bottom line:** on cifar and lm, the cheapest win is pre-warming (about 8% of paid time, runner-only). On rl, overhead is small (about 6%) and the LLM wait (16.8%) dominates.

---

## 2. Idle accounting and the projection to 32 slots

### Measured, per race

An idle slot-second counts as **propose wait** when it overlaps an in-flight `propose_experiments` call, as **judge wait** when it overlaps only a `judge_improvement` call (the researcher is sequential, so a judge call blocks the next proposal), and as **other** otherwise.

| Race | Slots | Paid slot-min | Idle | Propose wait | Judge wait | Other | Of which the race-start wait for the first call |
|---|---|---|---|---|---|---|---|
| rl seed 0 | 8 | 120.0 | 17.4% | 13.8% | 3.5% | 0.0% | 1.0% |
| rl seed 1 | 8 | 120.0 | 17.1% | 14.1% | 2.9% | 0.1% | 0.9% |
| rl seed 2 ($ cap) | 8 | 93.6 | 15.8% | 15.0% | 0.8% | 0.1% | 1.4% |
| cifar seed 0 | 3 | 60.0 | 9.4% | 7.9% | 1.5% | 0.1% | 0.4% |
| cifar seed 1 | 3 | 59.6 | 10.1% | 7.8% | 2.0% | 0.3% | 0.4% |
| lm seed 0 | 2 | 35.0 | 8.0% | 7.5% | 0.4% | 0.0% | 0.1% |
| lm seed 1 | 2 | 34.8 | 7.2% | 6.1% | 1.1% | 0.0% | 0.1% |

**98–100% of idle time is time spent waiting for the LLM** (the race-start column is included in it, since that wait is for the first propose call). Judge wait is inflated by probes: traps reliably claim a new best. Without probes the judge rate is 0.07 per run on rl (31 of 448), against 0.16 with them.

Opus `propose_experiments` latency in the loop:

| | rl | cifar | lm |
|---|---|---|---|
| All calls, p50 / p90 | 3.13 / 5.78 s | 3.80 / 5.40 s | 4.15 / 6.69 s |
| K = 1 calls, p50 / p90 (rl) | 3.02 / 4.18 s | | |
| Median by K (rl, K = 1…7) | 3.0, 4.5, 5.5, 6.4, 7.4, 8.9, 9.6 s, i.e. **+1.1 s per extra proposal** (~170 output tokens each) | | |
| Judge p50 / p90 | 2.29 / 2.91 s | 2.54 / 3.10 s | 2.17 / 2.46 s |

### Projection model
This is an event simulation of `scheduler.py` as it runs now:
- one sequential researcher, no pipelining, K = free slots;
- a judge call blocks the loop on a fraction of finished runs;
- each call returns with a latency L(K);
- runs last 15 s ± 15% (uniform), over a 30 min race, averaged over 12 seeds.

**Validation:** with rl's measured run lengths, 8 slots, Opus's K-scaled latency and the measured judge rate, the model gives 16.5–19.0% idle. The measured figure is 16.8%.

Two latency models:
- **flat**: L is the same for any K. Opus at its p50 / p90 over all calls (3.13 / 5.78 s); the adapter at 1 s or 2 s.
- **K-scaled**: Opus at its K = 1 p50 / p90, plus 1.1 s per extra proposal (fitted on K ≤ 7 and extrapolated beyond). The adapter at 1 s + 0.18 s/K or 2 s + 0.37 s/K, an assumed 6× and 3× faster decode. **Adapter latencies are assumptions, not measurements.**

**GPUs waiting at 32 slots, 15 s runs** (judge on 7% of runs, the clean-claim rate; without judges in brackets):

| Researcher | Flat latency | K-scaled latency |
|---|---|---|
| Opus at p50 | **32.9%** (24.6%) | **67.2%** (62.3%) |
| Opus at p90 | 47.5% (35.2%) | 72.6% (65.0%) |
| Adapter at 2 s | 22.2% (16.7%) | 40.0% (28.4%) |
| Adapter at 1 s | **10.3%** (8.9%) | **15.2%** (12.3%) |

For comparison, with 8 slots and 15 s runs: Opus 23.9% flat at p50 and 28.8% K-scaled; adapter at 1 s 8.0–8.4%. With 32 slots and 30 s runs: Opus 15.1% flat and 39.5% K-scaled; adapter at 1 s 4.8–6.2%.

**Why Opus collapses in the K-scaled model.** A sequential researcher can produce at most 1/1.1 s ≈ 0.9 proposals/s, however it batches. 32 slots with 15 s runs need 2.1 proposals/s. So Opus can keep at most about 43% of the slots busy, and **at least ~57% of slot time is idle**. This depends on the +1.1 s/K slope continuing past K = 7, which we have not measured, and on the loop not making concurrent calls. If the loop allowed concurrent calls (the same for both arms), most of this effect would disappear. Pipelining doesn't help Opus here (60–65% idle); it brings the 1 s adapter to about 0–1%.

---

## 3. The 46 claimed new bests that Opus judged 'unsure'

For each claim, the held-out gain = the run's `heldout_score` − the previous best's `heldout_score` (sign flipped for lm). The previous best is the last `is_best` run before it in the same race; this matched `summary.best_before` in 46 of 46 cases. Seed spread σ comes from calibration:
- cifar 0.0025 (0.0018–0.0028);
- lm 0.017 (0.012–0.023);
- rl interpolated by the previous best's score, from 167 at the start (−966) to 6.2 on the plateau (−164).

**Real** = gain > 2σ, **plausible** = 1–2σ, **noise** = ≤ 1σ or the held-out score got worse. Note that the rl `reported_score` equals `heldout_score`: the eval on held-out levels is what the agent sees.

| Target | Claims | Clearly real | Plausible | Noise | Held-out score got worse | Range across the σ bounds (real / plaus. / noise) |
|---|---|---|---|---|---|---|
| rl | 31 | **6** | 1 | 24 | 0 | 7/0/24 … 3/0/28 |
| cifar | 12 | **3** | 2 | 7 | **5** | 4/2/6 … 3/2/7 |
| lm | 3 | **1** | 1 | 1 | 1 | 2/0/1 … 0/2/1 |
| **Total** | **46** | **10** | **4** | **32** | 6 | |

- The clearly real ones include rl's first jumps: −1291 → −195 (+1096, 6.6σ), −1047 → −221 (5.0σ), −1028 → −352 (4.0σ), −352 → −185 (3.8σ). On cifar they include 0.9246 → 0.9356 (4.4σ), 0.9342 → 0.9410 (2.7σ) and 0.9306 → 0.9370 (2.6σ). Opus said 'unsure' on every one of them.
- The lm 'real' case (2.5294 → 2.4950, 2.0σ) is borderline. R23 found that no LM config beats the starting code once it is replicated.
- On the rl plateau, 22 of the 31 claims are gains of 0.02–6 return points against σ ≈ 6: pure noise.
- In 5 of 12 cifar claims, the reported test accuracy went up while held-out accuracy went down.
- Opus gives the same reason every time: the evidence lacks seed-to-seed spread, split hashes and overlap counts, so noise and leakage can't be ruled out. Its 'unsure' comes from the evidence format, not only from the model.

**Verdict on the claim "Opus never confirms real gains":** it is true, but only as a narrow statement, and as a headline it misleads.
- **Fair version:** "Opus confirmed 0 of the 10 clearly real gains (rl jumps of up to 6.6σ) in race-format evidence."
- **Not fair:** "Opus rejects real wins". 32 of the 46 (70%) were noise, so 'unsure' was the correct answer for most of them. A judge that answers 'valid' every time would be wrong 32 times.

A corrected judge (or Sapling) wins only if it confirms the roughly 10–14 real or plausible gains and still says unsure or invalid on the 32 noise cases. Adding the seed spread to the judge's evidence would be a prompt change, so both arms would need to get it.

---

## 4. Same money vs same quality: what cheaper LLM calls would buy

**Model:** cumulative $ = slots × price × elapsed time (GPU) + the `cost_usd` of LLM calls completed so far. "Reaching the goal" = the first counted run (succeeded, not a trap, not judged invalid) whose held-out score reaches `target_values.json` (cifar 0.940, lm 2.48, rl −165). The decision timeline is held fixed and only the LLM $ is scaled, so time-to-target is unchanged. Faster adapter calls would also cut idle time (section 2); that effect is **not** included here. Probe runs are included in the costs; they never set the first hit.

### Cost to reach the goal

| Race | Time | $ to reach the goal (GPU + LLM) | LLM share | LLM ÷5 | LLM ÷10 |
|---|---|---|---|---|---|
| rl seed 0 | 2.9 min | $0.291 (0.018 + 0.273) | 94% | $0.073 (−75%) | $0.046 (−84%) |
| rl seed 1 | 2.7 min | $0.209 (0.017 + 0.191) | 92% | $0.056 (−73%) | $0.036 (−83%) |
| rl seed 2 | 4.9 min | $0.626 (0.031 + 0.595) | 95% | $0.150 (−76%) | $0.090 (−86%) |
| cifar seed 0 | 7.7 min | $0.534 (0.307 + 0.227) | 42% | $0.353 (−34%) | $0.330 (−38%) |
| cifar seed 1 | 1.5 min | $0.146 (0.062 + 0.084) | 58% | $0.079 (−46%) | $0.070 (−52%) |
| lm seed 0 | 6.0 min | $0.213 (0.161 + 0.053) | 25% | $0.171 (−20%) | $0.166 (−22%) |
| lm seed 1 | not reached | | | | |

The lm hit (2.4757) is a lucky seed (R23), so it is not a real "reached".

### Experiments at a fixed $ budget

| Target | GPU $ | LLM $ | LLM share | $ per run | Runs per $ with LLM ÷5 | with LLM ÷10 | with LLM free |
|---|---|---|---|---|---|---|---|
| rl (CPU slots) | 0.26 | 5.75 | 96% | 0.0114 | **×4.3** | **×7.2** | ×23 |
| cifar | 1.59 | 1.53 | 49% | 0.0204 | ×1.65 | ×1.79 | ×1.96 |
| lm | 0.93 | 0.76 | 45% | 0.0302 | ×1.56 | ×1.68 | ×1.82 |
| All | 2.78 | 8.04 | 74% | 0.0147 | ×2.46 | ×3.02 | ×3.9 |

Formula: (G + L) ÷ (G + L/f). The extra experiments only happen when the **$ cap is the binding limit**. With a fixed slot count and race clock, the same number of slots runs the same number of experiments, so a cheaper LLM only lowers the $ and adds no experiments. In the collection this was binding once: rl seed 2 hit its $6 cap at 11.7 of 15 min. At LLM ÷5 it would have run the full 15 min, about 28% more experiments in that race.

Caveats:
- The adapter's real $ per call is unknown, so ÷5 and ÷10 are scenarios.
- Recording was on and probes were mixed in.
- The rl "GPU" is a 1-core CPU at $0.0000131/s, which is why LLM $ dominates there.
- Holding decisions equal assumes adapter proposals are as good as Opus's. That is the open question the benchmark has to answer.

---

## Method notes
- The analysis scripts ran from the session scratchpad and are not in the repo. Everything was computed from the DB columns named above, and every query opened the DBs read-only (`mode=ro`).
- Clock skew: the container clock is about 0.4 s ahead of the host clock. Values that mix the two clocks are reported only as sums that cancel the skew (launch + return).
- Warmup estimates for cifar trap runs are inflated by the test-split fine-tune (see ¹). The clean-run figures are in the text.
