# Review patches: not applied yet

These are fixes for the confirmed review findings (workflow output `woxf36z18`), written as patches while the races run from the current code. **None of them is applied.** No paid call, no Modal and no commit was made. `docs/results_log.md` and `docs/decisions.md` were not touched.

Each patch is a unified diff (`a/` and `b/` paths from the repo root) against the files as they were at about 17:50 BST on 24 Oct. That includes what the other workflow changed in the meantime:
- the `lcbench-wide-bm-c30-15m` preset and its pre-registration (D49);
- the D50 fingerprint rule that leaves `diagnose_run` out;
- `analyze_races.py`, which is now LF-only.

The patches were first written against an older state and then rebuilt on this one. Each patch also adds its own test file `tests/test_review_*.py`, and changes an existing test where the fix changes an asserted structure. Line endings are kept per file: CRLF files stay CRLF, and mixed files such as `tests/test_analyze_races.py` stay mixed.

Findings 10 and 11 (`scripts/eval_served_calls.py`, `scripts/push_judge_traffic.py`) are **not** covered here.

## Order by risk (lowest first) and what each one changes

| # | Patch | Finding | Files | Changes the fairness fingerprint? | Changes what a running race does? |
|---|---|---|---|---|---|
| 01 | `01-sampler-fakes-shipped.patch` | metrics.py `sampler_fakes_shipped` counts fakes ever accepted as best | `backend/benchmark/metrics.py`, `scripts/headline.py` | no | no (reporting only) |
| 02 | `02-shipped-trap-quality.patch` | metrics.py "shipped model" quality uses a contaminated held-out score | `metrics.py`, `analysis.py`, `scheduler.py` (one kwarg in `live_summary`), `analyze_races.py`, `headline.py` | no | no (reporting only; the arm report's `shipped_quality` value) |
| 03 | `03-headline-prereg-pinning-and-repeated-seeds.patch` | headline.py pre-registration matched by name only; repeated seeds kept one race silently | `scripts/headline.py`, `tests/test_planner_cadence.py` (helper) | no | no (reporting only) |
| 04 | `04-analyze-races-setups-and-skips.patch` | analyze_races silent target skips; pooling across fingerprints | `scripts/analyze_races.py`, `tests/test_analyze_races.py` (one test) | no | no (reporting only) |
| 05 | `05-bench-preflight-all-project-keys.patch` | bench.py live pre-flight checks only `MUBIT_API_KEY` | `scripts/bench.py` | no | no (only refuses to *start* a race with a missing or duplicated project key) |
| 06 | `06-deadline-cutoff-counters.patch` | scheduler.py: proposals after the deadline cutoff stay pending and are not counted | `backend/scheduler.py` | no | counters and labels only. Launches, $ and quality are unchanged. Late proposals become `cancelled` with the deadline-rule error, are counted in `launches_skipped_deadline` / `backfill_dropped`, and their TPE trials are closed (FAIL). Applies only to presets with `launch_cutoff_s` (lcbench/cifar 17 s) |
| 07 | `07-planner-sees-human-corrections.patch` | loop.py: the planner's best config ignores human fake-improvement flags | `backend/scheduler.py` (`ExperimentSnapshot`), `backend/agent/loop.py` | no | only after a correction: the propose user message hides a flagged run's score, so the "current best config" and the lean-planner diff base move to the honest best. The system prompt and schema are unchanged, so the Mubit task does not split. Benchmark races refuse corrections; `demo-3m` records them |
| 08 | `08-mubit-fallback-to-provider.patch` | mubit_client.py: the fallback is the same request, so it can be routed to the adapter | `backend/agent/mubit_client.py`, `backend/scheduler.py` (`llm_policy_of`), `scripts/analyze_races.py` (fingerprint), docstrings in `loop.py` and `app.py`, `tests/test_bench.py` (the record-mode test: it asserted the old rule, "fallbacks included") | **yes, for arms that record** (`metadata`/`full`, i.e. the mubit arm): `llm_policy.fallback_record = "off"`. Arms with record `off` (opus, opus_bm) keep their fingerprint | yes, on the mubit arm: every fallback goes out with `x-middleware-record: off` and no adapter header |
| 09 | `09-proposer-process-isolation.patch` | GIL contention: BackfillSampler and the Optuna arm's study share the race process | `backend/agent/study_worker.py` (new), `backend/agent/optuna_arm.py`, `backend/scheduler.py`, `backend/app.py`, `scripts/analyze_races.py` (fingerprint), `scripts/bench_proposer_contention.py` (new), `tests/test_backfill.py` (2 structural asserts) | **yes, for every benchmark race with a backfill arm or an Optuna arm**: `proposer_isolation = "process"` on those arms | yes: timing. The TPE studies run in child processes, so the Optuna arm's GPU waiting and experiment counts change. The configs do not change: same seeds, TPE settings and order of operations |

**Do not apply 08 or 09 in the middle of a race series**, including the c30 series now pre-registered (09 changes its fingerprint). They change the fingerprint, so the races run after them never pool with the ones before (by design). A series split across them can no longer reach n=3 for one setup. 06 and 07 leave the fingerprint alone. 06 only changes counters (`launches_skipped_deadline`, `backfill_dropped`, the sampler's `told_failed`/`open_trials`), so apply it between series too, or say in the log that the counters changed. 07 matters only in races with corrections. 01 to 05 change what the reports print, not what races do.

### Measured effect of 02 and 03 on the D47 races (read-only, three JSONL exports, `6978df02c4`)

- **Shipped model, held-out score, Optuna:** 0.8555 (n=3) before, 0.8509 (n=2/3) after, with "1 fake shipped (held-out not counted)". That is seed 2, race `c72b3af1`, a `best_test` trap. Opus and BM are unchanged (no fake shipped).
- **Pre-registration:** all three races match every frozen setting and have no repeated seeds. The verdict is the same before and after: `SUCCESS LINE FAILED`. The check statuses before the patch were `opus_waiting` FAIL, `bm_waiting` FAIL, `bm_llm_usd` PASS, `bm_best_vs_opus` FAIL, `bm_best_vs_optuna` PASS.

Recompute the D47 "shipped held-out" row with 02 applied before it is logged. Until then, label the pre-patch Optuna value as contaminated.

## What each patch does (short)

- **01:** `sampler_fakes_shipped` becomes 0/1, meaning the *final* shipped run is a sampler trap. The old count is kept as `sampler_fakes_accepted_as_best`. The headline engine line prints both.
- **02:** adds `metrics.TRAP_CONTAMINATES_HELDOUT = {"lcbench"}`. On LCBench the trap selects the epoch on the test split, and that test split is `heldout_score`. A shipped trap run on such a target gets `shipped_quality = None`, `shipped_is_fake = True` and `shipped_heldout_raw` (audit only). On rl, cifar and lm the trap does not touch the held-out split, so the honest (worse) number is kept, as it should be. `summarize_arm(..., target=)` is passed by the scheduler, `analysis.summaries_from_rows`, `analyze_races` and `headline`. The shipped cells say "N fake shipped".
- **03:** `PreRegistration.frozen` pins the settings frozen in `docs/preregistration_lcbench_wide.md`. The pins also cover `lcbench-wide-bm-c30-15m` (`docs/preregistration_lcbench_wide_c30.md`): the same setup with the cadence at 30 s, and a test checks that the two are never mistaken for each other. The pinned settings are: arm set, budget, slots, cap, goal, run budget, pipelining, cutoff, record modes, engine features, cadence 90 s, backfill settings, prompt fingerprints (v1/v2 propose, v1 judge), BM effort low, and project A. A setup that differs gets "NOT the pre-registered setup" in place of the banner, every check is "(not evaluable)", and the verdict is never MET or FAILED. A repeated seed is now a not-evaluable reason (it names the race ids, and per-seed cells list every value) and a setup blocker in `build_groups`.
- **04:** `analyse()` groups by (target, fingerprint id). It writes one report per setup with `setup `<id>`` in the heading and a warning when a target has several setups. A target it cannot analyse (mixed goals, no goal) goes into `Exclusions` ("Excluded from this analysis", JSON `excluded`) and to stderr. The heading counts distinct seeds ("N race(s) over M distinct seed(s)"). Paste rows and charts are stamped `MIXED SETUPS / REPEATED SEEDS` when the pooled races are not one setup over independent seeds. Chart files get a `-<setup>` suffix only when a target has several setups.
- **05:** `mubit_key_checks(cfg)` loads `data/mubit_projects.json`. It checks the key of every project that any LLM arm of the race routes a step to, plus the default project, and fails when two projects share a key. That is the rule `MubitClient._client_for` already enforces lazily. Keys are never printed.
- **06:** `_apply_cutoff` drops and counts buffered proposals on **every** call. The one-time work (releasing idle slots, the log line) runs once. A dropped sampler proposal is told to the sampler as no score. The arm shutdown then applies the sampler's queued tells, last and off the event loop.
- **07:** `ExperimentSnapshot.flagged_fake` / `override_genuine`. `RepoPrompts._row` hides a flagged run's score ("human: fake improvement, not counted"), and a human "genuine" lifts the LLM-judge `invalid`. Row ranking now matches `_eligible`, and `propose_base_config`/`expand_diff` follow. `DefaultPrompts` marks it in its brief.
- **08:** `MubitClient.call(is_fallback=True)` sends `record_headers(self.fallback_record)`, which is `off`, and skips `projects.headers_for`. The fallback stays on the routed project's key. `llm_policy_of` reports `fallback_record`. **This reverses a rule that was there on purpose.** Before, every attempt, fallbacks included, carried the arm's record mode, and `test_researcher_record_mode_reaches_the_wire_on_every_attempt` asserted it. The patch changes that test to expect `[mode, mode, "off"]`. A `metadata` arm's fallback is therefore no longer collected as a Mubit example. Decide this with that in mind.
- **09:** `StudyWorker` runs one TPE study per child process (spawn, daemon), with the same `TPESampler(seed, n_startup_trials, constant_liar=True)` and direction. `BackfillSampler(isolation=)` and `OptunaResearcher(isolation=)` send their queued operations with each ask, in the same order the in-thread code applies them. A test checks that thread and process modes give identical configs and counts. `study_worker.proposer_isolation(mode)` returns `process` for benchmark races and `thread` for dev, mock and dry-run races. `BUSY_PROPOSER_ISOLATION=thread|process` overrides it, and the value is recorded. The children start before t0 (`RaceEngine.run`) and close in `_finish`. New per-arm report fields:
  - `proposer` (Optuna arm): `isolation`, `asks`, `ask_s`, `ask_p50_s`, `ask_p95_s`, `ask_max_s`
  - `backfill`: adds `isolation` and the same ask percentiles
  - `event_loop_lag` (every arm): `samples`, `lag_p50_s`, `lag_p95_s`, `lag_max_s`, measured as how late each monitor tick wakes

  The engine manifest records `proposer_isolation` per arm.

## Not done (left on purpose, say so if it matters)

- 07: rows with an `unsure` verdict are still shown when `accept_unsure=False`. Every preset uses `True`.
- 08: `LLMCallRecord` gets no "record mode sent" column, which would need a DB migration. A fallback is always `off` by construction (`recorded=False`, `is_fallback=True`, manifest `fallback_record`). A judge with no usable reply on an adapter-served arm is still turned into `unsure` and accepted under `accept_unsure`. Whether `off` guarantees a provider answer is still open with Mubit (office hours item 2).
- 03: the c30 pre-registration's own check `bm_fakes_shipped` counts `fake_best_accepted`, i.e. trap runs accepted as a new best at any time. That is the same "accepted vs shipped" distinction as finding 8. Its label says so ("trap runs accepted as a new best"), and it is a frozen formula, so it is left unchanged.
- 03: bench.py does not tag overridden race names. The pinned settings make that unnecessary for the pre-registration.
- 09: in process mode, an Optuna `add_trial` error from `observe` surfaces at the next propose instead of at once. The arm then stops with "researcher crashed" either way. Offline replays should still not run on the race host.

## How to apply (after the current race series)

From the repo root, in order, with the races stopped:

```
git apply --check --whitespace=nowarn data/review_patches/01-sampler-fakes-shipped.patch   # repeat per patch
git apply --whitespace=nowarn data/review_patches/01-sampler-fakes-shipped.patch
```

`--whitespace=nowarn` only silences git's "trailing whitespace" note about the CR of CRLF lines. Every patch applies alone to the current files, and all nine apply stacked in the order 01→09. Several touch the same files (`scheduler.py`, `headline.py`, `analyze_races.py`, `metrics.py`, `app.py`), but never the same lines. Checked with `git apply --check` against the real working tree (`core.autocrlf=true`): all nine pass.

## How it was verified

- Each patch was applied alone to a fresh copy of the current files, and its new tests plus the existing test files it touches were run there. The results are in the table below.
- All nine were applied stacked to one copy, and the full suite was run there.
- Every new test file was also run against the unpatched code. Each file fails there. A few single tests pass on the old code too, because they guard behaviour that must not change, such as unflagged rows or "no cutoff configured".
  - Collection errors, because a new symbol is missing: 02, 03, 09.
  - Test failures: 01 (3 of 4), 04 (5/5), 05 (6/6), 06 (2/3), 07 (4/5), 08 (5/5).
- Two environment notes for scratch copies:
  - A scratch copy has no `.env`, so set `MUBIT_API_KEY_B=test-key-b` (a fake string). Without it 7 existing `tests/test_mubit_client.py` tests fail on the unpatched code too. The patches do not cause this.
  - `tests/test_fairness_fixes.py::test_early_stop_keeps_the_slot_busy_until_the_runner_confirms` is timing-sensitive and failed once on the unpatched code while races loaded the machine.

```
MUBIT_API_KEY_B=test-key-b PYTHONUTF8=1 python -m uv run python -m pytest -q tests/test_review_*.py
MUBIT_API_KEY_B=test-key-b PYTHONUTF8=1 python -m uv run python -m pytest -q
```

Results (24 Oct, ~18:10 to 18:40 BST, against the files as of ~17:50, on the race host while races ran):

| Patch | Applies alone | Tests run alone (its new file + the existing files it touches) |
|---|---|---|
| 01 | yes | 45 passed |
| 02 | yes | 126 passed |
| 03 | yes | 64 passed (incl. the c30 tests and `tests/test_bench.py`) |
| 04 | yes | 58 passed |
| 05 | yes | 41 passed |
| 06 | yes | 105 passed |
| 07 | yes | 86 passed, 4 skipped |
| 08 | yes | 186 passed, 1 skipped (incl. `tests/test_bench.py`) |
| 09 | yes | 129 passed, 2 skipped |
| **all 9 stacked, 01→09** | yes | **full suite: 743 passed, 11 skipped, 0 failed** |

Two failures came up during verification. Both are dealt with:

- An earlier run against the older base failed 2 tests: `tests/test_bench.py::test_researcher_record_mode_reaches_the_wire_on_every_attempt`. That test encoded the old fallback rule, and patch 08 now updates it.
- In an earlier run, patch 09's test set once failed 1 test while the machine was loaded. It could not be identified because the output was cut off, and the 4 reruns since were clean. If it shows up again, the timing-sensitive engine tests are the first suspects.

`git apply --check` against the real working tree (`core.autocrlf=true`) passes for all nine.

### The GIL micro-benchmark (patch 09)

`scripts/bench_proposer_contention.py` times the Optuna arm's ask (`--k-optuna 8`) alone and with a backfill sampler asking (`--k-sampler 30`) in another thread. It does this in-thread and in process mode, with both studies preloaded to `--trials` completed trials (default 600, rl space), and records the event-loop lag of an asyncio ticker. It is CPU-heavy, so run it after the races, not on the race host during a race:

```
python -m uv run python scripts/bench_proposer_contention.py                  # rl, 600 trials
python -m uv run python scripts/bench_proposer_contention.py --target lcbench --trials 1200 --json out.json
```

Smoke run only (`--quick`: 60 trials, 2 repeats, 24 Oct ~17:00, on the race host while races ran, so not a benchmark number and not for the log):

| Isolation | Sampler asking next to it | Optuna ask p50 |
|---|---|---|
| thread | no | 0.071 s |
| thread | yes | 0.173 s (2.44x) |
| process | no | 0.076 s |
| process | yes | 0.086 s (1.14x) |

Run the full version after the series and log it in `docs/results_log.md`, whichever way it comes out.
