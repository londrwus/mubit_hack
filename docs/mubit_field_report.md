# Field report for the Mubit team

From the Busy Machines team (Night Shift London, 23–25 Oct 2026). We built an ML research agent whose recurring decisions (plan experiments, diagnose failed runs, judge claimed improvements, and a tested early-stop step) all run through the Mubit gateway, and we took four real tasks from recorded Opus 5.5 calls towards trained adapters. This report lists what we ran into and what we'd suggest. We wrote it to be useful to you, and we know the platform is a day old.

**Evidence rules.** Every measured number comes from our results log (`docs/results_log.md`, entries R1–R39) and carries its R-id. Other sources are our probe answers and decisions (`docs/decisions.md`), our docs review (`docs/mubit_office_hours.md`) and our code reading of the Banyan Sapling release (`docs/sapling_gate.md`); where a point is a reading rather than a measurement, we say so. Anything we haven't measured is marked "not measured yet". Our test sets are small (10–14 exam cases per adapter, 7 real gains in one set), so treat this as early signal, not a verdict.

**What changed since the first version (R1–R19).** We now have adapters on two real tasks (R28, R30), the exam rules (R30), the daily review limit (R29), a closed-output judge that passes our self-agreement gate (R32, R39), an early-stop step that didn't (R31), a 503 episode (R39), and measured Sapling results, including our head fine-tune (R33). F8–F13 are new; F1–F7 are updated.

## Summary
| # | Finding | Evidence | Impact for AI-lab users | Suggested fix (short) |
|---|---|---|---|---|
| F8 | A per-task daily review limit decides what the first adapter learns from | R28, R29 | Agents make hundreds of calls an hour; the first adapter trains on a few dozen and uses a daily adapter start | Document the limit, show the quota, sample the queue for coverage, let users hold training |
| F9 | The exam rules (95% match, 0.50 floor, ≥20 test cases) are visible only after training | R28, R30 | A training can't qualify however well it does, and still costs one of 3 daily starts | Show the rules before training; warn when the test set is under 20 |
| F10 | Free text in a structured output makes a whole-answer match exam very hard to pass | R30, R31, R32, R39, D39 | Most lab outputs carry a reason or a note; the match exam punishes wording, not decisions | Field-level grading; show the provider's self-agreement as the ceiling |
| F11 | Generative steps (propose) can't pass a match exam | R28, R20, R21, R36 | The slowest, most frequent step, where GPUs wait, is the one that can't be served | Outcome or validity grading for generative tasks |
| F12 | HTTP 503 `engine_unavailable` from the gateway during a recording run | R39 | A lab loop stalls or loses inputs; no hint when to retry | `Retry-After`, a clear "Mubit vs provider" error, an optional pass-through to the provider |
| F13 | Hosted Sapling endpoint: no way to check it matches the release | D33 (not measured yet) | Labs can't tell whether published Sapling numbers apply to the endpoint | Revision and calibration hash in the response; reference logits |
| F14 | Diagnose task: "Can't train: the training run failed", no reason given | R46 | 86 recorded calls give no adapter and no next step | A failure reason (data, schema, quota) and a retry button |
| F15 | A closed-output task (3 verdicts × reason codes, provider self-agreement 40/40) trained to an adapter that matches only 11.4% of the provider's answers in one workspace and 58.3% in another from the same inputs; training took ~6.5–7 h | R39, R70, R79 | The output format we designed to make matching easy didn't help; we can't see why from the card | Show the adapter's answers next to the provider's on the test cases, and a format-validity rate |
| F1 | Adapter replies don't keep the JSON output format | R7, R8, R9 | Agents that parse replies break; the broken replies are also the fast ones | Constrained decoding for `output_config.format`, plus a "malformed outputs: needs 0" exam line |
| F2 | The API doesn't say per call who answered | probe answers, R29 | Labs can't audit which model made a decision or measure adapters in their own tooling | A `served-by` response header and a lookup by request ID |
| F3 | `record: off` also drops the task fingerprint | docs, `backend/benchmark/configs.py` | Privacy, routing and "force the provider" are tied to one header | Separate "store content" from "who answers" |
| F4 | One step across many models: the task fingerprint pushes users to a weaker schema | R17, R18, R28, D24 | Labs choose between schema enforcement and one shared task | A task key or schema-family grouping; document the fingerprint rules |
| F5 | Adapters are graded against the provider, not ground truth | R7, R9, R24, R25, R34, R37 | An adapter graded against Opus learns Opus's blind spots, e.g. answering `unsure` on real gains | Customer-supplied ground-truth sets, scored next to provider agreement |
| F6 | Docs stop before live serving; outcome links are 404 | office hours, R28, R30 | Users can train an adapter but can't find how to route, correct or score it | One "adapter to live traffic" page and one "feedback" page |
| F7 | Stock Sapling answers `unsure` on every AI-research claim we tried | R33, `sapling_gate.md` §2 | As a verifier for research claims it defers everything to a human | Per-domain calibration and a refit script; our head fine-tune shows the gain and its limits |

## What worked well
Measured, and worth saying first:
- **Gateway overhead under 1 ms** per call (provider 1.46 s; R3). **Prompt caching passes through** (10,353 tokens written, then read; R2). Structured outputs pass through for provider calls, and 6 parallel calls returned no `429` (probe answers).
- **Recording at scale works.** One collection recorded 489 valid `propose_experiments` and 138 valid `judge_improvement` calls (R20); the closed-output judge recorded **177 of 177** valid answers (R39).
- **One task across three targets held** (D24): each real task is one task, 100% `claude-opus-5-5` (R28, R29). Kept rates so far: propose 66% (29/44), judge 80% (56/70) (R29).
- **Training helps.** The judge v0 adapter matched 6/14 test cases vs 1/14 for the base model with the prompt (R30). With 14 cases this is an early signal, not a firm result: the evaluation page shows a 95% confidence interval of 21.4–67.4% for the adapter, which is exactly the right thing to show at this size.
- **Fast pipeline on the toy task:** 40 calls to a trained adapter in about 20–25 minutes (R6).
- **The Sapling release reproduces.** The package's reference example matched the published H200 logits exactly (max diff 0.0; R33).
- Every model we needed was available, and a hosted Sapling endpoint was shared with us quickly (D33).

## New findings (R20–R39)

### F8. A per-task daily review limit decides what the first adapter learns from
**What we saw.** On `propose_experiments` (`ln_42bac960…`): 499 calls, **29 kept, 15 rejected, 455 waiting**; review today: 200 queued, **299 over today's limit** (R29). Training started on those 29 kept examples and the result scored **14.3% vs a 95% target** (R28). From the queue numbers the limit looks like about 200 per task per UTC day, resetting at 00:00 UTC; that's our inference, not confirmed. The limit isn't in the docs. We adapted: at most 200 recorded calls per task per day, coverage over volume, and a second project for fresh queues (D38, `docs/demo.md` §0).

**Why it matters to AI labs.** A lab agent makes many calls: our RL collection alone made 337 propose calls, 333 of them valid (R20). If the first adapter trains on the first few dozen reviewed calls, those may all come from one target or one phase of a run, so the adapter is weak and still uses one of the 3 daily adapter starts. Sending more calls the same day adds nothing, and nothing tells the user so.

**Suggested fix.**
1. Document the limit and show the remaining quota on the task page (a response header would be even better).
2. Say how the reviewed calls are chosen. If it's first come, first served, sample across sessions and time instead, so the kept set covers the task.
3. Let users hold training until a set number of kept examples, or until today's queue is reviewed.
4. Let users mark calls as "for teaching", so the limited reviews go where they matter.

### F9. The exam rules are visible only after training
**What we saw.**
- Targets: **95%** for our new tasks, **85%** for the toy task (R7, R28, R30). We found no explanation of the difference.
- The evaluation page says "evidence for a move" needs the verdict passed, **conformance pass rate ≥ 0.50**, and **≥ 20 judged test cases** (R30). Both real-task adapters had **14** (R28, R30), so neither could qualify at any score. The test set grows with kept examples, which the review limit (F8) caps.
- Adapter numbers so far, all below target: toy triage **70%** (needs 85%, R7), propose v1 **14.3%** (needs 95%, R28), judge v0 **42.9%, 95% CI 21.4–67.4%** (needs 95%, R30). The judge's base model with the prompt scored **1/14 (7.1%)** (R30). The provider's own score on the same test cases is not shown (not measured).
- Some rows were "graded by the grader" (an LLM grader), others exact (R30). Propose v1 shows **"Prompt only"** and "training examples: Not reported" (R28), so we can't tell whether it's a trained adapter or the base model with the prompt.

**Why it matters to AI labs.** With 3 adapter starts a day, a training that can't qualify is an expensive mistake. Labs want to know before recording whether a step can ever pass.

**Suggested fix.**
1. Show the target, the floor and the minimum test-case count on the task page before training, and explain why a target is 95% or 85%.
2. Warn (or wait) before a training whose test set would be under 20 cases.
3. Add a "provider vs provider" reference row (see F10).
4. Mark each test row as exact or LLM-graded, and explain "Prompt only".

### F10. Free text in a structured output makes a match-graded exam very hard to pass
**What we saw.** Our judge v0 output had a free-text `reason` next to the verdict. Its adapter matched **42.9%** (R30). The most likely cause is ours: the exam compares whole answers, and a 4B model rarely words a reason the way Opus does, even with the same verdict. So we changed the output, not the model:
- **judge v1b (closed output: verdict + reason code from a fixed list):** Opus agrees with itself on **68/68** inputs (lower 95% bound about 95.7%), and Haiku 5.5 agrees with Opus on **65/68 (95.6%)** zero-shot (R32). Recorded **177/177** valid; a self-agreement re-check on the first 40 gave **40/40** (R39). Adapter result: **not measured yet**.
- **early_stop (closed output from the start):** Opus agrees with itself on only **37/40 (92.5%)** for decision + reason, **38/40 (95%)** for the decision alone (R31). Opus flips on borderline runs. An adapter can't beat its teacher's own consistency against a 95% exam, so we recorded **nothing** for this step and use a rule instead (D40).
- Our gate before recording any new task (D39): Opus self-agreement ≥ 97% on the full output, ≥ 150 distinct inputs, no class over 70%, no free text.

**Why it matters to AI labs.** Most real lab outputs carry a reason, a note or a hypothesis, because people read them. A match exam scores the wording, not the decision, so a good adapter fails. And when the teacher isn't consistent with itself, no adapter can pass, however much data it gets; labs should learn that before recording, not after a training.

**Suggested fix.**
1. Grade structured outputs field by field: exact match on enum and number fields, a grader (or nothing) on free-text fields, with users able to mark which fields count.
2. Re-ask the provider on a sample of test cases and show its **self-agreement as the ceiling** next to the target. If the ceiling is below the target, say so before training.
3. Warn at task creation when the schema has free-text fields and the exam is match-graded.

### F11. Generative steps (propose) can't pass a match exam
**What we saw.** `propose_experiments` returns a batch of new experiment configs. Its adapter matched **14.3% vs 95%** (R28). Many different configs are equally good, so matching Opus's exact proposals is a poor test. This is also the step with the most calls (489 propose vs 138 judge in one collection, R20) and the longest wait: in the loop Opus took a p50 of **3.1–4.2 s** (R21), and in a 32-slot race propose latency grew from a p50 of **3.5 s at 1 config to 16.0 s at 8** (R36). So we kept propose on Opus, and Mubit's small models are meant to serve the classification steps (D36, D37).

**Why it matters to AI labs.** In research agents the generative step is where GPUs wait and where most LLM money goes. If only classification steps can pass, the main promise (faster, cheaper agents) skips the step that costs the most.

**Suggested fix.** A grading mode for generative tasks: (1) schema and constraint validity, (2) a grader asking "would the provider accept this as a reasonable alternative?", and (3) optionally an outcome score the user attaches by request ID (the experiment's result), which links to the feedback API in F6. Shadow traffic graded on outcomes would be the strongest version.

### F12. HTTP 503 `engine_unavailable` from the gateway during a recording run
**What we saw.** While recording judge v1 (04:17–04:55 BST), **80 attempts failed with HTTP 503 `engine_unavailable`** at Mubit's gateway: no answer and no example (R39). 177 other calls in the same run succeeded. We don't know whether recording mode played a part.

**Why it matters to AI labs.** In a live loop a failed call means a GPU waits until the client retries; in a recording job it means lost teaching inputs. Without a retry hint, every client invents its own back-off.

**Suggested fix.**
1. Send `Retry-After` with 503s, and say in the error body whether Mubit or the provider failed.
2. Offer an optional pass-through: if the Mubit engine is down, forward the call to the provider and skip recording, so the user's agent keeps working.
3. A status page or incident note, and a sentence on whether failed calls count towards any limit.

### F13. Hosted Sapling endpoint: no way to check it matches the release
**What we saw.** Mubit admins shared a hosted Banyan Sapling endpoint (D33). We plan to use it as the official stock baseline and to check it gives the same logits as our self-hosted run of the release on the same rows. That parity check is **not measured yet**. Our self-hosted run does match the published reference logits (max diff 0.0; R33). The response doesn't say which Hub revision or `calibration.json` it serves, and the endpoint has no auth.

**Why it matters to AI labs.** A lab quoting Sapling numbers needs to know the hosted model, the open weights and the model card are the same thing. Calibration matters most: it decides whether Sapling answers or says `unsure` (F7).

**Suggested fix.** Return the Hub revision and a calibration hash in each response; publish reference logits for a few rows; document the input format and rate limits. A key (even a free one) would protect your bill and let you see usage.

### F14. Diagnose task training failed with no reason
The `diagnose_run` task ("On-call debugging expert", 86 recorded calls, R26) shows *Can't train: No adapter yet: the training run failed* (console, 2026-10-24 ~08:45, R46). The page gives no cause and no retry. A lab can't tell whether the data, the schema or a quota is at fault, so it can't fix the input.
**Ask:** show the failure reason and allow a retry without re-recording.

## Earlier findings, updated

### F1. Adapter replies don't keep the JSON output format
**What we saw.** We sent `output_config.format` with a JSON schema on every toy call, but the task page described the task as "text replies" (R7). In Try models, 3 of 7 hand-checked prompts came back malformed: the key `failure` instead of `failure_class`, and no `fix` field (R9). The malformed replies took **0.36–0.40 s**; the valid ones **1.57–1.83 s** (R9), about the same as Haiku 5.5 (1.0–3.0 s, R4; 1.61–1.74 s in R8). Schema validity of the real-task adapters is **not measured yet** (R30).

**Impact for AI labs.** Lab agents parse replies into configs and verdicts. A malformed reply costs a retry or a fallback while a GPU waits. Worse, the fastest replies are the broken ones, so a latency dashboard that doesn't check validity shows a speed win that isn't real.

**Suggested fix.**
1. Enforce `output_config.format` with constrained decoding when an adapter serves.
2. Until then, add an exam line: "{k} of {n} replies didn't match the output format · needs 0".
3. Leave format-invalid replies out of latency and cost figures, or show them separately.
4. State whether Try-models latency matches routed latency.

### F2. The API doesn't say per call who answered
**What we saw.** The only Mubit response header is `x-middleware-request-id` (probe answers). The console has a **Served by** field, but nothing in the API response tells the caller whether the provider or an adapter answered. All our tasks still show no routes (R29).

**Impact for AI labs.** Research teams need to know which model made each decision, to reproduce a result and to trust a benchmark. Our Mubit arm is designed so that Mubit's small model answers only the steps whose adapter passed and Opus answers the rest (D36); without a per-call signal, the share each model answered can only come from the console, not from our own logs.

**Suggested fix.** A response header such as `x-middleware-served-by: provider` or `adapter:<id>@<version>` (with the route ID), in streaming responses too, plus a read endpoint that returns the same fields for an `mreq_` ID with the read-only key.

### F3. `record: off` also drops the task fingerprint
**What we saw.** The docs say `x-middleware-record: off` *"also drops the call's task fingerprint"*, while `metadata` keeps the fingerprint and drops the content (office hours §2). Our benchmark relies on this: `off` on the Opus arm so (by our reading) its calls are never routed to an adapter, `metadata` on the Mubit arm (`backend/benchmark/configs.py`). Mubit hasn't confirmed this reading yet.

**Impact for AI labs.**
- **Privacy and routing are coupled.** A lab that sends `off` on sensitive calls silently loses routing for them.
- **No per-call way to force the provider while still recording**, so a "reference" arm gives up its examples.
- **Fallbacks may not fall back.** A fallback with the same model ID and fingerprint could be routed to the adapter that just failed. Not measured yet, and without F2 we can't see it.

**Suggested fix.** Two controls: `x-middleware-record: full | metadata | off` for what is stored, and something like `x-middleware-route: auto | provider` for who answers. Document what each combination does for task matching and routing.

### F4. One step across many models: the task fingerprint pushes users to a weaker schema
**What we saw.** Calls join one task only if instructions, tools, tool choice, output format and model are identical. A per-target `propose_experiments` schema would have made three tasks, so we moved to one target-agnostic schema validated in our code (D24). It worked: each real task is one task (R28, R29). Its measured costs:
- A `[{name, value}]` list roughly **doubled output tokens** (about 960–1,040 vs about 550) and raised Opus's median latency from **5.94 s to 9.36 s**; Haiku 5.5 went to **0/3 valid** without enums in the schema (R17).
- A compact settings string brought validity back (Opus 3/3, Haiku 3/3) and output to about 670 tokens, but Opus stayed at a **9.51 s** median (R18). We couldn't A/B it, because the old schema is a different task.

**Impact for AI labs.** Labs run one step over many models and configurations, and must choose between provider-enforced schemas (one task per model) and one shared task (weaker enforcement, more validation code).

**Suggested fix.** Let users group calls explicitly (a task key), or treat schemas that differ only in enum values and descriptions as one family. Document exactly what the fingerprint covers (exact bytes or normalised; whether `output_config.effort` counts).

### F5. Adapters are graded against the provider, not ground truth
**What we saw.** Every adapter grade measures agreement with the provider (R7, R28, R30). Ground truth tells a different story:
- On our planted trap set, Opus 5.5 is perfect: **50/50**, fakes let through **0/25** (R24). On that set an adapter can at best tie its teacher.
- In race-format evidence, Opus judged **92 of 92** trap runs `invalid` and **46 of 46** clean claimed bests `unsure` (R25). Against 10-seed re-runs, Opus confirmed **1 of 7** real gains, accepted **0 of 36** noise claims and let through **0 of 32** fakes (R34).
- Our engine's combination (Opus for leakage, a seed-spread rule for noise) confirmed **7/7** real gains, accepted **1/36** noise and let through **0/32** fakes (R37). The rule alone let **24/32** fakes through (R34). Small set: quote counts.

**Impact for AI labs.** An adapter graded against Opus learns Opus's habits, including answering `unsure` on almost every real gain. A provider-agreement grade can't show an adapter catches what the teacher misses, and 10–14 test cases can't separate 70% from 85% reliably.

**Suggested fix.** Let customers attach a ground-truth set to a task, scored next to provider agreement, with case counts and a confidence interval, and optionally used as a routing gate. Our method (`data/trap_set/README.md`, `scripts/eval_trap_set.py`): planted cases with known answers, a list of acceptable answers per case, a dev/test split where test is never recorded, the same evidence fields on every case, separate error rates (fakes approved, genuine rejected, `unsure`), and no fallback during grading. Adapter scores on these sets: **not measured yet**.

### F6. Docs stop before live serving; outcome links are 404
**What we saw.** The public docs stop at Try models (office hours). Gaps we hit:
- **Routes, shadow, live:** no page says how to create or enable a route, whether shadow needs a passed exam, or how long shadow lasts (R7).
- **Exam and limits:** the 95% target, the 0.50 floor, the 20-case minimum (R30), the review limit (R29), "Prompt only" (R28) and the daily training limit are not in the docs.
- **Corrections and ratings:** every request has "Rate this reply: Good / Bad", but nothing says what a rating does or whether a corrected answer can replace the provider's label.
- **Outcomes:** the outcome-feedback links on mubit.ai (`docs.mubit.ai/patterns/outcome-attribution-loop`, `/api-reference/control-http`) return **404**.
- **Smaller:** the "metrics needs manager" console error names a role the docs don't list; `GET /v1/messages/batches` returns 200 but is undocumented.

**Impact for AI labs.** Routing, correction and outcome feedback turn recorded calls into a cheaper, better agent, and research teams already have an outcome score for every decision. These are the steps the docs don't cover yet.

**Suggested fix.** One page "From adapter to live traffic" (route, shadow, live, rollback, with conditions); one page "Exam and limits" (F8, F9); one page "Feedback" (ratings, corrections, outcomes); redirect the 404s.

### F7. Stock Sapling answers `unsure` on every AI-research claim we tried
**What we saw (measured, R33).** On 500 held-out AI-research claims, stock Sapling gave an answer (not `unsure`) on **0%**; its raw best guess was right **22.0%** of the time. On our 50-case trap set it answered **0 of 50**, scoring 10/50 only where `unsure` counts as correct. Our code reading explains why (`sapling_gate.md` §2): `fit_multidomain` keeps the **maximum** conformal threshold across domains, so ContractNLI's very wide threshold applies everywhere.

**What our head fine-tune does (R33).** `sapling-research`: verification head fine-tuned, backbone frozen, calibration refitted on ML-research claims only. One H100 pass, **$0.64**.
- Raw accuracy **22.0% → 58.2%** (unseen targets 27.4% → 61.3%).
- Answers given **0% → 6.0%** at α 0.01 (29/30 right) and **18.0%** at α 0.05 (80/90 right).
- Fakes caught **0 → 21** (α 0.01) / **43** (α 0.05); fakes approved **0/157** for both.
- ContractNLI (400 cases): raw accuracy **77.3% → 83.8%**.

**Where it still loses (R33).**
- **Opus 5.5 wins the trap set outright: 50/50** vs 14/50 (α 0.01) and 18/50 (α 0.05), where α 0.05 lets **2/25** fakes through.
- 82–94% of research claims still come back `unsure`; genuine gains accepted **0/132** (α 0.01) or **1/132** (α 0.05).
- Without calibration our raw guess would approve **19/157** fakes (stock 1); on ContractNLI raw false-supported went **7 → 8**.
- Our ContractNLI input layout is a guess: stock scores 77% on it vs 86% reported.
- Sapling's edge is speed and price: **73 ms** vs about **2.2 s** for Opus, about **$0.00015** vs **$0.0046** per case.

**Impact for AI labs.** As shipped, Sapling can't be the automatic check in an experiment loop that produces many claims an hour: it hands every research claim back to a human. With domain data it becomes a fast, cheap filter that catches some fakes and approved none of the 157 held-out fakes in our synthetic test set (at the looser α 0.05 it let 2 of 25 trap-set fakes through), but it is not yet a replacement for an Opus judge.

**Suggested fix.**
1. Ship per-domain (group-conditional) thresholds chosen by a domain field, instead of one shared maximum. In our run, refitting calibration alone didn't make the stock head answer; the gain came from the head fine-tune (`docs/sapling_results.md` §1), so a domain-head path matters too.
2. Publish a refit script that writes a new `calibration.json` from labelled, group-split cases (`ConformalCalibrator.fit` already does most of it).
3. Publish the ContractNLI input layout, so outside regression checks are comparable.
4. State on the card that coverage depends on the calibration domain.

We'd be glad to share `sapling-research` as a head-only delta plus our `calibration.json` (ML-research scope only); nothing is published yet.

**Smaller notes from the code review** (`sapling_gate.md`): the demo Space returns hard-coded `mock_response` values; `inference.load_model` compares package versions exactly, so a `+cuXXX` torch wheel fails; `verify_files` re-hashes every weight file on every cold start; `handler.py` doesn't set `CUBLAS_WORKSPACE_CONFIG` the way `inference.py` does.

## Still open on our side
- How to create a route, and whether a per-call "force provider" exists (F2, F3).
- The review limit's exact value and selection rule, and whether it can be raised (F8).
- Why 95% vs 85%, and whether field-level grading is planned (F9, F10).
- Whether a corrections or outcomes API exists (F6, F11).
- Whether self-hosted Sapling inference needs to sit behind Mubit (mostly settled for the stock verifier by the hosted endpoint, D33).
- Adapter billing and the "metrics needs manager" role.

Full wording is in `docs/decisions.md` (Q1–Q15) and `docs/mubit_office_hours.md`.

## Where the evidence lives
- `docs/results_log.md`: R1–R39 with request IDs (`mreq_…`). Toy: task `ln_74da54ace35187e2085fc0e9`, adapter `ad_lane_fa72c0ab`. Propose: task `ln_42bac9601f877d03fc0999bc`, adapter `ad_lane_68db9770`, training job `ftjob_e983ef33`, test set `gs_r4_ckac12cace699e5f91`. Judge v0: test set `gs_r4_ck8a6cdc8cd493233d`.
- `data/gates/judge_v1/` (gate and recording), `data/early_stop/` (failed gate), `scripts/gate_judge_v1.py`, `scripts/record_judge_v1.py`.
- `data/trap_set/`, `data/noise_set/`, `scripts/eval_trap_set.py` (ground-truth grading).
- `data/sapling/sapling_v1/`, `docs/sapling_results.md`, `docs/sapling_gate.md` (Sapling results and code review).
- `backend/agent/mubit_client.py` (record modes, request-ID capture), `backend/benchmark/configs.py` (per-arm record modes), `backend/agent/prompts/` (frozen prompts).
