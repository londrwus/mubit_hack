# Field report for the Mubit team

From the Busy Machines team (Night Shift London, 23–25 Oct 2026). We built an ML research agent whose recurring decisions (plan experiments, diagnose failed runs, judge claimed improvements, and a tested early-stop step) all run through the Mubit gateway, and we took four real tasks from recorded Opus 5.5 calls to trained adapters, in three workspaces. This report lists what we ran into and what we'd suggest. We wrote it to be useful to you, and we know the platform is two days old.

**Evidence rules.** Every measured number comes from our results log (`docs/results_log.md`, entries R1–R92) and carries its R-id. Other sources are our probe answers and decisions (`docs/decisions.md`), our docs review and the Discord answer we quote (`docs/mubit_office_hours.md`) and our code reading of the Banyan Sapling release (`docs/sapling_gate.md`); where a point is a reading or a hypothesis rather than a measurement, we say so. Anything we haven't measured is marked "not measured yet". Our test sets are small (10–35 exam cases per adapter, 7 real gains in one set), so treat this as early signal, not a verdict.

**What changed since the second version (R1–R39).** Five new findings: the same data scoring very differently in two workspaces (F15), training jobs that sat in "Training" for hours (F16), going-live minimums that a low-traffic step can't reach in time (F17), no way yet to send a call to a chosen adapter (F18), and our Sapling fine-tune, offered as a contribution (F19). F6, F7, F13 and F14 are updated with what we learned since (R42, R56, R80, R81, R92).

## Summary
| # | Finding | Evidence | Impact for AI-lab users | Suggested fix (short) |
|---|---|---|---|---|
| F15 | Same 177 inputs, same prompt, same task key, two workspaces: the judge v1 adapter matched **11.4%** (B, 35 cases) vs **58.3%** (C, 24 cases); both tasks are listed as "text replies" although every call sent a JSON schema | R39, R59, R61, R70, R79, R80 | A lab can't tell whether a low score means "this step can't be learned" or "something went wrong in this run", so it can't decide what to fix | Show the adapter's answer next to the provider's per test case, a format-validity rate and how replies are compared; let users set the reply type to JSON |
| F16 | Training jobs sat in "Training" for ~3 h (C) and ~7 h (B) with no progress, queue position or error; the toy trained in ~5 min | R6, R47, R55, R58, R68, R70, R79 | Planning around a ~25 min pipeline breaks; we re-recorded 177 Opus calls into a new workspace as a hedge | Job state (queued / running / step), absolute timestamps, ETA, a status page, a notice on failure or retry |
| F17 | Going live needs a silent test (10% of calls, up to 100 a day) and a partial rollout (at least 100 calls); a judge that fires only on a new best makes ~45–57 calls per race | R56, R75, R82, D52, D54 | The closed-output steps most likely to pass are the rarest calls, so a passing adapter can't finish rollout within days | Minimums that scale with traffic, an offline shadow on recorded inputs, and an ETA to live on the task page |
| F18 | No way to send a call to a chosen adapter yet (Discord: a header is planned) | office hours (docs, Discord 24 Oct 11:51), D36 | Benchmarks can't pin the adapter on one arm and keep the baseline on the provider | Ship the header, echo who served, document whether below-target versions can be pinned |
| F19 | **Contribution:** stock Sapling answers `unsure` on every ML-research claim; a head-only fine-tune on one H100 makes it answer and catch fakes, now public | R33, R42, R76, R81, R83, R92 | Out of the box Sapling can't be the automatic check in a research loop | Per-domain heads and thresholds; we offer `sapling-research` and our `ml-claims` set |
| F8 | A per-task daily review limit decides what the first adapter learns from | R28, R29 | Agents make hundreds of calls an hour; the first adapter trains on a few dozen and uses a daily adapter start | Document the limit, show the quota, sample the queue for coverage, let users hold training |
| F9 | The exam rules (95% match, 0.50 floor, ≥20 test cases) are visible only after training | R28, R30, R56 | A training can't qualify however well it does, and still costs one of 3 daily starts | Show the rules before training; warn when the test set is under 20 |
| F10 | Free text in a structured output makes a whole-answer match exam very hard to pass | R30, R31, R32, R39, D39 | Most lab outputs carry a reason or a note; the match exam punishes wording, not decisions | Field-level grading; show the provider's self-agreement as the ceiling |
| F11 | Generative steps (propose) can't pass a match exam | R28, R20, R21, R36 | The slowest, most frequent step, where GPUs wait, is the one that can't be served | Outcome or validity grading for generative tasks |
| F12 | HTTP 503 `engine_unavailable` from the gateway during a recording run | R39 (none in R59) | A lab loop stalls or loses inputs; no hint when to retry | `Retry-After`, a clear "Mubit vs provider" error, an optional pass-through to the provider |
| F13 | Hosted Sapling endpoint: labels match our self-hosted release, logits don't exactly, and nothing says which revision it serves | R33, R42, D33 | Labs can't tell whether published Sapling numbers apply to the endpoint | Revision and calibration hash in the response; reference logits |
| F14 | Diagnose task: "Can't train: the training run failed", no reason; it later trained (8.3%) with no notice | R46, R80 | 86 recorded calls gave no adapter and no next step for hours | A failure reason (data, schema, quota), a retry button, a notice when Mubit retries |
| F1 | Adapter replies don't keep the JSON output format | R7, R8, R9, R79 | Agents that parse replies break; the broken replies are also the fast ones | Constrained decoding for `output_config.format`, plus a "malformed outputs: needs 0" exam line |
| F2 | The API doesn't say per call who answered | probe answers, R29 | Labs can't audit which model made a decision or measure adapters in their own tooling | A `served-by` response header and a lookup by request ID |
| F3 | `record: off` also drops the task fingerprint | docs, `backend/benchmark/configs.py` | Privacy, routing and "force the provider" are tied to one header | Separate "store content" from "who answers" |
| F4 | One step across many models: the task fingerprint pushes users to a weaker schema | R17, R18, R28, D24 | Labs choose between schema enforcement and one shared task | A task key or schema-family grouping; document the fingerprint rules |
| F5 | Adapters are graded against the provider, not ground truth | R7, R9, R24, R25, R34, R37, R81 | An adapter graded against Opus learns Opus's blind spots, e.g. answering `unsure` on real gains | Customer-supplied ground-truth sets, scored next to provider agreement |
| F6 | Docs stop before live serving; outcome links are 404 (the going-live controls exist, under Settings) | office hours, R28, R30, R56, R58 | Users can train an adapter but can't find how to route, correct or score it | One "adapter to live traffic" page and one "feedback" page |
| F7 | Stock Sapling answers `unsure` on every AI-research claim we tried (details and our fix in F19) | R33, R81, `sapling_gate.md` §2 | As a verifier for research claims it defers everything to a human | Per-domain calibration and a refit script |

## What worked well
Measured, and worth saying first:
- **Gateway overhead under 1 ms** per call (provider 1.46 s; R3). **Prompt caching passes through** (10,353 tokens written, then read; R2). Structured outputs pass through for provider calls, and 6 parallel calls returned no `429` (probe answers).
- **Recording at scale works.** One collection recorded 489 valid `propose_experiments` and 138 valid `judge_improvement` calls (R20); the closed-output judge recorded **177 of 177** valid answers in B (R39) and again **177/177 with no errors** in C (R59).
- **One task across three targets held** (D24): each real task is one task, 100% `claude-opus-5-5` (R28, R29). Kept rates so far: propose 66% (29/44), judge v0 80% (56/70) (R29), judge v1 **177/177** in B (R47) and **159 kept, 0 rejected** in C (R61).
- **A fast pipeline when it runs well.** Toy task: 40 calls to a trained adapter in about 20–25 minutes (R6). Workspace C: grouped about 7 minutes after recording, 120 kept about 20 minutes after that, training started about 20 minutes after recording ended (R59, R61).
- **Training helps.** The judge v0 adapter matched 6/14 test cases vs 1/14 for the base model with the prompt (R30), and the evaluation page showed a 95% confidence interval (21.4–67.4%), which is exactly the right thing to show at this size. Our best adapter, judge v1 in C, matched **58.3%** (R79).
- **The settings pages are clear once found.** Settings → Training says new versions come after **25 new examples**, examples up to **256 KB**, and **no daily training limit** (R58); Settings → Going live shows the approval rule, silent test and partial rollout (R56).
- **Mubit retried a failed training on its own:** the diagnose task, shown as failed at R46, had a trained v1 by R80.
- **The Sapling release reproduces.** The package's reference example matched the published H200 logits exactly (max diff 0.0; R33), and again on our H100 (R76).
- Every model we needed was available, a hosted Sapling endpoint was shared with us quickly (D33), and Mubit support answered the adapter-targeting question on Discord (to another team) with a clear plan (office hours).

## New findings (R40–R92)

### F15. Same inputs, two workspaces, very different adapter scores
**What we saw.** When B's training stalled (F16) we recorded the same judge v1 inputs again into a new workspace C, as a hedge (R59). Everything we control was identical; only the workspace differed:

| | Workspace B | Workspace C |
|---|---|---|
| Inputs, prompt, model | 177 judge v1 inputs, prompt `5c1d3cfd2941`, Opus 5.5, record full (R39) | same 177 inputs, same prompt, same model and mode (R59) |
| Task | `ln_5d9ea82de398d3c492ec873f` | `ln_dfe18721334e06d23c32f338`, **same task key** `lk1_be9a22ab…` (R59) |
| Review | 177 kept, 0 rejected (R47) | 159 kept, 0 rejected, **18 "couldn't be reviewed"**, no reason shown (R61) |
| Adapter | `ad_lane_9033f7ff`, Qwen3-4B-Instruct-2507 (R70) | `ad_lane_52d6d485`, Qwen3-4B-Instruct-2507, PEFT (R79) |
| Score vs Opus | **11.4% of 35 test cases** (R70) | **58.3% of 24 test cases** (R79) |

The output is closed: a verdict from 3 values plus a reason code from a fixed list, no free text (D37). Opus agrees with itself on **68/68** inputs (R32) and **40/40** on a re-check (R39), and Haiku 5.5 zero-shot agrees with Opus on **65/68** (R32). So 11.4% is far below what a cheap provider model (Haiku 5.5) gets with no training, and below the judge v0 adapter with free text (42.9%, R30). By our arithmetic on the logged counts (about 4/35 and 14/24), the two 95% intervals don't overlap (roughly 5–26% vs 39–76%), so different test cases alone are an unlikely explanation; we haven't verified the cause.

Two observations that may be related (hypotheses, not verified):
- **Both tasks are described as "· text replies"** (R79), although every call sent a JSON schema in `output_config.format`, the same as the toy task (R7). If the exam compares raw text, JSON spacing or key order could count as a miss even when the verdict and code agree; the toy adapter did produce wrong keys (R9, F1).
- C's "Its traffic, 7 days" panel showed **354 calls** while the task showed 177 (R59), as if traffic for the shared task key were counted across both workspaces.

Across all our tasks the adapter scores so far are 8.3%, 11.4%, 14.3%, 42.9%, 58.3% and 70% (toy) (R80); none passed its target.

**Why it matters to AI labs.** A lab decides from one score whether a step is worth teaching. Here the same data gave "near chance" in one place and "our best adapter" in another, and the card doesn't show what the adapter actually answered. Without that, the lab can't tell a learning problem from a format, review or scoring problem, and can't decide whether to re-record, reshape the output or give up on the step.

**Suggested fix.**
1. On the evaluation page, show the adapter's reply next to the provider's for each test case, with the reason it counted as a miss.
2. Report a **format-validity rate** (replies that parse against the task's schema) separately from the match score.
3. Say how replies are compared (parsed JSON vs raw text) and let users set the reply type to "JSON replies" when the calls carry a schema.
4. Show the confidence interval on the adapter card, not only on the evaluation page, so 11.4% of 35 and 58.3% of 24 read as what they are.
5. Explain "couldn't be reviewed" (R61), and say whether tasks with the same task key in different workspaces share anything (traffic counts, test sets, training).

### F16. Training jobs sat in "Training" for hours with no status
**What we saw.**
- **B** (`ftjob_10fa515e`): recording ended 04:55 (R39); grouped about an hour later, reviewed about 2 h after that (R47); "Training" at 08:55 (R47), at 10:05 (R55) and at ~14:10 (R68); trained by ~14:45 after **about 7 h in "Training"** (R70).
- **C** (`ftjob_2dd8c4a0`): training started 11:21:43 (R61, R79); still "Training v1" at ~14:10 (R68); trained **about 3 h after it started** (R79).
- For comparison, the toy task trained in **about 5 minutes** (R6).
- It wasn't a limit on our side: Settings shows **no daily training limit**, and our examples are 8.4 KB median, 10.0 KB max, against a 256 KB cap (R58).
- The console gives relative times ("3 h ago", so ±30 min, R47), no queue position or progress, and a footer "2/3" whose meaning isn't stated (R47).

**Why it matters to AI labs.** We planned the day around the ~25-minute pipeline from Friday's toy run. With no status we couldn't tell "queued", "slow" and "stuck" apart, so we spent **$0.639** re-recording 177 Opus calls into a new workspace as a hedge (R59), and the Mubit-arm race we'd planned for once an adapter served never ran (D46, D54). A lab with a training schedule (and 3 adapter starts a day) needs to know whether to wait or act.

**Suggested fix.**
1. Show the job state: queued (with position), running (with step or epoch), evaluating, failed (with a reason), and absolute start times.
2. An ETA, or at least a typical duration for the task's size.
3. A status page or incident note when training is slow across the platform.
4. A notice (email or webhook) when a job finishes, fails or is retried (see F14).
5. Make the training job ID something support can look up, and say whether two jobs with the same task key share a queue.

### F17. Going-live minimums are out of reach for low-traffic steps within an event
**What we saw.** Settings → Going live in B (R56): approval **automatically, once it passes**; silent test **10% of calls, up to 100 a day**; partial rollout **10% of calls, for at least 100 calls**; quality bar **90% of test cases** (the adapter cards showed 95%, R46, R70). The user raised B's settings to the highest the page appeared to allow (silent test 20%, partial rollout 50%); the 100-call minimum stayed (R56).

Our judge is called only when a run claims a new best (D16), so a 15-minute race makes about **45–57 judge calls per arm** (R75, R82). At the default shares, a passing judge adapter would need on the order of 1,000 judge calls to finish rollout (our estimate, R56): many races, or days of normal traffic. We lowered the quality bar to 55% in workspace C only, to give C's 58.3% adapter (R79) a chance to go live (D52), and said so wherever we mention it. **No adapter served a call in any of our races before the freeze** (D54).

**Why it matters to AI labs.** The steps most likely to pass an exam are closed-output classification steps (F10), and in a well-designed agent those are the rarest calls: a judge fires on a new best, a diagnose on a failure. A silent test and rollout sized for high-traffic chat tasks means the best candidates for adapters are the slowest to go live. That's sensible caution for production, but it hides the value from anyone evaluating Mubit in a week, or a hackathon.

**Suggested fix.**
1. Minimums that scale with traffic: "N calls or T days, whichever comes first", with N set per task.
2. An **offline shadow**: run the candidate on the task's recorded inputs (you already have them) and report agreement, so a low-traffic step doesn't wait for live calls to prove itself.
3. A "sandbox" share for a named session or key (100% to the candidate, provider fallback), clearly labelled as evaluation.
4. On the task page, an estimate of time to live at the current traffic rate, and one quality bar (or an explanation of why the Settings bar and the card target differ).

### F18. No way yet to send a call to a chosen adapter
**What we saw.** The docs say *"There is no Mubit model ID: you can't send a call to an adapter by naming it as the model"*, and *"A header can lower what Mubit keeps of a call, never raise it"* (office hours, docs re-check). Mubit support on Discord (24 Oct 11:51, to another team): *"Currently our auto router handles which reqs go where. We will push an update later which allows you to put adapter id in header to specify which adapter the req should go to."* We're ready for it: `data/mubit_projects.json` takes an `adapter_headers` block, and `backend/agent/mubit_client.py` sends it only on the Mubit arm and refuses it on any other arm (office hours). Our fallback path goes to the provider with record off and no adapter header (R88). The header name and the adapter-ID format aren't known yet.

**Why it matters to AI labs.** Labs evaluate a model before they trust it. With only automatic routing, a benchmark can't pin the adapter on one arm and keep the baseline arm on the provider (D36), can't A/B two versions, and can't reproduce a result later with the same version. It also ties into F2 (who answered) and F3 (forcing the provider).

**Suggested fix.**
1. Ship the header (something like `x-middleware-adapter: ad_lane_…@v1`) and document its name, ID format and errors.
2. Echo who served in the response (F2), so pinned calls can be checked from the caller's own logs.
3. Say whether a below-target version can be pinned (clearly labelled as evaluation), whether pinned calls are still recorded and shown as "Served by", and the latency and price per adapter call.
4. Pair it with a "force provider" control (F3), so a baseline arm can be guaranteed provider-only while still recording.

### F19. A contribution: Sapling for ML-research claims
This one isn't a complaint. Banyan Sapling is a strong, open release, and we used it as the independent checker for "is this improvement real?". Out of the box it defers every ML-research claim to a human; a small, cheap fine-tune changes that, and we've published it for anyone to use or improve.

**Stock Sapling on our claims.**
- Synthetic held-out research claims (500): it gave an answer (not `unsure`) on **0%**; its raw best guess was right 22.0% (R33).
- **226 real claims from our races** (judge-format, 152 with ground truth): `unsure` on **152 of 152**; fakes caught 0 of 109 (R81). Our code reading explains why: `fit_multidomain` keeps the maximum conformal threshold across domains, so ContractNLI's wide threshold applies everywhere (`sapling_gate.md` §2, F7).

**What a head-only fine-tune does** (`sapling-research` v2: verification head fine-tuned, backbone frozen, 5-seed head ensemble anchored to the stock head, calibration refitted on ML-research claims only; R42):
- Synthetic held-out claims: raw accuracy **22.0% → 66.0%**; answers given **0% → 19.0%** (α 0.01) / **63.8%** (α 0.05); fakes approved **0 of 157** in every column; fakes caught **0 → 52 / 118** (R42).
- **Real race claims (out of distribution for v2):** at α 0.05, fakes caught **57 of 109**, fakes approved **0**, `unsure` on 24 of 152 instead of 152; agreement with Opus 72.1% (R81).
- Cost: one H100 pass for the head (**$0.64**, R33) plus one more for the extra rows ($0.43, R42). End to end on an H100 the package reproduces its reference logits to 4.8e-7 and takes **99.4 ms per claim** vs about 1.7 s for an Opus judge (R76, R81).
- Our ContractNLI set (400 rows, our own input layout, not the card's 731 cases): 77.3% → 84.3%, no regression (R42, R81).

**Where it still falls short (R42, R77, R81, R83).**
- Opus 5.5 is still much better: 109 of 109 real-race fakes caught (R81) and 50/50 on our trap set vs v2's 36/50 at α 0.05 (R42).
- No verifier confirmed a real gain on the race claims: 0 of 7 for stock, v2 and Opus alike (R81).
- At α 0.05 the conformal guarantee breaks on new templates (77.8% coverage vs a nominal 95%, R42).
- Two later versions failed the gates we declared before training and weren't released: v3 (R77) and v4 (R83). v4 caught 97 of 109 real-race fakes at α 0.01 and confirmed 5 of 7 real gains, but regressed on ContractNLI (79.5%) and approved one fake at α 0.05 (R83).

**What we're offering.** Both are public under Apache-2.0, with a NOTICE crediting MubitLabs and a card that says it's a community fine-tune, not an official release (R92):
- the model, as a head-only delta plus our calibration: https://huggingface.co/busy-machines/sapling-research
- the claims set (test, trap50, trap set v1): https://huggingface.co/datasets/busy-machines/ml-claims

**Suggested fix.**
1. Per-domain (group-conditional) thresholds selected by a domain field, instead of one shared maximum; and since refitting calibration alone didn't make the stock head answer (`docs/sapling_results.md` §1), a supported path for per-domain heads too.
2. A refit script that writes a new `calibration.json` from labelled, group-split cases (`ConformalCalibrator.fit` already does most of it).
3. If useful, take `ml-claims` as an ML-research regression set, or link `sapling-research` from the Sapling card as a community domain head (Q8). We'd be glad to help.

## Findings from the second version (R20–R39)

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
- Targets: **95%** for our new tasks, **85%** for the toy task (R7, R28, R30). We found no explanation of the difference. Settings → Going live later showed a quality bar of **90%** (R56).
- The evaluation page says "evidence for a move" needs the verdict passed, **conformance pass rate ≥ 0.50**, and **≥ 20 judged test cases** (R30). Both early real-task adapters had **14** (R28, R30), so neither could qualify at any score. The test set grows with kept examples, which the review limit (F8) caps. The judge v1 adapters had 35 and 24 (R70, R79).
- Adapter numbers so far, all below target: toy triage **70%** (needs 85%, R7), propose v1 **14.3%** (needs 95%, R28), judge v0 **42.9%, 95% CI 21.4–67.4%** (needs 95%, R30), diagnose v1 **8.3%** (R80), judge v1 **11.4%** (B) and **58.3%** (C) (R70, R79). The judge v0 base model with the prompt scored **1/14 (7.1%)** (R30). The provider's own score on the same test cases is not shown (not measured).
- Some rows were "graded by the grader" (an LLM grader), others exact (R30). Propose v1 shows **"Prompt only"** and "training examples: Not reported" (R28); the docs later explained that "Prompt only" means the base model did as well as the adapter with the prompt (office hours, docs re-check).

**Why it matters to AI labs.** With 3 adapter starts a day, a training that can't qualify is an expensive mistake. Labs want to know before recording whether a step can ever pass.

**Suggested fix.**
1. Show the target, the floor and the minimum test-case count on the task page before training, and explain why a target is 95%, 90% or 85%.
2. Warn (or wait) before a training whose test set would be under 20 cases.
3. Add a "provider vs provider" reference row (see F10).
4. Mark each test row as exact or LLM-graded, and explain "Prompt only" on the card.

### F10. Free text in a structured output makes a match-graded exam very hard to pass
**What we saw.** Our judge v0 output had a free-text `reason` next to the verdict. Its adapter matched **42.9%** (R30). The most likely cause is ours: the exam compares whole answers, and a 4B model rarely words a reason the way Opus does, even with the same verdict. So we changed the output, not the model:
- **judge v1b (closed output: verdict + reason code from a fixed list):** Opus agrees with itself on **68/68** inputs (lower 95% bound about 95.7%), and Haiku 5.5 agrees with Opus on **65/68 (95.6%)** zero-shot (R32). Recorded **177/177** valid; a self-agreement re-check on the first 40 gave **40/40** (R39). Adapter results: 11.4% and 58.3% in two workspaces from the same data (F15).
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
**What we saw.** While recording judge v1 (04:17–04:55 BST), **80 attempts failed with HTTP 503 `engine_unavailable`** at Mubit's gateway: no answer and no example (R39). 177 other calls in the same run succeeded. The same recording into workspace C at ~11:20 had **no errors** (R59). We don't know whether recording mode or load played a part.

**Why it matters to AI labs.** In a live loop a failed call means a GPU waits until the client retries; in a recording job it means lost teaching inputs. Without a retry hint, every client invents its own back-off.

**Suggested fix.**
1. Send `Retry-After` with 503s, and say in the error body whether Mubit or the provider failed.
2. Offer an optional pass-through: if the Mubit engine is down, forward the call to the provider and skip recording, so the user's agent keeps working.
3. A status page or incident note, and a sentence on whether failed calls count towards any limit.

### F13. Hosted Sapling endpoint: labels match, logits don't exactly, and nothing says which revision it serves
**What we saw.** Mubit admins shared a hosted Banyan Sapling endpoint (D33), and we ran a parity check against our self-hosted run of the release on 30 rows: **raw label and status agree 30/30**, but logits are close rather than identical (max diff 0.87, median 0.24). On Sapling's own reference example the hosted endpoint differs by 0.115, while our self-hosted run matches the published logits exactly (R33, R42). The hosted endpoint took about 1.1 s per call vs 75 ms on our H100 (R42). The response doesn't say which Hub revision or `calibration.json` it serves, and the endpoint has no auth.

**Why it matters to AI labs.** A lab quoting Sapling numbers needs to know the hosted model, the open weights and the model card are the same thing. Calibration matters most: it decides whether Sapling answers or says `unsure` (F7, F19), and a logit gap near a threshold can flip a status even when the labels agree on most rows.

**Suggested fix.** Return the Hub revision, dtype and a calibration hash in each response; publish reference logits for a few rows; document the input format and rate limits. A key (even a free one) would protect your bill and let you see usage.

### F14. Diagnose task training failed with no reason, then trained with no notice
The `diagnose_run` task ("On-call debugging expert", 86 recorded calls, R26) showed *Can't train: No adapter yet: the training run failed* (console, 2026-10-24 ~08:45, R46). The page gave no cause and no retry. By ~18:25 it had a trained v1 matching **8.3% on 12 test cases** (target 95%), created about 22 minutes earlier (R80), so Mubit retried it; nothing told us. A lab can't tell whether the data, the schema or a quota was at fault, so it can't fix the input, and it doesn't know when to look again.
**Ask:** show the failure reason, allow a retry without re-recording, and notify when an automatic retry starts or finishes (F16).

## Earlier findings, updated

### F1. Adapter replies don't keep the JSON output format
**What we saw.** We sent `output_config.format` with a JSON schema on every toy call, but the task page described the task as "text replies" (R7); the judge v1 task in C is described the same way (R79, F15). In Try models, 3 of 7 hand-checked prompts came back malformed: the key `failure` instead of `failure_class`, and no `fix` field (R9). The malformed replies took **0.36–0.40 s**; the valid ones **1.57–1.83 s** (R9), about the same as Haiku 5.5 (1.0–3.0 s, R4; 1.61–1.74 s in R8). Schema validity of the real-task adapters is **not measured yet** (R30): no real-task adapter has served a call (D54).

**Impact for AI labs.** Lab agents parse replies into configs and verdicts. A malformed reply costs a retry or a fallback while a GPU waits. Worse, the fastest replies are the broken ones, so a latency dashboard that doesn't check validity shows a speed win that isn't real.

**Suggested fix.**
1. Enforce `output_config.format` with constrained decoding when an adapter serves.
2. Until then, add an exam line: "{k} of {n} replies didn't match the output format · needs 0".
3. Leave format-invalid replies out of latency and cost figures, or show them separately.
4. State whether Try-models latency matches routed latency.

### F2. The API doesn't say per call who answered
**What we saw.** The only Mubit response header is `x-middleware-request-id` (probe answers). The console has a **Served by** field, but nothing in the API response tells the caller whether the provider or an adapter answered. All our tasks still show no routes (R29, R79, R80).

**Impact for AI labs.** Research teams need to know which model made each decision, to reproduce a result and to trust a benchmark. Our Mubit arm is designed so that Mubit's small model answers only the steps whose adapter passed and Opus answers the rest (D36); without a per-call signal, the share each model answered can only come from the console, not from our own logs.

**Suggested fix.** A response header such as `x-middleware-served-by: provider` or `adapter:<id>@<version>` (with the route ID), in streaming responses too, plus a read endpoint that returns the same fields for an `mreq_` ID with the read-only key. This pairs naturally with the planned adapter header (F18).

### F3. `record: off` also drops the task fingerprint
**What we saw.** The docs say `x-middleware-record: off` *"also drops the call's task fingerprint"*, while `metadata` keeps the fingerprint and drops the content (office hours §2). Our benchmark relies on this: `off` on the Opus arm so (by our reading) its calls are never routed to an adapter, `metadata` on the Mubit arm (`backend/benchmark/configs.py`). Mubit hasn't confirmed this reading yet.

**Impact for AI labs.**
- **Privacy and routing are coupled.** A lab that sends `off` on sensitive calls silently loses routing for them.
- **No per-call way to force the provider while still recording**, so a "reference" arm gives up its examples.
- **Fallbacks may not fall back.** A fallback with the same model ID and fingerprint could be routed to the adapter that just failed. Not measured yet, and without F2 we can't see it; our client now sends fallbacks with record off and no adapter header (R88).

**Suggested fix.** Two controls: `x-middleware-record: full | metadata | off` for what is stored, and something like `x-middleware-route: auto | provider` for who answers. Document what each combination does for task matching and routing.

### F4. One step across many models: the task fingerprint pushes users to a weaker schema
**What we saw.** Calls join one task only if instructions, tools, tool choice, output format and model are identical. A per-target `propose_experiments` schema would have made three tasks, so we moved to one target-agnostic schema validated in our code (D24). It worked: each real task is one task (R28, R29). Its measured costs:
- A `[{name, value}]` list roughly **doubled output tokens** (about 960–1,040 vs about 550) and raised Opus's median latency from **5.94 s to 9.36 s**; Haiku 5.5 went to **0/3 valid** without enums in the schema (R17).
- A compact settings string brought validity back (Opus 3/3, Haiku 3/3) and output to about 670 tokens, but Opus stayed at a **9.51 s** median (R18). We couldn't A/B it, because the old schema is a different task.

**Impact for AI labs.** Labs run one step over many models and configurations, and must choose between provider-enforced schemas (one task per model) and one shared task (weaker enforcement, more validation code).

**Suggested fix.** Let users group calls explicitly (a task key), or treat schemas that differ only in enum values and descriptions as one family. Document exactly what the fingerprint covers (exact bytes or normalised; whether `output_config.effort` counts), and how a task key behaves across workspaces (F15).

### F5. Adapters are graded against the provider, not ground truth
**What we saw.** Every adapter grade measures agreement with the provider (R7, R28, R30, R70, R79). Ground truth tells a different story:
- On our planted trap set, Opus 5.5 is perfect: **50/50**, fakes let through **0/25** (R24). On that set an adapter can at best tie its teacher.
- In race-format evidence, Opus judged **92 of 92** trap runs `invalid` and **46 of 46** clean claimed bests `unsure` (R25). Against 10-seed re-runs, Opus confirmed **1 of 7** real gains, accepted **0 of 36** noise claims and let through **0 of 32** fakes (R34). On 152 ground-truth real race claims, Opus confirmed **0 of 7** real gains (R81).
- Our engine's combination (Opus for leakage, a seed-spread rule for noise) confirmed **7/7** real gains, accepted **1/36** noise and let through **0/32** fakes (R37). The rule alone let **24/32** fakes through (R34). Small set: quote counts.

**Impact for AI labs.** An adapter graded against Opus learns Opus's habits, including answering `unsure` on almost every real gain. A provider-agreement grade can't show an adapter catches what the teacher misses, and 10–35 test cases can't separate 70% from 85% reliably.

**Suggested fix.** Let customers attach a ground-truth set to a task, scored next to provider agreement, with case counts and a confidence interval, and optionally used as a routing gate. Our method (`data/trap_set/README.md`, `scripts/eval_trap_set.py`, and `scripts/eval_served_calls.py` for adapter-served calls): planted cases with known answers, a list of acceptable answers per case, a dev/test split where test is never recorded, the same evidence fields on every case, separate error rates (fakes approved, genuine rejected, `unsure`), and no fallback during grading. Adapter scores on these sets: **not measured yet** (no adapter served, D54).

### F6. Docs stop before live serving; outcome links are 404
**What we saw.** The public docs stop at Try models (office hours). The going-live controls do exist: we found them under the admin **Settings → Going live** and **Settings → Training** tabs (R56, R58), which answered several questions below. Gaps we hit:
- **Routes, shadow, live:** no docs page describes the approval rule, silent test or partial rollout, or how long each takes (R56; first asked at R7).
- **Exam and limits:** the 95% card target vs the 90% Settings bar, the 0.50 floor, the 20-case minimum (R30, R56), the review limit (R29) and "Prompt only" (R28) are not in the docs. Settings states "no daily training limit" (R58), while the docs mention one.
- **Corrections and ratings:** every request has "Rate this reply: Good / Bad", but nothing says what a rating does or whether a corrected answer can replace the provider's label.
- **Outcomes:** the outcome-feedback links on mubit.ai (`docs.mubit.ai/patterns/outcome-attribution-loop`, `/api-reference/control-http`) return **404**.
- **Smaller:** the "metrics needs manager" console error names a role the docs don't list; `GET /v1/messages/batches` returns 200 but is undocumented.

**Impact for AI labs.** Routing, correction and outcome feedback turn recorded calls into a cheaper, better agent, and research teams already have an outcome score for every decision. These are the steps the docs don't cover yet.

**Suggested fix.** One page "From adapter to live traffic" (approval, silent test, partial rollout, rollback, with the Settings fields and how long each step takes at a given traffic rate, F17); one page "Exam and limits" (F8, F9); one page "Feedback" (ratings, corrections, outcomes); redirect the 404s.

### F7. Stock Sapling answers `unsure` on every AI-research claim we tried
**What we saw (measured).** On 500 synthetic held-out AI-research claims, stock Sapling gave an answer (not `unsure`) on **0%**; its raw best guess was right **22.0%** of the time. On our 50-case trap set it answered **0 of 50**, scoring 10/50 only where `unsure` counts as correct (R33). On **226 real race claims** it answered `unsure` on all 152 ground-truth rows (R81). Our code reading explains why (`sapling_gate.md` §2): `fit_multidomain` keeps the **maximum** conformal threshold across domains, so ContractNLI's very wide threshold applies everywhere.

**Impact for AI labs.** As shipped, Sapling can't be the automatic check in an experiment loop that produces many claims an hour: it hands every research claim back to a human.

**Suggested fix and our contribution:** see F19 (per-domain thresholds and heads, a refit script, and our public `sapling-research` head and `ml-claims` set). Also: publish the ContractNLI input layout, so outside regression checks are comparable (our 400-row set isn't the card's 731 cases, R81), and state on the card that coverage depends on the calibration domain.

**Smaller notes from the code review** (`sapling_gate.md`): the demo Space returns hard-coded `mock_response` values; `inference.load_model` compares package versions exactly, so a `+cuXXX` torch wheel fails; `verify_files` re-hashes every weight file on every cold start; `handler.py` doesn't set `CUBLAS_WORKSPACE_CONFIG` the way `inference.py` does.

## Still open on our side
- The adapter header: its name, ID format and ETA; whether a below-target version can be pinned; whether pinned calls are recorded and shown as "Served by" (F18).
- Why B and C scored so differently from the same data, and what "text replies" means for scoring (F15).
- Whether a per-call "force provider" exists (F2, F3).
- What `ftjob_10fa515e` and `ftjob_2dd8c4a0` were waiting for (F16).
- Whether silent-test and rollout minimums can be lowered for low-traffic tasks (F17).
- The review limit's exact value and selection rule, and whether it can be raised (F8).
- Why 95% vs 90% vs 85%, and whether field-level grading is planned (F9, F10).
- Whether a corrections or outcomes API exists (F6, F11).
- Whether Mubit would link `sapling-research` from the Sapling card (Q8, F19).
- Adapter billing and the "metrics needs manager" role.

Full wording is in `docs/decisions.md` (Q1–Q15) and `docs/mubit_office_hours.md`.

## Where the evidence lives
- `docs/results_log.md`: R1–R92 with request IDs (`mreq_…`).
  - Toy: task `ln_74da54ace35187e2085fc0e9`, adapter `ad_lane_fa72c0ab`.
  - Propose: task `ln_42bac9601f877d03fc0999bc`, adapter `ad_lane_68db9770`, training job `ftjob_e983ef33`, test set `gs_r4_ckac12cace699e5f91`. Judge v0: test set `gs_r4_ck8a6cdc8cd493233d`.
  - Judge v1, workspace B (`ws-2202e825d19648d2`, project `13123`): task `ln_5d9ea82de398d3c492ec873f`, adapter `ad_lane_9033f7ff`, job `ftjob_10fa515e`.
  - Judge v1, workspace C (`ws-80b9fd6027bad9e4`): task `ln_dfe18721334e06d23c32f338`, adapter `ad_lane_52d6d485`, job `ftjob_2dd8c4a0`, test set `gs_r4_ck38eeb4e87c369a73`. Shared task key `lk1_be9a22ab…`.
- `data/gates/judge_v1/` (gate and recordings, incl. `record/calls.hedge.C.record-full.jsonl`), `data/early_stop/` (failed gate), `scripts/gate_judge_v1.py`, `scripts/record_judge_v1.py`, `data/adapter_golive/` and `scripts/push_judge_traffic.py` (go-live traffic).
- `data/trap_set/`, `data/noise_set/`, `scripts/eval_trap_set.py`, `scripts/eval_served_calls.py` (ground-truth grading).
- `data/sapling/sapling_v1/`, `data/sapling/sapling_v2/hf_release/`, `data/sapling/compare_published/`, `docs/sapling_results.md`, `docs/sapling_results_v2.md`, `docs/sapling_vs_published.md`, `docs/sapling_gate.md` (Sapling results, the published comparison and the code review). Public: https://huggingface.co/busy-machines/sapling-research and https://huggingface.co/datasets/busy-machines/ml-claims.
- `backend/agent/mubit_client.py` (record modes, request-ID capture, `adapter_headers`), `data/mubit_projects.json`, `backend/benchmark/configs.py` (per-arm record modes), `backend/agent/prompts/` (frozen prompts).
