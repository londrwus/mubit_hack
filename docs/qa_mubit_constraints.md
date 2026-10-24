# Q&A: Mubit's constraints, and why our numbers still hold

Short answers for judges, and the Mubit team in particular. The tone is constructive: what we measured, what we did about it, what would help every team. Never say "Mubit doesn't work". Every number carries its R-id from `results_log.md`; field-report items are `F#` in `mubit_field_report.md`.

**Spoken short form (15 s):** "None of Mubit's small models answers our calls yet. The latest checking model matches Opus on 11.4% of its exam (the earlier one 42.9%), and the bar is 95%. So Mubit keeps Opus answering, as designed. Our result comes from our engine running on Opus, through Mubit: the same model for a quarter of the AI bill on the game-playing task (R78), and a pre-registered tabular race that failed (R75). The day a model passes, it gets faster and cheaper on top."

---

## 1. Why no adapter serves
| Fact | Evidence |
|---|---|
| **The exam bar is high.** Real tasks show a **95%** match target on the adapter cards; the toy task showed 85%. Project B's Settings → Going live says the quality bar is **90%**. A move also needs a 0.50 floor and ≥20 test cases | R7, R28, R30, R46, R56 |
| **Scores so far, all below target:** planning 14.3% (the card said "Prompt only", which may mean the base model plus prompt, not a trained adapter; unconfirmed), checking v0 42.9% (6 of 14; the base model with the prompt 1 of 14), toy 70% vs 85% | R7, R28, R30 |
| **The closed-output checker scored lower, not higher:** **11.4% on 35 test cases after ~7 h in training** (project B), although Opus agrees with itself 40 of 40 on that output. We suspect a format or scoring problem, not learning; not verified | R39, R70 |
| **Training was slow or stalled:** B's job showed "Training" for ~7 h before v1 appeared (at 11.4%); workspace C's (same data, same task key) still showed "Training v1" 2 h after it started, with no adapter. The toy task trained in ~5 min on Friday | R47, R55, R68, R70, R6 |
| **Diagnose training failed**, with no reason given | R46 |
| **No routes:** no task has a route, so nothing has gone live, silent test included | R28–R30, R46, R70 |
| **No way to pin an adapter yet.** Mubit support (Discord, 24 Oct 11:51): the auto-router decides; a header to name the adapter per request is coming "later". It had not shipped at the time of writing | `mubit_office_hours.md` (Discord answer) |
| **The planning step can't pass a match exam by nature:** many different experiment plans are equally good | R28, F11 |
| **Review and data limits shaped the first adapters:** ~200 reviews per task per day (our reading of the queue), so the first planning adapter trained on 29 of 499 calls | R29, F8 |
| Early stopping never got an adapter: Opus agrees with itself only 92.5–95% on it, below the 97% line we set, so we made it a rule and then switched it off (no rule passed our offline gate) | R31, R43, R57, D39, D43 |

## 2. What we did about it
| Action | Evidence |
|---|---|
| **Changed our outputs to suit the exam:** the checker's free-text reason became a verdict plus a reason code from a fixed list. Opus self-agreement 68/68, Haiku vs Opus 65/68 zero-shot | D37, R32 |
| **A gate before recording any new task:** Opus must agree with itself ≥97%, ≥150 distinct inputs, no class >70%, no free text. Early stopping failed it, so we didn't spend a training on it | D39, R31 |
| **Fresh workspaces for fresh queues:** project **B** (judge v1, 177/177 recorded) and workspace **C** (identical hedge recording when B stalled; grouped in ~7 min vs ~1 h on B) | D38, R39, R59, R61 |
| **Go-live settings tuned on B:** silent test 20% of calls (up to 100 a day), partial rollout 50% (at least 100 calls); the quality bar left at 90% | R56 |
| **A go-live kit, ready and $0 until used:** `scripts/push_judge_traffic.py` (traffic for silent test and rollout, capped, never touches project A), `scripts/eval_served_calls.py` (grades the calls the console says the adapter answered, against ground truth **and** against Opus), 3 held-out Try-models cases with Opus's expected answers, and a runbook | `adapter_golive_runbook.md`, `data/adapter_golive/` |
| **v2 triggered:** 40 new Opus-labelled checks recorded into B, to reach "new versions after 25 new examples" | R58, R71 |
| **Adapter header ready:** `data/mubit_projects.json` takes an `adapter_headers` block. The client sends it only on the Mubit arm, and refuses it on any other arm | `mubit_office_hours.md`, `backend/agent/mubit_client.py` |
| **The product doesn't wait for Mubit:** the engine (lean planner, backfill, cadence, checker) runs on plain Opus through Mubit today, and each step switches to Mubit's model only when it passes (D36, D46) | D46 |
| **A field report for the Mubit team**, F1–F15 | `mubit_field_report.md` |

## 3. What Mubit would need to change (field report, one line each)
| # | Ask |
|---|---|
| F1 | Enforce `output_config.format` when an adapter serves; add "malformed outputs: needs 0" to the exam (3 of 7 toy replies were malformed, R9) |
| F2 | A `served-by` response header and a lookup by request ID (the console is the only place that says who answered) |
| F3 | Separate "what is stored" from "who answers" (`record: off` also drops the task fingerprint) |
| F4 | Group calls by an explicit task key or schema family; document exactly what the fingerprint covers (R17, R18) |
| F5 | Customer ground-truth sets scored next to provider agreement, so an adapter can beat its teacher's blind spots (Opus confirms 1 of 7 real gains, R34) |
| F6 | Docs from adapter to live traffic (routes, silent test, rollout) and for feedback; fix the 404 outcome links |
| F7 | Per-domain calibration for Sapling (stock answers `unsure` on every research claim we tried, R33) |
| F8 | Document the daily review limit, show the quota, sample the queue for coverage (R29) |
| F9 | Show the exam rules (target, floor, ≥20 cases) before training (R28, R30) |
| F10 | Field-level grading; show the provider's self-agreement as the ceiling (R30–R32) |
| F11 | Outcome or validity grading for generative steps like planning (R28) |
| F12 | `Retry-After` on 503s, and a provider pass-through when the engine is down (80 × 503 `engine_unavailable`, R39) |
| F13 | The hosted Sapling endpoint should return its revision and calibration hash (hosted logits differ by up to 0.87 from ours, though raw labels agree 30/30, R42) |
| F14 | Give a reason for a failed training, and a retry button (R46) |
| F15 | Show the adapter's answers next to the provider's on test cases, plus a format-validity rate (11.4% after ~7 h, R70) |
| (new) | A corrections API, or confirmation of what a Bad rating does (Q2, Q9); ship the adapter-pin header (Discord, 24 Oct) |

## 4. How our numbers stay valid
| Question | Answer | Evidence |
|---|---|---|
| "Did every LLM call go through Mubit?" | Yes, a hard rule. Every race call carries a Mubit request ID (`mreq_…`) stored in SQLite and the JSONL export; the app never reads a provider key. Claude Code wrote code, never answered a race call (D15) | CLAUDE.md, D15 |
| "Could Mubit's router have answered some of the Opus arm's calls?" | Race calls run with `x-middleware-record: off` on project A. The docs say `off` drops the task fingerprint, so (by our reading, not yet confirmed by Mubit) those calls can't match a task or a route. Project A also has no route on any task. New tasks live in B and C, away from the baseline | D38, R46, R72, office hours §2 |
| "So 'Busy Machines' is Mubit?" | No. In every race so far, both sides are Opus 5.5 through Mubit. The difference is our engine, disclosed per arm in the race's fairness manifest. "Mubit + Busy Machines" is a separate arm, raced only once an adapter serves; `headline.py` refuses any Mubit framing without evidence that an adapter served | D41, D44, D46 |
| "Is the comparison fair?" | Equal and fixed: model, machines, slots, wall clock, $ cap, seeds, starting code, judge prompt. The standard arm is the plain v1 loop. The LCBench wide race was pre-registered (setup, seeds, success line) before any race, every seed is reported, and it failed (R75). The game-playing race's 60 s planner interval was chosen by an offline gate **before** that race (projected 0.22× LLM, 3.6% waiting; measured 0.25×, 1.5%), all 3 seeds reported (R78). The 30 s tabular follow-up was chosen **after** seeing R75 and is disclosed as such (D49) | D46, D47, D49, R75, R78, `preregistration_lcbench_wide.md`, `preregistration_lcbench_wide_c30.md` |
| "Can you pool the game-playing seeds?" | Yes, after one rule fix: one diagnose call (made only after a failed run) split seed 2 off by fingerprint; D50 leaves that step out of the pooling key (it depends on whether a run failed, not on setup), and every earlier fingerprint id stayed unchanged | D50, R78 |
| "Are the no-AI tool's numbers in these races clean?" | Not fully. In races with a Busy Machines arm, Optuna shared one Python process with Busy Machines' sampler, which slowed its asks (~1.6×), so its waiting and experiment counts there are not isolated (its waiting may be inflated). Opus vs Busy Machines is essentially unaffected (~5 ms loop lag vs 4–13 s LLM calls). Fix planned after the races | R74 |
| "Can you pool races?" | Only with the same fairness fingerprint (arm set, features, prompt versions, record modes, settings). A race with a mid-race correction is blocked | D41 |
| "What does the LLM $ include?" | Token estimates at list price from the provider's usage fields, billed to our own Anthropic key. They exclude any Mubit fee (the org balance showed $50 with no visible run-fee deduction; race calls are record off, which keeps them outside learning tasks) | R72, D14 |
| "Is the GPU time real?" | Game-playing runs on real Modal CPU containers and image recognition on real L4s. Tabular tuning (LCBench) is a surrogate with simulated GPU time, and the wide race runs it at 0.1× (disclosed) | R38, R48, R53, D47 |
| "Did you cherry-pick?" | Every measurement is logged, worse ones included: R75 (our pre-registered tabular race failed: 7% more errors), R44 (our earlier engine made a model worse), R41 (a broken stopping rule), R67 (BM's AI bill doubled), R57 (Opus lost to a free rule on early stopping), R70 (11.4%), R77 (our Sapling v3 failed its gate) | `results_log.md` |
| "Is the Opus baseline handicapped?" | It gets the same caching, structured outputs and batching. The lean prompt is ours and is disclosed as a BM feature; its separate effect is quoted from its own test | R62, R63, D46 |

## 5. Hard questions, short answers
| Question | Answer |
|---|---|
| "11.4% is worse than your free-text 42.9%. Why?" | "We don't know yet. A 3-verdict answer at 11.4% is near or below chance, which points to format or scoring, not learning, but that is a guess (R70). We haven't yet checked the adapter's own answers on its test cases; showing them beside the provider's would settle it (F15)." |
| "Why not lower the quality bar so it serves?" | "Then a model that disagrees with Opus more often would make the call that decides what a lab believes. We left the bar at 90% (R56)." |
| "Then what is Mubit adding today?" | "Every call through one gateway, with a request ID we store next to each experiment; the calls we record (race calls are record off, R72) become tasks and examples; the training pipeline; and a router that by Mubit's stated design switches only when its model is better (a launch claim we haven't measured), which is why our result doesn't depend on a model passing. When one does, it should serve without a code change on our side." |
| "Will the planning step ever be served?" | "Not with a match exam: many plans are equally good (R28). It's the slowest and most frequent step (13.1 s for 8 proposals, R48), so outcome grading would be the biggest win (F11)." |
| "Your correction didn't change a model, did it?" | "No. It is stored against the request ID and rated Bad in Mubit. There's no corrections API, and we don't know whether a rating affects training (Q2, Q9). What does change the model is new examples: 25 trigger a new version (R58), and we sent 40 (R71)." |
| "Is Haiku your small model?" | "No. Haiku was a speed test of a smaller provider model in the same loop (R50). It's not Mubit's model and not Busy Machines." |
| "Why did the tabular race fail?" | "We fixed the line before the race and missed it: 8.2× lower AI bill, but 7% more errors in all 3 runs (R75). Measured: the 90-second pace left 80 AI proposals per race, and there the AI planner found the shipped model in 3 of 3 runs, so it had too few ideas; and the standard assistant waited only 29.4%, not the 45.8% our offline gate projected (R66, flagged fragile; the success line was ≥40%), so there was less idle time to fill. Not measured: that a faster pace closes the gap. That's the one follow-up, chosen after the fail: [R-pending: LCBench 30 s race, D49]." GPU $ there is virtual |
| "Is it just Optuna doing the work?" | "Partly, and here's what we measured. Game-playing (R78): the tuner inside Busy Machines ran ~1,466 of 1,587 experiments and found the shipped model in 2 of 3 runs. Tabular (R75): the AI found it in 3 of 3. In both, Busy Machines beat the plain no-AI tool on quality (−154.3 vs −157.3; 0.8816 vs 0.8493, R74 caveat), and no fake got past its checker, where the unchecked tool accepted 53 as its best (R78). What we haven't measured: a 'tuner plus checker, no AI' arm, so 'the AI picks the regions, the tuner fills the machines, the checker blocks the fakes' is the design, not a measured split. And in an earlier race without our arm the plain tool alone reached −155.2 on the same game-playing setup (R48, not pooled), close to ours." |
| "Where does the 4× come from? Mubit?" | "No, our engine on plain Opus, on the game-playing task only: planning calls at most once a minute, shorter plans, the tuner in between. In-race GPU money was the same on both sides (~$0.35, by arithmetic: total minus AI $); the AI bill was $0.83 vs $3.30 (list-price estimate), with a tied model, not a better one (R78). On tabular our pre-registered race was 8.2× cheaper on AI but failed on quality, 7% more errors (R75). The checker's calls aren't paced yet: one run's burst of 143 checks cost $0.56. No Mubit model has answered a call (R70)." |
