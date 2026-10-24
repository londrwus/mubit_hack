# Demo script: 3-minute live demo + 2-minute Q&A

> **The demo is now a 3-minute pre-recorded VIDEO, not a live run. See [`video_script.md`](video_script.md)** for the timed script, shot list, capture commands, the "never say" list and a 20-second backup cut. This file still holds the number rules, the 05:00 version decision (§0), the correction beat's facts and wording (§4) and the Q&A plan (§5), and `video_script.md` builds on them.
> **The video's race visual is the R78 replay on the live-compare screen** (Pen frame `eGVso`, 30×). It is not the 60-second live demo: in that demo's one test run the standard agent was slightly ahead on quality and AI bill at 60 s (R89, "demo tool works; not a result"). The 3-minute live race in §1 is a stage fallback only.

The stage plan for Sunday 25 Oct. It replaces the timing table in `demo.md` §3 for the D46–D48 framing. `demo.md` still holds the label rules (§1) and the measured-line rules (§3.1). Q&A answers are in `qa_mubit_constraints.md`, slide copy is in `slides_outline.md`, and what to have open on stage is in `evidence_pack.md`.

**Positioning (D54).** **"Busy Machines: compute intelligence for training sweeps."** Busy Machines is the compute layer under the AI planner of a hyperparameter or RL training sweep: it decides when the planner is worth calling, keeps every machine busy in between with a no-AI sampler, and blocks fake improvements. Mubit is the cheap-AI layer next to it: it makes each call cheaper. The two multiply, and our measured 4× came before any Mubit model served. Customer: ML and RL teams at AI labs. Never call Busy Machines an "AI research agent" or "research assistant" (that is the planner it sits under). Say "machines" or "compute", never "GPUs" alone: R78 ran on cloud CPU workers, R82 on simulated GPU time, and the real-GPU race (R85) failed its quality line.

**Framing (D46, D48).** Two sides with the same model (Opus 5.5), the same machines, budget, clock, seeds and starting code:
- **Standard AI assistant (the Opus-driven sweep):** the plain Opus 5.5 research loop a lab would run (propose v1, plain Opus judge).
- **Busy Machines:** the same Opus 5.5, plus our engine: the **lean planner** (shorter plans, propose v2 at effort low, R62/R63), **backfill** (a no-AI tuner fills idle machines while Opus thinks, fed only checker-approved results, R64/R65), the **planner cadence** (Opus is called at most every 60 s on the game-playing task, R78; every 30 s on tabular, R82, D49, after every 90 s failed, R75, D47), and the **checker** (Opus checks for leaks, a seed-spread rule checks for noise, D40, R37). Every call goes through Mubit.

Lead with **model quality and money** (D48). The measured headline (D54) is **"same model as an Opus-driven sweep, a quarter of the AI bill, machines busy ~98%, no fakes shipped"** on two tasks: game-playing (R78, a tie, machines waiting 1.5%, 0 of 847 trap runs accepted) and tabular (R82, within 0.0013, the standard assistant ahead in every run, machines waiting 1.1%, 0 of 85 accepted, pre-registered, success line met). R82 is always said with the **failed** first pre-registered tabular race beside it (R75) and the fact that R82's setting was chosen after R75; wherever the scorecard appears, the **failed** real-GPU image race (R85: AI bill 0.22×, 5% more errors) appears too. R84's corrections apply (R78's shipped model came from the sampler in 3 of 3 runs). Machines waiting is *why it works*, not the headline. The no-AI baseline (Optuna) is named on stage only as "a standard no-AI tuning tool", on the second results slide or in Q&A (D46).

**Number rules.** Every number on stage is either `live` (read off the screen while it runs, never memorised) or a logged line with its R-id. `[R-pending: …]` marks a result not logged yet (now only the go-live card and B's v2); the LCBench wide race is R75, the RL cadence race is R78, and the LCBench 30 s follow-up (D49, `preregistration_lcbench_wide_c30.md`) is R82. Fill it from `results_log.md` once it's logged, or drop the line. AI $ is a list-price estimate; tabular GPU $ is virtual; the no-AI tool's numbers from races with a Busy Machines arm carry the R74 caveat. Never fill it from memory, from the terminal, or from a race that isn't logged. Projections are said as projections. Never say "Mubit beats Opus" (no adapter has served a call, R46/R70). Never call Haiku (R50) Mubit's model.

---

## 0. The 05:00 decision (Sun 25 Oct, 05:00 London = 05:00 UTC; clocks go back at 02:00; freeze and submit 06:00, D53)
Check the Mubit console fresh at 05:00 (it signs out after 15 minutes idle) and pick **one** version. Write the choice and the evidence into `results_log.md` (new R-row) before rehearsing.

| Console at 05:00 | Version | Evidence needed (all of it) |
|---|---|---|
| A judge v1 adapter (workspace C `ln_dfe18721…` or B `ln_5d9ea82d…`) is **Passed**, is in silent test or partial rollout, and at least one request shows **Served by: Mubit model** | **PRIMARY** | The card text with score and test-case count (≥20), the bar it passed (the card has shown 95%, B's Settings → Going live 90%, R56; in workspace C the bar is to be lowered to 55%, D52, so a C pass is said as "we lowered the bar to 55%", never "passed Mubit's exam"), and ≥1 `mreq_…` under Served by, graded with `scripts/eval_served_calls.py` (runbook step 6). All logged as an R-row |
| No adapter serves, but **B has a v2** on the judge v1 task (triggered by R71's 40 new examples, "new versions after 25 new examples", R58) | **FALLBACK + v2** | B's Adapters → History shows v1 → v2, with v2's score as text (logged) |
| No adapter serves and no v2 | **FALLBACK** | B's card still shows v1 at 11.4% (R70), or newer logged text |

A passed adapter that **doesn't** show Served by on any request counts as FALLBACK ("passed, not yet answering"). Say "passed its exam" only with the card on screen and only for an adapter that passed its default bar (for workspace C at the lowered bar, D52, say "we lowered the bar to 55% and it cleared that"), and "answering" only with a Served-by request on screen.

**Before 05:00 (both versions):**
- Fill every `[R-pending]` you can from the log. The first LCBench wide race failed its pre-registered line (R75), so the results beat always says so beside the 30 s follow-up that met its line (R82; §3 wording).
- PRIMARY only: start the adapter in **Try models** at least 20 minutes before the slot. It uses one of 3 adapter starts per org per UTC day, and the first start can take up to 20 minutes (office hours §6, runbook).
- Rehearse the correction curl on a scratch copy (§4). It costs $0 and makes no network calls.

---

## 1. Screens (no frontend needed)
| Window | Shows | Frontend replacement (if it ships) |
|---|---|---|
| **T1 terminal: live race** | `bench.py run` progress lines every 10 s: `[ <t>s / 180s] opus: <n> done, idle <x>%, $<y>, best <z> \| opus_bm: …` (format from `scripts/bench.py`; no example numbers on purpose, so none can be mistaken for a prediction) | Split screen: GPU tiles, model-quality bars, cost panel (`demo.md` §1) |
| **T2 terminal: results** | `scripts/headline.py` table for the logged races (D47 race ids in `evidence_pack.md`) | Results panel |
| **T3 terminal: correction** | `curl` POST flag → JSON with `correction_id`, then `GET /api/corrections` showing the SQLite row (`sent_to_mubit_at: null`) | "Flag" button and corrections list |
| **Browser tab set 1: Mubit console** | Workspace B: Tasks → judge v1 → Adapters (card, History); Requests (filtered by session); workspace C: same; project A: Tasks page | Not replaced: the console is the proof |
| **Browser tab 2: Hugging Face** | `huggingface.co/MubitLabs/banyan-sapling` (stock card). Our `sapling-research` is **not published** (R69); show it only if it's published by then | n/a |
| **Slides** | `slides_outline.md` | n/a |

Plain-language rule for the terminal: the CLI says `idle` and `best`. Say "machines waiting" and "model quality" out loud, and never read the raw metric name.

**Live race command** (rehearse Saturday night; LLM only, no Modal: LCBench GPU time is simulated, R38). It uses its own DB, so it never mixes with `bench.db` or a headline:
```powershell
python -m uv run --extra ml python scripts/bench.py check lcbench-wide-bm-15m
python -m uv run --extra ml python scripts/bench.py run lcbench-wide-bm-15m --minutes 3 --arms opus,opus_bm --seed 7 --progress-s 10 --db <scratchpad>/demo_live.db --yes
```
- This is a **demo race, never a result**. The `--minutes` override changes its fingerprint, so `headline.py` won't pool it, and it's never logged as a comparison. It shows the loop working live. The numbers on stage come from the logged races.
- Cost: LLM only. **Estimate by arithmetic, not a result:** on this preset the standard arm spent $2.831 and Busy Machines $0.347 per 15-min race (R75), so roughly $0.6 + $0.1 for 3 minutes (more if the first calls cost more than the average). Check `bench.py run` without `--yes` first; it prints the plan and the $ guard.
- Option (not rehearsed in this doc): the R82 preset `lcbench-wide-bm-c30-15m` (planner every 30 s) matches the tabular headline better. By the same arithmetic from R82 ($2.774 vs $0.698 per 15-min race) it costs roughly $0.55 + $0.14 for 3 minutes. Run `bench.py check` on it first; if used, the bullets below about the 90 s pace change accordingly.
- Benchmark races refuse corrections by design (D41 #4), so the flag beat uses a stored case (§4), not this race.
- **No network / Mubit down:** replay the logged R78 seed 0 race (game-playing, the headline) at 10× instead and say "this is a recording of yesterday's race": `python -m backend.replay 4ca4ed27 --speed 10 --db data/bench/bench.db` (`backend/replay.py`; check in rehearsal that the race is in that DB). R82 seed 0 (`c1f4b1d9`) is the tabular alternative (say the standard assistant was ahead in every run); R75 seed 0 (`613e2537`) is the race that failed, and if shown, say so.
- The live preset is the tabular one with the 90 s pace, so in 3 minutes Busy Machines calls Opus only ~2 times (by arithmetic, not measured): narrate the machines staying busy, not who is ahead.
- Real-GPU alternative: `demo-3m` (CIFAR, 2 L4 per arm, ~$0.16 GPU by arithmetic: 2 arms × 2 L4 × 3 min at $0.80/h; the preset comment says ~$0.20; not measured, inside the ~$6 Modal demo reserve, R73). **Not recommended:** it races `opus` vs `mubit`, and with no adapter serving, the `mubit` arm is just Opus through Mubit, which doesn't match the D46 story.

---

## 2. PRIMARY version (an adapter passed and answers calls)
| Time | Beat | Screen | Say (exact words; fill brackets from the log only) |
|---|---|---|---|
| 0:00–0:20 | Hook | Slide 1 | "Busy Machines is compute intelligence for training sweeps. Labs pay for a sweep twice: once for the machines, once for the AI that plans it. On our short-run game-playing sweep, **90.6%** of a standard Opus 5.5 sweep's spend was its own thinking (R54). We cut that to a quarter, with the same model, on two tasks (R78, R82; the tabular one at the second try, R75). On long real-GPU runs, the model got worse (R85)." |
| 0:20–0:40 | Start the race | T1 (start the command at 0:20) | "Same model, Opus 5.5, on both sides. Same machines, same budget, same clock. Left: the standard assistant. Right: Busy Machines: shorter plans, a simple tuner that keeps the machines busy while Opus thinks, and a checker for fake results. Every call goes through Mubit." |
| 0:40–1:15 | The result | T2 `headline.py` table, then slide 4 | "Two tasks, three races each. Game-playing: **the same model, a tie** (−154.3 vs −154.5 penalty), for **a quarter of the AI bill: 83 cents vs $3.30 a race** (R78). Tabular tuning: **within 0.0013** (0.8884 vs 0.8897), with the standard assistant slightly ahead every time, again **a quarter of the AI bill: 70 cents vs $2.77** (R82). That tabular race was our second try. **The first one, pre-registered, failed:** eight times cheaper on AI, but 7% more errors (R75). We picked the faster pace after seeing that, froze it before the race, and report both." Glance at T1: "Live, the left side is waiting [`live`]% of the time; the right, [`live`]%. That waiting is lost experiments." (Not "where the money goes": the GPU $ lost to waiting is small next to the AI bill, R54.) |
| 1:15–1:30 | Why it's real | Slide 6 | "A better number only counts if it's real. Our checker confirmed **7 of 7** real improvements, where Opus alone confirmed **1**, and let **0 of 32** fakes through. It also accepted 1 of 36 lucky results, where Opus accepted none, and this was offline, on 75 stored cases (R37)." |
| 1:30–2:10 | **Correction → model update** | Console, T3 | §4, PRIMARY wording |
| 2:10–2:40 | What else we measured | Slide 5 (pilots), slide 7 (vs a no-AI tool) | Two lines from §3, each with its worse result |
| 2:40–3:00 | Close | Slide 8 | "Busy Machines: compute intelligence for training sweeps. Mubit makes each AI call cheaper; we decide how many a sweep needs. They multiply. Same model, a quarter of the AI bill. And with Mubit, the checks we record can become training for the next model." Never "a better model for less money": R78 is a tie, R82 is slightly behind in every run, R75 and R85 are worse |

## 3. FALLBACK version (no adapter answers calls)
Same as PRIMARY except the correction beat (§4, FALLBACK wording) and the close.

| Time | Beat | Screen | Say |
|---|---|---|---|
| 0:00–1:30 | as PRIMARY | as PRIMARY | as PRIMARY |
| 1:30–2:10 | **Correction → model update** | Console, T3 | §4, FALLBACK wording (with or without v2) |
| 2:10–2:40 | What else we measured | Slides 5, 7 | Two lines from below |
| 2:40–3:00 | Close | Slide 8 | "Busy Machines: compute intelligence for training sweeps. Mubit makes each AI call cheaper; we decide how many a sweep needs. Same model, a quarter of the AI bill, before any Mubit model served. When Mubit's small models pass, the two multiply." Same rule as PRIMARY: never "a better model for less money" |

**Results lines for 0:40–1:15 and 2:10–2:40** (say one win, then its worse result, in the same breath):
| Line | Source | Say beside it |
|---|---|---|
| Headline: "On the game-playing task, three runs: **the same model** (a tie) **for a quarter of the AI bill**, $0.83 vs $3.30 a race, and $1.18 vs $3.65 in total." | R78 (headline-eligible) | "Not a better model. In one run a burst of 143 checks cost $0.56, so that run's bill was a third of Opus's (0.34×), not a fifth (0.21× in the other two runs). It reached the standard assistant's final quality in 2 of 3 runs ($0.46 vs $3.10)." AI $ at list price |
| Headline, second task: "On tabular tuning, three runs, pre-registered: **within 0.0013** of the standard assistant's model (0.8884 vs 0.8897) **for a quarter of the AI bill**, $0.70 vs $2.77 a race, and $3.89 vs $5.97 in total." | R82 (headline-eligible, success line met 4 of 4) | "Not a better model: the standard assistant was ahead in every run; we had about 1% more errors. It never reached the standard assistant's final quality, and to 0.86 it was slower and dearer ($1.14 / 4.2 min vs $1.08 / 2.9 min). We chose this 30 s pace after the first tabular race failed (next line), then froze it before the race." GPU time simulated; AI $ at list price |
| The second limit, always with the scorecard: "On image recognition, long runs on real L4 GPUs, three runs: Busy Machines' AI bill was 0.22× ($0.233 vs $1.056), but its model had **5% more errors** (0.9424 vs 0.9451), in every run. The quality line failed." | R85 | "On long runs the machines barely wait (the standard sweep's 12.0%, ours 0.0%) and the tuner's runs are long and weak, so fewer AI calls cost quality. To the 0.94 goal it was 16% dearer and 2.1× slower, and its shipped model (0.9395) missed the goal. Our only Busy Machines race on real GPUs." |
| Mechanism: "Its machines waited 1.5% of the time, the standard assistant's 68.8%, so it ran 10× the experiments per dollar. Busy Machines' side ran a fake shortcut 847 times, mostly from its tuner; none got through." | R78 (seed 2 alone: 759 sampler trap runs, 128 claims rejected) | "Most experiments came from the no-AI tuner inside Busy Machines (Opus proposed 120 of 1,587), and the shipped model came from the tuner in all 3 runs (R84). The plain no-AI tool reached the easy goal cheapest, but with a worse model and 53 fakes accepted (R74 caveat)." |
| The fail, always beside R82: "Our first pre-registered tabular race failed: 8 times cheaper on AI ($0.347 vs $2.831), but **7% more errors** (0.8816 vs 0.8892), in every run. We report it anyway." | R75 | Also failed: the standard assistant waited 29.4% (line ≥40%), Busy Machines 2.5% (line ≤2%). Passed: AI bill ≤0.8×, better than the no-AI tool. GPU time simulated. "Its 90-second pace left only 80 AI proposals; the one follow-up, a 30-second pace chosen after this fail and pre-registered, met its line (R82)." |
| "Shorter plans alone: the model was at least as good (0.866 vs 0.849, 3 runs), at 10.5% less AI spend per experiment." | R63 | Dev race, not a headline. Pooled over 9 earlier runs the standard assistant scores 0.854, so "at least as good", not a proven gain |
| "Without a pace limit, on the game-playing task, Busy Machines found a better model (−153.9 vs −156.2 penalty), and reached the goal in 0.5 vs 1.1 minutes, for $0.11 vs $0.20." | R67 (2 runs) | "But with no limit on how often it called Opus, its AI bill **doubled**: $6.50 vs $3.07 per race. That's why we added the pace limit; with it (R78) the bill fell to a quarter and the model became a tie." 2 runs only, not a headline; different races, not pooled |
| "On tabular tuning, Busy Machines reached the goal in 1.2 vs 2.5 minutes, for $0.36 vs $0.74." | R65 (2 runs) | "Its best model was slightly worse (0.859 vs 0.861), within noise, and its AI bill was 1.19× higher." |
| Second slide, vs "a standard no-AI tuning tool": "Even the standard assistant beats it on model quality on 2 of 3 tasks: 28% fewer errors on tabular, 22% on images. It accepted fake improvements as its best 59 times in 9 races; the assistant, 0." | R38/R44, R53, R54 | "The tool runs far more experiments per dollar on every task, and on the game-playing task its best model is slightly better (−155.2 vs −156.9, within noise) and it reaches the goal sooner and cheaper (R48). The tool has no checker by design, and on image recognition the assistant never tried the shortcut, so its checker wasn't tested there (R53)." |
| If the "Busy Machines with Opus" column comes up: "An earlier version of our engine, with an early-stopping rule we've since removed, made the tabular model worse: 9% more errors." | R44, D43 | Always say it when that column is discussed |

---

## 4. The correction → model update beat (rule #3, 40 seconds)
### 4.1 What is true (say nothing beyond this)
- Mubit trains a new version of a task's small model **after 25 new kept examples** (B, Settings → Training, R58). That is the model-update path Mubit gives us.
- Examples are **labelled by Opus's own answers**, and adapters are graded on agreement with Opus, not with the truth (CLAUDE.md, field report F5).
- **Corrections:** we store each one in SQLite against the Mubit request ID, and we rate that request **Bad** in the console. Mubit has no corrections API, and we don't know whether a rating changes training (Q2, Q9). So a correction **has not** changed a model, and we never say it has.
- **B's v1 → v2:** v1 trained on 177 Opus-labelled examples (R39) and scored 11.4% after ~7 h of training (R70). We recorded **40 new** Opus-labelled checks into B to trigger v2 (R71). Opus made no wrong call on those 40 (11 fakes → 11 `invalid`; 29 others → `unsure`), so **v2 is not "learned from corrections"**. It is "trained on 40 more checks".

### 4.2 The case we correct (logged ground truth, real Mubit request IDs)
Opus's one consistent mistake here: it **won't confirm real improvements**. It left 46 of 46 clean claimed bests `unsure` in live races (R25) and confirmed 1 of 7 real gains against 10-seed re-runs (R34). On the held-out test split, the closed-output judge confirmed 0 of 4 (R40). Our checker confirmed 7 of 7 (R37).

The three real gains below are from the noise set's dev split (ground truth: 10 re-runs per side, gain > 2× pooled spread and Welch p < 0.05, R34). Opus answered `unsure` **every time it saw them, in every project**, and the judge v1 calls were recorded with full content, so they appear in the console's Requests and can be rated.

| Case | Task | Opus in the live race (project A, judge v0, recorded, R20) | Opus judge v1 in B (recorded, R39) | Opus judge v1 in C (recorded, R59) | Experiment id (collection DB) |
|---|---|---|---|---|---|
| **ns_v1_007** (use this one) | Game-playing AI | `unsure` · `mreq_01a16b31599b7651a654571b8bc455a7` | `unsure · insufficient_evidence` · `mreq_01a16c07cc1d7c70ae96381f78916256` | `unsure · insufficient_evidence` · `mreq_01a16d547a3770d186869108a305e42e` | `700507e1dfe046178c072aed2f9a3aa0` (`collect_rl.db`) |
| ns_v1_013 | Game-playing AI | `unsure` · `mreq_01a16b3181347a709987f2a83e251dd6` | `unsure · insufficient_evidence` · `mreq_01a16c0666de7752843c5cebf7a34c28` | `unsure · insufficient_evidence` · `mreq_01a16d5483437f528e6cba8c292ac4bf` | `edb29f1526f8411b916fa0d94c6750f3` (`collect_rl.db`) |
| ns_v1_048 | Image recognition | `unsure` · `mreq_01a16b33c0367780810aff88ec19fc1d` | `unsure · insufficient_evidence` · `mreq_01a16c066ed474829ae80db3319c2810` | `unsure · insufficient_evidence` · `mreq_01a16d5495d8735396de43380d04c841` | `38e15409466a4f6995db6ca11e8bbfac` (`collect_cifar.db`) |

Sources: `data/noise_set/noise_set_v1.jsonl` (ground truth), `data/gates/judge_v1/record/calls.{first,rest}.B.record-full.jsonl` and `calls.hedge.C.record-full.jsonl` (B/C answers), `data/collection/collect_{rl,cifar}.db` `llm_calls` (project A). The per-case gain sizes are in R34's evidence file but not in a results-log line, so **don't put a per-case gain on a slide** unless it's logged first. On stage, say "ten re-runs say it's real (R34)".

**Trap claims (confirmations, not corrections).** Opus marks planted fakes `invalid`: 92 of 92 in races (R25), 11 of 11 in the v2 batch (R71, e.g. `mreq_01a16e29a9867f41b205d354d152a8e3`), and test case `trap_v1_020` (`mreq_01a16c309a4571329445dceb3b68140b`, R40). A flag on one of those confirms the checker, and is rated **Good**, not Bad. Use them for "fake improvement caught", never as "a correction".

### 4.3 The path on screen (both versions)
1. **T3, the flag** (rehearsed; a scratch copy of the collection DB, so `bench.db` and the collection DB are never written):
   ```powershell
   Copy-Item data\collection\collect_rl.db <scratchpad>\demo_corrections.db
   $env:BUSY_MACHINES_DB="<scratchpad>\demo_corrections.db"; python -m uv run --extra ml python scripts/dev_server.py
   curl.exe -s -X POST http://127.0.0.1:8000/api/experiments/700507e1dfe046178c072aed2f9a3aa0/flag -H "content-type: application/json" -d '{\"label\":\"genuine\",\"reason\":\"10 re-runs per side: real gain (R34). Opus judge said unsure. mreq_01a16b31599b7651a654571b8bc455a7\"}'
   curl.exe -s http://127.0.0.1:8000/api/corrections
   ```
   Checked on 24 Oct in-process (FastAPI TestClient, scratch copy): **200**, `applied_to_live_race: false`, and the row is stored with `label: genuine`, `source: human`, `sent_to_mubit_at: null`. The collection races are dev races, so they accept corrections; benchmark races refuse them (D41). Rehearse the exact PowerShell quoting once before the demo.
2. **Console, workspace B → Requests →** search `mreq_01a16c07cc1d7c70ae96381f78916256` (session `collect-judgev1-2026-10-24/noise_set/7`). It shows Opus's reply `unsure · insufficient_evidence`. Click **Rate this reply: Bad**. (Optionally, also rate the C copy `mreq_01a16d547a37…`.)
3. **Console, B → Tasks → judge v1 → Adapters:** the card, then History.

### 4.4 PRIMARY wording (adapter answering)
| Time | Screen | Say |
|---|---|---|
| 1:30 | Console: Requests → filter **Served by: Mubit model** → one request (ID from runbook step 6, logged) | "This check wasn't answered by Opus. It was Mubit's own small model, trained on the assistant's past checks. Mubit switched to it after it passed its exam: it agrees with Opus on [score, test-case count: R-pending, the 05:00 card]." If it is workspace C at the lowered bar (D52), replace "after it passed its exam" with "after we lowered the bar to 55% and it cleared that". |
| 1:40 | Try models (adapter pre-started): `trap_v1_020` | "Here it catches a fake improvement: tested on the levels it trained on." Only if its answer matches; otherwise pick another pre-checked case (runbook). Never edit the expected answer |
| 1:48 | T3 flag, then Requests → `mreq_01a16c07cc1d…` → Bad | "But it learned from Opus, so it learned Opus's blind spot too: Opus called this real improvement 'unsure'. Ten re-runs say it's real. We flag it: the correction is stored against that exact Mubit request, and we rate it Bad in Mubit." |
| 2:00 | Adapters → History. If v2: v1 → v2 | **With v2:** "And this is how the model updates: after 25 new examples Mubit trains a new version. Here's v2, trained on 40 more checks: [v2 score and test-case count, R-pending]." **Without v2:** "After 25 new examples Mubit trains the next version; we've recorded 40 more (R71)." |
| 2:08 | Caption: **"Every check we record can become training for the next model."** | Say "Mubit" here. Not "every check": race calls are record off and never become examples (R72) |

### 4.5 FALLBACK wording (no adapter answering)
| Time | Screen | Say |
|---|---|---|
| 1:30 | Console B → Requests → `mreq_01a16c07cc1d…` (Opus: `unsure`) | "Every check the assistant makes goes through Mubit. Here's one: Opus looked at a real improvement and said 'unsure'. Ten re-runs say it's real (R34). That's Opus's one consistent mistake: of 7 improvements that ten re-runs showed were real, it confirmed 1 (R34)." |
| 1:42 | T3 flag → row; Requests → **Rate: Bad** | "We flag it. The correction is stored against that exact Mubit request, and we rate it Bad in Mubit." |
| 1:50 | B → judge v1 → Adapters card | "Mubit has already trained a small model on this step, from 177 of Opus's past checks (R39). Today it matches Opus on 11.4% of its 35 test cases, and the bar is 95% (R70). So Mubit keeps Opus answering, which is the right default for a lab." |
| 2:00 | History (v1 → v2 if it exists) | **With v2:** "After 25 new examples Mubit trains a new version. Here's v2, trained on 40 more checks: [v2 score and test-case count, R-pending]." **Without v2:** "After 25 new examples Mubit trains a new version. We've recorded 40 more (R71); v2 hasn't appeared yet." Don't say "queued": nothing logged shows it |
| 2:08 | Caption: **"Every check we record can become training for the next model."** | Say "Mubit" here |

**Allowed:** "stored against that exact Mubit request" · "rated Bad in Mubit" · "trained on 40 more checks" (only with v2 on screen) · "Mubit keeps Opus answering until its model passes" · "every check we record becomes training for the next model".
**Never (either version, unless the log shows it):** "it learned from our correction" · "Mubit is retraining it now" · "v2 fixed this case" · "Mubit's model beats Opus" · "Mubit is answering" (FALLBACK) · "the correction changed the model". If the v2 card shows v2 getting ns_v1_007-type cases right, that still came from Opus labels, not from our Bad rating, unless Mubit confirms otherwise.

---

## 5. Two-minute Q&A plan
Answer in 15–20 seconds, with one number and its R-id, then stop. Full answers are in `qa_mubit_constraints.md`.

| # | Likely question | Short answer | Proof on screen |
|---|---|---|---|
| 1 | "Is Mubit's model answering?" | PRIMARY: "Yes, for the checking step: Served by, here." FALLBACK: "Not yet. Our best checking model matches Opus on 58.3% of 24 test cases (workspace C, R79; the same data in B 11.4% of 35, R70; the earlier free-text one 42.9%, R30), and the bar is 95% (90% in B's go-live settings, R56). Mubit keeps Opus answering until it passes." If C's bar was lowered (D52), say "we lowered the bar to 55% in one workspace" | Console card |
| 1b | "Isn't that what Mubit does?" | "Different layer. Mubit makes each AI call cheaper. We decide how many calls a sweep needs, and keep the machines busy in between. They multiply. Our 4× is our layer alone, on plain Opus through Mubit, before any Mubit model served (R78, R82; R70, R79)." | Slide 3 |
| 2 | "So where do your savings come from?" | "From our compute layer on the same Opus: Opus plans at most once a minute on game-playing and every 30 s on tabular, with shorter plans, and a no-AI tuner fills the machines in between, behind a checker. Same in-race GPU money on both sides (~$0.35, total minus AI $); the AI bill is the difference: $0.83 vs $3.30 on the game-playing task, list-price estimate, with a tied model (R78), and $0.70 vs $2.77 on tabular, within 0.0013 but slightly behind in every run (R82; our first tabular race failed, R75). Mubit's models come on top once they pass. We never claim a Mubit-vs-Opus win." | Slide 4 |
| 2b | "Is it just Optuna doing the work?" | "Partly, and we measured how much. In the game-playing race the tuner ran most experiments and found the shipped model in all 3 runs (R84); on tabular, the AI found it in 3 of 3 in both races (R75, R82), and on image recognition all of Busy Machines' bests came from the AI (R85). Busy Machines beat the plain no-AI tool on quality in all four races (R78, R75, R82, R85), and the tool shipped a fake in 3 of 3 game-playing runs, 1 of 3 in each tabular race and 3 of 3 image runs (R84, R85; the tool's side carries the R74 caveat). In the game-playing race none of Busy Machines' 847 fake-shortcut runs got past the checker, though one run's 143 checks cost $0.56. We haven't run a 'tuner plus checker, no AI' arm, so how much Opus steers the tuner isn't measured." Full answer: `pitch.md` §6 | Slide 4 notes |
| 2c | "Why did the tabular race fail?" | "Its 90-second pace left only 80 AI proposals, and on that task the AI's ideas were what found the best model (3 of 3 runs). The standard assistant also waited less than we projected: 29.4%, not the 45.8% our offline gate projected (R75, R66), so there was less idle time to fill. So we ran one follow-up, a 30-second pace, chosen after the fail and pre-registered: 240 AI proposals, within 0.0013 for a quarter of the AI bill, though the standard assistant was still ahead in every run (R82)." | Slide 4 |
| 2d | "Did you pick the setting after seeing results?" | "For the tabular follow-up, yes. We chose the 30-second pace after the first pre-registered race failed (R75), froze its setup and success line before running it, said it was the last pace we'd try (D49), and we show R75 next to it every time. It met the line (R82), on 3-run means, with run 3 at the margin. The game-playing pace was set by an offline gate before its race (R78)." | `preregistration_lcbench_wide_c30.md` |
| 3 | "Is the race fair?" | "Same model, machines, slots, clock, $ cap, seeds, starting code and judge prompt. What differs is our engine, and it's disclosed in each race's fairness manifest (D46). We pre-registered both tabular races' success lines before they ran; the first failed (R75), the follow-up passed (R82)." | `preregistration_lcbench_wide.md`, `preregistration_lcbench_wide_c30.md` |
| 4 | "Why not just run Optuna?" | "It's in every race. It's cheap and fast, but it has no checker: it accepted fakes as its best 59 times across 9 races, and it shipped a fake in 6 of 9 (R54)." | Slide 7 |
| 5 | "Why did training take 7 hours / fail?" | `qa_mubit_constraints.md` §1 | Console History |
| 6 | "Are the GPUs real?" | "Our headline results didn't run on real GPUs, and we say so: game-playing ran on real Modal CPU workers (R78), and tabular tuning uses a published benchmark's recorded training times, so its GPU time is simulated (R82). Our one race on real L4 GPUs, image recognition with long runs, failed its quality line: 4.5× cheaper on AI, 5% more errors (R85). The AI calls are always real, through Mubit. That's why we say 'machines', not 'GPUs'." | Race manifest |
| 7 | "Did anything break?" | "The no-limit version doubled the AI bill (R67), an early-stopping rule wrecked an arm (R41), and the earlier engine made a model worse (R44). All logged. The stopping rule is gone, and the pace limit is our answer to the bill: on the game-playing task it cut the bill to a quarter, with a tied model (R78). Our first pre-registered tabular race failed (R75), and we logged it beside the follow-up that passed (R82). And the real-GPU image race failed its quality line (R85)." | `results_log.md` |

---

## 6. Rehearsal checklist
- [ ] 5 timed rehearsals. Cut anything that runs past 3:00 (the correction beat is the last thing to cut).
- [ ] `bench.py run … --minutes 3` rehearsed once on the demo DB (LLM $ logged as a demo cost, not a result).
- [ ] Replay fallback tested: `python -m backend.replay 4ca4ed27 --speed 10 --db data/bench/bench.db` (R78 seed 0).
- [ ] Correction curl rehearsed on a fresh scratch copy.
- [ ] Console logged in to A, B and C right before the slot. Tabs open per `evidence_pack.md`.
- [ ] Every `[R-pending]` either filled from the log or deleted from the slides.
- [ ] PRIMARY: Try models adapter started ≥20 minutes before; the 3 cases checked against the expected answers.
