# Demo: "The Night Shift Race"

> **Stage plan for Sunday (D46–D48): [`demo_script.md`](demo_script.md)** gives the timed PRIMARY/FALLBACK script and the correction beat, with logged cases and Mubit request IDs. It also points to `slides_outline.md`, `evidence_pack.md` and `qa_mubit_constraints.md`. §0's engine column and the §3 timing below predate D46: median stopping is off (D43), and the Busy Machines side is now Opus + lean planner + backfill + cadence + checker. Where they differ, `demo_script.md` wins. The label rules (§1) and the measured-line rules (§3.1) still apply. **Logged headline (~18:00 Sat):** game-playing, 3 runs, the same model for a quarter of the AI bill (R78), always said beside the failed pre-registered tabular race (R75); the D49 30 s follow-up is still running.

Judging criterion: "Clear, live, and under the time limit." The room should **feel** the result, not just read it from a chart. The event is called Night Shift, so lean in: two AI assistants work the night shift on identical GPUs. Everything on screen follows the plain-language rules (D19, `pitch.md`).

## 0. The two sides (D36) and the Mubit constraints we designed around
| | Standard AI assistant | Busy Machines (powered by Mubit) |
|---|---|---|
| LLM | Opus 5.5 for every step | Every call through Mubit: its small model where it passed the exam, Opus where it didn't |
| Engine | same scheduler, slots, budget, seeds, settings (including the research choices, D29) | + median-stopping rule and the Busy Machines judge (the language model checks for leaks, a calibrated seed-spread rule checks for noise, D40). Per-claim re-runs are not built (the calibrated spread replaces them), and Sapling verifies offline only, not in the live race (D28) |
| What the benchmark adds | | a "Busy Machines with Opus" column, so judges see what our engine vs Mubit's models contributed |

**Mubit constraints (measured) and what we did about each:**
| Constraint | Evidence | Our design |
|---|---|---|
| ~200 example reviews per task per day (our reading of the queue, to confirm) | R29: 299 propose calls over the limit | ≤200 recorded calls per task per day; coverage over volume |
| Exam = ≥95% match with the provider's answers | R28, R30 | classification-style steps (judge, diagnose) are the ones meant to pass; early_stop failed our self-agreement gate (R31) and became an engine rule (D40); creative propose stays on Opus if it doesn't |
| 3 adapter starts per org per day | docs | trainings planned, never casual |
| No per-call "who answered" in the API | R3, docs | console "Served by"; per-arm record modes (D27 bench config) |
| `record: off` drops the task fingerprint | docs | Opus arm off (never routed), Mubit arm metadata |
| No corrections API | docs, office hours | ground-truth checks in our engine + console rating; v2 from corrected examples where Mubit allows |

## 1. The screen (live split screen)
Two sides (D36): **Standard AI assistant** (subtitle: Opus 5.5) on the left, **Busy Machines** (subtitle: powered by Mubit) on the right. Same GPUs, budget, clock, seeds and starting code; only the assistant differs.

**Number rule for the whole demo.** Every value marked `live` below is filled by the race on stage; nobody types it in. Every other number on a slide or in the narration is a measured line from `results_log.md` and carries its R-id in the speaker notes (§3.1). If a number is neither, it doesn't appear.

```
┌──── STANDARD AI ASSISTANT · Opus 5.5 ──────────┬──── BUSY MACHINES · powered by Mubit ──────────┐
│ GPUs ▢▢■▢▢■▢▢…   GPUs waiting [live]%  planning…│ GPUs ■■■■■■■■…   GPUs waiting [live]%          │
│ Experiments finished: [live]                    │ Experiments finished: [live]                    │
│ Model quality  [live bar]  [live sparkline]     │ Model quality  [live bar]  [live sparkline]     │
│ Planning: "[live: latest hypothesis]"           │ Planning: "[live: latest hypothesis]"           │
│ Fake improvements caught: [live]                │ Fake improvements caught: [live]                │
│ Money wasted on waiting: $[live]                │ Money wasted on waiting: $[live]                │
└─────────────────────────────────────────────────┴─────────────────────────────────────────────────┘
          [live] remaining        Money spent: $[live] | $[live]   (same cap for both: $[config])
```
The tile pattern above is only a layout sketch, not a prediction of who waits more. The logged races (R78: 1.5% vs 68.8% waiting on game-playing; R75: 2.5% vs 29.4% on tabular) don't make a live 3-minute demo race a result, so the screen never hints at a winner before that race does.

### Visual elements
1. **GPU tiles** light up while a training run is active and go grey while that slot waits for the assistant to plan. **The single most important visual.** Whichever side flickers grey is decided by the race, not by us.
2. **Live training curves** stream from each run, one thin line per run, with the best run highlighted.
3. **Leaderboard ticker:** model quality per side, updated live, with a flash on a new best. A claimed best that the checker rejects flashes red and is labelled **"Fake improvement caught"**.
4. **Thought stream:** each proposal's short `hypothesis` (≤20 words, D26, already in the JSON, so it costs nothing).
5. **Counters and cost panel** (every cell `live`):
```
                          Standard assistant    Busy Machines
GPU money (paid)          $[live]               $[live]
  wasted on waiting       $[live] ([live]%)     $[live] ([live]%)
Assistant (LLM) money     $[live]               $[live]
Time to reach the goal    [live] or "not yet"   [live] or "not yet"
Cost to reach the goal    [live] or "—"         [live] or "—"
```
   "Not yet" stays until the goal is actually reached; a projected time or cost is never shown (R27 projections are not results).
6. **Who answered each step** is not on the dashboard: Mubit's API response doesn't say whether Opus or Mubit's model answered (`mubit_field_report.md` F2). It's shown in the Mubit console's **Served by** field (§2).

### Dashboard labels
| Technical name | What the screen says |
|---|---|
| GPU idle % / GPU busy | **GPUs waiting** / **GPUs working** |
| $-to-target / time-to-target | **Cost to reach the goal** / **Time to reach the goal** |
| best score at fixed budget | **Model quality** (a 0–100 bar, higher = better) |
| experiments per GPU-hour | **Experiments finished** |
| idle GPU $ | **Money wasted on waiting** |
| val_bpb / test_acc / mean return / balanced accuracy | **Model quality** (raw metric only on hover) |
| contamination trap, judge `invalid · leak` | **Fake improvement caught** (hover: "tested on data it had already seen") |
| `propose_experiments` / `judge_improvement` | **Planning** / **Checking the result** |
| propose latency | **Planning time** |
| adapter | **Mubit's own small model** (console only; never a dashboard label) |
| exam passed + route enabled | **Ready to take over** (console only) |
| adapter v1 → v2 | **"Learned from your correction"** (only if v2 was trained on our corrections, §2) |
| Opus 5.5 arm | **Standard AI assistant** (subtitle: Opus 5.5) |
| Mubit + BM arm | **Busy Machines** (subtitle: powered by Mubit) |
| LM / RL / CIFAR / LCBench targets | **Language model · Game-playing AI · Image recognition · Tabular model tuning** |

Model quality is normalised per target: 0 = the starting model, 100 = the best known result. The raw number is on hover and in the export, so the claim stays verifiable.

## 2. The correction → model update moment (mandatory, 40 seconds)
Judging criterion #1 is "a correction turning into a model update". Mubit's pipeline sets what we can honestly show: an adapter needs ≥95% match with Opus's answers, a 0.50 floor and ≥20 test cases before it can serve (R30), it serves only once a route points to it, and on the toy task calls became a trained adapter in ~20–25 minutes end to end, with training itself ~5 minutes after review (R6). State as of the last log update: judge v0 adapter **42.9% vs a 95% target** (R30), closed-output judge v1 adapter **11.4% on 35 test cases vs 95%** (R70), planning adapter **14.3%** (R28), **no routes** on any task (R28–R30, R70); B's go-live quality bar reads 90% (R56). So there are two versions, picked by a check, never by hope.

**Decision at Sun 08:00 London time** (clocks went back at 02:00), in the Mubit console:
| What the console shows at 08:00 | Version on stage |
|---|---|
| A `judge_improvement` adapter with the exam **Passed**, an **enabled route**, and at least one race call whose **Served by** is Mubit's model | **Primary** |
| Anything less (today's state) | **Fallback** |
| In either version: a **v2** trained from our corrections exists | add the v1 → v2 step; otherwise skip it and never mention v2 as done |

**Shared by both versions**
- **The fake improvement.** The planning prompt is neutral and the targets offer a setting that tests on data the model has already seen (D25), so searchers do fall into it (R38: 33 trap runs on the Opus side and 62 on Optuna's across 3 benchmark races; R48: Opus 15 and Optuna 4,661 across 3 game-playing races; R53: Opus 0 and Optuna 118 across 3 image-recognition races). Busy Machines' backfill sampler ran 15 trap runs in the 2-run LCBench pilot, none accepted as best (R65). On image recognition the Opus side never tried the shortcut (R53), so prefer the game-playing target for this moment. If Busy Machines hasn't hit one by 1:25, **replay a logged trap claim** from this race's log, or the latest benchmark race's, and say "this one is from the race log".
- **The correction.** The presenter clicks **"Flag: fake improvement"**. That stores a correction row in SQLite with the Mubit request ID, and we rate the same request **Bad** in the console. These are the only correction paths we have (no corrections API yet, Q2/Q9). Never say a flag changes a model on stage.

### Primary version (a passed, routed judge adapter exists)
| Time | Do | Say |
|---|---|---|
| 1:30 | Busy Machines' ticker flashes a suspicious jump | "It thinks it found a huge improvement." |
| 1:35 | The checker rejects it: **"Fake improvement caught"** | "But it's fake: the model was tested on data it had already seen. Busy Machines caught it." |
| 1:40 | Console → Requests → that call → **Served by: Mubit's model** | "That check wasn't Opus. It was Mubit's own small model, trained on this assistant's past checks. Mubit switched to it because it passed the exam: it agrees with Opus [score, R-id logged at 08:00] of the time." |
| 1:50 | Click **Flag** → correction saved (dashboard row with request ID) → console rating Bad | "We flag it: the correction is stored against that exact Mubit request, and we rate it Bad in Mubit." (Not "it becomes a training example": we don't know whether a rating changes training, Q2/Q9.) |
| 1:55 | **If v2 exists:** Adapters → History v1 → v2, then **Try models** on a case v1 got wrong and v2 gets right (chosen from v2's logged test results, R-id) | Superseded by `demo_script.md` §4.4: v2 would be trained on 40 more Opus-labelled checks (R71), not on our flags, so never say "Mubit trained a new version from our flags". **If no v2:** Adapters → History v1 only; don't say "queued" |
| 2:05 | Caption: **"Every check we record becomes training for the next model."** | Say "Mubit" out loud here. (Not "every correction becomes a model update": not shown, Q2/Q9.) |

### Fallback version (no passed, routed adapter at 08:00)
| Time | Do | Say |
|---|---|---|
| 1:30 | Busy Machines' ticker flashes a suspicious jump | "It thinks it found a huge improvement." |
| 1:35 | The Busy Machines checker rejects it (D40: the language model checks for leaks, a seed-spread rule checks for noise): **"Fake improvement caught"** | "But it's fake: the model was tested on data it had already seen. Busy Machines' checker caught it." |
| 1:42 | Click **Flag** → correction saved (dashboard row with the Mubit request ID) → same request in the console, rated Bad | "We flag it. The correction is stored against that exact request in Mubit." |
| 1:50 | Console → project B → judge v1 task → Adapters: the closed-output judge's card (R70: 11.4% on 35 test cases, target 95%), or newer logged text at 08:00 | "Mubit has already trained a small model on this checking step, from 177 of the assistant's past checks (R39). Today it matches Opus on 11.4% of its 35 test cases (R70). Mubit only switches to it once it passes, so Opus still answers." (Not "this correction goes into the next training": unknown, Q2/Q9.) |
| 2:05 | Caption: **"Every check we record becomes training for the next model."** | Say "Mubit" out loud here. |

Fallback trade-off: the caption is weaker than the rubric's own words, because in this version no correction has updated a served model yet. Saying the stronger line would be false.

**Don't say (either version):** "Mubit is retraining it now" (on the toy task, calls to a trained adapter took ~20–25 minutes, R6), "it learned from your correction" (unless v2 was trained on our corrections), "Mubit's model beats Opus" (adapters are graded against Opus, D3), "Mubit is answering" (fallback version: it isn't).

### Show the Mubit console three times
1. **Tasks page:** the research steps with example counts. *"Our assistant's work, as Mubit tasks."*
2. **Requests filtered by session:** one researcher's full run, with **Served by** visible. *"Its full log, every call."*
3. **Adapters → History / test results:** the correction moment above.

Log in fresh right before the demo (it signs out after 15 minutes idle).

## 3. The 3-minute script
Both correction versions fit the same 40-second slot, so the rest of the script doesn't change.

| Time | Beat | Screen |
|---|---|---|
| 0:00–0:20 | Hook: "AI labs are GPU-bound. A research assistant plans the next experiments while the GPUs wait. With 30 machines and 15-second runs, a standard Opus 5.5 assistant kept them waiting 69% of the time: each plan for eight experiments takes about 13 seconds (R48)." If asked: with fewer machines and longer runs the waiting is ~11% (R53) | Standard assistant side alone, planning |
| 0:20–0:40 | "Same GPUs, same budget, same clock. Left: the standard AI assistant, Opus 5.5. Right: Busy Machines, with every call going through Mubit." Start the race. One line (D36): "Our benchmark also runs Busy Machines on plain Opus, so you can see what our engine adds and what Mubit's models add." | Split screen goes live |
| 0:40–1:30 | Race runs. Narrate only `live` values: GPUs waiting, experiments finished, new bests | Tiles, curves, ticker |
| 1:30–2:10 | Correction moment: **primary or fallback** (§2) | Dashboard, then console |
| 2:10–2:40 | Results: measured only (§3.1). Say two or three lines; the slide holds the rest, each with its R-id | One clean results slide |
| 2:40–3:00 | "Same model. Same machines. A quarter of the AI bill. Built for AI labs." (D48, R78; not "a better model for less money": R78 is a tie and R75 failed, `demo_script.md` §2) | Logo and one line |

### 3.1 Measured lines (the only numbers allowed off the live screen)
Each line goes on the slide with its R-id. Where the log has a worse result on the same comparison, it goes on the same slide (results-log rule).

| Say (plain words) | R-id | Must go with it |
|---|---|---|
| "Busy Machines vs the standard assistant, game-playing, three runs: **the same model** (a tie, −154.3 vs −154.5) **for a quarter of the AI bill** ($0.83 vs $3.30 per race; $1.18 vs $3.65 total)." | R78 (headline-eligible; `scripts/headline.py`) | Same breath: "Our pre-registered tabular race failed: 8.2× lower AI bill, but 7% more errors (R75)." Also: same quality reached in 2 of 3 runs only; one run's check burst took its bill to 0.34×; AI $ at list price. The live demo race is never this result |
| "Busy Machines' checker confirmed **7 of 7** real improvements; Opus alone confirmed **1 of 7**. Both let through **0 of 32** fakes." | R37 (R34 for Opus alone) | It accepted 1 of 36 noise results vs Opus's 0 of 36. Offline on 75 stored cases, not yet live in a race; quote counts, not percentages |
| "Opus takes about 3 s to plan one experiment and 13 s to plan eight." | R49 (p50 3.15 s at 1), R48 (p50 13.1 s at 8) | A smaller model in the same loop (Haiku, not a Mubit model) was only ~28% faster per call (R50) |
| "With 30 machines and 15-second runs, the standard assistant kept its GPUs **waiting 69% of the time**; the no-AI search tool, under 1%." | R48 (69.0 ± 0.4% vs 0.9 ± 0.1%, 3 runs) | With 3 machines and longer runs the waiting is 10.9% (image recognition, R53), and 4.6–5.0% on tabular tuning (R38, R44). The GPU money lost to waiting is small next to the assistant's own bill ($3.69 per race, of which $3.27–3.40 is thinking, R48). Say it as the problem and the mechanism: Busy Machines cut it to 1.5% vs 68.8% on the same setup, 3 runs, at 0.25× the AI bill, with a tied model (R78); without a pace limit it was 2.0% vs 66.0% at 2.1× the bill (R67, 2 runs) |
| "Even the standard assistant beats a no-AI search tool on model quality in 2 of 3 tasks and ties on the third: **28% fewer errors** on tabular tuning, **22% fewer** on image recognition." | R38 (repeated in R44), R53, R48 | The search tool ran more experiments per dollar on every task (80.9 vs 22.7 on tabular, 81.2 vs 38.7 on images, 4,585 vs 137 on the game). On the game-playing AI its best model is slightly better (−155.2 vs −156.9) and it reached the goal far sooner and cheaper ($0.007 / 0.3 min vs $0.222 / 1.1 min). On tabular, in 1 of 3 tries it reached the goal faster and cheaper; GPU cost there is simulated, and Opus may know that benchmark from training |
| "On image recognition the assistant hit the goal in **3 of 3** tries, in about 2 minutes for about 23 cents; the search tool **never** hit it." | R53 ($0.229 ± 0.077, 2.0 ± 0.6 min) | Its race cost $1.65 vs the search tool's $0.60, and its GPUs waited 10.9% vs 0.0% |
| "The no-AI search tool accepted fake bests on every task: **36** on the game-playing AI, **15** on image recognition, **8** on tabular. Opus accepted **0**." | R48 (36 of 4,661 trap runs vs 0 of 15), R53 (15 of 118 vs 0), R38 (8 of 62 vs 0 of 33) | The search tool has no checker by design. Opus never tried the shortcut on image recognition, so its checker wasn't tested there |
| "The model the search tool handed over did worse on fresh data: **89.6%** vs the assistant's **94.3%** on image recognition, and it missed the goal on the game-playing AI." | R53 (0.8955 ± 0.016 vs 0.9431), R48 (−177.1 ± 8.8 vs −156.9; goal −170) | On the game-playing AI the search tool's best clean model (−155.2) is slightly better than the assistant's; it shipped a worse one because ~97% of its runs used the shortcut |
| "The assistant really improves models: the game-playing AI's penalty is **5.9× smaller**, image recognition makes **22% fewer errors**." | R23 (game), R53 (image) | **The language model did not improve** (R22's gain was seed luck). Say it if the slide shows targets |
| If asked about Busy Machines' own engine: "On plain Opus, our engine add-ons made the tabular model worse: 9% more errors." | R44 (cause unclear, R51) | Must be said whenever the "Busy Machines with Opus" column comes up |

### Narration (plain words)
| Moment | Say | Don't say |
|---|---|---|
| Opening | "Two AI assistants. Same GPUs. Same budget. Same clock. Let's race." | "Two agent arms with N GPU slots each…" |
| Grey tiles | "Grey means the GPU is waiting for the assistant to plan: experiments that aren't running." (Not "you pay for every grey second" as a money claim: idle GPU $ is small next to the AI bill, R54.) | "GPU idle %, utilisation, propose latency" |
| Counters | "This side has finished [`live`] experiments. That side, [`live`]." | "experiments per GPU-hour" |
| Leaderboard | "Higher is better." Then the `live` leader, whoever it is | "val_bpb went from 1.14 to 1.09" |
| Cost | "That side has wasted $[`live`] on waiting. This side, $[`live`]." | "idle GPU $, $-to-target" |
| The mistake | "It thinks it found a huge improvement. But it's fake: the model was tested on data it had already seen." | "eval contamination, train/val overlap" |
| The correction (primary) | "That check was Mubit's own small model… we flag it… stored against that exact Mubit request, rated Bad." | "adapter v1 → v2, corrections as training examples", "it becomes training for the next version" |
| The correction (fallback) | "We flag it… Mubit already trained a small model on this step… it switches only once it passes… Opus still answers." | "the adapter is below target, no route, advisory exam", "this correction goes into the next training" |
| Close | "Same model. Same machines. A quarter of the AI bill." (R78, as above) | "cost per discovery at fixed budget", "spend less on GPUs" |

(`live` numbers are read off the screen, never memorised. If the race fails, use the recorded run (§4) and say so.)

### 3.2 Q&A: what we say about Mubit's limits
Constructive and specific: what we measured, what we changed because of it, and what would help everyone. Never "Mubit doesn't work".

| Question | What we say |
|---|---|
| "Is Mubit's small model actually answering in your demo?" | **Primary:** "Yes, for the checking step. You saw it in the console's Served by field." **Fallback:** "Not yet, by design. Mubit only switches to its own model once it matches the big model on 95% of an exam. Our first checking model went from 1 in 14 untrained to 6 in 14 trained (R30); the closed-output one scored 11.4% on 35 test cases (R70). Until one passes, Mubit keeps sending the call to Opus. Mubit's stated design is to switch only when its model is good enough (a launch claim we haven't measured), which is the right default for a lab." |
| "So where does Busy Machines' advantage come from?" | "That's why the benchmark has a third column, Busy Machines on plain Opus: it splits our engine's share from Mubit's models' share. On game-playing our engine on plain Opus got the same model for a quarter of the AI bill (R78); on tabular our pre-registered race failed, 8.2× cheaper on AI but 7% more errors (R75). An earlier engine version did worse than plain Opus on tabular tuning, 9% more errors (R44), and the 2-run pilots were mixed (R65: faster and cheaper to the goal, slightly lower best model; R67: better model, 2.1× the AI bill). So all of today's measured savings come from our engine; Mubit's models add nothing yet, because none serves. Mubit is how each step is meant to get faster and cheaper as its models pass." |
| "Why so few training examples?" (review limit) | "Mubit reviews about 200 examples per step per day: that's our reading of the queue, not a published number (R29). We sent 499 planning calls and the first model trained on 29 (R28–R29). So we now record at most 200 calls per step per day, picked to cover different situations, and put new steps in a second project with its own queue (D38). Showing that quota in the console would help every team (field report F8)." |
| "95% sounds strict." (the exam) | "It is, and it should be for a step that decides what a lab believes. It also needs a 0.50 floor and at least 20 test cases (R30). It changed how we work: before recording a step we check Opus agrees with itself, because a student can't match its teacher more often than the teacher matches itself (D39). The checking step agreed with itself 68 of 68 times (R32). The early-stopping step only 37 of 40 (R31), so we didn't train it and use a simple rule instead (D40)." |
| "Why did the first checking model score 43%?" (free text) | "Our fault, not Mubit's: the answer had a free-text reason, and the exam matches whole answers, so a small model's wording rarely matched Opus's even when the verdict did (R30). We closed the output to a verdict plus a reason from a fixed list. Opus then agreed with itself 68 of 68 times, and the small Haiku model agreed with Opus 65 of 68 times without training (R32). We recorded 177 examples (R39). That model scored 11.4% on 35 test cases after ~7 h of training (R70), lower than the free-text one; we suspect a format or scoring problem, not verified." (Update with a newer R-id if the 08:00 card changed.) |
| "Why doesn't the planning step use Mubit's model?" | "Planning is creative: many different experiments are equally good, so matching Opus's exact proposal is a hard exam. Its first model scored 14.3% (R28). It stays on Opus for now. It's also the slowest step, about 13 s for eight experiments (R48), so it's the biggest prize once creative steps can be graded on outcomes." |
| "How does Mubit decide who answers? Can you see it?" (routes) | "Per step: a trained model serves only after it passes the exam and a route points to it. None of ours has a route yet (R28–R30). The console shows who answered each call (Served by); the API response doesn't yet, so our dashboard can't show it live. We've suggested a response header for it (field report F2)." |
| "How do corrections reach the model?" | "Every flag is stored against its Mubit request ID, and we rate that request in the console. There's no corrections API yet (Q2/Q9), so a corrected example reaches training only where Mubit allows it. [Primary with v2: name the v2 version and how many corrections it was trained on, from the console.]" |
| "Did anything break?" | "During recording, 80 attempts got a temporary 'engine unavailable' error from the gateway; we retried and kept 177 of 177 valid answers (R39). It's launch week, and the retry logic is part of what a lab needs anyway." |

## 4. Build and run notes
- **The race must be real:** 2 L4s per arm for 3 minutes (about $0.20, an estimate from the preset, not measured), or the CPU RL target with 30 workers per arm (the setting of R48). **Pre-warm the Modal containers** so there's no cold start on stage.
- **Stack:** FastAPI + WebSocket backend streaming `gpu_state`, `metric`, `new_best`… (`architecture.md`). Vite + React + shadcn/ui + Tailwind (+ Recharts), designed first in Pen.dev. Dark theme, large type readable from the back of the room.
- **Log in to the Mubit console fresh** right before the demo (it signs out after 15 minutes idle).
- **Fallback:** record a perfect run on Saturday night and keep it ready if Wi-Fi or Modal fails. Say so openly if it's used.
- **Expo table (10:30–12:30):** loop the race on a monitor, with a QR code to the repo and the results export. People stop to watch races.
- **Rehearse at least 5 times** with a timer. Cut anything that pushes past 3 minutes.
