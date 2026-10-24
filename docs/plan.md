# Build plan: effort, order, cuts

Available time: Fri 19:30 → Sun 10:00 ≈ **39 hours** (clocks go back, +1 h). Realistically ~26–30 working hours per person after sleep and food.
Estimates are **person-hours** for someone fluent in Python/React and new to Modal/Mubit. With Claude Code writing much of the boilerplate (API, schemas, UI components), expect the low end.

## 0. Next session: start here (afternoon brief, Sat 24 Oct ~15:30 BST)

**Since the morning brief:** headline sets done for image recognition (R53: 22% fewer errors than Optuna, Optuna never reached the goal) and game-playing (R48); cross-race fakes (R54: Optuna shipped a fake in 6/9 races, the AI agent 0/9). Busy Machines engine built: lean planner (R62/R63: better model 0.866 vs 0.849), backfill (R64/R65), adapter-header support. RL race without a cadence limit gave a better model but a 2.1x LLM bill (R67), so it was stopped. Mubit: judge v1 adapter in B trained after ~7 h and scored 11.4% (R70); C stuck (R68); v2 batch recorded into B (R71). sapling-research v2 packaged for Hugging Face and verified on CPU (R69). Modal switched to account `emeraldshuorbit` (R73). Decisions D45–D48 (D46: race = standard Opus vs Busy Machines; D47: pre-registered LCBench wide race; D48: lead with quality and cost).

**Running (workflows):** `bm-cost-cut` (cadence build → LCBench wide race, pre-registration in `docs/preregistration_lcbench_wide.md` → RL cadence gate → RL race on Modal); `sapling-v3-and-gpu-check` (H100 end-to-end check, gated v3, hard stop 18:00); `demo-and-pitch-pack` (demo script primary/fallback, correction beat, Mubit Q&A, slides outline, evidence pack).

**DEADLINE (D53): code freeze and submission Sun 06:00 London (06:00 UTC).** Races done ~00:30 → patches 06–09 + full tests ~02:30 → final numbers/docs ~04:30 → demo decision 05:00 → final push ~05:30.

**Needs the user:** HF name + `hf auth login` + "go" to publish; Discord replies (stuck jobs, adapter header); B's page when v2 appears; "commit" when wanted. Deadline: everything except frontend by Sun 09:00 UTC.

## 0b. Earlier: morning brief (Sat 24 Oct ~06:50 BST)
**Read first:** `docs/results_log.md` R30–R43 and `docs/decisions.md` D36–D43. Everything below is logged there.

**What we can say today (measured, with the worse side):**
- **LCBench, 3 seeds (R38):** the Opus 5.5 research agent's model has **28% fewer errors than Optuna's**; Optuna never reached Opus's quality; Optuna accepted 8/62 fake bests, Opus 0/33. Worse for Opus: 0.3× the experiments per $; in seed 2 Optuna reached the goal faster and cheaper.
- **Opus keeps 30 workers waiting 70% of the time** (RL, 15 s runs, R41, 1 seed); Opus planning slows from 3.5 s to 16 s as it fills more workers (R36). Validated simulator (R43: 71.2% vs 70.0% measured) projects **8–15% waiting with a 1–2 s planner**: this is the Mubit adapter's value, still a projection.
- **Busy Machines judge (R37, R40):** Opus for leakage + our seed-spread rule confirms **7/7 real gains (Opus alone 1/7) with 0/32 fakes let through**; judge v1 is ~22% faster than v0.
- **Without a checker, search ships fake wins:** Optuna's best RL model was a contaminated one (R36, held-out −201 vs Opus −164).
- **We improved Mubit's own model (R33, R42):** sapling-research v2 answers **64% of AI-research claims (stock: 0%) approving 0/157 fakes**, raw accuracy 22% → 66%, no regression on ContractNLI (77% → 84%). Still far below Opus as a judge (36/50 vs 50/50 on the trap set).

**What didn't work (logged):** our engine add-ons (judge policy in the planner's history + median stop) made Opus **9% worse** on LCBench, n=3 (R44); they are now verification-only (D44). LM target has no real gain (R23); early stopping fails on every target (R41, R43, now off: D43); early_stop as an LLM step fails the D39 gate (R31); Mubit adapters below target: propose 14.3%, judge v0 42.9% vs 95% (R28, R30).

**Mubit state:** project A = propose v1 / judge v0 / diagnose; project B = judge v1 (closed output), 177 examples recorded (R39). No routes anywhere. Daily review limit ~200/task (R29). Mubit arm can route its judge calls to B (built, tested).

**Running at 06:50:** D44 fixes (planner sees the same as the Opus arm; σ-key bug; early stop off), then **RL wide n=3 Opus vs Optuna**, **one CIFAR Opus vs Optuna**, and a **planner-speed ablation** (same loop with Haiku 5.5 as a smaller, faster model — tests the R43 projection with real data; not a Mubit adapter). Fairness fixes are in (R45: cancel lag measured and fixed; headline gates).

**Decisions for you when you wake:**
1. Check the Mubit console for **judge v1 in project B** (exam result once ≥25 kept) and whether propose v2 trained; paste the adapter pages.
2. Ask Mubit: how to create a route; can the 95% target be lowered; review limit; what "Prompt only" means (`docs/mubit_office_hours.md`).
3. Frontend is unstarted (by your instruction). Plan: start ~12:00 with design A+ and the C race screen; a replay API exists for the fallback (`backend/replay.py`).
4. HF publishing of sapling-research v2 (draft in `data/sapling/sapling_v2/release/`, not published): your call.

### Scope lock (D35): nothing new gets built after this list
| # | Track | Owner | Go/no-go | Cut if late |
|---|---|---|---|---|
| 1 | **Wide RL race**, Opus vs Optuna (then + Mubit arms) | agent (wave 5) | first race by Sat 06:00 | never cut |
| 2 | **Judge adapter v1 → v2 with corrections**; real-gain-vs-noise set (Opus, Opus+rule, rule) | agents + console | v1 quality known by Sat 04:00; v2 trained by Sat 16:00 | never cut (judging criterion #1) |
| 3 | **early_stop step** (D34): prompt frozen + ~250 examples tonight, scheduler integration Sat morning | agent | examples recorded by 04:00; integrated by Sat 12:00 | drop scheduler integration, keep the offline Opus-vs-rule-vs-adapter table |
| 4 | **Sapling stages 1–2** + parity vs Mubit's endpoint (D33) | agent | stage 2 numbers by Sat 14:00 | hard stop Sat 18:00 (D20) |
| 5 | **LCBench race** | agent (wave 5) | one Opus-vs-Optuna race by Sat 08:00 | cut first if it slips |
| 6 | **Benchmark proper:** 3 seeds × (Opus, Mubit stock, Mubit + ours, Optuna) on RL; CIFAR confirmation | lead | runs Sat 14:00–22:00 | CIFAR → 1 seed |
| 7 | **Frontend** (design A+ chosen direction; race screen = design C) | agents | start Sat 12:00, done Sat 22:00 | landing page → static; keep race screen + results |
| 8 | **Demo:** fallback recording Sat 23:00, rehearsals Sun 06:00–09:30 | humans | — | never cut |

**Not building (decided):** LM as a headline target (D32), LoRA/full Sapling fine-tune (D23), open-model arm (no Groq/Together key), any change to the three frozen prompts.

**Waiting on Mubit:** daily review limit for propose (`ln_42bac9601f877d03fc0999bc`), how to create a route, 95% target, "Prompt only" meaning (`docs/mubit_office_hours.md`).

## 1. Component by component
| # | Component | How | Hours | Difficulty | Main risk |
|---|---|---|---|---|---|
| 1 | **Mubit client + 3 prompts** | Anthropic SDK with `base_url` = Mubit; frozen system prompts; structured-output schemas; usage + request-ID logging | 3–4 | 🟢 Easy | Low: caching and structured outputs verified through Mubit |
| 2a | **Target: image recognition** (CIFAR-10 speedrun) | Port an open-source airbench-style script; expose ~8 settings; add the contamination option | 2–3 | 🟢 Easy | None big; trains in seconds |
| 2b | **Target: language model** | **nanoGPT on a small text dataset** (Shakespeare-char style), not full nanochat. ~60 s on an L4. Data cached in a Modal Volume | 4–6 | 🟡 Medium | Noisy in 60 s; needs 3 runs per setting |
| 2c | **Target: game-playing AI** (RL on CPU) | Single-file PPO for gymnax in JAX (purejaxrl-style); expose settings; evaluate on held-out seeds | 3–4 | 🟡 Medium | JAX install / CPU speed in Modal containers (test early) |
| 3 | **Runners** (local + Modal) | `Runner` interface; a Modal Function per target; **metrics back via `modal.Queue`** polled by the backend (Modal can't call our laptop); slot caps (10 GPUs / 100 containers); pre-warm | 3–5 | 🟡 Medium | Cold starts; queueing showing up as fake idle time |
| 4 | **Race engine** (scheduler) | asyncio loop per arm: free slots → batch proposal → launch → collect → leaderboard (code) → diagnose/judge when needed; busy/idle timers; $ caps | 5–7 | 🟠 Medium-hard | **The benchmark's credibility depends on this being correct and fair** |
| 5 | **Baseline arm** (Optuna) | Same scheduler, Optuna as the proposer, no LLM | 1–2 | 🟢 Easy | none |
| 6 | **Backend API** | FastAPI + SQLite (8 tables) + WebSocket events + JSONL export | 3–4 | 🟢 Easy | none |
| 7 | **Dashboard** | Pen.dev design (~2 h), then React + shadcn + Recharts: GPU tiles, live curves, leaderboard, cost panel, flag button, results page | 10–12 | 🟡 Medium (time sink) | Polish eats unlimited time; timebox it |
| 8 | **Mubit arm** (small models answering) | If Mubit routes automatically (Q1): **0 h**. If not: open-model fallback via Mubit's Groq/Together routes, 1–2 h, which changes the story slightly | 0–2 | ⚪ Depends on Mubit | **Biggest external risk** |
| 9 | **Correction → model update** | Flag button → stored correction → into Mubit (API or console button, Q2/Q9). v2 must be **trained in time**: 3 starts/day, up to 20 min each, slow review | 1–3 | ⚪ Depends on Mubit | Timing: plan trainings, don't trigger them casually |
| 10 | **Benchmark runs + results** | Overnight/Saturday runs; analysis script; 3–4 plain-language charts | 3–4 (+ machine time) | 🟢 Easy | GPU budget ($30) |
| 11 | **Demo prep** | Script, 5 rehearsals, a recorded fallback video, slides | 4–5 | 🟢 Easy | Starting it too late |
| | **MUST-HAVE total** | | **≈ 45–60** | | |
| 12 | *Optional:* **Sapling fine-tune** (`sapling_research.md`) | Dataset generator (3–4 h); load the 46 GB model on a Modal H100 + baseline eval (3–4 h); head fine-tune + recalibration (4–6 h); HF publish + model card (1–2 h) | **11–16** | 🟠 Medium-high | Custom loader; H100 cost; first-load time |

## 2. Does it fit?
| Team | Person-hours | Verdict |
|---|---|---|
| **1 person** | ~28 | ❌ Not everything. Cut to 2 targets (CIFAR + RL), a simpler dashboard, **no Sapling** |
| **2 people** | ~56 | ⚠️ Must-haves fit, just. Sapling only if things go smoothly |
| **3 people** | ~84 | ✅ Must-haves + Sapling, with buffer |
| **4 people** | ~110 | ✅ Comfortable; one person on Sapling full-time |

### Team split (3–4 people)
1. Agent loop + Mubit integration + corrections
2. Targets + runners (local, Modal) + metrics streaming
3. Race dashboard + demo script
4. Benchmark runs + results charts + pitch (+ Sapling if a 4th person)

## 3. What's actually hard (it's not the code)
1. **Mubit unknowns (Q1, Q2).** Can small models answer live calls, and can we see it? How do corrections enter training? Ask **in the first hour**; the plan adapts.
2. **Getting a real, visible win.** "Cheaper" and "faster" are very likely. **"Better model" is not guaranteed**: it only happens if the extra experiments find better settings within the budget. Short runs and many parallel CPU workers make it far more likely. Report it honestly either way.
3. **Adapter timing and the quality gate.** Calls → trained adapter took ~20–25 min on the toy task, so training itself isn't the bottleneck. **Passing the exam and getting routed is:** the toy v1 scored 70% vs an 85% target and has no route (R7), and there's a shadow stage before live. Collect well over 25 examples per task (aim for 75–100), leave time for a retrain, and max 3 adapter starts per UTC day. **Example collection must start Friday night**, so prompts must be frozen by ~01:00 Saturday.
4. **Benchmark correctness.** Fair slots, the same budgets, held-out scoring, 3 runs each. Bugs here cost credibility, not just time.
5. **Dashboard time sink.** Design first in Pen.dev, then build only what the demo shows.

## 4. Critical path (the order that de-risks fastest)
1. **Fri 19:30–21:00:** ask Mubit Q1/Q2; scaffold the repo; first call through Mubit ✅ (done in probes); Modal "hello world" with a CPU + GPU function.
2. **Fri 21:00–01:00:** CIFAR target on Modal; race engine with the **mock assistant**; prompts on Haiku → **freeze** → Opus.
3. **Fri 01:00 → Sat morning:** **run Opus races overnight** (collects Mubit examples and gives the baseline); RL target on CPU.
4. **Sat:** dashboard; language-model target; Mubit arm; corrections → v2; (optional) Sapling, with go/no-go at 14:00 and a hard stop at 18:00.
5. **Sat night:** the full benchmark; results charts; record the fallback video.
6. **Sun 06:00–10:00:** freeze, rehearse ×5. **Code freeze 10:00.**

## 5. Cut ladder (if behind, cut in this order)
1. Sapling fine-tune → keep only the baseline eval, or drop it
2. Language-model target (keep CIFAR + RL)
3. Results page polish (keep one chart)
4. Optuna baseline arm (mention it in Q&A instead)
5. **Never cut:** the live race, the correction moment, the Opus comparison with real numbers
