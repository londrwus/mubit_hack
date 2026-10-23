# How hard is it to build? (honest estimate)

Available time: Fri 19:30 → Sun 10:00 ≈ **39 hours** (the clocks go back, which gives us one extra hour). Realistically ~26–30 working hours per person after sleep and food.
Estimates are in **person-hours** for someone fluent in Python/React and new to Modal/Mubit.

## Component by component
| # | Component | How | Hours | Difficulty | Main risk |
|---|---|---|---|---|---|
| 1 | **Mubit client + 3 prompts** | Anthropic SDK with `base_url` = Mubit; frozen system prompts; structured-output schemas; usage + request-ID logging | 3–4 | 🟢 Easy | Whether Mubit passes caching/structured outputs through (Q11) |
| 2a | **Target: image recognition** (CIFAR-10 speedrun) | Port an open-source speedrun script (airbench-style); expose ~8 settings; add the contamination option | 2–3 | 🟢 Easy | None big; trains in seconds |
| 2b | **Target: language model** | **nanoGPT on a small text dataset** (Shakespeare-char style), not full nanochat. Trains meaningfully in ~60 s on an L4. Data cached in a Modal Volume | 4–6 | 🟡 Medium | Results noisy in 60 s; needs 3 runs per setting to compare |
| 2c | **Target: game-playing AI** (RL on CPU) | Single-file PPO for gymnax environments in JAX (purejaxrl-style); expose settings; evaluate on held-out seeds | 3–4 | 🟡 Medium | JAX install/CPU speed in Modal containers (test early) |
| 3 | **Runners** (local + Modal) | `Runner` interface; a Modal Function per target; **metrics back via `modal.Queue`** polled by the backend (Modal can't call our laptop); slot caps (10 GPUs / 100 containers); pre-warm | 3–5 | 🟡 Medium | Cold starts; queueing showing up as fake idle time |
| 4 | **Race engine** (scheduler) | asyncio loop per arm: free slots → batch proposal → launch → collect → leaderboard (code) → diagnose/judge when needed; busy/idle timers; $ caps | 5–7 | 🟠 Medium-hard | **The benchmark's credibility depends on this being correct and fair** |
| 5 | **Baseline arm** (Optuna) | Same scheduler, Optuna as the proposer, no LLM | 1–2 | 🟢 Easy | none |
| 6 | **Backend API** | FastAPI + SQLite (7 tables) + WebSocket events + JSONL export | 3–4 | 🟢 Easy | none |
| 7 | **Dashboard** | Pen.dev design (~2 h), then React + shadcn + Recharts: GPU tiles, live curves, leaderboard, cost panel, flag button, results page | 10–12 | 🟡 Medium (time sink) | Polish eats unlimited time; timebox it |
| 8 | **Mubit arm** (small models answering) | If Mubit routes automatically (Q1): **0 h**. If not: open-model fallback via Mubit's Groq/Together routes, 1–2 h, which changes the story slightly | 0–2 | ⚪ Depends on Mubit | **Biggest external risk** |
| 9 | **Correction → model update** | Flag button → stored correction → into Mubit (API or console button, Q2). v2 must be **trained in time**: 3 trainings/day, up to 20 min each | 1–3 | ⚪ Depends on Mubit | Timing: plan trainings, don't trigger them casually |
| 10 | **Benchmark runs + results** | Overnight/Saturday runs; analysis script; 3–4 plain-language charts; replication run | 3–4 (+ machine time) | 🟢 Easy | GPU budget ($30) |
| 11 | **Demo prep** | Script, 5 rehearsals, a recorded fallback video, slides | 4–5 | 🟢 Easy | Starting it too late |
| | **MUST-HAVE total** | | **≈ 45–60 person-hours** | | |
| 12 | *Optional:* **Sapling fine-tune** | Dataset generator (3–4 h); load the 46 GB model on a Modal H100 + baseline eval (3–4 h); head fine-tune + recalibration (4–6 h); HF publish + model card (1–2 h) | **11–16** | 🟠 Medium-high | Custom loader; H100 cost; first-load time |

## Does it fit?
| Team | Person-hours | Verdict |
|---|---|---|
| **1 person** | ~28 | ❌ Not everything. Cut to: 2 targets (CIFAR + RL), a simpler dashboard, **no Sapling** |
| **2 people** | ~56 | ⚠️ The must-haves fit, just. Sapling only if things go smoothly |
| **3 people** | ~84 | ✅ Must-haves + Sapling, with buffer |
| **4 people** | ~110 | ✅ Comfortable; one person on Sapling full-time |

With Claude Code doing much of the boilerplate (API, schemas, UI components), expect the low end of each range.

## What's actually hard (it's not the code)
1. **Mubit unknowns (Q1, Q2, Q11).** Can small models answer live calls? How do corrections enter training? Ask **in the first hour**; the plan adapts.
2. **Getting a real, visible win.**
   - "Cheaper" and "faster" are very likely: smaller models, fewer waits.
   - **"Better model" is not guaranteed.** It only happens if the extra experiments find better settings within the budget. Short training runs (seconds to ~1 min) and many parallel CPU workers make it far more likely.
   - Report honestly whichever way it lands.
3. **Small-model training timing.** Each task needs ≥25 good examples before Mubit trains a small model (up to 20 min, max 3 per day). **Example collection must start Friday night**, so prompts must be frozen by ~01:00 Saturday.
4. **Benchmark correctness.** Fair slots, the same budgets, held-out scoring, 3 runs each. Bugs here cost credibility, not just time.
5. **Dashboard time sink.** Design first in Pen.dev, then build only what the demo shows.

## Critical path (the order that de-risks fastest)
1. **Fri 19:30–21:00:** ask Mubit Q1/Q2/Q11; scaffold the repo; first call through Mubit; Modal "hello world" with a CPU + GPU function.
2. **Fri 21:00–01:00:** CIFAR target on Modal; race engine with the **mock assistant**; prompts on Haiku → **freeze** → Opus.
3. **Fri 01:00 → Sat morning:** **run Opus races overnight** (collects Mubit examples and gives the baseline); RL target on CPU.
4. **Sat:** dashboard; language-model target; Mubit arm; corrections → v2; (optional) Sapling.
5. **Sat night:** the full benchmark; results charts; replication.
6. **Sun 06:00–10:00:** freeze, record the fallback video, rehearse.

## Cut ladder (if we're behind, cut in this order)
1. Sapling fine-tune → keep only the "baseline eval" or drop it entirely
2. Language-model target (keep CIFAR + RL)
3. Results page polish (keep one chart)
4. Optuna baseline arm (mention it in Q&A instead)
5. **Never cut:** the live race, the correction moment, the Opus comparison with real numbers
