# Build plan: effort, order, cuts

Available time: Fri 19:30 → Sun 10:00 ≈ **39 hours** (clocks go back, +1 h). Realistically ~26–30 working hours per person after sleep and food.
Estimates are **person-hours** for someone fluent in Python/React and new to Modal/Mubit. With Claude Code writing much of the boilerplate (API, schemas, UI components), expect the low end.

## 0. Next session: start here (status at Fri 23 Oct ~23:15 BST)
**Where we are:** planning done, docs consolidated, **no code yet**. Mubit works end to end on a toy task (R6–R9): calls → trained adapter in ~20–25 min, but toy v1 scored 70% vs an 85% target, has no route, returns malformed JSON on unfamiliar cases, and valid replies (~1.6 s) are no faster than Haiku.

**Build order for the session** (backend, design and frontend can run in parallel):
1. **Scaffold:** `backend/` (uv, FastAPI, Pydantic, Anthropic SDK, SQLite), `frontend/` (Vite + React + TS, Tailwind, shadcn/ui), `workers/` (targets, runners), `.env.example` with `MUBIT_API_KEY`.
2. **`mubit_client.py`:** the only LLM client; session IDs, request-ID capture, the four usage meters, schema validation → retry → Opus fallback, `x-middleware-record: off` for dev. Plus the **mock LLM**.
3. **Opus latency probe** (first real measurement of the session): the same short prompt to Opus 5.5 and Haiku 5.5 through Mubit, ~3 calls each, recording off. If Opus is ~2 s, rethink the speed story before building further.
4. **Design:** the race screen in Pen.dev (`demo.md` layout and labels), then build it in React against mock WebSocket events.
5. **First target + runner:** CIFAR speedrun on a Modal Function, `local` runner for tests, metrics back via `modal.Queue`.
6. **Scheduler + race engine** with the mock LLM, end to end on one target. Then SQLite tables, REST/WS API and JSONL export (`architecture.md`).
7. **The three prompts:** draft on Haiku, **freeze**, then start Opus example collection. Examples must cover the full mix (healthy runs, overfitting, contamination), not just failures. Aim for 75–100 per task.

**Take to office hours:** Q1 (how to create a route, what shadow needs), Q2 (corrections overriding labels), Q14 (85% target, retraining), Q15 (is the JSON format enforced, is Try-it latency representative).

**Timing:** the critical path below assumes Opus example collection runs **overnight Fri → Sat**. Starting the build on Saturday morning loses that night, so in that case make step 7 the first thing done.

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
