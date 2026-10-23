# Demo: "The Night Shift Race"

Judging criterion: "Clear, live, and under the time limit." The room should **feel** the result, not just read it from a chart. The event is called Night Shift, so lean in: two AI assistants work the night shift on identical GPUs. Everything on screen follows the plain-language rules (D19, `pitch.md`).

## 1. The screen (live split screen)
```
┌─────── STANDARD AI ASSISTANT (Opus 5.5) ───────┬─────── BUSY MACHINES (powered by Mubit) ───────┐
│ GPUs  ▢▢■▢   GPUs waiting 77%  😴 "thinking…"  │ GPUs  ■■■■   GPUs waiting 6%  ⚡              │
│ Experiments finished: 3                        │ Experiments finished: 17                       │
│ Model quality  ███░░░░░░░  ▁▂▂▃                 │ Model quality  ██████░░░░  ▁▃▄▆▇█              │
│ Thinking: "Considering the learning rate…"     │ Thinking: ✓ warmup longer ✓ wider model ✓ …    │
│ Money wasted on waiting: $2.40                 │ Money wasted on waiting: $0.20                 │
└────────────────────────────────────────────────┴────────────────────────────────────────────────┘
                         ⏱ 02:41 remaining     💰 GPU money spent: $0.09 | $0.09
```
(All numbers are placeholders. The live screen shows live values; slides show only logged ones.)

### Visual elements
1. **GPU tiles** light up when a training run is active and go grey while the GPU waits for the assistant. The Opus side visibly flickers grey. **The single most important visual.**
2. **Live training curves** stream from each run: many thin lines on our side, few on the Opus side, best run highlighted.
3. **Leaderboard ticker:** model quality per side, updated live, with a flash on a new best.
4. **Thought stream:** each proposal's short `hypothesis` (≤25 words, already in the JSON output, so it costs nothing). Our side scrolls visibly faster.
5. **Counters and cost panel:** experiments finished, how fast it answers, and the money breakdown:
```
                         Standard assistant   Busy Machines
GPU money (paid)         $3.20                $3.20
  ↳ wasted on waiting    $2.40 (75%)          $0.20 (6%)
Assistant (LLM) money    $6.10                $0.40
Time to reach the goal   didn't reach it      38 min
Cost to reach the goal   —                    $2.90
```
(Placeholders.)

### Dashboard labels
| Technical name | What the screen says |
|---|---|
| GPU idle % / GPU busy | **GPUs waiting** / **GPUs working** |
| $-to-target / time-to-target | **Cost to reach the goal** / **Time to reach the goal** |
| best score at fixed budget | **Model quality** (a 0–100 bar, higher = better) |
| experiments per GPU-hour | **Experiments finished** |
| idle GPU $ | **Money wasted on waiting** |
| val_bpb / test_acc / mean return | **Model quality** (raw metric only on hover) |
| contamination trap | **Fake improvement caught** |
| adapter v1 → v2 | **"Learned from your correction"** |
| Opus 5.5 arm | **Standard AI assistant** (subtitle: Opus 5.5) |
| Mubit arm | **Busy Machines** (subtitle: powered by Mubit) |
| LM / RL / CIFAR targets | **Language model · Game-playing AI · Image recognition** |

Model quality is normalised per target: 0 = the starting model, 100 = the best known result. The raw number is on hover and in the export, so the claim stays verifiable.

## 2. The correction → model update moment (mandatory; make it dramatic)
1. Mid-race, our assistant claims a huge jump: **the planted contamination trap** (it tested on data it had already seen). The leaderboard flashes a suspicious ↑↑.
2. The presenter clicks **"🚩 Flag: fake improvement"**. This is the human correction (Sapling flags count too).
3. Cut to the **Mubit console**: the correction is recorded, and `judge_improvement` **v1 → v2** appears in the version history. v2 was trained beforehand; say so honestly ("trained last night from N corrections like this one").
4. **Try models**, side by side: the same contaminated claim goes to v1 and v2. v2 **rejects it** ("gain invalid: eval split overlaps train"). Red ✗ turns to green ✓.
5. Caption: **"Every correction becomes a model update."** The rubric's own words, on screen. Say "Mubit" out loud here.

### Show the Mubit console three times
1. **Tasks page:** three research tasks with example counts. *"Our assistant's brain, as three Mubit tasks."*
2. **Requests filtered by session:** one researcher's full run. *"Its procedural log."*
3. **Adapters → History:** judge v1 → v2, then Try models on the contaminated claim (the moment above).

## 3. The 3-minute script
| Time | Beat | Screen |
|---|---|---|
| 0:00–0:20 | Hook: "AI labs are GPU-bound. Their AI assistant's GPUs sit idle while it thinks, and they pay for every grey second." | Opus side alone, GPUs mostly grey |
| 0:20–0:40 | "Same GPUs, same budget, one change: its thinking runs through Mubit." Start the race. | Split screen goes live |
| 0:40–1:30 | Race runs. Narrate GPUs waiting, experiments finished, new bests. | Tiles, curves, ticker |
| 1:30–2:10 | Fake improvement → flag → v1→v2 in the Mubit console → rejection | Console, then the trap replay |
| 2:10–2:40 | Full results: 3 runs each on language model / game-playing AI / image recognition (measured only) | One clean results chart |
| 2:40–3:00 | "Spend less on GPUs. Get a better model. Train faster. Built for AI labs." | Logo and one line |

### Narration (plain words)
| Moment | Say | Don't say |
|---|---|---|
| Opening | "Two AI assistants. Same GPUs. Same budget. Same clock. Let's race." | "Two agent arms with N GPU slots each…" |
| Grey tiles | "Grey means the GPU is waiting. You're paying for every grey second." | "GPU idle %, utilisation" |
| Counters | "This one has finished 17 experiments. That one, 3." | "experiments per GPU-hour" |
| Leaderboard | "Higher is better. Ours is ahead." | "val_bpb went from 1.14 to 1.09" |
| Cost | "That side has wasted $2.40 on waiting. Ours, 20 cents." | "idle GPU $, $-to-target" |
| The mistake | "It thinks it found a huge improvement. But it's fake: the model was tested on data it had already seen." | "eval contamination, train/val overlap" |
| The correction | "We flag it once… Mubit retrains it… now it spots fake improvements on its own." | "adapter v1 → v2, corrections as training examples" |
| Close | "Spend less on GPUs. Get a better model. Train faster." | "cost per discovery at fixed budget" |

(Numbers in the narration come from the live screen, never memorised.)

## 4. Build and run notes
- **The race must be real:** 2 L4s per arm for 3 minutes (about $0.20), or the CPU RL target with 32 workers per arm. **Pre-warm the Modal containers** so there's no cold start on stage.
- **Stack:** FastAPI + WebSocket backend streaming `gpu_state`, `metric`, `new_best`… (`architecture.md`). Vite + React + shadcn/ui + Tailwind (+ Recharts), designed first in Pen.dev. Dark theme, large type readable from the back of the room.
- **Log in to the Mubit console fresh** right before the demo (it signs out after 15 minutes idle).
- **Fallback:** record a perfect run on Saturday night and keep it ready if Wi-Fi or Modal fails. Say so openly if it's used.
- **Expo table (10:30–12:30):** loop the race on a monitor, with a QR code to the repo and the results export. People stop to watch races.
- **Rehearse at least 5 times** with a timer. Cut anything that pushes past 3 minutes.
