# Demo design: make the room *watch a race*

Judging criterion: "Clear, live, and under the time limit." The room should **feel** the result, not just read it from a chart.

## Concept: "THE NIGHT SHIFT RACE" (a live split screen)
The event is called Night Shift, so lean in: two AI researchers work the night shift on identical GPUs.

```
┌──────────── OPUS 5.5 RESEARCHER ────────────┬──────────── MUBIT RESEARCHER ───────────────┐
│ GPU ▢▢■▢   utilisation 23%  😴 "thinking…"   │ GPU ■■■■   utilisation 94%  ⚡               │
│ experiments: 3        $/exp: $0.41           │ experiments: 17       $/exp: $0.03           │
│ best val_bpb: 1.142 ▁▂▂▃                     │ best val_bpb: 1.098 ▁▃▄▆▇█                   │
│ thought log: "Considering learning rate…"    │ thought log: ✓ lr=3e-3 ✓ warmup=200 ✓ ...    │
└──────────────────────────────────────────────┴──────────────────────────────────────────────┘
                         ⏱ 02:41 remaining     💰 GPU $ spent: $0.09 | $0.09
```

### Visual elements (the "wow" parts)
1. **GPU tiles light up** when a training run is active and go grey when the GPU is idle waiting for the LLM. The Opus side visibly flickers grey. This is the single most important visual.
2. **Live loss curves** stream from each training run. Many thin lines appear on the Mubit side and few on the Opus side, and the best run is highlighted.
3. **Leaderboard ticker:** best score per arm, updated live. A sound or flash plays when a new best appears.
4. **Thought stream:** the agent's decisions scroll by. The Mubit side is visibly faster.
5. **Counters:** experiments run, $ per experiment, mean LLM latency (e.g. 6.2 s vs 0.3 s).

## The "correction → model update" moment (mandatory, make it dramatic)
1. Mid-race, the Mubit agent claims a huge jump (**a planted leakage trap**: it evaluated on training data). The leaderboard flashes a suspicious ↑↑.
2. The presenter clicks **"🚩 Flag: leakage"** on screen. This is the human correction.
3. The screen cuts to the **Mubit console**: the correction is recorded, and adapter **v1 → v2** appears in the version history (pre-trained, shown honestly as "trained last night from 40 corrections like this").
4. Hot-swap to v2: the same trap rollout is fed in, and v2 **rejects it** ("gain invalid: eval split overlaps train"). Red ✗ turns to green ✓.
5. Caption: **"Every correction becomes a model update."** These are the rubric's own words, on screen.

## The 3-minute script
| Time | Beat | Screen |
|---|---|---|
| 0:00–0:20 | Hook: "Labs and banks are GPU-bound. Your AI researcher's GPUs sit idle while it thinks." | Opus side alone, GPUs mostly grey |
| 0:20–0:40 | "Same GPUs, same budget, one change: its brain runs through Mubit." Start the race. | Split screen goes live |
| 0:40–1:30 | Race runs. Narrate the idle %, experiment counts and new bests. | Tiles, curves, ticker |
| 1:30–2:10 | Leakage trap → flag → v1→v2 in the Mubit console → rejection | Console and trap replay |
| 2:10–2:40 | Full results: 1 hour × 3 seeds on nanochat / RL / fraud, plus the H100 replication | One clean results chart |
| 2:40–3:00 | "Same discoveries, X× more experiments per GPU-dollar. AI labs for discovery, banks for fraud models." | Logo and one line |

## Build notes
- **Stack:** a FastAPI + WebSocket backend. Modal training workers stream metrics (loss, step, GPU busy/idle) to it. The frontend is a single page (React or plain HTML with Chart.js). Use a dark theme and big fonts readable from the back of the room.
- **The race has to be real:** 2 L4s per arm for 3 minutes (about $0.20). Pre-warm the Modal containers so there's no cold start on stage.
- **Fallback:** record a perfect run on Saturday night and keep it ready to play if the Wi-Fi or Modal fails. Say so openly if it's used.
- **Expo table (10:30–12:30):** loop the race on a monitor continuously, with a QR code to the repo and results JSONL. People stop to watch races.
- **Rehearse at least 5 times** with a timer. Cut anything that pushes past 3 minutes.

## Team split (if 3–4 people)
1. Agent loop + Mubit integration + corrections
2. Modal training harness (nanochat, RL, fraud) + metrics streaming
3. Race dashboard UI + demo script
4. Benchmark runs + results chart + pitch
