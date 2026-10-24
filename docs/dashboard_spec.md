# Dashboard spec: the one-screen race view (for the Pen mock and the build)

One screen at **1920×1080**. It is recorded and sped up for the 3-minute video (`video_script.md`), so it has to read at a glance after compression. It is a minimal live view of one race: **two columns, same layout, only the numbers differ.** Plain-language rules: CLAUDE.md "UI copy", `demo.md` §1 label table. Framing: D46 (Opus 5.5 vs Busy Machines, the no-AI tuner kept off the main screen) and D48 (quality and money first; waiting is *why*, never the headline).

Every sample value below comes from a logged race export, so the mock looks like the real thing:
- **Game-playing:** R78 seed 0, race `4ca4ed27`, `data/bench/race-rl-seed0-4ca4ed27.jsonl`
- **Tabular:** R82 seed 0, race `c1f4b1d9`, `data/bench/race-lcbench-seed0-c1f4b1d9.jsonl`

The values were computed from those files with the replay's own rules (`backend/benchmark/metrics.py` `quality` / `is_counted`, and `scripts/replay_tui.py` for fakes caught). In the real dashboard **no number is typed in**: every value is streamed from the race or the replay.

---

## 1. Layout (1920×1080)

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐ 0
│ HEADER (96 px)  Task · Budget · [mode tag]                              Race clock 07:30  │
├────────────────────────────────────────────┬─────────────────────────────────────────────┤ 96
│ Standard AI assistant                      │ Busy Machines                               │
│ Opus 5.5                                   │ Opus 5.5                                    │
│ ┌ Machines grid ─────────┐ ┌ Modal live ─┐ │ ┌ Machines grid ─────────┐ ┌ Modal live ─┐ │
│ │ ■■□□□□ ...             │ │ demo only   │ │ │ ■■■■■■ ...             │ │ demo only   │ │
│ └────────────────────────┘ └─────────────┘ │ └────────────────────────┘ └─────────────┘ │
│ AI bill  $1.04        Experiments finished │ AI bill  $0.30        Experiments finished │
│                       153                  │                       523                   │
│ Model quality ─── line chart, goal line ── │ Model quality ─── line chart, goal line ── │
│ Fake improvement caught · ticker (3 rows)  │ Fake improvement caught · ticker (3 rows)  │
├────────────────────────────────────────────┴─────────────────────────────────────────────┤ 1040
│ FOOTER (40 px)  R-ids · AI $ at list price · [simulated machine time]                     │
└──────────────────────────────────────────────────────────────────────────────────────────┘ 1080
```
- Two columns of 928 px with a 32 px gutter and 16 px outer margins. **Left is always the Standard AI assistant, right is always Busy Machines.**
- Type (stage and video): big numbers 64 px, labels 24 px, captions and footer 18–20 px. Nothing smaller than 18 px.
- Dark theme. Busy Machines' column must not look styled as "the winner" (no glow or highlight colour of its own). Both columns use the same styling.
- **Without the Modal panel** (replay mode), the machines grid takes the full column width.

## 2. Header

| Element | Exact label / text | Source | Update | Sample: game-playing (R78 s0) | Sample: tabular (R82 s0) |
|---|---|---|---|---|---|
| Task | "Game-playing AI" / "Tabular model tuning" / "Image recognition" | race `config.target` mapped as in `demo.md` §1 (`rl` / `lcbench` / `cifar`) | static | Game-playing AI | Tabular model tuning |
| Budget | "15 min · up to $8.00 per side · 30 machines each" | `config.budget_s`, `cost_cap_usd` (per side), `gpus_per_arm` | static | 15 min · up to $8.00 per side · 30 machines each | 15 min · up to $9.20 per side · 16 machines each |
| Goal | "Goal: −170" (small, next to the budget) | `config.target_value` | static | Goal: −170 | Goal: 0.78 |
| Mode tag (top right, always visible) | Replay: **"Replay of logged race R78 seed 0 · 30× speed"**. Demo: **"Demo run, not a benchmark result"** | replay CLI args / `config.demo_label` (code constant `DEMO_LABEL`: "demo run for the video, not a benchmark result") | static | Replay of logged race R78 seed 0 · 30× speed | Replay of logged race R82 seed 0 · 30× speed |
| Race clock | "07:30 / 15:00" (race time, not wall time) | event `ts` minus race start | 1 Hz in race time | 07:30 / 15:00 | 07:30 / 15:00 |

Demo race header (preset `demo-cifar-modal-5m`): "Image recognition · 5 min · up to $3.00 per side · 3 GPUs each" + "Demo run, not a benchmark result". The word "Live" may appear only in the demo run, never over a replay.

## 3. Per-column elements

Column titles: **"Standard AI assistant"** with the subtitle **"Opus 5.5"**, and **"Busy Machines"** with the subtitle **"Opus 5.5"**. (Note for the build: `replay_tui.py` currently says "Standard Opus 5.5 agent" and "Busy Machines (Opus 5.5)". The dashboard uses the labels in this spec.)

### 3.1 Machines grid
| | |
|---|---|
| **Label** | Replay R78: **"Machines"** + caption "30 cloud CPU workers". Replay R82: **"Machines"** + caption "16 GPUs, simulated machine time". Demo: **"GPUs"** + caption "3 L4 GPUs on Modal". (R78 ran on CPU workers and R82's GPU time is simulated, so the label never says "GPUs" for them.) |
| **Cells** | One cell per slot (`slot_id`), in a fixed order. R78: 30 cells (6×5). R82: 16 (4×4). Demo: 3 (3×1) |
| **Colours** | **Working** = green, **Waiting** = amber, **Stopped early** = red (cancelled, failed or timed out), **Off** = dark grey (before the start or after the end). The TUI uses #3fb950 / #d29922 / #f85149 / #30363d. Working and waiting also differ in lightness or fill pattern (colour-blind safe). Small legend: "Working · Waiting for the assistant" |
| **Caption under the grid** | "8 of 30 working · waiting so far 71.9%" (small, secondary: D48) |
| **Source** | `gpu_state` events (`arm`, `slot_id`, `state` busy/idle); `experiment_finished.status` for red. Waiting so far = idle time ÷ paid time to date, from the same events (`ArmSummary.gpu_idle_pct` at the end) |
| **Update** | On every `gpu_state` event; render at most 10×/s |
| **Samples, R78 s0** | Standard: lights up in blocks of 8 (one plan = 8 runs). At 1:00 8 of 30 working; 2:00 16; 5:00 8; 10:00 8. Waiting so far 76.4% (1:00) → 71.9% (5:00) → **69.0% (end)**. Busy Machines: 30 of 30 working the whole race; waiting so far **1.4% (end)** |
| **Samples, R82 s0** | Standard: 7 of 16 working (0:30), 12 (1:00), 16 (5:00), 12 (10:00); waiting so far 57.3% (1:00) → 36.5% (5:00) → **26.2% (end)**. Busy Machines: 16 of 16 working; waiting so far **0.5% (end)** |

### 3.2 Modal live (demo race only)
Shown only when the mode is the demo run (preset `demo-cifar-modal-5m`, one Modal app per side: `busy-machines-opus` / `busy-machines-bm`). **Hidden in replays** (R82 never ran on Modal, and R78's Modal metering is not per side).

| Row label | Source | Update | Mock value |
|---|---|---|---|
| **"Containers running"** | Modal app stats per side (needs a backend feed; to be defined in `dashboard_api.md` / `dashboard_modal.md`, which don't exist yet) | poll every 2 s | 3 (the preset's `gpus_per_arm` = 3, `prewarm_spare` = 0) |
| **"GPU use"** (%) | per-container GPU utilisation, if the backend gets it from Modal or the worker. **No source exists in the code today** | poll every 2 s | **"—"**. No GPU use % has been logged, so the mock must not show an invented number. If there's no source on the day, drop the row |
| **"Runs in flight"** | count of slots in `busy` from `gpu_state` (= Modal function calls in progress) | per event | 0–3 |
| **"Latest Modal call"** | the latest run's Modal call id (`experiments.run_ref`). **Not in any event yet**: `experiment_started` has no `run_ref`, so it must be added, or served by the dashboard API | per `experiment_started` | format `fc-01M5Q9NGA0EVAPZG6DSSPJPX95` (a real id from R78 s0, shown here for its format only). Monospace, 18 px, truncate the middle if needed |

Never show the Modal account name, tokens, settings or any per-account e-mail in this panel.

### 3.3 AI bill
| | |
|---|---|
| **Label** | **"AI bill"**; hover "Assistant (AI) calls at list price, estimate" |
| **Source** | sum of `llm_call.call.cost_usd` per `call.arm` (`ArmSummary.llm_cost_usd` at the end) |
| **Update** | on every `llm_call` event; format `$0.00` (2 decimals) |
| **Samples, R78 s0** | Standard: $0.19 (1:00), $0.39 (2:00), $1.04 (5:00), $2.16 (10:00), **$3.30 (end)**. Busy Machines: $0.12, $0.18, $0.30, $0.51, **$0.70 (end)** |
| **Samples, R82 s0** | Standard: $0.15, $0.34, $0.97, $1.88, **$3.01 (end)**. Busy Machines: $0.11, $0.16, $0.28, $0.49, **$0.67 (end)** |
| **Rule** | Starts at $0.00. Never show a ratio ("4× cheaper") or a projection on this screen: the logged 3-run ratios belong in the video captions (R78, R82) |

### 3.4 Experiments finished
| | |
|---|---|
| **Label** | **"Experiments finished"**; hover on Busy Machines: "120 planned by the AI, 1,473 by its built-in tuner" (from `finished_by_source` at the end) |
| **Source** | count of `experiment_finished` with `status = succeeded`, per arm |
| **Update** | per event; thousands separator |
| **Samples, R78 s0** | Standard: 25 (1:00), 49 (2:00), 153 (5:00), 321 (10:00), **497 (end)**. Busy Machines: 90, 209, 523, 1,057, **1,594 (end)** |
| **Samples, R82 s0** | Standard: 22, 58, 153, 284, **420 (end)**. Busy Machines: 169, 273, 415, 593, **761 (end)** |

### 3.5 Model quality
| | |
|---|---|
| **Label** | **"Model quality"**, small "higher is better". Hover only: "best score so far on data the model never trained on (game-playing: average reward; tabular: accuracy)" |
| **Chart** | A step line of the **best score so far** over race time (x: 0 to 15:00), one line per column, plus a dashed **goal line** labelled "Goal". The big number above the chart is the current best |
| **Source** | `experiment_finished` + the experiment's held-out score, counted only if succeeded, not a fake and not rejected by the checker (`metrics.is_counted`, `metrics.quality`); `new_best` for the flash. Both columns' charts share one y-range |
| **Update** | on each new best; flash the big number for about 1.5 s of screen time |
| **Y-range** | R78: −200 to −150 (the starting model's −1,068.7 is drawn pinned to the bottom edge with a "starting model" label). R82: 0.75 to 0.90. Never rescale mid-race |
| **Format** | R78: 1 decimal (−155.0). R82: 4 decimals (0.8905) |
| **Samples, R78 s0** | Standard: −1,068.7 (0:30) → −187.6 (1:00) → −163.3 (2:00) → −160.3 (3:00) → −157.3 (5:00) → −155.4 (7:30) → **−155.0 (end)**. Busy Machines: −169.1 (0:30) → −163.0 (1:00) → −158.1 (2:00) → −157.0 (3:00) → −156.3 (5:00) → −155.1 (7:30) → **−154.9 (10:00, end)**. Goal −170 |
| **Samples, R82 s0** | Standard: 0.8249 (0:30) → 0.8409 (1:00) → 0.8534 (2:00) → 0.8747 (3:00) → 0.8864 (5:00) → 0.8887 (7:30) → 0.8903 (10:00) → **0.8905 (end)**. Busy Machines: 0.7600 → 0.8276 → 0.8670 → 0.8717 → 0.8845 → 0.8882 → **0.8903 (10:00, end)**. Goal 0.78 |
| **Goal reached** | small tag on the column, "Goal reached at 0:16", from the first counted score past the goal (race clock, rounded down). R78 s0: Standard 1:14, Busy Machines 0:16 (`time_to_target_s` 74.6 s / 16.9 s). R82 s0: Standard 0:24, Busy Machines 0:33 (24.4 s / 33.8 s). Before that, show nothing (no estimate) |

Don't normalise to a 0–100 bar unless the backend supplies that number: the shown values must match the log exactly (the video captions quote raw scores, R78/R82).

### 3.6 Fake improvement caught
| | |
|---|---|
| **Label** | Counter **"Fake improvements caught"** + a ticker of the latest 3 rows, each starting with **"Fake improvement caught"** |
| **Row text** | "0:16 · claimed −141.1, real score −162.4 (tested on data it had already seen)". Tabular rows have no "real score" (the shortcut also spoils that score): "0:10 · claimed 0.7028 (tested on data it had already seen)" |
| **Source** | an experiment that is a known fake (`experiment_finished.is_trap`) **and** whose checker verdict is invalid (the judge `llm_call` covering that `experiment_id`), the same rule as `replay_tui.py`. Also a `correction` event with label `fake_improvement` (a manual flag: append " · flagged by a person") |
| **Update** | per event. Rows hold at least 1.5 s of screen time; at 30× speed group bursts into one row: "+5 caught in 0:16" |
| **Samples, R78 s0** | Standard: **0** for the whole race (it never tried the shortcut). Busy Machines: 5 (0:30), 13 (1:00), 14 (2:00–5:00), 15 (7:30), 18 (10:00), **21 (end)**, all runs from its built-in tuner. First row: "0:16 · claimed −141.1, real score −162.4" |
| **Samples, R82 s0** | Standard: **0** (its checker rejected 2 claims that weren't fakes, so those aren't counted). Busy Machines: **4**, all by 0:10. Row: "0:10 · claimed 0.7028" |
| **Empty text** | "None so far. Every claimed new best is checked." |

This counts Busy Machines' own checker in the race (D40), not our Sapling fine-tune, which runs offline only (D28, R81).

## 4. Footer (one line, 18 px)
Replay R78: "Game-playing: logged race R78, seed 0 of 3 · 3-run result: a tie, for a quarter of the AI bill (R78) · AI $ at list price · every number: docs/results_log.md"
Replay R82: "Tabular: logged race R82, seed 0 of 3 · standard assistant ahead in every run (R82) · R82's pace was chosen after the first pre-registered setting FAILED (R75) · AI $ at list price · simulated machine time"
Demo: "Demo run, not a benchmark result · benchmark numbers: R78, R82, R75 in docs/results_log.md"

R75 always sits beside R82. R84's corrections apply to R75, R78 and R82 cells (e.g. the shipped model came from the tuner in 3 of 3 R78 races).

## 5. States
| State | Grid | Numbers | Chart | Ticker | Header |
|---|---|---|---|---|---|
| **Empty** (before the start) | all cells Off | AI bill $0.00 · Experiments finished 0 · Fakes caught 0 | axes and the goal line only, no line | "None so far…" | clock 00:00 / 15:00; mode tag already visible |
| **Running** | live colours | counting | step line grows, flash on a new best | rows appear | clock runs |
| **Finished** | all cells Off | final values (the "end" samples above), frozen | full line; final value labelled | last 3 rows stay | badge "Finished · 15:00"; the race's `race_finished.reason` if it isn't "budget" (e.g. "Stopped: cost cap reached") |

Mock frames to draw: Empty, Running at 5:00 (R78 s0 values), Finished (R78 s0), plus one Finished frame with R82 s0 values (footer with R75).

## 6. Never show
- **Jargon as a label:** adapter, val_bpb, held-out, RL, PPO, idle %, utilisation, sampler, trap, surrogate, fingerprint, judge, propose, Optuna, seed spread, arm. (Raw metric names only on hover.)
- **Unlogged numbers:** no projections, ratios, "x× cheaper", time-to-goal estimates or GPU use % without a real source. Numbers from the demo run are never presented as a result.
- **Mubit claims:** no "Served by Mubit", "Mubit's model answering" or any Mubit-arm number (no adapter has served a call, R70, R79). No "Learned from your correction".
- **Over a replay:** "Live", "right now", or the Modal panel.
- **"GPUs"** for R78 (CPU workers) or for R82 without "simulated".
- The no-AI tuner's column on the main screen (D46); it goes on a second slide or in Q&A.
- Any account e-mail, key, token or Modal settings page.
- A styling that tells viewers who wins before the numbers do.
