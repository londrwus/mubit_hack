# Final design exports

Source: `C:/Users/Lenovo/Documents/mubit.pen` (Pen.dev), exported with the pencil MCP `Export(..., "png", ..., {scale: 2})` on 2026-10-24. No nodes were modified. Each file is named `<nodeId>.png`.

Numbers below are checked against `docs/results_log.md`. The R-ids in brackets are the logged rows each number comes from.

## Landing page (frame `pOKAY`, "Landing v2 · Compute intelligence (A+)")

| File | Node id | Size (px) | What it shows | Logged numbers on it |
|---|---|---|---|---|
| `pOKAY.png` | `pOKAY` (whole frame) | 2377 × 8192 | The full landing page from the nav to the call to action. This is the image to use for the hero group: Nav + Hero + the hero engraving overlay (`QwFBT`) + the paper texture (`z1YMqM`) are separate children of `pOKAY`, so only this export shows them together. **Note:** the page is taller than the 8192 px export limit, so this file was scaled down to about 1.65× instead of 2×. | Everything listed in the section rows below |
| `KekD1.png` | `KekD1` (Hero) | 2880 × 2202 | Hero: "Your GPUs should never wait", "Same GPUs, more science", "Built for AI labs", "Compute intelligence · runs on Mubit". Exported alone, it does **not** include the nav (`fYbWi`) or the engraving overlay (`QwFBT`), which sit outside this frame. | None |
| `QYavo.png` | `QYavo` (S2 · The engine) | 2880 × 1792 | "I · The engine: Same model. We changed when it thinks." Waiting share, standard agent vs Busy Machines, with a busy/waiting strip, field note and three steps (plans less often, fills idle machines, checks every gain). | **R78**: 69% vs 2% waiting (logged 68.8% vs 1.5%, rounded), 30 machines each |
| `MgmgS.png` | `MgmgS` (S3 · Open model) | 2880 × 1876 | "II · Our model": sapling-research vs Mubit's published Banyan Sapling, Hugging Face / results links, field note. | **R81**: hallucinated wins caught 0 vs 57 of 109, none wrongly approved. Also cites R33, R42 (one H100, about $1 of compute) and R54 (99.9% vs 88.7% on fresh data), which are outside the four requested ids |
| `oT6Ad.png` | `oT6Ad` (S4 · Benchmarks) | 2880 × 2307 | "III · Benchmarks": Opus 5.5 lab vs Busy Machines, 3 runs each, with bar rows and the machines-busy figure (game-playing sweep, 30 machines). | **R78**: AI bill $3.30 vs $0.83; total cost $3.65 vs $1.18; experiments per $ 137 vs 1,376; model quality penalty 154.5 vs 154.3 ("same"); machines busy 31.2% vs 98.5%. **R87**: time to reach the goal 1.0 vs 0.3 min (R87's goal framing of R78's runs). **R82**: tabular-task AI bill $2.77 vs $0.70, quality within the set margin |
| `c50WCZ.png` | `c50WCZ` (S6 · Call to action) | 2880 × 1800 | Closing call to action: "Same model. Busy machines." with Try / demo / GitHub buttons and the footer "Every number comes from our public results log". | None |

Not exported on their own: `fYbWi` (Nav), `pE4fB` (ornament divider, no text), `z1YMqM` (paper texture) and `QwFBT` (hero engraving). These are decoration or chrome and appear only in `pOKAY.png`.

## Platform screens

| File | Node id | Size (px) | What it shows | Logged numbers on it |
|---|---|---|---|---|
| `bxWh3.png` | `bxWh3` ("Platform · Overview") | 2880 × 1920 | Product dashboard overview: sidebar, KPI cards, a machines-over-time block chart, where the AI bill goes, and model quality over time. | **R78**: AI bill saved 75% ($0.83 vs $3.30 per sweep); machines busy 98.5% vs 31.2%; experiments 1,587 vs 500 in 15 min; hallucinated wins 0 of 847 accepted; quality "same quality at a 4× lower AI bill"; run 1 final −154.9 vs −155.0 and AI bill $0.70 vs $3.30 (labelled "R78 · run 1": per-seed values, not the 3-seed mean). **R82**: AI bill breakdown $0.698 per sweep vs $2.774 for the standard agent; planning $0.552; checking $0.146 (derived as $0.698 − $0.552); 79% / 21% split. The block layout is labelled illustrative |
| `eGVso.png` | `eGVso` ("Platform · Sweep detail (live compare)") | 2880 × 1920 | Live compare of a game-playing sweep (run 1 of 3, replayed at 30×), standard Opus 5.5 agent vs Busy Machines side by side: a 30-machine grid, AI bill, machines waiting, experiments, model quality curve, Modal stats and the hallucinated-win check flow. | **R78 run 1** (per-seed values): AI bill $3.30 vs $0.70; machines waiting 69.0% vs 1.4%; experiments 497 vs 1,594; best quality −155.0 vs −154.9; goal −170; 30 machines, 15 min, 15 s runs. The machine-use sparkline shape is labelled illustrative |

## Caveats

- **R87** appears only on `oT6Ad.png` (and therefore on `pOKAY.png`), as "time to reach the goal". The results log says to quote it with R87's caveat (from R48): the −170 goal is easy with 30 machines, and the no-AI tool reached it cheapest (0.4 min, $0.009; Busy Machines was faster at 0.3 min) and most cheaply. That caveat is not on the design. The benchmarks figure's caption says the time panel uses the game task's fixed, easy goal.
- The per-seed run-1 values on `bxWh3` / `eGVso` (497, 1,594, 69.0%, 1.4%, $0.70, −155.0, −154.9) are logged in **R90** (from the stored race data, checked against the stored summaries in R88). The headline numbers remain the 3-run means (R78).
- `bxWh3` mixes R78 headline values with an R82 cost breakdown. Each one carries its own R-id label on the card.

## Minimal platform screens (added 2026-10-24, after the user asked for less on screen)

Two simpler frames next to `bxWh3` / `eGVso`, same sidebar, card and button style. Sidebar has only Get started, Compare and Docs, plus the workspace switcher and user row.

| File | Node id | Size (px) | What it shows | Logged numbers on it |
|---|---|---|---|---|
| `P6OjMG.png` | `P6OjMG` ("Platform · Get started") | 2880 × 1920 | "Start in three steps": (1) Add your Mubit key (masked `MUBIT_API_KEY` input, "Every AI call runs through Mubit"), (2) Connect your compute (Modal token chip, "Modal today; any runner"), (3) Start a sweep, with a code block `POST /api/races {"preset": "rl-wide-bm-cadence-15m"}`, the CLI form `python scripts/bench.py run rl-wide-bm-cadence-15m --yes`, and "Try it free: replay a logged sweep →" | None. The preset chips (30 machines each, 15 minutes) are the preset's setup (R78's setup column) |
| `GKsLR.png` | `GKsLR` ("Platform · Compare") | 2880 × 1920 | "See how it compares": three example cards (game-playing sweep selected, tabular tuning, catching hallucinated wins), then the standard Opus 5.5 agent and Busy Machines side by side: a 30-machine grid, AI bill, machines waiting, experiments, model quality. "Replay · 90×" button and the caption "Replay of a logged sweep" | Cards: **R78** (same model, 4× cheaper AI bill, machines 98% busy from 1.5% waiting), **R82** (same model within noise, 4× cheaper AI bill), **R81** (ours caught 57 of 109 fakes, published model 0, unsure on all 152 ground-truth claims). Panel: **R90** (R78 run 1): AI bill $3.30 vs $0.70, waiting 69.0% vs 1.4%, experiments 497 vs 1,594, model quality −155.0 vs −154.9. The "4× cheaper, mean of 3 runs" chip is R78's mean (0.25×). The grid's "9 of 30 / 30 of 30 training" is a snapshot from `eGVso`, not a logged number |

Caveat: the backend's `POST /api/races` takes a full `RaceConfig`, not `{"preset": ...}`. The code block on `P6OjMG` needs a preset shortcut on that endpoint, or the CLI line, to be literally true.

## Landing v3 · Cost, time, score (reworked 2026-10-25)

Frame `W5R8ch`, renamed from "Landing v3 · Three problems" (v2 `pOKAY` left unchanged). Founder feedback: content repeated from screen to screen. Each section now has one job and no number repeats in consecutive sections.

Section order: Hero → ornament → **I How it works** (`GqXzU`, new 2026-10-25) → **II Cost** (`bVVJQ`, trimmed) → **III Time** (`JacVJ` + `T3KBM`) → **IV Model score** (`KIak1`) → **V Proof · Benchmarks** (`Ju1Te`) → CTA (`y4d9W`). Nav links: HOW IT WORKS (`SL2Ne`, new) · COST · TIME · MODEL SCORE · BENCHMARK.

Update 2026-10-25 (founder feedback round 2): added an intro screen that says what the product is (compute intelligence) before any saving is pitched, and cut the Cost screen by about 40%. Sections III-V were only renumbered.

| File | Node id | Size (px) | What it shows | Logged numbers on it |
|---|---|---|---|---|
| `W5R8ch.png` | `W5R8ch` ("Landing v3 · Cost, time, score") | 1483 × 8192 (scaled to fit the 8192 px limit) | The whole reworked v3 landing | As per the rows below |
| `GqXzU.png` | `GqXzU` (S0 · How it works, eyebrow "I · HOW IT WORKS · COMPUTE INTELLIGENCE") | 2880 × 1626 | **New.** "A compute layer for training sweeps." Sub-line: "Point it at your training code and your machines. It plans, runs and checks the experiments." Flow `YNyBI`: 01 IN *Your code + machines* ("Any training code. Modal today, runners pluggable.") → 02 BUSY MACHINES ENGINE: PLAN *AI planner* "Proposes the next experiments to try." [RUNS ON MUBIT] · RUN *Scheduler* "Keeps every machine busy, never waiting." · CHECK *Checker* "Re-tests every claimed win on unseen data." → 03 OUT *A better model* ("A lower bill. Nothing fake shipped."). Foot: "PLAN · RUN · CHECK — ON ANY MACHINES" | None (no numbers on this screen by design) |
| `bVVJQ.png` | `bVVJQ` (S1 · Cost, eyebrow "II · COST") | 2880 × 1926 | **Trimmed.** "A quarter of the AI bill. Built on Mubit." Lead: "AI thinking is most of a sweep's bill. We ask the AI only when it pays." Bill plate: 3 comparisons (AI bill game-playing, AI bill tabular, total AI + machines game-playing); planner-calls row removed. Where the saving comes from: two items (MEASURED · Fewer, better-timed AI calls; MUBIT · TRAINED, NONE SERVING YET · Small models we own). Formula row `t0s7C` and Mubit strip `GU1sA` deleted. One field note | **R78**: $3.30 vs $0.83; total $3.65 vs $1.18. **R82**: $2.77 vs $0.70. **R72**: list price, Mubit fee excluded. **R79/R80**: trained, none serving |
| `JacVJ.png` | `JacVJ` (S2a · Time · Machines waiting, eyebrow "III · TIME · MACHINES WAITING ON THE AI") | 2880 × 1696 | "Same model. We changed when it thinks." Busy/waiting strips (the only place the waiting share appears), field note, three steps (plans less often, fills idle machines, checks every gain). Background now matches S2b so the two read as one section | **R78**: 69% vs 2% waiting (logged 68.8% vs 1.5%), 30 machines each |
| `T3KBM.png` | `T3KBM` (S2b · Time · What it turns into, eyebrow "III · TIME → RESULT") | 2880 × 2532 | "What busy machines turn into." Chain cut from four steps to three (same budget → experiments run → time to the goal); the "machines busy 31.2% vs 98.5%" step was **removed** (it repeated S2a). Best-score-so-far chart unchanged | **R78**: 15 min, 30 machines; experiments 500 vs 1,587. **R87**: time to the goal 1.0 vs 0.3 min (easy-goal caveat in the field note). Chart: R78 run 1 series (race `4ca4ed27`); 16.9 s vs 74.6 s (**R90**); final scores tie over 3 runs (−154.5 vs −154.3, R78); no-AI tuner 0.4 min (R87) |
| `KIak1.png` | `KIak1` (S3 · Model score, eyebrow "IV · MODEL SCORE · OUR MODEL") | 2880 × 1876 | sapling-research vs Mubit's published Banyan Sapling; only the eyebrow changed | **R81**: 0 vs 57 of 109 caught, none wrongly approved. R33, R42 (one H100, about $1). R54 (99.9% vs 88.7%) |
| `Ju1Te.png` | `Ju1Te` (S4 · Benchmarks, eyebrow "V · PROOF · BENCHMARKS") | 2880 × 2307 | Unchanged figure (summary of all three); sub-line now "Cost, time and model score, measured. Same Opus 5.5 model, same machines, 3 runs each." | Same as `oT6Ad` (v2) |
| `m4npbu.png` | `m4npbu` (S1b · Three problems) | 2880 × 1662 | **Removed from the design** (node deleted); the section order now tells that story. File kept only as history | R78 (as before) |

The old v3 order (Three problems → Time · engine → Time → result → Model score → Cost) is superseded.
