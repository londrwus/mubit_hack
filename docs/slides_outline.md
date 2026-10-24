# Slides outline (8 slides, plus optional slide 7b)

Exact copy for each slide, the logged numbers that go on it (with R-ids for the speaker notes), and the worse result that has to sit on the same slide. Order follows D48: **better model and less money first**, GPUs waiting as the reason it works, and the no-AI baseline on a second results slide only. Plain words on the slides (D19); R-ids go in the speaker notes and a small footer.

`[R-pending: …]` = a result not logged yet (no race is pending since R82 logged the LCBench 30 s follow-up, D49; only the go-live card and B's v2 remain). Fill it from `results_log.md` once logged, or delete the line. Never fill it from the terminal or from memory. AI $ on every slide is a list-price token estimate (R72); tabular GPU $ is virtual; any Optuna number from a race with a Busy Machines arm carries the R74 caveat (notes).

---

## Slide 1: Hook
**Title:** Busy Machines
**Line:** The same model for a quarter of the AI bill, from the AI research assistant you already use.
**Small line under it:** Two tasks, 3 runs each: game-playing AI (R78) and tabular tuning (R82). Our first tabular race failed (slide 4).
**Condition (notes):** "a better model for less money" is **not** supported: R78 is a tie on quality, in R82 the standard assistant is ahead in every run (within 0.0013), R75 is worse, and R67 (better) cost 2.1× the AI bill. Use it only as "Built to get a better model for less money".
**Sub-line:** Built for AI labs' experiment loops: sweeps, ablations, tuning.
**Footer number:** "90.6% of a standard Opus 5.5 assistant's spend on our game-playing task was its own thinking." (R54: $10.01 of $11.06 over 3 runs)
**Beside it (notes):** On image recognition the share is 63.6% (R54). Tabular tuning's 30.8% uses simulated GPU $ (R54).

## Slide 2: The problem
**Title:** Research assistants are slow and expensive to think
**Bullets:**
- An AI assistant plans the next experiments while the machines wait.
- Planning 8 experiments takes Opus 5.5 about **13 seconds** (R48, p50 13.1 s); 1 takes about **3 seconds** (R49).
- With 30 machines and 15-second runs, a standard Opus 5.5 assistant left them **waiting 69% of the time** (R48, 69.0 ± 0.4%, 3 runs).
**Must be on the slide:** "With 3 GPUs and longer runs: 10.9% (R53); on tabular tuning: 4.6–5.0% (R38, R44)." Waiting is the mechanism, not the money: the GPU $ lost to waiting is small next to the AI bill (R54: 13.9× on the game-playing task).

## Slide 3: What Busy Machines is
**Title:** Same Opus 5.5. Our engine around it.
**Left column, "Standard AI assistant":** Opus 5.5 plans every experiment and checks every result.
**Right column, "Busy Machines":**
- **Shorter plans:** asks only for what changes (−71 to −73% output tokens per proposal, R62)
- **No idle machines:** a simple no-AI tuner fills machines while Opus thinks, and learns only from checked results (R64 gate, R65)
- **Opus at a steady pace:** at most one planning call every 60 s on game-playing (R78), 30 s on tabular (R82, D49; 90 s failed first, R75)
- **A checker for fake results:** Opus looks for leaks, a seed-spread rule looks for noise (R37)
- **Every call through Mubit:** each step can become a small, fast model once it passes Mubit's exam (none has passed yet: best so far 58.3%, bar 95%; R79, with 11.4% for the same data in another workspace, R70)
**Footer:** Same model, machines, budget, clock, seeds, starting code and judge prompt on both sides. Differences disclosed per race (D46).

## Slide 4: The result (the headline)
**Title:** Same model, a quarter of the AI bill, on two tasks. And the race we lost first.
**Top: game-playing AI, 3 runs, 30 machines per side, 15-second runs, 15 min (R78, headline-eligible)**
| | Standard assistant (Opus 5.5) | Busy Machines (Opus 5.5 + our engine) |
|---|---|---|
| Model quality (score, closer to 0 is better) | −154.5 | **−154.3** (a tie) |
| AI bill per race | $3.30 | **$0.83** (4.0× less) |
| Total per race | $3.65 | **$1.18** (3.1× less) |
| Cost to reach the standard assistant's final quality | $3.10 | **$0.46**, in 2 of 3 runs |
| GPUs waiting (why) | 68.8% | **1.5%** |
| Experiments per $ | 137 | **1,376** (10×) |
| Fake improvements accepted | 0 (of 10 shortcut runs) | **0** (of 847 shortcut runs) |
**Must be on the slide, same size as the table (worse side):** "A tie on quality, not a better model. In 1 of 3 runs a burst of checks raised its AI bill to 0.34×. It reached the standard assistant's quality in 2 of 3 runs. Most experiments came from the no-AI tuner inside Busy Machines."

**Middle: tabular tuning, pre-registered, 3 runs, 16 machines per side, planner every 30 s (R82, headline-eligible): success line MET**
| | Standard assistant (Opus 5.5) | Busy Machines (Opus 5.5 + our engine) |
|---|---|---|
| Model quality (score, higher is better) | **0.8897** | 0.8884 (within 0.0013) |
| AI bill per race | $2.77 | **$0.70** (4.0× less) |
| Total per race (GPU $ simulated) | $5.97 | **$3.89** (35% less) |
| Reached the standard assistant's final quality | n/a | **no** (0 of 3 runs) |
| GPUs waiting (why) | 30.3% | **1.1%** |
| Fake improvements shipped | 0 of 70 trap runs (R84) | **0** (of 85 shortcut runs) |
**Must be on the slide, same size as the table (worse side):** "Not a better model: the standard assistant was ahead in every run, by 0.0001 to 0.0030 (about 1% more errors on our side). It never reached the standard assistant's final quality, and to 0.86 it was slower and dearer ($1.14 / 4.2 min vs $1.08 / 2.9 min). **We chose this setting after the race below failed**, and froze it before running it."

**Bottom: tabular tuning, the first pre-registered race, planner every 90 s, 3 runs (R75): FAILED**
"AI bill 8.2× lower ($0.347 vs $2.831), but **7% more errors** (0.8816 vs 0.8892), in every run. We report it anyway." Success line: AI bill ≤0.8× ✅ · better than a no-AI tool ✅ · model ≥ standard assistant ❌ · standard assistant waiting ≥40% ❌ (29.4%) · Busy Machines waiting ≤2% ❌ (2.5%).
**Rule:** the middle and bottom blocks are never split across slides; R82 is never shown without R75 and "chosen after".
**Footer:** "Same model, machines, clock, $ cap, seeds, starting code and judge prompt on both sides. AI $ at list price. Total = AI + compute used inside the race. Game-playing on real cloud CPU workers; tabular GPU time simulated (runs at 0.1× a published benchmark's times)." (R78, R82, R75, D46, D47, D49, `preregistration_lcbench_wide.md`, `preregistration_lcbench_wide_c30.md`)
**Notes (Optuna column, small type or notes only):** R78: best −157.3, total $0.35, reached the easy −170 goal cheapest ($0.009 / 0.4 min vs BM $0.022 / 0.3 min vs Opus $0.20 / 1.0 min), 53 of 4,648 trap runs accepted as best. R82: 0.8576 ± 0.018, lowest total ($3.20, virtual GPU), 15 of 236 trap runs accepted as best, shipped a fake in 1 of 3 runs. R75: 0.8493, 19 fakes of 618 trap runs. R74 caveat on all three. R82 logs fakes for Busy Machines and the tool only, so the standard assistant's R82 cell stays "—" until a logged number exists. R82 pass is on 3-run means; run 3 alone sits at the margin (−0.0030 vs the −0.003 line). Total $ counts in-race GPU only (Modal metered ~$2.5 per race vs $1.05 in-race, same for all arms, R78).

## Slide 5: Each part, tested on its own (pilots, not headline races)
**Title:** Each part, tested on its own
| Part | What we measured | The worse side (same row) |
|---|---|---|
| Shorter plans | Model 0.866 vs 0.849, AI $ per experiment −10.5% (3 runs, R63) | Dev race; "at least as good", not a proven gain (pooled baseline 0.854) |
| No idle machines (tabular) | Goal in 1.2 vs 2.5 min, $0.36 vs $0.74; GPUs waiting 0.0% vs 4.0% (2 runs, R65) | Best model slightly lower (0.859 vs 0.861), AI bill 1.19× |
| No idle machines, no pace limit (game-playing AI) | Best model −153.9 vs −156.2; goal in 0.5 vs 1.1 min, $0.11 vs $0.20; GPUs waiting 2.0% vs 66.0% (2 runs, R67) | **AI bill 2.1× higher** ($6.50 vs $3.07) with no pace limit, which is why the pace limit exists |
| Opus at a steady pace | With it, on the same game-playing setup: AI bill 0.25× ($0.83 vs $3.30), 3 runs (R78, slide 4). On tabular at 30 s: AI bill 0.25× ($0.70 vs $2.77), within 0.0013 (R82) | The model went from better (R67) to a tie (R78); different races, not pooled. On tabular the 90 s pace left only 80 AI proposals and the model was 7% worse (R75); the 30 s pace (240 AI proposals, chosen after R75) still left the standard assistant ahead in every run (R82). Not tested on its own (always with the other parts) |
| Checker | Real improvements confirmed 7/7 vs Opus alone 1/7; fakes through 0/32 vs 0/32 (R37) | Accepts 1/36 noise vs 0/36; offline, small set |
**Footer:** "An earlier engine version, with an early-stopping rule we removed, made the tabular model worse: 9% more errors (R44, D43)."

## Slide 6: Fake improvements
**Title:** A better number only counts if it's real
**Line:** The checker confirmed **7 of 7** real improvements; Opus alone confirmed **1**. Neither let a fake through: **0 of 32** (R37, from R34).
**Held-out test split (closed-output checker):** real 4/4, noise 1/20, fakes through 0/16 (R40).
**Must be on the slide:** "It accepted 1 of 36 lucky results as real; Opus accepted 0 (R37). Only 7 real cases, tested offline on stored verdicts. The seed-spread rule alone would let 24 of 32 fakes through (R34), which is why Opus still does the leak check."

## Slide 7: Versus a standard no-AI tuning tool (second results slide)
**Title:** Even the standard assistant beats a no-AI tuning tool on quality, on 2 of 3 tasks
| Task (3 runs each) | Standard Opus assistant vs the tool | The tool's side |
|---|---|---|
| Tabular tuning | 28% fewer errors (0.856 vs 0.799, R38; repeated in R44) | More experiments per $ (80.9 vs 22.7); faster and cheaper to the goal in 1 of 3 runs (R38) |
| Image recognition | 22% fewer errors (0.9448 vs 0.9294); goal reached 3/3 vs 0/3 (R53) | Cheaper per race ($0.60 vs $1.65), more experiments per $ (81.2 vs 38.7) |
| Game-playing AI | A tie on the best model (−156.9 vs −155.2, within noise); shipped model −156.9 vs −177.1 (R48) | Its best model is slightly better (−155.2); goal reached sooner and cheaper ($0.007 / 0.3 min vs $0.222 / 1.1 min); 4,585 vs 137 experiments per $ |
**Line:** Unchecked, the tool accepted fakes as its best **59 times** in 9 races and shipped a fake in **6 of 9**; the assistant **0** (R54).
**Must be on the slide:** "The tool has no checker by design. On image recognition the assistant never tried the shortcut, so its checker wasn't tested there (R53)."

## Slide 7b (optional, before the close): Our fine-tune of Banyan Sapling
Show it if the timing allows, or as the first backup slide in Q&A. It keeps the number "Slide 8" for the close, so the `demo_script.md` timings stay valid. Full sources and caveats: `sapling_vs_published.md`.
**Title:** We didn't just reuse Mubit's checker. We fine-tuned it for AI research.
**Line:** "We fine-tuned Banyan Sapling's verification head, backbone frozen. It is our community fine-tune, not an official MubitLabs model."
| | Published Banyan Sapling | Our fine-tune (strict / looser) | Opus 5.5 |
|---|---|---|---|
| Real claims from our races: fakes caught (of 109, R81) | 0 | 8 / 57 | 109 |
| Real claims: fakes approved | 0 | 0 / 0 | 0 |
| Real claims: no answer (of 152) | 152 | 105 / 24 | never abstains |
| 500 synthetic claims: answers given | 0% | 19% / 64% | not run |
| 500 synthetic claims: fakes caught (of 157) | 0 | 52 / 118 | not run |
| Fakes approved (of 157) | 0 | 0 / 0 | not run |
| Trap set: correct (of 50) | 10 | 18 / 36 | **50** |
| Time per check | ~0.1 s | ~0.1 s | ~1.7 s |
**Must be on the slide, same size as the table (worse side):** "Opus is still the better judge: it scores 50 of 50 on our trap set, against 18 or 36 for ours. None of the three confirms a real gain on its own: ours confirms 0 of 132 at the strict setting. At the looser setting its 95% guarantee falls to 78% on new claim types. The 500 claims are synthetic, from our own generator. A later version trained on race claims failed its gate, so we kept this one."
**Sources (notes):** the 500-claim and trap-set rows are R42; Opus 50/50 is R24; time per check is R76 (99.4 ms) and R40 (Opus judge v1 p50 1.65–1.70 s); the v3 fail is R77. The real-claims rows are R81 (`data/sapling/compare_published/compare_published.json`): 226 judge-format claims, 152 with ground truth. Caveats for that block: 186 of the rows trained the failed v3; 77 carry labels taught by Opus, so Opus is graded on the 152 ground-truth rows only; there are only 7 real gains. ContractNLI goes in the notes only: 77.3% → 84.3% on our own 400 cases (R42). It is not a reproduction of the card's 86.32%.

## Slide 8: Mubit, and the close
**Title:** Every check we record becomes training for the next model
(Not "every check": race calls are record off and never become examples, R72.)
**Bullets (pick one version at 05:00 Sun, per `demo_script.md` §0, D53):**
- PRIMARY: "Mubit's own checking model now answers: it passed at [score, R-pending: go-live card] and shows as **Served by: Mubit model**."
- FALLBACK: "Mubit trained a checking model from 177 of the assistant's checks (R39). Today it matches Opus on 11.4% of its 35 test cases, against a 95% bar (R70), so Opus keeps answering until it passes." (If asked: the same data in workspace C scored 58.3% on 24 cases, R79.)
- Both: "Corrections are stored against their Mubit request ID. After 25 new examples, Mubit trains a new version (R58); 40 are in (R71)." [If B's v2 exists: "v2: [score, R-pending]".]
- Both, small (only if slide 7b was skipped): "We fine-tuned Banyan Sapling's verification head, backbone frozen. It catches 52 of 157 fakes the published model caught none of, and approves none (R42). Opus is still the better judge: 50/50 vs 18/50 on our trap set. At that strict setting it confirms none of the genuine gains, and the claims are synthetic."
**Close line:** Same model. Same machines. A quarter of the AI bill. (R78, R82; never "a better model for less money", see slide 1.)

---

## Numbers that must never appear on a slide
- "Mubit beats Opus", or any Mubit-arm number (no adapter served, R46, R70; `headline.py` gate D41).
- Haiku's R50 numbers as Mubit's model or as Busy Machines.
- R66, R64, R52, R43 or R27 projections, or R78's gate projection, unless labelled "projection" (Q&A only).
- R78's or R82's "4×" (or R78's "3.1×", R82's "35%") without the task and the R75 fail on the same slide; "a better model" for R78 (a tie) or R82 (the standard assistant ahead in every run); "$0.46 vs $3.10" without "in 2 of 3 runs".
- R82 without "chosen after R75 failed"; R82 as reaching the standard assistant's quality (0/3) or as faster to 0.86 (it was slower and dearer).
- R36, R22, R33, R35 (superseded), or R41's Busy Machines arm (broken rule).
- R38 + R44 pooled as 6 runs.
- Per-case gains from the noise set (not separately logged).
- On slide 7b: real-claims numbers that aren't in R81 (e.g. the per-source split); "matches Opus" or "beats Opus"; "confirms real gains"; our fine-tune called "Mubit's model" or official; the 77.3% → 84.3% ContractNLI figure as a reproduction of the card's 86.32%.
- Any number from the live demo race as a result (fingerprint differs, never logged as a comparison).
