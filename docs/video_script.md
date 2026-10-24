# Video script: the 3-minute demo video (pre-recorded)

The demo is a **3:00 pre-recorded video**, not a live run. This file is the script, the shot list, the commands for every terminal visual, the "never say" list and a 20-second backup cut. It reuses the framing and number rules of `demo_script.md` (D46, D48), the slide copy of `slides_outline.md`, the IDs in `evidence_pack.md` and the Q&A answers in `qa_mubit_constraints.md`.

**Rules that hold for every frame**
- **Only logged numbers go on screen**, each with its R-id as a small caption. A number that is not in `results_log.md` is not shown, said or captioned. `[R-new: …]` marks a result that must be **logged as a new R-row by the main session before the final edit**. If it isn't logged, use the fallback for that beat.
- **The worse result sits beside each win, in the same caption or sentence.** Put R75 (failed) beside R82 every time. Call R78 "a tie". For R82, say the standard agent was ahead in every run.
- **D48 order:** same model and less money first. GPUs waiting is *why it works*, never the headline.
- **AI $ is a list-price estimate. Tabular GPU time is simulated** (virtual GPU $). Say so in the caption footer whenever a $ figure is on screen.
- **Labels in plain language:** "AI bill", "model quality", "GPUs waiting", "fake improvement caught", "small model". No "adapter", "val", "RL", "sampler", "trap" or "idle %" as a main label.
- **Recordings are labelled as recordings.** The race replay says "replayed from a logged race, 30× speed". The Modal dashboards say **"demo run, not a benchmark result"**, top-right, for every second they're on screen.
- No adapter has served a call (R70, R79). So never say "Mubit beats Opus", and never give a Mubit-arm number.
- **Workspace C's bar (D52):** if C's card or settings are shown with the bar at 55%, say "we lowered the bar to 55% in that one test workspace". Never say "passed Mubit's exam". No R-row yet logs that the bar was changed or that C went live. Show it only once that row exists.

**Word budget:** 418 spoken words over 3:00 with the 2-word Try-models placeholder, 423–426 with the real line (about 141–142 words a minute; at 145 words a minute 426 words take 2:56). Beat timings below are proportional to the words in each beat.

---

## (a) Timed script

### Beat 1 · 0:00–0:16 · The problem (money)
| | |
|---|---|
| **Narration (38 words)** | "AI labs now let agents run their experiments, and the agent is the expensive part. On our game-playing task, 90.6% of a standard Opus 5.5 agent's spend was its own thinking. And while it thinks, the machines wait." |
| **Visual** | Clip **C01**: the replay TUI on race `4ca4ed27` (R78 seed 0) at its first frame, both sides' machine slots and AI bill counters at $0. Let it start running from ~0:06 |
| **Caption** | "90.6% of a standard Opus 5.5 agent's spend was AI calls: $10.01 of $11.06 over 3 earlier game-playing runs (R48 races, R54)" · top-right tag: "Replayed from a logged race (R78) · 30× speed" |
| **Fallback** | Replay TUI not ready: clip **C01-alt** (`backend.replay` text timeline, same race, `--play --speed 30`), same caption. Terminal unusable: chart 2 (`2-gpus-waiting-r78-dark.png`) as a still, with a slow zoom |

### Beat 2 · 0:16–0:40 · What Busy Machines is (same model, our engine)
| | |
|---|---|
| **Narration (57 words)** | "This is Busy Machines, replayed from a logged race. Both sides run the same Opus 5.5, on the same machines, budget, clock and starting code. On the right, our engine: shorter plans, Opus at a steady pace, a simple tuner that keeps machines busy in between, and a checker for fake results. Every call goes through Mubit." |
| **Visual** | C01 continues: the 15-minute race plays in 30 s, ending ~0:34. **Hold the last frame to 0:40**, showing both final AI bill counters and model quality |
| **Caption** | "Same model, machines, clock, $ cap, seeds, starting code; only the engine differs (D46)" → at the end frame: "One run of 3. Counters are from this race's log; 3-run means are on the next screen (R78)" |
| **Fallback** | As in beat 1. If the TUI's end frame shows any label in jargon, crop it, or hold on chart 1's left panel instead |

### Beat 3 · 0:40–1:11 · The result: same model, a quarter of the AI bill, and the race we lost first
| | |
|---|---|
| **Narration (71 words)** | "The result, three runs per task. Game-playing: a tie on model quality, for a quarter of the AI bill, 83 cents against $3.30 a race. Tabular tuning: also a quarter of the bill, within 0.0013, but the standard agent was ahead in every run. And the first pre-registered setting failed: eight times cheaper, 7% more errors. We picked the faster pace after that, froze it before racing, and we report both." |
| **Visual** | `data/charts/final/1-headline-two-tasks-dark.png` full screen, 0:40–1:03. Slow push-in on the left panel ("Game-playing"), then the middle, then the right (R75, red). At ~1:03 cut to `3-quality-over-time-r82-dark.png` ("keeps pace, not ahead") until 1:11 |
| **Captions (in order, ≥4 s each)** | 1. "Game-playing: model quality −154.3 vs −154.5, a tie · AI bill $0.83 vs $3.30 per race (R78)" · 2. "Tabular, AI planner at most every 30 s: 0.8884 vs 0.8897; standard agent ahead in all 3 runs · AI bill $0.70 vs $2.77 (R82)" · 3. "**First pre-registered setting FAILED:** AI bill $0.35 vs $2.83, 7% more errors, 0.8816 vs 0.8892 (R75). R82's pace was chosen after it" · 4. (on chart 3) "One run of 3 shown (R82 seed 1). Standard agent ahead in every run" · footer throughout: "Busy Machines first, standard Opus agent second · AI $ at list price · tabular machine time simulated" |
| **Fallback** | None needed: the charts exist. If the charts are regenerated (`final_charts.py`), check that their numbers still match R75/R78/R82 |

### Beat 4 · 1:11–1:33 · Why it works: the waiting
| | |
|---|---|
| **Narration (51 words)** | "Why it works: in the game-playing race the standard agent's machines sat waiting 68.8% of the time while Opus planned. Busy Machines' waited 1.5%. With longer runs the standard agent waits less: 10.9% in our image races. Here is a five-minute demo on real cloud GPUs, one Modal app per side." |
| **Visual** | 1:11–1:24: `2-gpus-waiting-r78-dark.png`. 1:24–1:33: clip **C04**, the two Modal app dashboards side by side, left `busy-machines-opus`, right `busy-machines-bm` (Apps → app → containers / GPU view), from the `demo-cifar-modal-5m` run. Sped up in the edit (8×), and say so in the tag |
| **Captions** | "Machines waiting (here 30 cloud CPU workers per side): 68.8% vs 1.5%, game-playing, 3-run means (R78) · chart: one run of 3 (seed 1)" → "Longer image-recognition runs: the standard agent waited 10.9%, 3-run mean; Busy Machines not raced on this task (R53)" → on C04, top-right, fixed: "**Demo run, not a benchmark result** · image recognition, 3 L4 per side, 5 min · sped up 8×" |
| **Fallback** | Demo race not run, or the dashboards show nothing readable: keep chart 2 for the whole beat, and replace the last sentence with "Here is one of those races, minute by minute." (same word count). Never describe what the dashboards "show" beyond what is visibly on screen. The demo run's numbers are never read out |

### Beat 5 · 1:33–1:47 · Fake improvement caught
| | |
|---|---|
| **Narration (34 words)** | "A better number only counts if it's real. Busy Machines' side tried a fake shortcut 847 times, mostly from its tuner. The checker let none through. One burst of checks did cost 56 cents." |
| **Visual** | Clip **C02**: replay TUI on race `145c2a2d` (R78 seed 2), the stretch where "Fake improvement caught" fires repeatedly and Busy Machines' AI bill counter climbs (the burst). About 14 s cut from the 30 s replay |
| **Caption** | "Fake shortcut runs accepted as the best model: 0 of 847 on Busy Machines' side, 3 runs (R78)" → "That run's burst: 143 checks, $0.56, that run's AI bill 0.34× of the standard agent's, vs 0.21× in the other two runs (R78)" · tag: "Replayed from a logged race · 30× speed" |
| **Fallback** | TUI not ready: C02-alt (`backend.replay 145c2a2d … --play`) with the same captions. Or slide 6's checker line as a still: "Real improvements confirmed 7 of 7 vs Opus alone 1 of 7; fakes through 0 of 32, offline on stored verdicts (R37). It accepted 1 of 36 lucky results; Opus 0 (R34)" (and change the narration to match these R37 words) |

### Beat 6 · 1:47–2:23 · Mubit: a correction, and how the model updates (rule #3)
| | |
|---|---|
| **Narration (84 words incl. the 2-word placeholder, FALLBACK wording: no v2, nothing served)** | "Mubit turns the agent's recurring steps into small models. Here's one check where Opus missed: it called a real improvement 'unsure'. Ten re-runs say it's real. We store the correction with the case's Mubit request id, and rate this reply Bad. That hasn't changed a model yet; new examples do. After 25, Mubit trains a new version. Our best small checker matches Opus on 58.3% of its test cases; the bar is 95%, so Opus keeps answering. On this held-out fake, it [Try-models line]." |
| **Visual, in order** | (1) 1:47–1:51 console project A → **Tasks** (the agent's steps as Mubit tasks) · (2) 1:51–1:58 workspace B → **Requests** → `mreq_01a16c07cc1d7c70ae96381f78916256`, Opus's reply `unsure · insufficient_evidence` · (3) 1:58–2:03 click **Rate this reply: Bad**, with a 3 s inset of clip **C05** (the flag `curl` returning the stored correction row, `sent_to_mubit_at: null`) · (4) 2:03–2:09 B → Tasks → judge v1 `ln_5d9ea82de398d3c492ec873f` → Adapters → **History** · (5) 2:09–2:16 workspace C → judge v1 `ln_dfe18721334e06d23c32f338` → Adapters card (`ad_lane_52d6d485`, 58.3%) · (6) 2:16–2:23 **Try models**: the adapter's answer next to Opus's on `trap_v1_020` |
| **Captions** | (1) "The agent's steps as Mubit tasks: planning, checking, debugging (R29, R46)" · (2) "Opus: 'unsure'. Ten re-runs: a real improvement (R34, R39)" · (3) "Correction stored with the original race call's Mubit request id (same case, mreq_01a16b31…, R34); this re-recorded reply (R39) rated Bad. Mubit has no corrections API, so this hasn't changed a model" · (4) "New version after 25 new examples (R58); 40 more recorded (R71)" · (5) "Best small checker: 58.3% match with Opus on 24 test cases; bar 95% (R79). Same data in another workspace: 11.4% of 35 (R70)" · (6) "One held-out case, not a score. Opus: 'invalid' (R40). The truth: it was tested on the levels it trained on · small model: as shown [R-new: Try-models answers, all 3 cases]" |
| **Variant lines (swap in only with the R-row logged)** | **B v2 exists:** replace "After 25, Mubit trains a new version." with "After 25, Mubit trains a new version: here's v2, trained on 40 more checks." Caption (4): "v1 → v2, v2 trained on 40 more checks: [v2 score and test-case count, R-new]". Never say v2 learned from the correction. **C's bar lowered (D52), card shows it:** add "We lowered the bar to 55% in that one test workspace" and caption "(bar lowered to 55% by us, workspace C only, D52) [R-new]". **C serves calls (Served by: Mubit model, graded):** use `demo_script.md` §4.4 wording, with "after we lowered the bar to 55% and it cleared that", and caption the served-call grade [R-new] |
| **Try-models line** | Same verdict as Opus: "gives the same answer as Opus: invalid." · Different: "gets it wrong, which is why it doesn't serve yet." · Not recorded: drop the sentence and hold shot (5) for 7 s more. Show `trap_v1_020` whatever the answer is. Don't swap in another case because it looks better: record all 3 prepared cases and log all three |
| **Fallback** | Console signed out or Mubit down during capture: use the console stills captured in step 1 of the shot list, with the same captions. B's history shows only v1: keep the FALLBACK wording ("trains a new version" plus "40 more recorded"). Don't say "queued" |

### Beat 7 · 2:23–2:44 · Our fine-tune of Mubit's verifier
| | |
|---|---|
| **Narration (48 words)** | "We also fine-tuned Mubit's open verifier, Banyan Sapling, for research claims. On real claims from our races, the published model caught none of 109 fakes; ours caught 57 and approved none, in a tenth of a second each. Opus is still the better judge: it caught all 109." |
| **Visual** | 2:23–2:37 `4-sapling-real-claims-dark.png` (pan from "Fakes caught" to "Real improvements confirmed: 0 for all four"). 2:37–2:44 clip **C08**: the Hugging Face model page **if published**, else the model card file (`data/sapling/sapling_v2/hf_release/README.md`, Markdown preview, scrolled from "# sapling-research v2" to "## Where it loses") |
| **Captions** | "152 real race claims with known answers: fakes caught 0 / 57 / 109 of 109 (published Sapling / ours, looser setting / Opus); fakes approved 0 for all (R81)" → "Real improvements confirmed: 0 of 7 for all, Opus included (R81)" → "~0.1 s per check on an H100 (R76) vs ~1.7 s for Opus (R81)" → on C08: "A community fine-tune, not an official MubitLabs model" + (file version) "Model card file · not published" |
| **Fallback** | HF not published: show the file (as above), never a link that doesn't exist. Editing time short: drop C08 and hold the chart |

### Beat 8 · 2:44–3:00 · Close
| | |
|---|---|
| **Narration (35 words)** | "Same Opus, same machines, a quarter of the AI bill, on two tasks. Not a better model yet: one tie, one slightly behind. And every check we record can become training for Mubit's next version." |
| **Visual** | End card (dark, large type): "Busy Machines" / "Same Opus 5.5. Same machines. A quarter of the AI bill." / small: "Game-playing: a tie (R78) · Tabular: slightly behind in every run (R82), after a failed first setting (R75) · every number in docs/results_log.md" / repo URL `github.com/londrwus/mubit_hack` |
| **Caption** | On the card itself (above). Last 1 s silent |
| **Fallback** | None needed |

**Spoken total:** 38 + 57 + 71 + 51 + 34 + 84 + 48 + 35 = **418 words** (`wc -w`, FALLBACK wording, counting `[Try-models line]` as 2 words); the real Try-models line makes it 423 ("same answer") or 426 ("gets it wrong"), 2:55–2:56 at 145 words a minute. The v2 variant adds 7 words and the D52 line 11: both together reach ~444 words (~3:04 at 145), so with both variants beat 2 must be shortened. If a variant pushes a take past 3:00, shorten beat 2's list of engine parts ("shorter plans, a steady pace, a tuner in between, and a checker"). Never cut a worse-result line or a "replayed" or "demo run" label.

---

## (b) Shot list and recording steps

### Global settings
- **OBS:** Settings → Video: base and output 1920×1080, 30 fps. Output: recording format MKV (remux to MP4 afterwards), encoder hardware H.264 (or x264 CRF 18), audio 48 kHz. Windows display scaling 100% on the capture monitor, so terminal text is sharp.
- **OBS scenes** (one scene per source; titles are added in the editor, not burned in during capture):
  | Scene | Sources |
  |---|---|
  | `S1 Terminal` | Window capture: Windows Terminal (maximised, 1920×1080) |
  | `S2 Browser` | Window capture: Chrome, "video" profile, 1920×1080 window |
  | `S3 Modal side by side` | Two window captures, Chrome window A (`busy-machines-opus`) left 960×1080 and window B (`busy-machines-bm`) right 960×1080; text source fixed top-right "Demo run, not a benchmark result" |
  | `S4 VO` | Microphone only (for the voice-over takes) |
  Charts are not captured: they're placed in the editor as stills (2000×1125 is exactly 16:9, so scale to 1920×1080 with no crop).
- **Terminal:** Windows Terminal, focus mode (no tabs or title bar), dark scheme, **Cascadia Mono 20 pt** (about 120 columns × 36 rows at 1920×1080). If the replay TUI wraps at 20 pt, go to 18 pt, never below 16 pt. Hide the cursor, and clear the scrollback before each take (`Clear-Host`).
- **Browser:** Chrome, a separate profile "video": bookmarks bar hidden, zoom **125%** for the Mubit console and Modal, 110% for Hugging Face, extensions off. Never open Mubit → Provider keys, Modal → Settings/tokens or `.env` on screen. Crop or blur any account e-mail in the top-right menus.
- **Captions (in the editor):** a lower third, white Segoe UI / Inter at 30 px on a 70% black bar, max 2 lines, on screen for ≥4 s. Tags top-right at 24 px. R-ids always inside the caption text.
- **Replay speed:** **30×** for every race replay (15-minute race → 30 s on screen). The Modal dashboards are sped up **8×** in the editor (the tag says so). Nothing else is sped up.

### Browser tabs (open in this order before the console capture; log in fresh: the console signs out after 15 minutes idle)
1. Mubit, project A → **Tasks** (propose, judge v0, diagnose, toy)
2. Mubit, workspace B → **Requests** → search `mreq_01a16c07cc1d7c70ae96381f78916256` (session `collect-judgev1-2026-10-24/noise_set/7`)
3. Mubit, workspace B → Tasks → judge v1 `ln_5d9ea82de398d3c492ec873f` → **Adapters** → **History**
4. Mubit, workspace C → Tasks → judge v1 `ln_dfe18721334e06d23c32f338` → **Adapters** (`ad_lane_52d6d485`); also Settings → Going live, but only if the bar change is logged
5. Mubit, workspace C → **Try models** with adapter `ad_lane_52d6d485`, system prompt from `data/adapter_golive/try_models/system_prompt.judge_improvement.v1.txt`, user message from `fake_caught.trap_v1_020.user.txt` (then `real_gain.trap_v1_024.user.txt`, `noise.trap_v1_048.user.txt`)
6. Modal (account `emeraldshuorbit`) → Apps → `busy-machines-opus` (window A)
7. Modal → Apps → `busy-machines-bm` (window B)
8. Hugging Face model page (only if published), or VS Code with the model card file in Markdown preview

### Capture order (fits D53: freeze and submit at 06:00 Sun, the version decision at 05:00)
| # | Clip | Scene | What to do | Raw length | Used |
|---|---|---|---|---|---|
| 0 | **VO guide take** | S4 | Read the script once at pace, against a stopwatch; adjust words, not speed, if a beat runs long | ~3:10 | timing only |
| 1 | **Console stills (insurance)** | S2 | Screenshot tabs 1–4 before anything is clicked (Win+Shift+S, save to the scratchpad, not the repo) | ~2 min | fallback for beat 6 |
| 2 | **C01** replay, R78 seed 0 | S1 | Command C01 (section c). Start recording, wait 2 s, press Enter, record until 5 s after the end frame | ~40 s | beats 1–2 (40 s) |
| 3 | **C02** replay, R78 seed 2 | S1 | Command C02. Same procedure | ~40 s | beat 5 (14 s) |
| 4 | **C03** spare replay, R82 seed 0 | S1 | Command C03. Not in the main cut. Spare for Q&A or a re-edit; if used, the R75 caption goes with it | ~40 s | spare |
| 5 | **C05** correction flag | S1 | Commands C05 (scratch DB, local dev server). Record the POST and the GET output | ~60 s | beat 6 inset (3 s) |
| 6 | Start the Try-models adapter | S2 | In tab 5 press start, **≥20 min before clip 8**. This uses 1 of 3 adapter starts per org per UTC day, and a cold start takes up to ~20 min | n/a | n/a |
| 7 | **C04** demo race dashboards | S3 | Deploy both apps, then run the demo race (commands C04). Start recording at "pre-warm", stop after scale-down. One take | ~8 min | beat 4 (9 s, at 8×) |
| 8 | **C06** console walk-through | S2 | Tabs 1 → 2 → **click Rate this reply: Bad once** → 3 → 4, slow mouse, ~3 s still on each page. **The rating can be done only once.** If the take fails after the click, retake with the C copy `mreq_01a16d547a3770d186869108a305e42e` and log both ratings | ~4 min | beat 6 (29 s) |
| 9 | **C07** Try models | S2 | Run `trap_v1_020`, then the other two cases. Record all three; screenshot each answer; the main session logs all three as an R-row before the edit | ~3 min | beat 6 (7 s) |
| 10 | Re-check at 05:00 | S2 | Re-open tabs 3 and 4. If B has a v2 or C's state changed (and it's logged), re-record only shots (4)/(5) of beat 6 | ~1 min | variant |
| 11 | **C08** model card | S2 / VS Code | HF page if published; else the README in Markdown preview, slow scroll | ~30 s | beat 7 (7 s) |
| 12 | **Final VO** | S4 | 3 takes; keep the best whole take (no splicing inside a number) | 3 × ~3:05 | full |
| 13 | Edit and export | n/a | Picture to VO; captions per section (a); export 1920×1080, 30 fps, H.264 ~12 Mbps, AAC 48 kHz, ≤ 3:00.0. File `busy-machines-demo-3min.mp4`. Also export the 20-second backup cut (section e) | n/a | n/a |

Before export, check every caption against `results_log.md` by R-id. Remove any caption whose `[R-new]` was not logged, and use that beat's fallback instead.

---

## (c) Commands for each terminal visual
All read-only on `data/bench/bench.db` unless marked. Run them from the repo root in PowerShell.

**C01 / C02 / C03: race replay TUI** (`scripts/replay_tui.py` is still being built. Check its flags once it lands; the form below is what the script should take, not yet verified):
```powershell
python -m uv run --extra ml python scripts/replay_tui.py --help
python -m uv run --extra ml python scripts/replay_tui.py 4ca4ed27 --speed 30 --db data/bench/bench.db   # C01: R78 seed 0
python -m uv run --extra ml python scripts/replay_tui.py 145c2a2d --speed 30 --db data/bench/bench.db   # C02: R78 seed 2 (checker burst)
python -m uv run --extra ml python scripts/replay_tui.py c1f4b1d9 --speed 30 --db data/bench/bench.db   # C03 spare: R82 seed 0 (show R75 beside it)
```
Check before recording: the TUI shows both sides' slots, the two AI bill counters, model quality and "Fake improvement caught" in plain words. It must not show `val_*`, "idle %", "sampler" or "trap" as a main label.

**C01-alt / C02-alt: text replay fallback** (`backend/replay.py` exists; flags checked in its argparse: `--db`, `--speed`, `--play`, `--limit`):
```powershell
python -m backend.replay --list --db data/bench/bench.db
python -m backend.replay 4ca4ed27 --speed 30 --play --limit 0 --db data/bench/bench.db
python -m backend.replay 145c2a2d --speed 30 --play --limit 0 --db data/bench/bench.db
```

**Optional still: the headline table** (if a terminal shot of the logged numbers is wanted instead of chart 1; it prints "NOT A HEADLINE RESULT" if a gate fails, and then it isn't used):
```powershell
python -m uv run --extra ml python scripts/headline.py --race 4ca4ed27 --race d4913e95 --race 145c2a2d   # R78
python -m uv run --extra ml python scripts/headline.py --race c1f4b1d9 --race 9d0cfd77 --race b3922d0e   # R82
python -m uv run --extra ml python scripts/headline.py --race 613e2537 --race 6afbbe8a --race c72b3af1   # R75, the fail
```

**C04: demo race on two Modal apps** (PAID: Modal GPU + Opus calls. Preset `demo-cifar-modal-5m` in `backend/benchmark/configs.py`: dev mode, labelled "demo run for the video, not a benchmark result", refused by `headline.py`; one Modal app per side via `PER_ARM_MODAL_APPS`. Being built, so confirm the deploy step with whoever builds it):
```powershell
$env:BUSY_MODAL_APP="busy-machines-opus"; python -m uv run modal deploy workers/runners/modal_runner.py
$env:BUSY_MODAL_APP="busy-machines-bm";   python -m uv run modal deploy workers/runners/modal_runner.py
Remove-Item Env:BUSY_MODAL_APP
python -m uv run --extra ml python scripts/bench.py check demo-cifar-modal-5m
python -m uv run --extra ml python scripts/bench.py run demo-cifar-modal-5m --seed 0 --db data/bench/demo.db --progress-s 10          # prints the plan and $ guard, no --yes
python -m uv run --extra ml python scripts/bench.py run demo-cifar-modal-5m --seed 0 --db data/bench/demo.db --progress-s 10 --yes    # the one take
python -m uv run modal app stop busy-machines-opus
python -m uv run modal app stop busy-machines-bm
```
- **Cost (estimate by arithmetic, not a result):** GPU ≤ 2 sides × 3 L4 × 5 min at $0.80/h ≈ $0.40, plus pre-warm and scale-down (the preset comment). Opus calls: the standard side spent ~$1.05 of AI per 15-minute image race (R53: $1.65 total − $0.60 GPU), so roughly $0.35 for 5 minutes; Busy Machines' side is not measured on this task. Check that the Modal demo reserve (≥$6, R73) covers it before `--yes`.
- Stop both apps afterwards, so idle containers aren't billed.
- The demo run is **never logged as a comparison** and its numbers are never read out. Log its cost as a demo cost row (main session).

**C05: the correction flag** (from `demo_script.md` §4.3: a scratch copy, never `bench.db` or the original collection DB; $0, local only):
```powershell
Copy-Item data\collection\collect_rl.db <scratchpad>\demo_corrections.db
$env:BUSY_MACHINES_DB="<scratchpad>\demo_corrections.db"; python -m uv run --extra ml python scripts/dev_server.py
# second terminal:
curl.exe -s -X POST http://127.0.0.1:8000/api/experiments/700507e1dfe046178c072aed2f9a3aa0/flag -H "content-type: application/json" -d '{\"label\":\"genuine\",\"reason\":\"10 re-runs per side: real gain (R34). Opus judge said unsure. mreq_01a16b31599b7651a654571b8bc455a7\"}'
curl.exe -s http://127.0.0.1:8000/api/corrections
```
Rehearse the PowerShell quoting once before recording.

**Charts** (already rendered; regenerate only if the DB changed. The script stops if the DB disagrees with R75/R78/R82/R81):
```powershell
python -m uv run python scripts/final_charts.py
```

---

## (d) Never say (voice-over and captions)
- "Mubit beats Opus", "Mubit's model beats Opus", or any Mubit-arm number (no adapter has served, R70, R79).
- "A better model for less money", "better than Opus", "beats the standard agent". R78 is a tie, in R82 the standard agent was ahead in every run, and R75 was worse.
- "Mubit saved us 4×" or "Busy Machines is Mubit". The 4× comes from our engine on plain Opus, through Mubit.
- "4× cheaper" without naming the task, or R82 without "chosen after the first setting failed" (R75).
- "Passed Mubit's exam" for workspace C at 55%. Say "we lowered the bar to 55% in one test workspace" (D52), and only once that's logged.
- "Mubit's model is answering" or "served by Mubit", unless a graded Served-by request is logged.
- "It learned from our correction", "the correction changed the model", "v2 fixed this case", "Mubit is retraining it now", "queued".
- "Every check becomes training". Say "every check **we record can become** training" (race calls are record off, R72; Mubit's review keeps only some recorded calls, R29).
- "Live" or "right now" over a replay. Say "replayed from a logged race".
- Any number read off the demo dashboards or the replay counters as a result.
- "Reached the standard agent's quality" or "faster to the goal" for R82 (0 of 3 runs; slower and dearer to 0.86).
- "Real GPUs" for tabular tuning (simulated machine time), and "the Modal bill" for "total $" (in-race compute only).
- "Zero fakes, ever". Say "none got through in these races".
- "Optuna" in the narration. If needed, say "a standard no-AI tuning tool" (D46). Its numbers carry the R74 caveat.
- "Haiku" as Mubit's model or as Busy Machines (R50).
- About our Sapling fine-tune: "Mubit's model", "official", "matches Opus", "beats Opus", "confirms real gains", "reproduces Sapling's ContractNLI score", "full" or "LoRA fine-tune".
- Any projection (R43, R52, R64, R66, the R78 gate) as a measured result.
- Jargon as a main label: adapter, val_bpb, RL, idle %, surrogate, sampler, trap, fingerprint.

---

## (e) 20-second backup cut
For a submission slot that wants a short clip, or for when the 3:00 cut fails review. One visual, one take.

| Time | Narration (48 words) | Visual | Caption |
|---|---|---|---|
| 0:00–0:20 | "Busy Machines runs the same Opus 5.5 research agent with our engine around it, every call through Mubit. Game-playing: a tie on model quality, for a quarter of the AI bill. Tabular: a quarter of the bill, slightly behind in every run, after our first pre-registered setting failed." | `1-headline-two-tasks-dark.png`, full screen, slow push-in left → right | 0–7 s "Busy Machines vs standard Opus agent · Game-playing: a tie, −154.3 vs −154.5 · AI bill $0.83 vs $3.30 per race (R78)" · 7–14 s "Tabular: 0.8884 vs 0.8897, standard agent ahead in all 3 runs · $0.70 vs $2.77 (R82)" · 14–20 s "First pre-registered setting failed: 7% more errors (R75) · AI $ at list price, tabular machine time simulated" |
