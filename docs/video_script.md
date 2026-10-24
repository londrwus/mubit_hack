# Video script: the 3-minute demo video (pre-recorded)

The demo is a **3:00 pre-recorded video**, not a live run. This file is the script, the shot list, the commands for every terminal visual, the "never say" list and a 20-second backup cut. It reuses the framing and number rules of `demo_script.md` (D46, D48), the slide copy of `slides_outline.md`, the IDs in `evidence_pack.md` and the Q&A answers in `qa_mubit_constraints.md`.

**Rules that hold for every frame**
- **Only logged numbers go on screen**, each with its R-id as a small caption. A number that is not in `results_log.md` is not shown, said or captioned. `[R-new: …]` marks a result that must be **logged as a new R-row by the main session before the final edit**. If it isn't logged, use the fallback for that beat.
- **The worse result sits beside each win, in the same caption or sentence.** Put R75 (failed) beside R82 every time. Call R78 "a tie". For R82, say the standard agent was ahead in every run.
- **Positioning (D54):** "Busy Machines: compute intelligence for training sweeps". Busy Machines is the compute layer under the AI planner (when to call it, every machine busy, fakes blocked); Mubit is the cheap-AI layer next to it; they multiply. Never call Busy Machines an "AI research agent" or "research assistant" (that is the planner we sit under). Say "machines" or "compute", never "GPUs" alone: R78 ran on cloud CPU workers, R82 on simulated GPU time, and the real-GPU race (R85) failed its quality line.
- **D48 order:** same model and less money first. Machines waiting is *why it works*, never the headline. Wherever the scorecard appears, R85 (the real-GPU limit) appears with it.
- **AI $ is a list-price estimate. Tabular GPU time is simulated** (virtual GPU $). Say so in the caption footer whenever a $ figure is on screen.
- **Labels in plain language:** "AI bill", "model quality", "machines waiting", "fake improvement caught", "small model". No "adapter", "val", "RL", "sampler", "trap" or "idle %" as a main label.
- **Recordings are labelled as recordings.** The race replay's header tag reads "Replay of logged race R78, run 1 of 3 · 30× speed"; the dashboard API sets this tag itself (`docs/dashboard_api.md` §6). The Modal dashboards, if used, say **"demo run, not a benchmark result"** in the top-right corner for every second they're on screen.
- **The race visual is the R78 replay on the live-compare screen** (Pen frame `eGVso`, "Platform · Sweep detail (live compare)"). It is **not** the 60-second live demo (`demo-live-60s`). In that demo's one test run the standard agent was slightly ahead on model quality and AI bill at 60 s (R89: 0.8090 vs 0.8064, $0.133 vs $0.142). R89 shows that the demo tool works. It is not a result, and none of its numbers go in the video.
- **Design frames** (`C:/Users/Lenovo/Documents/mubit.pen`; open it only through the pencil MCP, never Read it). The PNG exports are in `design/exports/final/<id>.png`, and their numbers are checked in that folder's `README.md`. A frame goes on screen only if every number on it is logged, and it is tagged "design screen" whenever it is a still rather than the built app. The landing page `pOKAY` has five screens: hero `KekD1`, engine `QYavo`, our model `MgmgS`, benchmarks `oT6Ad`, call to action `c50WCZ`. The platform frames are `bxWh3` (Overview) and `eGVso` (Sweep detail, live compare). Conditions before use:
  - `KekD1` says "Your GPUs should never wait" and "Same GPUs, more science", which is "GPUs" alone (against D54). It is **not used** until that copy is changed in Pen (on the user's go).
  - `MgmgS` cites "99.9% vs 88.7% on fresh data", a figure that is not in `results_log.md`. It is **not used** until that figure is removed or logged.
  - `oT6Ad` shows R87's time to the goal without its easy-goal caveat, and it shows neither R75 nor R85. It is **not in the main cut**.
  - The `bxWh3` and `eGVso` run-1 values are logged in R90. Their headline numbers are 3-run means (R78).
- **Built app or stills.** The frontend is built only on the user's go. If it is built, record the `eGVso` layout live from the replay WebSocket. If it isn't, the moving race comes from `scripts/replay_tui.py` (same race, same speed), and the Pen exports appear only as stills.
- No adapter has served a call (R70, R79). So never say "Mubit beats Opus", and never give a Mubit-arm number.
- **Workspace C's bar (D52):** if C's card or settings are shown with the bar at 55%, say "we lowered the bar to 55% in that one test workspace". Never say "passed Mubit's exam". No R-row yet logs that the bar was changed or that C went live. Show it only once that row exists.

**Word budget:** 423 spoken words over 3:00 with the 2-word Try-models placeholder, 428–431 with the real line (about 141–144 words a minute; at 145 words a minute 431 words take 2:58). Beat timings below are proportional to the words in each beat.

---

## (a) Timed script

### Beat 1 · 0:00–0:17 · The problem (money)
| | |
|---|---|
| **Narration (40 words)** | "AI labs now let an AI plan their training sweeps, and the AI is the expensive part. On our game-playing sweep, 90.6% of a standard Opus 5.5 sweep's spend was its own thinking. And while it thinks, the machines wait." |
| **Visual** | Clip **C01**: the **live-compare screen** (`eGVso` layout) replaying race `4ca4ed27` (R78 run 1) at 30×. It opens on the first frame: both sides' 30 machines and AI bill counters at $0. Let it start running from ~0:06. Variant: if the hero `KekD1` has had its "GPUs" copy fixed, show it for 0:00–0:04 first, then cut to C01 |
| **Caption** | "90.6% of a standard Opus 5.5 sweep's spend was AI calls: $10.01 of $11.06 over 3 earlier game-playing runs on cloud CPU workers (R48 races, R54)" · top-right tag (from the screen's header): "Replay of logged race R78, run 1 of 3 · 30× speed" |
| **Fallback** | Frontend not built: clip **C01-tui** (`scripts/replay_tui.py 4ca4ed27 --speed 30 --record`), with the same caption and the tag "Replayed from a logged race (R78) · 30× speed". Terminal unusable: chart 2 (`2-gpus-waiting-r78-dark.png`) as a still, with a slow zoom. Never the 60-second live demo (R89) |

### Beat 2 · 0:17–0:40 · What Busy Machines is (compute intelligence for training sweeps)
| | |
|---|---|
| **Narration (56 words)** | "This is Busy Machines: compute intelligence for training sweeps, replayed from a logged race. Both sides run the same Opus 5.5 planner on the same machines, budget and clock. Ours decides when the planner is worth calling, keeps every machine busy in between with a no-AI tuner, and blocks fake results. Every call goes through Mubit." |
| **Visual** | C01 continues: the 15-minute race plays in 30 s and ends at ~0:36. **Hold the end frame to 0:40.** It shows both final AI bill counters ($3.30 vs $0.70), machines waiting (69.0% vs 1.4%), experiments (497 vs 1,594) and model quality (−155.0 vs −154.9), all R78 run 1 values logged in R90. This end frame is what the `eGVso` design shows |
| **Caption** | "Busy Machines: compute intelligence for training sweeps · same Opus 5.5 planner, machines, clock, $ cap, seeds, starting code; only the compute layer differs (D46) · here: 30 cloud CPU workers per side" → at the end frame: "One run of 3 (R90). 3-run means are on the next screen (R78)" |
| **Fallback** | As in beat 1. If the end frame shows any label in jargon, crop it, or hold on the `eGVso` export (`design/exports/final/eGVso.png`) tagged "design screen · R78 run 1 values (R90)". Its sparkline is labelled illustrative, so crop it out if that label isn't readable at 1080p |

### Beat 3 · 0:40–1:11 · The result: same model, a quarter of the AI bill, and the race we lost first
| | |
|---|---|
| **Narration (71 words)** | "The result, three runs per task. Game-playing: a tie on model quality, for a quarter of the AI bill, 83 cents against $3.30 a race. Tabular tuning: also a quarter of the bill, within 0.0013, but the standard agent was ahead in every run. And the first pre-registered setting failed: eight times cheaper, 7% more errors. We picked the faster pace after that, froze it before racing, and we report both." |
| **Visual** | `data/charts/final/1-headline-two-tasks-dark.png` full screen, 0:40–1:03. Slow push-in on the left panel ("Game-playing"), then the middle, then the right (R75, red). At ~1:03 cut to `3-quality-over-time-r82-dark.png` ("keeps pace, not ahead") until 1:11 |
| **Captions (in order, ≥4 s each)** | 1. "Game-playing: model quality −154.3 vs −154.5, a tie · AI bill $0.83 vs $3.30 per race (R78)" · 2. "Tabular, AI planner at most every 30 s: 0.8884 vs 0.8897; standard agent ahead in all 3 runs · AI bill $0.70 vs $2.77 (R82)" · 3. "**First pre-registered setting FAILED:** AI bill $0.35 vs $2.83, 7% more errors, 0.8816 vs 0.8892 (R75). R82's pace was chosen after it" · 4. (on chart 3) "One run of 3 shown (R82 seed 1). Standard agent ahead in every run" · 5. (last 4 s, on chart 3) "Also logged, not in the chart: long image runs on real L4 GPUs: AI bill $0.23 vs $1.06, but 5% more errors and slower to the goal; quality line FAILED (R85)" · footer throughout: "Busy Machines first, standard Opus agent second · AI $ at list price · tabular machine time simulated" |
| **Fallback** | None needed: the charts exist. If the charts are regenerated (`final_charts.py`), check that their numbers still match R75/R78/R82. **Not used here:** the landing page's benchmarks screen `oT6Ad`. It shows R87's time to the goal (1.0 vs 0.3 min) without its caveat (the goal is easy, and the no-AI tool was cheaper still), and it has no R75 or R85 on it. It may replace chart 1 only if captions 3 and 5 stay on screen with it and the R87 caveat is added |

### Beat 4 · 1:11–1:33 · Why it works: the waiting
| | |
|---|---|
| **Narration (51 words)** | "Why it works: in the game-playing race the standard agent's machines sat waiting 68.8% of the time while Opus planned. Busy Machines' waited 1.5%. With longer runs the standard agent waits less: 10.9% in our image races. That waiting is where the experiments are lost; our engine fills those idle machines." |
| **Visual** | 1:11–1:24: `2-gpus-waiting-r78-dark.png`. 1:24–1:33: the landing page's **engine screen** `QYavo` ("I · The engine: Same model. We changed when it thinks"; `design/exports/final/QYavo.png`, slow push-in on its busy/waiting strip and three steps). Or, if the frontend is built, the same section of the live landing page |
| **Captions** | "Machines waiting (here 30 cloud CPU workers per side): 68.8% vs 1.5%, game-playing, 3-run means (R78) · chart: one run of 3 (seed 1)" → "Longer image-recognition runs on real L4 GPUs: the standard agent waited 10.9% (R53) and 12.0% (R85); there Busy Machines was 4.5× cheaper on AI but had 5% more errors (R85, failed)" → on `QYavo`: "The engine: plans less often, fills idle machines, checks every gain · 69% vs 2% waiting is R78's 68.8% vs 1.5%, rounded" · tag: "Design screen" (stills only) |
| **Variant (paid, only on the user's go)** | Clip **C04**: the two Modal app dashboards side by side, left `busy-machines-opus` and right `busy-machines-bm` (Apps → app → containers / GPU view), from a `demo-cifar-modal-5m` run (not run yet). Sped up 8× in the edit. Tag, top-right, fixed: "**Demo run, not a benchmark result** · image recognition, 3 L4 per side, 5 min · sped up 8×". To use it, swap the last sentence for "Here is a five-minute demo on real cloud GPUs, one Modal app per side." (14 words, same count). Never describe what the dashboards "show" beyond what is visibly on screen, and never read out the demo run's numbers. This is not the 60-second live demo (R89), which is not used |
| **Fallback** | `QYavo` export unreadable at 1080p: keep chart 2 for the whole beat, same narration |

### Beat 5 · 1:33–1:47 · Fake improvement caught
| | |
|---|---|
| **Narration (34 words)** | "A better number only counts if it's real. Busy Machines' side tried a fake shortcut 847 times, mostly from its tuner. The checker let none through. One burst of checks did cost 56 cents." |
| **Visual** | Clip **C02**: the live-compare screen (`eGVso` layout) replaying race `145c2a2d` (R78 run 3) at 30×, over the stretch where "Fake improvement caught" fires again and again and Busy Machines' AI bill counter climbs (the burst). About 14 s cut from the 30 s replay. Use `start_s` to begin just before the burst (section c) |
| **Caption** | "Fake shortcut runs accepted as the best model: 0 of 847 on Busy Machines' side, 3 runs (R78)" → "That run's burst: 143 checks, $0.56, that run's AI bill 0.34× of the standard agent's, vs 0.21× in the other two runs (R78)" · tag (from the header): "Replay of logged race R78, run 3 of 3 · 30× speed" |
| **Fallback** | Frontend not built: C02-tui (`scripts/replay_tui.py 145c2a2d --speed 30 --record`) with the same captions. Or slide 6's checker line as a still: "Real improvements confirmed 7 of 7 vs Opus alone 1 of 7; fakes through 0 of 32, offline on stored verdicts (R37). It accepted 1 of 36 lucky results; Opus 0 (R34)" (and change the narration to match these R37 words) |

### Beat 6 · 1:47–2:23 · Mubit: a correction, and how the model updates (rule #3)
| | |
|---|---|
| **Narration (84 words incl. the 2-word placeholder, FALLBACK wording: no v2, nothing served)** | "Mubit turns the planner's recurring steps into small models. Here's one check where Opus missed: it called a real improvement 'unsure'. Ten re-runs say it's real. We store the correction with the case's Mubit request id, and rate this reply Bad. That hasn't changed a model yet; new examples do. After 25, Mubit trains a new version. Our best small checker matches Opus on 58.3% of its test cases; the bar is 95%, so Opus keeps answering. On this held-out fake, it [Try-models line]." |
| **Visual, in order** | (1) 1:47–1:51 console project A → **Tasks** (the agent's steps as Mubit tasks) · (2) 1:51–1:58 workspace B → **Requests** → `mreq_01a16c07cc1d7c70ae96381f78916256`, Opus's reply `unsure · insufficient_evidence` · (3) 1:58–2:03 click **Rate this reply: Bad**, with a 3 s inset of clip **C05** (the flag `curl` returning the stored correction row, `sent_to_mubit_at: null`) · (4) 2:03–2:09 B → Tasks → judge v1 `ln_5d9ea82de398d3c492ec873f` → Adapters → **History** · (5) 2:09–2:16 workspace C → judge v1 `ln_dfe18721334e06d23c32f338` → Adapters card (`ad_lane_52d6d485`, 58.3%) · (6) 2:16–2:23 **Try models**: the adapter's answer next to Opus's on `trap_v1_020` |
| **Captions** | (1) "The agent's steps as Mubit tasks: planning, checking, debugging (R29, R46)" · (2) "Opus: 'unsure'. Ten re-runs: a real improvement (R34, R39)" · (3) "Correction stored with the original race call's Mubit request id (same case, mreq_01a16b31…, R34); this re-recorded reply (R39) rated Bad. Mubit has no corrections API, so this hasn't changed a model" · (4) "New version after 25 new examples (R58); 40 more recorded (R71)" · (5) "Best small checker: 58.3% match with Opus on 24 test cases; bar 95% (R79). Same data in another workspace: 11.4% of 35 (R70)" · (6) "One held-out case, not a score. Opus: 'invalid' (R40). The truth: it was tested on the levels it trained on · small model: as shown [R-new: Try-models answers, all 3 cases]" |
| **Variant lines (swap in only with the R-row logged)** | **B v2 exists:** replace "After 25, Mubit trains a new version." with "After 25, Mubit trains a new version: here's v2, trained on 40 more checks." Caption (4): "v1 → v2, v2 trained on 40 more checks: [v2 score and test-case count, R-new]". Never say v2 learned from the correction. **C's bar lowered (D52), card shows it:** add "We lowered the bar to 55% in that one test workspace" and caption "(bar lowered to 55% by us, workspace C only, D52) [R-new]". **C serves calls (Served by: Mubit model, graded):** use `demo_script.md` §4.4 wording, with "after we lowered the bar to 55% and it cleared that", and caption the served-call grade [R-new] |
| **Try-models line** | Same verdict as Opus: "gives the same answer as Opus: invalid." · Different: "gets it wrong, which is why it doesn't serve yet." · Not recorded: drop the sentence and hold shot (5) for 7 s more. Show `trap_v1_020` whatever the answer is. Don't swap in another case because it looks better: record all 3 prepared cases and log all three |
| **Fallback** | Console signed out or Mubit down during capture: use the console stills captured in step 1 of the shot list, with the same captions. B's history shows only v1: keep the FALLBACK wording ("trains a new version" plus "40 more recorded"). Don't say "queued" |

### Beat 7 · 2:23–2:44 · Our fine-tune of Mubit's verifier
| | |
|---|---|
| **Narration (48 words)** | "We also fine-tuned Mubit's open verifier, Banyan Sapling, for research claims. On real claims from our races, the published model caught none of 109 fakes; ours caught 57 and approved none, in a tenth of a second each. Opus is still the better judge: it caught all 109." |
| **Visual** | 2:23–2:37 `4-sapling-real-claims-dark.png` (pan from "Fakes caught" to "Real improvements confirmed: 0 for all four"). 2:37–2:44 clip **C08**: the Hugging Face model page **if published** (needs the user's HF name, login and go), else the model card file (`data/sapling/sapling_v2/hf_release/README.md`, Markdown preview, scrolled from "# sapling-research v2" to "## Where it loses"). The landing page's **our-model screen** `MgmgS` may replace the file view only after its unlogged "99.9% vs 88.7% on fresh data" figure is removed or logged. Its R81 numbers (0 vs 57 of 109) are fine |
| **Captions** | "152 real race claims with known answers: fakes caught 0 / 57 / 109 of 109 (published Sapling / ours, looser setting / Opus); fakes approved 0 for all (R81)" → "Real improvements confirmed: 0 of 7 for all, Opus included (R81)" → "~0.1 s per check on an H100 (R76) vs ~1.7 s for Opus (R81)" → on C08: "A community fine-tune, not an official MubitLabs model" + (file version) "Model card file · not published". Small, optional: "A later version (v4) failed its pre-declared gate; v2 stays the release (R83)" |
| **Fallback** | HF not published: show the file (as above), never a link that doesn't exist. Editing time short: drop C08 and hold the chart |

### Beat 8 · 2:44–3:00 · Close
| | |
|---|---|
| **Narration (39 words)** | "Mubit makes each AI call cheaper; Busy Machines decides how many a sweep needs, and the savings multiply. Our layer alone: same model, a quarter of the AI bill. On long image runs, not yet: we lost on quality." |
| **Visual** | 2:44–2:52: **Platform · Overview** `bxWh3` (built app, or the export tagged "design screen"). Its cards carry their own R-ids: R78 AI bill $0.83 vs $3.30 and machines busy, R82's AI bill split (R90), and run-1 values (R90). 2:52–3:00: the **call-to-action screen** `c50WCZ` ("Same model. Busy machines.") as the end card. Overlay, dark, large type: "Busy Machines" / "Compute intelligence for training sweeps · Runs on Mubit" / "Same Opus 5.5. Same machines. A quarter of the AI bill." Small line: "Game-playing: a tie (R78) · Tabular: slightly behind in every run (R82), after a failed first setting (R75) · Long image runs on real L4 GPUs: 5% more errors (R85) · measured before any Mubit model served · every number in docs/results_log.md" / repo URL `github.com/londrwus/mubit_hack` |
| **Caption** | On `bxWh3`: "Overview of the game-playing sweep (R78); limits: R75 and R85 failed, see 1:03". On the end card: the overlay above. Last 1 s silent |
| **Fallback** | `bxWh3` or `c50WCZ` unusable: the plain dark end card with the same text for the whole beat |

**Spoken total:** 40 + 56 + 71 + 51 + 34 + 84 + 48 + 39 = **423 words** (FALLBACK wording, counting `[Try-models line]` as 2 words); the real Try-models line makes it 428 ("same answer") or 431 ("gets it wrong"), 2:57–2:58 at 145 words a minute. The v2 variant adds 7 words and the D52 line 11: both together reach ~449 words (~3:06 at 145), so with either variant beat 2 must be shortened. If a variant pushes a take past 3:00, shorten beat 2's list ("Ours calls the planner only when it's worth it, keeps every machine busy, and blocks fakes"). Never cut a worse-result line or a "replayed" or "demo run" label.

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
  | `S5 Dashboard` | Window capture: Chrome "video" profile, 1920×1080, the built frontend's live-compare page (`eGVso` layout) at 100% zoom. Only if the frontend is built (user's go) |
  Charts are not captured: they're placed in the editor as stills (2000×1125 is exactly 16:9, so scale to 1920×1080 with no crop). **Design stills** come from `design/exports/final/` (`QYavo.png` 2880×1792, `bxWh3.png` and `eGVso.png` 2880×1920, `c50WCZ.png` 2880×1800). None is exactly 16:9, so pad them with the frame's own background colour, don't stretch them, and tag every still "design screen". Export again with the pencil MCP only if a frame changes in Pen.
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
| 2 | **C01** replay, R78 run 1 (`4ca4ed27`) on the live-compare screen | S5 (frontend built) or S1 (TUI) | Server running (section c). Create the 30× replay, open it on the live-compare page, start recording, wait 2 s, start playback, and record until 5 s after the end frame. TUI: command C01-tui | ~40 s | beats 1–2 (40 s) |
| 3 | **C02** replay, R78 run 3 (`145c2a2d`) | S5 or S1 | Same procedure, starting from `start_s` just before the burst | ~40 s | beat 5 (14 s) |
| 4 | **C03** spare replay, R82 run 1 (`c1f4b1d9`) | S5 or S1 | Not in the main cut. Spare for Q&A or a re-edit; if used, the R75 caption goes with it | ~40 s | spare |
| 4b | Design stills | editor | Place `QYavo`, `bxWh3`, `c50WCZ` (and `eGVso` as the beat-2 fallback) from `design/exports/final/`. Check each number on them against `design/exports/final/README.md` and the log row it cites. `KekD1`, `MgmgS` and `oT6Ad` stay out unless their conditions in the rules above are met | n/a | beats 2, 4, 8 |
| 5 | **C05** correction flag | S1 | Commands C05 (scratch DB, local dev server). Record the POST and the GET output | ~60 s | beat 6 inset (3 s) |
| 6 | Start the Try-models adapter | S2 | In tab 5 press start, **≥20 min before clip 8**. This uses 1 of 3 adapter starts per org per UTC day, and a cold start takes up to ~20 min | n/a | n/a |
| 7 | **C04** demo race dashboards (**variant, paid, only on the user's go**) | S3 | Deploy both apps, then run the demo race (commands C04). Start recording at "pre-warm", stop after scale-down. One take. Skipped by default: beat 4 uses `QYavo` | ~8 min | beat 4 variant (9 s, at 8×) |
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

**C01 / C02 / C03: the R78 replay on the live-compare screen** (backend server + replay endpoint, `docs/dashboard_api.md` §1, §6; read-only on `bench.db`, $0):
```powershell
# terminal 1: the server
$env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"
$env:BUSY_MACHINES_DB="data/busy_machines.db"
$env:BUSY_MACHINES_REPLAY_SOURCES="data/bench/bench.db"
python -m uv run --extra ml --with onnxruntime==1.29.0 uvicorn --factory backend.app:create_app --port 8765
# terminal 2: create the replay sessions (each returns a replay_id; the screen opens WS /ws/replays/<replay_id>/dashboard?tick_s=0.25&raw=false)
Invoke-RestMethod -Method Post "http://127.0.0.1:8765/api/races/4ca4ed27/replay?speed=30&metric_stride=0"   # C01: R78 run 1
Invoke-RestMethod -Method Post "http://127.0.0.1:8765/api/races/145c2a2d/replay?speed=30&metric_stride=0"   # C02: R78 run 3 (checker burst)
Invoke-RestMethod -Method Post "http://127.0.0.1:8765/api/races/c1f4b1d9/replay?speed=30&metric_stride=0"   # C03 spare: R82 run 1 (show R75 beside it)
# quick check of the stream at 90x (10 s), no frontend needed
python -m uv run python scripts/dashboard_client.py replay 4ca4ed27 --speed 90 --seconds 12 --raw false
# find the C02 burst: compare fakes_caught and ai_bill_usd between race times, then use start_s=<s> on the dashboard WS
python -m uv run python scripts/dashboard_client.py snapshot 145c2a2d --at 300
python -m uv run python scripts/dashboard_client.py snapshot 145c2a2d --at 600
```
A replay session is held in memory (at most 16), so create it just before the take. The header tag is set by the server ("Replay of logged race R78, run 1 of 3 · 30× speed"); never overlay "Live" on it.

**C01-tui / C02-tui / C03-tui: replay TUI fallback** (`scripts/replay_tui.py`; flags from its argparse: `--db`, `--jsonl`, `--speed` (default 30), `--fps`, `--record`, `--hold-start`, `--at`, `--svg`):
```powershell
python -m uv run python scripts/replay_tui.py --help
python -m uv run python scripts/replay_tui.py 4ca4ed27 --speed 30 --record   # C01-tui: R78 run 1
python -m uv run python scripts/replay_tui.py 145c2a2d --speed 30 --record   # C02-tui: R78 run 3 (checker burst)
python -m uv run python scripts/replay_tui.py c1f4b1d9 --speed 30 --record   # C03-tui spare: R82 run 1
python -m uv run python scripts/replay_tui.py 145c2a2d --at 450              # one frame, to find the burst
```
Check before recording (screen or TUI): both sides' machines, the two AI bill counters, model quality and "Fake improvement caught" are shown in plain words. None of `val_*`, "idle %", "sampler" or "trap" may appear as a main label.

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

**C04: demo race on two Modal apps** (VARIANT ONLY, on the user's go; not the 60-second `demo-live-60s`, which the video doesn't use, R89. PAID: Modal GPU + Opus calls. Preset `demo-cifar-modal-5m` in `backend/benchmark/configs.py`: dev mode, labelled "demo run for the video, not a benchmark result", refused by `headline.py`; one Modal app per side via `PER_ARM_MODAL_APPS`. Being built, so confirm the deploy step with whoever builds it):
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
- "Mubit saved us 4×" or "Busy Machines is Mubit". The 4× comes from the Busy Machines layer on plain Opus, through Mubit, before any Mubit model served. Say "different layers: Mubit makes each call cheaper, we decide how many calls a sweep needs".
- "AI research agent", "research assistant" or "AI scientist" as what Busy Machines is (D54). It is compute intelligence for training sweeps.
- "GPUs" alone for the headline races, or anything implying the best results ran on GPUs (R78: cloud CPU workers; R82: simulated GPU time; R85, the real-GPU race, failed its quality line).
- The headline without its limits: R75 (planned too rarely) beside R82, and R85 (long real-GPU runs, 5% worse) wherever the scorecard is shown.
- "4× cheaper" without naming the task, or R82 without "chosen after the first setting failed" (R75).
- "Passed Mubit's exam" for workspace C at 55%. Say "we lowered the bar to 55% in one test workspace" (D52), and only once that's logged.
- "Mubit's model is answering" or "served by Mubit", unless a graded Served-by request is logged.
- "It learned from our correction", "the correction changed the model", "v2 fixed this case", "Mubit is retraining it now", "queued".
- "Every check becomes training". Say "every check **we record can become** training" (race calls are record off, R72; Mubit's review keeps only some recorded calls, R29).
- "Live" or "right now" over a replay. Say "replayed from a logged race".
- Any number read off the demo dashboards or the replay counters as a result. Any number from the 60-second live demo (R89): it shows the demo tool works and is not a result. And no claim that Busy Machines wins in 60 seconds, because there the standard agent was slightly ahead on quality and AI bill.
- The hero copy "Your GPUs should never wait" or "Same GPUs, more science" (`KekD1`) until it is changed (D54). Any design-frame number that isn't logged (for example `MgmgS`'s "99.9% vs 88.7%"). "Powered by Mubit's models" or "Mubit cut our bill" (D54): say "Runs on Mubit".
- R87's "89% cheaper / 72% faster to the goal" without its caveat: the −170 goal is easy with 30 machines, and the no-AI tool reached it cheapest ($0.009 / 0.4 min).
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

| Time | Narration (51 words) | Visual | Caption |
|---|---|---|---|
| 0:00–0:20 | "Busy Machines is compute intelligence for training sweeps: same Opus 5.5 planner, fewer calls, every machine busy, every call through Mubit. Game-playing: a tie on model quality, for a quarter of the AI bill. Tabular: a quarter of the bill, slightly behind in every run, after our first pre-registered setting failed." | `1-headline-two-tasks-dark.png`, full screen, slow push-in left → right | 0–7 s "Busy Machines vs standard Opus agent · Game-playing: a tie, −154.3 vs −154.5 · AI bill $0.83 vs $3.30 per race (R78)" · 7–14 s "Tabular: 0.8884 vs 0.8897, standard agent ahead in all 3 runs · $0.70 vs $2.77 (R82)" · 14–20 s "First pre-registered setting failed: 7% more errors (R75) · long image runs on real L4 GPUs: 5% more errors (R85) · AI $ at list price, tabular machine time simulated" |
