# Busy Machines: compute intelligence for training sweeps

Night Shift London, the Mubit hackathon (23–25 Oct 2026). Repo: https://github.com/londrwus/mubit_hack

Submission summary. Results log up to R92. **Submission window (D55): Sun 25 Oct 00:00–08:00**, via `/submit`; the latest submission counts. Judged: the repo's `main` branch as it is at 08:00, the README, and a demo video of at most 2:00. Our own code freeze stays **06:00** (D53), with 08:00 as the hard limit.

## Submission form (D55)
| Field | Value |
|---|---|
| Team name | Busy Machines |
| Members (names and e-mails) | [member 1 name], [member 1 e-mail] · [member 2 name], [member 2 e-mail] · [add or remove rows] (the user fills these in) |
| Repo URL | https://github.com/londrwus/mubit_hack |
| README URL | https://github.com/londrwus/mubit_hack/blob/main/README.md |
| Demo video | `busy-machines-demo-2min.mp4`: **≤ 2:00**, MP4 or MOV, **≤ 300 MB**, uploaded as a file (no YouTube link, no deck). Script and recording checklist: [`video_script.md`](video_script.md) (2:00 version) |
| When | Between **00:00 and 08:00** Sun 25 Oct; the **latest submission counts**, so a re-submit replaces the earlier one. `main` is judged as it is at 08:00: push before submitting, and don't push after 08:00 |

**Before submitting:** (1) `git status` clean and `main` pushed (only on the user's word); (2) the README top names the user and the recurring task, says what the demo shows, and points to where every LLM call goes to `https://api.mubit.ai/anthropic` (D55, done); (3) the video is under 2:00.0 and under 300 MB; (4) every number in the README and the video is in `results_log.md`.

**One line:** Busy Machines is the compute layer for AI-planned training sweeps: it decides when the AI planner is worth calling, keeps every machine busy in between, and blocks fake improvements. Measured on two tasks, three runs each, before any Mubit model served: **the same model as an Opus-driven sweep, a quarter of the AI bill, machines busy ~98%, no fakes shipped** (R78, R82). It is not a better model at the same money, and it has two logged limits: planning too rarely made the model worse (R75, our first pre-registered tabular race, failed), and on long runs on real L4 GPUs it was cheaper but 5% worse (R85, failed its quality line).

**Two layers that multiply:** Mubit is built to make each AI call cheaper; Busy Machines decides how many calls a sweep needs. Our 4× is the Busy Machines layer alone, on plain Opus 5.5 through Mubit.

Every number below comes from [`results_log.md`](results_log.md) (ids in brackets). Worse results sit beside the wins. AI $ is a list-price token estimate. Where it ran: game-playing on cloud CPU workers (R78), tabular on simulated GPU time (R82, R75), image recognition on real L4 GPUs (R85).

## The problem
ML and RL teams at AI labs now let an AI plan their hyperparameter and RL training sweeps. The planner is the expensive part, and while it thinks, the machines (GPUs or CPUs) wait: with 30 cloud CPU workers and 15-second runs, a standard Opus 5.5 sweep left them waiting 69% of the time (R48), and 90.6% of its bill was its own thinking (R54). Nothing in the sweep decides when the planner is worth calling, so the AI bill grows with every call and the compute idles between them.

## What we built
- **Busy Machines, compute intelligence for training sweeps**: a compute layer that sits under any AI planner (Opus 5.5 here, or a lab's own research agent) and runs its sweep on a compute pool. It decides when the planner is worth calling (a pace limit and a shorter planner prompt), keeps every machine busy with a no-AI sampler fed only checked results, and blocks fake improvements (test-set tuning, noise) with a checker.
- **Sweeps we ran it on:** game-playing (an RL sweep: Pendulum, PPO on cloud CPU workers), tabular model tuning (LCBench, simulated GPU time) and image recognition (CIFAR-10 on real L4 GPUs).
- **`night/bench`**: a fair race harness. Same machines, wall-clock, $ cap, seeds, starting code and model; only the engine differs. Optuna (no AI) is the baseline in every race. Two races were pre-registered (D47, D49).
- **sapling-research**: our fine-tune of Mubit's Banyan Sapling verifier for fake research gains. A community fine-tune, not an official MubitLabs model. Published at https://huggingface.co/busy-machines/sapling-research, with its data at https://huggingface.co/datasets/busy-machines/ml-claims (R92). v2 is the release. v3 and v4 failed their pre-declared gates (R77, R83).
- **A dashboard API** for live races and replays of logged races (R86, R88), with screens designed in Pen (see "Design" below), and a **frontend** (Vite + React + Tailwind: landing `/`, Get started `/app`, live replay `/app/compare`) that builds and replays R78 run 1 to exactly the logged R90 values (R91).

## How we use Mubit
**Runs on Mubit** (D54). Every AI call goes through Mubit's gateway and is recorded and traceable. Mubit trains small models on our calls, and we fine-tuned Mubit's Sapling. We don't say "powered by Mubit's models" or "Mubit cut our bill": the measured 4× came from our engine, with Opus answering through Mubit and no Mubit model serving.
- **Mubit is the cheap-AI layer next to ours:** it is built to make each call cheaper, we decide how many calls a sweep needs, so the savings multiply. Our measured 4× is our layer alone; Mubit's models add on top once they pass.
- **Every LLM call goes through the Mubit gateway** at **`https://api.mubit.ai/anthropic`** with `MUBIT_API_KEY` (model `claude-opus-5-5`), including both race sides and the checker. The URL is set in `backend/agent/llm_base.py` line 29; `backend/agent/mubit_client.py` rejects any other host (lines 354–357), reads the key from `MUBIT_API_KEY`, never a provider key (lines 120–136) and builds the only SDK client (lines 379–386).
- The recurring steps (propose, diagnose a failed run, judge a claim) use frozen prompts, so Mubit groups them into tasks and trains small models ("adapters") from them.
- Corrections are stored against the exact Mubit request ID and rated in the console.

**Constraints we hit** (details in [`mubit_field_report.md`](mubit_field_report.md)):
- **No adapter has answered a call** (all below the 95% target, R70, R79); judge v1 training took about 7 hours (R70).
- Exam rules are visible only after training (F9); 3 adapter starts per day; adapter replies dropped the JSON format (F1); generative steps can't pass a match-based exam (F11).
- No corrections API, so **correction → model update is not shown end to end**.
- We lowered the quality bar to 55% in one test workspace (D52). That is never "passed Mubit's exam".

## Results
| Comparison (3 runs each) | Win | Worse side |
|---|---|---|
| **Game-playing (RL sweep, cloud CPU workers): Busy Machines vs standard Opus [R78]** | Same model, a tie: −154.3 vs −154.5 penalty. AI bill **$0.83 vs $3.30 per race (4.0× less)**; total $1.18 vs $3.65. Machines waiting 1.5% vs 68.8%. 0 of 847 fakes accepted. To the −170 goal: 89% cheaper and 72% faster ($0.022 vs $0.200, 0.3 vs 1.0 min) [R87] | Not a better model. Reached Opus's final quality in 2 of 3 runs only. In one run a burst of checker calls took the bill to 0.34×. The shipped model came from the sampler in 3 of 3 runs (R84). The −170 goal is easy with 30 machines: the no-AI tool reached it cheapest ($0.009 / 0.4 min) [R87] |
| **Tabular (simulated GPU time), planner every ≥30 s [R82], pre-registered, success line met** | Within 0.0013: 0.8884 vs 0.8897. AI bill **$0.698 vs $2.774 (4.0× less)**; total $3.89 vs $5.97 (−35%, GPU $ virtual). Machines waiting 1.1% vs 30.3%. 0 of 85 fakes accepted | Standard Opus ahead in every run (about 1% more errors). Never reached Opus's final best (0/3); slower and dearer to 0.86 ($1.14 / 4.2 min vs $1.08 / 2.9 min). **This setting was chosen after R75 failed** |
| **Tabular, planner every ≥90 s [R75], pre-registered, FAILED** | AI bill $0.347 vs $2.831 (8.2× less) | **7% more errors:** 0.8816 vs 0.8892, worse in all 3 runs (planned too rarely). The no-AI tool shipped a fake in 1 of 3 runs (R84) |
| **Image recognition, real L4 GPUs, planner every ≥60 s [R85], quality gate FAILED** | AI bill **$0.233 vs $1.056 (0.22×, 4.5× less)**; total $0.83 vs $1.66. Machines waiting 0.0% vs 12.0%. 0 of 12 fakes accepted | **5% more errors:** 0.9424 vs 0.9451, worse in all 3 runs. To the 0.94 goal 16% dearer and 2.1× slower; shipped model 0.9395, below the goal. On long runs the machines barely wait, so fewer AI calls cost quality |
| Standard Opus vs Optuna, tabular [R38] | 28% fewer errors: 0.856 vs 0.799 | 0.3× the experiments per $. In one run Optuna reached the goal faster and cheaper |
| Standard Opus vs Optuna, game-playing [R48] | Shipped model −156.9 vs −177.1 (Optuna shipped fakes) | Optuna won every search number: best clean model −155.2 vs −156.9, $0.35 vs $3.69 total |
| Standard Opus vs Optuna, image recognition [R53] | 22% fewer errors: 0.9448 vs 0.9294. Goal reached 3/3 vs 0/3 | $1.65 vs $0.60 per race; 0.5× the experiments per $ |
| Sapling on 152 real race claims with known answers [R81]: published / ours strict / ours looser / Opus | Fakes caught 0 / 8 / 57 / 109 of 109; fakes approved 0 for all four | Real gains confirmed 0 of 7 for all four. Opus is still the better judge. v4 failed its gate (R83) |
| Mubit adapters [R70, R79, R80] | Trained from our recorded calls | Judge v1 11.4% (B) and 58.3% (C), diagnose 8.3%, vs the 95% target; none served, so there is no Mubit-arm number |
| Backend and demo tooling [R86, R88, R89, R91] (**checks, not results**) | API, live events and replay smoke test passed (R86). Review patches 06–09 applied, dashboard API added, full suite 817 passed, 3 skipped, 0 failed (R88); after live Modal metrics and a timer-flake fix, **840 passed, 3 skipped, 0 failed**, and the frontend builds with no console errors (R91). The 60-second live demo endpoint works (R89) | **R89 is a demo run, not a result**: at 60 s the standard agent was slightly ahead on quality (0.8090 vs 0.8064) and AI bill ($0.133 vs $0.142), which is why the video uses the R78 replay. Running the samplers in their own processes cut Optuna's contention from 2.52× / 2.49× to 1.40× / 1.46× slower, not to zero, and only for future races (R88). Logged races keep the R74 caveat |

## Design (Pen)
Designs are in `mubit.pen` (Pen.dev; open it only through the pencil MCP). PNG exports and their number check are in `design/exports/final/README.md`. Per-run values on the frames are logged in R90.
- **Landing page `pOKAY`** ("Landing v2 · Compute intelligence (A+)"): hero `KekD1`, engine `QYavo`, our model `MgmgS`, benchmarks `oT6Ad`, call to action `c50WCZ`.
- **Platform:** "Platform · Overview" `bxWh3`, "Platform · Sweep detail (live compare)" `eGVso` (R78 run 1 replayed at 30×).
- **Known issues** (to fix in Pen before these frames are shown):
  - `KekD1` says "GPUs" alone, against D54 (the built hero in `frontend/` has the same copy).
  - `MgmgS` cites "99.9% vs 88.7%": that is R54's figure for one run of the no-AI tool's shipped image model (self-reported 0.9994, held-out 0.8874), so it is shown only with that label.
  - `oT6Ad` shows R87 without its caveat, and has no R75 or R85.
- **Frontend:** built (R91); see `frontend/README.md`. Its 'View on Hugging Face' link points at the published model page (R92).

## Links
- Docs index: [`README.md`](README.md). Results: [`results_log.md`](results_log.md). Decisions: [`decisions.md`](decisions.md). Race ids and commands: [`evidence_pack.md`](evidence_pack.md)
- Pre-registrations: [`preregistration_lcbench_wide.md`](preregistration_lcbench_wide.md), [`preregistration_lcbench_wide_c30.md`](preregistration_lcbench_wide_c30.md)
- Sapling: [`sapling_vs_published.md`](sapling_vs_published.md)

## Reproduce a race
Needs Python 3.12, uv, `MUBIT_API_KEY` in `.env`, and Modal for game-playing. Paid: from R78, roughly $4.1 of AI (both AI sides, list price) plus ~$2.5 of Modal per game-playing race.

```powershell
python -m uv sync --extra ml
python -m uv run modal deploy workers/runners/modal_runner.py
# game-playing (R78 setup), seeds 0-2, into a fresh DB
python -m uv run --extra ml python scripts/bench.py check rl-wide-bm-cadence-15m
python -m uv run --extra ml python scripts/bench.py run rl-wide-bm-cadence-15m --db data/bench/repro.db --yes
python -m uv run --extra ml python scripts/headline.py --db data/bench/repro.db --preset rl-wide-bm-cadence-15m
# tabular (R82 setup), no Modal needed
python -m uv run --extra ml --with onnxruntime==1.29.0 python scripts/bench.py run lcbench-wide-bm-c30-15m --db data/bench/repro.db --yes
python -m uv run --extra ml python scripts/headline.py --db data/bench/repro.db --preset lcbench-wide-bm-c30-15m
# re-read the logged races (read-only, $0)
python -m uv run --extra ml python scripts/headline.py --race 4ca4ed27 --race d4913e95 --race 145c2a2d
python -m uv run --extra ml python scripts/headline.py --race c1f4b1d9 --race 9d0cfd77 --race b3922d0e
python -m uv run --extra ml python scripts/headline.py --race 613e2537 --race 6afbbe8a --race c72b3af1
```
`headline.py` prints "NOT A HEADLINE RESULT" whenever a fairness gate fails.

## Team and attribution
- Team: Busy Machines. Members: [names and e-mails, to fill in; see the submission form above]. All project code was written during the event.
- Built on: Banyan Sapling (MubitLabs, Apache 2.0), Optuna, JAX, PyTorch, FastAPI, LCBench data; Modal compute; Mubit routing.
- Claude Code was our coding tool only, never a runtime LLM call.

## What's next
1. Get one adapter to answer real calls, then race "Mubit + Busy Machines" against standard Opus.
2. Throttle checker calls, the remaining cost lever (R78).
3. Isolate Optuna's numbers fully. Running each sampler in its own process (patch 09, R88) cut the contention from 2.52× / 2.49× to 1.40× / 1.46× slower, but not to zero, and it applies only to future races. Logged races keep the R74 caveat.
4. Put the Sapling verifier in the live race loop.
5. A corrections API, so a flag becomes training data.

---

## For the presenter
**Corrections from the verify step: all logged as R84** (the original rows are kept as written; R84 supersedes the cells below).
1. **R78 shipped model:** came from the sampler in 3 of 3 races (R84), not 2 of 3. In seed 1 (`d4913e95`) an AI run was best earlier, then replaced by a sampler run.
2. **R75 "fakes shipped 19 of 618":** those are trap runs Optuna accepted as a best at some point. Optuna actually shipped a fake in 1 of 3 races (seed 2, `c72b3af1`). Its shipped held-out score is 0.8509 ± 0.018 (n = 2), not 0.8555. Opus and Busy Machines are unchanged.
3. **R75 experiment counts:** the log's 425 / 1,450 / 1,419 include runs cancelled at the deadline. Experiments finished: 412.7 / 1,434.7 / 1,404.0.
4. **R75 checker counts:** "5 vs 11" counts trap claims only. All rejections: Opus 7, Busy Machines 12.
5. **R82 additions (R84):**
   - Standard Opus accepted 0 of 70 trap runs (this fills slide 4's empty cell).
   - Shipped held-out scores: Opus 0.8896 ± 0.0015, Busy Machines 0.8883 ± 0.0017, Optuna 0.8563 ± 0.025 (n = 2, 1 fake shipped).
6. **R78:** Optuna shipped a fake in 3 of 3 races. Its −157.7 is still a real held-out score.
7. **R41 run totals:** if you count only runs that started, they are 466 of 497 and 1,649 of 2,216. Only its 70.0% waiting is quoted anywhere.
8. Stale race `d17fc68f` (R36) is now marked "aborted" in the DB (R88). It was never pooled.
9. Review patches 06–09 **are applied** (R88), and no existing fingerprint changed. Patch 09 changes the fingerprint of future paid races only.
10. Items 1–7 are logged as R84; quote them from there.
11. Fill in the team members' names and e-mails (submission form above).
12. The 60-second live demo (R89) is "demo tool works; not a result". Never quote its numbers as a comparison. The video uses the R78 replay instead.
13. R87's goal framing (89% cheaper, 72% faster) is always quoted with its caveat: the goal is easy, and the no-AI tool was cheapest.

**Charts** (`data/charts/final/`, 2000×1125, light and dark; regenerate with `python -m uv run python scripts/final_charts.py`, which stops if the DB disagrees with R75/R78/R82/R81):
- `1-headline-two-tasks-light.png`, `1-headline-two-tasks-dark.png`: AI bill and model quality for R78, R82 and R75 (R75 labelled as failed)
- `2-gpus-waiting-r78-light.png`, `2-gpus-waiting-r78-dark.png`: R78 seed 1 (`d4913e95`); quote the logged 3-run means, 68.8% vs 1.5% waiting (R78), since the seed-1 values on the chart are not logged
- `3-quality-over-time-r82-light.png`, `3-quality-over-time-r82-dark.png`: R82 seed 1 (`9d0cfd77`), "keeps pace, not ahead"
- `4-sapling-real-claims-light.png`, `4-sapling-real-claims-dark.png`: R81, including the "real improvements confirmed 0 of 7" panel

Charts 2 and 3 use seed 1, the middle run on the gap between the two sides; other seeds look slightly different. The R75 gap "0.0076" is the difference of the logged means, not a logged number itself.
