# Busy Machines: submission summary

Night Shift London, the Mubit hackathon (23–25 Oct 2026). Repo: https://github.com/londrwus/mubit_hack

**One line:** the same model as a standard Opus 5.5 research agent for a quarter of the AI bill, on two tasks, three runs each (R78, R82). It is not a better model at the same money, and our first pre-registered tabular race failed (R75).

Every number below comes from [`results_log.md`](results_log.md) (ids in brackets). Worse results sit beside the wins. AI $ is a list-price token estimate. On the tabular task, GPU time is simulated.

## The problem
AI labs let agents run ML experiments. While the agent thinks, the GPUs wait: with 30 workers and 15-second runs, a standard Opus 5.5 agent left them waiting 69% of the time (R48), and 90.6% of its bill was its own thinking (R54). The agent is both the bottleneck and the main cost.

## What we built
- **An ML research agent** that improves any model whose training can be scripted. Tasks: game-playing (Pendulum, PPO on CPU), image recognition (CIFAR-10 on L4 GPUs) and tabular model tuning (LCBench, simulated GPU time).
- **The Busy Machines engine**, on the same Opus 5.5: a shorter planner prompt, a no-AI sampler that fills idle machines while Opus thinks, a limit on how often the planner is called, and a checker that rejects fake improvements (test-set tuning, noise).
- **`night/bench`**: a fair race harness. Same machines, wall-clock, $ cap, seeds, starting code and model; only the engine differs. Optuna (no AI) is the baseline in every race. Two races were pre-registered (D47, D49).
- **sapling-research**: our fine-tune of Mubit's Banyan Sapling verifier for fake research gains. A community fine-tune, not an official MubitLabs model, and not published.

## How we use Mubit
- **Every LLM call goes through the Mubit gateway** (Anthropic endpoint, model `claude-opus-5-5`), including both race sides and the checker.
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
| **Game-playing: Busy Machines vs standard Opus [R78]** | Same model, a tie: −154.3 vs −154.5 penalty. AI bill **$0.83 vs $3.30 per race (4.0× less)**; total $1.18 vs $3.65. GPUs waiting 1.5% vs 68.8% | Not a better model. Reached Opus's final quality in 2 of 3 runs only. In one run a burst of checker calls took the bill to 0.34× |
| **Tabular, planner every ≥30 s [R82], pre-registered, success line met** | Within 0.0013: 0.8884 vs 0.8897. AI bill **$0.70 vs $2.77 (4.0× less)** | Standard Opus ahead in every run (about 1% more errors). **This setting was chosen after R75 failed** |
| **Tabular, planner every ≥90 s [R75], pre-registered, FAILED** | AI bill $0.35 vs $2.83 (8.2× less) | **7% more errors:** 0.8816 vs 0.8892, worse in all 3 runs |
| Standard Opus vs Optuna, tabular [R38] | 28% fewer errors: 0.856 vs 0.799 | 0.3× the experiments per $. In one run Optuna reached the goal faster and cheaper |
| Standard Opus vs Optuna, game-playing [R48] | Shipped model −156.9 vs −177.1 (Optuna shipped fakes) | Optuna won every search number: best clean model −155.2 vs −156.9, $0.35 vs $3.69 total |
| Standard Opus vs Optuna, image recognition [R53] | 22% fewer errors: 0.9448 vs 0.9294. Goal reached 3/3 vs 0/3 | $1.65 vs $0.60 per race; 0.5× the experiments per $ |
| Sapling on 152 real race claims with known answers [R81]: published / ours strict / ours looser / Opus | Fakes caught 0 / 8 / 57 / 109 of 109; fakes approved 0 for all four | Real gains confirmed 0 of 7 for all four. Opus is still the better judge. v4 failed its gate (R83) |
| Mubit adapters [R70, R79] | Trained from our recorded calls | 11.4% and 58.3% vs the 95% target; none served, so there is no Mubit-arm number |

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
- Team: [names to fill in]. All project code was written during the event.
- Built on: Banyan Sapling (MubitLabs, Apache 2.0), Optuna, JAX, PyTorch, FastAPI, LCBench data; Modal compute; Mubit routing.
- Claude Code was our coding tool only, never a runtime LLM call.

## What's next
1. Get one adapter to answer real calls, then race "Mubit + Busy Machines" against standard Opus.
2. Throttle checker calls, the remaining cost lever (R78).
3. Run each sampler in its own process, so Optuna's numbers are isolated (R74).
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
8. Stale race `d17fc68f` is marked "running" in the DB. It was never pooled and is harmless.
9. Review patches 06–09 are not applied, and every fingerprint matches the log.
10. Before quoting items 1–7, wait until they are logged.
11. Fill in the team names.

**Charts** (`data/charts/final/`, 2000×1125, light and dark; regenerate with `python -m uv run python scripts/final_charts.py`, which stops if the DB disagrees with R75/R78/R82/R81):
- `1-headline-two-tasks-light.png`, `1-headline-two-tasks-dark.png`: AI bill and model quality for R78, R82 and R75 (R75 labelled as failed)
- `2-gpus-waiting-r78-light.png`, `2-gpus-waiting-r78-dark.png`: R78 seed 1 (`d4913e95`), 68.9% vs 1.5% waiting
- `3-quality-over-time-r82-light.png`, `3-quality-over-time-r82-dark.png`: R82 seed 1 (`9d0cfd77`), "keeps pace, not ahead"
- `4-sapling-real-claims-light.png`, `4-sapling-real-claims-dark.png`: R81, including the "real improvements confirmed 0 of 7" panel

Charts 2 and 3 use seed 1, the middle run on the gap between the two sides; other seeds look slightly different. The R75 gap "0.0076" is the difference of the logged means, not a logged number itself.
