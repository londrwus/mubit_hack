# Pre-registration 2: LCBench wide setup, planner every ≥30 s (frozen 2026-10-24 ~16:45 BST, before any race)

**This is a second, follow-up setting chosen after the first pre-registered race failed (R75).** Both are reported together; neither replaces the other.

**Why this setting:** in R75 Busy Machines' 90 s cadence kept only 80 AI proposals per race and its best model was 7% worse than the standard Opus agent's (0.8816 vs 0.8892), while its LLM bill was 8.2× lower. A 30 s cadence gives the planner ~3× more proposals. The R66 simulator projected BM LLM $0.85 per race at 30 s (≈0.30× the measured Opus bill; it over-estimated BM's bill at 90 s: $0.44 projected vs $0.347 measured).

**Identical to `preregistration_lcbench_wide.md` (preset `lcbench-wide-bm-15m`) except one parameter:** Busy Machines' `planner_cadence` `min_propose_interval_s = 30` (was 90). Same task, 16 virtual L4/arm, runtime multiplier 0.1, 900 s, $9.20 cap, arms (standard Opus v1/medium, BM, Optuna), record off, project A, goal 0.78 frozen, 0.87 reported, **seeds 0, 1, 2**.

**Primary endpoints (D48), reported first:** same quality (total $ and time to reach the standard Opus arm's final best, paired; and to 0.86); same money (best score at Opus's total spend); best score and total / LLM $ per race.

**Success line ("same quality for much less AI money"), all on 3-seed means:**
1. BM best held-out ≥ standard Opus best held-out − 0.003 (about the per-arm seed spread in R75: 0.0017 / 0.0038).
2. BM LLM $ per race ≤ 0.4 × standard Opus.
3. BM best ≥ Optuna best.
4. 0 fakes shipped by BM.

GPUs waiting is reported, not part of the line. A fail is logged as a fail; no re-runs, no seed swaps, no further cadence settings after this one.
