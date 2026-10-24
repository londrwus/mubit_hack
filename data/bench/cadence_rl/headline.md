### Headline: Game-playing AI (`rl`), 1 seed(s)

- Races: RL wide 15 min (15 s runs), Busy Machines with Opus + planner cadence 60 s seed 2 · ids `145c2a2d` · seeds 2.
- Setup: 30 slots per arm, 15-min budget, 15 s runs, runner modal.
- Goal: held-out score ≥ -170 (race config (frozen)).
- Starting code held-out: -1065 (mean of every arm's starting-config runs).
- Status: **eligible for the headline** (finished, benchmark mode, frozen goal, equal slots, one fairness fingerprint, no corrections); each framing still needs its own n and its arm gates.
- Fairness fingerprint: `7e42794ae2` (arm set, per-arm engine features, prompt versions called, record modes, race settings, Mubit routing; only races with the same fingerprint are pooled).
- Mubit project routing: `{"config": {"default": "A", "keys": {"A": "MUBIT_API_KEY", "B": "MUBIT_API_KEY_B"}, "routes": {"mubit": {"early_stop": "B", "judge_improvement": "B"}}, "source": "data/mubit_projects.json"}, "config_identical": true, "note": "D38: project B holds the closed-output judge v1 / early_stop tasks; A everything else", "per_arm": {"opus": {"diagnose_run": "A", "early_stop": "A", "judge_improvement": "A", "propose_experiments": "A"}, "opus_bm": {"diagnose_run": "A", "early_stop": "A", "judge_improvement": "A", "propose_experiments": "A"}}}`.
- Standard AI assistant (Opus 5.5): prompts listed but never called in these races (not in the fingerprint): early_stop=v2:f1f3bf21aa79.
- Busy Machines with Opus: prompts listed but never called in these races (not in the fingerprint): early_stop=v2:f1f3bf21aa79.
- ⚠️ Standard AI assistant (Opus 5.5): LLM $ of 74 call(s) is an estimate (cost_source=estimate@list), not a provider bill
- ⚠️ Busy Machines with Opus: LLM $ of 159 call(s) is an estimate (cost_source=estimate@list), not a provider bill
- Cells: mean ± std over seeds (n = seeds with a value). Same quality / same money are paired per race with that race's Opus arm.

| | Standard AI assistant (Opus 5.5) | Busy Machines with Opus | Automatic search (no AI, Optuna) |
|---|---|---|---|
| Engine features (Busy Machines) | none | backfill (keeps GPUs busy with a no-AI sampler while the AI thinks) + judge policy + lean planner (shorter planner replies: only the settings it changes) + planner cadence (calls the AI planner at most every 60 s; the no-AI sampler keeps GPUs busy in between) | none |
| LLM setup (model · record · prompts called, planner effort) | claude-opus-5-5 · record off · diagnose v0, judge v1, propose v1 (effort medium) | claude-opus-5-5 · record off · diagnose v0, judge v1, propose v2 (effort low) | no LLM |
| Cost to reach the goal | $0.083 (n=1) | $0.018 (n=1) | $0.006 (n=1) |
| Time to reach the goal | 0.5 min (n=1) | 0.3 min (n=1) | 0.3 min (n=1) |
| GPUs waiting | 68.4% (n=1) | 1.8% (n=1) | 2.2% (n=1) |
| Model quality @ same budget (best held-out score) | -154.6 (n=1) | -154.1 (n=1) | -158.5 (n=1) |
| Same quality: cost to reach the Opus agent's final quality | $3.49 (n=1) | $0.298 (n=1) | didn't reach it (n=0/1) |
| Same quality: time to reach the Opus agent's final quality | 14.4 min (n=1) | 3.4 min (n=1) | didn't reach it (n=0/1) |
| Same money: best held-out score at the Opus agent's total spend | -154.6 (n=1) | -154.1 (n=1) | -158.5 (n=1) |
| Experiments finished | 502.0 (n=1) | 1576.0 (n=1) | 1565.0 (n=1) |
| Experiments per $ | 139.4 (n=1) | 1087.3 (n=1) | 4471.2 (n=1) |
| Total spend (GPU $ + LLM $) | $3.60 (n=1) | $1.45 (n=1) | $0.350 (n=1) |
| Shipped model, held-out score | -154.6 (n=1) | -154.1 (n=1) | -158.6 (n=1) |
| Fake improvements accepted as a new best (in-race traps) | 0 of 4 trap runs | 0 of 759 trap runs | 24 of 1529 trap runs |

**Busy Machines engine** (slot time freed is not converted to $ and is in no $ framing):

| Engine | Standard AI assistant (Opus 5.5) | Busy Machines with Opus | Automatic search (no AI, Optuna) |
|---|---|---|---|
| Planner cadence (calls the AI planner at most every N seconds; the no-AI sampler keeps GPUs busy in between) | off · 64.0 planner calls per race | on · at most one AI planner call per 60 s · 15.0 planner calls per race | off (no AI planner) |
| Lean planner (shorter planner replies: only the settings it changes) | off · propose v1, effort medium | on · propose v2, effort low | off (no planner) |
| Backfill (keeps GPUs busy with a no-AI sampler while the AI thinks) | off | 120 runs proposed by the AI · 1456 by the no-AI sampler · shipped model from the sampler in 1 of 1 race(s) · sampler fakes: 759 run(s), 128 claim(s) rejected by the checker, 0 shipped | off |
| Judge policy (LLM for leakage + seed-spread rule) | off | 143 claims · LLM said invalid 128 · rule: real 5, noise 138 · final invalid 128, unsure 12, valid 3 | off |

- Judge policy: the judge policy cannot move race headline numbers while unsure claims are accepted (accept_unsure); quote it only through its own ground-truth stats (R37/R40, D41 #5).

- Backfill: backfill runs come from a no-AI Optuna sampler that fills the GPUs the AI leaves free; its claims go through the same checker, and it is told a rejected claim's worst-so-far score, never which option is the trap (the backfill gate).

- Lean planner: the lean planner sends the propose prompt v2 ('changes only': each experiment lists only the settings that differ from the current best) at effort low; judge prompt and what the planner sees are the same as the standard arm's; its separate effect is quoted from its own test (R63 ablation), D46.

- Planner cadence: the planner cadence starts an AI planner (propose) call at most once per interval, measured between call starts; K = min(the per-call cap of 8, the slots the AI's own runs leave free) when a call is due; judge calls are never held back; the no-AI sampler (backfill) fills the slots in between (D47, pre-registered).

**Headline framings** (benchmark.md §6b, fixed formulas; worse results printed as worse):

- Model quality vs the starting code, the Opus 5.5 agent vs start: not measured (the Opus 5.5 agent: 1 distinct seed(s) (needs >= 3))
- Model quality vs the starting code, Busy Machines with Opus vs start: not measured (Busy Machines with Opus: 1 distinct seed(s) (needs >= 3))
- Model quality vs the starting code, Optuna (no AI) vs start: not measured (Optuna (no AI): 1 distinct seed(s) (needs >= 3))
- Head-to-head, Mubit + Busy Machines vs the Opus 5.5 agent: not measured: no evidence an adapter served (also: no Mubit + Busy Machines arm in these races)
- Model quality @ same budget, Busy Machines with Opus vs the Opus 5.5 agent: not measured (Busy Machines with Opus: 1 distinct seed(s) (needs >= 3); the Opus 5.5 agent: 1 distinct seed(s) (needs >= 3))
- Model quality @ the Opus agent's spend, Busy Machines with Opus vs the Opus 5.5 agent: not measured (Busy Machines with Opus: 1 distinct seed(s) (needs >= 3); the Opus 5.5 agent: 1 distinct seed(s) (needs >= 3))
- Cost to reach the goal (held-out score ≥ -170), Busy Machines with Opus vs the Opus 5.5 agent: not measured (Busy Machines with Opus: 1 distinct seed(s) (needs >= 3); the Opus 5.5 agent: 1 distinct seed(s) (needs >= 3))
- Time to reach the goal (held-out score ≥ -170), Busy Machines with Opus vs the Opus 5.5 agent: not measured (Busy Machines with Opus: 1 distinct seed(s) (needs >= 3); the Opus 5.5 agent: 1 distinct seed(s) (needs >= 3))
- Cost to reach the Opus 5.5 agent's final model quality, Busy Machines with Opus vs the Opus 5.5 agent: not measured (Busy Machines with Opus: 1 distinct seed(s) (needs >= 3); the Opus 5.5 agent: 1 distinct seed(s) (needs >= 3))
- Time to reach the Opus 5.5 agent's final model quality, Busy Machines with Opus vs the Opus 5.5 agent: not measured (Busy Machines with Opus: 1 distinct seed(s) (needs >= 3); the Opus 5.5 agent: 1 distinct seed(s) (needs >= 3))
- Experiments per dollar, Busy Machines with Opus vs the Opus 5.5 agent: not measured (Busy Machines with Opus: 1 distinct seed(s) (needs >= 3); the Opus 5.5 agent: 1 distinct seed(s) (needs >= 3))
- Head-to-head, Mubit + Busy Machines vs Busy Machines with Opus: not measured: no evidence an adapter served (also: no Mubit + Busy Machines arm in these races)
- Head-to-head, Mubit + Busy Machines vs Optuna (no AI): not measured: no evidence an adapter served (also: no Mubit + Busy Machines arm in these races)
- Model quality @ same budget, the Opus 5.5 agent vs Optuna (no AI): not measured (the Opus 5.5 agent: 1 distinct seed(s) (needs >= 3); Optuna (no AI): 1 distinct seed(s) (needs >= 3))
- Model quality @ the Opus agent's spend, the Opus 5.5 agent vs Optuna (no AI): not measured (the Opus 5.5 agent: 1 distinct seed(s) (needs >= 3); Optuna (no AI): 1 distinct seed(s) (needs >= 3))
- Cost to reach the goal (held-out score ≥ -170), the Opus 5.5 agent vs Optuna (no AI): not measured (the Opus 5.5 agent: 1 distinct seed(s) (needs >= 3); Optuna (no AI): 1 distinct seed(s) (needs >= 3))
- Time to reach the goal (held-out score ≥ -170), the Opus 5.5 agent vs Optuna (no AI): not measured (the Opus 5.5 agent: 1 distinct seed(s) (needs >= 3); Optuna (no AI): 1 distinct seed(s) (needs >= 3))
- Experiments per dollar, the Opus 5.5 agent vs Optuna (no AI): not measured (the Opus 5.5 agent: 1 distinct seed(s) (needs >= 3); Optuna (no AI): 1 distinct seed(s) (needs >= 3))
- Fake improvements let through: not measured here (trap-set result, scripts/eval_trap_set.py; the in-race trap counts are in the table). The Mubit adapter's version is not measured: no evidence an adapter served.

**Evidence** behind each number (summed over races; LLM calls completed and runs started inside the window the number uses):

| Number | Standard AI assistant (Opus 5.5) | Busy Machines with Opus | Automatic search (no AI, Optuna) |
|---|---|---|---|
| Cost to reach the goal | 2/2 Mubit req IDs · 17/17 Modal IDs · 1 race(s) | 1/1 Mubit req IDs · 38/38 Modal IDs · 1 race(s) | no LLM · 33/33 Modal IDs · 1 race(s) |
| Time to reach the goal | 2/2 Mubit req IDs · 17/17 Modal IDs · 1 race(s) | 1/1 Mubit req IDs · 38/38 Modal IDs · 1 race(s) | no LLM · 33/33 Modal IDs · 1 race(s) |
| GPUs waiting | 74/74 Mubit req IDs · 502/510 Modal IDs · 1 race(s) | 159/159 Mubit req IDs · 1577/1579 Modal IDs · 1 race(s) | no LLM · 1566/1568 Modal IDs · 1 race(s) |
| Model quality @ same budget (best held-out score) | 74/74 Mubit req IDs · 502/510 Modal IDs · 1 race(s) | 159/159 Mubit req IDs · 1577/1579 Modal IDs · 1 race(s) | no LLM · 1566/1568 Modal IDs · 1 race(s) |
| Same quality: cost to reach the Opus agent's final quality | 71/71 Mubit req IDs · 494/494 Modal IDs · 1 race(s) | 25/25 Mubit req IDs · 377/377 Modal IDs · 1 race(s) | no LLM · 0/0 Modal IDs · 1 race(s) |
| Same quality: time to reach the Opus agent's final quality | 71/71 Mubit req IDs · 494/494 Modal IDs · 1 race(s) | 25/25 Mubit req IDs · 377/377 Modal IDs · 1 race(s) | no LLM · 0/0 Modal IDs · 1 race(s) |
| Same money: best held-out score at the Opus agent's total spend | 74/74 Mubit req IDs · 502/502 Modal IDs · 1 race(s) | 159/159 Mubit req IDs · 1577/1577 Modal IDs · 1 race(s) | no LLM · 1496/1496 Modal IDs · 1 race(s) |
| Experiments finished | 74/74 Mubit req IDs · 502/510 Modal IDs · 1 race(s) | 159/159 Mubit req IDs · 1577/1579 Modal IDs · 1 race(s) | no LLM · 1566/1568 Modal IDs · 1 race(s) |
| Experiments per $ | 74/74 Mubit req IDs · 502/510 Modal IDs · 1 race(s) | 159/159 Mubit req IDs · 1577/1579 Modal IDs · 1 race(s) | no LLM · 1566/1568 Modal IDs · 1 race(s) |
| Total spend (GPU $ + LLM $) | 74/74 Mubit req IDs · 502/510 Modal IDs · 1 race(s) | 159/159 Mubit req IDs · 1577/1579 Modal IDs · 1 race(s) | no LLM · 1566/1568 Modal IDs · 1 race(s) |
| Shipped model, held-out score | 74/74 Mubit req IDs · 502/510 Modal IDs · 1 race(s) | 159/159 Mubit req IDs · 1577/1579 Modal IDs · 1 race(s) | no LLM · 1566/1568 Modal IDs · 1 race(s) |
| Fake improvements accepted as a new best (in-race traps) | 74/74 Mubit req IDs · 502/510 Modal IDs · 1 race(s) | 159/159 Mubit req IDs · 1577/1579 Modal IDs · 1 race(s) | no LLM · 1566/1568 Modal IDs · 1 race(s) |

### Headline: Game-playing AI (`rl`), 2 seed(s)

- Races: RL wide 15 min (15 s runs), Busy Machines with Opus + planner cadence 60 s seed 0; RL wide 15 min (15 s runs), Busy Machines with Opus + planner cadence 60 s seed 1 · ids `4ca4ed27`, `d4913e95` · seeds 0, 1.
- Setup: 30 slots per arm, 15-min budget, 15 s runs, runner modal.
- Goal: held-out score ≥ -170 (race config (frozen)).
- Starting code held-out: -1035 (mean of every arm's starting-config runs).
- Status: **eligible for the headline** (finished, benchmark mode, frozen goal, equal slots, one fairness fingerprint, no corrections); each framing still needs its own n and its arm gates.
- Fairness fingerprint: `e0d1cc7906` (arm set, per-arm engine features, prompt versions called, record modes, race settings, Mubit routing; only races with the same fingerprint are pooled).
- Mubit project routing: `{"config": {"default": "A", "keys": {"A": "MUBIT_API_KEY", "B": "MUBIT_API_KEY_B"}, "routes": {"mubit": {"early_stop": "B", "judge_improvement": "B"}}, "source": "data/mubit_projects.json"}, "config_identical": true, "note": "D38: project B holds the closed-output judge v1 / early_stop tasks; A everything else", "per_arm": {"opus": {"diagnose_run": "A", "early_stop": "A", "judge_improvement": "A", "propose_experiments": "A"}, "opus_bm": {"diagnose_run": "A", "early_stop": "A", "judge_improvement": "A", "propose_experiments": "A"}}}`.
- Standard AI assistant (Opus 5.5): prompts listed but never called in these races (not in the fingerprint): diagnose_run=v0:22ecb45368e3, early_stop=v2:f1f3bf21aa79.
- Busy Machines with Opus: prompts listed but never called in these races (not in the fingerprint): diagnose_run=v0:22ecb45368e3, early_stop=v2:f1f3bf21aa79.
- ⚠️ Standard AI assistant (Opus 5.5): LLM $ of 157 call(s) is an estimate (cost_source=estimate@list), not a provider bill
- ⚠️ Busy Machines with Opus: LLM $ of 114 call(s) is an estimate (cost_source=estimate@list), not a provider bill
- Cells: mean ± std over seeds (n = seeds with a value). Same quality / same money are paired per race with that race's Opus arm.

| | Standard AI assistant (Opus 5.5) | Busy Machines with Opus | Automatic search (no AI, Optuna) |
|---|---|---|---|
| Engine features (Busy Machines) | none | backfill (keeps GPUs busy with a no-AI sampler while the AI thinks) + judge policy + lean planner (shorter planner replies: only the settings it changes) + planner cadence (calls the AI planner at most every 60 s; the no-AI sampler keeps GPUs busy in between) | none |
| LLM setup (model · record · prompts called, planner effort) | claude-opus-5-5 · record off · judge v1, propose v1 (effort medium) | claude-opus-5-5 · record off · judge v1, propose v2 (effort low) | no LLM |
| Cost to reach the goal | $0.259 ± 0.008 (n=2) | $0.024 ± 0.009 (n=2) | $0.010 ± 0.004 (n=2) |
| Time to reach the goal | 1.3 ± 0.0 min (n=2) | 0.3 ± 0.0 min (n=2) | 0.4 ± 0.2 min (n=2) |
| GPUs waiting | 69.0 ± 0.1% (n=2) | 1.4 ± 0.1% (n=2) | 1.8 ± 0.1% (n=2) |
| Model quality @ same budget (best held-out score) | -154.4 ± 0.88 (n=2) | -154.4 ± 0.64 (n=2) | -156.7 ± 1.4 (n=2) |
| Same quality: cost to reach the Opus agent's final quality | $2.90 ± 0.650 (n=2) | $0.616 (n=1/2 reached) | didn't reach it (n=0/2) |
| Same quality: time to reach the Opus agent's final quality | 11.9 ± 2.6 min (n=2) | 8.1 min (n=1/2 reached) | didn't reach it (n=0/2) |
| Same money: best held-out score at the Opus agent's total spend | -154.4 ± 0.88 (n=2) | -154.4 ± 0.64 (n=2) | -156.7 ± 1.4 (n=2) |
| Experiments finished | 499.5 ± 3.5 (n=2) | 1592.5 ± 2.1 (n=2) | 1587.0 ± 1.4 (n=2) |
| Experiments per $ | 136.1 ± 0.2 (n=2) | 1520.0 ± 8.2 (n=2) | 4530.4 ± 4.9 (n=2) |
| Total spend (GPU $ + LLM $) | $3.67 ± 0.031 (n=2) | $1.05 ± 0.007 (n=2) | $0.350 ± 0.000 (n=2) |
| Shipped model, held-out score | -154.4 ± 0.88 (n=2) | -154.4 ± 0.64 (n=2) | -157.2 ± 0.69 (n=2) |
| Fake improvements accepted as a new best (in-race traps) | 0 of 6 trap runs | 0 of 88 trap runs | 29 of 3119 trap runs |

**Busy Machines engine** (slot time freed is not converted to $ and is in no $ framing):

| Engine | Standard AI assistant (Opus 5.5) | Busy Machines with Opus | Automatic search (no AI, Optuna) |
|---|---|---|---|
| Planner cadence (calls the AI planner at most every N seconds; the no-AI sampler keeps GPUs busy in between) | off · 63.5 planner calls per race | on · at most one AI planner call per 60 s · 15.0 planner calls per race | off (no AI planner) |
| Lean planner (shorter planner replies: only the settings it changes) | off · propose v1, effort medium | on · propose v2, effort low | off (no planner) |
| Backfill (keeps GPUs busy with a no-AI sampler while the AI thinks) | off | 240 runs proposed by the AI · 2943 by the no-AI sampler · shipped model from the sampler in 2 of 2 race(s) · sampler fakes: 88 run(s), 38 claim(s) rejected by the checker, 0 shipped | off |
| Judge policy (LLM for leakage + seed-spread rule) | off | 84 claims · LLM said invalid 38 · rule: real 22, noise 62 · final invalid 38, unsure 41, valid 5 | off |

- Judge policy: the judge policy cannot move race headline numbers while unsure claims are accepted (accept_unsure); quote it only through its own ground-truth stats (R37/R40, D41 #5).

- Backfill: backfill runs come from a no-AI Optuna sampler that fills the GPUs the AI leaves free; its claims go through the same checker, and it is told a rejected claim's worst-so-far score, never which option is the trap (the backfill gate).

- Lean planner: the lean planner sends the propose prompt v2 ('changes only': each experiment lists only the settings that differ from the current best) at effort low; judge prompt and what the planner sees are the same as the standard arm's; its separate effect is quoted from its own test (R63 ablation), D46.

- Planner cadence: the planner cadence starts an AI planner (propose) call at most once per interval, measured between call starts; K = min(the per-call cap of 8, the slots the AI's own runs leave free) when a call is due; judge calls are never held back; the no-AI sampler (backfill) fills the slots in between (D47, pre-registered).

**Headline framings** (benchmark.md §6b, fixed formulas; worse results printed as worse):

- Model quality vs the starting code, the Opus 5.5 agent vs start: not measured (the Opus 5.5 agent: 2 distinct seed(s) (needs >= 3))
- Model quality vs the starting code, Busy Machines with Opus vs start: not measured (Busy Machines with Opus: 2 distinct seed(s) (needs >= 3))
- Model quality vs the starting code, Optuna (no AI) vs start: not measured (Optuna (no AI): 2 distinct seed(s) (needs >= 3))
- Head-to-head, Mubit + Busy Machines vs the Opus 5.5 agent: not measured: no evidence an adapter served (also: no Mubit + Busy Machines arm in these races)
- Model quality @ same budget, Busy Machines with Opus vs the Opus 5.5 agent: not measured (Busy Machines with Opus: 2 distinct seed(s) (needs >= 3); the Opus 5.5 agent: 2 distinct seed(s) (needs >= 3))
- Model quality @ the Opus agent's spend, Busy Machines with Opus vs the Opus 5.5 agent: not measured (Busy Machines with Opus: 2 distinct seed(s) (needs >= 3); the Opus 5.5 agent: 2 distinct seed(s) (needs >= 3))
- Cost to reach the goal (held-out score ≥ -170), Busy Machines with Opus vs the Opus 5.5 agent: not measured (Busy Machines with Opus: 2 distinct seed(s) (needs >= 3); the Opus 5.5 agent: 2 distinct seed(s) (needs >= 3))
- Time to reach the goal (held-out score ≥ -170), Busy Machines with Opus vs the Opus 5.5 agent: not measured (Busy Machines with Opus: 2 distinct seed(s) (needs >= 3); the Opus 5.5 agent: 2 distinct seed(s) (needs >= 3))
- Cost to reach the Opus 5.5 agent's final model quality, Busy Machines with Opus vs the Opus 5.5 agent: not measured (Busy Machines with Opus: 2 distinct seed(s) (needs >= 3); the Opus 5.5 agent: 2 distinct seed(s) (needs >= 3))
- Time to reach the Opus 5.5 agent's final model quality, Busy Machines with Opus vs the Opus 5.5 agent: not measured (Busy Machines with Opus: 2 distinct seed(s) (needs >= 3); the Opus 5.5 agent: 2 distinct seed(s) (needs >= 3))
- Experiments per dollar, Busy Machines with Opus vs the Opus 5.5 agent: not measured (Busy Machines with Opus: 2 distinct seed(s) (needs >= 3); the Opus 5.5 agent: 2 distinct seed(s) (needs >= 3))
- Head-to-head, Mubit + Busy Machines vs Busy Machines with Opus: not measured: no evidence an adapter served (also: no Mubit + Busy Machines arm in these races)
- Head-to-head, Mubit + Busy Machines vs Optuna (no AI): not measured: no evidence an adapter served (also: no Mubit + Busy Machines arm in these races)
- Model quality @ same budget, the Opus 5.5 agent vs Optuna (no AI): not measured (the Opus 5.5 agent: 2 distinct seed(s) (needs >= 3); Optuna (no AI): 2 distinct seed(s) (needs >= 3))
- Model quality @ the Opus agent's spend, the Opus 5.5 agent vs Optuna (no AI): not measured (the Opus 5.5 agent: 2 distinct seed(s) (needs >= 3); Optuna (no AI): 2 distinct seed(s) (needs >= 3))
- Cost to reach the goal (held-out score ≥ -170), the Opus 5.5 agent vs Optuna (no AI): not measured (the Opus 5.5 agent: 2 distinct seed(s) (needs >= 3); Optuna (no AI): 2 distinct seed(s) (needs >= 3))
- Time to reach the goal (held-out score ≥ -170), the Opus 5.5 agent vs Optuna (no AI): not measured (the Opus 5.5 agent: 2 distinct seed(s) (needs >= 3); Optuna (no AI): 2 distinct seed(s) (needs >= 3))
- Experiments per dollar, the Opus 5.5 agent vs Optuna (no AI): not measured (the Opus 5.5 agent: 2 distinct seed(s) (needs >= 3); Optuna (no AI): 2 distinct seed(s) (needs >= 3))
- Fake improvements let through: not measured here (trap-set result, scripts/eval_trap_set.py; the in-race trap counts are in the table). The Mubit adapter's version is not measured: no evidence an adapter served.

**Evidence** behind each number (summed over races; LLM calls completed and runs started inside the window the number uses):

| Number | Standard AI assistant (Opus 5.5) | Busy Machines with Opus | Automatic search (no AI, Optuna) |
|---|---|---|---|
| Cost to reach the goal | 18/18 Mubit req IDs · 82/82 Modal IDs · 2 race(s) | 2/2 Mubit req IDs · 70/70 Modal IDs · 2 race(s) | no LLM · 93/93 Modal IDs · 2 race(s) |
| Time to reach the goal | 18/18 Mubit req IDs · 82/82 Modal IDs · 2 race(s) | 2/2 Mubit req IDs · 70/70 Modal IDs · 2 race(s) | no LLM · 93/93 Modal IDs · 2 race(s) |
| GPUs waiting | 157/157 Mubit req IDs · 999/1015 Modal IDs · 2 race(s) | 114/114 Mubit req IDs · 3185/3185 Modal IDs · 2 race(s) | no LLM · 3174/3176 Modal IDs · 2 race(s) |
| Model quality @ same budget (best held-out score) | 157/157 Mubit req IDs · 999/1015 Modal IDs · 2 race(s) | 114/114 Mubit req IDs · 3185/3185 Modal IDs · 2 race(s) | no LLM · 3174/3176 Modal IDs · 2 race(s) |
| Same quality: cost to reach the Opus agent's final quality | 126/126 Mubit req IDs · 802/802 Modal IDs · 2 race(s) | 41/41 Mubit req IDs · 879/879 Modal IDs · 2 race(s) | no LLM · 0/0 Modal IDs · 2 race(s) |
| Same quality: time to reach the Opus agent's final quality | 126/126 Mubit req IDs · 802/802 Modal IDs · 2 race(s) | 41/41 Mubit req IDs · 879/879 Modal IDs · 2 race(s) | no LLM · 0/0 Modal IDs · 2 race(s) |
| Same money: best held-out score at the Opus agent's total spend | 156/156 Mubit req IDs · 999/999 Modal IDs · 2 race(s) | 114/114 Mubit req IDs · 3185/3185 Modal IDs · 2 race(s) | no LLM · 3088/3088 Modal IDs · 2 race(s) |
| Experiments finished | 157/157 Mubit req IDs · 999/1015 Modal IDs · 2 race(s) | 114/114 Mubit req IDs · 3185/3185 Modal IDs · 2 race(s) | no LLM · 3174/3176 Modal IDs · 2 race(s) |
| Experiments per $ | 157/157 Mubit req IDs · 999/1015 Modal IDs · 2 race(s) | 114/114 Mubit req IDs · 3185/3185 Modal IDs · 2 race(s) | no LLM · 3174/3176 Modal IDs · 2 race(s) |
| Total spend (GPU $ + LLM $) | 157/157 Mubit req IDs · 999/1015 Modal IDs · 2 race(s) | 114/114 Mubit req IDs · 3185/3185 Modal IDs · 2 race(s) | no LLM · 3174/3176 Modal IDs · 2 race(s) |
| Shipped model, held-out score | 157/157 Mubit req IDs · 999/1015 Modal IDs · 2 race(s) | 114/114 Mubit req IDs · 3185/3185 Modal IDs · 2 race(s) | no LLM · 3174/3176 Modal IDs · 2 race(s) |
| Fake improvements accepted as a new best (in-race traps) | 157/157 Mubit req IDs · 999/1015 Modal IDs · 2 race(s) | 114/114 Mubit req IDs · 3185/3185 Modal IDs · 2 race(s) | no LLM · 3174/3176 Modal IDs · 2 race(s) |

#### Notes

- rl races with the same budget/slots/goal were NOT pooled because their fairness fingerprints differ: fingerprint `7e42794ae2` (`145c2a2d`) | fingerprint `e0d1cc7906` (`4ca4ed27`, `d4913e95`) — differs from `7e42794ae2` in: arms.opus.llm_policy.prompts.diagnose_run: v0:22ecb45368e3 vs absent; arms.opus_bm.llm_policy.prompts.diagnose_run: v0:22ecb45368e3 vs absent

#### Excluded

- Nothing excluded.
