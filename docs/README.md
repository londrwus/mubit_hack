# Docs index

`decisions.md` is the source of truth. If another doc disagrees with it, the decision wins, and the doc gets fixed.

| Doc | What's in it |
|---|---|
| [decisions.md](decisions.md) | **Decisions (D1–D48), answers from our probes, open questions for Mubit** |
| [results_log.md](results_log.md) | **Every measurement, including where we're worse.** The only source for numbers on slides |
| [context.md](context.md) | Hackathon rules and judging; how Mubit works (docs + what we observed); the 23 Oct launch (platform, Banyan Sapling, latte/bench); Anil's worldview; Rekursiv and related benchmarks |
| [pitch.md](pitch.md) | Customer (AI labs only), the three promises, pitch scripts and lines, plain-language rules, the "Mubit for Research" commercial story, Q&A prep |
| [benchmark.md](benchmark.md) | **`night/bench` (the MUST):** claims we can defend, arms, metrics, targets, fairness, verifiability |
| [demo.md](demo.md) | The live "Night Shift Race": screen, dashboard labels, the correction moment, 3-minute script, narration |
| [architecture.md](architecture.md) | System design, Mubit integration and capability map, runners, data model, API |
| [budget.md](budget.md) | GPU (Modal $30) and LLM ($287 + $25) budgets, the LLM cost design, guards, pre-flight checklist |
| [plan.md](plan.md) | Effort per component, team fit, critical path, cut ladder |
| [sapling_research.md](sapling_research.md) | **OPTIONAL:** our fine-tune of Banyan Sapling for spotting fake improvements |
| [demo_script.md](demo_script.md) | Timed 3-minute demo, primary and fallback versions (chosen Sun 08:00), the correction → model update beat, Q&A plan, rehearsal checklist |
| [slides_outline.md](slides_outline.md) | 8 slides with exact copy and the logged numbers for each |
| [evidence_pack.md](evidence_pack.md) | What to have open on stage: console pages, task/adapter/job IDs, request IDs, race IDs, links |
| [qa_mubit_constraints.md](qa_mubit_constraints.md) | Judge Q&A on Mubit's constraints: why no adapter serves, what we did, what Mubit would need to change |
| [preregistration_lcbench_wide.md](preregistration_lcbench_wide.md) | Frozen pre-registration of the LCBench wide race (D47), with the quality-and-cost addendum (D48) |
| [adapter_golive_runbook.md](adapter_golive_runbook.md) | Step-by-step for the first hour after an adapter appears in workspace B or C |
| [mubit_field_report.md](mubit_field_report.md) | Findings for the Mubit team (F1–F15) |
| [mubit_office_hours.md](mubit_office_hours.md) | Questions for Mubit, what their docs and Discord say |
| [sapling_vs_published.md](sapling_vs_published.md) | Our sapling-research v2 vs Mubit's published Banyan Sapling vs Opus 5.5 (R42, R81) |
| [sapling_results.md](sapling_results.md) / [sapling_results_v2.md](sapling_results_v2.md) / [sapling_gate.md](sapling_gate.md) | sapling-research v1/v2 results and the gate |
| [optimisation_analysis.md](optimisation_analysis.md) / [opportunity_benchmarks.md](opportunity_benchmarks.md) / [research_task_pack.md](research_task_pack.md) | Engine analysis, benchmark opportunities, the task pack spec |

The early brainstorming docs (critique of the original idea, pivot options, bank/fraud framing) were folded in on 23 Oct: their still-valid points are in `pitch.md` (Q&A prep) and the rest is in git history before this change.
