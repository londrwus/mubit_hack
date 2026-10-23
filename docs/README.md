# Docs index

The docs are numbered in the order they were written, so later docs refine or override earlier ones. **If two docs conflict, the higher number wins**, and `decisions.md` beats all of them.

## Start here (current plan)
| Doc | What it is |
|---|---|
| [decisions.md](decisions.md) | **Source of truth:** decisions made and open questions |
| [results_log.md](results_log.md) | **Every measurement, including ones where we're worse.** The only source for numbers on slides |
| [20_plain_language.md](20_plain_language.md) | **Stage and UI language:** the three promises, plain-word pitch and narration, dashboard labels, words to avoid |
| [19_ai_labs_focus.md](19_ai_labs_focus.md) | **AI labs only:** personas, targets (LM / RL / CIFAR speedruns), contamination trap |
| [architecture.md](architecture.md) | Planned system design, data model and API |
| [22_implementation_difficulty.md](22_implementation_difficulty.md) | **How hard each part is**: hours, risks, team-size fit, critical path, cut ladder |
| [16_full_mubit_integration.md](16_full_mubit_integration.md) | **Every Mubit capability mapped to the product**; "Mubit for Research" packaging and sales story |
| [14_pitch_options.md](14_pitch_options.md) | **Pitch wording:** taglines, the 30-second pitch, the four-pillar slide (money / speed / busy / better) |
| [10_pitch_cost_per_discovery.md](10_pitch_cost_per_discovery.md) | **Current pitch:** cost per discovery, GPU idle %, $-to-target |
| [09_demo_showcase.md](09_demo_showcase.md) | The live "Night Shift Race" demo, the 3-minute script and the correction moment |
| [06_benchmark_beat_opus.md](06_benchmark_beat_opus.md) | Benchmark design and which claims are defensible |
| [21_sapling_research_model.md](21_sapling_research_model.md) | **Optional "wow" piece:** our fine-tune of Banyan Sapling for spotting fake improvements in AI research, staged plan, gates, benchmark |
| [15_hf_model_release.md](15_hf_model_release.md) | Earlier HF-release options (**superseded by 21**) |
| [11_launch_day_news.md](11_launch_day_news.md) | Mubit's 23 Oct launch: platform, Banyan Sapling, latte/bench |

## Context and research
| Doc | What it is |
|---|---|
| [01_mubit_research.md](01_mubit_research.md) | What Mubit does, how adapters work, Anil Peri's worldview |
| [02_hackathon_brief.md](02_hackathon_brief.md) | Rules, tracks, judging criteria, practical limits |
| [07_existing_benchmarks.md](07_existing_benchmarks.md) | MLE-bench, RE-Bench, MLGym, MLAgentBench, nanochat; Rekursiv as a competitor |
| [08_modal_compute_budget.md](08_modal_compute_budget.md) | Modal prices and how the $30 budget is allocated |
| [18_llm_cost_optimization.md](18_llm_cost_optimization.md) | **LLM cost design**: fewer calls, caching, structured outputs, effort, dev modes, guards, pre-flight checklist |
| [17_llm_budget.md](17_llm_budget.md) | Mubit $25 balance, Anthropic prices, call estimates, short-run design insight |
| [12_commercial_view_as_mubit.md](12_commercial_view_as_mubit.md) | AI-lab buyer segments, go-to-market, attacks on our own idea |
| [13_tech_stack.md](13_tech_stack.md) | Stack summary |

## History (superseded, kept for reasoning)
| Doc | Status |
|---|---|
| [03_idea_attack.md](03_idea_attack.md) | Critique of the original "ML sandbox / beat Opus / fraud" idea. Its points still apply as Q&A prep. |
| [04_recommended_pivots.md](04_recommended_pivots.md) | Options A/B/C. **Superseded by 05 and 10.** Option A (the RL judge) lives on as a commercial segment in 12. |
| [05_broad_pitch_reframe.md](05_broad_pitch_reframe.md) | The "research agent + Mubit" reframe. **Refined by 10.** |
