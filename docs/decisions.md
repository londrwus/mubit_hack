# Decisions and open questions

The source of truth. Add new entries at the bottom of each section with a date.

## Decisions
| # | Date | Decision | Why | Ref |
|---|---|---|---|---|
| D1 | 2026-10-23 | Build **Busy Machines** (working name was "Night Scientist", see D11): an ML research agent whose LLM calls run through Mubit | Broad (any ML/DL/RL model) and technically true: Mubit improves the *agent's* LLM calls, not the models directly | 05 |
| D2 | 2026-10-23 | The pitch is **cost per discovery**: $-to-target, time-to-target, GPU idle % | Relatable to labs and banks; matches Anil's "agents wait less, do more" | 10 |
| D3 | 2026-10-23 | The "beat Opus 5.5" claim applies **at a fixed GPU/time/$ budget**, not per call | Adapters are graded against the provider, so a per-call win is unprovable | 06 |
| D4 | 2026-10-23 | Benchmark tasks: nanochat-style LM (GPU), RL (Craftax/gymnax), fraud (CPU); optionally MLGym/MLAgentBench tasks | Standard, cheap and verifiable; covers labs and banks | 07, 08 |
| D5 | 2026-10-23 | Arms: Opus 5.5 agent, Mubit agent, and an **Optuna/random-search baseline** | Pre-empts "why not just Optuna?" | 12 |
| D6 | 2026-10-23 | Correction moment: a **planted leakage trap**, then a flag, then adapter v1 → v2 | Fulfils judging criterion #1, dramatically | 09 |
| D7 | 2026-10-23 | Use **Banyan Sapling** as the independent verifier of improvement claims | Uses Mubit's own launch; matches the "separate verifier from context" thesis | 11 |
| D8 | 2026-10-23 | Stack: FastAPI + SQLite, Vite React + shadcn/ui + Tailwind, Pen.dev, Modal | Team choice | 13 |
| D9 | 2026-10-23 | Compute: Modal Starter, $30. L4 is the default GPU; H100 only for the final replication | Budget | 08 |
| D10 | 2026-10-23 | Commercial lead: RL/eval teams (volume) and enterprise ML (ease of sale); frontier labs are not customers | Fit with Mubit's usage model | 12 |
| D23 | 2026-10-23 | **Sapling stage 3 (LoRA) dropped.** The plan ends at stage 2 (frozen backbone, retrained verification head + recalibration). Describe it precisely: "fine-tuned the verification head, backbone frozen". Never claim a full or LoRA fine-tune; if only calibration works, say "recalibrated" | Saves time and GPU money; stage 2 is genuine fine-tuning, and precise wording survives judges' questions (Mubit built Sapling) | 21 |
| D22 | 2026-10-23 | **No local GPUs: Modal ($30, 10 GPUs / 100 containers) is the only compute.** The laptop runs only dev/mock and tiny CPU tests. Sapling stage 3 depends on extra GPU credits | Corrects the earlier assumption in doc 07 ("we have GPUs") | 08, 21 |
| D21 | 2026-10-23 | Plan around Modal Starter limits: **10 GPUs** (4 per arm + 1 H100 for Sapling + 1 spare) and **100 CPU containers**. **The RL target moves to CPU**, with 32 workers per arm and all seeds in parallel; this is the cheap, large-scale "GPUs waiting" demo. GPU seeds run sequentially; one GPU target gets the full 3 × 1 h, the others 30-minute races | Fits the $30 budget and turns the concurrency limit into the most dramatic version of the race | 08 |
| D20 | 2026-10-23 | **Priorities. MUST:** a fair benchmark vs Opus 5.5 proving we **train faster, get a better model, and spend less**. **OPTIONAL (stretch "wow"):** our own model, a fine-tune of Mubit's Banyan Sapling for spotting fake improvements in AI-research results, published with a benchmark so Mubit can ship or sell it. Staged with go/no-go gates (Sat 14:00, hard stop Sat 18:00) | The benchmark is what wins the rubric; the model is what makes it memorable | 06, 21 |
| D19 | 2026-10-23 | **Plain language on stage and on screen.** Three promises: spend less on GPUs, get a better model, train faster. No ML/DL/RL jargon in the demo or UI labels; technical depth only in Q&A, the results export and the docs | The audience isn't all ML people; a demo must be understood in seconds | 20 |
| D18 | 2026-10-23 | **AI labs are the only focus.** Banks and fraud are dropped from the pitch, targets and demo. Targets: **LM pretraining speedrun** (nanochat/nanoGPT), **RL** (gymnax/Craftax), **vision speedrun** (CIFAR-10, airbench-style); stretch: tiny-LM post-training. The leakage trap becomes a **contamination trap** (val/test overlap with train, eval on training seeds). Buyers: RL/post-training and eval teams, research teams at mid-size labs and neolabs, AI-scientist startups; GPU clouds as a channel | One clear buyer; contamination is a real, current AI-lab pain; supersedes the bank parts of D2, D4, D10, D17 | 12, 19 |
| D17 | 2026-10-23 | **The product is compute-agnostic.** Training goes through a pluggable `Runner` (`local`, `modal`; Slurm/K8s/Ray are pitch-only). Modal is only our test/demo compute. GPU $/hour is configurable per runner | Labs and banks run on their own clusters; we're not building for Modal | architecture |
| D16 | 2026-10-23 | LLM cost design: **batch proposals (K per call)**, `summarize_result` → code, diagnose only on failures, judge only on a new best, **stateless cached calls** (system → target spec → dynamic digest), structured outputs, effort per step decided by small measured sweeps, `x-middleware-record: off` on non-example calls. The baseline arm gets the same optimisations. No Batch API (not proxied by Mubit) | ~10× cheaper per experiment (estimate); fair vs an optimised Opus baseline | 18 |
| D15 | 2026-10-23 | **Claude Code (the Opus 5.5 subscription session) is a dev tool only:** writing code, docs, data generators and reviews. **It is never the product's LLM:** not the Opus arm, not example collection, not labelling or verification | Those calls would bypass Mubit (breaking the hard rule), have no request IDs or cost/latency data (the benchmark couldn't be verified), and Mubit can't train adapters from calls it never saw | 17 |
| D14 | 2026-10-23 | Mubit $25 balance = dev + example collection; the race uses **our own Anthropic key ($287 balance) added as a Mubit provider key**, so calls are still routed through Mubit. Training runs are kept **short (20–90 s)** so LLM wait time is a meaningful share | $25 ≈ 800 Opus calls vs ~2k needed; with 5-min runs the idle effect is too small to matter | 17 |
| D13 | 2026-10-23 | Position Busy Machines as **"Mubit for Research"**, a sellable vertical using every Mubit capability (gateway, tasks, sessions, request IDs, recording controls, adapters, Try models, compat open models, imports, Sapling) | Maximises "Use of Mubit"; gives Mubit a product to sell in launch week | 16 |
| D12 | 2026-10-23 | Primary pitch: **"cost and efficiency" for AI labs' ML/DL/RL experimentation**. Cost = less GPU money; efficiency = better models from the same compute | Anil's own words; scoped to the experiment loop, not giant pretraining runs, to stay defensible | 14 |
| D11 | 2026-10-23 | Project name: **Busy Machines** | The name is the pitch (GPU idle % → 0). Avoids "cheap" (Anil: cost isn't the motivation, efficiency is) and "continual learning" (a phrase he dropped) | — |

## Answers from our own probes (2026-10-23, ~22:30 BST, Anthropic route, `x-middleware-record: off`)
| Question | Result |
|---|---|
| Models available through Mubit (Q10 part) | ✅ **All**, including `claude-opus-5-5`, `sonnet-5-5`, `haiku-5-5`, `fable-5-1`. Opus 5.5 answered a tiny call |
| Prompt caching passthrough (Q11) | ✅ **Works.** 1st call `cache_write` 10,353 tokens; identical 2nd call `cache_read` 10,353 |
| Structured outputs passthrough (Q11) | ✅ **Works.** `output_config.format` json_schema returned valid JSON |
| Recording headers + `metadata.user_id` | ✅ Accepted (`off`, `metadata`). ❓ Whether `off` fully excludes a call from training must be checked in the console |
| Concurrency | ✅ 6 parallel calls, no `429` |
| `count_tokens` | ✅ Works |
| Batch API (`/v1/messages/batches`) | ⚠️ **GET returns 200** (undocumented). Unknown whether batch calls are recorded as Mubit examples, so **still not used** for anything that must be learned from |
| Latency via Mubit (tiny calls) | ~0.6–1.0 s (Haiku/Sonnet/Opus) |
| Response headers | Only `x-middleware-request-id` from Mubit (plus Anthropic's own). **No header shows adapter vs provider yet** |
| Billing | Console → Provider keys: **Anthropic key "Valid"** (our $287 key). Our calls run on **our Anthropic account**, not the $25 Mubit credit. The console shows "Connected, 25+ calls in the last hour" |
| Tasks after 40 toy calls | ❓ The console says **"no tasks left"**. Meaning unclear: no task formed yet, or a plan/quota limit? Investigating |
| Request details (console) | **"Served by: Your provider"** field exists, so the console shows whether the provider or a Mubit model answered (**strong hint that Q1 = automatic routing**; the API response doesn't expose it). **Mubit overhead "<1 ms"**; the provider took 1.46 s. Session shown (`toy-triage`). "Rate this reply: Good/Bad" is on every request. **No Task field shown yet** |
| Requests list (console) | All toy calls show **Task "—"**, **Served by "Your provider"**, 1.0–3.0 s. Combined with "no tasks left", **our calls are not being grouped into tasks**. ~~Most likely our workspace has no task/training allowance enabled.~~ **B1 resolved:** grouping was just **delayed by ~10 minutes** |
| Task lifecycle (toy task "Training-run triage", 40 calls) | **Grouped ~10 min after the calls** ("same instructions and tools"). Then an **automatic reviewer** decides which calls become kept examples: "40 waiting for review · 2 running · 0 over today's limit", so **review has a daily limit**. Needs **25 kept** before the first training. Console fields seen: **Routes** (none yet; this is where small-model serving shows → Q1), **Answered by your model**, **Regression check**, **Match cut-off**. ⏳ Next: watch the kept/rejected ratio and whether training starts |
| Tasks page (~15 min later) | Still "Reviewing examples, 0 of 25". Task category: **Learning** (others: Live, Needs review). **Review is slow** (~2 at a time), so it's a pipeline bottleneck: start real tasks early. Console error: *"reading lane … metrics needs manager; a key restricted to a project reads that project's metrics…"*, a **permissions bug/limit on metrics**. Task details still load. Q12: ask Mubit about the "needs manager" metrics error |
| Q1 (live routing to adapters) | ⏳ **Experiment running:** seeded a throwaway task "training-run triage" (40 recorded `claude-haiku-5-5` calls, structured output, user_id `toy-triage`). Next: check the console for task status / training, then re-call and compare `model`, headers and latency |

## Open questions (ask Mubit at office hours)
| # | Question | Blocks |
|---|---|---|
| Q1 | Can adapters be used for **live inference** from our code (automatic routing), or only in Try models? | The whole Mubit arm |
| Q2 | How do **corrections** enter training? Can a corrected output override the provider's label? | The correction demo, and the "v2 beats Opus on leakage" claim |
| Q3 | Can our training limits be raised (3 adapter starts per day, the daily training limit)? | The overnight schedule |
| Q4 | Which **base models** are served for adapters? | Latency estimates |
| Q5 | Does running **Banyan Sapling** locally (not an API call) violate "every LLM call through Mubit"? | Verifier design |
| Q6 | Are any of the $35k hackathon credits **Modal/GPU** credits? | Scaling to 10 GPUs per arm |
| Q7 | Who are the judges? (Not announced as of 23 Oct.) | Pitch tuning |
| Q11 | Does Mubit pass through **`cache_control`** (cache reads show in `usage`) and **`output_config.format`** (structured outputs)? Does `x-middleware-record: off` fully exclude a call from training examples? | Cost levers in doc 18 |
| Q10 | How is **adapter inference billed**, and does it draw from the $25 Mubit balance? Is `claude-opus-5-5` on the starter-credit model list? | LLM budget (doc 17) |
| Q9 | Is there an **API** for ratings, outcomes or corrections (beyond the console's Good/Bad button)? Does a rating affect training? Can outcome scores (e.g. val_bpb) be attached to a request ID? | Correction automation, outcome learning |
| Q8 | Can trained adapters be **exported** (weights) to publish on Hugging Face? Would Mubit co-release or link an ML-research extension of Banyan Sapling? | The HF release (doc 15) |
