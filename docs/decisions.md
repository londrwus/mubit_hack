# Decisions and open questions

The source of truth. If another doc disagrees with a decision here, the decision wins (and fix the doc). Add new decisions at the bottom with a date; when one replaces another, say so in both rows.

## Decisions
| # | Date | Decision | Why | Ref |
|---|---|---|---|---|
| D1 | 2026-10-23 | Build an **ML research agent whose LLM calls run through Mubit** (named Busy Machines, D11) | Broad (any ML/DL/RL model whose training can be scripted) and technically true: Mubit improves the *agent's* LLM calls, not the models directly | pitch |
| D2 | 2026-10-23 | The pitch is **cost per discovery**: $-to-target, time-to-target, GPU idle %. Audience narrowed to AI labs by D18; stage wording set by D19 | Relatable to GPU-bound labs; matches Anil's "agents wait less, do more" | pitch |
| D3 | 2026-10-23 | The "beat Opus 5.5" claim applies **at a fixed GPU/time/$ budget**, not per call | Adapters are graded against the provider, so a per-call win is unprovable | benchmark |
| D4 | 2026-10-23 | Benchmark targets: **LM pretraining speedrun** (GPU), **RL on gymnax** (CPU, D21), **CIFAR-10 speedrun** (GPU); stretch: tiny-LM post-training. (The original fraud target was dropped by D18) | Standard, cheap, verifiable, and the work labs repeat thousands of times | benchmark |
| D5 | 2026-10-23 | Arms: Opus 5.5 agent, Mubit agent, and an **Optuna/random-search baseline** | Pre-empts "why not just Optuna?" | benchmark |
| D6 | 2026-10-23 | Correction moment: a **planted contamination trap** → flag → adapter v1 → v2 | Fulfils judging criterion #1, dramatically, with ground truth | demo |
| D7 | 2026-10-23 | Use **Banyan Sapling** as the independent verifier of improvement claims | Uses Mubit's own launch; matches the "separate verifier from context" thesis | context, sapling_research |
| D8 | 2026-10-23 | Stack: FastAPI + SQLite, Vite React + shadcn/ui + Tailwind, Pen.dev, Modal | Team choice | architecture |
| D9 | 2026-10-23 | Compute: Modal Starter, $30. **L4 is the default GPU**; H100 only where needed (Sapling, D21; a replication run only if money remains) | Budget | budget |
| D10 | 2026-10-23 | Commercial lead: **RL/post-training and eval teams** (volume) and **research teams at mid-size labs and neolabs**; frontier labs are credibility, not customers. (Enterprise ML was dropped by D18) | Fit with Mubit's usage model | pitch |
| D11 | 2026-10-23 | Project name: **Busy Machines** (working name was "Night Scientist") | The name is the pitch (GPU idle % → 0). Avoids "cheap" (Anil: efficiency, not cost, is the motivation) and "continual learning" (a phrase he dropped) | — |
| D12 | 2026-10-23 | Primary pitch: **"cost and efficiency" for AI labs' ML/DL/RL experimentation**. Cost = less GPU money; efficiency = better models from the same compute | Anil's own words; scoped to the experiment loop, not giant pretraining runs, to stay defensible | pitch |
| D13 | 2026-10-23 | Position Busy Machines as **"Mubit for Research"**, a sellable vertical using every Mubit capability | Maximises "Use of Mubit"; gives Mubit a product to sell in launch week | pitch, architecture |
| D14 | 2026-10-23 | Mubit $25 balance = development; the race and example collection use **our own Anthropic key ($287) added as a Mubit provider key**, so calls still route through Mubit. Training runs are kept **short (20–90 s)** so LLM wait time is a meaningful share. Observed: with the provider key added, our calls bill to Anthropic, not the $25 | $25 ≈ 800 Opus calls vs ~2k needed; with 5-min runs the idle effect is too small to matter | budget |
| D15 | 2026-10-23 | **Claude Code (the Opus 5.5 subscription session) is a dev tool only:** writing code, docs, data generators and reviews. **It is never the product's LLM:** not the Opus arm, not example collection, not labelling or verification | Those calls would bypass Mubit (breaking the hard rule), have no request IDs or cost/latency data (the benchmark couldn't be verified), and Mubit can't train adapters from calls it never saw | budget |
| D16 | 2026-10-23 | LLM cost design: **batch K proposals per call**, `summarize_result` → code, diagnose only on failures, judge only on a new best, **stateless cached calls** (system → target spec → dynamic digest), structured outputs, effort per step from small measured sweeps, `x-middleware-record: off` on non-example calls. The Opus arm gets the same optimisations. No Batch API | ~10× cheaper per experiment (estimate); fair vs an optimised Opus baseline | budget |
| D17 | 2026-10-23 | **The product is compute-agnostic.** Training goes through a pluggable `Runner` (`local`, `modal`; Slurm/K8s/Ray are pitch-only). Modal is only our test/demo compute. GPU $/hour is configurable per runner | Labs run on their own clusters; we're not building for Modal | architecture |
| D18 | 2026-10-23 | **AI labs are the only focus.** Banks and fraud are dropped from the pitch, targets and demo. The leakage trap becomes a **contamination trap** (val/test overlap with train, eval on training seeds). Buyers: RL/post-training and eval teams, research teams at mid-size labs and neolabs, AI-scientist startups; GPU clouds as a channel | One clear buyer; contamination is a real, current AI-lab pain. Updates D2, D4, D6, D10 | pitch |
| D19 | 2026-10-23 | **Plain language on stage and on screen.** Three promises: spend less on GPUs, get a better model, train faster. No ML/DL/RL jargon in the demo or UI labels; technical depth only in Q&A, the results export and the docs | The audience isn't all ML people; a demo must be understood in seconds | pitch, demo |
| D20 | 2026-10-23 | **Priorities. MUST:** a fair benchmark vs Opus 5.5 proving we **train faster, get a better model, and spend less**. **OPTIONAL:** `sapling-research`, our fine-tune of Banyan Sapling for spotting fake improvements in AI-research results, published with a benchmark. Go/no-go Sat 14:00, hard stop Sat 18:00 | The benchmark wins the rubric; the model makes it memorable | benchmark, sapling_research |
| D21 | 2026-10-23 | Plan around Modal Starter limits: **10 GPUs** (4 per arm + 1 H100 for Sapling + 1 spare) and **100 CPU containers**. **The RL target moves to CPU**, 32 workers per arm, all seeds in parallel. GPU seeds run sequentially; one GPU target gets the full 3 × 1 h, the other 30-minute races | Fits $30 and turns the concurrency limit into the most dramatic version of the race | budget, benchmark |
| D22 | 2026-10-23 | **No local GPUs: Modal ($30, 10 GPUs / 100 containers) is the only compute.** The laptop runs only dev/mock and tiny CPU tests | Corrects an earlier assumption that we had GPUs | budget |
| D23 | 2026-10-23 | **Sapling LoRA/full fine-tune dropped.** The plan ends at stage 2 (frozen backbone, retrained verification head + recalibration). Say "fine-tuned the verification head, backbone frozen"; never claim a full or LoRA fine-tune; if only calibration works, say "recalibrated" | Saves time and GPU money; stage 2 is genuine fine-tuning, and precise wording survives questions from the people who built Sapling | sapling_research |

## Answers from our own probes (2026-10-23, ~22:30 BST, Anthropic route, `x-middleware-record: off`)
Measurements are also in `results_log.md` (R1–R5).

| Question | Result |
|---|---|
| Models available through Mubit | ✅ **All**, including `claude-opus-5-5`, `sonnet-5-5`, `haiku-5-5`, `fable-5-1` |
| Prompt caching passthrough | ✅ 1st call `cache_write` 10,353 tokens; identical 2nd call `cache_read` 10,353 |
| Structured outputs passthrough | ✅ `output_config.format` json_schema returned valid JSON |
| Recording headers + `metadata.user_id` | ✅ Accepted. ❓ Whether `off` fully excludes a call from training: check in the console (Q11) |
| Concurrency | ✅ 6 parallel calls, no `429` |
| `count_tokens` | ✅ Works |
| Batch API (`/v1/messages/batches`) | ⚠️ GET returns 200 (undocumented). Unknown whether batch calls are recorded as examples, so **not used** |
| Latency (tiny calls) | ~0.6–1.0 s (Haiku/Sonnet/Opus). Structured toy-triage calls on Haiku 5.5: 1.0–3.0 s |
| Mubit overhead | **<1 ms** (console "Where the time went"; provider took 1.46 s) |
| Response headers | Only `x-middleware-request-id` from Mubit. **No header shows adapter vs provider** |
| Console request details | **"Served by"** field (provider vs Mubit model): a strong hint that Q1 = automatic routing. Session shown. "Rate this reply: Good/Bad" on every request |
| Billing | Provider keys: Anthropic key **"Valid"**. Our calls bill to **our Anthropic account**, not the $25 Mubit credit |
| Task lifecycle (toy task "Training-run triage", 40 Haiku calls, session `toy-triage`) | Calls **grouped ~10 min after arrival** ("same instructions and tools"). An **automatic reviewer** then decides which become kept examples: "40 waiting for review · 2 running · 0 over today's limit", so review is **slow (~2 at a time) and has a daily limit**. 25 kept needed before training. Task fields: **Routes**, **Answered by your model**, **Regression check**, **Match cut-off**. Category: **Learning** (others: Live, Needs review). After ~15 min: still 0 of 25 kept. **Update ~22:52:** review then finished in one batch (**39 kept, 1 rejected**), training ran ~5 min, adapter v1 exists. Calls → trained adapter ≈ 20–25 min (R6) |
| Toy adapter v1 (R7) | Base **Qwen3-4B-Instruct-2507** (4B, 256K context). **70% quality on 10 test cases vs an 85% target** → exam "not passed (advisory)". Rollout is staged: **candidate → shadow → live**, and it needs a **route ("request path" / "pathway") pointing to the adapter**. None exists, so nothing is shadowed or served. "Figures unavailable" on the task page = the Q12 metrics error again. The task page says "text replies" although we sent a JSON schema: check that the adapter keeps the output format |
| Console error | *"reading lane … metrics needs manager; a key restricted to a project reads that project's metrics…"*: a permissions limit on metrics (Q12). Task details still load |

## Open questions (ask Mubit at office hours)
| # | Question | Status | Blocks |
|---|---|---|---|
| Q1 | Do adapters serve **live inference** from our code (automatic routing)? How do we observe, per call, whether an adapter answered, and can we force it for the benchmark? | Partly answered: the launch says Mubit routes to the dedicated model only when it's better; the console shows "Served by" and "Routes". The API response doesn't. Toy adapter (R7): routing needs an **enabled route/pathway** and passes through a **shadow** stage after the exam. Ask: how do we create a route, does shadow need the 85% target, how much shadow traffic/time before live? | The whole Mubit arm |
| Q2 | How do **corrections** enter training? Can a corrected output override the provider's label? | Open | The correction demo and Claim 2 ("v2 beats Opus on the trap") |
| Q3 | Can our training limits be raised (3 adapter starts per day, the daily training limit)? | Open | The overnight schedule |
| Q4 | Which **base models** are served for adapters? | **Answered for the toy task:** Qwen3-4B-Instruct-2507 (4B). Ask whether a bigger base can be chosen for harder steps | Latency estimates |
| Q5 | Does running **Banyan Sapling** ourselves (not an API call) violate "every LLM call through Mubit"? | Open | Verifier design |
| Q6 | Are any of the $35k hackathon credits **Modal/GPU** credits? | Open | More GPUs per arm, more seeds |
| Q7 | Who are the judges? | Not announced as of 23 Oct | Pitch tuning |
| Q8 | Would Mubit co-release `sapling-research` or link it from the Sapling card? | Open | The Sapling pitch line |
| Q9 | Is there an **API** for ratings, outcomes or corrections (beyond the console's Good/Bad button)? Does a rating affect training? Can outcome scores (e.g. val_bpb) be attached to a request ID? | Open | Correction automation, outcome learning |
| Q10 | How is **adapter inference billed**, and from which balance? | Open (model availability and provider billing answered by probes) | LLM budget |
| Q11 | Does `x-middleware-record: off` fully exclude a call from training examples? | Open (caching and structured outputs ✅ verified) | Keeping tasks clean |
| Q12 | The "metrics needs manager" console error with a project-restricted key: what role do we need? | Open | Reading task metrics |
| Q13 | Can example review be sped up or its daily limit raised? (R5: 0 of 40 decided after ~15 min) | **Mostly answered (R6):** review finished in one batch, ~20–25 min calls → trained. Still ask about the daily review limit | Getting 25 kept examples per task in time |
| Q15 | Does adapter serving **enforce `output_config.format`** (JSON schema), or only imitate it? Is "Try it" latency representative of routed latency? (R8–R9: malformed JSON on 3 of 7 prompts; ~1.6 s per valid reply, same as Haiku) | Open | Parsing in the race loop; the speed claim |
| Q14 | Is the **85% quality target** fixed? What raises quality most: more examples, retraining settings, a bigger base? Does a retraining count as one of the 3 daily adapter starts? | Open | Getting adapters past the exam in time |
