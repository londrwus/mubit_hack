# Context: the hackathon, Mubit, and related work

Facts we build on. Sources: the Luma event page, mubit.ai, docs.mubit.ai (Concepts, First adapter, Model IDs), Hoxton Ventures, Anil's posts and launch post (x.com/notanilp/status/2113651825752629256), the Banyan Sapling model card, and our own probes on 23 Oct (`decisions.md`, `results_log.md`).

## 1. Night Shift London: rules, tracks, judging
**Fri 23 Oct, 19:30 hacking → Sun 25 Oct, 10:00 code freeze** (clocks go back at 02:00 on Sunday, which gives us one extra hour). Expo judging at 10:30, finalist demos at 12:30, awards at 13:30.
Prizes: **$15k / $10k / $5k**, judged across all tracks, plus $35k in credits and Mubit credits.

### Hard rules (disqualifiers)
- **Every LLM call must route through Mubit.**
- All code must be written during the event. Open-source libraries are fine.
- At least one team member must attend the Sunday demo.

### Tracks
1. **Recurring tasks:** a repeated task (extraction, triage, support replies) where **corrections become model updates**.
2. **Own your intelligence layer:** move a closed-model workload to a task-adapted open model and **compare quality, cost and latency**.
3. **Fast refresh:** a changing workload where **model updates keep pace**.

### Judging (equal weight)
| Criterion | Verbatim | What it means for us |
|---|---|---|
| Use of Mubit | "Show a correction turning into a model update in your demo." | **Mandatory demo beat.** Show the before output, the human or verifier correction, the new adapter version, and the after output. Precompute it, because training is not instant. |
| Usefulness | "A real recurring task someone would want handled." | It must be high-volume and repetitive. One-off research does not count, which is why we sell the agent's *recurring steps*. |
| Technical execution | "It works, and it holds up under questions." | Real numbers on a held-out set. Honest answers about failure cases. |
| Demo | "Clear, live, and under the time limit." | About 3 minutes, one story, one chart, one live race. |

## 2. Mubit: what it does
Mubit is a **drop-in LLM gateway that turns your recurring LLM calls into small task-specific open models ("adapters")**. You change one base URL. Mubit records every call, groups similar calls into tasks, and once a task has enough examples it trains and tests a small model for it.
Tagline: *"Build reliable agents with a model for every task."* The site also calls it *"agentic runtime calibration."*

### Mechanics (from the docs)
| Concept | Meaning |
|---|---|
| **Gateway** | `https://api.mubit.ai/openai/v1`, `/anthropic`, `/gemini`. Uses `MUBIT_API_KEY`. Model IDs stay the same. |
| **OpenAI-compatible services** | `/compat/{openrouter,groq,together,mistral}/v1`: open models, still through Mubit. |
| **Request** | One recorded call. Response header `x-middleware-request-id`. |
| **Session** | `metadata.user_id` (Anthropic) or `metadata.session_id` (OpenAI). Groups one run's calls in the console. |
| **Recording controls** | `x-middleware-record: full / metadata / off`. |
| **Task** | Calls with the **same model, instructions, tools, tool choice and output format**. The unit of learning. Docs: *"put the text that changes per call in the user message, not in the instructions."* Numbers and IDs don't split tasks. |
| **Kept example** | A call accepted for training/testing. The first training happens at **25 kept examples**. |
| **Adapter** | A small open base model trained for one task. Each training run creates a new version. |
| **Test** | Each version is graded against the **provider model's answers**: `Passed`, `Below target`, `Malformed tool calls`, `Prompt only` (base model + prompt already matches), `Not tested`. |
| **Try models** | Side-by-side comparison of an adapter and the provider. |
| **Limits** | Training must be enabled per workspace. Daily training limit (resets 00:00 UTC), a pause between trainings, **3 adapter starts per org per UTC day**, adapter cold start of up to about 20 minutes. |
| **Imports** | Past calls from Langfuse, Braintrust and Pydantic Logfire (admins only; only calls before a project's first live call). |
| **Coding tools** | Claude Code, Codex, Gemini CLI and Aider work through the gateway. |

The marketing site also lists **outcome learning** (linking decisions to verified results), **experience reuse**, **prompt/skill refinement**, **change control** (evaluate before activation, roll back) and **task-specific models**. Its metrics are **task success, repeated failures, latency, and cost per successful task**. No fixed promise of improvement.

### What we observed in our own probes (23 Oct, details in `decisions.md`)
- All Anthropic models are available, including `claude-opus-5-5`. Prompt caching, structured outputs (`output_config.format`), `count_tokens` and 6 parallel calls all work through the gateway. Mubit overhead is **<1 ms**.
- With our Anthropic key in **Provider keys**, calls are billed to our Anthropic account, not the $25 Mubit credit.
- **Task lifecycle (toy task, R5–R7):** calls are grouped into a task **~10 minutes** after they arrive. An **automatic reviewer** decides which become kept examples (39 of 40 kept), under a **daily review limit**. Training took **~5 minutes**: calls → trained adapter ≈ **20–25 min**.
- **Adapter rollout:** each version takes an exam against a **quality target (85%)**, then goes **candidate → shadow → live**. It only receives traffic once a **route ("request path") points to it**. Our toy v1 scored 70% on 10 test cases and has no route, so it serves nothing yet. The base model was **Qwen3-4B-Instruct-2507** (4B, 256K context).
- The **API response doesn't say who answered** (adapter or provider). The console does: each request has a **"Served by"** field, and each task has **Routes**, **Answered by your model**, **Regression check** and **Match cut-off** fields. Every request has a "Rate this reply: Good/Bad" button.

### Gotchas from the docs
- Starter-credit calls are limited to listed models (`400 starter_credit_model`) and concurrency (`429 starter_credit_busy`). Our provider key avoids both.
- The console signs you out after **15 minutes idle**: log in fresh right before the demo.
- **One project per workspace**: keep races apart by session IDs, not projects.
- Only admins can import calls or manage keys.
- Call timeouts: 10 s to connect, 60 min total per call.

### How Mubit's positioning has moved
Anil's late-September/early-October posts talk about an 85–90% token cut, "agents wait less", cost and efficiency, and look back at an earlier memory / procedural-log positioning (a product called Middleware, middleware.now, handles the design layer). The site and docs now sell **distilling recurring calls into adapters, where corrections become model updates**. Judges will reward the *adapter / model-update loop*; pitching "agent memory" or "prompt optimisation" pitches what they've moved past.

## 3. The 23 Oct launch (launch week 26–30 Oct)
1. **Mubit platform** (public launch). "Turns your agents' work into models you own": dedicated models for recurring tasks, frontier models for everything else. It **switches to the dedicated model only when it performs better**, so adapters *are* used at inference (details still Q1). Third parties report a one-line base-URL swap and traces refreshing every few minutes. Critics asked "won't agents just repeat existing tasks?"; our pitch answers it (recurring steps powering non-repeating discovery).
2. **Banyan Sapling** (`MubitLabs/banyan-sapling`, Apache 2.0), their first open model. It checks whether a claim is supported by the supplied policy and evidence and flags conflicts, missing info and inconsistent context. Model card: 94.22% raw accuracy on 1,731 cases, but only 45% coverage and weak out of domain; they state it has **no proven real-world accuracy**. It matches Anil's "keep the verifier separate from the context" thesis, which is why it's our verifier. Full facts in `sapling_research.md`.
3. **latte/bench** (mubit.ai/bench): agents plan simulated latte pours, get feedback and retry, with and without a retained note. It measures **client-side notes, "not model-weight learning"**, and **does not test held-out transfer**. That's our opening: `night/bench` measures weight learning on real tasks with held-out metrics, costed in GPU $.

## 4. Company and Anil's worldview
- Founder/CEO **Anil Peri** (@notanilp). Founded 2025, HQ Aberdeen. Backed by **Hoxton Ventures** (Hoxton IV, 2026; partners Hussein Kanji and Rishabh Kaul).
- Customers: **we couldn't verify Spotify, Salesforce or Tokio publicly** and the site lists no logos. Don't name them on stage unless Mubit does first.
- Judges aren't announced (Q7). Expect Anil, Mubit engineers and probably a Hoxton partner.

Use his vocabulary:
1. **Token efficiency is about agent wait time, not dollars.** "Agents wait less and do more." Latency is a first-class metric.
2. **Keep the verifier separate from the context layer.** A verifier that sees the same context becomes biased; a judge must not share context with the actor.
3. "Steering agents using procedural logs was a problem no one thought they needed."
4. **"Continual learning is a process, not the goal."** Don't sell it as a buzzword.
5. **After the shiny parts, only cost and efficiency matter.**
6. "Combine this with **parallelism and cloud agents**, you have the **full utilisation cycle**."
7. **"Managing a farm of models that continuously evolve from your pointers" is herculean**, and so is keeping it low-latency.
8. Product-first, no forward-deployed engineers. **The metric is "time to visibility"**: how fast the value becomes visible.

## 5. Related work and competitors
### Rekursiv (rekursiv.ai): the closest competitor
- "AI Scientists" that form hypotheses, run experiments, review each other's work and update leaderboards. Founders are ex-Google Research, DeepMind and Luma (TensorFlow Probability, VideoPoet, PyMC, BlackJAX).
- Self-reported: new ML methods beating state of the art; a factory scheduler winning 29 of 30 problems; a 15M-parameter RL agent beating Craftax's final boss in 44% of games; state of the art on Karpathy's nanochat in 3 days.
- **priml** (Apache 2.0): typed building blocks for A/B-testing ML research. Baselines: **nanochat** (300-second budget, validation bits-per-byte) and sudoku; includes a FLOPs/roofline cost model. No results tables. Usable as a starting point for our LM target.
- **trackinizer** (Apache 2.0): an "epistemological DB" linking every claim to the experiments behind it. No benchmarks.

**What this means:** Rekursiv competes on *"agents that do better research."* We don't. Our angle is the one they lack: **the research agent's own LLM calls become cheap owned models that improve from corrections**, so the same budget buys more experiments.

### Benchmarks for ML-research agents
| Benchmark | By | What it is | Use for us |
|---|---|---|---|
| **MLE-bench** | OpenAI | 75 Kaggle competitions, scored by medals ("lite" split exists) | Too heavy. Cite, don't run |
| **RE-Bench** | METR | 7 open-ended ML research environments, **fixed time budget**, scored against human experts | **We copy its fixed-budget method** |
| **MLGym** | Meta | 13 AI-research tasks (vision, NLP, RL, game theory), Gym-style | Possible extra tasks; not planned |
| **MLAgentBench** | Stanford | 13 tasks improving a baseline `train.py` | Possible extra tasks; not planned |
| **Speedruns** (nanoGPT speedrun, nanochat, airbench) | community | Best val loss / accuracy within a fixed training budget | **Our targets** (`benchmark.md`) |
| DSBench / ScienceAgentBench / PaperBench / CORE-Bench | various | Data science, scientific coding, paper reproduction | Off-target; mention at most |
| **Craftax** / **gymnax** | Matthews et al. / community | JAX RL environments | gymnax is our RL target |

Using public task families and their own scoring is what stops judges dismissing the result as a benchmark built to win.
