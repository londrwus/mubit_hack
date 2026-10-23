# Using Mubit fully: Busy Machines as a solution Mubit can sell

**Positioning:** Busy Machines is the **reference solution for "Mubit for Research"**: the vertical Mubit could take to AI labs during their launch week (26–30 Oct). Every layer of the product runs on a Mubit capability. The pitch to the judges:

> **"We built the AI-lab product you can sell on Monday. Same base URL, three research tasks, and a GPU bill that shrinks every night."**

## 1. Capability map
Status key: ✅ documented in docs.mubit.ai · 📣 on the marketing site or in the launch post (verify at office hours) · ❓ not documented.

| Mubit capability | Status | How Busy Machines uses it |
|---|---|---|
| **Gateway** (base-URL swap) | ✅ | Every LLM call: agent steps, the Opus arm, data labelling. One client module (`mubit_client.py`). |
| **Tasks** (calls with the same instructions, tools and output format) | ✅ | The research steps are **three clean Mubit tasks**: `propose_experiments`, `diagnose_run`, `judge_improvement` (summarising is deterministic code, see doc 18). Per docs: *"put the text that changes per call in the user message, not in the instructions."* Numbers and IDs don't split tasks. |
| **Sessions** (`metadata.user_id` on Anthropic, `metadata.session_id` on OpenAI) | ✅ | One session = one researcher run, e.g. `race-12/mubit/seed-1`. In the console, a session is the agent's full **procedural log**, which is Anil's own phrase. |
| **Request IDs** (`x-middleware-request-id`) | ✅ | Stored on every experiment row, so each GPU result links to the exact decision that caused it. This is the basis of outcome attribution and of the verifiable benchmark. |
| **Recording controls** (`x-middleware-record: full / metadata / off`) | ✅ | **The enterprise answer to "labs won't send their traces":** sensitive steps (e.g. proprietary architectures in `propose_experiment`) can record metadata only. |
| **Adapters + versions + grading** | ✅ | Each research step gets its own small model; `judge_improvement` v1 → v2 is the correction demo. |
| **Try models** (side by side) | ✅ | A live demo moment: adapter vs Opus on the same leakage claim. |
| **Rate reply Good/Bad** (console) | ✅ UI / ❓ effect | The correction channel if there's no API: flag → rating. **Q9.** |
| **OpenAI-compatible services** (`/compat/{openrouter,groq,together,mistral}/v1`) | ✅ | An **open-model arm** (e.g. Qwen/Llama on Groq or Together) *through Mubit*. This covers the "Own your intelligence layer" track, and is the **fallback if adapters can't serve live traffic** (Q1). |
| **Import past calls** (Langfuse, Braintrust, Logfire) | ✅ | **Day-0 onboarding for labs:** import existing research-agent traces, so adapters can train before the first night. Pitch only: imports cover calls before a project's first live call, and there is one project per workspace, so we can't demo it on our project. |
| **Coding-tool integration** (Claude Code, Codex, Gemini CLI, Aider) | ✅ | **Zero-build adoption:** a researcher's existing Claude Code or Codex sessions also flow through Mubit. Busy Machines' task pack works alongside them. |
| **Outcome learning** (link decisions to verified results) | 📣 | **The training metric IS the verified outcome.** `propose_experiment` → val_bpb / PR-AUC / return. Outcomes are attached by request ID. This is how adapters can become *better than the teacher*, not just a copy of it. |
| **Routing to the dedicated model only when it's better** | 📣 (launch) | The Mubit arm runs on adapters automatically once they beat the provider. **Q1.** |
| **Change control / rollback** | 📣 | If the verifier's accuracy drops after an update, roll the adapter back. Show the version history. |
| **Experience reuse** | 📣 | Lessons from one target carry over to the next (e.g. LR-warmup findings from the LM speedrun to the CIFAR speedrun). |
| **Prompt / skill refinement** | 📣 | Reviewable prompt suggestions from experiment history. ⚠️ Applying them creates a new task, so this is a post-event feature. |
| **Banyan Sapling** | ✅ (HF) | Independent verifier of improvement claims (doc 15). |
| **latte/bench** | ✅ | Our `night/bench` is its research counterpart: weight learning on real tasks, costed in GPU $. |
| **Mubit's own metrics**: task success, repeated failures, latency, cost per successful task | 📣 | **Our dashboard shows exactly these four per research step**, plus GPU $. Speak their language. |

## 2. The product Mubit sells: "Mubit for Research"
| Component | What the customer gets |
|---|---|
| **Research Task Pack** | Four prompt templates shaped to form clean Mubit tasks (stable instructions, variable content in the user message, JSON output) |
| **Outcome connector** | Training metrics (W&B / MLflow / custom) attached to Mubit request IDs |
| **Verifier** | Banyan Sapling calibrated for ML-research claims (leakage, overfit, eval overlap) |
| **GPU efficiency dashboard** | $-to-target, GPU idle %, best score at a fixed budget, cost per successful experiment |
| **Onboarding** | Import Langfuse/Braintrust/Logfire traces; swap the base URL in Claude Code/Codex |
| **Data control** | Per-step recording levels (`full / metadata / off`) |

### Customer journey (the sales story)
| When | What happens | What the customer sees |
|---|---|---|
| Day 0 | Base-URL swap + trace import | Nothing changes in their workflow |
| Night 1 | The agent runs on the frontier model through Mubit; tasks fill up | Three research tasks appear in the console |
| Day 2 | Adapters pass grading; routing begins | Step latency drops, GPU idle % falls |
| Week 1 | Corrections and outcomes produce v2, v3… | **$-to-target report: less GPU money, better models** |
| Ongoing | Weekly report: GPU $ saved, quality gained | Expansion to more teams and tasks |

This matches Anil's **"time to visibility"** metric: value shows on Day 2, without forward-deployed engineers.

### Who Mubit sells it to
AI labs only (D18; see doc 12): RL/post-training and eval teams (volume), research teams at mid-size labs and neolabs, AI-scientist startups, and GPU clouds as a channel.

## 3. Demo: show the Mubit console three times
1. **Tasks page:** three research tasks with example counts. *"Our researcher's brain, as three Mubit tasks."*
2. **Requests filtered by session:** one researcher's full trajectory. *"The procedural log."*
3. **Adapters → History:** judge v1 → v2 after corrections, then **Try models** side by side on the leakage claim.

## 4. Constraints from the docs that affect us
- **Starter-credit concurrency limit** returns `429 starter_credit_busy`. Our parallel agents could hit it, so **use our own provider key** for the race.
- **The console signs you out after 15 minutes idle.** Log in fresh right before the demo.
- **One project per workspace.** Keep races distinct by session IDs, not projects.
- **Only admins can import calls or manage keys.** The account owner should be an admin.
- Call timeouts: 10 s to connect, 60 min total per call. Fine for us.

## 5. New open questions
See `decisions.md` Q9: is there an API for ratings, outcomes or corrections, or only the console button?
