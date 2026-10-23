# Mubit: what they actually do (as of 23 Oct 2026)

Sources: mubit.ai, docs.mubit.ai (Concepts, First adapter, Model IDs), Hoxton Ventures portfolio page, the Luma event page.

## One-line summary
Mubit is a **drop-in LLM gateway that turns your recurring LLM calls into small task-specific open models ("adapters")**. You change one base URL. Mubit records every call, groups similar calls into tasks, and once a task has enough examples it trains and tests a small model for it.

Tagline: *"Build reliable agents with a model for every task."* The site also calls it *"agentic runtime calibration."*

## The mechanics (from the docs)
| Concept | Meaning |
|---|---|
| **Gateway** | `https://api.mubit.ai/openai/v1`, `/anthropic`, `/gemini`. Uses `MUBIT_API_KEY`. Model IDs stay the same. |
| **Request** | One recorded call. |
| **Task** | Calls with the **same model, instructions, tools, tool choice and output format**. This is the unit of learning. |
| **Kept example** | A call accepted for training/testing. By default the first training happens at **25 kept examples**. |
| **Adapter** | A small open base model trained for one task. Each training run creates a new version. |
| **Test** | Each version is graded against the **provider model's answers**: `Passed`, `Below target`, `Malformed tool calls`, `Prompt only` (base model + prompt already matches), `Not tested`. |
| **Try models** | A side-by-side comparison of an adapter and the provider. |
| **Limits** | Training must be enabled per workspace. Daily training limit (resets 00:00 UTC), a pause between trainings, **3 adapter starts per org per UTC day**, and an adapter cold start of up to about 20 minutes. |
| **Imports** | Past calls can be imported from Langfuse, Braintrust and Pydantic Logfire. |
| **Coding tools** | Works with Claude Code, Codex, Gemini CLI and Aider through the gateway. |

The marketing site also lists **outcome learning** (linking decisions to verified results), **experience reuse**, **prompt/skill refinement**, **change control** (evaluate before activation, roll back) and **task-specific models**. Its metrics are **task success, repeated failures, latency, and cost per successful task**. The site makes no fixed promise of improvement.

## How the product has changed
- **Anil's posts (Sep 29 and Oct 1. X omits the year for current-year posts, so these are very likely from 2026, i.e. weeks old):** an 85–90% token cut, "agents wait less", cost and efficiency. Some of them look back at the earlier memory / procedural-log positioning. A product called Middleware (middleware.now) handles the design layer.
- **Now (site and docs):** distilling recurring calls into adapters, where **corrections become model updates**. The memory and token-efficiency story has become an **owned, cheap, fast, per-task model** story.
- **What this means for us:** Judges will reward the *adapter/model-update loop*. Pitching "agent memory" or "prompt optimization" means pitching what they have already moved past.

## Company
- Founder/CEO: **Anil Peri** (@notanilp). Founded 2025, HQ Aberdeen, Scotland. Backed by **Hoxton Ventures** (Hoxton IV, 2026; partners Hussein Kanji and Rishabh Kaul).
- Customers: **I could not verify Spotify, Salesforce or Tokio publicly.** The site lists no logos. Don't name them on stage unless Mubit does first.
- No public model releases or changelog. Their "models" are the per-customer adapters.

## Anil's worldview (from the posts you pasted). Use this vocabulary.
1. **Token efficiency is about agent wait time, not dollars.** "Agents wait less and do more." Latency is a first-class metric.
2. **Keep the verifier separate from the context layer.** A verifier that sees the same context becomes biased. This shapes architecture: a judge must not share context with the actor.
3. **"Steering agents using procedural logs was a problem no one thought they needed."**
4. **"Continual learning is a process, not the goal."** Don't sell it as a buzzword.
5. **After the shiny parts, only cost and efficiency matter.**
6. **"Managing a farm of models that continuously evolve from your pointers" is herculean**, and so is keeping it low-latency.
7. **Product-first, no forward-deployed engineers. The metric is "time to visibility":** how fast the value becomes visible.

## Open gaps
- **Judges are not announced.** Expect Anil, Mubit engineers and probably a Hoxton partner. We should check Luma and X at kickoff.
- The X profile can't be fetched without login, so this file relies only on the posts you pasted.
- The docs don't explain how **corrections** enter training (edited outputs? feedback API?). **Ask at office hours on Saturday morning**, because criterion #1 depends on it.
