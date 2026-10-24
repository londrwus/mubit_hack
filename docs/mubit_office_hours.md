# Mubit office hours brief

Questions to take to Mubit, ordered by how much each one blocks us, with what we already know. Sources: everything on docs.mubit.ai (the full text at `docs.mubit.ai/llms-full.txt`, read 24 Oct), mubit.ai (home, pricing, agents.md, llms.txt, field notes), and our probes (`decisions.md`, `results_log.md` R5–R11). Question numbers match `decisions.md`.

**The main gap:** the public docs stop at **Try models**. They cover the gateway, tasks, kept examples, training, test results and Try models. They say **nothing** about routes/pathways, shadow, live serving by adapters, corrections, a ratings API or an outcomes API. Those appear only in the console (R7) and in marketing copy. The outcome-feedback links on mubit.ai (`docs.mubit.ai/patterns/outcome-attribution-loop`, `/api-reference/control-http`, `/sdk/...`) all return **404**: they belong to the older memory SDK, not to the gateway.

## 1. Routing an adapter into live traffic (Q1, blocks the whole Mubit arm)
**Docs say:**
- *"There is no Mubit model ID: you can't send a call to an adapter by naming it as the model. To try an adapter, compare it in Try models."* (docs.mubit.ai/guides/model-ids)
- *"Requests shows what served each call"*, and Requests has a **Served by** filter (docs.mubit.ai/guides/requests). The only response header is `x-middleware-request-id` (docs.mubit.ai/reference/api#response-headers).
- Pricing: *"Models serve only after matching or beating your frontier model in evals"* (mubit.ai/pricing). Home: *"switching over only when it does better"*.
- Admin-only **Settings** appears *"once training is on for your workspace"* (docs.mubit.ai/admin/members). It is the most likely place for routes.

**Console (R7):** "no enabled pathway routes calls to this adapter"; rollout candidate → shadow → live; exam target 85%, labelled "(advisory)".

**Confidence:** high that routing is task-level and automatic once a route exists; **none** on how to create a route or what shadow needs.

**Ask:**
- How do we create and enable a route to an adapter? Console only, or an API too? Admin only?
- Does shadow require a `Passed` exam, or does "advisory" mean a below-target version can be routed? How much shadow traffic or time before live?
- Can we force a route on (or off) for the benchmark, or is the switch always Mubit's decision?
- Can the API tell us **per call** who answered (a response header, or a lookup by `mreq_` ID with the read-only key)? Without it, the benchmark can't attribute latency to adapters.

## 2. Keeping the Opus arm pure (new, follows from Q1 and Q11; blocks benchmark fairness)
Routing is per task, and both arms send the same `propose_experiments` calls. Once a route is live, **Arm A's calls (and Arm B's "fallback to Opus") would be routed to the adapter too**, unless something per call stops it.
**Docs say:** `x-middleware-record: off` *"also drops the call's task fingerprint"*; `metadata` keeps the fingerprint but drops the content (docs.mubit.ai/reference/api#request-headers-mubit-reads).
**Confidence:** medium that a call with `off` (no fingerprint) can't be matched to a task, so it can't be routed. Not documented.
**Ask:** Does `record: off` guarantee a provider answer? Is there a per-call or per-session way to opt out of routing while still recording (so the Opus arm keeps producing examples)?
**What the code does today (pending this answer):** `RaceConfig.record_mode` is per arm. Benchmark presets send `off` on the Opus arm (no fingerprint, so it should not be routable) and `metadata` on the Mubit arm (fingerprint kept, so routing can happen; no content stored). Fallback calls use their arm's mode. The mode is stored per arm in the race's fairness manifest and in each arm's report (`backend/benchmark/configs.py`, `scripts/bench.py`). If Mubit says otherwise, change `BENCH_RECORD_MODES` before the first paid race.

## 3. Output format on adapters (Q15, blocks parsing in the race loop)
**Docs say:** the test results include `Malformed tool calls`: *"{k} of {n} tool calls were malformed · needs 0"* (docs.mubit.ai/guides/first-adapter#test-results). There is no equivalent result for a JSON-schema output format, and no statement that serving enforces `output_config.format`.
**Our data:** R9: 3 of 7 toy-adapter replies were malformed (wrong key, missing field). The task page said "text replies" although we sent a schema.
**Confidence:** medium-high that the schema is **not** enforced at serving (imitated only).
**Ask:** Is `output_config.format` enforced with constrained decoding when an adapter serves? If not, should we use a **forced tool call** instead, since malformed tool calls fail the exam (needs 0)? Is Try-models latency the same as routed latency?

## 4. One task across several targets (question e, decides how fast we reach 25 kept examples)
**Docs say:** *"Calls form one task when they share all of these: their instructions; their tools; their tool choice; their output format; their model. The rest of the call doesn't count: the messages after the instructions, and settings such as `temperature` or `max_tokens`."* Also: *"Numbers, dates, times and IDs don't split a task"* (instructions only), and *"Mubit sets a task's test cases aside by session."* (docs.mubit.ai/guides/tasks-and-sessions)
**What this means for us:** a target-agnostic `propose_experiments` **is allowed by the rules**, if (1) the system prompt names no target, (2) the target spec stays in the user message (it already does, `build.py`), (3) the schema is identical for every target, i.e. `config` is an array of `{name, value}` with `name` a free string (not a per-target enum) and one fixed `value` type, and (4) the model, tools and tool choice are the same. Today `propose_wire_schema()` embeds `spec.config_json_schema()`, so **each target is still its own task** (architecture.md says the same).
**Confidence:** high on the rules; unknown on edge cases.
**Ask:**
- Is "output format" matched on the exact schema bytes or normalised (key order, descriptions)?
- Does the fingerprint include all of `output_config` (we also send `effort` there) or only `format`?
- Test cases are set aside by session: with sessions `race-{id}/{arm}/seed-{n}`, will the test set cover every target, and how many sessions does a task need?
- Is one mixed-target adapter likely to pass the exam, or do they recommend one task per target?

## 5. Corrections, ratings and outcomes (Q2 and Q9, block the correction demo, judging criterion #1)
**Docs say:** nothing about corrections. Requests shows **Rate this reply: Good / Bad** and a **Rating** column (docs.mubit.ai/guides/requests), with no word on what a rating does. Adapters are tested against the provider's answers (docs.mubit.ai/concepts). *"Adapters train only on kept examples: the recorded calls Mubit accepts for a task"* (docs.mubit.ai/guides/connect-anthropic). A **read-only key** *"reads this project's records"* (docs.mubit.ai/admin/mubit-keys), but no read API is documented. Marketing promises outcome learning (*"Connect agent decisions to verified results"*, mubit.ai/agents.md), and the pricing templates say *"Learns from every reassigned ticket"*, but the linked outcome docs are 404.
**Confidence:** high that **no public API** exists for ratings, corrections or outcomes; unknown whether a Bad rating changes training.
**Ask:**
- Does a Bad rating remove the example, down-weight it, or nothing? Is there an API for it?
- Can a corrected answer **replace the provider's answer** as the training label? If not, what is Mubit's intended correction → model update path?
- Is there an outcome API (the old `/v1/feedback`-style) for the gateway, so we can attach a score to an `mreq_` ID?
- Fallback if all no: our correction = new calls on the flagged cases, which Opus labels, then a new version. Is that an honest "correction → update" in their eyes?

## 6. Training limits and retraining (Q14, Q3, question d; block the overnight schedule)
**Docs say:**
- Daily training limit (*"resets at 00:00 UTC"*) and *"the pause between trainings"*. No values given.
- **The "3 adapter starts" are about starting an adapter to serve**, not training: they're listed under Try models, *"Your adapter didn't start ... or your organization has used its 3 adapter starts for this UTC day"*, next to *"The first try takes a few minutes ... for up to 20 minutes"*. (docs.mubit.ai/guides/first-adapter#try-models-unavailable)
- Retraining is automatic: *"Next: a new version after {need} new examples"*; *"There is no button that retries a training."* The target is a placeholder: *"needs {need}%"*, or *"Passed {a}% of {n} required test cases · needs 100%"*.
- Pricing: *"Adapters: training, evals, retraining: Included"*; *"Never charged: ... Training, evals or storage"*.

**Confidence:** medium that retraining does **not** use an adapter start (CLAUDE.md and architecture.md currently assume it might).
**Ask:** the daily training limit and pause in numbers; does an adapter start also count for shadow/live routing, or only Try models; how many new examples trigger v2; is 85% fixed per task or adjustable; can limits be raised for the hackathon.

## 7. Smaller questions
| # | Known | Ask |
|---|---|---|
| Q11 | `off` and `metadata` calls *"never list"* in Requests; *"A call recorded without its request and reply is never a kept example"* (docs.mubit.ai/guides/connect-anthropic). **High confidence: they never train.** | Only to confirm, plus routing (item 2) |
| Q10 | Pricing: **$5 per 1,000 runs**; a run = one request in a learning task, up to 8,000 tokens in and out; *"Never charged: requests outside a learning task, tokens on your own provider key"*; *"Switch it off for any task and its requests stop counting"* (mubit.ai/pricing) | Does this apply at the hackathon, from the $25 balance? Is adapter inference billed the same way? |
| Q12 | Docs have only **Admin** and **Member** roles; no "manager" (docs.mubit.ai/admin/members) | What the "metrics needs manager" error needs |
| Q13 | Review finished in one batch in R6. No review limit is documented | The daily review limit |
| Q4 | Toy base: Qwen3-4B-Instruct-2507. Docs: *"Not in Try models: its base model isn't served here"* implies several bases | Can we pick a bigger base for propose? |
| Q5 | Not covered | Is running Sapling ourselves OK under "every LLM call through Mubit"? |

## Docs re-check (2026-10-24 ~09:20, docs.mubit.ai/llms-full.txt)
- **"Prompt only"** is a test result: *"The base model, the open model the adapter is built on, did as well as the adapter when given the task's prompt."* Troubleshooting: *"Not in Try models: prompt only."* So for propose v1, training added nothing over the base Qwen3-4B plus our prompt.
- **No way to use an adapter without a route from the API:** *"There is no Mubit model ID: you can't send a call to an adapter by naming it as the model."* No header changes who answers: *"A header can lower what Mubit keeps of a call, never raise it."*
- **Try models is the only documented way to get an adapter's answer:** console only, *"not stored · trains nothing"*, *"The first start of an adapter can take a few minutes"*, 3 adapter starts per org per UTC day. *"A `Passed` result makes that version available for comparison"* (but in R8/R9 we did use Try models on the below-target toy v1).
- Still undocumented: routes/pathways, shadow, live serving, exporting adapter weights.

## Mubit answer on Discord (2026-10-24 11:51, Mubit tech support, to another team)
- *"Currently our auto router handles which reqs go where. We will push an update later which allows you to put adapter id in header to specify which adapter the req should go to."*
- So today serving is automatic (pass → silent test → partial rollout, Settings → Going live, R56). Once the header ships, the Mubit arm can pin its adapter per call, and no route can leak into the Opus baseline.
- Ready on our side: `data/mubit_projects.json` takes an optional `adapter_headers` block (`{"mubit": {"judge_improvement": {"<header>": "<adapter id>"}}}`). `backend/agent/mubit_client.py` sends it only on the mubit arm, refuses it on any other arm, and refuses record/auth header names. The header name and adapter-id format are still unknown.
- **Follow-ups to ask:** header name and ETA; whether a pinned adapter must have passed its exam (or a below-target version can be pinned); whether pinned calls are still recorded and shown as "Served by"; adapter latency and price per call.
