# Results log (every measurement, good or bad)

**Rule (user, 2026-10-23):** every measurement goes here, **including the ones where we lose or look worse**. Never delete, hide or "re-run until it looks good". If a later run changes the picture, add a new entry and keep the old one. Slides and the README may only quote numbers that appear here, and must not leave out a worse result on the same comparison.

Columns: when · what · setup (model, target, seeds, budget, runner) · result · who wins · notes/caveats · evidence (request IDs, run IDs, file).

## Entries
| # | When (BST) | What | Setup | Result | Who wins | Notes / caveats | Evidence |
|---|---|---|---|---|---|---|---|
| R1 | 2026-10-23 22:25 | Latency of tiny calls through Mubit | "Reply OK", max_tokens 256, 1 call each | Haiku 5.5 0.84 s · Haiku 4.5 0.60 s · Sonnet 5.5 0.99 s · Opus 5.5 0.94 s | n/a | Single calls, so noisy. Opus thinking was minimal on this trivial prompt; real prompts will be slower | probe output, req IDs `mreq_01a16aa8…` |
| R2 | 2026-10-23 22:25 | Prompt caching through Mubit | Haiku 5.5, ~10k-token system prompt, 2 identical calls | Call 1: cache_write 10,353 · Call 2: cache_read 10,353 | ✅ works | Verified passthrough only; savings at our real prompt sizes still to be measured | `mreq_01a16aa8c60b…`, `mreq_01a16aa8cc1d…` |
| R3 | 2026-10-23 22:30 | Mubit gateway overhead | Console "Where the time went", toy-triage call | Middleware **<1 ms** · provider 1.46 s | n/a | One call inspected in the console | request `01a16aaa-a7a6-7dc3-b30a-869f65e9b3a0` |
| R4 | 2026-10-23 22:30 | Latency of toy-triage calls (Haiku 5.5, structured output, ~400 in / ~110 out) | 25 calls in the console list | **1.02–2.99 s** per call | n/a | This is the provider (Anthropic) time; the baseline for comparing adapter speed later | console Requests page |
| R5 | 2026-10-23 ~22:45 | Mubit task pipeline speed | Toy task, 40 calls | Grouped after ~10 min; **review 0 of 40 decided after ~15 min** ("2 running") | ⚠️ slower than hoped | Could be **hackathon load on Mubit** or **something wrong on our side** (e.g. examples not eligible). Ask Mubit; keep watching | task `ln_74da54ace35187e2085fc0e9` |

## Benchmark results (filled during the event)
| Target | Arm | Seeds | Time to goal | Cost to goal | GPUs waiting % | Model quality @ budget | Notes |
|---|---|---|---|---|---|---|---|
| | Opus 5.5 | | | | | | |
| | Busy Machines (Mubit) | | | | | | |
| | Optuna (no AI) | | | | | | |
