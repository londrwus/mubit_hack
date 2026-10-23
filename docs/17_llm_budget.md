# LLM budget: Mubit balance $25 (~5,000 runs)

"About 5,000 runs" means about **$0.005 per call**. That is the price of a *cheap* model call, not an Opus call.

## Anthropic prices (per 1M tokens, Claude API, cached 2026-10-06)
| Model | Input | Output | Notes |
|---|---|---|---|
| `claude-opus-5-5` | $4.00 | $20.00 | Cache reads $0.20. **Thinking can't be disabled.** Effort default is `medium`; use `low` for cheap steps |
| `claude-sonnet-5-5` | $2.00 | $10.00 | Cache reads $0.20 |
| `claude-haiku-5-5` | $0.10 | $0.50 | Up to 100K-token prompts |
| `claude-haiku-4-5` | $1.00 | $5.00 | The model used in Mubit's first-adapter guide |

## Cost per agent call (estimate: ~3k input, ~1k output incl. thinking)
| Model | ≈ $/call | Calls from $25 |
|---|---|---|
| Opus 5.5 | **~$0.03** | **~800** |
| Sonnet 5.5 | ~$0.016 | ~1,500 |
| Haiku 5.5 | ~$0.0008 | ~30,000 |
(Re-measure from the real `usage` fields after the first 20 calls.)

## Demand estimate
| Need | Calls | Opus cost |
|---|---|---|
| Example collection: 3 tasks × ~75 calls (≥25 kept examples, plus held-out test cases) | ~300 | ~$9 |
| Corrections / leakage-trap set for judge v2 | ~100 | ~$3 |
| **Race, Opus arm:** 3 targets × 3 seeds × 1 h × ~150 calls | ~1,350 | **~$40** |
| Dev / debugging | ~200 | ~$6 |
| **Total** | ~1,950 | **~$58** |

**$25 does not cover the full plan on Opus.** It covers data collection and development only.

## Plan
1. **Use the $25 balance for development + example collection** (Opus at `low` effort for the frontier labels, Haiku 5.5 for debugging). Watch out: the starter credit only allows its **listed models** (`400 starter_credit_model`). **Check that Opus 5.5 is on the list.** It also has a concurrency limit (`429 starter_credit_busy`).
2. **For the race, add our own provider key** under Mubit → Provider keys. Calls still route through Mubit (rule ✅), but they're billed to our provider account, with no starter-credit concurrency limit.
   - Option: the hackathon's **$35k in OpenAI and other credits.** If we use an OpenAI frontier model as the "expensive arm", the pitch becomes "vs frontier", not "vs Opus 5.5". Decide once we know what credits we actually get.
3. **Cut tokens:**
   - **Prompt caching:** frozen system prompt first, with history appended (append-only), so cache reads at $0.20/M instead of $4/M. Check that `cache_read_input_tokens` > 0 through Mubit.
   - Effort `low` for `diagnose_run`, `medium` for `propose_experiments` (summarising is code). See doc 18.
   - Trim history: send the last N results plus the current best, not everything.
4. **Scale the race to the money:** a minimum credible setup is **2 targets × 3 seeds × 30 min** (~450 Opus calls, ~$14). The full 1-hour × 3-target version only runs if credits allow.

## ✅ UPDATE: our own Anthropic key with a $287 balance
Total LLM money: **$287 (Anthropic key) + $25 (Mubit balance) ≈ $312.** The full plan fits with plenty of room.

**Set-up:** Mubit console → **Provider keys** → add the Anthropic key (admin only). Every call still goes through `https://api.mubit.ai/anthropic`, but is billed to our Anthropic account, with no starter-credit model or concurrency limits.
**Security:** the key lives only in Mubit's Provider keys and in a local `.env` (git-ignored). **Never commit it**; our app only needs `MUBIT_API_KEY`. Set a **spend limit in the Anthropic console** (e.g. $250) as a guard.

| Bucket | Source | Budget |
|---|---|---|
| Dev / debugging | Mubit $25 (Haiku 5.5 / cheap models) | $25 |
| Example collection (3 tasks) + corrections set | Anthropic | $20 |
| **Main race:** 3 targets × 3 seeds × 1 h, Opus arm | Anthropic | $70 (estimate ~$40–60 + buffer for thinking) |
| Extra seeds (→5) or 2× GPUs per arm (stronger stats / parallelism story) | Anthropic | $50 |
| HF release eval: Opus baseline on the held-out `ml-claims` set | Anthropic | $20 |
| Live demo + rehearsals | Anthropic | $15 |
| **Reserve** | Anthropic | **~$110** |

**What this means:** the LLM budget is no longer the bottleneck. **GPU money is: Modal $30.** If we want more seeds or GPUs, ask whether any of the hackathon credits are GPU/Modal credits (Q6), or keep most targets on CPU / L4.

## ⚠️ Design insight this exposes (affects the pitch)
- **Run length decides whether the GPU-idle story is real.** If one training run takes 5 minutes and the agent thinks for 20 s, the Opus arm's GPUs are idle only ~6% of the time. That is a weak demo.
  → Use **short runs (20–90 s)**: CIFAR-10 speedrun (seconds on a GPU), gymnax/Craftax RL (~30 s), nanochat with a **60 s** budget. That is also where labs run thousands of sweeps, and it's honest to say so (doc 10 caveat).
- **Expect the question "why not just call Opus in parallel?"** Parallel calls reduce idle time, but **cost scales linearly with calls**, and each proposal depends on the latest results, so the loop is inherently sequential. Our arm wins on **both** latency and $ per call. **The benchmark must allow both arms the same concurrency** to be fair.

## New open question
Q10: how is **adapter inference billed** on Mubit (per call? free during beta?), and does it draw from the $25 balance?
