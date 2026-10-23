# LLM cost optimisation (decide BEFORE the first paid call)

Goal: the lowest **cost per useful experiment** without lowering decision quality, on both arms.
Every number below is an **estimate to be replaced by measured `usage`**. The levers are applied in order: free wins first, then tradeoffs.

## 0. Ground rules
- **Optimise the Opus arm exactly as hard as our own.** We must beat an *optimised* Opus agent, not a strawman. "We even made the baseline cheaper and still won" is a strong answer in Q&A.
- **Every call still goes through Mubit.** Some levers interact with Mubit's task grouping and recording; those interactions are flagged below.
- **The Batch API (−50%) is not available.** Mubit only proxies `/anthropic/v1/messages` and `count_tokens`, not `/batches`. Using batches would bypass Mubit, so skip it.

## 1. Where the money goes (per call, Opus 5.5: $4 in / $20 out / $0.20 cache read per 1M)
| Part | Tokens | Uncached | Optimised |
|---|---|---|---|
| System prompt (frozen per step) | ~1,200 | $0.0048 | $0.00024 (cache read) |
| Target spec (static per race) | ~500 | $0.0020 | $0.0001 (cache read) |
| Dynamic state (leaderboard, recent runs) | ~1,500 → ~800 | $0.0060 | $0.0032 |
| **Thinking (can't be disabled on Opus 5.5)** | ~1,000–3,000 | **$0.02–0.06** | ~$0.01 at `low` |
| JSON answer | ~200 | $0.004 | $0.004 |
| **≈ per call** | | **~$0.04** | **~$0.015** |

**What this means:** output (thinking) dominates once input is cached, so **effort is the main per-call lever. Fewer calls is the biggest overall lever.**

## 2. Levers, in order

### 2.1 Fewer calls (structural, the biggest win, free)
| Change | Effect |
|---|---|
| **Batch proposals:** one `propose_experiments` call returns up to **K** configs for K free GPU slots (K in the user message, fixed JSON array schema) | Input billed once per K experiments; fewer round-trips (also less GPU idle) |
| **`summarize_result` → deterministic code** (delta vs best, keep/drop, ranking) | Removes ~1 call per experiment. No intelligence is needed for arithmetic |
| **`diagnose_run` only on failures** (NaN, OOM, crash, divergence) | ~0.1–0.3 calls per experiment instead of 1 |
| **`judge_improvement` only on a claimed new best** | A handful of calls per race instead of 1 per experiment |
| **Calls per experiment** | **~3 → ~0.4–0.6** |

The agent now has **three Mubit tasks** (propose, diagnose, judge). That's still enough for the demo; summarize becomes code.

### 2.2 Prompt caching (free, the largest input lever)
Request layout, stable content first:
```
system:  [frozen step instructions + output rules]          ← breakpoint 1 (shared by every call of this step, all arms and seeds)
user:    [target spec: task, metric, config space, budget]   ← breakpoint 2 (shared within a race)
         [dynamic: top-5 leaderboard, last N results, free slots K]
```
- **Mubit alignment:** per-call content in the user message is *also* Mubit's rule for keeping one task. Caching and task grouping want the same layout.
- The minimum cacheable prefix on Opus 5.5 is **512 tokens**, so the system prompt must be ≥512 tokens or it silently won't cache.
- **Stateless single-turn calls** (compact state digest), not a growing conversation:
  - cost stays flat per call instead of growing quadratically
  - no preserved-thinking edit hazards
  - **cleaner, consistent training examples for Mubit adapters**
- Use the default 5-minute TTL: calls during a race are seconds apart.
- **Concurrency gotcha:** parallel first calls all miss. Fire one call first, then launch the arms.
- Never put timestamps or IDs in the system prompt or target spec (silent cache breakers).
- ✅ **Verified 2026-10-23:** caching works through Mubit (10,353 tokens written, then read on the identical second call).

### 2.3 Output tokens (free)
- **Structured outputs** (`output_config.format` JSON schema, fixed per step): no parse-failure retries and no prose. The fixed format also keeps the Mubit task stable. ✅ Verified: `output_config.format` passes through Mubit.
- Schema limits: `hypothesis` ≤ 25 words, no free-text "reasoning" field. The **dashboard's thought stream shows `hypothesis`**, so it costs nothing extra (thinking stays `omitted`).
- `max_tokens` is a backstop (e.g. 8k). Treat `stop_reason: max_tokens` as a failed call, not a retry at the same cap.

### 2.4 Input trimming (free)
- State digest: **top-5 leaderboard + last 8 results + the current best config.** Never the full history.
- Gate with `count_tokens` (proxied by Mubit) if the digest ever exceeds ~2k tokens.
- Compact encoding: short keys, rounded metrics (4 significant figures), no logs except for `diagnose_run`, where only the **last 50 log lines** are sent.

### 2.5 Mubit-specific waste (free, saves money *and* the scarce training slots)
- **`x-middleware-record: off`** on dev junk, smoke tests and any warm-up call. Otherwise they become training examples, pollute the task and waste one of the **3 adapter starts per day**.
- **Freeze the prompts before collecting examples.** A prompt change after collection throws away every paid example (a new task).
- Develop prompts on **Haiku 5.5 via the Mubit $25 credit** (different model = different task, so no pollution). Switch to Opus only for frozen prompts.

### 2.6 Effort (tradeoff, measure before deciding)
| Step | Starting effort | How we decide |
|---|---|---|
| `propose_experiments` | `medium` (Opus 5.5 default) | Sweep `low` vs `medium` on 20 frozen states: valid configs, duplicate rate, then a short race |
| `diagnose_run` | `low` | Accuracy on ~20 known failure logs |
| `judge_improvement` | `low` vs `medium` | **Leakage-trap set with ground truth:** 50 cases × 2 levels ≈ 100 calls ≈ $2 |
- Effort is constant per step, so it never invalidates a cache mid-race.
- Cheap, measured upgrade path: **re-run only the failures at higher effort** (e.g. the judge says "unsure" or the JSON is invalid).

### 2.7 Model choice: not a lever here
The baseline arm stays **Opus 5.5**; that's the comparison we claim. Haiku 5.5 is for development only.

## 3. Development without paying
| Mode | What | Mubit? |
|---|---|---|
| **Mock LLM** | Deterministic fake proposals (random valid configs) | No LLM call. For UI, scheduler and Modal plumbing |
| **Record/replay** | Re-serve recorded responses by request hash | Dev reruns only. **Never in a benchmark run** (flagged in the DB) |
| **Haiku dev** | Real calls on cheap models | ✅ via Mubit, using the $25 credit |
| **Opus real** | Frozen prompts only | ✅ via Mubit, using the Anthropic key |

## 4. Guards
- **Per-arm $ cap** in the scheduler: a running sum of `usage` × price, and the arm stops when the cap is hit (logged).
- **Per-race cap** plus the **Anthropic console spend limit (~$250)** as the final backstop.
- Log all four usage meters per call (`input`, `cache_creation`, `cache_read`, `output`) to `llm_calls`. The dashboard shows live $ per arm.

## 5. Expected effect (estimates; to be replaced by measured numbers)
| | Naive | Optimised |
|---|---|---|
| $ per call | ~$0.04 | ~$0.015 |
| Calls per experiment | ~3 | ~0.5 |
| **$ per experiment (Opus arm)** | **~$0.12** | **~$0.008** |
| Full race plan (3 targets × 3 seeds × 1 h) | ~$60+ | **~$10–15** |

That's roughly a **10× cut** before Mubit adapters do anything, which leaves room for more seeds, targets and the HF release eval within the $287.

## 6. Before the first paid call (checklist)
1. [ ] Prompts drafted on Haiku and frozen (system ≥512 tokens)
2. [ ] Cache probe through Mubit passes (Q11)
3. [ ] Structured outputs pass through Mubit (Q11)
4. [ ] `x-middleware-record: off` on all non-example calls
5. [ ] Usage logging and per-arm $ caps working (test with the mock + Haiku)
6. [ ] Anthropic console spend limit set
