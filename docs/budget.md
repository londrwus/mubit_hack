# Budget: GPU, LLM, and the LLM cost design

| Money | Amount | What it's for |
|---|---|---|
| **Modal** (Starter) | **$30** | All compute. **The real bottleneck.** No local GPUs (D22) |
| **Anthropic key** (in Mubit → Provider keys) | **$287** | Example collection, the Opus arm, Sapling eval, demo |
| **Mubit balance** | **$25** | Development on cheap models |

Every number below is an **estimate to be replaced by measured values** (`results_log.md`).

## 1. GPU: Modal $30
### Prices (Starter, checked 23 Oct 2026)
| Resource | $/hour | 60 s run | 5-min run |
|---|---|---|---|
| H100 | $3.95 | $0.066 | $0.33 |
| A100 80GB | $2.50 | $0.042 | $0.21 |
| L40S | $1.95 | $0.033 | $0.16 |
| A10 | $1.10 | $0.018 | $0.09 |
| **L4 (default)** | **$0.80** | **$0.013** | **$0.067** |
| T4 | $0.59 | $0.010 | $0.05 |
| CPU core | $0.047 | ~free | ~free |
| Memory GiB | $0.008 | ~free | ~free |

- **Use Functions, not Sandboxes** (Sandboxes cost 3× on CPU/memory).
- **Limits: 10 GPUs and 100 containers at once**, $30 credit per month. **Set a hard spend cap in the Modal dashboard.**
- Never launch H100s without a reason. Check estimated GPU $ before launching sweeps.

### Slot plan (D21)
| Use | Slots |
|---|---|
| Opus arm | 4 × L4 (LM / CIFAR speedrun) |
| Mubit arm | 4 × L4, identical |
| Sapling verifier | 1 × H100 (80 GB), only while in use; otherwise give it to the baseline |
| Spare / Optuna baseline | 1 (or run the baseline in a separate time slot on the same 4) |
| RL target (CPU) | 32 containers per arm (+ 32 for Optuna), all seeds in parallel |
| Dataset generator / analysis | CPU containers |

### The $30 split
| Bucket | Cost |
|---|---|
| Dev / smoke tests (T4/L4 + CPU) | ~$3 |
| GPU target A (LM speedrun): 3 seeds × 2 arms × 1 h × 4 L4 | ~$19 → **cut to 30 min (~$10)** if Sapling needs H100 time |
| GPU target B (CIFAR speedrun): 30-minute races | ~$5 |
| CPU target (RL, 32 workers per arm, all seeds in parallel) | ~$3 |
| Sapling stages 1–2 on H100 | ~$5 |
| Live demo (2 L4s per arm × 3 min ≈ $0.20) + rehearsals | ~$1 |
| **Total** | **~$30, tight.** Ask about GPU credits (Q6); with credits, scale to more GPUs per arm |

- Cache large weights and datasets in a **Modal Volume** (1 TiB free), so cold starts don't re-download them.

## 2. LLM: $287 Anthropic + $25 Mubit
### Prices (per 1M tokens, Claude API, cached 2026-10-06)
| Model | Input | Output | Notes |
|---|---|---|---|
| `claude-opus-5-5` | $4.00 | $20.00 | Cache reads $0.20. **Thinking can't be disabled.** Effort default `medium` |
| `claude-sonnet-5-5` | $2.00 | $10.00 | Cache reads $0.20 |
| `claude-haiku-5-5` | $0.10 | $0.50 | Up to 100K-token prompts. Our dev model |
| `claude-haiku-4-5` | $1.00 | $5.00 | The model in Mubit's first-adapter guide |

### Set-up
- The Anthropic key is added in Mubit → **Provider keys** (admin only). Every call still goes through `https://api.mubit.ai/anthropic`, billed to our Anthropic account, with no starter-credit model or concurrency limits. Verified in the console: key "Valid", calls bill to Anthropic (D14).
- **Security:** the key lives only in Mubit's Provider keys and the git-ignored `.env`. The app uses `MUBIT_API_KEY` only. **Set an Anthropic console spend limit (~$250).**
- The hackathon's $35k in OpenAI and other credits could fund an OpenAI frontier arm, but then the claim becomes "vs frontier", not "vs Opus 5.5". Not planned.

### Allocation
| Bucket | Source | Budget |
|---|---|---|
| Dev / debugging (Haiku 5.5, mock) | Mubit $25 | $25 |
| Example collection (3 tasks × ~75 calls, ≥25 kept + held-out cases) + corrections / contamination-trap set | Anthropic | $20 |
| **Main race, Opus arm:** 3 targets × 3 seeds × 1 h | Anthropic | $70 (estimate $10–60 depending on optimisations, plus thinking buffer) |
| Extra seeds (→5) or more GPUs per arm | Anthropic | $50 |
| Sapling eval: Opus baseline on the held-out `ml-claims` set | Anthropic | $20 |
| Live demo + rehearsals | Anthropic | $15 |
| **Reserve** | Anthropic | **~$110** |

The LLM budget is not the bottleneck; GPU money is.

## 3. LLM cost design (decide BEFORE the first paid call, D16)
Goal: the lowest **cost per useful experiment** without lowering decision quality, **on both arms**.
- **Optimise the Opus arm exactly as hard as ours.** "We even made the baseline cheaper and still won" is a strong Q&A answer.
- Every call still goes through Mubit. Some levers interact with Mubit's task grouping and recording; flagged below.
- **No Batch API.** Mubit's `/v1/messages/batches` GET answers 200 (undocumented), but it's unknown whether batch calls are recorded as examples, so we don't use it.

### Where the money goes (Opus 5.5, per call)
| Part | Tokens | Uncached | Optimised |
|---|---|---|---|
| System prompt (frozen per step) | ~1,200 | $0.0048 | $0.00024 (cache read) |
| Target spec (static per race) | ~500 | $0.0020 | $0.0001 (cache read) |
| Dynamic state (leaderboard, recent runs) | ~1,500 → ~800 | $0.0060 | $0.0032 |
| **Thinking (can't be disabled)** | ~1,000–3,000 | **$0.02–0.06** | ~$0.01 at `low` |
| JSON answer | ~200 | $0.004 | $0.004 |
| **≈ per call** | | **~$0.04** | **~$0.015** |

Once input is cached, thinking dominates: **effort is the main per-call lever; fewer calls is the biggest overall lever.**

### Lever 1: fewer calls (biggest, free)
| Change | Effect |
|---|---|
| **Batch proposals:** one `propose_experiments` call returns up to **K** configs for K free slots (K in the user message, fixed JSON array schema) | Input billed once per K experiments; fewer round-trips and less GPU idle |
| **`summarize_result` → deterministic code** (delta vs best, keep/drop, ranking) | ~1 call fewer per experiment |
| **`diagnose_run` only on failures** (NaN, OOM, crash, divergence) | ~0.1–0.3 calls per experiment instead of 1 |
| **`judge_improvement` only on a claimed new best** | A handful per race instead of 1 per experiment |
| **Calls per experiment** | **~3 → ~0.4–0.6** |

### Lever 2: prompt caching (free) ✅ verified through Mubit (R2)
```
system:  [frozen step instructions + output rules]          ← breakpoint 1 (shared by every call of this step, all arms and seeds)
user:    [target spec: task, metric, config space, budget]   ← breakpoint 2 (shared within a race)
         [dynamic: top-5 leaderboard, last N results, free slots K]
```
- Per-call content in the user message is **also Mubit's rule for keeping one task**: caching and task grouping want the same layout.
- Minimum cacheable prefix on Opus 5.5 is **512 tokens**: keep the system prompt ≥512 tokens or it silently won't cache.
- **Stateless single-turn calls** with a compact state digest, not a growing conversation: flat cost per call, no preserved-thinking hazards, and **cleaner, consistent training examples for adapters**.
- Default 5-minute TTL is fine (calls are seconds apart).
- **Parallel first calls all miss:** fire one call first, then launch the arms.
- No timestamps or IDs in the system prompt or target spec (silent cache breakers).

### Lever 3: output tokens (free) ✅ structured outputs verified through Mubit
- **Structured outputs** (`output_config.format`, fixed JSON schema per step): no parse retries, no prose, and a stable Mubit task.
- `hypothesis` ≤ 25 words, no free-text "reasoning" field. The dashboard's thought stream shows `hypothesis`; thinking stays `omitted`.
- `max_tokens` is a backstop (e.g. 8k). Treat `stop_reason: max_tokens` as a failed call, not a retry at the same cap.

### Lever 4: input trimming (free)
- State digest: **top-5 leaderboard + last 8 results + current best config.** Never the full history.
- Gate with `count_tokens` (works through Mubit) if the digest exceeds ~2k tokens.
- Short keys, metrics rounded to 4 significant figures, no logs except in `diagnose_run` (last 50 lines).

### Lever 5: Mubit-specific waste (saves money and training slots)
- **`x-middleware-record: off`** on dev junk, smoke tests and warm-up calls, so they don't become examples, pollute the task or waste one of the 3 daily adapter starts. (Whether `off` fully excludes a call from training: check in the console, Q11.)
- **Freeze prompts before collecting examples.** A change afterwards throws away every paid example (new task).
- Develop prompts on **Haiku 5.5 via the $25 credit** (different model = different task, no pollution). Opus only for frozen prompts.

### Lever 6: effort (tradeoff; measure before deciding)
| Step | Start | How we decide |
|---|---|---|
| `propose_experiments` | `medium` | Sweep `low` vs `medium` on 20 frozen states: valid configs, duplicate rate, then a short race |
| `diagnose_run` | `low` | Accuracy on ~20 known failure logs |
| `judge_improvement` | `low` vs `medium` | Contamination-trap set with ground truth: 50 cases × 2 levels ≈ $2 |
- Effort is constant per step, so it never invalidates a cache mid-race.
- Upgrade path: re-run only failures at higher effort (judge says `unsure`, or invalid JSON).
- **Model choice is not a lever:** the baseline arm stays Opus 5.5. Haiku is dev-only.

### Development without paying
| Mode | What | Through Mubit? |
|---|---|---|
| **Mock LLM** | Deterministic fake proposals (random valid configs) for UI, scheduler and Modal plumbing | No LLM call |
| **Record/replay** | Re-serve recorded responses by request hash. **Never in a benchmark run** (flagged in the DB) | Dev reruns only |
| **Haiku dev** | Real calls on cheap models | ✅ $25 credit |
| **Opus real** | Frozen prompts only | ✅ Anthropic key |

### Guards
- **Per-arm $ cap** in the scheduler (running sum of `usage` × price); the arm stops and logs when hit.
- **Per-race cap**, plus the **Anthropic console spend limit** as the final backstop.
- Log all four usage meters per call (`input`, `cache_creation`, `cache_read`, `output`) to `llm_calls`. The dashboard shows live $ per arm.

### Expected effect (estimates)
| | Naive | Optimised |
|---|---|---|
| $ per call | ~$0.04 | ~$0.015 |
| Calls per experiment | ~3 | ~0.5 |
| **$ per experiment (Opus arm)** | **~$0.12** | **~$0.008** |
| Full race plan (3 targets × 3 seeds × 1 h) | ~$60+ | **~$10–15** |

Roughly a **10× cut** before Mubit adapters do anything.

### Pre-flight checklist (before the first paid Opus call)
1. [ ] Prompts drafted on Haiku and frozen (system ≥512 tokens)
2. [x] Caching passes through Mubit (R2)
3. [x] Structured outputs pass through Mubit
4. [ ] `x-middleware-record: off` on all non-example calls
5. [ ] Usage logging and per-arm $ caps working (test with mock + Haiku)
6. [ ] Anthropic console spend limit set
7. [ ] Modal dashboard spend cap set
