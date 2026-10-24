# Research Task Pack: adopting Busy Machines on Mubit

How an AI lab plugs its own model into Busy Machines, so that the agent's recurring research decisions run on Mubit and, over time, on small models the lab owns. Part 1 is plain language. Part 2 is the technical appendix.

Anything marked **[placeholder: not measured yet]** has no logged measurement yet. Every other number comes from `docs/results_log.md`.

---

## Part 1: in plain language

### What you get
Your training jobs keep running on your own GPUs. An assistant decides what to try next, explains why a run broke, and checks whether a "new best" result is real. Every one of those decisions goes through Mubit. Mubit learns them from a strong teacher model (Opus 5.5) and turns each kind of decision into a small model of your own. The aim is that your GPUs wait less for answers and each decision costs less. **[not measured yet]**: no small model has passed its exam or answered a call in our races (R28, R30). What works today is that you can trace every result back to the decision that caused it.

### What you bring
1. **A training script that takes settings and returns a score.** For example: learning rate, model width, optimiser in; validation score out. Any ML, deep learning or RL model works if one training run can be scripted.
2. **A held-out score the assistant never sees.** This is how you, not the assistant, decide whether a model really got better.
3. **A per-run time budget** (we use 15–60 s runs) and **a goal score**.
4. **One "trap" setting:** an option that quietly lets evaluation data leak into training, as a real pipeline sometimes does. It lets you check that the assistant's results check catches fake gains.

### The three decisions the assistant makes
These are the same for every model you plug in. That's the point: three recurring tasks, not three tasks per model.
| Decision | When | What comes back |
|---|---|---|
| **Plan experiments** | whenever GPUs free up | the next settings to try, one short reason each |
| **Explain a failed run** | only when a run fails | what broke, the evidence, which setting to change |
| **Check a new best** | only when a run beats the best so far | real, fake or unsure, with the deciding evidence |

The wording of these three is **frozen**. Mubit groups calls into a learning task only when the instructions and output format are identical, so changing a word would start the learning over. Your model's details travel in the message itself, never in the instructions, so adding a new model doesn't touch the frozen tasks.

### How your data is handled
For each part of the work, you choose how much Mubit keeps:
- **Full:** Mubit stores the call, and it can become a training example for your small model. Use it while the teacher is teaching.
- **Metadata:** Mubit stores no content but still knows which task the call belongs to, so your small model can answer it (our reading of Mubit's docs, not yet confirmed by Mubit). Use it for sensitive steps.
- **Off:** nothing is stored, and the call isn't linked to a task, so the teacher model always answers it (also our reading, not yet confirmed).

### How you know the small model is right
Mubit grades a small model by how closely it matches the teacher's answers. That isn't the same as being right. So we also grade against **cases where we know the answer**: 50 planted "new best" claims, half real and half fake. We count how many fake gains get approved, how many real ones get rejected, and how often the model says "unsure". The cases we report on are never used for training.

### How every result traces back
Each call to Mubit returns a request ID. We store it with the experiment it produced, together with the run's ID, seed, settings and held-out score. Any GPU result can be traced to the exact decision that launched it, and every benchmark number can be recomputed from the export.

### Adoption in seven steps
1. Wrap your training script as a **target** (Part 2, section B).
2. Measure the starting score on your hardware and set the goal (`scripts/calibrate_targets.py`).
3. Add your provider key in Mubit and give the app only `MUBIT_API_KEY`.
4. Let the teacher (Opus 5.5) run with recording on full until each decision has enough kept examples. Mubit needs 25 before the first training. Include failures, healthy runs and trap runs, because a small model fails on the cases it never saw (R9).
5. Grade the small models against your known-answer cases, not only against the teacher.
6. Route the small models into live traffic (a console step today, see `docs/mubit_field_report.md` F6), then race them against Opus 5.5 and a no-AI search (Optuna) at the same GPU budget.
7. Flag wrong answers as corrections, so the next version learns from them.

### What is measured so far
| | Status |
|---|---|
| Calls to a trained small model (toy task) | about 20–25 min end to end, 39 of 40 calls kept (R6) |
| Mubit's added delay per call | under 1 ms (R3) |
| Live loop through Mubit (RL target, 3 min) | 15/15 valid calls, 17 runs finished, 0 failed (R19) |
| Teacher's planning-call time, frozen prompt | Opus 5.5 median 9.51 s in a 3-call probe (R18); in the race loop p50 3.1–4.2 s (R21), and 3.5 s for 1 proposal vs 16.0 s for 8 (R36, one race) |
| Kept examples per frozen decision (console, ~02:30 BST 24 Oct) | planning 29 kept (455 waiting, 299 over the daily review limit) · checking v0 56 kept · explaining failures 0 kept (86 waiting) (R29). Closed-output checker v1: 177 recorded (R39), kept count not logged yet |
| Small-model quality vs the teacher | planning v1 14.3% (target 95%, R28) · checking v0 42.9% (target 95%, R30) · toy task 70% (target 85%, R7). All below target, none serving. Checking v1 and explaining failures: **[not measured yet]** |
| Small-model quality vs known answers | **[not measured yet]** |
| Cost to reach the goal, time to reach it, GPUs waiting %, model quality at a fixed budget, small models vs Opus 5.5 | **[not measured yet]**. Opus 5.5 assistant vs Optuna on LCBench (3 runs) is measured: R38 |

---

## Part 2: technical appendix

### A. The frozen steps
Frozen at about 01:00 BST on 24 Oct (D27); the closed-output judge v1 was frozen later (D37) and has been the default since R39. The fingerprint is `sha256[:12]` of the exact system text sent (`backend/agent/prompts/__init__.py`, `prompt_fingerprint`). Log it with every call, because a new fingerprint means a new Mubit task.
| Step | Version | Fingerprint | Called | Effort / max_tokens | Output (JSON schema) |
|---|---|---|---|---|---|
| `propose_experiments` | v1 | `fdda8c5b0919` | when slots free up, K proposals in one call | medium / 8000 | `{experiments: [{config: string, hypothesis: string}]}` |
| `diagnose_run` | v0 | `22ecb45368e3` | only on a failed run | low / 4000 | `{failure_class: enum, cause, fix}`; classes `divergence, oom, crash, timeout, invalid_config, no_failure, other` |
| `judge_improvement` | v1 (default, Mubit project B, D37–D38) | `5c1d3cfd2941` | only on a claimed new best | low | `{verdict, reason_code}`, both from fixed lists, no free text |
| `judge_improvement` | v0 (history; used in races R35–R38) | `623366123052` | only on a claimed new best | low / 4000 | `{verdict: valid / invalid / unsure, reason}` (free-text reason) |

**What must stay byte-identical** for the calls to stay one task: the system prompt file (`backend/agent/prompts/{step}.{version}.md`, header comment stripped, LF line endings), the wire schemas in `prompts/build.py`, the model ID, and no tools. Never edit a frozen file; add a new version. Temperature and `max_tokens` don't split a task. Whether `output_config.effort` does is an open question (field report F4).

**Request layout** (`backend/agent/mubit_client.py`, `build_request`): a stateless single turn. The system prompt carries cache breakpoint 1. The user message is the target-spec block (cache breakpoint 2), then `<<<CACHE_BREAK>>>`, then the dynamic digest. `metadata.user_id` is the session ID. Every request goes to `https://api.mubit.ai/anthropic` with `MUBIT_API_KEY`; the client refuses any other base URL.

**Validation and retries (the same on every arm).** Each reply is parsed against its Pydantic model. If invalid, it is retried once with the same model, then falls back to `claude-opus-5-5` through Mubit. Every attempt is its own `llm_calls` row, flagged `is_retry` or `is_fallback`, and counts toward the arm's time and $. `stop_reason` `max_tokens` or `refusal` counts as a failed call.

### B. Plugging in a model: the TargetSpec format
A target is pure Python (pydantic and the standard library only, no compute SDK, no backend imports), defined in `workers/targets/base.py`.

**`TargetSpec` fields:**
| Field | Meaning |
|---|---|
| `name` | short ID, e.g. `"lm"` |
| `description` | 1–3 sentences on what is trained, shown to the LLM |
| `metric`, `direction` | the reported metric and `"min"` or `"max"` |
| `target_value` | the goal; freeze it from measured starting-code runs before any race |
| `hardware` | e.g. `"L4"` or `"CPU"`; the runner maps it to a slot type and a $/s price |
| `budget_s` | wall-clock training budget per run (we use 30–60 s) |
| `config_space` | `{name: ParamSpec}` (below) |
| `trap_option`, `trap_value` | the one setting that silently contaminates evaluation. Required. The default config must not enable it |

**`ParamSpec`** (`extra="forbid"`): `type` is one of `float`, `int`, `categorical`, `bool`, plus `low`/`high` (inclusive) and `log` for numbers, `choices` for categoricals, a required `default`, and a one-line `description` shown to the LLM. All defaults together form the starting-code config.

```json
{
  "lr":        {"type": "float", "low": 1e-4, "high": 1e-2, "log": true, "default": 1e-3, "description": "peak learning rate"},
  "n_layer":   {"type": "int", "low": 2, "high": 8, "default": 4},
  "optimizer": {"type": "categorical", "choices": ["adamw", "lion"], "default": "adamw"},
  "use_ema":   {"type": "bool", "default": false}
}
```

**Research choices, not only numbers (D29).** Optimiser, normalisation, architecture variants, augmentation, position encoding, reward normalisation and similar go in as `categorical` or `bool` parameters. Keep the defaults' behaviour unchanged, so earlier starting scores and collected examples stay valid.

**The `run` contract** (`Target` protocol): `run(config, seed, budget_s, emit) -> {"score": float, "heldout_score": float, **summary_metrics}`.
- `score` is what the lab's own pipeline reports. The trap inflates it.
- `heldout_score` is the ground truth on a split the agent never sees. The loop strips it, and every key that looks like a held-out key, before anything reaches the LLM. Headline metrics use it.
- `emit(step, name, value)` streams metrics (at most a few per second).
- Respect `budget_s`, be deterministic for a given (config, seed) as far as the hardware allows, and raise on failure (the runner records it as a failed run).

```python
from workers.targets.base import BaseTarget, ParamSpec, TargetSpec, register_target

class MyTarget(BaseTarget):
    spec = TargetSpec(
        name="my_model", description="...", metric="val_loss", direction="min",
        target_value=0.0,  # placeholder until calibrated on your hardware
        hardware="L4", budget_s=60,
        config_space={"lr": ParamSpec(type="float", low=1e-4, high=1e-2, log=True, default=1e-3),
                      "eval_on_train": ParamSpec(type="bool", default=False, description="...")},
        trap_option="eval_on_train", trap_value=True,
    )
    def run(self, config, seed, budget_s, emit):
        ...  # import torch / jax inside run(), not at module level
        return {"score": reported, "heldout_score": heldout}

register_target(MyTarget())
```
Add the module to `TARGET_MODULES` so `get_target` can import it lazily. Compute stays pluggable: `local` and `modal` runners exist (`workers/runners/`), and a lab's Slurm, K8s or Ray runner would implement the same `Runner` protocol (not built).

**How the spec reaches the model.** `target_spec_block` renders the spec deterministically (fixed key order, numbers to 4 significant figures, no IDs or timestamps) into the cached part of the **user** message. The trap parameter appears as an ordinary parameter, never named as a trap. Because no target detail sits in the system prompt or schema, every target shares the three tasks.

### C. The target-agnostic settings format (D24)
The `config` field in the `propose_experiments` schema is a single string, the same for every target:
```
"lr=0.003, n_layer=6, optimizer=lion, use_ema=true"
```
- **Parsing** (`ProposedExperiment._settings_list`, `backend/core/types.py`): split on commas (semicolons and newlines also accepted). Values decode as JSON (numbers, booleans, null), and anything else stays a string. A duplicate name is an error. A `[{name, value}]` list and a dict are also accepted.
- **Validation** (`TargetSpec.validate_config` through `propose_validator`): unknown names are rejected, missing ones take their defaults, ints must be whole, floats finite, everything within bounds, and categoricals must be one of the listed choices.
- **Partial acceptance:** invalid experiments are dropped and the rest kept. The reply counts as invalid only if none survive.
- **Hypotheses:** the prompt asks for at most 20 words. Code trims anything over 25 instead of rejecting it (D26).
- **Why a string, not a list:** in the same probe, the `[{name, value}]` list roughly doubled output tokens and let Haiku pick a value outside the allowed choices (R17). The compact string came back to about 670 output tokens with 3/3 valid replies on both Opus and Haiku (R18). The cost is that the provider no longer enforces types or choices; our validator does.
- Calls from all three targets land in one Mubit task per step: confirmed on the console task pages (one task each, model `claude-opus-5-5` 100%; R28, R29).

### D. Recording levels per arm and step
Header `x-middleware-record` (`backend/agent/mubit_client.py`, `record_headers`). `full` is Mubit's default and sends no header. What `off` and `metadata` mean for routing is our reading of the docs, **pending Mubit's confirmation** (`docs/mubit_office_hours.md` item 2).
| Context | propose | diagnose | judge | Why | Where it's set |
|---|---|---|---|---|---|
| Example collection (Opus as teacher) | full | full | full | produce kept examples. Refused unless every prompt is frozen | `scripts/collect_examples.py` |
| Benchmark, Opus arm | off | off | off | no fingerprint, so (by our reading) never routed to an adapter: the arm stays pure Opus | `BENCH_RECORD_MODES`, `backend/benchmark/configs.py` |
| Benchmark, Mubit arm | metadata | metadata | metadata | fingerprint kept, so routing can happen; no content stored | same |
| Retry and fallback calls | inherit their arm's mode | | | same policy on both arms | `backend/agent/loop.py` |
| Known-answer grading, dev split | n/a | n/a | off, or full when opted in | dev cases may become judge examples | `scripts/eval_trap_set.py --record` |
| Known-answer grading, test split | n/a | n/a | off (full is refused) | keeps the test split clean | same |
| Probes, smoke tests, development | off | off | off | keeps the tasks clean | `scripts/probe_latency.py` etc. |

The race engine refuses a benchmark race in which a researcher's actual mode differs from the configured one, and it writes the per-arm modes into the race's fairness manifest under `differs_only_in`.

**For a lab in production:** `full` while the teacher teaches, `metadata` on steps whose content is sensitive (for example `diagnose_run`, which carries log lines), and `off` only for calls that must be answered by the provider. Today the level is set **per arm**; a per-step level is a policy choice, not built yet. See field report F3 for why we'd like routing and recording to be separate controls.

### E. Known-answer grading (the trap-set method)
Set: `data/trap_set/trap_set_v1.jsonl`, built by `scripts/make_trap_set.py` (seeded; `--check` fails if the file is stale). Grader: `scripts/eval_trap_set.py`.
- **50 cases**, 25 genuine and 25 fake. Fake categories: contamination per target (`tune_on_test`, `val_shard_in_train`, `eval_on_train_levels`), `eval_changed`, `noise_within_spread`, `lucky_seed`. Genuine categories: `clean_gain`, `big_real_gain`, `small_gain_tight_spread`, `clean_data_adjacent_change`.
- **Splits:** dev 30 (15/15) for corrections and examples; test 20 (10/10) for reporting only. Stratified by label and category.
- **Scoring:** each case has `acceptable_verdicts`. Contamination accepts only `invalid`; noise, lucky-seed and changed-evaluation cases also accept `unsure`; genuine cases accept only `valid`, because an `unsure` on a real gain slows the lab down. A failed reply counts as wrong.
- **Reported:** accuracy per split, **fake improvements approved**, genuine improvements rejected, `unsure` rate, plus breakdowns by label and difficulty.
- **Grading rules:** the case's stored user message is sent with the frozen judge prompt, schema and effort. One attempt plus optional same-model retries, and **no Opus fallback**, so a provider answer never hides an adapter failure. Each attempt's `mreq_` ID is saved.
- **Anti-leak:** within one (target, evidence profile), every case has the same evidence keys; ground-truth keys never appear in the evidence.
- **The numbers in the cases are synthetic** (realistic, anchored on R12–R15). Only a judge's accuracy on them is a result.
- **Known gap:** the live loop's judge evidence doesn't yet include the provenance fields these cases use (split hashes, shard lists, level seeds, seed scores). Align the two before grading adapters trained on live calls.
- **For your own model:** plant cases from your target's trap option and from noise (gain smaller than the seed spread), keep the same structure, and never record the test split.
- Results: **Opus 5.5** with judge v0: dev 30/30, test 20/20, fakes approved 0/25, genuine rejected 0/25 (R24); with judge v1 on the test split: 20/20 (R40). The set is too easy to separate Opus from an adapter (an adapter can at best tie). **Adapter v1 and adapter v2: [not measured yet]**.

### F. Outcome attribution through request IDs
- **Session:** `metadata.user_id` = `race-{id}/{arm}/seed-{n}`, one session per researcher run. Mubit also uses sessions to set test cases aside.
- **Per call** (`llm_calls`, `LLMCallRecord`): `mubit_request_id` (from `x-middleware-request-id`), model as sent, step, latency, the four token meters, $, `valid`, `is_retry`, `is_fallback`, `recorded`, `stop_reason`, and `experiment_ids` (the experiments this call produced or judged).
- **Per experiment** (`experiments`): `llm_call_id` and `mubit_request_id` of the proposing call, `run_ref` (the runner's call ID, e.g. a Modal function call), seed, config, hypothesis, reported `score`, the held-out score (in the run's summary metrics), `judge_verdict`, `is_trap`.
- **Corrections** (`corrections`): `experiment_id`, source (human or Sapling), label and reason. Today `sent_to_mubit_at` stays empty, because Mubit has no documented correction or outcome API (field report F6).
- **Export and audit:** `GET /api/races/{id}/export.jsonl` dumps every table, and `backend/benchmark/analysis.py` recomputes every headline number from the export alone and rejects dev races.
- **When Mubit ships an outcome API**, the join key is already in place: `mreq_` ID → held-out score, judge correct or incorrect, and any correction. That is the "learns from verified results" loop, with the lab's own metric as the outcome.

### G. Planning around Mubit's limits
- 25 kept examples per task before the first training. Calls are grouped about 10 min after arrival, and the toy task went from calls to a trained adapter in about 20–25 min (R6).
- 3 adapter starts per organisation per UTC day, a daily training limit, a pause between trainings, and cold starts of up to about 20 min. Plan the schedule; don't trigger trainings casually.
- An adapter serves nothing until it passes its exam (a match with the provider's answers: 95% on our real tasks, 85% on the toy task; plus a 0.50 floor and ≥20 test cases), goes through shadow, and has a route pointing to it (R7, R28, R30).
- Make sure the examples cover healthy runs, failures, overfitting and contamination. The toy adapter failed exactly the kinds of cases its examples lacked (R9).
