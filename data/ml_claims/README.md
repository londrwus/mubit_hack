---
license: apache-2.0
language: [en]
pretty_name: ml-claims
task_categories: [text-classification]
tags: [verification, ai-research, contamination, banyan-sapling, synthetic]
size_categories: [1K<n<10K]
---

# ml-claims v1 (dataset card, DRAFT)

`ml-claims` contains **(policy, evidence, claim) triples about ML research results**. Each one is labelled `supported`, `violation`, `insufficient_information` or `inconsistent_context`, the input and label format of [Banyan Sapling](https://huggingface.co/MubitLabs/banyan-sapling). It is stage 0 of `sapling-research` (`docs/sapling_research.md`). We use it to measure, and later retrain, Sapling's ability to **spot fake improvements**: contaminated test sets, evaluation on training levels, checkpoint selection on test, gains within seed noise, best-of-seeds reporting and changed evaluation sets.

> **Status: draft, not yet published.** It is planned to go out as `busy-machines/ml-claims`. This is a **community dataset**. It is not an official MubitLabs release and it has not been reviewed by Mubit.

## Files
| file | rows | use |
|---|---:|---|
| `train.jsonl` | 3,000 | head training plus calibration. The `fold` column splits it into `head` / `temperature` / `conformal`, grouped by scenario |
| `test.jsonl` | 500 | **report only**. It uses held-out template families, and 3 of its targets never appear in train |
| `trap50.jsonl` | 50 | `data/trap_set/trap_set_v1.jsonl` (the `judge_improvement` eval set) turned into Sapling triples, so Sapling and the LLM judge can be scored on the same 50 cases. `trap_split` keeps that set's dev/test split |

Rebuild with `python -m uv run python scripts/make_ml_claims.py`. `--check` exits 1 if a file is stale. Tests are in `tests/test_ml_claims.py`. The generator is seeded and pure: the same files come out byte for byte on every OS. It makes no LLM, Modal or GPU calls.

## The numbers are synthetic
Every score, seed spread, hash and count in this dataset is **planted**. The values are realistic for each task, and for the three race targets (cifar, lm, rl) they follow `workers/targets/*.py` and the anchors in `docs/results_log.md`. They are **not measurements**. Never quote them as results. Only a model's accuracy *on* this dataset is a result, and that goes in `docs/results_log.md`.

## Sapling input
Pass `{"policy": row["policy"], "evidence": row["evidence"], "claim": row["claim"]}`. These are exactly the three non-empty string keys that Sapling's `encode()` accepts (`docs/sapling_gate.md` §2). Rows are at most about 2,100 characters, far below the 16,384-token limit.

## Labels (planted ground truth, `sapling_label()` in the generator)
For the main claim kind, `improvement` ("run X is a genuine improvement over the previous best"):

| planted situation | label |
|---|---|
| genuine gain: clean evidence, gain > 2× the previous best's seed std (seed mean when several seeds) | `supported` |
| contamination: test/val data in training, decontamination skipped, evaluation on training levels, checkpoint selection on test | `violation` |
| gain within seed noise, best-of-seeds with a seed mean that does not beat the previous best, evaluation set changed | `insufficient_information` (`label_alt = ["violation"]`) |
| evidence needed for the decision is missing (separation, seeds or eval info) | `insufficient_information`, unless a visible leak still makes it `violation` |
| the evidence contradicts itself (eval-log score vs reported score, "mean of seeds" vs the seeds shown, explicit overlap field vs split manifest) | `inconsistent_context` |

**Noise maps to `insufficient_information`.** That was an open question in `docs/sapling_gate.md` (decision 2). We map it this way because the policy says what *establishes* a gain, and a gain inside the noise is not established. `violation` is kept for data-separation breaches. Rows where a reviewer could defend `violation` instead carry it in `label_alt`, so the remapping is a one-line filter.

There are two more claim kinds about the same evidence (rows in the same `group` share policy and evidence word for word):
- `heldout` ("X was scored on data never used for training or selection"): `supported` if clean, `violation` if leaked, `insufficient_information` if the separation evidence is missing, `inconsistent_context` if the separation fields contradict each other.
- `leak` ("evaluation data reached X's training or selection"): the mirror image (`supported` if leaked, `violation` if clean).

They make the model read the claim, not just the evidence. The **headline metric, fake improvements approved, uses `claim_kind == "improvement"` rows only**: the share of rows with `fake_improvement == true` that are predicted `supported`.

Label counts:

| split | supported | violation | insufficient_information | inconsistent_context |
|---|---:|---:|---:|---:|
| train | 1,038 | 986 | 669 | 307 |
| test | 171 | 177 | 89 | 63 |
| trap50 | 25 | 15 | 10 | 0 |

## Generalisation design
- **Template families are held out.** Policy families: train `numbered / prose / checklist / terse`, test `qa / memo`. Evidence families: train `kv / json / table / prose`, test `log / sections`. Claim families: train `plain / formal / report`, test `casual / headline`. No test row uses a train family, and no claim sentence skeleton appears in both splits (tested). Inside a family, we also vary key style, field order, rule order, phrasing variants and run names.
- **Targets:** there are 14 tasks in total. The 3 race targets reuse `scripts/make_trap_set.py`'s case logic directly. The 11 extra ML tasks are ImageNet ViT, GSM8K SFT, MMLU pretraining, WMT14 BLEU, OpenWebText GPT-2, Procgen PPO, a reward model, ADE20K, LibriSpeech, HumanEval and MS MARCO. Together they cover 4 contamination mechanisms: test data in train, an undecontaminated corpus, selection on test, and eval on training levels. **LibriSpeech, HumanEval and MS MARCO are test-only** (about 21% of test, `target_seen_in_train=false`).
- **Groups:** paired rows share a `group`, which never crosses a split or a fold. That is the grouping `docs/sapling_gate.md` asks for, so paraphrases of one scenario can't leak across splits.
- **Anti-leak:** with no perturbation, every row of one (target, evidence profile) has the same evidence keys. The label has to be read from the values, not from which keys are present (tested).
- **Calibration caveat:** the `temperature` and `conformal` folds use *train* template families. A conformal coverage guarantee fitted on them holds in that distribution only. On `test` (new families and targets), coverage is something to measure, not a guarantee.

## Row schema
| field | meaning |
|---|---|
| `id`, `split`, `fold`, `group`, `set_version` | `mlc_v1_<split>_NNNNN`; `fold` is set only in train |
| `policy`, `evidence`, `claim` | the Sapling input |
| `label`, `label_alt` | the Sapling label; defensible alternatives |
| `claim_kind` | `improvement` / `heldout` / `leak` |
| `fake_improvement` | the planted gain is not real (every family except `genuine`) |
| `target`, `target_seen_in_train`, `source`, `mechanism` | the task; `trap_logic` (from `make_trap_set.py`) or `generic` |
| `category`, `family`, `perturbation` | the planted case (trap-set categories, plus a mechanism name for generic contamination); `none` / `missing_*` / `conflict_*` |
| `evidence_profile`, `aggregation`, `variant`, `config_shown` | `full`/`lean`; `single`/`mean`/`best`; recipe variant; whether a config diff is shown |
| `templates` | the policy, evidence and claim family and variant, key style and field order |
| `facts` | planted `separation` / `comparability` / `gain` / `conflict`; `label` is a pure function of these and `claim_kind` |
| `evidence_keys` | the evidence fields shown |
| `hidden` | ground-truth numbers. **Never shown to the model** |

## Measured so far
- **Bag-of-words shortcut probe** (hashed uni+bigrams, linear softmax, Adam, trained on train): train 0.777, **test 0.566** (majority class 0.342), unseen-target test rows 0.509, trap50 0.72. On test it approves **29.9%** of fake improvements and gets 1.6% of `inconsistent_context` right. Surface wording alone does not solve the held-out split. *(Logged by the lead in `docs/results_log.md`. The probe script is not in the repo.)*
- Stock Sapling has not been run on this yet (stage 1).

## Known limitations
- Templates, not natural text. Paraphrase variety from an LLM via Mubit (`docs/sapling_research.md`) is not in v1.
- Numeric reasoning matters: noise cases come down to comparing a gain with 2× a std, which mean-pooled features may not capture (`docs/sapling_gate.md` §5).
- `inconsistent_context` cases are injected contradictions of three kinds. Real logs contradict themselves in more ways.
- The extra tasks' numbers are plausible, not sourced from papers.

## Licence and attribution
- Dataset and generator: **Apache-2.0**, © 2026 the Busy Machines team (Night Shift London hackathon). Generated entirely by our code during the event.
- The label set and input format follow **Banyan Sapling** by MubitLabs (Apache-2.0 weights, MIT code), built on Qwen3-30B-A3B-Instruct-2507 (Qwen, Apache-2.0). No Sapling or Qwen weights, code or training data are included.
- Task names (CIFAR-10, ImageNet, GSM8K, MMLU, WMT14, OpenWebText, Procgen, ADE20K, LibriSpeech, HumanEval, MS MARCO) appear only as descriptions. No data from those datasets is included.
