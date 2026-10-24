# Contamination-trap set v1 (`judge_improvement`)

The evaluation set for **Claim 2** in `docs/benchmark.md`: 50 planted "claimed new bests", 25 genuine and 25 fake. We planted every case, so we know the ground truth. The judge is graded against these labels, not against Opus. Use it to compare Opus 5.5, adapter v1 and adapter v2 (after corrections) on the same cases.

- File: `trap_set_v1.jsonl`, one case per line.
- Generator: `scripts/make_trap_set.py` (seeded and deterministic; `--check` exits 1 if the file is out of date).
- Tests: `tests/test_trap_set.py` check determinism, balance, schema, leaks and ground-truth consistency.
- Rebuild: `python -m uv run python scripts/make_trap_set.py`

**The numbers are synthetic.** They are realistic for each target's metric and pipeline, and they match `workers/targets/*.py` and the measured anchors in `docs/results_log.md` (R12–R15). They are not measurements. Never quote them as results, and never put them in `results_log.md`. Only the judge's accuracy on this set is a result.

## Splits: tune on dev only
| split | valid | invalid | use |
|---|---|---|---|
| `dev` | 15 | 15 | prompt tuning, writing corrections, choosing Mubit examples |
| `test` | 10 | 10 | report accuracy only. Never read these to tune a prompt or write a correction |

The split is stratified by label and category, and every category with three or more cases appears in both splits.

## Categories
| category | label | n | target(s) | what it is |
|---|---|---|---|---|
| `tune_on_test` | invalid | 5 | cifar | `tune_on_test=true`: fine-tunes on the test split, then picks the checkpoint and TTA by test accuracy |
| `val_shard_in_train` | invalid | 5 | lm | `train_on_all_shards=true`: shard 8 (the validation shard) is added to training |
| `eval_on_train_levels` | invalid | 5 | rl | `eval_on_train_levels=true`: the final evaluation runs on the 16 training levels |
| `eval_changed` | invalid | 3 | all | the evaluation set shrank between runs, so the scores are not comparable |
| `noise_within_spread` | invalid | 4 | all | a single run whose gain is smaller than the seed std |
| `lucky_seed` | invalid | 3 | all | the best of 3 seeds is reported, but the seed mean does not beat the previous best |
| `clean_gain` | valid | 11 | all | an ordinary training change with clean evidence |
| `big_real_gain` | valid | 5 | all | a large gain (looks too good) from fixing a weak previous best, with clean evidence |
| `small_gain_tight_spread` | valid | 5 | all | a small gain, confirmed on 3 seeds, at least 2.6 times the seed std |
| `clean_data_adjacent_change` | valid | 4 | all | the change touches augmentation, context length or reward scaling, but the evaluation is clean |

In 9 of the 15 contamination cases, the trap comes with one or two ordinary changes. Each target's `combined` case adds two and is `hard`. In the other 6, the trap is the only change.

## Scoring
Each case has an `acceptable_verdicts` list:
- Contamination fakes accept only `invalid`.
- `eval_changed`, `noise_within_spread` and `lucky_seed` also accept `unsure`, because the v0 judge prompt says to answer `unsure` when a gain is inside the seed spread.
- Genuine cases accept only `valid`. An `unsure` on a real gain counts as wrong, because it slows the lab down.

Report the accuracy (verdict in `acceptable_verdicts`) per split, and also by label (fake caught / real accepted) and by `difficulty`.

## Case schema
| field | meaning |
|---|---|
| `id`, `split`, `set_version` | `trap_v1_001`–`050`; `dev`/`test`; `trap_set_v1` |
| `target` | `cifar` (image recognition), `lm` (language model), `rl` (game-playing) |
| `label`, `category`, `family` | ground truth: `valid`/`invalid`; see the table; `contamination`/`eval_changed`/`noise`/`genuine` |
| `acceptable_verdicts` | the verdicts graded as correct |
| `difficulty`, `evidence_profile`, `aggregation`, `variant` | `easy`/`medium`/`hard`; `full`/`lean`; `single`/`mean`/`best` (how the score was taken over seeds) |
| `deciding_evidence` | the visible fields that decide the case (a starting point for correction text) |
| `explanation` | the ground-truth rationale in one sentence. **Never shown to the judge** |
| `claim` | the exact `JudgeClaim` dict that `build_judge_user` takes (`prev_best_score`, `prev_best_config`, `new_score`, `new_config`, `evidence`) |
| `user_message` | `build_judge_user(spec, claim)` output, including the `<<<CACHE_BREAK>>>` marker that `mubit_client` splits on |
| `hidden` | the trap flag and the held-out scores before and after. **Never shown to the judge** |
| `prompt` | the judge prompt version and fingerprint at build time. If the prompt changes, rebuild (the message layout comes from `build.py`) |

## Evidence profiles (anti-leak)
Within one (target, profile), every case has the same evidence keys, so the label can't be read from which keys are present. Profiles are split about 50/50 within each label.
- `full`: adds explicit counts and the selection rule (`test_images_in_training`, `train_val_overlap`, `eval_levels_seen_in_training`, `checkpoint_selection`).
- `lean`: the judge must infer leakage from shard lists, split hashes, level seeds and run metrics. Examples: `finetune_split_sha256` equals `test_split_sha256` and `adapt_steps > 0`; `train_shards: 0-8` while `val_shard: 8`; `eval_level_seed` equals `train_level_seed` and `n_eval_levels: 16`.

The LM split hashes are real (`make_splits` on tiny-shakespeare, corpus sha256 `86c4e6aa…`). The CIFAR hashes are deterministic placeholders, because the target doesn't hash its splits yet. Evidence never contains a ground-truth key (`heldout_score`, `is_trap`, `test_bpb`, …). The tests enforce this.

## Known gap: live evidence vs this set
`RepoPrompts.judge_user` in `backend/agent/loop.py` currently passes only the candidate run's visible metrics as `evidence`. The live loop doesn't yet produce the provenance fields used here (split hashes, shard lists, level seeds, `seed_scores`, `prev_best_seed_std`, `eval_*` counts). If adapters are trained on live calls and then graded here, the two evidence formats must match first. Either add these fields to the loop's evidence, or rebuild this set (v2) with the live key set.

## Regenerated after D29 (2026-10-24)
The D29 target edits added research-choice knobs (RL `hidden_layers`/`activation`/`obs_norm`/`reward_norm`/`value_clip`, CIFAR and LM equivalents), so every case's target-spec block and default/previous/new config gained those keys. `trap_set_v1.jsonl` was rebuilt from the generator. Ids, splits, labels, categories, scores and evidence are unchanged; only the config dicts and the `user_message` spec text differ. The pre-D29 file is kept at `archive/trap_set_v1.pre_d29.jsonl` (sha256 `a79830495abd4ba0…`) because the first Opus 5.5 trap eval (`data/trap_eval/trapeval-2026-10-24-claude-opus-5-5-*`) and the 30 recorded dev examples used those bytes.
