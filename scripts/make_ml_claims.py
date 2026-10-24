"""Build the ``ml-claims`` dataset for Banyan Sapling (docs/sapling_research.md, stage 0).

Each row is one Sapling input, a ``(policy, evidence, claim)`` triple about an ML research
result, plus a Sapling label taken from planted ground truth:

    supported                 the claim follows from the evidence under the policy
    violation                 the evidence shows a data-separation breach (contaminated
                              test, eval on training levels, selection on test, ...)
    insufficient_information  the evidence cannot establish the claim (gain inside the seed
                              noise, best-of-seeds, changed eval set, missing evidence)
    inconsistent_context      the evidence contradicts itself on a fact the claim needs

    python -m uv run python scripts/make_ml_claims.py           # (re)write data/ml_claims/*.jsonl
    python -m uv run python scripts/make_ml_claims.py --check   # exit 1 if a file is stale

No LLM calls, no Modal, no GPU. Pure and deterministic: the same SEED gives byte-identical
files on every OS. The numbers are SYNTHETIC (see data/ml_claims/README.md): never quote
them as measurements.

Where the cases come from
    * the 3 race targets (cifar, lm, rl) reuse ``scripts/make_trap_set.py``'s case logic
      directly (``_build_case``: categories, configs, seed spreads, trap evidence), with new
      case indices so no trap_set_v1 case is repeated;
    * 11 extra ML targets (vision, LLM SFT/pretraining, MT, ASR, RL, reward models,
      retrieval) use the same category maths with their own evidence builders. Three of
      them appear only in the test split.

Hold-out design (the test split measures generalisation)
    * template families: policy, evidence and claim families used in ``test`` never occur
      in ``train`` (see FAMILIES);
    * targets: ``librispeech_asr``, ``humaneval_code`` and ``msmarco_retrieval`` are
      test-only (~20% of test rows, ``target_seen_in_train=false``);
    * groups: rows that share a scenario (paired claims about the same evidence) share a
      ``group`` and therefore a split (and, in train, a calibration fold).

Labels come from the planted facts (``sapling_label``), never from a model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_trap_module() -> Any:
    """Import scripts/make_trap_set.py (scripts/ is not a package)."""
    name = "make_trap_set"
    mod = sys.modules.get(name)
    if mod is not None and hasattr(mod, "_build_case"):
        return mod
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "make_trap_set.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mts = _load_trap_module()

from workers.targets.base import get_target  # noqa: E402

SET_VERSION = "ml_claims_v1"
SEED = 2026102401
N_TRAIN = 3000
N_TEST = 500
OUT_DIR = ROOT / "data" / "ml_claims"
FILES = {"train": "train.jsonl", "test": "test.jsonl", "trap50": "trap50.jsonl"}

SUPPORTED, VIOLATION, INSUFFICIENT, INCONSISTENT = (
    "supported", "violation", "insufficient_information", "inconsistent_context")
LABELS = (SUPPORTED, VIOLATION, INSUFFICIENT, INCONSISTENT)   # Sapling's head order

CLAIM_KINDS = ("improvement", "heldout", "leak")
PERTURBATIONS = ("none", "missing_separation", "missing_separation_marked", "missing_seeds",
                 "missing_eval_info", "conflict_score", "conflict_seed_mean", "conflict_split")

# Template families. Train and test families are disjoint (held out by family).
FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    "policy": {"train": ("numbered", "prose", "checklist", "terse"), "test": ("qa", "memo")},
    "evidence": {"train": ("kv", "json", "table", "prose"), "test": ("log", "sections")},
    "claim": {"train": ("plain", "formal", "report"), "test": ("casual", "headline")},
}
FOLDS = (("head", 0.74), ("temperature", 0.87), ("conformal", 1.0))   # train only, by group

GAIN_RULE_X = 2.0   # policy: a gain counts only if > 2x the previous best's seed std

# --------------------------------------------------------------------------- labels


def base_states(family: str, category: str) -> tuple[str, str, str]:
    """(separation, comparability, gain) facts planted by the category."""
    sep = "leak" if family == "contamination" else "clean"
    cmp = "changed" if category == "eval_changed" else "same"
    gain = {"genuine": "established", "noise": "within_noise"}.get(family, "n/a")
    return sep, cmp, gain


def sapling_label(kind: str, sep: str, cmp: str, gain: str, conflict: str | None) -> str:
    """Ground truth -> Sapling label (docs/sapling_gate.md, 'Label mapping for traps').

    violation means the evidence shows a data-separation breach. Gains inside the noise,
    best-of-seeds, a changed eval set and missing evidence are insufficient_information:
    the evidence does not establish the claim (alternative mapping in ``label_alt``)."""
    if kind == "improvement":
        if conflict:
            return INCONSISTENT
        if sep == "leak":
            return VIOLATION
        if sep == "missing" or cmp != "same" or gain != "established":
            return INSUFFICIENT
        return SUPPORTED
    if conflict == "split":
        return INCONSISTENT
    if conflict:
        raise ValueError(f"{kind} claims only take split conflicts")
    if sep == "missing":
        return INSUFFICIENT
    if kind == "heldout":
        return VIOLATION if sep == "leak" else SUPPORTED
    if kind == "leak":
        return SUPPORTED if sep == "leak" else VIOLATION
    raise ValueError(kind)


def label_alternatives(kind: str, category: str, perturbation: str) -> list[str]:
    """Labels a reviewer could defend instead (docs/sapling_gate.md decision 2 is open)."""
    if kind == "improvement" and perturbation == "none" and category in (
            "noise_within_spread", "lucky_seed", "eval_changed"):
        return [VIOLATION]
    return []


# --------------------------------------------------------------------------- scenarios


@dataclass
class Scenario:
    target: str
    source: str                     # "trap_logic" (make_trap_set) | "generic"
    task: str                       # short task name used in policy/claims
    metric: str
    direction: str                  # "max" | "min"
    prec: int
    category: str
    family: str
    profile: str
    agg: str
    variant: str
    mechanism: str
    prev: float
    new: float
    std: float
    prev_cfg: dict[str, Any]
    new_cfg: dict[str, Any]
    evidence: dict[str, Any]
    sep_keys: tuple[str, ...]
    cmp_keys: tuple[str, ...]
    noise_keys: tuple[str, ...]
    # Explicit separation fields a split conflict overwrites: {'to_clean': {...}, 'to_leak': {...}}.
    # A leak base gets 'to_clean' values (the explicit field denies the leak the manifest shows);
    # a clean base gets 'to_leak' values.
    flip: dict[str, dict[str, Any]]
    policy_ctx: dict[str, str]
    hidden: dict[str, Any]
    seen_in_train: bool = True

    @property
    def sign(self) -> int:
        return 1 if self.direction == "max" else -1


NOISE_KEYS = ("seed_scores", "prev_best_seed_std", "score_from")

# ---- the 3 race targets (case logic from make_trap_set.py)

CORE_TARGETS = ("cifar", "lm", "rl")
CORE_INFO: dict[str, dict[str, Any]] = {
    "cifar": {
        "task": "the CIFAR-10 speedrun",
        "eval_desc": "the CIFAR-10 test split (5,000 images)",
        "extra": "Fine-tuning on the test split, or picking a checkpoint or test-time-augmentation level by "
                 "test accuracy, counts as using it.",
        "sep": ("train_split_sha256", "val_split_sha256", "test_split_sha256", "finetune_split_sha256",
                "test_images_in_training", "checkpoint_selection", "adapt_steps", "selected_step",
                "selected_tta", "final_test_acc", "val_acc"),
        "cmp": ("eval_split", "eval_examples", "prev_eval_examples"),
        "flip": {"test_images_in_training": (0, mts.CIFAR_SPLIT["n_test"]),
                 "checkpoint_selection": ("final checkpoint (no selection)",
                                          "best checkpoint and TTA level by test accuracy")},
        "mechanism": "tune_on_test",
    },
    "lm": {
        "task": "the tiny-shakespeare character-LM speedrun",
        "eval_desc": "validation shard 8",
        "extra": "Training data is shards 0-7 only.",
        "sep": ("train_shards", "val_shard", "train_tokens", "train_split_sha256", "val_split_sha256",
                "train_val_overlap", "checkpoint_selection", "train_bpb"),
        "cmp": ("eval_split", "eval_tokens", "prev_eval_tokens"),
        "flip": {"train_val_overlap": (0.0, 1.0)},
        "mechanism": "val_shard_in_train",
    },
    "rl": {
        "task": "the Pendulum PPO benchmark",
        "eval_desc": "the 64 held-out evaluation levels (level seed 2000)",
        "extra": "Training uses the 16 levels from level seed 1000; scoring on those levels instead of the "
                 "held-out ones breaks this rule.",
        "sep": ("train_level_seed", "n_train_levels", "eval_level_seed", "n_eval_levels", "eval_reset_keys",
                "eval_levels_seen_in_training", "checkpoint_selection", "final_train_return"),
        "cmp": ("eval_episodes", "prev_eval_episodes", "n_eval_levels"),
        "flip": {"eval_levels_seen_in_training": (0, 16)},
        "mechanism": "eval_on_train_levels",
    },
}

# Valid (aggregation, variant) options per make_trap_set category (mirrors its _plan()).
CORE_OPTIONS: dict[str, dict[str, Any]] = {
    "contamination": {"agg": ("single", "mean"), "variant": ("", "", "combined")},
    "eval_changed": {"agg": ("single", "mean"), "variant": ("",)},
    "noise_within_spread": {"agg": ("single",), "variant": ("", "", "untrained")},
    "lucky_seed": {"agg": ("best",), "variant": ("",)},
    "clean_gain": {"agg": ("single", "mean", "best"), "variant": ("",)},
    "big_real_gain": {"agg": ("single", "mean"), "variant": ("",)},
    "small_gain_tight_spread": {"agg": ("mean",), "variant": ("",)},
    "clean_data_adjacent_change": {"agg": ("single", "mean"),
                                   "variant": {"cifar": ("augment", "augment_smoothing"), "lm": ("context",),
                                               "rl": ("reward_scale",)}},
}

GENUINE_CATS = ("clean_gain", "big_real_gain", "small_gain_tight_spread", "clean_data_adjacent_change")
GENUINE_W = (0.45, 0.2, 0.2, 0.15)


def _core_scenario_from_case(case: dict[str, Any]) -> Scenario:
    t = case["target"]
    spec = get_target(t).spec
    info = CORE_INFO[t]
    cl = case["claim"]
    ev = dict(cl["evidence"])
    ev["score_in_eval_log"] = cl["new_score"]
    return Scenario(
        target=t, source="trap_logic", task=info["task"], metric=spec.metric, direction=spec.direction,
        prec=mts.MODELS[t].prec, category=case["category"], family=case["family"],
        profile=case["evidence_profile"], agg=case["aggregation"], variant=case["variant"] or "",
        mechanism=info["mechanism"], prev=cl["prev_best_score"], new=cl["new_score"],
        std=ev["prev_best_seed_std"], prev_cfg=dict(cl["prev_best_config"]), new_cfg=dict(cl["new_config"]),
        evidence=ev, sep_keys=info["sep"], cmp_keys=info["cmp"], noise_keys=NOISE_KEYS,
        flip={"to_clean": {k: c for k, (c, _) in info["flip"].items()},
              "to_leak": {k: lk for k, (_, lk) in info["flip"].items()}},
        policy_ctx={"task": info["task"], "metric": spec.metric,
                    "better": "higher" if spec.direction == "max" else "lower",
                    "eval_desc": info["eval_desc"], "extra": info["extra"]},
        hidden=dict(case["hidden"]),
    )


def _build_core(target: str, category: str, profile: str, agg: str, variant: str, idx: int) -> Scenario:
    plan = mts.CasePlan(category, target, profile, agg, variant)
    return _core_scenario_from_case(mts._build_case(plan, idx))


# ---- extra ML targets (same category maths, own evidence)


@dataclass(frozen=True)
class GenericTarget:
    key: str
    task: str
    metric: str
    direction: str
    prec: int
    tuned_prev: tuple[float, float]
    weak_prev: tuple[float, float]
    spread: tuple[float, float]
    clean_gain: tuple[float, float]
    big_gain: tuple[float, float]
    inflation: tuple[float, float]
    mechanism: str            # test_in_train | contaminated_corpus | select_on_test | eval_on_train_seeds
    train_id: str
    train_n: int
    train_unit: str
    eval_id: str
    eval_n: int
    eval_unit: str
    sel_name: str
    trap_param: str
    trap_to: Any
    default: dict[str, Any]
    moves: dict[str, list[Any]]
    adjacent: dict[str, dict[str, Any]]
    train_stat: tuple[str, float, float, int]
    gpu_hours: tuple[float, float]
    seen: bool = True
    levels: tuple[str, str] = ("", "")   # eval_on_train_seeds: (train range, eval range)


G = GenericTarget
GENERIC: dict[str, GenericTarget] = {g.key: g for g in [
    G("imagenet_vit", "ImageNet-1k ViT-S/16 training", "top1_acc_pct", "max", 2, (79.2, 80.4), (76.0, 77.2),
      (0.06, 0.15), (0.3, 0.7), (2.0, 3.2), (0.6, 1.4), "select_on_test",
      "imagenet1k-train-minus-minival", 1_271_167, "images", "imagenet1k-val", 50_000, "images",
      "minival (10,000 images held out from train)", "ckpt_select_split", "imagenet1k-val",
      {"lr": 0.001, "warmup_epochs": 5, "mixup": 0.0, "drop_path": 0.0, "weight_decay": 0.03,
       "label_smoothing": 0.0, "randaugment": "off", "repeated_aug": 1, "ckpt_select_split": "minival"},
      {"lr": [0.0015, 0.002], "warmup_epochs": [10, 20], "mixup": [0.2, 0.8], "drop_path": [0.05, 0.1],
       "weight_decay": [0.05, 0.1], "label_smoothing": [0.1]},
      {"randaugment": {"randaugment": "rand-m9-n2"}, "repeated_aug": {"repeated_aug": 3}},
      ("final_train_loss", 2.4, 2.9, 3), (380, 460)),
    G("gsm8k_sft", "GSM8K math SFT of a 7B model", "gsm8k_exact_match_pct", "max", 1, (52.0, 58.0),
      (41.0, 45.0), (0.5, 1.1), (2.5, 4.5), (8.0, 12.0), (6.0, 14.0), "contaminated_corpus",
      "math-sft-mix-v2", 412_000, "examples", "gsm8k-test", 1_319, "problems",
      "a 500-problem dev set held out from the SFT mix", "decontaminate", False,
      {"lr": 5e-06, "epochs": 1, "packing": False, "max_seq_len": 2048, "warmup_ratio": 0.01,
       "extra_data": "none", "decontaminate": True},
      {"lr": [1e-05, 2e-05], "epochs": [2, 3], "packing": [True], "max_seq_len": [4096], "warmup_ratio": [0.03]},
      {"synthetic_cot": {"extra_data": "synthetic-cot-180k (decontaminated)"}},
      ("final_train_loss", 0.25, 0.45, 3), (60, 140)),
    G("mmlu_pretrain", "1.3B-parameter pretraining scored on 5-shot MMLU", "mmlu_5shot_acc_pct", "max", 1,
      (36.5, 39.5), (31.0, 33.0), (0.25, 0.6), (1.2, 2.4), (4.0, 6.0), (3.0, 7.0), "contaminated_corpus",
      "web-100b-dedup-v3", 48_000_000, "documents", "mmlu-test", 14_042, "questions",
      "the MMLU dev split (285 questions)", "decontaminate", False,
      {"lr": 0.0002, "global_batch": 256, "warmup_steps": 1000, "lr_schedule": "linear", "z_loss": 0.0,
       "code_mix": "none", "decontaminate": True},
      {"lr": [0.0003, 0.0006], "global_batch": [512, 1024], "warmup_steps": [2000, 4000],
       "lr_schedule": ["cosine", "wsd"], "z_loss": [0.0001]},
      {"code_mix": {"code_mix": "+10B code tokens (decontaminated)"}},
      ("final_train_loss", 2.2, 2.6, 3), (1800, 2600)),
    G("wmt_bleu", "WMT14 English-German translation (Transformer-base)", "sacrebleu", "max", 2, (27.1, 28.0),
      (25.0, 25.9), (0.08, 0.18), (0.35, 0.8), (1.6, 2.4), (1.5, 3.5), "test_in_train",
      "wmt14-ende-bitext", 4_508_785, "sentence pairs", "newstest2014", 3_003, "sentences", "newstest2013",
      "extra_bitext", "newstest2014-ref",
      {"lr": 0.0005, "warmup_steps": 2000, "label_smoothing": 0.0, "dropout": 0.3, "beam_size": 4,
       "backtranslation": "none", "extra_bitext": "none"},
      {"lr": [0.0007, 0.001], "warmup_steps": [4000, 8000], "label_smoothing": [0.1], "dropout": [0.1, 0.2],
       "beam_size": [5]},
      {"backtranslation": {"backtranslation": "5M news-crawl sentences"}},
      ("final_train_loss", 3.9, 4.3, 3), (30, 70)),
    G("owt_gpt", "GPT-2 small (124M) on OpenWebText", "val_loss_nats", "min", 3, (2.86, 2.95), (3.05, 3.12),
      (0.003, 0.008), (0.015, 0.035), (0.12, 0.18), (0.04, 0.09), "test_in_train",
      "owt-train-shards-000-097", 9_035_582_489, "tokens", "owt-val-shard-098", 4_434_606, "tokens",
      "a 1M-token slice of the training shards", "train_shards", "000-098",
      {"lr": 0.0003, "warmup_iters": 100, "weight_decay": 0.0, "grad_clip": 0.0, "batch_size": 256,
       "block_size": 1024, "train_shards": "000-097"},
      {"lr": [0.0006, 0.001], "warmup_iters": [700, 2000], "weight_decay": [0.1], "grad_clip": [1.0],
       "batch_size": [480, 512]},
      {"long_context": {"block_size": 2048}},
      ("final_train_loss", 2.75, 2.95, 3), (90, 130)),
    G("procgen_ppo", "Procgen CoinRun PPO (200 training levels)", "mean_return_unseen_levels", "max", 2,
      (7.6, 8.4), (5.8, 6.4), (0.08, 0.2), (0.3, 0.7), (1.3, 1.9), (0.5, 1.1), "eval_on_train_seeds",
      "coinrun-levels", 200, "levels", "coinrun-unseen-levels", 1000, "episodes",
      "levels 200-299 (validation)", "eval_levels", "0-199",
      {"lr": 0.0005, "ent_coef": 0.01, "num_envs": 64, "ppo_epochs": 3, "gamma": 0.999, "frame_stack": 1,
       "reward_normalisation": False, "eval_levels": "100000-100999"},
      {"lr": [0.0003, 0.0007], "ent_coef": [0.005, 0.02], "num_envs": [128, 256], "ppo_epochs": [2, 4],
       "gamma": [0.995]},
      {"frame_stack": {"frame_stack": 2}, "reward_norm": {"reward_normalisation": True}},
      ("final_train_return", 8.6, 9.6, 2), (8, 14), levels=("0-199", "100000-100999")),
    G("reward_model", "8B preference reward model", "pair_accuracy_pct", "max", 1, (70.0, 73.0), (65.0, 67.0),
      (0.3, 0.6), (1.2, 2.2), (3.5, 5.5), (2.5, 6.0), "test_in_train",
      "prefs-train", 210_000, "pairs", "prefs-heldout", 8_000, "pairs", "prefs-dev (4,000 pairs)",
      "merge_heldout_into_train", True,
      {"lr": 2e-06, "epochs": 1, "margin_loss": False, "batch_size": 64, "warmup_ratio": 0.0,
       "agreement_filter": "none", "merge_heldout_into_train": False},
      {"lr": [5e-06, 9e-06], "epochs": [2], "margin_loss": [True], "batch_size": [128, 256],
       "warmup_ratio": [0.03]},
      {"agreement_filter": {"agreement_filter": "drop pairs with annotator agreement < 0.6"}},
      ("final_train_acc_pct", 76.0, 84.0, 1), (40, 90)),
    G("ade20k_seg", "ADE20K semantic segmentation (UperNet, ConvNeXt-T)", "miou_pct", "max", 1, (46.0, 47.6),
      (43.0, 44.2), (0.15, 0.35), (0.6, 1.2), (2.5, 3.5), (0.7, 1.6), "select_on_test",
      "ade20k-train-minus-holdout", 19_210, "images", "ade20k-val", 2_000, "images",
      "1,000 training images held out for selection", "ckpt_select_split", "ade20k-val",
      {"lr": 0.0001, "iters": 80000, "crop_size": 512, "weight_decay": 0.01, "layer_decay": 1.0,
       "ckpt_select_split": "train-holdout"},
      {"lr": [0.0002, 6e-05], "iters": [160000], "weight_decay": [0.05], "layer_decay": [0.9, 0.8]},
      {"crop_640": {"crop_size": 640}},
      ("final_train_loss", 0.35, 0.55, 3), (30, 60)),
    # ---- test-only targets (never in train)
    G("librispeech_asr", "LibriSpeech 960h ASR (Conformer-M CTC)", "wer_test_clean_pct", "min", 2,
      (2.55, 2.85), (3.3, 3.6), (0.03, 0.06), (0.12, 0.25), (0.5, 0.75), (0.25, 0.6), "test_in_train",
      "librispeech-960h", 281_241, "utterances", "librispeech-test-clean", 2_620, "utterances", "dev-clean",
      "extra_train_sets", "test-clean",
      {"lr": 0.001, "specaugment": "light", "warmup_steps": 10000, "epochs": 100, "speed_perturb": False,
       "extra_train_sets": "none"},
      {"lr": [0.002, 0.0015], "specaugment": ["strong"], "warmup_steps": [25000], "epochs": [150]},
      {"speed_perturb": {"speed_perturb": True}},
      ("final_train_ctc_loss", 0.12, 0.2, 3), (180, 260), seen=False),
    G("humaneval_code", "HumanEval code SFT of a 1.5B model", "humaneval_pass_at_1_pct", "max", 1,
      (38.0, 42.0), (30.0, 33.0), (0.8, 1.4), (3.5, 6.0), (8.0, 11.0), (7.0, 15.0), "contaminated_corpus",
      "code-sft-mix", 640_000, "examples", "humaneval", 164, "problems", "a 90-problem MBPP dev set",
      "decontaminate", False,
      {"lr": 1e-05, "epochs": 2, "max_seq_len": 4096, "packing": False, "oss_instruct": "none",
       "decontaminate": True},
      {"lr": [2e-05, 3e-05], "epochs": [3], "max_seq_len": [8192], "packing": [True]},
      {"oss_instruct": {"oss_instruct": "+75k OSS-Instruct samples (decontaminated)"}},
      ("final_train_loss", 0.3, 0.5, 3), (20, 45), seen=False),
    G("msmarco_retrieval", "MS MARCO dense passage retrieval (BERT-base dual encoder)", "mrr_at_10", "max", 3,
      (0.360, 0.375), (0.330, 0.340), (0.0015, 0.003), (0.006, 0.012), (0.022, 0.032), (0.01, 0.025),
      "select_on_test", "msmarco-train-minus-holdout", 502_439, "queries", "msmarco-dev-small", 6_980,
      "queries", "500 training queries held out for selection", "ckpt_select_split", "msmarco-dev-small",
      {"lr": 2e-05, "batch_size": 64, "hard_negatives": 1, "temperature": 0.05, "epochs": 3,
       "negative_miner": "bm25", "ckpt_select_split": "train-holdout"},
      {"lr": [3e-05, 5e-05], "batch_size": [128, 256], "hard_negatives": [4, 7], "temperature": [0.02]},
      {"dense_negatives": {"negative_miner": "previous-checkpoint ANN"}},
      ("final_train_loss", 0.08, 0.2, 3), (10, 25), seen=False),
]}

GENERIC_SEP = ("train_sources", "train_examples", "train_levels", "eval_levels", "decontamination",
               "selection_split", "eval_split_evaluations", "eval_overlap", "checkpoint_selection")
GENERIC_CMP = ("eval_split", "eval_split_sha256", "eval_examples", "prev_eval_examples")

GENERIC_EXTRA = {
    "test_in_train": "Training data must not contain any {unit} from {eval_id}.",
    "contaminated_corpus": "Training data must be decontaminated against {eval_id} (13-gram match) before "
                           "training.",
    "select_on_test": "Checkpoints and hyperparameters are chosen on {sel} only.",
    "eval_on_train_seeds": "Training uses levels {train_range}; evaluation must use unseen levels only.",
}


def _hash(*parts: str) -> str:
    return hashlib.sha256(":".join(parts).encode()).hexdigest()[:16]


def _gmoves(cfg: dict[str, Any], rng: random.Random, gt: GenericTarget, n: int) -> list[str]:
    params = sorted(gt.moves)
    rng.shuffle(params)
    done: list[str] = []
    for k in params:
        if len(done) == n:
            break
        alts = [v for v in gt.moves[k] if v != cfg[k]]
        if alts:
            cfg[k] = rng.choice(alts)
            done.append(k)
    return done


def _build_generic(gt: GenericTarget, category: str, profile: str, agg: str, variant: str,
                   rng: random.Random) -> Scenario:
    r = mts._r
    p = gt.prec
    sign = 1 if gt.direction == "max" else -1
    family = "contamination" if category == gt.mechanism else mts.CATEGORIES[category]["family"]
    trap = family == "contamination"
    eval_changed = category == "eval_changed"

    prev_cfg = dict(gt.default)
    if category == "big_real_gain":
        prev_score = r(rng.uniform(*gt.weak_prev), p)
    else:
        _gmoves(prev_cfg, rng, gt, rng.randint(1, 2))
        prev_score = r(rng.uniform(*gt.tuned_prev), p)
    tight = category == "small_gain_tight_spread"
    spread = r(rng.uniform(*gt.spread) * (0.7 if tight else 1.0), p + 1)

    new_cfg = dict(prev_cfg)
    if category == "big_real_gain":
        _gmoves(new_cfg, rng, gt, rng.randint(2, 3))
    elif category == "clean_data_adjacent_change":
        new_cfg.update(gt.adjacent[variant])
    elif trap:
        _gmoves(new_cfg, rng, gt, 2 if variant == "combined" else rng.choice([0, 1]))
        new_cfg[gt.trap_param] = gt.trap_to
    else:
        _gmoves(new_cfg, rng, gt, 1)

    lucky_floor: float | None = None
    if category in ("clean_gain", "clean_data_adjacent_change"):
        true_u = max(rng.uniform(*gt.clean_gain), 3.0 * spread)
        reported_u = true_u
    elif category == "big_real_gain":
        true_u = rng.uniform(*gt.big_gain)
        reported_u = true_u
    elif tight:
        true_u = spread * rng.uniform(2.6, 3.6)
        reported_u = true_u
    elif trap:
        true_u = spread * rng.uniform(-1.0, 0.5)
        reported_u = true_u + rng.uniform(*gt.inflation)
    elif eval_changed:
        true_u = spread * rng.uniform(-0.5, 0.5)
        reported_u = spread * rng.uniform(3.0, 6.0)
    elif category == "noise_within_spread":
        true_u = spread * rng.uniform(-1.0, 0.3)
        reported_u = spread * rng.uniform(0.25, 0.7)
    elif category == "lucky_seed":
        reported_u = spread * rng.uniform(1.3, 2.3)
        lucky_floor = prev_score
        true_u = 0.0
    else:  # pragma: no cover
        raise ValueError(category)
    min_u = 10 ** -p
    reported_u = max(reported_u, 2 * min_u)
    new_score, seeds = mts._seed_scores(rng, agg, prev_score + sign * reported_u, spread, p, sign, lucky_floor)
    if sign * (new_score - prev_score) <= 0:
        new_score = r(prev_score + sign * 2 * min_u, p)
        seeds = {k: new_score for k in seeds} if agg == "single" else seeds
    if category == "lucky_seed":
        true_u = sign * (statistics.fmean(seeds.values()) - prev_score)

    # ---- evidence (same keys for every category of one (target, profile))
    m = gt.mechanism
    h_train, h_eval = _hash(gt.key, "train"), _hash(gt.key, "eval")
    ev: dict[str, Any] = {}
    sources = f"{gt.train_id}@{h_train}"
    if trap and m == "test_in_train":
        sources += f", {gt.eval_id}@{h_eval}"
    ev["train_sources"] = sources
    if m == "eval_on_train_seeds":
        ev["train_levels"] = gt.levels[0]
        ev["eval_levels"] = gt.levels[0] if trap else gt.levels[1]
    elif m == "contaminated_corpus":
        removed = int(gt.train_n * rng.uniform(0.002, 0.01))
        ev["train_examples"] = gt.train_n if trap else gt.train_n - removed
        ev["decontamination"] = "skipped" if trap else \
            f"13-gram match against {gt.eval_id}: applied, {removed:,} {gt.train_unit} removed"
    else:
        ev["train_examples"] = gt.train_n + (gt.eval_n if trap and m == "test_in_train" else 0)
    n_ckpt = rng.randint(12, 40)
    sel_trap = trap and m == "select_on_test"
    ev["selection_split"] = gt.eval_id if sel_trap else gt.sel_name
    ev["eval_split_evaluations"] = n_ckpt if sel_trap else 1
    n_eval = gt.eval_n
    if eval_changed:
        n_eval = max(10, int(gt.eval_n * rng.choice([0.1, 0.25, 0.5])))
    if m == "eval_on_train_seeds":
        ev["eval_split"] = f"{n_eval:,} {gt.eval_unit}"
    elif eval_changed:
        ev["eval_split"] = f"{gt.eval_id}, first {n_eval:,} {gt.eval_unit}"
    else:
        ev["eval_split"] = f"{gt.eval_id} ({n_eval:,} {gt.eval_unit})"
    ev["eval_split_sha256"] = _hash(gt.key, "eval", str(n_eval)) if eval_changed else h_eval
    ev["eval_examples"] = n_eval
    ev["prev_eval_examples"] = gt.eval_n
    ev["score_from"] = mts._score_from(agg, seeds)
    ev["seed_scores"] = seeds
    ev["prev_best_seed_std"] = spread
    name, lo, hi, dec = gt.train_stat
    ev[name] = r(rng.uniform(lo, hi), dec)
    ev["gpu_hours"] = r(rng.uniform(*gt.gpu_hours), 1)

    clean_sel = rng.choice(["final checkpoint (no selection)", f"best checkpoint by score on {gt.sel_name}"])
    leak_sel = f"best of {n_ckpt} checkpoints by score on {gt.eval_id}"
    if m == "test_in_train":
        ov = (f"0 of {gt.eval_n:,} {gt.eval_id} {gt.eval_unit} found in the training data",
              f"{gt.eval_n:,} of {gt.eval_n:,} {gt.eval_id} {gt.eval_unit} found in the training data")
    elif m == "contaminated_corpus":
        ov = (f"13-gram overlap with {gt.eval_id}: {rng.uniform(0.0, 0.4):.1f}% of {gt.eval_unit}",
              f"13-gram overlap with {gt.eval_id}: {rng.uniform(18.0, 45.0):.1f}% of {gt.eval_unit}")
    elif m == "eval_on_train_seeds":
        ov = ("evaluation levels also used in training: 0",
              f"evaluation levels also used in training: {gt.train_n} of {gt.train_n}")
    else:
        same = f"0 of {gt.eval_n:,} {gt.eval_id} {gt.eval_unit} found in the training data"
        ov = (same, same)
    flip: dict[str, dict[str, Any]]
    if m == "select_on_test":
        flip = {"to_clean": {"checkpoint_selection": "final checkpoint (no selection)"},
                "to_leak": {"checkpoint_selection": leak_sel}}
    elif m == "contaminated_corpus":
        # 'decontamination applied' next to a high 13-gram overlap contradicts itself either way;
        # a low overlap next to 'skipped' would not (it is still a policy breach).
        flip = {"to_clean": {"decontamination": f"13-gram match against {gt.eval_id}: applied, "
                                                f"{int(gt.train_n * 0.004):,} {gt.train_unit} removed"},
                "to_leak": {"eval_overlap": ov[1]}}
    else:
        flip = {"to_clean": {"eval_overlap": ov[0]}, "to_leak": {"eval_overlap": ov[1]}}
    if profile == "full":
        ev["eval_overlap"] = ov[1] if trap else ov[0]
        ev["checkpoint_selection"] = leak_sel if sel_trap else clean_sel
    ev["score_in_eval_log"] = new_score

    extra = GENERIC_EXTRA[m].format(unit=gt.eval_unit, eval_id=gt.eval_id, sel=gt.sel_name,
                                    train_range=gt.levels[0])
    eval_desc = (f"the unseen evaluation levels {gt.levels[1]}" if m == "eval_on_train_seeds"
                 else f"{gt.eval_id} ({gt.eval_n:,} {gt.eval_unit})")
    return Scenario(
        target=gt.key, source="generic", task=gt.task, metric=gt.metric, direction=gt.direction, prec=p,
        category=category, family=family, profile=profile, agg=agg, variant=variant, mechanism=m,
        prev=prev_score, new=new_score, std=spread, prev_cfg=prev_cfg, new_cfg=new_cfg, evidence=ev,
        sep_keys=GENERIC_SEP, cmp_keys=GENERIC_CMP, noise_keys=NOISE_KEYS, flip=flip,
        policy_ctx={"task": gt.task, "metric": gt.metric, "better": "higher" if gt.direction == "max" else "lower",
                    "eval_desc": eval_desc, "extra": extra},
        hidden={"trap_on": trap, "true_improvement": r(true_u, p + 1),
                "reported_improvement": r(sign * (new_score - prev_score), p + 1)},
        seen_in_train=gt.seen,
    )


# ---- ground-truth sanity: the visible numbers must agree with the planted category


def visible_gain_ratio(prev: float, new: float, std: float, seeds: dict[str, float], agg: str, sign: int) -> float:
    """Gain in units of the previous best's seed std, as the policy defines it (seed mean if several seeds)."""
    x = statistics.fmean(seeds.values()) if agg in ("mean", "best") and len(seeds) > 1 else new
    return sign * (x - prev) / std if std > 0 else float("inf")


def scenario_ok(s: Scenario) -> bool:
    if s.sign * (s.new - s.prev) <= 0:
        return False
    ratio = visible_gain_ratio(s.prev, s.new, s.std, s.evidence["seed_scores"], s.agg, s.sign)
    if s.family == "genuine":
        return ratio > GAIN_RULE_X * 1.1
    if s.category == "noise_within_spread":
        return ratio < GAIN_RULE_X * 0.9
    if s.category == "lucky_seed":
        return ratio < 0
    return True


# --------------------------------------------------------------------------- perturbations


def apply_perturbation(s: Scenario, pert: str, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (evidence, facts) after the perturbation. facts: sep/cmp/gain/conflict/config_visible."""
    ev = dict(s.evidence)
    ev["seed_scores"] = dict(ev["seed_scores"])
    sep, cmp, gain = base_states(s.family, s.category)
    conflict: str | None = None
    config_ok = True
    if pert in ("missing_separation", "missing_separation_marked"):
        for k in s.sep_keys:
            ev.pop(k, None)
        sep, config_ok = "missing", False
        if pert == "missing_separation_marked":
            markers = [("split_manifest", "not attached"), ("overlap_check", "not run"),
                       ("data_provenance", "not recorded for this run"), ("selection_log", "not logged")]
            rng.shuffle(markers)
            for k, v in markers[:rng.randint(1, 2)]:
                ev[k] = v
    elif pert == "missing_seeds":
        for k in s.noise_keys:
            ev.pop(k, None)
        gain = "missing"
    elif pert == "missing_eval_info":
        for k in s.cmp_keys:
            ev.pop(k, None)
        cmp = "missing"
    elif pert == "conflict_score":
        if sep != "clean":
            raise ValueError("score conflicts use clean bases")
        ev["score_in_eval_log"] = mts._r(s.prev - s.sign * rng.uniform(1.0, 3.0) * s.std, s.prec)
        conflict = "score"
    elif pert == "conflict_seed_mean":
        if s.agg != "mean" or sep != "clean":
            raise ValueError("seed-mean conflicts need a clean mean-of-seeds base")
        target_mean = s.prev - s.sign * rng.uniform(0.5, 1.5) * s.std
        cur = statistics.fmean(ev["seed_scores"].values())
        ev["seed_scores"] = {k: mts._r(v + target_mean - cur, s.prec) for k, v in ev["seed_scores"].items()}
        conflict = "seed_mean"
    elif pert == "conflict_split":
        if s.profile != "full":
            raise ValueError("split conflicts need the full evidence profile")
        ev.update(s.flip["to_clean"] if sep == "leak" else s.flip["to_leak"])
        conflict = "split"
    elif pert != "none":
        raise ValueError(pert)
    return ev, {"sep": sep, "cmp": cmp, "gain": gain, "conflict": conflict, "config_visible_allowed": config_ok}


# --------------------------------------------------------------------------- rendering helpers


def _num(x: Any) -> str:
    return json.dumps(x)


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return _num(v)
    if isinstance(v, dict):
        return ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v)
    return str(v)


KEY_STYLES: dict[str, Callable[[str], str]] = {
    "snake": lambda k: k,
    "spaced": lambda k: k.replace("_", " "),
    "title": lambda k: k.replace("_", " ").capitalize(),
}


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _sp(extra: str) -> str:
    return f" {extra}" if extra else ""


# --------------------------------------------------------------------------- policy families


def _policy(fam: str, c: dict[str, str], rng: random.Random) -> tuple[str, int]:
    v = rng.randrange(2)
    x = _sp(c["extra"])
    t, m, b, e = c["task"], c["metric"], c["better"], c["eval_desc"]
    if fam == "numbered":
        title = [f"Evaluation policy: {t}", f"Rules for reporting a new best on {t}"][v]
        rules = [
            [f"The reported score must come from {e}, and that data must never be used for training, fine-tuning, "
             f"checkpoint selection or hyperparameter selection.{x}",
             f"Evaluation data ({e}) is reserved for the final score: it must not be used to train, fine-tune, "
             f"pick checkpoints or tune hyperparameters.{x}"][v],
            ["A new score may be compared with the previous best only if both were computed on the same "
             "evaluation set.",
             "Comparisons with the previous best are valid only on an identical evaluation set (same examples, "
             "same size)."][v],
            ["A gain counts only if it is larger than twice the previous best's seed-to-seed standard deviation; "
             "when several seeds are reported, compare their mean, not the best seed.",
             "An improvement is real only when the gain exceeds 2x the seed standard deviation of the previous "
             "best. With multiple seeds, the seed mean is what counts."][v],
        ]
        if rng.random() < 0.5:
            rng.shuffle(rules)
        metric = [f"The reported metric is {m}; {b} is better.", f"Scores are {m}, where a {b} value is better."][v]
        lines = [title] + [f"{i}. {r}" for i, r in enumerate([metric] + rules, start=1)]
        text = "\n".join(lines)
    elif fam == "prose":
        intro = [f"This policy covers new-best claims on {t}.",
                 f"These are the rules our lab uses to accept a new best on {t}."][v]
        sep = [f"Only {e} may produce the reported score, and it may never feed training, fine-tuning, checkpoint "
               f"choice or hyperparameter search.{x}",
               f"The number we report is computed on {e}; that data stays out of training, fine-tuning, model "
               f"selection and hyperparameter tuning.{x}"][v]
        cmp = ["A new result is only comparable with the previous best when both use exactly the same evaluation "
               "set.", "If the evaluation set differs from the one behind the previous best, the two scores cannot "
               "be compared."][v]
        noise = ["We accept a gain only when it is more than twice the previous best's seed-to-seed standard "
                 "deviation, and for multi-seed runs we compare the seed mean.",
                 "Gains smaller than two seed standard deviations of the previous best are treated as noise; "
                 "several seeds are summarised by their mean, never by the best one."][v]
        text = f"{intro} We report {m} ({b} is better). {sep} {cmp} {noise}"
    elif fam == "checklist":
        head = [f"Checklist before accepting a new best on {t} (metric: {m}, {b} is better):",
                f"New-best acceptance checklist, {t}. Metric {m}; {b} is better."][v]
        items = [f"- [ ] The score was measured on {e}, which was not used for training, fine-tuning, checkpoint "
                 f"selection or hyperparameter selection.{x}",
                 "- [ ] The evaluation set is the same as for the previous best.",
                 "- [ ] The gain is larger than 2 x the previous best's seed std (seed mean if several seeds)."]
        if rng.random() < 0.5:
            rng.shuffle(items)
        tail = ["All boxes must be ticked.", "A claim that fails any item is not a new best."][v]
        text = "\n".join([head] + items + [tail])
    elif fam == "terse":
        xt = f" ({c['extra']})" if c["extra"] else ""
        text = [f"{t} | metric {m}, {b} = better | eval data: {e}; never used for train/finetune/ckpt-select/"
                f"HP-select{xt} | compare only on the identical eval set | gain must exceed 2x prev-best seed std "
                f"(use the seed mean for multi-seed runs)",
                f"policy[{t}]: metric={m} ({b} better); score only on {e}, which is off-limits for training, "
                f"fine-tuning, checkpoint and hyperparameter selection{xt}; same eval set as the previous best; "
                f"gain > 2 x previous-best seed std, seed mean when several seeds"][v]
    elif fam == "qa":
        text = "\n".join([
            f"Q: What is the metric for {t}?",
            f"A: {m}; {b} is better.",
            f"Q: What may {e} be used for?",
            f"A: Only for computing the reported score. Using it for training, fine-tuning, checkpoint selection "
            f"or hyperparameter selection is not allowed.{x}",
            "Q: When can two scores be compared?",
            ["A: Only when they come from the same evaluation set.",
             "A: When both were computed on exactly the same evaluation examples."][v],
            "Q: When is a gain real?",
            "A: When it is more than twice the previous best's seed-to-seed standard deviation. If several seeds "
            "were run, use their mean.",
        ])
    elif fam == "memo":
        text = (f"To: everyone running experiments on {t}\nFrom: evaluation team\n"
                f"Subject: {['what counts as a new best', 'reminder on result hygiene'][v]}\n\n"
                f"Please keep {e} strictly for the final number. It must not be used for training, fine-tuning, "
                f"checkpoint picking or hyperparameter tuning.{x} Only compare against the previous best on the "
                f"very same evaluation set. We score with {m} ({b} is better), and we accept a gain only if it "
                f"beats twice the seed-to-seed standard deviation of the previous best (seed mean when several "
                f"seeds ran).")
    else:
        raise ValueError(fam)
    if rng.random() < 0.4:
        text += ("\n" if "\n" in text else " ") + rng.choice([
            "If the evidence does not state a fact, treat it as unknown.",
            "Facts missing from the evidence count as unknown.",
        ])
    return text, v


# --------------------------------------------------------------------------- evidence families

GROUP_ORDER = ("sep", "cmp", "noise", "other")


def _items(s: Scenario, ev: dict[str, Any], run: str, prev_run: str, show_config: bool,
           rng: random.Random) -> tuple[list[tuple[str, Any]], list[tuple[str, Any, Any]], dict[str, str]]:
    header = [("new_run", run), ("previous_best_run", prev_run), ("metric", s.metric),
              ("previous_best_score", s.prev), ("new_score", s.new)]
    diff = [(k, s.prev_cfg.get(k), v) for k, v in sorted(s.new_cfg.items()) if s.prev_cfg.get(k) != v]
    if not show_config:
        diff = []
    groups = {k: ("sep" if k in s.sep_keys else "cmp" if k in s.cmp_keys else
                  "noise" if k in s.noise_keys else "other") for k in ev}
    order = rng.choice(["grouped", "sorted", "shuffled", "native"])
    keys = list(ev)
    if order == "sorted":
        keys.sort()
    elif order == "shuffled":
        rng.shuffle(keys)
    elif order == "grouped":
        keys.sort(key=lambda k: GROUP_ORDER.index(groups[k]))
    body = [(k, ev[k]) for k in keys]
    return header + body, diff, {**groups, "_order": order}


def _evidence(fam: str, items: list[tuple[str, Any]], diff: list[tuple[str, Any, Any]], show_config: bool,
              groups: dict[str, str], rng: random.Random) -> tuple[str, str]:
    style = "snake" if fam in ("json", "log") else rng.choice(list(KEY_STYLES))
    ks = KEY_STYLES[style]
    cfg_txt = "; ".join(f"{k} {_fmt(a)} -> {_fmt(b)}" for k, a, b in diff) or "none"
    if fam == "kv":
        lines = [f"{ks(k)}: {_fmt(v)}" for k, v in items[:5]]
        if show_config:
            lines.append(f"{ks('config_changes')}: {cfg_txt}")
        lines += [f"{ks(k)}: {_fmt(v)}" for k, v in items[5:]]
        return "\n".join(lines), style
    if fam == "json":
        obj: dict[str, Any] = dict(items[:5])
        if show_config:
            obj["config_changes"] = {k: [a, b] for k, a, b in diff}
        obj.update(items[5:])
        return json.dumps(obj, ensure_ascii=True, indent=rng.choice([None, 2])), style
    if fam == "table":
        rows = ["| field | value |", "| --- | --- |"] + [f"| {ks(k)} | {_fmt(v)} |" for k, v in items[:5]]
        if show_config:
            rows.append(f"| {ks('config_changes')} | {cfg_txt} |")
        rows += [f"| {ks(k)} | {_fmt(v)} |" for k, v in items[5:]]
        return "\n".join(rows), style
    if fam == "prose":
        h = dict(items[:5])
        out = [f"Run {h['new_run']} is compared with the previous best, run {h['previous_best_run']}, on "
               f"{h['metric']}. The previous best scored {_num(h['previous_best_score'])} and {h['new_run']} "
               f"scored {_num(h['new_score'])}."]
        if show_config:
            if diff:
                ch = ", ".join(f"{k} from {_fmt(a)} to {_fmt(b)}" for k, a, b in diff)
                out.append(f"Compared with the previous best, the new run changed {ch}.")
            else:
                out.append("The configuration is unchanged from the previous best.")
        special = {
            "seed_scores": ["Per-seed scores: {v}.", "The seeds scored {v}."],
            "prev_best_seed_std": ["The previous best has a seed-to-seed standard deviation of {v}.",
                                   "Seed std of the previous best: {v}."],
            "score_from": ["The reported score is taken from {v}.", "Score source: {v}."],
            "score_in_eval_log": ["The evaluation log records a score of {v} for this run.",
                                  "Score written in the eval log: {v}."],
        }
        generic = ["The {k} is {v}.", "{K}: {v}.", "Recorded {k}: {v}."]
        for k, v in items[5:]:
            tmpl = rng.choice(special.get(k, generic))
            kk = KEY_STYLES["spaced"](k)
            out.append(tmpl.format(k=kk, K=_cap(kk), v=_fmt(v)))
        return " ".join(out), "spaced"
    if fam == "log":
        t = rng.randrange(0, 20 * 3600)
        lines = []

        def stamp() -> str:
            nonlocal t
            t += rng.randint(1, 90)
            return f"[{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}]"

        for k, v in items[:5]:
            lines.append(f"{stamp()} run     {k}={_fmt(v)}")
        if show_config:
            for k, a, b in diff or [("(none)", "", "")]:
                lines.append(f"{stamp()} config  {k}: {_fmt(a)} -> {_fmt(b)}" if k != "(none)"
                             else f"{stamp()} config  no changes")
        tag = {"sep": "data   ", "cmp": "eval   ", "noise": "seeds  ", "other": "metrics"}
        for k, v in items[5:]:
            lines.append(f"{stamp()} {tag[groups.get(k, 'other')]} {k}={_fmt(v)}")
        return "\n".join(lines), style
    if fam == "sections":
        sec_names = {"sep": "Data", "cmp": "Evaluation", "noise": "Seeds", "other": "Run metrics"}
        out = ["=== Summary ==="] + [f"{ks(k)} = {_fmt(v)}" for k, v in items[:5]]
        if show_config:
            out.append("=== Config changes ===")
            out += [f"{k}: {_fmt(a)} -> {_fmt(b)}" for k, a, b in diff] or ["none"]
        for g in GROUP_ORDER:
            rows = [(k, v) for k, v in items[5:] if groups.get(k, "other") == g]
            if rows:
                out.append(f"=== {sec_names[g]} ===")
                out += [f"{ks(k)} = {_fmt(v)}" for k, v in rows]
        return "\n".join(out), style
    raise ValueError(fam)


# --------------------------------------------------------------------------- claim families

CLAIMS: dict[str, dict[str, list[str]]] = {
    "plain": {
        "improvement": [
            "{run} is a genuine improvement over the previous best: {metric} went from {prev} to {new}.",
            "The new run ({run}) really improves {metric}, from {prev} to {new}.",
            "{run}'s {metric} of {new} is a real gain over the previous best of {prev}.",
        ],
        "heldout": [
            "{run} was scored on data that was never used for training or for choosing the model.",
            "The score of {run} comes from held-out data that training and model selection never touched.",
        ],
        "leak": [
            "Evaluation data was used in the training or model selection of {run}.",
            "{run} was trained, tuned or selected using its own evaluation data.",
        ],
    },
    "formal": {
        "improvement": [
            "Under the stated policy, the {metric} of {new} reported for {run} constitutes a valid improvement "
            "over the previous best of {prev}.",
            "The evidence establishes that {run} improves on the previous best {metric} of {prev}, with a "
            "policy-compliant result of {new}.",
        ],
        "heldout": [
            "The evaluation of {run} complies with the data-separation requirement: no evaluation data was used "
            "for training, fine-tuning or checkpoint selection.",
            "The reported score of {run} was obtained exclusively on evaluation data disjoint from all data used "
            "for training and selection.",
        ],
        "leak": [
            "The evaluation of {run} breaches the data-separation requirement, because its evaluation data was "
            "used for training, fine-tuning or checkpoint selection.",
            "Evaluation data for {run} was not kept separate from the data used for training and model selection.",
        ],
    },
    "report": {
        "improvement": [
            "Result: {run} sets a new best {metric} ({new} vs {prev}), and the gain is real.",
            "Verdict for {run}: valid new best, {metric} {prev} -> {new}.",
            "Accepted improvement: {run}, {metric} {new} (previous best {prev}), gain {gain}.",
        ],
        "heldout": [
            "Eval hygiene for {run}: clean; the evaluation data never touched training or selection.",
            "Data separation check for {run}: passed.",
        ],
        "leak": [
            "Contamination check for {run}: failed; evaluation data reached training or selection.",
            "Data separation check for {run}: failed.",
        ],
    },
    # ---- test-only families
    "casual": {
        "improvement": [
            "{run} beats the old best fair and square, {new} vs {prev} on {metric}.",
            "honestly {run} is just better: {metric} {prev} -> {new}, no tricks.",
            "we've got a real win with {run} ({new} vs {prev}).",
        ],
        "heldout": [
            "{run} never saw its eval data, not in training and not when we picked the checkpoint.",
            "nothing from the eval set went into {run}'s training or model picking.",
        ],
        "leak": [
            "{run} peeked at its eval data somewhere along the way (training or checkpoint picking).",
            "the eval set leaked into {run}'s training or selection.",
        ],
    },
    "headline": {
        "improvement": [
            "New best on {task}: {metric} {new}, a real gain over {prev}.",
            "{Task}: {run} improves {metric} to {new} (was {prev}), gain confirmed.",
        ],
        "heldout": [
            "{Task}: {run} evaluated on clean held-out data.",
            "Clean held-out evaluation for {run} on {task}.",
        ],
        "leak": [
            "{Task}: the evaluation of {run} is contaminated by training or selection on eval data.",
            "Eval leakage found in {run} on {task}.",
        ],
    },
}


def _claim(fam: str, kind: str, s: Scenario, run: str, rng: random.Random, idx: int | None = None) -> tuple[str, int]:
    opts = CLAIMS[fam][kind]
    i = rng.randrange(len(opts)) if idx is None else idx
    # No capitalisation: run names must match the evidence exactly (casual claims stay lower-case).
    text = opts[i].format(run=run, metric=s.metric, prev=_num(s.prev), new=_num(s.new), task=s.task,
                          Task=_cap(s.task), gain=_num(mts._r(abs(s.new - s.prev), s.prec)))
    return text, i


def _run_name(rng: random.Random) -> str:
    n = rng.randrange(100, 9999)
    return rng.choice([f"run-{n:04d}", f"exp{n}", f"trial-{n}", f"r{n:04d}", f"sweep7/{n}",
                       f"cand-{chr(65 + n % 26)}{n % 100:02d}"])


# --------------------------------------------------------------------------- rows


def compatible(kind: str, pert: str) -> bool:
    return kind == "improvement" or pert not in ("conflict_score", "conflict_seed_mean")


def make_row(s: Scenario, kind: str, pert: str, fams: dict[str, str], rng: random.Random,
             shared: dict[str, Any] | None = None, claim_idx: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render one row. ``shared`` (from a previous row of the same scenario) reuses its policy/evidence."""
    if shared is None:
        ev, facts = apply_perturbation(s, pert, rng)
        run, prev_run = _run_name(rng), _run_name(rng)
        while prev_run == run:
            prev_run = _run_name(rng)
        show_config = facts["config_visible_allowed"] and rng.random() < 0.55
        items, diff, groups = _items(s, ev, run, prev_run, show_config, rng)
        policy, pv = _policy(fams["policy"], s.policy_ctx, rng)
        evidence, kstyle = _evidence(fams["evidence"], items, diff, show_config, groups, rng)
        shared = {"ev": ev, "facts": facts, "run": run, "policy": policy, "evidence": evidence,
                  "show_config": show_config, "templates": {"policy_family": fams["policy"], "policy_variant": pv,
                                                            "evidence_family": fams["evidence"], "key_style": kstyle,
                                                            "evidence_order": groups["_order"]}}
    facts = shared["facts"]
    claim, ci = _claim(fams["claim"], kind, s, shared["run"], rng, claim_idx)
    label = sapling_label(kind, facts["sep"], facts["cmp"], facts["gain"], facts["conflict"])
    ev = shared["ev"]
    row = {
        "policy": shared["policy"],
        "evidence": shared["evidence"],
        "claim": claim,
        "label": label,
        "label_alt": label_alternatives(kind, s.category, pert),
        "claim_kind": kind,
        "fake_improvement": s.family != "genuine",
        "target": s.target,
        "target_seen_in_train": s.seen_in_train,
        "source": s.source,
        "category": s.category,
        "family": s.family,
        "perturbation": pert,
        "mechanism": s.mechanism,
        "evidence_profile": s.profile,
        "aggregation": s.agg,
        "variant": s.variant or None,
        "config_shown": shared["show_config"],
        "templates": {**shared["templates"], "claim_family": fams["claim"], "claim_template": ci},
        "facts": {"separation": facts["sep"], "comparability": facts["cmp"], "gain": facts["gain"],
                  "conflict": facts["conflict"]},
        "evidence_keys": sorted(ev),
        "hidden": {**s.hidden, "prev_score": s.prev, "new_score": s.new, "prev_best_seed_std": s.std,
                   "seed_scores_shown": ev.get("seed_scores"), "score_in_eval_log": ev.get("score_in_eval_log"),
                   "visible_gain_ratio": round(visible_gain_ratio(s.prev, s.new, s.std, s.evidence["seed_scores"],
                                                                  s.agg, s.sign), 3)},
    }
    return row, shared


# --------------------------------------------------------------------------- sampling plan

# (weight, base, perturbation). Bases: genuine / contamination / a named category /
# clean_any (genuine or noise/eval_changed) / any / genuine_mean / leak_or_genuine.
RECIPES: dict[str, list[tuple[float, str, str]]] = {
    "improvement": [
        (32, "genuine", "none"), (22, "contamination", "none"), (3, "contamination", "missing_seeds"),
        (3, "contamination", "missing_eval_info"), (6, "noise_within_spread", "none"), (5, "lucky_seed", "none"),
        (6, "eval_changed", "none"), (4, "any", "missing_separation"), (3, "any", "missing_separation_marked"),
        (3, "clean_any", "missing_seeds"), (2, "genuine", "missing_eval_info"), (4, "clean_any", "conflict_score"),
        (4, "genuine_mean", "conflict_seed_mean"), (4, "leak_or_genuine", "conflict_split"),
    ],
    "claim_about_data": [
        (34, "clean_any", "none"), (34, "contamination", "none"), (4, "contamination", "missing_seeds"),
        (4, "clean_any", "missing_eval_info"), (8, "any", "missing_separation"),
        (6, "any", "missing_separation_marked"), (10, "leak_or_genuine", "conflict_split"),
    ],
}
KIND_W = (("improvement", 0.72), ("heldout", 0.14), ("leak", 0.14))
SIBLING_P = 0.3


def _wchoice(rng: random.Random, pairs: list[tuple[Any, float]]) -> Any:
    tot = sum(w for _, w in pairs)
    x = rng.random() * tot
    for v, w in pairs:
        x -= w
        if x < 0:
            return v
    return pairs[-1][0]


def _category(rng: random.Random, base: str) -> str:
    gen = list(zip(GENUINE_CATS, GENUINE_W, strict=True))
    if base in ("genuine", "genuine_mean"):
        return _wchoice(rng, gen)
    if base == "contamination":
        return "contamination"
    if base == "clean_any":
        return _wchoice(rng, [(c, w * 0.6) for c, w in gen] + [("noise_within_spread", 0.15), ("lucky_seed", 0.1),
                                                              ("eval_changed", 0.15)])
    if base == "any":
        return _wchoice(rng, [("contamination", 0.4)] + [(c, w * 0.4) for c, w in gen]
                        + [("noise_within_spread", 0.08), ("lucky_seed", 0.05), ("eval_changed", 0.07)])
    if base == "leak_or_genuine":
        return _wchoice(rng, [("contamination", 0.5)] + [(c, w * 0.5) for c, w in gen])
    return base


def _targets(split: str) -> list[tuple[str, float]]:
    seen = [k for k, g in GENERIC.items() if g.seen]
    unseen = [k for k, g in GENERIC.items() if not g.seen]
    if split == "train":
        return [(t, 0.5 / 3) for t in CORE_TARGETS] + [(t, 0.5 / len(seen)) for t in seen]
    return ([(t, 0.4 / 3) for t in CORE_TARGETS] + [(t, 0.4 / len(seen)) for t in seen]
            + [(t, 0.2 / len(unseen)) for t in unseen])


def _scenario(target: str, cat_or_family: str, base: str, pert: str, rng: random.Random, idx: int) -> Scenario:
    profile = "full" if pert == "conflict_split" else rng.choice(["full", "lean"])
    for attempt in range(60):
        if cat_or_family == "contamination":
            category = mts.TRAP_CATEGORY[target] if target in CORE_TARGETS else GENERIC[target].mechanism
            opts = CORE_OPTIONS["contamination"]
        else:
            category = cat_or_family
            opts = CORE_OPTIONS[category]
        agg = "mean" if base == "genuine_mean" else rng.choice(opts["agg"])
        if pert == "conflict_seed_mean":
            agg = "mean"
        if category == "clean_data_adjacent_change":
            variants = opts["variant"][target] if target in CORE_TARGETS else tuple(GENERIC[target].adjacent)
        else:
            variants = opts["variant"]
            if target not in CORE_TARGETS:
                variants = tuple(v for v in variants if v != "untrained") or ("",)
        variant = rng.choice(variants)
        if target in CORE_TARGETS:
            s = _build_core(target, category, profile, agg, variant, idx + 7919 * attempt)
        else:
            s = _build_generic(GENERIC[target], category, profile, agg, variant,
                               random.Random(f"{SEED}:generic:{idx}:{attempt}"))
        if scenario_ok(s):
            return s
    raise RuntimeError(f"no valid scenario for {target}/{cat_or_family} after 60 attempts")  # pragma: no cover


def build_split(split: str, n_rows: int) -> list[dict[str, Any]]:
    rng = random.Random(f"{SEED}:{split}")
    fams = FAMILIES
    idx_base = {"train": 100_000, "test": 200_000}[split]
    rows: list[dict[str, Any]] = []
    g = 0
    while len(rows) < n_rows:
        g += 1
        group = f"{split}-{g:05d}"
        target = _wchoice(rng, _targets(split))
        kind = _wchoice(rng, list(KIND_W))
        recipe = RECIPES["improvement" if kind == "improvement" else "claim_about_data"]
        _, base, pert = _wchoice(rng, [((w, b, p), w) for w, b, p in recipe])
        cat = _category(rng, base)
        s = _scenario(target, cat, base, pert, rng, idx_base + g)
        f = {k: rng.choice(fams[k][split]) for k in fams}
        row, shared = make_row(s, kind, pert, f, rng)
        batch = [row]
        if rng.random() < SIBLING_P and len(rows) + 2 <= n_rows:
            others = [k for k in CLAIM_KINDS if k != kind and compatible(k, pert)]
            if others:
                k2 = rng.choice(others)
                f2 = {**f, "claim": rng.choice(fams["claim"][split])}
                row2, _ = make_row(s, k2, pert, f2, rng, shared=shared)
                batch.append(row2)
        fold = None
        if split == "train":
            x = random.Random(f"{SEED}:fold:{group}").random()
            fold = next(name for name, hi in FOLDS if x < hi)
        for r in batch:
            rows.append({"id": f"mlc_v1_{split}_{len(rows) + 1:05d}", "split": split, "fold": fold, "group": group,
                         "set_version": SET_VERSION, **r})
    return rows


def build_trap50() -> list[dict[str, Any]]:
    """trap_set_v1 (the judge_improvement eval set) as Sapling triples, canonical train templates."""
    rows = []
    fams = {"policy": "numbered", "evidence": "kv", "claim": "plain"}
    for case in mts.build_cases():
        s = _core_scenario_from_case(case)
        rng = random.Random(f"{SEED}:trap50:{case['id']}")
        row, _ = make_row(s, "improvement", "none", fams, rng, claim_idx=0)
        rows.append({"id": f"mlc_v1_trap50_{case['id'][-3:]}", "split": "trap50", "fold": None,
                     "group": case["id"], "set_version": SET_VERSION, "trap_id": case["id"],
                     "trap_split": case["split"], **row})
    return rows


def build_all() -> dict[str, list[dict[str, Any]]]:
    return {"train": build_split("train", N_TRAIN), "test": build_split("test", N_TEST), "trap50": build_trap50()}


def render_jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows)


def summary(data: dict[str, list[dict[str, Any]]]) -> str:
    lines = []
    for split, rows in data.items():
        lines.append(f"{split}: {len(rows)} rows, {len({r['group'] for r in rows})} groups")
        for key in ("label", "claim_kind", "perturbation", "family", "target"):
            cnt = Counter(r[key] for r in rows)
            lines.append(f"  {key}: " + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
        lens = sorted(len(r["policy"]) + len(r["evidence"]) + len(r["claim"]) for r in rows)
        lines.append(f"  chars per row: median {lens[len(lens) // 2]}, max {lens[-1]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--check", action="store_true", help="exit 1 if any file differs from a fresh build")
    args = ap.parse_args(argv)
    data = build_all()
    stale = []
    for split, rows in data.items():
        path = args.out_dir / FILES[split]
        text = render_jsonl(rows)
        if args.check:
            cur = path.read_bytes().decode("utf-8") if path.exists() else ""
            if cur != text:
                stale.append(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
            print(f"wrote {path}")
    if args.check:
        if stale:
            print("stale (rerun scripts/make_ml_claims.py): " + ", ".join(stale), file=sys.stderr)
            return 1
        print("data/ml_claims is up to date")
        return 0
    print(summary(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
