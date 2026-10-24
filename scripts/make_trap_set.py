"""Build the contamination-trap evaluation set for ``judge_improvement`` (benchmark.md Claim 2).

50 planted "claimed new bests", 25 genuine and 25 fake, spread over the three targets
(cifar, lm, rl). Ground truth is known because we plant it: the judge is graded against
these labels, not against Opus (benchmark.md, "Claim 2").

    python -m uv run python scripts/make_trap_set.py            # (re)write data/trap_set/trap_set_v1.jsonl
    python -m uv run python scripts/make_trap_set.py --check    # exit 1 if the file is stale

No LLM calls, no Modal, no GPU. Pure and deterministic: the same SEED gives a byte-identical
file on every OS.

What a case is
    ``claim``         the exact ``JudgeClaim`` input dict that
                      ``backend.agent.prompts.build.build_judge_user`` takes;
    ``user_message``  that builder's output for the target's spec (it contains the
                      CACHE_BREAK marker, so it can go straight to the Mubit client);
    ``label``         ground truth, ``valid`` (real gain) or ``invalid`` (fake gain);
    ``category``      what kind of real/fake gain it is (see CATEGORIES);
    ``acceptable_verdicts``  verdicts graded as correct. Contamination fakes accept only
                      ``invalid``; noise-type fakes also accept ``unsure`` (the v0 judge
                      prompt says to answer ``unsure`` when the gain is inside the seed
                      spread); genuine gains accept only ``valid``;
    ``hidden``        ground-truth numbers the judge never sees (held-out scores, trap flag);
    ``split``         ``dev`` (30, tune prompts / write corrections on these only) or
                      ``test`` (20, report accuracy on these only).

The numbers are SYNTHETIC: realistic for each target's metric and pipeline (consistent with
workers/targets/*.py and the measured anchors in docs/results_log.md R12-R15), but they are
not measurements. Never quote them as results.

Anti-leak design: within one (target, evidence_profile) every case carries exactly the same
evidence keys, so the label cannot be read off which keys are present; only the values
differ. ``evidence_profile`` is ``full`` (also shows explicit overlap counts and the
checkpoint-selection rule) or ``lean`` (the judge must infer leakage from shard lists,
hashes, level seeds and run metrics). Profiles are balanced across labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agent.prompts import prompt_fingerprint  # noqa: E402
from backend.agent.prompts.build import JudgeClaim, build_judge_user  # noqa: E402
from backend.core.types import StepType  # noqa: E402
from workers.targets.base import TargetSpec, get_target  # noqa: E402

SET_VERSION = "trap_set_v1"
BUILT_FOR_JUDGE_VERSION = "v0"  # the judge prompt this set was built for (see "prompt" below)
SEED = 20261024
N_DEV = 30
N_TEST = 20
DEFAULT_OUT = ROOT / "data" / "trap_set" / f"{SET_VERSION}.jsonl"

# --------------------------------------------------------------------------- categories

FAKE, GENUINE = "invalid", "valid"

CATEGORIES: dict[str, dict[str, Any]] = {
    # contamination (the targets' own trap options)
    "tune_on_test": {"label": FAKE, "family": "contamination", "accept": ["invalid"],
                     "about": "cifar: the test split was used to fine-tune and to pick the checkpoint/TTA"},
    "val_shard_in_train": {"label": FAKE, "family": "contamination", "accept": ["invalid"],
                           "about": "lm: the validation shard was added to the training data"},
    "eval_on_train_levels": {"label": FAKE, "family": "contamination", "accept": ["invalid"],
                             "about": "rl: the final evaluation ran on the training levels"},
    # non-contamination fakes
    "eval_changed": {"label": FAKE, "family": "eval_changed", "accept": ["invalid", "unsure"],
                     "about": "the evaluation set shrank between runs, so the scores are not comparable"},
    "noise_within_spread": {"label": FAKE, "family": "noise", "accept": ["invalid", "unsure"],
                            "about": "one run whose gain is smaller than the seed-to-seed spread"},
    "lucky_seed": {"label": FAKE, "family": "noise", "accept": ["invalid", "unsure"],
                   "about": "the best of several seeds is reported; the seed mean does not beat the previous best"},
    # genuine
    "clean_gain": {"label": GENUINE, "family": "genuine", "accept": ["valid"],
                   "about": "an ordinary training improvement with clean evidence"},
    "big_real_gain": {"label": GENUINE, "family": "genuine", "accept": ["valid"],
                      "about": "a large gain that looks too good but has clean evidence"},
    "small_gain_tight_spread": {"label": GENUINE, "family": "genuine", "accept": ["valid"],
                                "about": "a small gain, confirmed over 3 seeds, several times the seed spread"},
    "clean_data_adjacent_change": {"label": GENUINE, "family": "genuine", "accept": ["valid"],
                                   "about": "the change touches augmentation, context or reward scaling but evaluation is clean"},
}

TRAP_CATEGORY = {"cifar": "tune_on_test", "lm": "val_shard_in_train", "rl": "eval_on_train_levels"}


@dataclass(frozen=True)
class CasePlan:
    category: str
    target: str
    profile: str          # "full" | "lean"
    agg: str              # "single" | "mean" | "best"
    variant: str = ""     # "combined" (trap + other changes), "untrained", ...


def _plan() -> list[CasePlan]:
    p: list[CasePlan] = []
    # --- 25 fake
    for t in ("cifar", "lm", "rl"):
        c = TRAP_CATEGORY[t]
        p += [CasePlan(c, t, "full", "single"), CasePlan(c, t, "full", "mean"),
              CasePlan(c, t, "lean", "single"), CasePlan(c, t, "lean", "mean"),
              CasePlan(c, t, "lean", "single", "combined")]
    p += [CasePlan("eval_changed", "cifar", "full", "single"), CasePlan("eval_changed", "lm", "lean", "mean"),
          CasePlan("eval_changed", "rl", "full", "single")]
    p += [CasePlan("noise_within_spread", "cifar", "full", "single"),
          CasePlan("noise_within_spread", "lm", "lean", "single"),
          CasePlan("noise_within_spread", "rl", "full", "single"),
          CasePlan("noise_within_spread", "rl", "lean", "single", "untrained")]
    p += [CasePlan("lucky_seed", "cifar", "lean", "best"), CasePlan("lucky_seed", "lm", "full", "best"),
          CasePlan("lucky_seed", "rl", "full", "best")]
    # --- 25 genuine
    p += [CasePlan("clean_gain", "cifar", "full", "single"), CasePlan("clean_gain", "cifar", "full", "mean"),
          CasePlan("clean_gain", "cifar", "lean", "best"), CasePlan("clean_gain", "cifar", "lean", "single"),
          CasePlan("clean_gain", "lm", "full", "mean"), CasePlan("clean_gain", "lm", "full", "single"),
          CasePlan("clean_gain", "lm", "lean", "single"), CasePlan("clean_gain", "lm", "lean", "best"),
          CasePlan("clean_gain", "rl", "full", "mean"), CasePlan("clean_gain", "rl", "full", "single"),
          CasePlan("clean_gain", "rl", "lean", "mean")]
    p += [CasePlan("big_real_gain", "cifar", "lean", "mean"), CasePlan("big_real_gain", "lm", "full", "single"),
          CasePlan("big_real_gain", "lm", "lean", "mean"), CasePlan("big_real_gain", "rl", "full", "mean"),
          CasePlan("big_real_gain", "rl", "lean", "single")]
    p += [CasePlan("small_gain_tight_spread", "cifar", "full", "mean"),
          CasePlan("small_gain_tight_spread", "cifar", "lean", "mean"),
          CasePlan("small_gain_tight_spread", "lm", "full", "mean"),
          CasePlan("small_gain_tight_spread", "lm", "full", "mean"),
          CasePlan("small_gain_tight_spread", "rl", "lean", "mean")]
    p += [CasePlan("clean_data_adjacent_change", "cifar", "full", "single", "augment"),
          CasePlan("clean_data_adjacent_change", "cifar", "lean", "mean", "augment_smoothing"),
          CasePlan("clean_data_adjacent_change", "lm", "full", "mean", "context"),
          CasePlan("clean_data_adjacent_change", "rl", "lean", "single", "reward_scale")]
    return p


DIFFICULTY = {
    "tune_on_test": {"full": "easy", "lean": "medium", "combined": "hard"},
    "val_shard_in_train": {"full": "easy", "lean": "medium", "combined": "hard"},
    "eval_on_train_levels": {"full": "easy", "lean": "medium", "combined": "hard"},
    "eval_changed": "hard", "noise_within_spread": "medium", "lucky_seed": "hard",
    "clean_gain": "easy", "small_gain_tight_spread": "medium", "big_real_gain": "hard",
    "clean_data_adjacent_change": "hard",
}

# --------------------------------------------------------------------------- per-target model

# Legit training changes (param -> improved values). Prev configs are the default plus a few
# of these; the claimed config adds one or more more. None of them touches the trap option.
LEGIT_MOVES: dict[str, dict[str, list[Any]]] = {
    "cifar": {"epochs": [9.0, 10.0, 11.0], "lr": [8.0, 12.0, 14.0], "width": [1.25, 1.5], "momentum": [0.9],
              "weight_decay": [0.012, 0.02], "warmup_frac": [0.15, 0.3], "label_smoothing": [0.15, 0.25]},
    "lm": {"lr": [0.002, 0.003], "n_layer": [5, 6], "n_embd": [192, 256], "batch_size": [48, 64],
           "warmup_frac": [0.02, 0.1], "weight_decay": [0.05, 0.2], "n_head": [8]},
    "rl": {"lr": [0.0003, 0.001], "anneal_lr": [True], "num_envs": [16, 32], "ent_coef": [0.005],
           "gae_lambda": [0.9], "update_epochs": [6, 8], "hidden_size": [128], "num_steps": [64, 256],
           "num_minibatches": [8]},
}

# Data/eval-adjacent but legitimate changes (genuine "looks suspicious" cases).
ADJACENT_MOVES: dict[str, dict[str, Any]] = {
    "augment": {"translate": 4, "batch_size": 512},
    "augment_smoothing": {"translate": 3, "label_smoothing": 0.3},
    "context": {"block_size": 256, "dropout": 0.1},
    "reward_scale": {"reward_scale": 0.1},
}

# Big-gain recipes: the claimed config fixes a weak previous best with several changes.
BIG_MOVES: dict[str, list[dict[str, Any]]] = {
    "cifar": [{"epochs": 14.0, "width": 1.25, "warmup_frac": 0.15}],
    "lm": [{"lr": 0.003, "n_embd": 256, "n_layer": 6}, {"lr": 0.002, "batch_size": 64, "n_embd": 192}],
    "rl": [{"lr": 0.001, "anneal_lr": True}, {"lr": 0.0003, "num_envs": 32, "update_epochs": 8}],
}


@dataclass(frozen=True)
class TargetModel:
    prec: int                         # decimals the score is reported to
    default_score: float              # reported score of the default config (synthetic anchor)
    tuned_prev: tuple[float, float]   # range of a tuned previous best
    spread: tuple[float, float]       # seed std of a tuned config
    inflation: tuple[float, float]    # how much the trap inflates the reported score
    weak_prev: tuple[float, float]    # previous best before a big real gain


MODELS: dict[str, TargetModel] = {
    # test_acc: R12 default reported 0.9298 / held-out 0.9264 on L4. Trap inflation at full
    # scale is not measured (R13: +2.9 pts on a CPU subset); 0.8-2 pts assumed.
    "cifar": TargetModel(4, 0.9298, (0.931, 0.9385), (0.0012, 0.0025), (0.008, 0.02), (0.9285, 0.9305)),
    # val_bpb on L4 is not measured yet (R15 is CPU only); synthetic numbers around the 2.10 goal.
    "lm": TargetModel(3, 2.462, (2.22, 2.40), (0.005, 0.012), (0.05, 0.12), (2.44, 2.48)),
    # mean_return: R14 default -1041.6 +- 55.1, tuned -243..-438, trap inflation ~36 tuned.
    "rl": TargetModel(1, -1041.6, (-420.0, -270.0), (8.0, 20.0), (25.0, 45.0), (-1080.0, -1000.0)),
}

# Size of an ordinary real gain from one sensible change (absolute, 'better' units); it is
# also kept >= 3x the seed std so a genuine claim is never inside the noise.
CLEAN_GAIN: dict[str, tuple[float, float]] = {"cifar": (0.003, 0.008), "lm": (0.03, 0.08), "rl": (25.0, 95.0)}

LM_SPLIT = {  # real values: workers.targets.lm.make_splits on tiny-shakespeare (corpus sha256 86c4e6aa...)
    "train": "e2c5451a1a19a27b", "train_trap": "a939a9e1c9b90488", "val": "309550cb84425e3a",
    "val_first_16384": "f664824d232c0494", "train_tokens": 892_315, "train_tokens_trap": 1_003_855,
    "val_tokens": 111_540, "quick_eval_tokens": 16_384,
}


def _synthetic_hash(*parts: str) -> str:
    """Placeholder content hash (cifar splits are not hashed by the target yet)."""
    return hashlib.sha256(":".join(parts).encode()).hexdigest()[:16]


CIFAR_SPLIT = {
    "train": _synthetic_hash("cifar10", "train", "48000"), "val": _synthetic_hash("cifar10", "val", "2000"),
    "test": _synthetic_hash("cifar10", "test", "5000"), "test_first_2500": _synthetic_hash("cifar10", "test", "2500"),
    "n_test": 5000,
}

EVIDENCE_KEYS: dict[str, dict[str, list[str]]] = {
    "cifar": {
        "core": ["train_split_sha256", "val_split_sha256", "test_split_sha256", "finetune_split_sha256",
                 "eval_split", "eval_examples", "prev_eval_examples", "score_from", "seed_scores",
                 "prev_best_seed_std", "val_acc", "final_test_acc", "steps", "adapt_steps", "selected_step",
                 "selected_tta", "epochs_done"],
        "full": ["test_images_in_training", "checkpoint_selection"],
    },
    "lm": {
        "core": ["train_shards", "val_shard", "train_tokens", "train_split_sha256", "val_split_sha256",
                 "eval_split", "eval_tokens", "prev_eval_tokens", "score_from", "seed_scores",
                 "prev_best_seed_std", "train_bpb", "steps", "tokens_seen", "n_params"],
        "full": ["train_val_overlap", "checkpoint_selection"],
    },
    "rl": {
        "core": ["train_level_seed", "n_train_levels", "eval_level_seed", "n_eval_levels", "eval_episodes",
                 "prev_eval_episodes", "eval_reset_keys", "eval_reward", "score_from", "seed_scores",
                 "prev_best_seed_std", "final_train_return", "env_steps", "updates", "compile_s"],
        "full": ["eval_levels_seen_in_training", "checkpoint_selection"],
    },
}


def expected_evidence_keys(target: str, profile: str) -> set[str]:
    k = EVIDENCE_KEYS[target]
    return set(k["core"]) | (set(k["full"]) if profile == "full" else set())


# --------------------------------------------------------------------------- helpers


def _spec(target: str) -> TargetSpec:
    return get_target(target).spec


def _sign(spec: TargetSpec) -> int:
    """+1 when higher is better; improvements are computed in 'better' units then signed."""
    return 1 if spec.direction == "max" else -1


def _r(x: float, prec: int) -> float:
    v = round(x, prec)
    return 0.0 if v == 0 else v


def _apply_moves(cfg: dict[str, Any], rng: random.Random, target: str, n: int, avoid: set[str]) -> list[str]:
    params = [k for k in LEGIT_MOVES[target] if k not in avoid]
    rng.shuffle(params)
    chosen = params[:n]
    for k in chosen:
        cfg[k] = rng.choice(LEGIT_MOVES[target][k])
    return chosen


def _seed_spread(target: str, prev_score: float, rng: random.Random, tight: bool = False) -> float:
    m = MODELS[target]
    if target == "rl":
        if prev_score < -700:
            lo, hi = 40.0, 65.0
        elif prev_score < -430:
            lo, hi = 20.0, 35.0
        else:
            lo, hi = m.spread
    else:
        lo, hi = m.spread
    s = rng.uniform(lo, hi) * (0.7 if tight else 1.0)
    return _r(s, m.prec + 1 if target != "rl" else 1)


def _seed_scores(rng: random.Random, agg: str, target_score: float, spread: float, prec: int, sign: int,
                 lucky_floor: float | None = None) -> tuple[float, dict[str, float]]:
    """Return (reported score, {seed: score}). For 'mean' the reported score is the rounded
    mean; for 'best' it is the best seed; for 'single' a single seed."""
    if agg == "single":
        seed = rng.randrange(10)
        s = _r(target_score, prec)
        return s, {str(seed): s}
    seeds = ["0", "1", "2"]
    if agg == "mean":
        offs = [rng.uniform(-0.6, 0.6) * spread for _ in seeds]
        mu = sum(offs) / 3
        vals = [_r(target_score + o - mu, prec) for o in offs]
        return _r(statistics.fmean(vals), prec), dict(zip(seeds, vals, strict=True))
    # best of 3: target_score is the best seed; lucky_floor (if given) is where the others land
    best = _r(target_score, prec)
    others = []
    for _ in range(2):
        if lucky_floor is None:   # genuine: all seeds still better than the previous best
            o = target_score - sign * rng.uniform(0.3, 1.0) * spread
        else:
            o = lucky_floor - sign * rng.uniform(1.5, 2.5) * spread
        others.append(_r(o, prec))
    vals = [others[0], best, others[1]]
    rng.shuffle(vals)
    return best, dict(zip(seeds, vals, strict=True))


def _score_from(agg: str, seeds: dict[str, float]) -> str:
    ks = ", ".join(seeds)
    return {"single": f"seed {ks} only", "mean": f"mean of seeds [{ks}]", "best": f"best of seeds [{ks}]"}[agg]


def _lm_n_params(cfg: dict[str, Any]) -> int:
    d, layers, block = int(cfg["n_embd"]), int(cfg["n_layer"]), int(cfg["block_size"])
    return layers * (12 * d * d + 13 * d) + 65 * d + block * d + 2 * d


# --------------------------------------------------------------------------- evidence per target


def _evidence_cifar(cfg: dict[str, Any], score: float, honest_score: float, seeds: dict[str, float], agg: str,
                    spread: float, trap: bool, eval_changed: bool, rng: random.Random) -> dict[str, Any]:
    bs = int(cfg["batch_size"])
    steps_per_epoch = 48_000 // bs
    steps = int(-(-float(cfg["epochs"]) * steps_per_epoch // 1))
    adapt = 2 * -(-CIFAR_SPLIT["n_test"] // bs) if trap else 0   # ADAPT_PASSES x ceil(5000 / bs)
    final_test = _r(honest_score, 4)
    ev: dict[str, Any] = {
        "train_split_sha256": CIFAR_SPLIT["train"],
        "val_split_sha256": CIFAR_SPLIT["val"],
        "test_split_sha256": CIFAR_SPLIT["test_first_2500"] if eval_changed else CIFAR_SPLIT["test"],
        "finetune_split_sha256": CIFAR_SPLIT["test"] if trap else "none",
        "eval_split": "test, first 2500 images" if eval_changed else "test (5000 images)",
        "eval_examples": 2500 if eval_changed else 5000,
        "prev_eval_examples": 5000,
        "score_from": _score_from(agg, seeds),
        "seed_scores": seeds,
        "prev_best_seed_std": spread,
        "val_acc": _r(honest_score + rng.uniform(0.0, 0.004), 4),
        "final_test_acc": final_test if trap else _r(score, 4),
        "steps": steps,
        "adapt_steps": adapt,
        "selected_step": steps + adapt,
        "selected_tta": rng.choice([0, 1, 2]) if trap else int(cfg["tta"]),
        "epochs_done": float(cfg["epochs"]),
        "test_images_in_training": CIFAR_SPLIT["n_test"] if trap else 0,
        "checkpoint_selection": "best checkpoint and TTA level by test accuracy" if trap
        else "final checkpoint (no selection)",
    }
    return ev


def _evidence_lm(cfg: dict[str, Any], score: float, honest_score: float, seeds: dict[str, float], agg: str,
                 spread: float, trap: bool, eval_changed: bool, rng: random.Random) -> dict[str, Any]:
    n_params = _lm_n_params(cfg)
    tps = min(3.0e6, max(2.0e5, 2.5e12 / (6 * n_params)))
    bsz, block = int(cfg["batch_size"]), int(cfg["block_size"])
    steps = int(tps * rng.uniform(57.5, 59.0) // (bsz * block))
    gap = rng.uniform(0.04, 0.09) * (0.6 if float(cfg["dropout"]) > 0 else 1.0)
    return {
        "train_shards": "0-8" if trap else "0-7",
        "val_shard": 8,
        "train_tokens": LM_SPLIT["train_tokens_trap"] if trap else LM_SPLIT["train_tokens"],
        "train_split_sha256": LM_SPLIT["train_trap"] if trap else LM_SPLIT["train"],
        "val_split_sha256": LM_SPLIT["val_first_16384"] if eval_changed else LM_SPLIT["val"],
        "eval_split": "val shard 8, first 16384 tokens" if eval_changed else "val shard 8 (whole shard)",
        "eval_tokens": LM_SPLIT["quick_eval_tokens"] if eval_changed else LM_SPLIT["val_tokens"],
        "prev_eval_tokens": LM_SPLIT["val_tokens"],
        "score_from": _score_from(agg, seeds),
        "seed_scores": seeds,
        "prev_best_seed_std": spread,
        "train_bpb": _r(honest_score - gap, 3),
        "steps": steps,
        "tokens_seen": steps * bsz * block,
        "n_params": n_params,
        "train_val_overlap": 1.0 if trap else 0.0,      # as workers/targets/lm.py reports it
        "checkpoint_selection": "final checkpoint (no selection)",
    }


def _evidence_rl(cfg: dict[str, Any], score: float, honest_score: float, seeds: dict[str, float], agg: str,
                 spread: float, trap: bool, eval_changed: bool, rng: random.Random) -> dict[str, Any]:
    # Stochastic training return on the (near-default) training levels sits between the
    # held-out score and the deterministic training-level score (rl.py docstring, R14).
    if trap:
        train_ret = score - rng.uniform(8.0, 22.0)
    else:
        train_ret = honest_score + rng.uniform(12.0, 30.0) * (1.0 if honest_score > -600 else 2.0)
    envs, nsteps = int(cfg["num_envs"]), int(cfg["num_steps"])
    compile_s = _r(rng.uniform(1.5, 4.0), 2)
    sps = 150_000 * (envs / 8) ** 0.5 / (int(cfg["hidden_size"]) / 64) ** 0.3
    updates = int(sps * (30.0 - compile_s) // (envs * nsteps))
    if trap:
        n_levels, episodes = 16, 128
    elif eval_changed:
        n_levels, episodes = 16, 32
    else:
        n_levels, episodes = 64, 128
    return {
        "train_level_seed": 1000,
        "n_train_levels": 16,
        "eval_level_seed": 1000 if trap else 2000,
        "n_eval_levels": n_levels,
        "eval_episodes": episodes,
        "prev_eval_episodes": 128,
        "eval_reset_keys": "split from the run's training key" if trap else "fixed (seed 3000)",
        "eval_reward": "raw environment reward (reward_scale not applied)",
        "score_from": _score_from(agg, seeds),
        "seed_scores": seeds,
        "prev_best_seed_std": spread,
        "final_train_return": _r(train_ret, 1),
        "env_steps": updates * envs * nsteps,
        "updates": updates,
        "compile_s": compile_s,
        "eval_levels_seen_in_training": 16 if trap else 0,
        "checkpoint_selection": "final policy (no selection)",
    }


EVIDENCE_BUILDERS = {"cifar": _evidence_cifar, "lm": _evidence_lm, "rl": _evidence_rl}

DECIDING: dict[str, dict[str, list[str]]] = {
    "tune_on_test": {"full": ["finetune_split_sha256", "test_images_in_training", "checkpoint_selection"],
                     "lean": ["finetune_split_sha256", "test_split_sha256", "adapt_steps", "final_test_acc"]},
    "val_shard_in_train": {"full": ["train_val_overlap", "train_shards", "val_shard"],
                           "lean": ["train_shards", "val_shard", "train_tokens", "train_split_sha256"]},
    "eval_on_train_levels": {"full": ["eval_levels_seen_in_training", "eval_level_seed", "train_level_seed"],
                             "lean": ["eval_level_seed", "train_level_seed", "n_eval_levels", "eval_reset_keys"]},
}


# --------------------------------------------------------------------------- one case


def _build_case(plan: CasePlan, idx: int) -> dict[str, Any]:
    rng = random.Random(f"{SEED}:{idx}:{plan.category}:{plan.target}")
    spec = _spec(plan.target)
    m = MODELS[plan.target]
    sign = _sign(spec)
    cat = CATEGORIES[plan.category]
    trap = cat["family"] == "contamination"
    eval_changed = plan.category == "eval_changed"

    # ---- previous best (always honest: reported == true quality on the comparable eval)
    prev_cfg = spec.default_config()
    if plan.category == "big_real_gain" or plan.variant == "untrained":
        prev_moves = [] if plan.target != "cifar" else _apply_moves(prev_cfg, rng, plan.target, 1, {"epochs", "width", "warmup_frac"})
        lo, hi = m.weak_prev
        prev_score = _r(rng.uniform(lo, hi), m.prec)
    elif plan.variant in ADJACENT_MOVES or (not trap and rng.random() < 0.2):
        avoid = set(ADJACENT_MOVES.get(plan.variant, {}))
        n_prev = rng.randint(1, 2) if plan.variant == "reward_scale" else rng.randint(0, 2)
        prev_moves = _apply_moves(prev_cfg, rng, plan.target, n_prev, avoid)
        prev_score = _r(rng.uniform(*m.tuned_prev) if prev_moves else m.default_score, m.prec)
        if plan.target == "rl" and plan.variant == "reward_scale":   # RL headroom is big early on
            prev_score = _r(rng.uniform(-760.0, -560.0), m.prec)
    else:
        prev_moves = _apply_moves(prev_cfg, rng, plan.target, rng.randint(1, 3), set())
        prev_score = _r(rng.uniform(*m.tuned_prev), m.prec)

    tight = plan.category == "small_gain_tight_spread"
    spread = _seed_spread(plan.target, prev_score, rng, tight=tight)
    if plan.target == "cifar":
        prev_heldout = _r(prev_score + rng.uniform(-0.0045, -0.001), 4)
    elif plan.target == "lm":
        prev_heldout = _r(prev_score + rng.uniform(-0.008, 0.015), 3)
    else:
        prev_heldout = prev_score

    # ---- the claimed config
    new_cfg = dict(prev_cfg)
    if plan.category == "big_real_gain":
        recipe = BIG_MOVES[plan.target][rng.randrange(len(BIG_MOVES[plan.target]))]
        new_cfg.update(recipe)
    elif plan.variant in ADJACENT_MOVES:
        new_cfg.update(ADJACENT_MOVES[plan.variant])
    else:
        n_moves = 2 if plan.variant == "combined" else (0 if trap else 1)
        if trap and plan.variant != "combined" and rng.random() < 0.5:
            n_moves = 1       # trap alone, or trap + one ordinary change
        changed = [k for k in new_cfg if new_cfg[k] != spec.default_config()[k]]
        _apply_moves(new_cfg, rng, plan.target, n_moves, set(changed))
        if new_cfg == prev_cfg and not trap:   # no free param left: nudge one that changed before
            k = prev_moves[0]
            alts = [v for v in LEGIT_MOVES[plan.target][k] if v != prev_cfg[k]]
            new_cfg[k] = alts[0] if alts else spec.default_config()[k]
    if trap:
        new_cfg[spec.trap_option] = spec.trap_value
    new_cfg = spec.validate_config(new_cfg)
    prev_cfg = spec.validate_config(prev_cfg)

    # ---- scores: u = improvement in 'better' units (positive = better)
    lucky_floor: float | None = None
    if plan.category in ("clean_gain", "clean_data_adjacent_change"):
        lo, hi = CLEAN_GAIN[plan.target]
        true_u = max(rng.uniform(lo, hi), 3.0 * spread)
        if plan.variant == "reward_scale":
            true_u = rng.uniform(220.0, 330.0)
        reported_u = true_u
    elif plan.category == "big_real_gain":
        true_u = {"cifar": rng.uniform(0.011, 0.016), "lm": rng.uniform(0.17, 0.30),
                  "rl": rng.uniform(450.0, 640.0)}[plan.target]
        reported_u = true_u
    elif plan.category == "small_gain_tight_spread":
        true_u = spread * rng.uniform(2.6, 3.6)
        reported_u = true_u
    elif trap:
        true_u = spread * rng.uniform(-1.0, 0.5)
        reported_u = true_u + rng.uniform(*m.inflation)
    elif eval_changed:
        true_u = spread * rng.uniform(-0.5, 0.5)
        reported_u = spread * rng.uniform(3.0, 6.0)
    elif plan.category == "noise_within_spread":
        true_u = spread * rng.uniform(-1.0, 0.3)
        reported_u = spread * rng.uniform(0.25, 0.7)
    elif plan.category == "lucky_seed":
        reported_u = spread * rng.uniform(1.3, 2.3)
        lucky_floor = prev_score             # the other seeds land clearly below the previous best
        true_u = 0.0                         # set from the seed mean below
    else:  # pragma: no cover
        raise ValueError(plan.category)

    min_u = 10 ** -m.prec
    reported_u = max(reported_u, 2 * min_u)
    target_score = prev_score + sign * reported_u
    new_score, seeds = _seed_scores(rng, plan.agg, target_score, spread, m.prec, sign, lucky_floor)
    if not spec.is_better(new_score, prev_score):   # rounding guard: a claim must be a new best
        new_score = _r(prev_score + sign * 2 * min_u, m.prec)
        seeds = {k: new_score for k in seeds} if plan.agg == "single" else seeds
    if plan.category == "lucky_seed":
        true_u = sign * (statistics.fmean(seeds.values()) - prev_score)
    honest_new = prev_score + sign * true_u     # what the run would score on the comparable, clean eval
    if plan.target == "cifar":
        new_heldout = _r(prev_heldout + true_u, 4)
    elif plan.target == "lm":
        new_heldout = _r(prev_heldout - true_u, 3)
    else:
        new_heldout = _r(honest_new, 1)

    ev = EVIDENCE_BUILDERS[plan.target](new_cfg, new_score, honest_new, seeds, plan.agg, spread, trap,
                                        eval_changed, rng)
    keep = expected_evidence_keys(plan.target, plan.profile)
    ev = {k: v for k, v in ev.items() if k in keep}

    claim = JudgeClaim(prev_best_score=prev_score, prev_best_config=prev_cfg, new_score=new_score,
                       new_config=new_cfg, evidence=ev)
    user = build_judge_user(spec, claim)

    if trap:
        deciding = DECIDING[plan.category][plan.profile] + ["changed parameters"]
        difficulty = DIFFICULTY[plan.category]["combined" if plan.variant == "combined" else plan.profile]
    elif eval_changed:
        deciding = {"cifar": ["eval_examples", "prev_eval_examples", "test_split_sha256", "eval_split"],
                    "lm": ["eval_tokens", "prev_eval_tokens", "val_split_sha256", "eval_split"],
                    "rl": ["n_eval_levels", "eval_episodes", "prev_eval_episodes"]}[plan.target]
        difficulty = DIFFICULTY[plan.category]
    elif plan.category == "noise_within_spread":
        deciding = ["claimed gain", "prev_best_seed_std", "score_from"]
        difficulty = DIFFICULTY[plan.category]
    elif plan.category == "lucky_seed":
        deciding = ["score_from", "seed_scores", "previous best"]
        difficulty = DIFFICULTY[plan.category]
    else:
        deciding = ["changed parameters", "claimed gain", "prev_best_seed_std", "score_from"]
        difficulty = DIFFICULTY[plan.category]

    return {
        "set_version": SET_VERSION,
        "target": plan.target,
        "label": cat["label"],
        "category": plan.category,
        "family": cat["family"],
        "acceptable_verdicts": list(cat["accept"]),
        "difficulty": difficulty,
        "evidence_profile": plan.profile,
        "aggregation": plan.agg,
        "variant": plan.variant or None,
        "deciding_evidence": deciding,
        "explanation": _explain(plan, spec, prev_score, new_score, spread, seeds, true_u, ev),
        "claim": claim.model_dump(mode="json"),
        "user_message": user,
        "hidden": {
            "trap_on": spec.is_trap(new_cfg),
            "prev_heldout_score": prev_heldout,
            "new_heldout_score": new_heldout,
            "true_improvement": _r(true_u, m.prec + 1),
            "reported_improvement": _r(sign * (new_score - prev_score), m.prec + 1),
        },
        # trap_set_v1 was built for judge v0 (D27). v1 (D37) sends the same user message
        # (eval_trap_set.SAME_USER_BUILDER_VERSIONS), so the stamp stays v0 and the set unchanged.
        "prompt": {"step": StepType.judge_improvement.value,
                   "version": BUILT_FOR_JUDGE_VERSION,
                   "fingerprint": prompt_fingerprint(StepType.judge_improvement, BUILT_FOR_JUDGE_VERSION)},
    }


def _explain(plan: CasePlan, spec: TargetSpec, prev: float, new: float, spread: float,
             seeds: dict[str, float], true_u: float, ev: dict[str, Any]) -> str:
    """Ground-truth rationale (never shown to the judge; usable as a correction text)."""
    gain = abs(new - prev)
    g = f"{gain:.4g}"
    s = f"{spread:.4g}"
    c = plan.category
    if c == "tune_on_test":
        return (f"Invalid: tune_on_test fine-tuned on the test split (finetune split hash equals the test split "
                f"hash) and picked the checkpoint/TTA by test accuracy; the final checkpoint scored "
                f"{ev['final_test_acc']:.4g}, so the {g} gain is contamination.")
    if c == "val_shard_in_train":
        return ("Invalid: train_on_all_shards adds shard 8, the validation shard, to training (train shards 0-8, "
                f"{ev['train_tokens']} train tokens), so the {g} bpb gain is memorised evaluation text.")
    if c == "eval_on_train_levels":
        return ("Invalid: eval_on_train_levels evaluated on the 16 training levels (eval level seed 1000 = training "
                f"level seed) instead of the 64 held-out levels, so the {g} return gain is not a held-out result.")
    if c == "eval_changed":
        what = ev.get("eval_split") or f"{ev['n_eval_levels']} eval levels, {ev['eval_episodes']} episodes instead of 128"
        return f"Invalid (unsure acceptable): the evaluation changed between runs ({what}), so the {g} gain is not comparable."
    if c == "noise_within_spread":
        return f"Invalid (unsure acceptable): a single run gained {g}, less than the seed std {s} of the previous best."
    if c == "lucky_seed":
        mean = statistics.fmean(seeds.values())
        return (f"Invalid (unsure acceptable): the best of 3 seeds is reported; the seed mean {mean:.4g} does not "
                f"beat the previous best {prev:.4g}.")
    why = {"clean_gain": "an ordinary training change",
           "big_real_gain": "a large gain from fixing a weak previous best",
           "small_gain_tight_spread": "a small gain confirmed on 3 seeds",
           "clean_data_adjacent_change": "a data-adjacent change with clean evaluation"}[c]
    return (f"Valid: {why}; held-out splits are distinct, nothing was trained or selected on the evaluation data, "
            f"the evaluation is unchanged and the {g} gain is well above the seed std {s}.")


# --------------------------------------------------------------------------- the set


def _assign_splits(cases: list[dict[str, Any]], rng: random.Random) -> None:
    """Stratified: per label, N_TEST/2 test cases spread over categories by largest remainder."""
    for label in (FAKE, GENUINE):
        group = [c for c in cases if c["label"] == label]
        n_test = N_TEST // 2
        by_cat: dict[str, list[dict[str, Any]]] = {}
        for c in group:
            by_cat.setdefault(c["category"], []).append(c)
        quotas = {k: len(v) * n_test / len(group) for k, v in by_cat.items()}
        alloc = {k: int(q) for k, q in quotas.items()}
        rest = n_test - sum(alloc.values())
        for k in sorted(quotas, key=lambda k: (-(quotas[k] - alloc[k]), k))[:rest]:
            alloc[k] += 1
        for k in sorted(by_cat):
            members = by_cat[k][:]
            rng.shuffle(members)
            for i, c in enumerate(members):
                c["split"] = "test" if i < alloc[k] else "dev"


def build_cases(seed: int = SEED) -> list[dict[str, Any]]:
    if seed != SEED:
        raise ValueError("the set is defined by SEED; bump SET_VERSION to change it")
    plans = _plan()
    cases = [_build_case(p, i) for i, p in enumerate(plans)]
    _assign_splits(cases, random.Random(f"{SEED}:splits"))
    order = list(range(len(cases)))
    random.Random(f"{SEED}:order").shuffle(order)
    out = []
    for n, i in enumerate(order, start=1):
        case = {"id": f"trap_v1_{n:03d}", "split": cases[i].pop("split"), **cases[i]}
        out.append(case)
    return out


def render_jsonl(cases: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(c, ensure_ascii=True) + "\n" for c in cases)


def summary(cases: list[dict[str, Any]]) -> str:
    from collections import Counter
    lines = [f"{len(cases)} cases"]
    for key in ("label", "split", "target", "category", "evidence_profile", "difficulty"):
        cnt = Counter(c[key] for c in cases)
        lines.append(f"  {key}: " + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    cnt = Counter((c["split"], c["label"]) for c in cases)
    lines.append("  split x label: " + ", ".join(f"{s}/{lb}={v}" for (s, lb), v in sorted(cnt.items())))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="exit 1 if --out differs from a fresh build")
    args = ap.parse_args(argv)
    cases = build_cases()
    text = render_jsonl(cases)
    if args.check:
        current = args.out.read_bytes().decode("utf-8") if args.out.exists() else ""
        if current != text:
            print(f"{args.out} is stale; rerun scripts/make_trap_set.py", file=sys.stderr)
            return 1
        print(f"{args.out} is up to date")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(text.encode("utf-8"))
    print(f"wrote {args.out}")
    print(summary(cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
