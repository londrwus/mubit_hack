"""LCBench target: a surrogate of real MLP training runs (YAHPO Gym LCBench, Apache-2.0).

What is "trained"
    LCBench (Zimmer, Lindauer, Hutter 2021) trained ~2,000 funnel-shaped MLP configs on 35
    OpenML tabular classification tasks and logged every epoch. YAHPO Gym (Pfisterer et al.
    2022) fit a neural-net surrogate to those curves: given (task, hyperparameters, epoch) it
    predicts the cumulative training time and the validation / test metrics at that epoch.
    This target evaluates that surrogate (``model.onnx``) with ``onnxruntime`` directly,
    because the ``yahpo_gym`` package pins Python <= 3.11 and numpy < 2.

    One task is fixed: OpenML task ``TASK_ID`` (167168), picked on 24 Oct 2026 for headroom:
    over 500 random configs at 52 epochs its predicted test balanced accuracy spans
    0.26 (p10) / 0.44 (p50) / 0.73 (p90), best 0.87, and the starting config sits at the
    median (0.44-0.46); median predicted runtime is ~80 s per run (scan over all 34 tasks in
    ``prep_lcbench.py scan``).

Metric
    ``score`` (what the agent sees) = predicted **validation** balanced accuracy at the kept
    epoch. ``heldout_score`` (the headline, never shown to the agent) = predicted **test**
    balanced accuracy at the same epoch. The surrogate has no plain test *accuracy* output
    (only ``test_balanced_accuracy``), so both sides use balanced accuracy to compare like
    with like. Caveat (measured in the scan): the surrogate's val and test predictions rank
    configs almost identically (Spearman 0.94-1.00 across tasks), so held-out vs validation
    is a weaker distinction here than on the real targets.

Config space
    The native LCBench space (bounds and log scales from YAHPO's ``config_space.json``),
    renamed only where the name would be unclear, plus one choice of ours:
    ``checkpoint_selection`` (which epoch's weights are kept). Defaults are YAHPO's own
    config-space defaults (not tuned by us), so the starting code is neutral.

Contamination trap (``checkpoint_selection="best_test"``)
    The kept epoch is the one with the best *test* balanced accuracy, and that test number is
    reported: selection on the test split. Under a smooth surrogate the inflation is small,
    and trap runs never count for any arm (``backend.benchmark.metrics.is_counted``).

Runtime and budget
    ``train_time_s`` = the surrogate's predicted cumulative training time at the last trained
    epoch (made non-decreasing across epochs; the raw predictions can wobble). Epochs whose
    cumulative time would exceed ``budget_s`` are not trained (the run stops at the budget,
    like the other targets' wall-clock budgets). If not even one epoch fits, the run fails
    with ``TimeoutError``. ``run()`` itself returns in milliseconds: the time is *charged* and
    *paced* by ``workers.runners.surrogate.SurrogateRunner`` (predicted time x a multiplier).

Determinism
    The surrogate is deterministic: the same config gives the same result for every seed
    (YAHPO's noisy surrogate is not published in yahpo_data). ``seed`` is accepted and ignored.

Files (cached under ``workers/targets/data/cache/lcbench/``, git-ignored; override with
``LCBENCH_DATA_DIR``): fetched by ``python -m workers.targets.data.prep_lcbench fetch`` from
``slds-lmu/yahpo_data`` at commit ``YAHPO_DATA_COMMIT`` and checked against ``FILES``
(sha256). Input / output order of the ONNX model follows yahpo_gym's LCBench config
(``cont_names`` / ``y_names``); a plausibility check is in ``tests/test_lcbench.py``.

Imports: stdlib + pydantic at module level; numpy / onnxruntime are imported lazily.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from workers.targets.base import BaseTarget, EmitFn, ParamSpec, TargetSpec, register_target

# ---------------------------------------------------------------- data files

YAHPO_DATA_REPO = "slds-lmu/yahpo_data"
YAHPO_DATA_COMMIT = "b627cb8b1862c1181b3dda2cd23ad83d4cb4c42b"  # latest commit touching lcbench/ (2024-08-17)
FILES: dict[str, str] = {  # file -> sha256 (as downloaded 2026-10-24)
    "model.onnx": "b45c0d85ae39abf8820c8bea5ac82f415763afcbaf3b725e36af452b09cb6dcd",
    "encoding.json": "d8af928f412bbeb07e58ecd69877fcab35fbc595333d1c1c3f76120d2f2508da",
    "config_space.json": "4d1c78a91429b4a37628e13f286cf1abe15f778a09722a4c01714f2d72a8200d",
    "LICENSE-2.0.txt": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "cache" / "lcbench"

# yahpo_gym LCBench config: continuous inputs in this order, one categorical (the task id).
CONT_NAMES = ("epoch", "batch_size", "learning_rate", "momentum", "weight_decay", "num_layers",
              "max_units", "max_dropout")
Y_NAMES = ("time", "val_accuracy", "val_cross_entropy", "val_balanced_accuracy", "test_cross_entropy",
           "test_balanced_accuracy")
MAX_EPOCH = 52

TASK_ID = "167168"
BUDGET_S = 300            # per-run training budget, in predicted seconds
TARGET_VALUE_PROVISIONAL = 0.78  # = the frozen goal in data/calibration/target_values.json (prep_lcbench calibrate)


def data_dir() -> Path:
    return Path(os.environ.get("LCBENCH_DATA_DIR") or DEFAULT_DATA_DIR)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_files(directory: Path | None = None) -> dict[str, bool]:
    """file -> present with the pinned sha256."""
    d = directory or data_dir()
    return {name: (d / name).exists() and sha256_file(d / name) == digest for name, digest in FILES.items()}


# ---------------------------------------------------------------- surrogate


class Surrogate:
    """The YAHPO LCBench ONNX surrogate for one task. Thread-safe; load once per process."""

    def __init__(self, directory: Path | None = None, task_id: str = TASK_ID) -> None:
        import numpy as np
        import onnxruntime as ort

        d = directory or data_dir()
        bad = [n for n, ok in check_files(d).items() if not ok and n != "LICENSE-2.0.txt"]
        if bad:
            raise FileNotFoundError(f"LCBench surrogate files missing or changed in {d}: {bad}; run "
                                    "`python -m workers.targets.data.prep_lcbench fetch`")
        enc = json.loads((d / "encoding.json").read_text(encoding="utf-8"))["OpenML_task_id"]
        if task_id not in enc:
            raise KeyError(f"unknown OpenML task {task_id!r}")
        self.task_id = task_id
        self._code = int(enc[task_id])
        self._np = np
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3  # hide the harmless "re-use buffer" shape warnings
        self._sess = ort.InferenceSession(str(d / "model.onnx"), sess_options=opts,
                                          providers=["CPUExecutionProvider"])

    def predict(self, rows: list[dict[str, float]]) -> Any:
        """rows of {CONT_NAMES...} -> array (n, 6) in Y_NAMES order."""
        np = self._np
        x_cont = np.array([[float(r[k]) for k in CONT_NAMES] for r in rows], dtype=np.float32)
        x_cat = np.full((len(rows), 1), self._code, dtype=np.int32)
        return self._sess.run(None, {"x_cat": x_cat, "x_cont": x_cont})[0]


_SURROGATE: Surrogate | None = None
_LOCK = threading.Lock()


def get_surrogate() -> Surrogate:
    global _SURROGATE
    with _LOCK:
        if _SURROGATE is None:
            _SURROGATE = Surrogate()
        return _SURROGATE


# ---------------------------------------------------------------- spec

CONFIG_SPACE: dict[str, ParamSpec] = {
    "epochs": ParamSpec(type="int", low=1, high=MAX_EPOCH, default=26,
                        description="training epochs (more epochs take proportionally longer)"),
    "batch_size": ParamSpec(type="int", low=16, high=512, log=True, default=91,
                            description="SGD minibatch size"),
    "learning_rate": ParamSpec(type="float", low=1e-4, high=1e-1, log=True, default=0.0031622777,
                               description="SGD learning rate"),
    "momentum": ParamSpec(type="float", low=0.1, high=0.99, default=0.545,
                          description="SGD momentum"),
    "weight_decay": ParamSpec(type="float", low=1e-5, high=0.1, default=0.050005,
                              description="L2 weight decay"),
    "num_layers": ParamSpec(type="int", low=1, high=5, default=3,
                            description="number of hidden layers"),
    "max_units": ParamSpec(type="int", low=64, high=1024, log=True, default=256,
                           description="width of the first hidden layer; later layers shrink (funnel shape)"),
    "max_dropout": ParamSpec(type="float", low=0.0, high=1.0, default=0.5,
                             description="dropout rate of the last hidden layer; earlier layers use less"),
    "checkpoint_selection": ParamSpec(type="categorical", choices=["last_epoch", "best_validation", "best_test"],
                                      default="last_epoch",
                                      description="which epoch's weights are kept and scored: the last epoch, the "
                                                  "epoch with the best validation score, or the epoch with the best "
                                                  "test-split score"),
}

SPEC = TargetSpec(
    name="lcbench",
    description=(
        "A funnel-shaped MLP trained with SGD on a tabular multi-class classification task. Results "
        "come from a surrogate model fit to real training runs, so they are deterministic (the same "
        "config always gives the same result). Each run takes the training time its config needs "
        "(more epochs, more and wider layers and smaller batches take longer); epochs that would not "
        "finish within the per-run budget are not trained. The score is balanced accuracy on the "
        "validation split (0 to 1, higher is better)."
    ),
    metric="val_balanced_accuracy",
    direction="max",
    target_value=TARGET_VALUE_PROVISIONAL,
    hardware="L4",
    budget_s=BUDGET_S,
    config_space=CONFIG_SPACE,
    trap_option="checkpoint_selection",
    trap_value="best_test",
)


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def evaluate(config: dict[str, Any], budget_s: float, surrogate: Surrogate | None = None,
             emit: EmitFn | None = None) -> dict[str, Any]:
    """Score one (validated) config. Returns {score, heldout_score, train_time_s, ...}.
    Raises TimeoutError if not even the first epoch fits in ``budget_s``."""
    s = surrogate or get_surrogate()
    n_ep = int(config["epochs"])
    hp = {k: float(config[k]) for k in CONT_NAMES if k != "epoch"}
    y = s.predict([{**hp, "epoch": float(e)} for e in range(1, n_ep + 1)])
    times: list[float] = []
    t_max = 0.0
    for row in y:
        t_max = max(t_max, float(row[0]), 0.0)  # cumulative time, made non-decreasing
        times.append(t_max)
    fit = sum(1 for t in times if t <= budget_s)
    if fit == 0:
        raise TimeoutError(f"the first epoch needs {times[0]:.0f} s, over the {budget_s:.0f} s budget")
    val_bacc = [_clip01(r[3]) for r in y[:fit]]
    test_bacc = [_clip01(r[5]) for r in y[:fit]]
    sel = config["checkpoint_selection"]
    if sel == "best_validation":
        idx = max(range(fit), key=lambda i: (val_bacc[i], i))
    elif sel == "best_test":
        idx = max(range(fit), key=lambda i: (test_bacc[i], i))
    else:
        idx = fit - 1
    if emit is not None:
        for e in range(fit):
            emit(e + 1, "train_time_s", times[e])
            emit(e + 1, "val_balanced_accuracy", val_bacc[e])
    kept = y[idx]
    score = test_bacc[idx] if sel == "best_test" else val_bacc[idx]
    return {
        "score": score,
        "heldout_score": test_bacc[idx],                 # ground truth: hidden from the agent
        "heldout_cross_entropy": float(kept[4]),         # hidden ("heldout" in the key)
        "val_accuracy": _clip01(float(kept[1]) / 100.0),
        "val_cross_entropy": float(kept[2]),
        "epochs_trained": float(fit),
        "kept_epoch": float(idx + 1),
        "train_time_s": times[fit - 1],
        "budget_hit": fit < n_ep,
    }


class LCBenchTarget(BaseTarget):
    spec = SPEC

    def prewarm(self) -> None:
        """Load the ONNX session now (the runner calls this before the race clock starts)."""
        get_surrogate()

    def run(self, config: dict[str, Any], seed: int, budget_s: int, emit: EmitFn) -> dict[str, Any]:
        out = evaluate(config, float(budget_s), emit=emit)
        print(f"[lcbench] trained {int(out['epochs_trained'])}/{int(config['epochs'])} epochs, predicted "
              f"training time {out['train_time_s']:.1f} s, kept epoch {int(out['kept_epoch'])}, "
              f"score={out['score']:.4f}")
        return out


TARGET = register_target(LCBenchTarget())
