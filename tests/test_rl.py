"""Tests for the RL target (workers/targets/rl.py): spec, trap, dynamics and short runs.

The training tests need jax + optax (``uv pip install jax optax``); they are skipped if
those are missing. The dynamics cross-check also needs gymnax (skipped otherwise).
"""

from __future__ import annotations

import math
import subprocess
import sys

import pytest

from workers.targets.base import get_target
from workers.targets.rl import (
    HELDOUT_LEVEL_RANGES,
    SPEC,
    TARGET,
    TRAIN_LEVEL_RANGES,
    Levels,
    PState,
    pendulum_step,
    sample_levels,
)

TINY = {
    "num_envs": 4,
    "num_steps": 16,
    "hidden_size": 16,
    "update_epochs": 1,
    "num_minibatches": 1,
    "lr": 1e-3,
}

SUMMARY_KEYS = {
    "score", "reported_score", "heldout_score", "train_levels_score", "is_trap",
    "final_train_return", "mean_train_return", "env_steps", "updates", "compile_s",
    "train_s", "eval_s", "steps_per_s", "final_entropy",
}


# ---------------------------------------------------------------- spec / registry


def test_registry_returns_rl_target() -> None:
    t = get_target("rl")
    assert t is TARGET
    assert t.name == "rl" and t.metric == "mean_return" and t.direction == "max"
    assert t.hardware == "CPU" and t.budget_s == 30
    assert t.trap_option == "eval_on_train_levels"


def test_default_config_is_valid_and_not_trap() -> None:
    cfg = SPEC.validate_config({})
    assert cfg == SPEC.default_config()
    assert not SPEC.is_trap(cfg)
    assert SPEC.is_trap({"eval_on_train_levels": True})
    assert 8 <= len(SPEC.config_space) <= 19


def test_config_json_schema_is_closed() -> None:
    schema = SPEC.config_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(SPEC.config_space)
    assert schema["properties"]["num_envs"]["enum"] == [4, 8, 16, 32, 64, 128]


def test_random_configs_validate_and_divide_batch() -> None:
    import random

    rng = random.Random(0)
    for _ in range(200):
        cfg = SPEC.validate_config(SPEC.random_config(rng))
        assert (cfg["num_envs"] * cfg["num_steps"]) % cfg["num_minibatches"] == 0


def test_invalid_config_raises_before_training() -> None:
    with pytest.raises(ValueError):
        TARGET.run({"lr": 1.0}, 0, 30, lambda *a: None)
    with pytest.raises(ValueError):
        TARGET.run({"not_a_param": 1}, 0, 30, lambda *a: None)


def test_module_import_does_not_import_jax() -> None:
    code = "import sys, workers.targets.rl; print('jax' in sys.modules, 'modal' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False"


def test_heldout_levels_are_wider_than_training_levels() -> None:
    for name, (lo, hi) in TRAIN_LEVEL_RANGES.items():
        hlo, hhi = HELDOUT_LEVEL_RANGES[name]
        assert hlo <= lo and hi <= hhi
    a = sample_levels(1, 16, TRAIN_LEVEL_RANGES)
    b = sample_levels(1, 16, TRAIN_LEVEL_RANGES)
    assert all((x == y).all() for x, y in zip(a, b, strict=True))  # deterministic


# ---------------------------------------------------------------- dynamics


def test_pendulum_dynamics_match_gymnax() -> None:
    pytest.importorskip("jax")
    gymnax = pytest.importorskip("gymnax")
    import jax
    import jax.numpy as jnp
    import numpy as np

    env, params = gymnax.make("Pendulum-v1")
    from gymnax.environments.classic_control.pendulum import EnvState

    levels = Levels(np.ones(1, np.float32), np.ones(1, np.float32), np.ones(1, np.float32))
    rng = np.random.default_rng(0)
    for _ in range(20):
        th, thd, a = rng.uniform(-math.pi, math.pi), rng.uniform(-8, 8), rng.uniform(-3, 3)
        g_state = EnvState(theta=jnp.float32(th), theta_dot=jnp.float32(thd), last_u=jnp.float32(0.0), time=0)
        _, g_next, g_rew, _, _ = env.step_env(jax.random.PRNGKey(0), g_state, jnp.float32(a), params)
        s = PState(jnp.float32(th), jnp.float32(thd), jnp.int32(0), jnp.int32(0))
        ns, rew = pendulum_step(jnp, s, jnp.float32(a), levels)
        assert float(ns.theta) == pytest.approx(float(g_next.theta), abs=1e-5)
        assert float(ns.theta_dot) == pytest.approx(float(g_next.theta_dot), abs=1e-5)
        assert float(rew) == pytest.approx(float(g_rew), abs=1e-4)


# ---------------------------------------------------------------- short runs


def _run(cfg: dict, seed: int = 0, budget_s: int = 6) -> tuple[dict, list[tuple[int, str, float]]]:
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    emitted: list[tuple[int, str, float]] = []
    res = TARGET.run(cfg, seed, budget_s, lambda step, name, value: emitted.append((step, name, value)))
    return res, emitted


def test_short_run_returns_summary_and_streams_metrics() -> None:
    res, emitted = _run(TINY)
    assert SUMMARY_KEYS <= set(res)
    assert all(isinstance(v, float) and math.isfinite(v) for v in res.values())
    assert res["is_trap"] == 0.0
    assert res["score"] == res["heldout_score"] == res["reported_score"]
    assert -2000.0 < res["score"] <= 0.0  # Pendulum returns are negative, 200 steps
    assert res["updates"] >= 1 and res["env_steps"] == res["updates"] * 4 * 16
    names = {n for _, n, _ in emitted}
    assert {"train_return", "env_steps", "loss", "entropy"} <= names
    steps = [s for s, n, _ in emitted if n == "env_steps"]
    assert steps == sorted(steps)


def test_trap_reports_training_level_score() -> None:
    res, _ = _run({**TINY, "eval_on_train_levels": True})
    assert res["is_trap"] == 1.0
    assert res["score"] == res["reported_score"] == res["train_levels_score"]
    assert res["heldout_score"] != res["score"]


def test_zero_budget_raises_timeout() -> None:
    pytest.importorskip("jax")
    with pytest.raises(TimeoutError):
        TARGET.run(TINY, 0, 0, lambda *a: None)


# ---------------------------------------------------------------- research choices (D29)

RESEARCH_DEFAULTS = {"hidden_layers": 2, "activation": "tanh", "obs_norm": False, "reward_norm": False,
                     "value_clip": True}

# Recorded from the pre-D29 code (jax 0.12.0.1 CPU) on 24 Oct via the _build path (fixed number
# of updates; the time loop is machine-dependent): the default of every research choice must
# reproduce the original code path exactly.
RECORDED_JAX = "0.12"
RT = {"num_envs": 4, "num_steps": 16, "hidden_size": 16, "update_epochs": 2, "num_minibatches": 2, "lr": 1e-3}
RECORDED = [
    ({}, 0, [841.361328125, 500.85772705078125, 666.9578247070312, 798.5294189453125, 617.420654296875],
     -1307.38525390625),
    ({"hidden_size": 32, "reward_scale": 0.1, "ent_coef": 0.01}, 7,
     [8.151901245117188, 7.371377944946289, 6.530877590179443, 7.336994171142578, 7.491920471191406],
     -1210.326416015625),
]


def _fixed_updates(cfg: dict, seed: int, n: int = 5) -> tuple[list[float], float]:
    """init + n jitted updates + held-out eval (1 episode per level): deterministic."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("optax")
    import jax.numpy as jnp
    import numpy as np

    from workers.targets import rl

    c = SPEC.validate_config(cfg)
    built = rl._build(c, sample_levels(rl.TRAIN_LEVEL_SEED, rl.N_TRAIN_LEVELS, TRAIN_LEVEL_RANGES))
    st = built.init(seed, jax.random.PRNGKey(seed))
    losses = []
    for _ in range(n):
        st, stats = built.update(st, jnp.float32(c["lr"]))
        losses.append(float(stats["loss"]))
    idx = np.arange(rl.N_HELDOUT_LEVELS, dtype=np.int32)
    keys = jax.random.split(jax.random.PRNGKey(rl.HELDOUT_RESET_SEED), idx.shape[0])
    hl = sample_levels(rl.HELDOUT_LEVEL_SEED, rl.N_HELDOUT_LEVELS, HELDOUT_LEVEL_RANGES)
    norm_stats = st[6] if len(st) > 6 else None
    return losses, float(np.asarray(built.evaluate(st[0], hl, idx, keys, norm_stats)).mean())


def test_research_choice_defaults_and_descriptions() -> None:
    for name, default in RESEARCH_DEFAULTS.items():
        p = SPEC.config_space[name]
        assert p.default == default and p.description
        assert p.type in ("categorical", "bool", "int")
    assert SPEC.config_space["activation"].choices == ["tanh", "relu"]
    assert not SPEC.is_trap({**SPEC.default_config(), "obs_norm": True, "reward_norm": True})


@pytest.mark.parametrize("extra,seed,losses,eval_mean", RECORDED)
def test_default_config_reproduces_recorded_values(extra: dict, seed: int, losses: list, eval_mean: float) -> None:
    jax = pytest.importorskip("jax")
    if not jax.__version__.startswith(RECORDED_JAX):
        pytest.skip(f"values recorded with jax {RECORDED_JAX}")
    got, ev = _fixed_updates(RT | extra, seed)
    assert got == pytest.approx(losses, rel=1e-6)
    assert ev == pytest.approx(eval_mean, rel=1e-6)
    got2, ev2 = _fixed_updates(RT | extra | RESEARCH_DEFAULTS, seed)  # explicit == omitted
    assert got2 == got and ev2 == ev


@pytest.mark.parametrize("extra", [
    {"hidden_layers": 1}, {"hidden_layers": 3}, {"activation": "relu"}, {"obs_norm": True},
    {"reward_norm": True}, {"value_clip": False}, {"obs_norm": True, "reward_norm": True},
])
def test_research_choice_trains(extra: dict) -> None:
    ref, ref_ev = _fixed_updates(RT, 0)
    got, ev = _fixed_updates(RT | extra, 0)
    assert all(math.isfinite(x) for x in got) and math.isfinite(ev)
    assert -2000.0 < ev <= 0.0
    assert got != ref  # the option changes training


def test_rms_update_matches_numpy() -> None:
    import numpy as np

    from workers.targets.rl import _rms_update

    rng = np.random.default_rng(0)
    a, b = rng.normal(2.0, 3.0, (50, 3)), rng.normal(-1.0, 0.5, (70, 3))
    m, v, c = _rms_update(a.mean(0), a.var(0), 50.0, b)
    both = np.concatenate([a, b])
    assert c == 120.0
    assert np.allclose(m, both.mean(0)) and np.allclose(v, both.var(0))


def test_normalised_run_end_to_end_and_trap() -> None:
    """Full run() with both normalisers: the frozen obs statistics reach the final eval."""
    res, _ = _run({**TINY, "obs_norm": True, "reward_norm": True, "hidden_layers": 1}, budget_s=15)  # compile counts; 4 s flaked under load
    assert res["is_trap"] == 0.0 and res["score"] == res["heldout_score"]
    assert -2000.0 < res["score"] <= 0.0
    trap, _ = _run({**TINY, "obs_norm": True, "eval_on_train_levels": True}, budget_s=15)
    assert trap["is_trap"] == 1.0 and trap["score"] == trap["train_levels_score"]
