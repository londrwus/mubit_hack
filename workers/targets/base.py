"""Target protocol, TargetSpec, config-space format and the target registry.

A target is pure Python: no compute SDK imports, no backend imports (it runs inside
Modal containers and local subprocesses). Only pydantic + stdlib here.

Config-space format (one entry per tunable parameter), shared by LLM prompts and Optuna:

    {
      "lr":        {"type": "float", "low": 1e-4, "high": 1e-2, "log": true, "default": 1e-3},
      "n_layer":   {"type": "int", "low": 2, "high": 8, "default": 4},
      "optimizer": {"type": "categorical", "choices": ["adamw", "lion"], "default": "adamw"},
      "use_ema":   {"type": "bool", "default": false}
    }

``int`` and ``float`` bounds are inclusive. Every param has a default; the defaults form the
starting-code config. Exactly one param is the contamination trap (``trap_option``); the
trap is active when ``config[trap_option] == trap_value``.
"""

from __future__ import annotations

import importlib
import math
import random
from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

ParamType: TypeAlias = Literal["float", "int", "categorical", "bool"]
Scalar: TypeAlias = float | int | str | bool

EmitFn: TypeAlias = Callable[[int, str, float], None]
"""emit(step, name, value): a target reports a streamed metric. The runner adds t and
experiment_id. Call it at most a few times per second."""


class ParamSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ParamType
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list[Scalar] | None = None
    default: Scalar
    description: str = ""  # one line, shown to the LLM

    @model_validator(mode="after")
    def _check(self) -> ParamSpec:
        if self.type in ("float", "int"):
            if self.low is None or self.high is None or self.low > self.high:
                raise ValueError(f"{self.type} param needs low <= high")
            if self.log and self.low <= 0:
                raise ValueError("log scale needs low > 0")
            if isinstance(self.default, bool) or not isinstance(self.default, (int, float)):
                raise ValueError("numeric param needs a numeric default")
            if not (self.low <= self.default <= self.high):
                raise ValueError("default outside [low, high]")
        elif self.type == "categorical":
            if not self.choices:
                raise ValueError("categorical param needs choices")
            if self.default not in self.choices:
                raise ValueError("default not in choices")
        elif self.type == "bool":
            if not isinstance(self.default, bool):
                raise ValueError("bool param needs a bool default")
        return self


ConfigSpace: TypeAlias = dict[str, ParamSpec]


class TargetSpec(BaseModel):
    """Static description of a target. Serializable; goes into the cached target-spec
    block of every LLM prompt (no timestamps/IDs here: they would break the cache)."""

    model_config = ConfigDict(extra="forbid")

    name: str                         # "lm" | "rl" | "cifar"
    description: str                  # what is trained, in 1-3 sentences, for the LLM
    metric: str                       # "val_bpb" | "mean_return" | "test_acc"
    direction: Literal["min", "max"]
    target_value: float               # for $-to-target / time-to-target; frozen before races
    hardware: str                     # "L4" | "CPU"
    budget_s: int                     # per-run training budget (20-90 s)
    config_space: ConfigSpace
    trap_option: str                  # name of the contamination-trap param
    trap_value: Scalar = True         # value that silently contaminates the eval

    @model_validator(mode="after")
    def _check(self) -> TargetSpec:
        if self.trap_option not in self.config_space:
            raise ValueError("trap_option must be a key of config_space")
        trap = self.config_space[self.trap_option]
        if trap.type == "categorical" and self.trap_value not in (trap.choices or []):
            raise ValueError("trap_value not in trap_option choices")
        if trap.default == self.trap_value:
            raise ValueError("the default config must not enable the trap")
        return self

    # ---- helpers ---------------------------------------------------------------

    def default_config(self) -> dict[str, Any]:
        return {k: p.default for k, p in self.config_space.items()}

    def is_trap(self, config: dict[str, Any]) -> bool:
        return config.get(self.trap_option, self.config_space[self.trap_option].default) == self.trap_value

    def is_better(self, a: float, b: float | None) -> bool:
        """True if score a beats b (b=None counts as no score yet)."""
        if b is None:
            return True
        return a < b if self.direction == "min" else a > b

    def reaches_target(self, score: float) -> bool:
        return score <= self.target_value if self.direction == "min" else score >= self.target_value

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Fill defaults, coerce types, reject unknown keys / out-of-range values.
        Raises ValueError. Use on every LLM-proposed config before launching."""
        unknown = set(config) - set(self.config_space)
        if unknown:
            raise ValueError(f"unknown params: {sorted(unknown)}")
        out: dict[str, Any] = {}
        for name, p in self.config_space.items():
            v = config.get(name, p.default)
            if p.type == "bool":
                if not isinstance(v, bool):
                    raise ValueError(f"{name}: expected bool")
            elif p.type == "categorical":
                if v not in (p.choices or []):
                    raise ValueError(f"{name}: {v!r} not in {p.choices}")
            else:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(f"{name}: expected number")
                if p.type == "int":
                    if float(v) != int(v):
                        raise ValueError(f"{name}: expected int")
                    v = int(v)
                else:
                    v = float(v)
                    if not math.isfinite(v):
                        raise ValueError(f"{name}: not finite")
                assert p.low is not None and p.high is not None
                if not (p.low <= v <= p.high):
                    raise ValueError(f"{name}: {v} outside [{p.low}, {p.high}]")
            out[name] = v
        return out

    def random_config(self, rng: random.Random) -> dict[str, Any]:
        """Uniform (log-uniform where log=True) sample. For the mock LLM / random search."""
        out: dict[str, Any] = {}
        for name, p in self.config_space.items():
            if p.type == "bool":
                out[name] = rng.random() < 0.5
            elif p.type == "categorical":
                out[name] = rng.choice(p.choices or [])
            else:
                assert p.low is not None and p.high is not None
                if p.log:
                    x = math.exp(rng.uniform(math.log(p.low), math.log(p.high)))
                else:
                    x = rng.uniform(p.low, p.high)
                out[name] = min(max(int(round(x)), int(p.low)), int(p.high)) if p.type == "int" else x
        return out

    def suggest_optuna(self, trial: Any) -> dict[str, Any]:
        """Sample a config from an ``optuna.Trial`` (duck-typed; no optuna import here)."""
        out: dict[str, Any] = {}
        for name, p in self.config_space.items():
            if p.type == "bool":
                out[name] = trial.suggest_categorical(name, [False, True])
            elif p.type == "categorical":
                out[name] = trial.suggest_categorical(name, list(p.choices or []))
            elif p.type == "int":
                assert p.low is not None and p.high is not None
                out[name] = trial.suggest_int(name, int(p.low), int(p.high), log=p.log)
            else:
                assert p.low is not None and p.high is not None
                out[name] = trial.suggest_float(name, p.low, p.high, log=p.log)
        return out

    def config_json_schema(self) -> dict[str, Any]:
        """JSON schema for one config object (closed, all params required), for
        structured-output schemas of propose_experiments."""
        props: dict[str, Any] = {}
        for name, p in self.config_space.items():
            if p.type == "bool":
                props[name] = {"type": "boolean"}
            elif p.type == "categorical":
                props[name] = {"enum": list(p.choices or [])}
            elif p.type == "int":
                props[name] = {"type": "integer"}
            else:
                props[name] = {"type": "number"}
            if p.description:
                props[name]["description"] = p.description
        return {
            "type": "object",
            "properties": props,
            "required": list(self.config_space),
            "additionalProperties": False,
        }


@runtime_checkable
class Target(Protocol):
    """A training task. ``run`` executes one training run in-process and returns
    ``{"score": float, **summary_metrics}`` where score is the *reported* metric (what the
    agent's pipeline reports; inflated when the trap is on) and ``heldout_score`` is the
    ground truth on a split the agent never sees. Headline numbers use ``heldout_score``;
    the agent loop hides it from the LLM. It may raise on failure; the runner turns the
    exception into RunResult(status=failed). It must respect budget_s (wall-clock training
    budget, excluding final eval) and be deterministic given (config, seed) as far as the
    hardware allows."""

    spec: TargetSpec

    @property
    def name(self) -> str: ...            # "lm" | "rl" | "cifar"
    @property
    def metric(self) -> str: ...          # "val_bpb" | "mean_return" | "test_acc"
    @property
    def direction(self) -> Literal["min", "max"]: ...
    @property
    def target_value(self) -> float: ...
    @property
    def hardware(self) -> str: ...        # "L4" | "CPU"
    @property
    def budget_s(self) -> int: ...
    @property
    def config_space(self) -> ConfigSpace: ...
    @property
    def trap_option(self) -> str: ...

    def run(self, config: dict[str, Any], seed: int, budget_s: int, emit: EmitFn) -> dict[str, Any]: ...


class BaseTarget:
    """Convenience base: subclass, set ``spec``, implement ``run``. Exposes the spec
    fields (name, metric, direction, ...) as attributes."""

    spec: TargetSpec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def metric(self) -> str:
        return self.spec.metric

    @property
    def direction(self) -> Literal["min", "max"]:
        return self.spec.direction

    @property
    def target_value(self) -> float:
        return self.spec.target_value

    @property
    def hardware(self) -> str:
        return self.spec.hardware

    @property
    def budget_s(self) -> int:
        return self.spec.budget_s

    @property
    def config_space(self) -> ConfigSpace:
        return self.spec.config_space

    @property
    def trap_option(self) -> str:
        return self.spec.trap_option

    def run(self, config: dict[str, Any], seed: int, budget_s: int, emit: EmitFn) -> dict[str, Any]:
        raise NotImplementedError


# --------------------------------------------------------------------------- registry

TARGETS: dict[str, Target] = {}
"""name -> target instance. Target modules call register_target() at import time."""

# Module that defines each known target; get_target() imports it lazily. Importing a target
# module only builds its spec (ML libs are imported inside run()), so this is cheap.
# Hardware (from each spec): lm -> L4, cifar -> L4, rl -> CPU.
TARGET_MODULES: dict[str, str] = {
    "lm": "workers.targets.lm",
    "rl": "workers.targets.rl",
    "cifar": "workers.targets.cifar",
    "lcbench": "workers.targets.lcbench",  # surrogate (YAHPO LCBench); runs on SurrogateRunner
}


def register_target(target: Target) -> Target:
    TARGETS[target.spec.name] = target
    return target


def list_targets() -> list[str]:
    """Names of every known target (registered or importable)."""
    return sorted(set(TARGETS) | set(TARGET_MODULES))


def load_all_targets() -> dict[str, Target]:
    """Import every known target module and return the registry."""
    for name in TARGET_MODULES:
        get_target(name)
    return dict(TARGETS)


def get_target(name: str) -> Target:
    """Return a registered target, importing its module on first use."""
    if name not in TARGETS and name in TARGET_MODULES:
        importlib.import_module(TARGET_MODULES[name])
    try:
        return TARGETS[name]
    except KeyError:
        raise KeyError(f"unknown target {name!r}; known: {list_targets()}") from None
