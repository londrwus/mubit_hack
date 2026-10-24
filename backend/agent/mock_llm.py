"""MockLLM: deterministic fake LLMClient for UI, scheduler and runner plumbing. No network.

- propose_experiments: seeded random valid configs from the target's config space (K parsed
  from the user message, or ``k=`` kwarg), no duplicates within a batch; the trap param is
  left at its default except with probability ``trap_rate``.
- diagnose_run: keyword heuristics on the run block (nan -> divergence, OOM -> oom, ...).
- judge_improvement: ``invalid`` when the evidence shows overlap / test-split selection or the
  trap param changed to its trap value; otherwise ``valid``.

Every record has ``model="mock-llm"``, ``is_replay=True`` and cost 0, and the class has
``benchmark_safe = False``, so a benchmark run can reject it (``is_benchmark_safe``).
Output is a pure function of (seed, step, session_id, attempt flags, user text).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from backend.agent.mubit_client import parse_and_validate
from backend.agent.prompts.build import FREE_SLOTS_LABEL, split_user
from backend.core.types import ArmKind, FailureClass, LLMCallRecord, StepType, Verdict
from workers.targets.base import TargetSpec

T = TypeVar("T", bound=BaseModel)

MOCK_MODEL = "mock-llm"

_HYPOTHESES = [
    "Testing whether {p} near {v} speeds up convergence within the run budget.",
    "Exploring {p}={v} to probe an untested region of the space.",
    "Refining around the best run by moving {p} to {v}.",
    "Checking if {p}={v} reduces instability seen in earlier runs.",
]


class MockLLM:
    benchmark_safe = False

    def __init__(
        self,
        spec: TargetSpec | None = None,
        *,
        seed: int = 0,
        latency_s: float = 0.0,
        latency_jitter_s: float = 0.0,
        invalid_rate: float = 0.0,
        trap_rate: float = 0.1,
    ) -> None:
        self.spec = spec
        self.seed = seed
        self.latency_s = latency_s
        self.latency_jitter_s = latency_jitter_s
        self.invalid_rate = invalid_rate
        self.trap_rate = trap_rate
        self.calls: list[LLMCallRecord] = []

    # ------------------------------------------------------------------ LLMClient

    async def call(
        self,
        step: StepType,
        system: str,
        user: str,
        schema: type[T],
        arm: ArmKind,
        session_id: str,
        record: bool = False,
        *,
        validator: Callable[[T], T | None] | None = None,
        is_retry: bool = False,
        is_fallback: bool = False,
        experiment_ids: list[str] | None = None,
        arm_id: str | None = None,
        k: int | None = None,
        **_: Any,  # accepts MubitClient-only kwargs (wire_schema, model, effort, ...)
    ) -> tuple[T | None, LLMCallRecord]:
        step = StepType(step)
        rng = self._rng(step, session_id, user, is_retry, is_fallback)
        rec = LLMCallRecord(
            arm=ArmKind(arm),
            arm_id=arm_id,
            session_id=session_id,
            step_type=step,
            model=MOCK_MODEL,
            is_replay=True,
            recorded=False,  # nothing is ever sent
            is_retry=is_retry,
            is_fallback=is_fallback,
            stop_reason="end_turn",
            experiment_ids=list(experiment_ids or []),
        )
        t0 = time.perf_counter()
        delay = self.latency_s + (rng.uniform(0, self.latency_jitter_s) if self.latency_jitter_s else 0.0)
        if delay > 0:
            await asyncio.sleep(delay)

        if not is_fallback and self.invalid_rate > 0 and rng.random() < self.invalid_rate:
            text: str | None = '{"oops": "simulated malformed reply"}'
        else:
            try:
                text = json.dumps(self._answer(step, user, rng, k))
            except ValueError as e:
                rec.latency_ms = (time.perf_counter() - t0) * 1000
                rec.error = f"mock: {e}"
                self.calls.append(rec)
                return None, rec

        parsed, err = parse_and_validate(text, schema, validator)
        rec.latency_ms = (time.perf_counter() - t0) * 1000
        rec.valid = parsed is not None
        rec.error = err
        self.calls.append(rec)
        return parsed, rec

    # ------------------------------------------------------------------ internals

    def _rng(self, step: StepType, session_id: str, user: str, is_retry: bool, is_fallback: bool) -> random.Random:
        key = f"{self.seed}|{step.value}|{session_id}|{int(is_retry)}{int(is_fallback)}|{user}"
        return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))

    def _answer(self, step: StepType, user: str, rng: random.Random, k: int | None) -> dict[str, Any]:
        _, digest = split_user(user)
        if step is StepType.propose_experiments:
            return self._propose(digest, rng, k)
        if step is StepType.diagnose_run:
            return self._diagnose(digest)
        return self._judge(digest)

    def _propose(self, digest: str, rng: random.Random, k: int | None) -> dict[str, Any]:
        if self.spec is None:
            raise ValueError("MockLLM needs a TargetSpec for propose_experiments")
        spec = self.spec
        if k is None:
            m = re.search(re.escape(FREE_SLOTS_LABEL) + r"\s*(\d+)", digest)
            k = int(m.group(1)) if m else 1
        trap_default = spec.config_space[spec.trap_option].default
        seen: set[str] = set()
        exps: list[dict[str, Any]] = []
        for _ in range(max(1, k) * 20):
            if len(exps) >= max(1, k):
                break
            cfg = spec.random_config(rng)
            cfg[spec.trap_option] = spec.trap_value if rng.random() < self.trap_rate else trap_default
            cfg = spec.validate_config(cfg)
            key = json.dumps(cfg, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            p = rng.choice([n for n in spec.config_space if n != spec.trap_option] or list(spec.config_space))
            v = cfg[p]
            v_s = f"{v:.3g}" if isinstance(v, float) else str(v)
            exps.append({"config": cfg, "hypothesis": rng.choice(_HYPOTHESES).format(p=p, v=v_s)})
        return {"experiments": exps}

    @staticmethod
    def _diagnose(digest: str) -> dict[str, Any]:
        t = digest.lower()
        rules: list[tuple[tuple[str, ...], FailureClass, str, str]] = [
            (("out of memory", "outofmemory", " oom"), FailureClass.oom,
             "The log reports running out of memory.", "Reduce the batch size or the model size."),
            ((" nan", "nan ", "=nan", " inf", "diverg", "overflow"), FailureClass.divergence,
             "The loss became NaN or exploded.", "Lower the learning rate and add warmup."),
            (("timed_out", "timeout", "timed out"), FailureClass.timeout,
             "The run exceeded its wall-clock budget.", "Use a smaller model or larger batch."),
            (("invalid config", "invalid_config"), FailureClass.invalid_config,
             "The config was rejected.", "Keep every value inside its bounds."),
            (("traceback", "exception", "error:"), FailureClass.crash,
             "The process raised an exception.", "Revert the config change that triggered it."),
        ]
        haystack = t.replace("error: none", "")  # the builder's placeholder is not an error
        for keys, cls, cause, fix in rules:
            if any(kw in haystack for kw in keys):
                return {"failure_class": cls.value, "cause": cause, "fix": fix}
        return {"failure_class": FailureClass.no_failure.value,
                "cause": "The loss decreased steadily and no error appears.", "fix": "No change needed."}

    def _judge(self, digest: str) -> dict[str, Any]:
        t = digest.lower()
        for m in re.finditer(r"overlap[^:\n]*:\s*([0-9.]+)", t):
            if float(m.group(1)) > 0:
                return {"verdict": Verdict.invalid.value, "reason": "Evidence shows overlap between eval and training data."}
        if re.search(r"selection[^:\n]*:\s*test", t):
            return {"verdict": Verdict.invalid.value, "reason": "The test split was used for selection."}
        if self.spec is not None:
            spec = self.spec
            changed = re.search(r"changed parameters:(.*)", digest)
            tv = "true" if spec.trap_value is True else ("false" if spec.trap_value is False else str(spec.trap_value).lower())
            if changed and re.search(rf"\b{re.escape(spec.trap_option.lower())}\b[^,]*->\s*{re.escape(tv)}\b", changed.group(1).lower()):
                return {"verdict": Verdict.invalid.value, "reason": f"Gain coincides with enabling {spec.trap_option}, which touches evaluation data."}
        if "(none provided)" in t:
            return {"verdict": Verdict.unsure.value, "reason": "No evidence was provided to rule out leakage."}
        return {"verdict": Verdict.valid.value, "reason": "Splits are distinct and the change is a plausible training improvement."}


__all__ = ["MOCK_MODEL", "MockLLM"]
