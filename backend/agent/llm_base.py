"""LLMClient protocol: the only interface the agent loop uses to talk to an LLM.

Implementations: backend/agent/mubit_client.py (real, the ONLY place SDK clients are
constructed; base_url https://api.mubit.ai/anthropic, api_key MUBIT_API_KEY) and a mock.

Contract for ``call``:
- Sends exactly ONE request (no internal retry/fallback). Retry-once-then-Opus-fallback is
  policy in the agent loop; it calls again with ``is_retry`` / ``is_fallback`` set on the
  returned records (``record.model_copy(update=...)``) or via implementation kwargs.
- Never raises for model-output or transport problems: returns ``(None, record)`` with
  ``record.valid = False`` and ``record.error`` set.
- ``record=False`` (the default) MUST send header ``x-middleware-record: off`` and set
  ``LLMCallRecord.recorded = False``. Only frozen-prompt example-collection / race calls
  pass ``record=True``.
- ``session_id`` goes to ``metadata.user_id`` (Anthropic); build it with ``session_id_for``.
- ``system`` is the frozen step prompt; all per-call content goes in ``user``.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from backend.core.types import ArmKind, LLMCallRecord, StepType

T = TypeVar("T", bound=BaseModel)

MUBIT_ANTHROPIC_BASE_URL = "https://api.mubit.ai/anthropic"
RECORD_HEADER = "x-middleware-record"
REQUEST_ID_HEADER = "x-middleware-request-id"

OPUS_MODEL = "claude-opus-5-5"
DEV_MODEL = "claude-haiku-5-5"


@runtime_checkable
class LLMClient(Protocol):
    async def call(
        self,
        step: StepType,
        system: str,
        user: str,
        schema: type[T],
        arm: ArmKind,
        session_id: str,
        record: bool = False,
    ) -> tuple[T | None, LLMCallRecord]: ...


def session_id_for(race_id: str, arm: ArmKind | str, seed: int) -> str:
    """Mubit session per researcher run: ``race-{id}/{arm}/seed-{n}``."""
    return f"race-{race_id}/{ArmKind(arm).value}/seed-{seed}"


# USD per 1M tokens (budget.md, cached 2026-10-06). Estimates: prefer the cost Mubit reports
# when available. cache_write = 1.25 x input (5-minute TTL).
PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.00, "cache_write": 5.00, "cache_read": 0.20, "output": 20.00},
    "claude-sonnet-5-5": {"input": 2.00, "cache_write": 2.50, "cache_read": 0.20, "output": 10.00},
    "claude-haiku-5-5": {"input": 0.10, "cache_write": 0.125, "cache_read": 0.01, "output": 0.50},
    "claude-haiku-4-5": {"input": 1.00, "cache_write": 1.25, "cache_read": 0.10, "output": 5.00},
}


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
    output_tokens: int,
) -> float:
    """Cost from the four usage meters. Unknown models cost as Opus (conservative)."""
    p = PRICES_PER_MTOK.get(model, PRICES_PER_MTOK[OPUS_MODEL])
    return (
        input_tokens * p["input"]
        + cache_creation_tokens * p["cache_write"]
        + cache_read_tokens * p["cache_read"]
        + output_tokens * p["output"]
    ) / 1_000_000
