"""Runners: compute backends behind the Runner protocol (workers/runners/base.py).

``ModalRunner`` is imported lazily so that importing this package never imports ``modal``
(only workers/runners/modal_runner.py may).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from workers.runners.base import (
    DEFAULT_GRACE_S,
    DEFAULT_PRICE_PER_SECOND,
    Runner,
    RunnerCapacityError,
    RunnerError,
)
from workers.runners.local import LocalRunner

if TYPE_CHECKING:
    from workers.runners.modal_runner import ModalRunner

__all__ = [
    "DEFAULT_GRACE_S",
    "DEFAULT_PRICE_PER_SECOND",
    "LocalRunner",
    "ModalRunner",
    "Runner",
    "RunnerCapacityError",
    "RunnerError",
    "get_runner",
]


def __getattr__(name: str) -> Any:
    if name == "ModalRunner":
        from workers.runners.modal_runner import ModalRunner

        return ModalRunner
    raise AttributeError(name)


def get_runner(name: str, **kwargs: Any) -> Runner:
    """Build a runner by name: "local" or "modal"."""
    if name == "local":
        return LocalRunner(**kwargs)
    if name == "modal":
        from workers.runners.modal_runner import ModalRunner

        return ModalRunner(**kwargs)
    raise ValueError(f"unknown runner {name!r}; known: local, modal")
