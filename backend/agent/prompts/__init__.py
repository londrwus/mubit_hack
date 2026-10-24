"""Versioned system prompts and user-message builders for the research steps (4th: early_stop, D34).

STATUS: DRAFT v0 - NOT frozen. A human freezes a version (sets FROZEN[step] = version)
before any recorded example collection. Changing a frozen prompt's text creates a new
Mubit task and resets example collection, so never edit a frozen file: add a new version.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

from backend.core.types import StepType

PROMPTS_DIR = Path(__file__).resolve().parent

# The version each step uses by default. Bump by adding a new file, never by editing.
CURRENT_VERSION: dict[StepType, str] = {
    StepType.propose_experiments: "v1",
    StepType.diagnose_run: "v0",
    StepType.judge_improvement: "v1",
    StepType.early_stop: "v2",
}

# step -> frozen version, or None while the prompt is still a draft. Only a human edits this.
FROZEN: dict[StepType, str | None] = {
    StepType.propose_experiments: "v1",  # frozen 2026-10-24 ~01:00 BST (D27)
    StepType.diagnose_run: "v0",         # frozen 2026-10-24 ~01:00 BST (D27)
    StepType.judge_improvement: "v1",    # frozen 2026-10-24 ~04:40 BST (D37, D39 gate R32); v0 frozen ~01:00 (D27)
    StepType.early_stop: "v2",           # frozen 2026-10-24 (D34/D39), decision only; v0, v1 never recorded.
                                         # Recording is gated by D39 (gate_summary_<fp>.json), not by this entry.
}

# Leading HTML comment(s) hold file metadata (status, notes); they are stripped before sending.
_HEADER_RE = re.compile(r"\A(?:\s*<!--.*?-->\s*)+", re.DOTALL)


def prompt_path(step: StepType | str, version: str | None = None) -> Path:
    step = StepType(step)
    return PROMPTS_DIR / f"{step.value}.{version or CURRENT_VERSION[step]}.md"


@lru_cache(maxsize=None)
def _load(step: StepType, version: str) -> str:
    raw = prompt_path(step, version).read_text(encoding="utf-8")
    text = _HEADER_RE.sub("", raw, count=1).strip()
    # Normalise line endings so the cached prefix is byte-identical on every OS.
    return text.replace("\r\n", "\n") + "\n"


def load_system_prompt(step: StepType | str, version: str | None = None) -> str:
    """The exact system prompt text sent for ``step`` (metadata header removed)."""
    step = StepType(step)
    return _load(step, version or CURRENT_VERSION[step])


def prompt_fingerprint(step: StepType | str, version: str | None = None) -> str:
    """sha256[:12] of the sent text. Log it with every call; a change means a new Mubit task."""
    return hashlib.sha256(load_system_prompt(step, version).encode("utf-8")).hexdigest()[:12]


def is_frozen(step: StepType | str) -> bool:
    return FROZEN[StepType(step)] is not None
