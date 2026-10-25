"""MubitClient: the ONLY place an LLM SDK client is constructed.

Every request goes to the Mubit gateway (``https://api.mubit.ai/anthropic``) with
``MUBIT_API_KEY``. Never construct an Anthropic client elsewhere and never read
``ANTHROPIC_API_KEY`` (the provider key lives in Mubit -> Provider keys).

Request layout (budget.md, levers 2-6):
- stateless single turn; ``system`` = frozen step prompt with cache breakpoint 1;
- user message = target-spec block (cache breakpoint 2) + dynamic digest, split on
  ``prompts.build.CACHE_BREAK``;
- structured output: ``output_config.format`` json_schema (closed wire schema), then
  Pydantic validation (+ optional extra validator, e.g. ``spec.validate_config``);
- ``output_config.effort`` per step (propose medium, diagnose low, judge low, early_stop low);
- ``metadata.user_id`` = session id ``race-{id}/{arm}/seed-{n}``;
- header ``x-middleware-record: off`` unless ``record=True`` (``record`` also accepts the
  header values ``"full"`` / ``"metadata"`` / ``"off"``; ``"metadata"`` keeps the task
  fingerprint but drops the content, see docs/mubit_office_hours.md item 2);
- a FALLBACK attempt (``is_fallback=True``) is always sent with ``x-middleware-record: off`` and
  without adapter headers, whatever the arm's record mode: without the task fingerprint Mubit
  cannot route it to an adapter, so "fall back to Opus" is a provider answer and not the same
  request a second time (``FALLBACK_RECORD``);
- ``x-middleware-request-id`` captured; latency and the four usage meters recorded;
- ``stop_reason`` max_tokens / refusal count as failed calls;
- per-step **Mubit project routing** (D38): ``data/mubit_projects.json`` maps (arm, step) to a
  project (A / B) and each project to the *name* of the env var holding its key. The project
  used is stored on every record (``LLMCallRecord.mubit_project``). Keys are read with
  ``dotenv_values`` (env first, then the git-ignored ``.env``) and are never printed or logged.

``MubitClient.call`` sends exactly one HTTP request (SDK auto-retries disabled) and never
raises. ``call_with_fallback`` is the agent-loop policy: retry once on an invalid reply,
then fall back to ``claude-opus-5-5`` (records flagged ``is_retry`` / ``is_fallback``).
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from backend.agent.llm_base import (
    DEV_MODEL,
    MUBIT_ANTHROPIC_BASE_URL,
    OPUS_MODEL,
    RECORD_HEADER,
    REQUEST_ID_HEADER,
    LLMClient,
    estimate_cost_usd,
)
from backend.agent.prompts.build import split_user
from backend.core.types import ArmKind, LLMCallRecord, StepType

T = TypeVar("T", bound=BaseModel)

REPO_ROOT = Path(__file__).resolve().parents[2]
MUBIT_KEY_ENV = "MUBIT_API_KEY"
FAILED_STOP_REASONS = frozenset({"max_tokens", "refusal"})
MUBIT_BASE_PREFIX = "https://api.mubit.ai/"

PROJECTS_CONFIG_PATH = REPO_ROOT / "data" / "mubit_projects.json"
_ENV_VAR_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_HEADER_NAME = re.compile(r"^x-[a-z0-9-]+$")  # lower-case x- header; never a record/auth header

RecordMode = Literal["full", "metadata", "off"]
RECORD_MODES: frozenset[str] = frozenset({"full", "metadata", "off"})


def record_mode(record: bool | str) -> RecordMode:
    """``True`` -> ``"full"``, ``False`` -> ``"off"``; a header value passes through."""
    if isinstance(record, bool):
        return "full" if record else "off"
    if record in RECORD_MODES:
        return record  # type: ignore[return-value]
    raise ValueError(f"record must be a bool or one of {sorted(RECORD_MODES)}, got {record!r}")


def record_headers(record: bool | str) -> dict[str, str]:
    """Extra headers for a recording mode. ``full`` is Mubit's default, so it sends none."""
    mode = record_mode(record)
    return {} if mode == "full" else {RECORD_HEADER: mode}


@dataclass(frozen=True)
class StepConfig:
    effort: str          # output_config.effort: low | medium | high | xhigh | max
    max_tokens: int      # backstop; hitting it is a failed call (budget.md lever 3)


DEFAULT_STEP_CONFIGS: dict[StepType, StepConfig] = {
    StepType.propose_experiments: StepConfig(effort="medium", max_tokens=8000),
    StepType.diagnose_run: StepConfig(effort="low", max_tokens=4000),
    StepType.judge_improvement: StepConfig(effort="low", max_tokens=4000),
    StepType.early_stop: StepConfig(effort="low", max_tokens=4000),  # D34: latency-critical classification
}

# Arm -> provider model ID as sent. The mubit arm sends the SAME ID as the opus arm; Mubit
# routes to an adapter when one is live. ``dev=True`` swaps both for Haiku 5.5.
DEFAULT_ARM_MODELS: dict[ArmKind, str] = {
    ArmKind.opus: OPUS_MODEL,
    ArmKind.opus_bm: OPUS_MODEL,   # "Busy Machines with Opus": Opus on every step, like the opus arm
    ArmKind.mubit: OPUS_MODEL,
    # Ablation only (rl-planner-ablation): the same loop on a smaller, faster provider model.
    # A smaller faster model in the same loop, NOT a Mubit adapter.
    ArmKind.fast: DEV_MODEL,
}

# LLMCallRecord.cost_source. The mubit arm sends the Opus model ID, but Mubit may answer with an
# adapter whose price we do not see: its cost is an estimate at Opus list price, tagged as such.
COST_SOURCE_LIST = "estimate@list"
COST_SOURCE_MUBIT = "estimate@opus_list"


def load_mubit_api_key(env_file: Path | None = None, env_var: str = MUBIT_KEY_ENV) -> str:
    """The key in ``env_var`` (default MUBIT_API_KEY) from the environment, else from the repo's
    git-ignored ``.env``.

    Reads only that one key (``dotenv_values``, not ``load_dotenv``), so no other secret in
    ``.env`` is loaded into the process environment. Errors name the variable, never a key."""
    key = os.environ.get(env_var, "").strip()
    if key:
        return key
    from dotenv import dotenv_values

    path = env_file or REPO_ROOT / ".env"
    if path.exists():
        key = (dotenv_values(path).get(env_var) or "").strip()
    if not key:
        raise RuntimeError(f"{env_var} is not set (env or {path})")
    return key


# --------------------------------------------------------------------------- project routing (D38)


@dataclass(frozen=True)
class ProjectRouting:
    """Which Mubit project answers which (arm, step). ``keys`` maps a project to the NAME of the
    env var holding its key (never the key itself). Unlisted (arm, step) pairs use ``default``."""

    default: str
    routes: Mapping[str, Mapping[str, str]]   # arm value -> step value -> project
    keys: Mapping[str, str]                   # project -> env var name
    source: str = "built-in"
    # arm value -> step value -> extra request headers. For Mubit's announced "adapter id in a
    # header" (Discord, 24 Oct: not shipped yet); only Mubit arms may pin an adapter.
    adapter_headers: Mapping[str, Mapping[str, Mapping[str, str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.default not in self.keys:
            raise ValueError(f"default project {self.default!r} has no key entry")
        for arm, steps in self.adapter_headers.items():
            if ArmKind(arm) is not ArmKind.mubit:
                raise ValueError(f"adapter headers on arm {arm!r}: only the mubit arm may pin an adapter")
            for step, hdrs in steps.items():
                StepType(step)
                for h, v in hdrs.items():
                    if (not _HEADER_NAME.match(h) or h in (RECORD_HEADER, "x-middleware-store", "x-api-key")
                            or not v or any(c.isspace() for c in v)):
                        raise ValueError(f"adapter header {arm}/{step}: bad header {h!r}")
        for name, var in self.keys.items():
            if not _ENV_VAR_NAME.match(var):
                # Guards against a real key pasted into the (committed) JSON file.
                raise ValueError(f"project {name!r}: keys must be env var names (A-Z, 0-9, _)")
        for arm, steps in self.routes.items():
            ArmKind(arm)  # a typo in an arm or step name would silently misroute
            for step, project in steps.items():
                StepType(step)
                if project not in self.keys:
                    raise ValueError(f"route {arm}/{step} -> {project!r}: unknown project")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], source: str = "dict") -> ProjectRouting:
        extra = set(data) - {"default", "routes", "keys", "adapter_headers"}
        if extra:
            raise ValueError(f"unknown keys in project routing: {sorted(extra)}")
        return cls(
            default=str(data["default"]),
            routes={str(a): {str(s): str(p) for s, p in steps.items()}
                    for a, steps in (data.get("routes") or {}).items()},
            keys={str(p): str(v) for p, v in data["keys"].items()},
            source=source,
            adapter_headers={str(a): {str(s): {str(h): str(v) for h, v in hdrs.items()}
                                      for s, hdrs in steps.items()}
                             for a, steps in (data.get("adapter_headers") or {}).items()},
        )

    def headers_for(self, arm: ArmKind | str, step: StepType | str) -> dict[str, str]:
        return dict(self.adapter_headers.get(ArmKind(arm).value, {}).get(StepType(step).value, {}))

    def project_for(self, arm: ArmKind | str, step: StepType | str) -> str:
        return self.routes.get(ArmKind(arm).value, {}).get(StepType(step).value, self.default)

    def arm_map(self, arm: ArmKind | str) -> dict[str, str]:
        """step -> project for one arm, every step listed."""
        return {s.value: self.project_for(arm, s) for s in StepType}

    def to_dict(self) -> dict[str, Any]:
        out = {"default": self.default,
               "routes": {a: dict(steps) for a, steps in self.routes.items()},
               "keys": dict(self.keys), "source": self.source}
        if self.adapter_headers:
            out["adapter_headers"] = {a: {s: dict(h) for s, h in steps.items()}
                                      for a, steps in self.adapter_headers.items()}
        return out


# The D38 rule, used only if data/mubit_projects.json is missing (the file is the source of truth).
DEFAULT_PROJECT_ROUTING = ProjectRouting(
    default="A",
    routes={ArmKind.mubit.value: {StepType.judge_improvement.value: "B", StepType.early_stop.value: "B"}},
    keys={"A": MUBIT_KEY_ENV, "B": "MUBIT_API_KEY_B"},
)


def load_project_routing(path: Path | str | None = None) -> ProjectRouting:
    """Project routing from ``data/mubit_projects.json`` (or ``path``). A missing default file
    falls back to the built-in D38 rule; a missing explicit ``path`` or a malformed file raises."""
    p = Path(path) if path is not None else PROJECTS_CONFIG_PATH
    if not p.exists():
        if path is not None:
            raise FileNotFoundError(p)
        return DEFAULT_PROJECT_ROUTING
    try:
        source = p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        source = str(p)
    return ProjectRouting.from_dict(json.loads(p.read_text(encoding="utf-8")), source=source)


def mubit_projects_manifest(clients: Mapping[Any, Any]) -> dict[str, Any]:
    """Fairness-manifest block: per arm, which Mubit project each step goes to.

    ``clients`` maps an arm (``ArmKind`` or its value) to that arm's LLM client; a client without
    project routing (mock, replay, Optuna's ``None``) maps to ``None``. Env var names only, never
    keys. Routing differs per arm by design (D38), so it sits next to ``differs_only_in``."""
    per_arm: dict[str, dict[str, str] | None] = {}
    configs: dict[str, dict[str, Any]] = {}
    for arm, client in clients.items():
        arm_v = arm.value if isinstance(arm, ArmKind) else str(arm)
        routing = getattr(client, "projects", None)
        if not isinstance(routing, ProjectRouting):
            per_arm[arm_v] = None
            continue
        per_arm[arm_v] = routing.arm_map(arm_v)
        configs[arm_v] = routing.to_dict()
    cvals = list(configs.values())
    return {"per_arm": per_arm,
            "config": cvals[0] if cvals else None,
            "config_identical": all(c == cvals[0] for c in cvals),
            "note": "D38: project B holds the closed-output judge v1 / early_stop tasks; A everything else"}


def _sanitize_schema(schema: Any) -> Any:
    """Make a JSON schema acceptable to structured outputs: drop numeric / length / array-size
    constraints and titles, close every object. Validation of the dropped constraints is
    Pydantic's job."""
    drop = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
            "minLength", "maxLength", "minItems", "maxItems", "title", "default"}
    if isinstance(schema, dict):
        out = {k: _sanitize_schema(v) for k, v in schema.items() if k not in drop}
        if out.get("type") == "object" or "properties" in out:
            out["additionalProperties"] = False
            out.setdefault("required", list(out.get("properties", {})))
        return out
    if isinstance(schema, list):
        return [_sanitize_schema(x) for x in schema]
    return schema


def schema_from_model(model: type[BaseModel]) -> dict[str, Any]:
    """Wire schema from a Pydantic model (fine for diagnose/judge; propose needs the
    per-target schema from ``prompts.build.propose_wire_schema``)."""
    return _sanitize_schema(model.model_json_schema())


def _text_of(message: Any) -> str | None:
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) == "text":
            return block.text
    return None


def _usage(message: Any) -> tuple[int, int, int, int]:
    u = getattr(message, "usage", None)
    if u is None:
        return 0, 0, 0, 0
    return (
        int(getattr(u, "input_tokens", 0) or 0),
        int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        int(getattr(u, "cache_read_input_tokens", 0) or 0),
        int(getattr(u, "output_tokens", 0) or 0),
    )


def parse_and_validate(
    text: str | None,
    schema: type[T],
    validator: Callable[[T], T | None] | None = None,
) -> tuple[T | None, str | None]:
    """JSON text -> validated model, or (None, error). Never raises."""
    if text is None:
        return None, "no text block in reply"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e.msg} at {e.pos}"
    try:
        parsed = schema.model_validate(data)
    except ValidationError as e:
        return None, "schema: " + "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()[:5]
        )
    if validator is not None:
        try:
            res = validator(parsed)
        except Exception as e:  # noqa: BLE001 - never raise; any validator error = invalid reply
            return None, f"validator: {type(e).__name__}: {e}"
        if res is not None:
            parsed = res
    return parsed, None


class MubitClient:
    """LLMClient over the Mubit Anthropic gateway. See module docstring."""

    benchmark_safe = True  # real calls; MockLLM sets False
    # Record mode of every fallback attempt (manifest: llm_policy.fallback_record). "off" drops the
    # task fingerprint, so Mubit cannot route the fallback to an adapter (bench.py, configs.py).
    fallback_record: RecordMode = "off"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = MUBIT_ANTHROPIC_BASE_URL,
        dev: bool = False,
        arm_models: Mapping[ArmKind, str] | None = None,
        step_configs: Mapping[StepType, StepConfig] | None = None,
        fallback_model: str = OPUS_MODEL,
        timeout_s: float = 120.0,
        http_client: Any | None = None,
        projects: ProjectRouting | Path | str | None = None,
        project_keys: Mapping[str, str] | None = None,
    ) -> None:
        """``api_key`` is the DEFAULT project's key (back-compat); other projects' keys come from
        ``project_keys`` or, lazily on first use, from the env var named in the routing config."""
        if not base_url.startswith(MUBIT_BASE_PREFIX):
            # Hard rule: every LLM call goes through Mubit. No exemption, not even with a
            # custom http_client (tests use a fake transport on the real Mubit URL).
            raise ValueError(f"base_url must be the Mubit gateway, got {base_url!r}")
        models = dict(DEFAULT_ARM_MODELS)
        if dev:
            models = {arm: DEV_MODEL for arm in models}
        if arm_models:
            models.update(arm_models)
        self.arm_models: dict[ArmKind, str] = models
        self.step_configs: dict[StepType, StepConfig] = {**DEFAULT_STEP_CONFIGS, **(step_configs or {})}
        self.fallback_model = fallback_model
        self.base_url = base_url
        self.projects: ProjectRouting = (
            projects if isinstance(projects, ProjectRouting) else load_project_routing(projects)
        )
        self._timeout_s = timeout_s
        self._http_client = http_client
        self._project_keys: dict[str, str] = dict(project_keys or {})
        default = self.projects.default
        key = api_key or self._project_keys.get(default) or load_mubit_api_key(env_var=self.projects.keys[default])
        self._project_keys[default] = key
        self._client = self._make_sdk_client(key)
        self._clients: dict[str, Any] = {default: self._client}

    def _make_sdk_client(self, api_key: str) -> Any:
        return anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=self.base_url,  # checked against MUBIT_BASE_PREFIX in __init__
            timeout=self._timeout_s,
            max_retries=0,  # exactly one HTTP request per call(); retry policy lives in the loop
            http_client=self._http_client,
        )

    def project_for(self, arm: ArmKind | str, step: StepType | str) -> str:
        return self.projects.project_for(arm, step)

    def _client_for(self, project: str) -> Any:
        """The SDK client for a project, built on first use. Raises (caught in ``call``) if the
        key is missing or equals another project's key (a misrouted race must not look routed)."""
        client = self._clients.get(project)
        if client is not None:
            return client
        key = self._project_keys.get(project) or load_mubit_api_key(env_var=self.projects.keys[project])
        for other, other_key in self._project_keys.items():
            if other != project and other_key == key:
                raise RuntimeError(f"key for project {project} ({self.projects.keys[project]}) "
                                   f"equals project {other}'s key")
        self._project_keys[project] = key
        client = self._clients[project] = self._make_sdk_client(key)
        return client

    def model_for(self, arm: ArmKind | str) -> str:
        arm = ArmKind(arm)
        if arm not in self.arm_models:
            raise ValueError(f"no model configured for arm {arm.value!r}")
        return self.arm_models[arm]

    def build_request(
        self,
        step: StepType,
        system: str,
        user: str,
        wire_schema: dict[str, Any],
        session_id: str,
        model: str,
        effort: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        """The exact kwargs for ``messages.create`` (exposed for tests and the probe)."""
        prefix, rest = split_user(user)
        content: list[dict[str, Any]] = []
        if prefix is not None and prefix.strip():
            content.append({"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}})
        content.append({"type": "text", "text": rest})
        return {
            "model": model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": wire_schema}},
            "metadata": {"user_id": session_id},
        }

    async def call(
        self,
        step: StepType,
        system: str,
        user: str,
        schema: type[T],
        arm: ArmKind,
        session_id: str,
        record: bool | RecordMode = False,
        *,
        wire_schema: dict[str, Any] | None = None,
        validator: Callable[[T], T | None] | None = None,
        model: str | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        is_retry: bool = False,
        is_fallback: bool = False,
        experiment_ids: list[str] | None = None,
        arm_id: str | None = None,
    ) -> tuple[T | None, LLMCallRecord]:
        step = StepType(step)
        arm = ArmKind(arm)
        cfg = self.step_configs[step]
        if is_fallback:
            # The fallback must reach the provider model: the first try and the retry were this
            # same request, and an adapter that answered them badly must not answer it again.
            # Record off (no task fingerprint: not routable) and no adapter-pinning header.
            headers = record_headers(self.fallback_record)
        else:
            try:
                headers = record_headers(record)
            except ValueError:
                headers = {RECORD_HEADER: "off"}  # unknown mode: fail safe (never record by accident)
            headers.update(self.projects.headers_for(arm, step))
        rec = LLMCallRecord(
            arm=arm,
            arm_id=arm_id,
            session_id=session_id,
            step_type=step,
            model=model or self.arm_models.get(arm, OPUS_MODEL),
            recorded=RECORD_HEADER not in headers,  # True only for a full recording
            is_retry=is_retry,
            is_fallback=is_fallback,
            experiment_ids=list(experiment_ids or []),
            mubit_project=self.project_for(arm, step),
        )
        try:
            sdk = self._client_for(rec.mubit_project)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001 - never raise from call(); never fall back to another project
            rec.error = f"mubit project {rec.mubit_project}: {e}"[:300]
            return None, rec
        if wire_schema is None and step is StepType.propose_experiments:
            # ProposeOutput.config is a free dict; the closed per-target schema must be passed.
            rec.error = "propose_experiments needs wire_schema=prompts.build.propose_wire_schema(spec)"
            return None, rec
        try:
            kwargs = self.build_request(
                step, system, user, wire_schema or schema_from_model(schema), session_id,
                rec.model, effort or cfg.effort, max_tokens or cfg.max_tokens,
            )
        except Exception as e:  # noqa: BLE001 - never raise from call()
            rec.error = f"request build failed: {e}"
            return None, rec

        t0 = time.perf_counter()
        try:
            raw = await sdk.messages.with_raw_response.create(**kwargs, extra_headers=headers)
            rec.mubit_request_id = raw.headers.get(REQUEST_ID_HEADER)
            message = await raw.parse()
        except anthropic.APIStatusError as e:
            rec.latency_ms = (time.perf_counter() - t0) * 1000
            rec.mubit_request_id = e.response.headers.get(REQUEST_ID_HEADER) if e.response is not None else None
            rec.error = f"http {e.status_code}: {str(e.message)[:300]}"
            return None, rec
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            rec.latency_ms = (time.perf_counter() - t0) * 1000
            rec.error = f"transport: {type(e).__name__}: {e}"[:300]
            return None, rec
        except Exception as e:  # noqa: BLE001 - never raise from call()
            rec.latency_ms = (time.perf_counter() - t0) * 1000
            rec.error = f"unexpected: {type(e).__name__}: {e}"[:300]
            return None, rec
        rec.latency_ms = (time.perf_counter() - t0) * 1000

        rec.input_tokens, rec.cache_creation_tokens, rec.cache_read_tokens, rec.output_tokens = _usage(message)
        rec.cost_usd = estimate_cost_usd(
            rec.model, rec.input_tokens, rec.cache_creation_tokens, rec.cache_read_tokens, rec.output_tokens
        )
        rec.cost_source = COST_SOURCE_MUBIT if arm == ArmKind.mubit else COST_SOURCE_LIST
        rec.stop_reason = getattr(message, "stop_reason", None)
        if rec.stop_reason in FAILED_STOP_REASONS:
            rec.error = f"stop_reason={rec.stop_reason}"
            return None, rec

        parsed, err = parse_and_validate(_text_of(message), schema, validator)
        rec.valid = parsed is not None
        rec.error = err
        return parsed, rec

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.close()


async def call_with_fallback(
    client: LLMClient,
    step: StepType,
    system: str,
    user: str,
    schema: type[T],
    arm: ArmKind,
    session_id: str,
    record: bool | RecordMode = False,
    *,
    fallback_model: str = OPUS_MODEL,
    **kwargs: Any,
) -> tuple[T | None, list[LLMCallRecord]]:
    """Agent-loop policy (architecture.md section 1): try; on an invalid reply retry once with
    the same model; if still invalid, fall back to ``fallback_model`` (Opus) through Mubit, sent
    with record off and no adapter headers (``MubitClient.call`` with ``is_fallback``).
    Every attempt is its own record and counts toward the arm's time and $. Same policy on
    every arm (fairness)."""
    records: list[LLMCallRecord] = []
    attempts: list[dict[str, Any]] = [
        {},
        {"is_retry": True},
        {"is_fallback": True, "model": fallback_model},
    ]
    for extra in attempts:
        parsed, rec = await client.call(  # type: ignore[call-arg]
            step, system, user, schema, arm, session_id, record, **kwargs, **extra
        )
        records.append(rec)
        if parsed is not None:
            return parsed, records
    return None, records


def is_benchmark_safe(client: object) -> bool:
    """Benchmark runs must reject mock / replay clients."""
    return bool(getattr(client, "benchmark_safe", False))


# --------------------------------------------------------------------------- key check (dashboard)

KEY_CHECK_MODEL = DEV_MODEL   # the cheapest model: one call with max_tokens=1 costs a fraction of a cent


def _scrub(text: str, secret: str) -> str:
    """An error message without the key (or any 8+ character slice of it)."""
    out = text.replace(secret, "[key]") if secret else text
    if len(secret) >= 16:
        for i in range(0, len(secret) - 8):
            out = out.replace(secret[i:i + 8], "[key]")
    return out[:300]


async def check_mubit_key(api_key: str, *, model: str = KEY_CHECK_MODEL, timeout_s: float = 20.0,
                          http_client: Any | None = None) -> dict[str, Any]:
    """Test a Mubit key with the smallest possible call: one message, ``max_tokens=1``, the cheapest
    model, ``x-middleware-record: off`` (never an example), through the Mubit gateway only.

    Returns ``{ok, latency_ms, model, request_id, error}`` and never raises; the key is never
    returned, logged or kept (the SDK client is closed before returning)."""
    key = (api_key or "").strip()
    out: dict[str, Any] = {"ok": False, "latency_ms": None, "model": model, "request_id": None, "error": None}
    if not key:
        out["error"] = "no key given"
        return out
    client = anthropic.AsyncAnthropic(api_key=key, base_url=MUBIT_ANTHROPIC_BASE_URL, timeout=timeout_s,
                                      max_retries=0, http_client=http_client)
    t0 = time.perf_counter()
    try:
        raw = await client.messages.with_raw_response.create(
            model=model, max_tokens=1, messages=[{"role": "user", "content": "ping"}],
            extra_headers={RECORD_HEADER: "off"})
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000)
        out["request_id"] = raw.headers.get(REQUEST_ID_HEADER)
        msg = raw.parse()
        out["model"] = getattr(msg, "model", None) or model
        out["ok"] = True
    except anthropic.APIStatusError as e:
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000)
        reason = {401: "Mubit rejected the key (401 unauthorized)",
                  403: "the key is not allowed to call this model (403 forbidden)"}.get(e.status_code)
        out["error"] = reason or _scrub(f"Mubit answered {e.status_code}: {e.message}", key)
        try:
            out["request_id"] = e.response.headers.get(REQUEST_ID_HEADER)
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001 - network errors, timeouts
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000)
        out["error"] = _scrub(f"could not reach Mubit: {type(e).__name__}: {e}", key)
    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass
    return out


__all__ = [
    "KEY_CHECK_MODEL",
    "check_mubit_key",
    "COST_SOURCE_LIST",
    "COST_SOURCE_MUBIT",
    "DEFAULT_ARM_MODELS",
    "DEFAULT_PROJECT_ROUTING",
    "DEFAULT_STEP_CONFIGS",
    "MubitClient",
    "PROJECTS_CONFIG_PATH",
    "ProjectRouting",
    "RECORD_MODES",
    "RecordMode",
    "StepConfig",
    "call_with_fallback",
    "is_benchmark_safe",
    "load_mubit_api_key",
    "load_project_routing",
    "mubit_projects_manifest",
    "parse_and_validate",
    "record_headers",
    "record_mode",
    "schema_from_model",
]
