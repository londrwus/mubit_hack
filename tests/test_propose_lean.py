"""Lean propose prompts (v2/v3 diff form): selectable by config, default and frozen v1 unchanged."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import workers.targets  # noqa: F401  (registers targets)
from backend.agent.loop import LLMResearcher, RepoPrompts, load_prompts
from backend.agent.mubit_client import parse_and_validate
from backend.agent.prompts import CURRENT_VERSION, FROZEN, prompt_fingerprint
from backend.agent.prompts import build as B
from backend.core.types import ArmKind, LLMCallRecord, ProposeOutput, StepType
from backend.scheduler import ExperimentSnapshot, ResearchDigest
from backend.core.types import ExperimentStatus
from workers.targets.base import get_target

SPEC = get_target("rl").spec


def test_v1_default_and_fingerprint_unchanged() -> None:
    assert CURRENT_VERSION[StepType.propose_experiments] == "v1"
    assert FROZEN[StepType.propose_experiments] == "v1"
    assert prompt_fingerprint(StepType.propose_experiments) == "fdda8c5b0919"
    assert prompt_fingerprint(StepType.judge_improvement) == "5c1d3cfd2941"
    p = RepoPrompts()
    assert p.propose_version == "v1"
    assert "propose_experiments=v1:fdda8c5b0919" in p.version
    assert p.propose_schema() is None
    # v1 wire schema is exactly the frozen one (config + hypothesis, 20 words)
    ws = p.wire_schema(StepType.propose_experiments, SPEC)
    assert ws == B.propose_wire_schema(SPEC)
    item = ws["properties"]["experiments"]["items"]
    assert set(item["properties"]) == {"config", "hypothesis"}
    assert item["properties"]["hypothesis"]["description"] == "At most 20 words."


def test_v2_selectable_and_distinct() -> None:
    p = load_prompts("v2")
    assert isinstance(p, RepoPrompts) and p.propose_version == "v2"
    assert p.system(StepType.propose_experiments) != RepoPrompts().system(StepType.propose_experiments)
    assert "propose_experiments=v2:" in p.version
    # other steps keep their current prompts
    assert p.system(StepType.judge_improvement) == RepoPrompts().system(StepType.judge_improvement)
    item = p.wire_schema(StepType.propose_experiments, SPEC)["properties"]["experiments"]["items"]
    assert set(item["properties"]) == {"changes", "hypothesis"}
    assert item["properties"]["hypothesis"]["description"] == "At most 10 words."
    assert p.propose_schema() is B.ProposeDiffOutput
    assert "6 words" in load_prompts("v3").wire_schema(StepType.propose_experiments, SPEC)[
        "properties"]["experiments"]["items"]["properties"]["hypothesis"]["description"]


def _rows() -> list[B.ResultRow]:
    d = SPEC.default_config()
    best = d | {"lr": 0.001, "gamma": 0.95}
    return [B.ResultRow(label="#1", config=d, score=-300.0),
            B.ResultRow(label="#2", config=best, score=-160.0),
            B.ResultRow(label="#3", config=d | {"lr": 0.01}, score=-250.0)]


def test_expand_diff_against_current_best() -> None:
    rows = _rows()
    base = B.propose_base_config(SPEC, rows)
    assert base == rows[1].config  # leaderboard top = the 'current best config' line
    assert B.propose_base_config(SPEC, []) == SPEC.default_config()
    text = json.dumps({"experiments": [
        {"changes": "gamma=0.97", "hypothesis": "Longer horizon helps credit assignment."},
        {"changes": "lr=0.0005, hidden_size=128", "hypothesis": "Wider net, gentler steps."},
        {"changes": "not_a_param=3", "hypothesis": "bad"},
        {"changes": "lr=5", "hypothesis": "out of range"},
    ]})
    out, err = parse_and_validate(text, B.ProposeDiffOutput, B.propose_validator(SPEC, 8, base=base))
    assert err is None and isinstance(out, ProposeOutput)
    assert len(out.experiments) == 2  # invalid ones dropped, the rest kept (same rule as v1)
    c0, c1 = (e.config for e in out.experiments)
    assert c0 == SPEC.validate_config(base | {"gamma": 0.97})
    assert c1["lr"] == 0.0005 and c1["hidden_size"] == 128 and c1["gamma"] == base["gamma"]
    # every expanded config is a full config
    assert set(c0) == set(SPEC.config_space)


def test_long_hypothesis_trimmed_not_rejected() -> None:
    long = " ".join(["word"] * 40)
    out = B.ProposeDiffOutput.model_validate({"experiments": [{"changes": "gamma=0.97", "hypothesis": long}]})
    assert len(out.experiments[0].hypothesis.split()) <= 25


class _FakeLLM:
    """Returns a fixed diff reply; records what it was asked."""

    benchmark_safe = False

    def __init__(self, text: str) -> None:
        self.text = text
        self.seen: dict[str, Any] = {}

    async def call(self, step, system, user, schema, arm, session_id, record=False, **kw):  # type: ignore[no-untyped-def]
        self.seen = {"system": system, "schema": schema, "wire": kw.get("wire_schema")}
        parsed, err = parse_and_validate(self.text, schema)
        rec = LLMCallRecord(arm=arm, session_id=session_id, step_type=step, model="fake", valid=parsed is not None,
                            error=err)
        return parsed, rec


def _snap(i: int, cfg: dict[str, Any], score: float) -> ExperimentSnapshot:
    return ExperimentSnapshot(id=f"e{i}", label=f"#{i}", config=cfg, status=ExperimentStatus.succeeded, score=score)


def test_researcher_expands_diff_end_to_end() -> None:
    rows = _rows()
    hist = [_snap(i + 1, r.config, r.score or 0.0) for i, r in enumerate(rows)]
    digest = ResearchDigest(k=2, n_slots=4, elapsed_s=10, remaining_s=800, history=hist)
    llm = _FakeLLM(json.dumps({"experiments": [{"changes": "gamma=0.97", "hypothesis": "Longer horizon."},
                                               {"changes": "num_envs=32", "hypothesis": "More envs."}]}))
    r = LLMResearcher(kind=ArmKind.opus, llm=llm, spec=SPEC, session_id="t", prompts=load_prompts("v2"))
    props = asyncio.run(r.propose(digest, 2))
    assert llm.seen["schema"] is B.ProposeDiffOutput
    assert set(llm.seen["wire"]["properties"]["experiments"]["items"]["properties"]) == {"changes", "hypothesis"}
    assert len(props) == 2
    assert props[0].config == SPEC.validate_config(rows[1].config | {"gamma": 0.97})
    assert props[1].config["num_envs"] == 32 and props[1].config["gamma"] == rows[1].config["gamma"]


def test_researcher_v1_unchanged() -> None:
    d = SPEC.default_config()
    settings = ", ".join(f"{k}={json.dumps(v)}" for k, v in (d | {"gamma": 0.97}).items())
    llm = _FakeLLM(json.dumps({"experiments": [{"config": settings, "hypothesis": "Longer horizon."}]}))
    r = LLMResearcher(kind=ArmKind.opus, llm=llm, spec=SPEC, session_id="t", prompts=load_prompts())
    props = asyncio.run(r.propose(ResearchDigest(k=1, n_slots=1, elapsed_s=0, remaining_s=900), 1))
    assert llm.seen["schema"] is ProposeOutput
    assert len(props) == 1 and props[0].config["gamma"] == 0.97
