"""early_stop (D34) as a 4th step, without touching the three frozen steps.

1. The three frozen steps are byte-identical: prompt fingerprints, raw prompt files, registry
   entries, step configs, user-message builders and wire schemas (golden sha256 on a fixed
   synthetic spec, verified against the pre-D34 build.py).
2. The early_stop prompt / builder / schema / output model.
3. scripts/collect_early_stop.py: case building (context, labels, median rule) on a synthetic
   read-only DB, the policy evaluation, the D39 record gate.
No network.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import httpx2
import pytest

from backend.agent.llm_base import OPUS_MODEL, RECORD_HEADER, REQUEST_ID_HEADER
from backend.agent.mubit_client import DEFAULT_STEP_CONFIGS, MubitClient, StepConfig
from backend.agent.prompts import CURRENT_VERSION, FROZEN, PROMPTS_DIR, load_system_prompt, prompt_fingerprint
from backend.agent.prompts.build import (
    CACHE_BREAK,
    CurvePoint,
    EarlyStopState,
    JudgeClaim,
    ResultRow,
    build_diagnose_user,
    build_early_stop_user,
    build_judge_user,
    build_propose_user,
    downsample_curve,
    early_stop_wire_schema,
    judge_wire_schema,
    split_user,
    target_spec_block,
    wire_schema_for,
)
from backend.core.types import STEP_OUTPUT_SCHEMAS, ArmKind, EarlyStopOutput, StepType, StopDecision
from workers.targets.base import ParamSpec, TargetSpec

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- 1. frozen steps

FROZEN_FINGERPRINTS = {
    StepType.propose_experiments: "fdda8c5b0919",
    StepType.diagnose_run: "22ecb45368e3",
    StepType.judge_improvement: "623366123052",
}
FROZEN_VERSIONS = {StepType.propose_experiments: "v1", StepType.diagnose_run: "v0", StepType.judge_improvement: "v0"}
FROZEN_FILE_SHA256 = {
    "propose_experiments.v1.md": "15dc0ce389e1cd5cfd4ade51730b1a77b9b8ea70873c2725fe46857b712c30e4",
    "diagnose_run.v0.md": "a62df6a759a4b4ceff02d06bf5b6993fa54f1e665d7f4820a8920fbf793d1fbb",
    "judge_improvement.v0.md": "f9708f2b33415540834d64fb17968594907b54fd02c94b3f3095dfe2d1f85c05",
}
# sha256 of [propose user, diagnose user, judge user, propose / diagnose / judge wire schema]
# for the fixed inputs below; computed with build.py before the early_stop section existed.
GOLDEN_BUILDERS = [
    "b5f070e6d62c3cb8e6237572786b5e0745faba0b83edfa0798ea041c2700d05d",
    "02cb66ed0641b7579bb27c1d30fd8625bdc41ff7b33e6d2b5038f387e44dec4a",
    "37c43ae88591fc2b14ff42f06feec54536c9a82414645b1d923f5840793caee2",
    "10f29ba09bdb8403a67ff65f0778328bde84bc10a35f391020ffa179d423d186",
    "74058ea713dd0640e657b106155dc77129ffeb2d11cea9b588bbc7e0406fe420",
    "0cc297af8b121594c3b3335cb760de9c87e8fc1a6aaff8a16cd9a2383cea2bff",
]


@pytest.fixture
def toy() -> TargetSpec:
    return TargetSpec(
        name="toy",
        description="A tiny MLP regression task for tests.",
        metric="val_loss",
        direction="min",
        target_value=0.05,
        hardware="CPU",
        budget_s=20,
        config_space={
            "lr": ParamSpec(type="float", low=1e-4, high=1e-1, log=True, default=1e-3, description="learning rate"),
            "width": ParamSpec(type="int", low=8, high=256, default=32, description="hidden width"),
            "act": ParamSpec(type="categorical", choices=["relu", "gelu", "tanh"], default="relu", description="activation"),
            "val_from_train_pool": ParamSpec(type="bool", default=False, description="draw validation rows from the training pool"),
        },
        trap_option="val_from_train_pool",
    )


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("step", list(FROZEN_FINGERPRINTS))
def test_three_frozen_fingerprints_unchanged(step: StepType) -> None:
    assert prompt_fingerprint(step, FROZEN_VERSIONS[step]) == FROZEN_FINGERPRINTS[step]
    if step is StepType.judge_improvement:
        # D37: judge moved to the closed-output v1 (tests/test_judge_v1.py); v0 stays as history.
        assert CURRENT_VERSION[step] == FROZEN[step] == "v1"
    else:
        assert CURRENT_VERSION[step] == FROZEN_VERSIONS[step] and FROZEN[step] == FROZEN_VERSIONS[step]


@pytest.mark.parametrize("name", list(FROZEN_FILE_SHA256))
def test_three_frozen_prompt_files_untouched(name: str) -> None:
    assert hashlib.sha256((PROMPTS_DIR / name).read_bytes()).hexdigest() == FROZEN_FILE_SHA256[name]


def test_three_frozen_step_configs_unchanged() -> None:
    assert DEFAULT_STEP_CONFIGS[StepType.propose_experiments] == StepConfig(effort="medium", max_tokens=8000)
    assert DEFAULT_STEP_CONFIGS[StepType.diagnose_run] == StepConfig(effort="low", max_tokens=4000)
    assert DEFAULT_STEP_CONFIGS[StepType.judge_improvement] == StepConfig(effort="low", max_tokens=4000)


def test_three_frozen_builders_and_schemas_byte_identical(toy: TargetSpec) -> None:
    hist = [ResultRow(label=f"#{i + 1}", config=toy.default_config(), score=0.5 - 0.01 * i if i != 3 else None,
                      status="succeeded" if i != 3 else "failed", note="diverged" if i == 3 else None, hypothesis="h")
            for i in range(12)]
    outs = [
        build_propose_user(toy, hist, 3, running=[toy.default_config()], time_left_s=123.4),
        build_diagnose_user(toy, toy.default_config(), "failed", "boom", ["a", "b nan"], {"loss": 1.23456}),
        build_judge_user(toy, JudgeClaim(prev_best_score=0.3, prev_best_config=toy.default_config(), new_score=0.25,
                                         new_config=toy.default_config(), evidence={"x": 1})),
        json.dumps(wire_schema_for(StepType.propose_experiments, toy), sort_keys=True),
        json.dumps(wire_schema_for(StepType.diagnose_run), sort_keys=True),
        json.dumps(judge_wire_schema("v0"), sort_keys=True),  # v0 schema unchanged (current is v1, D37)
    ]
    assert [_sha(o) for o in outs] == GOLDEN_BUILDERS


# --------------------------------------------------------------------------- 2. early_stop step

EARLY_STOP_FINGERPRINT = "f1f3bf21aa79"   # v2, decision only (D39)
GOLDEN_EARLY_STOP_USER = "db16c350cb5af923d54a6cfde9189ca1263d4419bef130a5494c005ef2069200"


def _state(toy: TargetSpec, n_points: int = 21) -> EarlyStopState:
    return EarlyStopState(config=toy.default_config(), curve_metric="val_loss", elapsed_s=10.0, budget_s=20.0,
                          curve=[CurvePoint(step=10 * i, t=0.5 * i, value=1.0 / (1 + i)) for i in range(n_points)],
                          n_completed=5, best_final=0.04321, median_final=0.0612345, median_at_point=0.0911,
                          best_run_at_point=None)


def test_early_stop_registered_frozen_and_cacheable() -> None:
    s = StepType.early_stop
    assert CURRENT_VERSION[s] == FROZEN[s] == "v2"
    assert prompt_fingerprint(s) == EARLY_STOP_FINGERPRINT
    text = load_system_prompt(s)
    assert len(text) / 4 >= 600 and "<!--" not in text and "\r" not in text
    assert "reason" not in text.split("## What you return")[1].split("## Style")[0]  # decision only
    assert DEFAULT_STEP_CONFIGS[s] == StepConfig(effort="low", max_tokens=4000)
    assert STEP_OUTPUT_SCHEMAS[s] is EarlyStopOutput


def test_superseded_versions_kept_unrecorded() -> None:
    # v0 (free-text reason) and v1 (reason_code) stay on disk; neither is the current version.
    for v in ("v0", "v1"):
        assert (PROMPTS_DIR / f"early_stop.{v}.md").exists()
        assert prompt_fingerprint(StepType.early_stop, v) != EARLY_STOP_FINGERPRINT


def test_early_stop_builder_layout_and_determinism(toy: TargetSpec) -> None:
    u = build_early_stop_user(toy, _state(toy))
    assert _sha(u) == GOLDEN_EARLY_STOP_USER
    assert u == build_early_stop_user(toy, _state(toy))
    prefix, rest = split_user(u)
    assert prefix == target_spec_block(toy)        # same cached spec block as the other steps
    assert rest.startswith("RUN IN PROGRESS") and "COMPLETED RUNS IN THIS RACE" in rest
    rows = [ln for ln in rest.splitlines() if ln.startswith("  ")]
    assert len(rows) == 12 and rows[0] == "  0 0 1" and rows[-1] == "  200 10 0.04762"  # first + latest kept
    assert "best run had n/a" in rest and "(50%)" in rest
    for leak in ("heldout", "held-out", "held out", "trap", "is_trap", "label"):
        assert leak not in rest.lower()
    with pytest.raises(ValueError):
        build_early_stop_user(toy, _state(toy, 0))


def test_downsample_curve() -> None:
    pts = [CurvePoint(step=i, t=float(i), value=float(i)) for i in range(57)]
    d = downsample_curve(pts)
    assert len(d) == 12 and d[0] == pts[0] and d[-1] == pts[-1]
    assert downsample_curve(pts[:5]) == pts[:5]


def test_early_stop_wire_schema_and_output() -> None:
    assert early_stop_wire_schema() == {
        "type": "object", "properties": {"decision": {"type": "string", "enum": ["continue", "stop"]}},
        "required": ["decision"], "additionalProperties": False}
    assert wire_schema_for(StepType.early_stop) == early_stop_wire_schema()
    assert EarlyStopOutput.model_validate({"decision": "stop"}).decision is StopDecision.stop
    assert EarlyStopOutput.model_validate({"decision": "continue"}).decision is StopDecision.continue_
    with pytest.raises(ValueError):
        EarlyStopOutput.model_validate({"decision": "stop", "reason": "x"})
    with pytest.raises(ValueError):
        EarlyStopOutput.model_validate({"decision": "maybe"})


async def test_early_stop_request_through_mubit(toy: TargetSpec) -> None:
    seen: list[httpx2.Request] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        seen.append(req)
        body = {"id": "m", "type": "message", "role": "assistant", "model": OPUS_MODEL,
                "content": [{"type": "text", "text": json.dumps({"decision": "stop"})}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 300, "output_tokens": 40, "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 1800}}
        return httpx2.Response(200, json=body, headers={REQUEST_ID_HEADER: "mreq_es"})

    c = MubitClient(api_key="k", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    out, rec = await c.call(StepType.early_stop, load_system_prompt(StepType.early_stop),
                            build_early_stop_user(toy, _state(toy)), EarlyStopOutput, ArmKind.opus, "race-r/opus/seed-0",
                            wire_schema=wire_schema_for(StepType.early_stop))
    assert out is not None and out.decision is StopDecision.stop and rec.valid and rec.mubit_request_id == "mreq_es"
    assert rec.step_type is StepType.early_stop and not rec.recorded
    req = seen[0]
    assert str(req.url).startswith("https://api.mubit.ai/anthropic/") and req.headers[RECORD_HEADER] == "off"
    body = json.loads(req.content)
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["schema"] == early_stop_wire_schema()
    assert CACHE_BREAK.strip() not in json.dumps(body)
    await c.aclose()


# --------------------------------------------------------------------------- 3. collection script


@pytest.fixture
def ces() -> Any:
    name = "collect_early_stop_under_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "collect_early_stop.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_db(path: Path) -> None:
    """One rl race: 3 clean runs + 1 probe + 1 trap finish first; then a winner and a loser."""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE experiments (id TEXT, race_id TEXT, started_at REAL, ended_at REAL, config_json TEXT,
            metrics_json TEXT, is_trap INTEGER, judge_verdict TEXT, status TEXT, score REAL);
        CREATE TABLE collection_probes (experiment_id TEXT, kind TEXT);
        CREATE TABLE metrics (experiment_id TEXT, step INTEGER, t REAL, name TEXT, value REAL);
    """)
    runs = [("A", 0.0, -300.0, 0, None), ("B", 0.0, -250.0, 0, None), ("C", 0.0, -200.0, 0, None),
            ("P", 0.0, -350.0, 0, "diverge"), ("T", 0.0, -100.0, 1, None),
            ("W", 100.0, -150.0, 0, None), ("L", 100.0, -400.0, 0, None)]
    for rid, start, final, trap, probe in runs:
        con.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (rid, "R", start, start + 33.0, json.dumps({"lr": 0.001}),
                     json.dumps({"train_s": 30.0, "heldout_score": final}), trap,
                     "invalid" if trap else None, "succeeded", final))
        if probe:
            con.execute("INSERT INTO collection_probes VALUES (?,?)", (rid, probe))
        for k in range(31):   # training starts at launch t=2 s, one point per second
            v = -1000.0 + (final + 1000.0) * k / 30
            con.execute("INSERT INTO metrics VALUES (?,?,?,?,?)", (rid, k, 2.0 + k, "train_return", v))
    con.commit()
    con.close()


def test_build_cases_context_labels_and_median_rule(ces: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_db(tmp_path / "collect_rl.db")
    monkeypatch.setattr(ces, "COLLECTION_DIR", tmp_path)
    cases = {c["key"]: c for c in ces.build_cases("rl")}
    # only W and L are cases (A/B/C have < 3 completed runs before them; P is a probe; T a trap)
    assert set(cases) == {"rl:W:25", "rl:W:50", "rl:L:25", "rl:L:50"}
    w, lo = cases["rl:W:25"], cases["rl:L:25"]
    st = w["state"]
    assert st["n_completed"] == 4                       # A, B, C, P; the trap run is excluded
    assert st["best_final"] == pytest.approx(-200.0)    # not the trap's -100
    assert st["elapsed_s"] == 7.5 and all(p["t"] <= 7.5 for p in st["curve"]) and len(st["curve"]) == 8
    assert w["labels"]["stop_correct"] is False and w["labels"]["beats_val"] and w["labels"]["beats_heldout"]
    assert lo["labels"]["stop_correct"] is True
    assert w["median_rule"] == "continue" and lo["median_rule"] == "stop"
    # the model-visible message never carries the labels
    msg = ces.user_for(w)
    assert "-150" not in msg.split("<<<CACHE_BREAK>>>")[1] and "stop_correct" not in msg


def test_evaluate_policy_simulation(ces: Any) -> None:
    def case(run: str, f: float, ok_stop: bool) -> dict[str, Any]:
        return {"key": f"{run}:{f}", "experiment_id": run, "target": "rl", "fraction": f, "train_time_s": 30.0,
                "state": {"budget_s": 30.0}, "labels": {"stop_correct": ok_stop}}

    cs = [case("w", 0.25, False), case("w", 0.5, False), case("l", 0.25, True), case("l", 0.5, True)]
    dec = {"w:0.25": "continue", "w:0.5": "stop", "l:0.25": "stop", "l:0.5": "stop"}
    e = ces.evaluate(cs, dec)
    assert e["accuracy"] == pytest.approx(0.75) and e["stop_but_winner"] == 1 and e["winner_runs_wrongly_stopped"] == 1
    # w stopped at 50% (15 s saved), l stopped at 25% (22.5 s saved) of 60 s
    assert e["training_s_saved"] == pytest.approx(37.5) and e["budget_saved_pct"] == pytest.approx(62.5)


async def test_record_refused_without_passing_gate(ces: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ces, "OUT_DIR", tmp_path)
    monkeypatch.setattr(ces, "CASES_PATH", tmp_path / "cases.jsonl")
    monkeypatch.setattr(ces, "CALL_FILES", {k: tmp_path / f"{k}.jsonl" for k in ces.CALL_FILES})
    (tmp_path / "cases.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="D39 gate"):
        await ces.run_phase("record", None, True, False, 4, project="b")
    fp = ces.system_prompt()[1]
    (tmp_path / f"gate_summary_{fp}.json").write_text(json.dumps({"recordable_variants": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="D39 gate"):
        await ces.run_phase("record", None, True, False, 4, project="b")
    with pytest.raises(SystemExit, match="recorded calls only"):
        await ces.run_phase("eval", None, True, False, 4, project="b")
