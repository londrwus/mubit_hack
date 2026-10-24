"""Review fix: bench.py's live pre-flight checks every Mubit project key the race routes to.

Before the fix only MUBIT_API_KEY (project A) was checked. MubitClient loads project B's key
lazily on the first mubit-arm judge / early_stop call, so a missing MUBIT_API_KEY_B (or one equal
to A's) failed every such call at $0 mid-race, the judge policy read the missing verdict as
'unsure', and under accept_unsure a contamination fake could ship. Fake key loader only: no
real key is read, no network.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

import backend.agent.mubit_client as mc
from backend.benchmark.configs import PRESETS

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bench_script_keys", ROOT / "scripts" / "bench.py")
assert _spec and _spec.loader
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

A, B = "test-key-a", "test-key-b"


def _loader(env: dict[str, str]) -> Any:
    def load(env_file: Any = None, env_var: str = "MUBIT_API_KEY") -> str:
        if not env.get(env_var):
            raise RuntimeError(f"{env_var} is not set (env or .env)")
        return env[env_var]
    return load


def _by_name(res: list[tuple[str, bool, str]]) -> dict[str, tuple[bool, str]]:
    return {n: (ok, note) for n, ok, note in res}


MUBIT_RACE = PRESETS["demo-3m"]          # opus + mubit: the mubit judge routes to project B (D38)
NO_MUBIT_ARM = PRESETS["lcbench-wide-bm-15m"]


def test_both_routed_keys_present_and_distinct() -> None:
    res = _by_name(bench.mubit_key_checks(MUBIT_RACE, load_key=_loader({"MUBIT_API_KEY": A, "MUBIT_API_KEY_B": B})))
    assert res["MUBIT_API_KEY available"][0] and res["MUBIT_API_KEY_B available"][0]
    assert "mubit/judge_improvement" in res["MUBIT_API_KEY_B available"][1]
    assert res["Mubit project keys distinct"][0]
    assert all(A not in note and B not in note for _, note in res.values())     # keys are never printed


def test_missing_b_key_fails_the_preflight() -> None:
    res = _by_name(bench.mubit_key_checks(MUBIT_RACE, load_key=_loader({"MUBIT_API_KEY": A})))
    ok, note = res["MUBIT_API_KEY_B available"]
    assert not ok and "MUBIT_API_KEY_B is not set" in note
    assert res["MUBIT_API_KEY available"][0]


def test_b_key_equal_to_a_fails_the_preflight() -> None:
    res = _by_name(bench.mubit_key_checks(MUBIT_RACE, load_key=_loader({"MUBIT_API_KEY": A, "MUBIT_API_KEY_B": A})))
    ok, note = res["Mubit project keys distinct"]
    assert not ok and "MUBIT_API_KEY equals MUBIT_API_KEY_B" in note and A not in note


def test_a_race_without_a_project_b_route_needs_only_the_default_key() -> None:
    res = _by_name(bench.mubit_key_checks(NO_MUBIT_ARM, load_key=_loader({"MUBIT_API_KEY": A})))
    assert set(res) == {"MUBIT_API_KEY available"} and res["MUBIT_API_KEY available"][0]


def test_live_checks_gate_on_the_b_key(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MUBIT_RACE.model_copy(update={"runner": "surrogate"})   # no Modal lookup in this test
    monkeypatch.setattr(mc, "load_mubit_api_key", _loader({"MUBIT_API_KEY": A}))
    res = bench.live_checks(cfg)
    assert not all(ok for _, ok, _ in res)
    monkeypatch.setattr(mc, "load_mubit_api_key", _loader({"MUBIT_API_KEY": A, "MUBIT_API_KEY_B": B}))
    assert all(ok for _, ok, _ in bench.live_checks(cfg))


def test_a_broken_routing_fails_the_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoKey:
        default = "A"
        keys = {"A": "MUBIT_API_KEY"}

        def project_for(self, *_: Any) -> str:
            return "C"

    res = _by_name(bench.mubit_key_checks(MUBIT_RACE, routing=NoKey(), load_key=_loader({"MUBIT_API_KEY": A})))
    assert not res["Mubit project C key"][0]

    def unreadable(path: Any = None) -> Any:
        raise ValueError("malformed data/mubit_projects.json")

    monkeypatch.setattr(mc, "load_project_routing", unreadable)
    res = _by_name(bench.mubit_key_checks(MUBIT_RACE, load_key=_loader({"MUBIT_API_KEY": A})))
    assert set(res) == {"Mubit project routing"} and not res["Mubit project routing"][0]
