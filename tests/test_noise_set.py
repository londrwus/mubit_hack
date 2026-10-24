"""Tests for scripts/make_noise_set.py and scripts/eval_noise_set.py (no network, no Modal:
the LLM goes through a fake httpx2 transport on the real Mubit URL)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx2
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_noise_set as ens  # noqa: E402
import make_noise_set as mk  # noqa: E402

from backend.agent.llm_base import OPUS_MODEL, RECORD_HEADER, REQUEST_ID_HEADER  # noqa: E402
from backend.agent.mubit_client import MubitClient  # noqa: E402
from backend.agent.prompts import load_system_prompt  # noqa: E402
from backend.agent.prompts.build import CACHE_BREAK, JudgeClaim, build_judge_user, judge_wire_schema, split_user  # noqa: E402
from backend.core.types import StepType  # noqa: E402
from backend.scheduler import GROUND_TRUTH_METRIC_KEYS  # noqa: E402
from workers.targets.base import get_target  # noqa: E402

HAVE_DBS = all((mk.COLLECTION_DIR / f"collect_{t}.db").exists() for t in mk.TARGETS)
needs_dbs = pytest.mark.skipif(not HAVE_DBS, reason="collection DBs not present")


@pytest.fixture(scope="module")
def claims() -> list[dict[str, Any]]:
    if not HAVE_DBS:
        pytest.skip("collection DBs not present")
    return mk.load_all_claims(mk.TARGETS)


# --------------------------------------------------------------------------- ground truth


def test_ground_truth_real_noise_and_direction() -> None:
    prev = [-200.0 + d for d in (-3, -2, -1, 0, 1, 2, 3, -1, 1, 0)]
    big = [x + 50 for x in prev]
    small = [x + 1 for x in prev]
    assert mk.ground_truth(big, prev, "max")["label"] == "real"
    gt = mk.ground_truth(small, prev, "max")
    assert gt["label"] == "noise" and gt["gain"] == pytest.approx(1.0)
    # min metric: a higher mean is worse, never real
    assert mk.ground_truth(big, prev, "min")["label"] == "noise"
    assert mk.ground_truth([x - 50 for x in prev], prev, "min")["label"] == "real"


def test_ground_truth_needs_both_conditions_and_min_seeds() -> None:
    prev = [0.0, 10.0] * 5                     # std ~5.3
    new = [x + 11 for x in prev]               # gain 11 > 2 x 5.27, p ~ 1e-4 -> real
    assert mk.ground_truth(new, prev, "max")["label"] == "real"
    new = [x + 10 for x in prev]               # gain 10 < 2 x pooled std -> noise even though p < 0.05
    gt = mk.ground_truth(new, prev, "max")
    assert gt["welch_p"] < 0.05 and gt["label"] == "noise"
    assert mk.ground_truth(new[:5], prev, "max")["label"] == "insufficient"


def test_label_cases_uses_config_keys_and_traps_are_fake() -> None:
    clean = {"kind": "clean", "target": "rl", "candidate": {"config_key": "a"}, "previous_best": {"config_key": "b"}}
    trap = {"kind": "trap", "target": "rl", "candidate": {"config_key": "c"}, "previous_best": {"config_key": "b"}}
    reruns = ([{"config_key": "a", "seed": s, "status": "succeeded", "heldout_score": -100.0 + s % 3} for s in range(10)]
              + [{"config_key": "b", "seed": s, "status": "succeeded", "heldout_score": -300.0 + s % 3} for s in range(10)]
              + [{"config_key": "b", "seed": 99, "status": "failed", "heldout_score": None}])
    out = mk.label_cases([clean, trap], reruns)
    assert out[0]["truth"] == "real" and out[0]["ground_truth"]["n_prev"] == 10
    assert out[1]["truth"] == "fake"


# --------------------------------------------------------------------------- set extraction


@needs_dbs
def test_collection_spec_matches_pre_d29_archive() -> None:
    if not mk.ARCHIVE_PRE_D29.exists():
        pytest.skip("pre-D29 archive not present")
    for t in mk.TARGETS:
        spec_c = ens._collection_spec(mk, t)
        assert mk.check_against_archive(spec_c) is True
        assert set(spec_c.config_space) < set(get_target(t).spec.config_space)  # D29 params removed


@needs_dbs
def test_claims_cover_every_clean_judged_claim(claims: list[dict[str, Any]]) -> None:
    clean = [c for c in claims if c["kind"] == "clean"]
    assert sum(c["target"] == "rl" for c in clean) == 31       # R25: rl 31, cifar 12 clean claims
    assert sum(c["target"] == "cifar" for c in clean) == 12
    assert all(c["recorded_judge"]["verdict"] == "unsure" for c in clean)
    traps = [c for c in claims if c["kind"] == "trap"]
    assert traps and all(c["candidate"]["is_trap"] for c in traps)
    assert len({c["id"] for c in claims}) == len(claims)


@needs_dbs
def test_messages_rebuild_and_hide_ground_truth(claims: list[dict[str, Any]]) -> None:
    for t in mk.TARGETS:
        spec_c = ens._collection_spec(mk, t)
        for c in (x for x in claims if x["target"] == t):
            assert build_judge_user(spec_c, JudgeClaim(**c["claim"])) == c["user_message"]
            assert CACHE_BREAK in c["user_message"]
            ev = c["claim"]["evidence"]
            assert not (set(ev) & GROUND_TRUTH_METRIC_KEYS) and not any("heldout" in k for k in ev)
            assert c["claim"]["prev_best_score"] == pytest.approx(c["previous_best"]["score"])
    assert ens.case_problems([c | {"truth": "noise"} for c in claims]) == []


@needs_dbs
def test_splits_are_by_group(claims: list[dict[str, Any]]) -> None:
    by_group: dict[str, set[str]] = {}
    for c in claims:
        by_group.setdefault(c["group"], set()).add(c["split"])
    assert all(len(s) == 1 for s in by_group.values())
    assert {c["split"] for c in claims} == {"dev", "test"}


@needs_dbs
def test_rerun_jobs_clean_only_seed_major(claims: list[dict[str, Any]]) -> None:
    jobs = mk.rerun_jobs(claims, [100, 101])
    keys = {j["config_key"] for j in jobs}
    clean_keys = {c[r]["config_key"] for c in claims if c["kind"] == "clean" for r in ("candidate", "previous_best")}
    assert keys == clean_keys
    assert [j["seed"] for j in jobs] == [100] * len(keys) + [101] * len(keys)
    for j in jobs:
        spec = get_target(j["target"]).spec
        assert spec.validate_config(j["config"]) == j["config"]
        assert not spec.is_trap(j["config"])
        assert j["budget_s"] == {"rl": 30, "cifar": 45}[j["target"]]


# --------------------------------------------------------------------------- eval


def _case(truth: str = "noise", target: str = "rl", **kw: Any) -> dict[str, Any]:
    spec = get_target(target).spec
    prev_cfg = spec.default_config()
    new_cfg = dict(prev_cfg, lr=prev_cfg["lr"] * 2)
    claim = JudgeClaim(prev_best_score=kw.get("prev", -300.0), prev_best_config=prev_cfg,
                       new_score=kw.get("new", -290.0), new_config=new_cfg, evidence={"reported_score": -290.0})
    return {"id": kw.get("id", "ns_t_001"), "split": kw.get("split", "dev"), "target": target, "group": f"{target}/seed-0",
            "kind": "trap" if truth == "fake" else "clean", "truth": truth, "claim": claim.model_dump(mode="json"),
            "user_message": build_judge_user(spec, claim),
            "recorded_judge": {"verdict": "unsure", "total_input_tokens": 1000}}


def test_rule_is_appended_to_user_only() -> None:
    c = _case()
    u = ens.user_for("opus_rule", c)
    p0, r0 = split_user(c["user_message"])
    p1, r1 = split_user(u)
    assert p0 == p1 and r1 == r0 + "\n" + ens.RULE_TEXT
    assert ens.user_for("opus", c) == c["user_message"]


def test_rule_baselines() -> None:
    c = _case(prev=-300.0, new=-290.0)
    assert ens.rule_msg_verdict(c)[0] == "unsure"          # race evidence has no spread
    cal = {"summary": {"rl": {"default": {"heldout": {"n": 10, "mean": -966.0, "std": 167.0}},
                              "best": {"heldout": {"n": 3, "mean": -164.0, "std": 6.0}}}}}
    sigma = ens.calib_sigma(cal, "rl", -300.0)
    assert 6.0 < sigma < 167.0
    assert ens.rule_calib_verdict(c, cal)[0] == "unsure"    # +10 < 2 sigma (~2 x 33)
    big = _case(prev=-1000.0, new=-200.0)
    assert ens.rule_calib_verdict(big, cal)[0] == "valid"


def test_grade() -> None:
    assert ens.grade("real", "valid")["correct"] and ens.grade("real", "valid")["real_confirmed"]
    assert not ens.grade("real", "unsure")["correct"]
    assert ens.grade("noise", "unsure")["correct"] and ens.grade("noise", "valid")["noise_accepted"]
    assert ens.grade("fake", "invalid")["fake_caught"] and ens.grade("fake", "valid")["fake_let_through"]
    assert not ens.grade("fake", "unsure")["correct"]


class Gateway:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.requests: list[httpx2.Request] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        body = {"id": "msg_t", "type": "message", "role": "assistant", "model": OPUS_MODEL,
                "content": [{"type": "text", "text": json.dumps({"verdict": self.verdict, "reason": "r"})}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 600, "output_tokens": 50, "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 400}}
        return httpx2.Response(200, json=body, headers={REQUEST_ID_HEADER: f"mreq_{len(self.requests)}"})

    def client(self) -> MubitClient:
        return MubitClient(api_key="test-mubit-key",
                           http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler)))


def test_eval_sends_frozen_prompt_record_off_and_summarizes() -> None:
    gw = Gateway("valid")
    cases = [_case("real", id="a"), _case("noise", id="b", split="test"), _case("fake", id="c")]

    async def run() -> list[dict[str, Any]]:
        client = gw.client()
        try:
            return await ens.eval_cases(cases, client, ["opus", "opus_rule", "rule_msg"], usd_cap=1.0, concurrency=2)
        finally:
            await client.aclose()

    res = asyncio.run(run())
    assert len(gw.requests) == 6 and len(res) == 9
    for req in gw.requests:
        body = json.loads(req.content)
        assert req.headers[RECORD_HEADER] == "off"
        assert body["system"][0]["text"] == load_system_prompt(StepType.judge_improvement)
        assert body["output_config"]["format"]["schema"] == judge_wire_schema()
        assert body["model"] == OPUS_MODEL
    users = ["".join(b["text"] for b in json.loads(r.content)["messages"][0]["content"]) for r in gw.requests]
    assert sum(u.endswith(ens.RULE_TEXT) for u in users) == 3
    s = ens.summarize(res)
    a = s["opus"]["all"]
    assert (a["real_confirmed"], a["noise_accepted"], a["fake_let_through"]) == (1, 1, 1)
    assert a["tokens_match_race"] == "3/3" and a["same_verdict_as_race"] == "0/3"
    assert s["rule_msg"]["all"]["unsure_rate"] == 1.0 and s["rule_msg"]["all"]["cost_usd"] == 0.0
    assert s["opus"]["dev"]["n_graded"] == 2 and s["opus"]["test"]["n_graded"] == 1


def test_eval_respects_usd_cap() -> None:
    gw = Gateway("unsure")

    async def run() -> list[dict[str, Any]]:
        client = gw.client()
        try:
            return await ens.eval_cases([_case(id=str(i)) for i in range(5)], client, ["opus"], usd_cap=0.0)
        finally:
            await client.aclose()

    res = asyncio.run(run())
    assert not gw.requests and all(r["skipped"] == "usd_cap" and not r["graded"] for r in res)


def test_regrade_applies_current_truth_without_calls() -> None:
    saved = [{"id": "a", "arm": "opus", "split": "dev", "target": "rl", "group": "rl/seed-0", "kind": "clean",
              "truth": "insufficient", "recorded_verdict": "unsure", "recorded_total_input_tokens": 10,
              "verdict": "unsure", "reason": "r", "attempts": [], "skipped": None}]
    out = ens.regrade(saved, [_case("real", id="a"), _case("noise", id="zz")])
    assert len(out) == 1 and out[0]["truth"] == "real" and not out[0]["correct"]
    assert ens.summarize(out)["opus"]["all"]["real"] == 1
