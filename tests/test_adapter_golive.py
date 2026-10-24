"""Adapter go-live kit: scripts/push_judge_traffic.py and scripts/eval_served_calls.py.

No network, no paid call: the real MubitClient talks to a fake httpx2 transport on the Mubit URL,
and the dry-run path uses the script's mock. The keys below are test strings, not real keys.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import httpx2
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_served_calls as esc  # noqa: E402
import push_judge_traffic as pjt  # noqa: E402

from backend.agent.llm_base import OPUS_MODEL, RECORD_HEADER, REQUEST_ID_HEADER  # noqa: E402
from backend.agent.mubit_client import MubitClient, ProjectRouting  # noqa: E402
from backend.agent.prompts import load_system_prompt, prompt_fingerprint  # noqa: E402
from backend.agent.prompts.build import judge_wire_schema  # noqa: E402
from backend.core.types import StepType  # noqa: E402

SECRET = "test-secret-key-workspace-C-0123456789"
STEP = StepType.judge_improvement
SYSTEM = load_system_prompt(STEP, "v1")
SCHEMA = judge_wire_schema("v1")


def _cases(sources: tuple[str, ...] = ("trap_test",), n: int = 4) -> list[dict[str, Any]]:
    return pjt.load_cases(sources)[:n]


def _reply(req_id: str, verdict: str = "invalid", code: str = "contamination_or_eval_leak",
           in_tok: int = 3000, out_tok: int = 30) -> httpx2.Response:
    body = {"id": "msg_t", "type": "message", "role": "assistant", "model": OPUS_MODEL,
            "content": [{"type": "text", "text": json.dumps({"verdict": verdict, "reason_code": code})}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": in_tok, "output_tokens": out_tok,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}
    return httpx2.Response(200, json=body, headers={REQUEST_ID_HEADER: req_id})


class Gateway:
    def __init__(self, script: list[str] | None = None) -> None:
        self.script = list(script or [])   # "503" / "429" / "ok", consumed in order; then "ok"
        self.requests: list[httpx2.Request] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        n = len(self.requests)
        step = self.script.pop(0) if self.script else "ok"
        if step in ("503", "429"):
            return httpx2.Response(int(step), json={"type": "error", "error": {"type": "overloaded_error",
                                                                                "message": "engine_unavailable"}},
                                   headers={REQUEST_ID_HEADER: f"mreq_err{n:04d}"})
        return _reply(f"mreq_ok{n:04d}")

    def client(self, workspace: str = "C") -> MubitClient:
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler))
        routing = ProjectRouting(default=workspace, routes={}, keys={workspace: pjt.WORKSPACE_KEYS[workspace]})
        return MubitClient(api_key=SECRET, projects=routing, http_client=http)

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


async def _no_sleep(_: float) -> None:
    return None


def _push(cases: list[dict[str, Any]], client: Any, tmp_path: Path, **kw: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    out = tmp_path / "traffic.jsonl"
    kw.setdefault("record", "metadata")
    summary = asyncio.run(pjt.push(cases, client, workspace="C", out_path=out, system=SYSTEM, wire_schema=SCHEMA,
                                   fingerprint=prompt_fingerprint(STEP), sleep=_no_sleep, log=lambda s: None, **kw))
    return summary, pjt._jsonl(out)


# --------------------------------------------------------------------------- frozen prompt / inputs


def test_judge_v1_fingerprint_is_unchanged() -> None:
    assert prompt_fingerprint(STEP, "v1") == pjt.EXPECTED_FINGERPRINT == "5c1d3cfd2941"
    system, schema = pjt._rjv1().preflight()
    assert system == SYSTEM and schema == SCHEMA


def test_sources_have_ground_truth_and_opus_reference() -> None:
    jv1 = pjt.load_cases(["jv1"])
    assert len(jv1) == 177 and all(c["in_training_pool"] for c in jv1)
    assert all(c["ground_truth"] for c in jv1) and all(c["opus_ref"] for c in jv1)
    held = pjt.load_cases(["trap_test", "noise_test"])
    assert len(held) == 60 and not any(c["in_training_pool"] for c in held)
    assert all(c["opus_ref"] and c["opus_ref"]["reason_code"] for c in held)
    # held-out messages are never among the 177 recorded inputs
    assert not {c["msg_sha256"] for c in held} & {c["msg_sha256"] for c in jv1}
    # dedup across sources: trap_dev is inside jv1
    assert len(pjt.load_cases(["jv1", "trap_dev"])) == 177


def test_trap_noise_family_is_noise_not_fake() -> None:
    trap = {r["id"]: r for r in pjt._jsonl(pjt.TRAP_SET)}
    assert pjt.gt_trap(trap["trap_v1_048"])["truth"] == "noise"       # noise_within_spread
    assert pjt.gt_trap(trap["trap_v1_020"])["truth"] == "fake"        # eval_on_train_levels
    assert pjt.gt_trap(trap["trap_v1_024"])["truth"] == "real"


# --------------------------------------------------------------------------- record guards and headers


def test_record_guards() -> None:
    held, jv1 = _cases(("trap_test",)), _cases(("jv1",))
    assert pjt.record_problems("metadata", held + jv1, yes=False, allow_duplicates=False) == []
    assert pjt.record_problems("off", held, yes=False, allow_duplicates=False) == []
    errs = pjt.record_problems("full", held, yes=True, allow_duplicates=True)
    assert any("held-out" in e for e in errs)
    assert any("--yes" in e for e in pjt.record_problems("full", jv1, yes=False, allow_duplicates=True))
    assert any("duplicate" in e for e in pjt.record_problems("full", jv1, yes=True, allow_duplicates=False))
    assert pjt.record_problems("full", jv1, yes=True, allow_duplicates=True) == []
    assert pjt.record_problems("bogus", jv1, yes=True, allow_duplicates=True)


@pytest.mark.parametrize("record,header", [("metadata", "metadata"), ("off", "off"), ("full", None)])
def test_record_mode_header_on_the_wire(record: str, header: str | None, tmp_path: Path) -> None:
    gw = Gateway()
    summary, rows = _push(_cases(("jv1",), 2), gw.client(), tmp_path, record=record)
    assert len(gw.requests) == 2 and summary["valid_calls"] == 2
    for req, body in zip(gw.requests, gw.bodies()):
        assert str(req.url).startswith("https://api.mubit.ai/anthropic")
        assert req.headers.get(RECORD_HEADER) == header
        assert body["model"] == OPUS_MODEL
        assert body["system"][0]["text"] == SYSTEM
        assert body["output_config"]["format"]["schema"] == SCHEMA
    assert all(r["record"] == record and r["prompt_fingerprint"] == "5c1d3cfd2941" for r in rows)
    assert all(r["recorded_full"] == (record == "full") for r in rows)
    if record != "full":
        assert all(r["session_id"].startswith("golive-") for r in rows)


def test_v2_full_sessions_one_per_case() -> None:
    c = {"source": "v2", "session_n": 7, "case_id": "jv1v2_race_007"}
    assert pjt.session_for(c, "full", "C", "s", "2026-10-25") == "collect-judgev1v2-2026-10-24/race/7"
    assert pjt.session_for(c, "metadata", "C", "s", "2026-10-25") == "golive-2026-10-25/C/metadata/s"


def test_adapter_header_only_with_flag(tmp_path: Path) -> None:
    gw = Gateway()
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(gw.handler))
    routing = ProjectRouting(default="C", routes={}, keys={"C": "MUBIT_API_KEY_C"},
                             adapter_headers={"mubit": {"judge_improvement": {"x-mubit-adapter": "adp_123"}}})
    client = MubitClient(api_key=SECRET, projects=routing, http_client=http)
    _push(_cases(("jv1",), 1), client, tmp_path)
    assert gw.requests[0].headers.get("x-mubit-adapter") == "adp_123"
    gw2 = Gateway()
    _push(_cases(("jv1",), 1), gw2.client(), tmp_path / "b")
    assert "x-mubit-adapter" not in gw2.requests[0].headers


# --------------------------------------------------------------------------- backoff, cap, concurrency


def test_backoff_on_503_and_429(tmp_path: Path) -> None:
    gw = Gateway(["503", "429", "ok"])
    slept: list[float] = []

    async def sleep(s: float) -> None:
        slept.append(s)

    out = tmp_path / "t.jsonl"
    summary = asyncio.run(pjt.push(_cases(("jv1",), 1), gw.client(), workspace="C", record="metadata", out_path=out,
                                   system=SYSTEM, wire_schema=SCHEMA, fingerprint="5c1d3cfd2941", sleep=sleep,
                                   log=lambda s: None))
    rows = pjt._jsonl(out)
    assert [r["http_status"] for r in rows] == [503, 429, None]
    assert rows[-1]["valid"] and summary["valid_calls"] == 1 and summary["backoffs"] == 2
    assert len(slept) == 2 and slept[0] < slept[1] <= pjt.BACKOFF_MAX_S
    assert rows[0]["mubit_fee_usd_est"] == 0.0 and rows[-1]["mubit_fee_usd_est"] == pjt.DEFAULT_MUBIT_FEE_USD


def test_backoff_gives_up(tmp_path: Path) -> None:
    gw = Gateway(["503"] * 50)
    summary, rows = _push(_cases(("jv1",), 1), gw.client(), tmp_path, max_backoff_retries=2, invalid_retries=0)
    assert len(rows) == 3 and summary["valid_calls"] == 0


def test_hard_cap_is_never_exceeded(tmp_path: Path) -> None:
    gw = Gateway()   # every reply: 3000 in / 30 out at Opus list = $0.0126
    cap = 0.06
    summary, rows = _push(_cases(("jv1",), 30), gw.client(), tmp_path, usd_cap=cap, mubit_fee_usd=0.005)
    spent = sum(r["cost_usd"] + r["mubit_fee_usd_est"] for r in rows)
    assert summary["stopped_by_cap"] is True
    assert 0 < len(rows) < 30 and spent <= cap + 1e-9


def test_budget_reserves_before_first_cost() -> None:
    b = pjt.Budget(0.03, fee_usd=0.005)
    r1 = b.try_reserve()
    assert r1 == pytest.approx(pjt.FIRST_RESERVE_USD + 0.005)
    assert b.try_reserve() is None  # a second concurrent reservation would overshoot
    b.settle(r1, 0.004, answered=True)
    assert b.spent == pytest.approx(0.009)


def test_concurrency_is_capped_at_4(tmp_path: Path) -> None:
    class Slow(pjt.MockJudgeClient):
        inflight = 0
        peak = 0

        async def call(self, *a: Any, **kw: Any) -> Any:
            Slow.inflight += 1
            Slow.peak = max(Slow.peak, Slow.inflight)
            await asyncio.sleep(0.01)
            Slow.inflight -= 1
            return await super().call(*a, **kw)

    _push(_cases(("jv1",), 20), Slow({}, err_rate=0.0), tmp_path, concurrency=10)
    assert Slow.peak == 4


# --------------------------------------------------------------------------- dry run CLI, keys


def test_dry_run_cli_uses_the_mock_and_never_reads_a_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("MUBIT_API_KEY_C", SECRET)

    def boom(*a: Any, **kw: Any) -> Any:
        raise AssertionError("dry run must not build a real client")

    monkeypatch.setattr(pjt, "make_client", boom)
    rc = pjt.main(["push", "--dry-run", "--sources", "trap_test,noise_test", "--out-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert SECRET not in out and "DRY RUN" in out
    traffic = next(tmp_path.glob("traffic-*.jsonl"))
    rows = pjt._jsonl(traffic)
    assert rows and all(r["dry_run"] and r["cost_usd"] == 0 for r in rows)
    assert all(r["mubit_request_id"].startswith("mock_") for r in rows)
    assert SECRET not in traffic.read_text(encoding="utf-8")
    ids = traffic.with_suffix(".request_ids.txt").read_text(encoding="utf-8").split()
    assert len(ids) == len({r["case_id"] for r in rows if r["valid"]}) and all(i in out for i in ids)
    served = traffic.with_suffix(".served_by_mubit.mock.txt")
    assert served.exists()
    summary = json.loads(traffic.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["dry_run"] and summary["llm_cost_usd_est"] == 0
    # the eval script reads the dry-run files end to end
    assert esc.main([str(traffic), "--served-ids", str(served)]) == 0
    assert esc.main([str(traffic)]) == 0
    assert SECRET not in capsys.readouterr().out


def test_cli_refuses_full_without_yes_and_on_held_out(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert pjt.main(["push", "--dry-run", "--sources", "trap_test", "--record", "full", "--yes",
                     "--out-dir", str(tmp_path)]) == 2
    assert pjt.main(["push", "--dry-run", "--sources", "v2", "--record", "full", "--out-dir", str(tmp_path)]) == 2
    assert pjt.main(["push", "--dry-run", "--sources", "v2", "--record", "full", "--yes", "--repeat", "2",
                     "--out-dir", str(tmp_path)]) == 2
    assert not list(tmp_path.glob("traffic-*.jsonl"))


def test_paid_push_needs_yes_and_never_targets_a(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **kw: Any) -> Any:
        raise AssertionError("no client may be built without --yes")

    monkeypatch.setattr(pjt, "make_client", boom)
    assert pjt.main(["push", "--sources", "trap_test", "--out-dir", str(tmp_path)]) == 2
    assert not list(tmp_path.glob("traffic-*"))
    with pytest.raises(SystemExit):  # argparse rejects workspace A
        pjt.main(["push", "--dry-run", "--workspace", "A", "--out-dir", str(tmp_path)])
    monkeypatch.undo()
    with pytest.raises(SystemExit) as e:
        pjt.make_client("A")  # refused before any key is read
    assert "Opus baseline" in str(e.value)


def test_runs_in_the_same_second_get_separate_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import datetime as dt

    fixed = dt.datetime(2026, 10, 24, 12, 0, 0, tzinfo=dt.UTC)
    monkeypatch.setattr(pjt, "utc_now", lambda: fixed)
    for _ in range(2):
        assert pjt.main(["push", "--dry-run", "--sources", "trap_test", "--limit", "3", "--out-dir", str(tmp_path)]) == 0
    files = sorted(tmp_path.glob("traffic-*.jsonl"))
    assert len(files) == 2 and all(len(pjt._jsonl(f)) >= 3 for f in files)
    for f in files:
        s = json.loads(f.with_suffix(".summary.json").read_text(encoding="utf-8"))
        assert s["planned_calls"] == 3 and s["valid_calls"] == 3


def test_make_client_refuses_shared_keys_without_printing_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUBIT_API_KEY_C", SECRET)
    monkeypatch.setenv("MUBIT_API_KEY_B", SECRET)
    monkeypatch.setenv("MUBIT_API_KEY", "test-other-key-a")  # never falls through to the real .env
    with pytest.raises(SystemExit) as e:
        pjt.make_client("C")
    assert "MUBIT_API_KEY_C" in str(e.value) and SECRET not in str(e.value)


def test_make_client_routes_only_to_the_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUBIT_API_KEY_C", SECRET)
    monkeypatch.setenv("MUBIT_API_KEY_B", "test-other-key-b")
    monkeypatch.setenv("MUBIT_API_KEY", "test-other-key-a")
    c = pjt.make_client("C", {"x-mubit-adapter": "adp_1"})
    try:
        assert c.project_for("mubit", "judge_improvement") == "C" and c.project_for("opus", "judge_improvement") == "C"
        assert c.projects.headers_for("mubit", "judge_improvement") == {"x-mubit-adapter": "adp_1"}
        assert SECRET not in json.dumps(c.projects.to_dict())
    finally:
        asyncio.run(c.aclose())


# --------------------------------------------------------------------------- v2 batch


def test_v2_batch_is_new_and_labelled() -> None:
    rows = pjt._jsonl(pjt.V2_INPUTS)
    assert len(rows) >= 25
    known = {r["msg_sha256"] for r in pjt._jsonl(pjt.JV1_INPUTS)}
    held = {pjt._sha(r["user_message"]) for r in pjt._jsonl(pjt.TRAP_SET) + pjt._jsonl(pjt.NOISE_SET)
            if r["split"] != "dev"}
    shas = [pjt._sha(r["user_message"]) for r in rows]
    assert len(set(shas)) == len(rows) and not set(shas) & known and not set(shas) & held
    assert all(r["msg_sha256"] == s for r, s in zip(rows, shas))
    assert len({r["id"] for r in rows}) == len({r["session_id"] for r in rows}) == len(rows)
    assert all(r["ground_truth"]["truth"] in ("fake", "not_fake") for r in rows)
    assert {r["ground_truth"]["truth"] for r in rows} == {"fake", "not_fake"}
    # R71 recorded the v2 batch full into B (data/adapter_golive/B/traffic-*.jsonl): in B's pool, not C's
    in_b, in_c = pjt.load_cases(["v2"], workspace="B"), pjt.load_cases(["v2"], workspace="C")
    assert len(in_b) == len(in_c) == len(rows)
    assert all(c["in_training_pool"] and "B" in c["recorded_in"] for c in in_b)
    assert not any(c["in_training_pool"] for c in in_c)
    assert pjt.record_problems("full", in_c, yes=True, allow_duplicates=False, workspace="C") == []
    errs = pjt.record_problems("full", in_b, yes=True, allow_duplicates=False, workspace="B")
    assert any("duplicate" in e and "workspace B" in e for e in errs) and not any("--sources v2" in e for e in errs)
    assert pjt.record_problems("full", in_b, yes=True, allow_duplicates=True, workspace="B") == []


def test_cli_blocks_re_recording_v2_into_b(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert pjt.main(["push", "--dry-run", "--workspace", "B", "--sources", "v2", "--record", "full", "--yes",
                     "--out-dir", str(tmp_path)]) == 2
    assert "already recorded full into workspace B" in capsys.readouterr().err
    assert not list(tmp_path.glob("traffic-*.jsonl"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_recorded_full_ledger_is_per_workspace(tmp_path: Path) -> None:
    rec, root = tmp_path / "record", tmp_path / "golive"
    _write_jsonl(rec / "inputs.jsonl", [{"id": "j1", "user_message": "m1"}, {"id": "j2", "user_message": "m2"}])
    _write_jsonl(rec / "calls.first.B.record-full.jsonl", [{"case_id": "j1", "project": "B", "record": "full",
                                                            "recorded": True, "valid": False}])
    _write_jsonl(rec / "calls.hedge.C.record-full.jsonl", [{"case_id": "j2", "project": "C", "record": "full",
                                                            "recorded": True, "valid": True}])
    _write_jsonl(rec / "calls.recheck.A.record-off.jsonl", [{"case_id": "j2", "project": "A", "record": "off",
                                                             "recorded": False}])
    base = {"workspace": "C", "dry_run": False}
    _write_jsonl(root / "C" / "traffic-1.jsonl", [
        {**base, "msg_sha256": "s_full", "record": "full", "recorded_full": True},
        {**base, "msg_sha256": "s_meta", "record": "metadata", "recorded_full": False},
        {**base, "msg_sha256": "s_notrec", "record": "full", "recorded_full": False},
        {**base, "msg_sha256": "s_dry", "record": "full", "recorded_full": True, "dry_run": True}])
    _write_jsonl(root / "C" / "dryrun" / "traffic-2.jsonl",
                 [{**base, "msg_sha256": "s_sub", "record": "full", "recorded_full": True}])
    led = pjt.recorded_full_ledger(root=root, record_dir=rec)
    assert led == {"B": {pjt._sha("m1")}, "C": {pjt._sha("m2"), "s_full"}}


def test_real_ledger_has_the_177_in_b_and_c_and_v2_in_b() -> None:
    led = pjt.recorded_full_ledger()
    jv1 = {r["msg_sha256"] for r in pjt._jsonl(pjt.JV1_INPUTS)}
    v2 = {pjt._sha(r["user_message"]) for r in pjt._jsonl(pjt.V2_INPUTS)}
    assert len(jv1) == 177 and jv1 <= led["B"] and jv1 <= led["C"]
    assert v2 <= led["B"] and not v2 & led["C"]
    held = {c["msg_sha256"] for c in pjt.load_cases(["trap_test", "noise_test"])}
    assert not held & set().union(*led.values())
    for ws in ("B", "C"):
        assert all(c["in_training_pool"] for c in pjt.load_cases(["jv1", "trap_dev", "noise_dev"], workspace=ws))


def test_select_v2_round_robin_is_deterministic() -> None:
    cands = [{"target": t, "expected": {"is_trap": trap, "recorded_v0_verdict": v},
              "msg_sha256": hashlib.sha256(f"{t}{trap}{v}{i}".encode()).hexdigest()}
             for t in ("rl", "cifar") for trap, v in ((True, "invalid"), (False, "unsure")) for i in range(10)]
    cands.append({"target": "rl", "expected": {"is_trap": False, "recorded_v0_verdict": "valid"}, "msg_sha256": "rare"})
    a, b = pjt.select_v2(cands, 9), pjt.select_v2(list(reversed(cands)), 9)
    assert [x["msg_sha256"] for x in a] == [x["msg_sha256"] for x in b]
    assert "rare" in {x["msg_sha256"] for x in a}  # every stratum is covered first


def test_try_models_cases_are_held_out_with_opus_answers(tmp_path: Path) -> None:
    res = pjt.export_try_models(tmp_path)
    roles = {c["role"]: c for c in res["cases"]}
    assert set(roles) == {"fake_caught", "real_gain", "noise"}
    assert all(c["split"] == "test" for c in roles.values())
    assert roles["fake_caught"]["opus_answer_R40"]["verdict"] == "invalid"
    assert roles["real_gain"]["opus_answer_R40"]["verdict"] == "valid"
    assert roles["noise"]["opus_answer_R40"]["reason_code"] == "gain_within_noise"
    assert (tmp_path / "system_prompt.judge_improvement.v1.txt").read_text(encoding="utf-8") == SYSTEM


# --------------------------------------------------------------------------- eval_served_calls


def _row(cid: str, rid: str, verdict: str, code: str, lat: float, *, truth: str = "fake", pool: bool = False,
         ref: tuple[str, str] | None = ("invalid", "contamination_or_eval_leak"), set_: str = "trap") -> dict[str, Any]:
    acc = {"fake": ["invalid"], "real": ["valid"], "noise": ["unsure", "invalid"], "not_fake": ["valid", "unsure"]}[truth]
    return {"case_id": cid, "source": "trap_test", "valid": True, "mubit_request_id": rid, "verdict": verdict,
            "reason_code": code, "latency_ms": lat, "in_training_pool": pool, "workspace": "C", "record": "metadata",
            "prompt_fingerprint": "5c1d3cfd2941",
            "ground_truth": {"set": set_, "truth": truth, "acceptable": acc},
            "opus_ref": {"verdict": ref[0], "reason_code": ref[1]} if ref else None}


def test_eval_split_by_served_by() -> None:
    rows = [
        _row("t1", "mreq_a1", "invalid", "contamination_or_eval_leak", 300),
        _row("t2", "mreq_a2", "valid", "clean_gain", 320),                         # fake let through
        _row("t1", "mreq_o1", "invalid", "contamination_or_eval_leak", 1700),
        _row("t3", "mreq_o2", "valid", "clean_gain", 1800, truth="real", ref=("valid", "clean_gain")),
        _row("t4", "mreq_o3", "unsure", "gain_within_noise", 1650, truth="noise", pool=True,
             ref=("unsure", "gain_within_noise"), set_="noise"),
        {**_row("t5", "mreq_bad", "invalid", "x", 50), "valid": False},   # no reply: failed, counted wrong
    ]
    rep = esc.report(rows, {"mreq_a1", "mreq_a2", "mreq_missing"})
    adp, opus = rep["groups"]["mubit_model"]["all"], rep["groups"]["opus"]["all"]
    assert adp["calls"] == 2 and adp["correct"] == 1 and adp["by_set"]["trap"]["fake_let_through"] == "1/2"
    assert adp["agree_with_opus_ref"]["verdict_and_code_count"] == "1/2"
    assert opus["calls"] == 4 and opus["failed"] == 1 and opus["correct"] == 3 and opus["accuracy"] == 0.75
    assert opus["valid_only"]["accuracy"] == 1.0 and "VALID ANSWERS ONLY" in opus["valid_only"]["label"]
    assert rep["calls"] == 6 and rep["valid_calls"] == 5 and rep["failed_calls"] == 1
    assert rep["groups"]["opus"]["training_pool"]["calls"] == 1
    assert rep["n_served_ids_not_in_traffic"] == 1
    assert rep["paired_same_cases"]["cases"] == 1  # t1 answered by both
    assert adp["latency_p50_ms"] < opus["latency_p50_ms"]
    assert rep["disagreements_with_opus_ref"][0]["case_id"] == "t2"
    assert "Mubit model" in esc.format_report(rep)


def test_invalid_adapter_replies_count_as_wrong() -> None:
    """An adapter call with an unusable reply (tokens spent, no reason_code) is a miss, stays in the
    denominators, and its id is known (not 'not in traffic'), as Mubit grades it."""
    bad = {**_row("t2", "mreq_a2", "valid", "clean_gain", 300), "valid": False, "verdict": None, "reason_code": None,
           "error": "no reason_code", "input_tokens": 3300, "output_tokens": 12}
    retry = _row("t2", "mreq_o9", "invalid", "contamination_or_eval_leak", 1700)   # Opus answers the retry
    rows = [_row("t1", "mreq_a1", "invalid", "contamination_or_eval_leak", 300), bad, retry]
    rep = esc.report(rows, {"mreq_a1", "mreq_a2"})
    assert rep["n_served_ids_not_in_traffic"] == 0
    adp = rep["groups"]["mubit_model"]["all"]
    assert adp["calls"] == 2 and adp["invalid"] == 1 and adp["failed"] == 0
    assert adp["correct"] == 1 and adp["graded"] == 2 and adp["accuracy"] == 0.5
    assert adp["by_set"]["trap"]["fake_caught"] == "1/2" and adp["by_set"]["trap"]["no_valid_answer"] == 1
    assert adp["agree_with_opus_ref"]["verdict_and_code_count"] == "1/2" and adp["agree_with_opus_ref"]["verdict"] == 0.5
    assert adp["valid_only"]["accuracy"] == 1.0 and adp["valid_only"]["agree_verdict_and_code"] == 1.0
    assert adp["latency_p50_ms"] == 300.0                   # latency from valid answers only
    assert rep["disagreements_with_opus_ref"][0]["outcome"] == "invalid"
    text = esc.format_report(rep)
    assert "invalid 1" in text and "valid answers only, optimistic" in text
    # overall mode keeps it too
    ov = esc.report(rows, None)["overall"]["all"]
    assert ov["calls"] == 3 and ov["invalid"] == 1 and ov["correct"] == 2


def test_read_ids_from_pasted_console_text(tmp_path: Path) -> None:
    p = tmp_path / "ids.txt"
    p.write_text("Served by: Mubit model  mreq_01a16d543b287c13b3345b1bbb17aaf2, 1.2s\n"
                 "mreq_01a16d543acf71e1b7611e7bd21bc8ba\tjudge\n mock_adp_0123abcd\n", encoding="utf-8")
    assert esc.read_ids(p) == {"mreq_01a16d543b287c13b3345b1bbb17aaf2", "mreq_01a16d543acf71e1b7611e7bd21bc8ba",
                               "mock_adp_0123abcd"}


def test_bimodal_latency_is_only_a_hint() -> None:
    fast = [300 + 5 * i for i in range(20)]
    slow = [1600 + 20 * i for i in range(60)]
    h = esc.bimodal_hint(fast + slow)
    assert h and h["fast_calls"] == 20 and "HINT ONLY" in h["label"]
    assert esc.bimodal_hint([1500 + 10 * i for i in range(80)]) is None
    assert esc.bimodal_hint([300, 1700]) is None
    rows = [_row(f"c{i}", f"mreq_{i}", "invalid", "contamination_or_eval_leak", x) for i, x in enumerate(fast + slow)]
    rep = esc.report(rows, None)
    assert rep["mode"] == "overall" and rep["latency_hint"]["fast_calls"] == 20
    assert "HINT ONLY" in esc.format_report(rep)
