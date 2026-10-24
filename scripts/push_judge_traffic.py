"""Push judge_improvement v1 traffic through one Mubit workspace so a new adapter can go live.

Why: a version goes live in steps (R56): pass (quality bar 90%) -> silent test (a share of
calls, up to 100 a day) -> partial rollout (a share of calls, for at least 100 calls). Races send
~15-20 judge calls each, so without extra traffic C's adapter would barely move through those
steps before the freeze. This script replays the frozen judge v1 calls (prompt fingerprint
``5c1d3cfd2941``, Opus 5.5 model id, closed wire schema) so the task sees real traffic.

Recording modes (``--record``):
- ``metadata`` (default): the docs say *"`metadata` keeps the fingerprint but drops the content"*
  (docs.mubit.ai/reference/api#request-headers-mubit-reads, quoted in docs/mubit_office_hours.md
  item 2). The call stays in the task (so it can be routed and should count toward silent test /
  partial rollout) and adds no example (*"A call recorded without its request and reply is never
  a kept example"*, office hours Q11). Caveat from the same docs: `off` and `metadata` calls
  *"never list"* in Requests, so their request ids may NOT be findable under 'Served by'.
- ``full``: Mubit stores the call as an example. Allowed only on cases never recorded into the
  TARGET workspace, needs ``--yes``. Held-out test splits are never recorded; re-recording a case
  already recorded into that workspace needs ``--allow-duplicates``. What was recorded where comes
  from a ledger (``recorded_full_ledger``): the 177 judge v1 inputs recorded into B and C
  (``data/gates/judge_v1/record/calls.*.<ws>.record-full.jsonl``) plus every row of
  ``data/adapter_golive/<ws>/traffic-*.jsonl`` with ``record=full`` and ``recorded_full=true``
  (e.g. R71: the ``v2`` batch into B), keyed by the exact user message (``msg_sha256``). So a
  case's ``in_training_pool`` depends on the workspace: ``v2`` is in B's pool, not C's. A full
  push written with ``--out-dir`` outside ``data/adapter_golive/<ws>/`` is NOT in the ledger.
- ``off``: drops the task fingerprint (our reading: never routed, always Opus). A control only.

Inputs (``--sources``, comma list): ``jv1`` (the 177 recorded judge v1 inputs,
data/gates/judge_v1/record/inputs.jsonl), ``trap_test`` / ``noise_test`` (held-out splits, never
recorded anywhere, so the cleanest adapter check), ``trap_dev`` / ``noise_dev`` (already inside
``jv1``), ``v2`` (the new batch). Each case carries its ground truth where known and Opus's
reference answer (C's recorded teacher answer for ``jv1``, the R40 Opus answer for test splits).

Every attempt (case id, ground truth, verdict, reason_code, Mubit request id, latency, tokens,
estimated $) goes to ``data/adapter_golive/<workspace>/traffic-<UTC stamp>.jsonl``; the valid
request ids also go to ``....request_ids.txt`` (failed / invalid attempts to
``....failed_request_ids.txt``) and are printed so you can look them up in the
console (Requests -> Served by). Concurrency <= 4, exponential backoff on 429/503/529 and
transport errors, a hard $ cap (``--usd-cap``, default $2) on estimated Opus $ + Mubit's run fee.

Every real call goes through ``backend.agent.mubit_client.MubitClient`` (the Mubit gateway) with
the workspace's key read from the env var named in ``WORKSPACE_KEYS`` (never printed).
``--dry-run`` uses a local mock (no network, no key, $0) that also simulates some "adapter"
answers and 503s, so the whole kit can be rehearsed. A push without ``--dry-run`` makes paid calls
and refuses to start without ``--yes``. Workspace A (the Opus baseline, D38) is never a target.

    python -m uv run --extra ml python scripts/push_judge_traffic.py push --dry-run
    python -m uv run --extra ml python scripts/push_judge_traffic.py push --sources trap_test,noise_test --yes
    python -m uv run --extra ml python scripts/push_judge_traffic.py push --sources jv1,trap_test,noise_test --usd-cap 2.5 --yes
    python -m uv run --extra ml python scripts/push_judge_traffic.py build-v2
    python -m uv run --extra ml python scripts/push_judge_traffic.py push --sources v2 --record full --yes
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import hashlib
import importlib.util
import json
import random
import re
import sqlite3
import statistics
import sys
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.core.types import JUDGE_REASON_VERDICT, ArmKind, JudgeOutput, JudgeReasonCode, LLMCallRecord, StepType  # noqa: E402

STEP = StepType.judge_improvement
OPUS = "claude-opus-5-5"
EXPECTED_FINGERPRINT = "5c1d3cfd2941"
WORKSPACE_KEYS = {"A": "MUBIT_API_KEY", "B": "MUBIT_API_KEY_B", "C": "MUBIT_API_KEY_C"}
# A serves the Opus baseline (D38): never push traffic there. A's key stays in WORKSPACE_KEYS only
# for the "two workspaces share a key" check.
PUSH_WORKSPACES = ("B", "C")
DEFAULT_WORKSPACE = "C"
RECORD_CHOICES = ("metadata", "full", "off")
SOURCES = ("jv1", "trap_test", "noise_test", "trap_dev", "noise_dev", "v2")
DEFAULT_SOURCES = ("trap_test", "noise_test", "jv1")
HELD_OUT_SOURCES = frozenset({"trap_test", "noise_test"})  # never recorded anywhere (benchmark held-out splits)
MAX_CONCURRENCY = 4
DEFAULT_CAP_USD = 2.00
# R59: 177 Opus judge v1 calls into C cost $0.639 ($0.0036/call); R40 test splits $0.0043/call.
FIRST_RESERVE_USD = 0.02
# mubit.ai/pricing: "$5 per 1,000 runs" (a run = one request in a learning task). Whether it
# applies at the hackathon is open (office hours Q10); counted against the cap to be safe.
DEFAULT_MUBIT_FEE_USD = 0.005
RETRY_STATUS = frozenset({429, 503, 529})
MAX_BACKOFF_RETRIES = 5
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 30.0

RECORD_DIR = REPO / "data" / "gates" / "judge_v1" / "record"
JV1_INPUTS = RECORD_DIR / "inputs.jsonl"
OPUS_REF_CALLS = (RECORD_DIR / "calls.hedge.C.record-full.jsonl",   # what C's adapter was trained on
                  RECORD_DIR / "calls.first.B.record-full.jsonl",
                  RECORD_DIR / "calls.rest.B.record-full.jsonl")
TRAP_SET = REPO / "data" / "trap_set" / "trap_set_v1.jsonl"
NOISE_SET = REPO / "data" / "noise_set" / "noise_set_v1.jsonl"
# R40: Opus 5.5 with judge v1 on the held-out test splits (record off, project A).
R40_TRAP_TEST = REPO / "data" / "trap_eval" / "trapeval-2026-10-24-claude-opus-5-5-test-043437.jsonl"
R40_NOISE_TEST = REPO / "data" / "noise_set" / "eval" / "noiseeval-2026-10-24-043502.jsonl"
BENCH_DB = REPO / "data" / "bench" / "bench.db"
ABLATION_DB = REPO / "data" / "bench" / "ablation.db"
OUT_ROOT = REPO / "data" / "adapter_golive"
V2_DIR = OUT_ROOT / "v2_batch"
V2_INPUTS = V2_DIR / "inputs.jsonl"
V2_N = 40          # 25 kept examples trigger "New versions after 25 new examples" (R58); margin for review losses (R61: 18 of 177 not reviewed)
V2_SEED = 20261024
V2_DATE = "2026-10-24"
NOISE_ACCEPTABLE = {"real": ["valid"], "noise": ["unsure", "invalid"], "fake": ["invalid"]}

Sleep = Callable[[float], Awaitable[None]]


def _rjv1() -> Any:
    """scripts/record_judge_v1.py as a module (preflight, race_claims, helpers)."""
    spec = importlib.util.spec_from_file_location("record_judge_v1", REPO / "scripts" / "record_judge_v1.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


# --------------------------------------------------------------------------- ground truth


def gt_trap(row: dict[str, Any]) -> dict[str, Any]:
    """Trap-set case: planted label; ``acceptable`` is the set's own list. The 'noise' family
    (noise_within_spread, lucky_seed) is labelled invalid-or-unsure in the set: truth 'noise'."""
    truth = "real" if row["label"] == "valid" else ("noise" if row.get("family") == "noise" else "fake")
    return {"set": "trap", "truth": truth,
            "acceptable": list(row["acceptable_verdicts"]), "category": row.get("category")}


def gt_noise(truth: str, kind: str | None = None) -> dict[str, Any]:
    """Noise-set case: 10-seed re-run ground truth (real / noise / fake)."""
    return {"set": "noise", "truth": truth, "acceptable": list(NOISE_ACCEPTABLE[truth]), "category": kind}


def gt_fake_flag(set_name: str, is_fake: bool) -> dict[str, Any]:
    """Race / collection claim: only 'was the trap option on' is known. A non-trap claim may be a
    real gain or noise, so the only wrong answer there is rejecting it as invalid."""
    return ({"set": set_name, "truth": "fake", "acceptable": ["invalid"], "category": "trap_run"} if is_fake
            else {"set": set_name, "truth": "not_fake", "acceptable": ["valid", "unsure"], "category": "non_trap_run"})


def ground_truth_for_jv1(r: dict[str, Any], trap: dict[str, dict[str, Any]], noise: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    e = r.get("expected") or {}
    if r["source"] == "trap_set":
        return gt_trap(trap[r["orig_id"]]) if r["orig_id"] in trap else None
    if r["source"] == "noise_set":
        t = noise.get(r["orig_id"], {}).get("truth") or e.get("truth")
        return gt_noise(t, noise.get(r["orig_id"], {}).get("kind")) if t in NOISE_ACCEPTABLE else None
    if r["source"] == "collection":
        return gt_fake_flag("collection", e.get("kind") == "trap")
    if r["source"] == "race":
        return gt_fake_flag("race", bool(e.get("is_trap")))
    return None


# --------------------------------------------------------------------------- Opus reference answers


def opus_refs_jv1() -> dict[str, dict[str, Any]]:
    """case id -> Opus's recorded answer (C's hedge first: the answers C's adapter learned from)."""
    out: dict[str, dict[str, Any]] = {}
    for path in OPUS_REF_CALLS:
        for r in _jsonl(path):
            if r.get("valid") and r["case_id"] not in out:
                out[r["case_id"]] = {"verdict": r["verdict"], "reason_code": r["reason_code"],
                                     "from": path.name, "mubit_request_id": r.get("mubit_request_id")}
    return out


def opus_refs_test() -> dict[str, dict[str, Any]]:
    """trap/noise set id -> R40's Opus judge v1 answer on the test split (record off)."""
    out: dict[str, dict[str, Any]] = {}
    for path, arm in ((R40_TRAP_TEST, None), (R40_NOISE_TEST, "opus")):
        for r in _jsonl(path):
            if arm is not None and r.get("arm") != arm:
                continue
            if r.get("verdict") is None:
                continue
            code = r.get("reason") if r.get("reason") in {c.value for c in JudgeReasonCode} else None
            out[r["id"]] = {"verdict": r["verdict"], "reason_code": code, "from": f"R40 {path.name}",
                            "mubit_request_id": (r.get("attempts") or [{}])[-1].get("mubit_request_id")}
    return out


# --------------------------------------------------------------------------- cases


def recorded_full_ledger(root: Path | None = None, record_dir: Path | None = None) -> dict[str, set[str]]:
    """workspace -> msg_sha256 of every user message recorded full into it (so the workspace's
    adapter may have trained on it). Sources: the judge v1 record files (the 177 inputs into B and
    C, rows with record=full and recorded=true, mapped to messages via inputs.jsonl) and
    ``<root>/<ws>/traffic-*.jsonl`` rows with record=full, recorded_full=true and not a dry run
    (dry runs live in ``<ws>/dryrun/`` and are skipped anyway: the glob is not recursive)."""
    root = OUT_ROOT if root is None else root
    record_dir = RECORD_DIR if record_dir is None else record_dir
    out: dict[str, set[str]] = {}
    by_id = {r["id"]: r.get("msg_sha256") or _sha(r["user_message"]) for r in _jsonl(record_dir / "inputs.jsonl")}
    for path in sorted(record_dir.glob("calls.*.record-full.jsonl")):
        for r in _jsonl(path):
            ws = r.get("project")
            if r.get("record") == "full" and r.get("recorded") and ws and r.get("case_id") in by_id:
                out.setdefault(str(ws), set()).add(by_id[r["case_id"]])
    if root.exists():
        for wsdir in sorted(p for p in root.iterdir() if p.is_dir()):
            for path in sorted(wsdir.glob("traffic-*.jsonl")):
                for r in _jsonl(path):
                    if (r.get("record") == "full" and r.get("recorded_full") is True and not r.get("dry_run")
                            and r.get("msg_sha256")):
                        out.setdefault(str(r.get("workspace") or wsdir.name), set()).add(r["msg_sha256"])
    return out


def _case(case_id: str, source: str, r: dict[str, Any], *, gt: dict[str, Any] | None,
          ref: dict[str, Any] | None, session_n: Any = None) -> dict[str, Any]:
    return {"case_id": case_id, "source": source, "split": r.get("split"), "target": r.get("target"),
            "in_training_pool": False, "ground_truth": gt, "opus_ref": ref,
            "msg_sha256": _sha(r["user_message"]), "user_message": r["user_message"],
            "record_session_id": r.get("session_id"), "session_n": session_n}


def load_cases(sources: Iterable[str], workspace: str | None = None,
               ledger: dict[str, set[str]] | None = None) -> list[dict[str, Any]]:
    """Cases for the chosen sources, deduplicated on the exact user message (first source wins).

    ``in_training_pool`` = the message was recorded full into ``workspace`` (per the ledger,
    ``recorded_full_ledger()`` unless given); with no workspace, recorded full into ANY workspace.
    ``recorded_in`` lists the workspaces it was recorded into."""
    sources = list(dict.fromkeys(sources))
    bad = [s for s in sources if s not in SOURCES]
    if bad:
        raise SystemExit(f"unknown sources {bad}; choose from {list(SOURCES)}")
    ledger = recorded_full_ledger() if ledger is None else ledger
    dedup = _load_cases(sources)
    for c in dedup:
        c["recorded_in"] = sorted(ws for ws, shas in ledger.items() if c["msg_sha256"] in shas)
        c["in_training_pool"] = (workspace in c["recorded_in"]) if workspace is not None else bool(c["recorded_in"])
    return dedup


def _load_cases(sources: list[str]) -> list[dict[str, Any]]:
    """Cases (in_training_pool unset: load_cases fills it from the ledger), deduplicated."""
    trap = {r["id"]: r for r in _jsonl(TRAP_SET)}
    noise = {r["id"]: r for r in _jsonl(NOISE_SET)}
    out: list[dict[str, Any]] = []
    for s in sources:
        if s == "jv1":
            refs = opus_refs_jv1()
            for r in _jsonl(JV1_INPUTS):
                out.append(_case(r["id"], "jv1", r, gt=ground_truth_for_jv1(r, trap, noise),
                                 ref=refs.get(r["id"]), session_n=r.get("n")))
        elif s in ("trap_test", "trap_dev"):
            split = s.split("_")[1]
            refs = opus_refs_test() if split == "test" else {}
            jv1_ids = {r["orig_id"]: r["id"] for r in _jsonl(JV1_INPUTS) if r["source"] == "trap_set"}
            jrefs = opus_refs_jv1() if split == "dev" else {}
            for r in trap.values():
                if r["split"] == split:
                    ref = refs.get(r["id"]) or jrefs.get(jv1_ids.get(r["id"], ""))
                    out.append(_case(r["id"], s, r, gt=gt_trap(r), ref=ref))
        elif s in ("noise_test", "noise_dev"):
            split = s.split("_")[1]
            refs = opus_refs_test() if split == "test" else {}
            jv1_ids = {r["orig_id"]: r["id"] for r in _jsonl(JV1_INPUTS) if r["source"] == "noise_set"}
            jrefs = opus_refs_jv1() if split == "dev" else {}
            for r in noise.values():
                if r["split"] == split and r.get("truth") in NOISE_ACCEPTABLE:
                    ref = refs.get(r["id"]) or jrefs.get(jv1_ids.get(r["id"], ""))
                    out.append(_case(r["id"], s, r, gt=gt_noise(r["truth"], r.get("kind")), ref=ref))
        elif s == "v2":
            if not V2_INPUTS.exists():
                raise SystemExit(f"{V2_INPUTS.relative_to(REPO)} missing; run `build-v2` first")
            for r in _jsonl(V2_INPUTS):
                out.append(_case(r["id"], "v2", r, gt=r.get("ground_truth"), ref=None,
                                 session_n=r.get("n")))
    seen: set[str] = set()
    dedup: list[dict[str, Any]] = []
    for c in out:
        if c["msg_sha256"] not in seen:
            seen.add(c["msg_sha256"])
            dedup.append(c)
    return dedup


def record_problems(record: str, cases: Sequence[dict[str, Any]], *, yes: bool, allow_duplicates: bool,
                    workspace: str | None = None) -> list[str]:
    """Why this push must not run (empty = OK). Duplicates come from ``in_training_pool``, which
    ``load_cases(..., workspace)`` sets per workspace from the recording ledger."""
    if record not in RECORD_CHOICES:
        return [f"--record must be one of {list(RECORD_CHOICES)}"]
    if record != "full":
        return []
    out: list[str] = []
    if not yes:
        out.append("--record full stores examples in the workspace: pass --yes")
    held = sorted({c["source"] for c in cases if c["source"] in HELD_OUT_SOURCES})
    if held:
        out.append(f"held-out test splits are never recorded (sources {held}); use --record metadata for them")
    dup = [c["case_id"] for c in cases if c.get("in_training_pool")]
    if dup and not allow_duplicates:
        where = f"into workspace {workspace}" if workspace else "(per the recording ledger)"
        srcs = sorted({c["source"] for c in cases if c.get("in_training_pool")})
        out.append(f"{len(dup)} cases (sources {srcs}) were already recorded full {where} (e.g. {dup[:3]}); "
                   "re-recording adds duplicate examples: pass --allow-duplicates if that is really wanted, or "
                   "record only cases not yet recorded there (see `list`)")
    return out


def session_for(case: dict[str, Any], record: str, workspace: str, stamp: str, date: str) -> str:
    """metadata.user_id. Full recordings get one session per case (like record_judge_v1, so Mubit can
    set test cases aside by session); traffic pushes share one session per run (easy to filter)."""
    if record == "full":
        if case["source"] == "v2":
            return f"collect-judgev1v2-{V2_DATE}/race/{case['session_n']}"
        return f"collect-judgev1-dup-{date}/{case['case_id']}"
    return f"golive-{date}/{workspace}/{record}/{stamp}"


# --------------------------------------------------------------------------- budget / backoff


class Budget:
    """Hard $ cap with per-call reservation, so concurrent calls cannot overshoot it."""

    def __init__(self, cap_usd: float, fee_usd: float = DEFAULT_MUBIT_FEE_USD, first_reserve: float = FIRST_RESERVE_USD) -> None:
        self.cap = float(cap_usd)
        self.fee = float(fee_usd)
        self.first_reserve = first_reserve
        self.llm = 0.0
        self.fees = 0.0
        self.reserved = 0.0
        self.n_paid = 0

    @property
    def spent(self) -> float:
        return self.llm + self.fees

    def est(self) -> float:
        llm = self.first_reserve if self.n_paid == 0 else max(1.5 * self.llm / self.n_paid, 0.001)
        return llm + self.fee

    def try_reserve(self) -> float | None:
        e = self.est()
        if self.spent + self.reserved + e > self.cap + 1e-12:
            return None
        self.reserved += e
        return e

    def settle(self, reserved: float, llm_cost: float, answered: bool) -> None:
        """A call that got an HTTP reply from the provider (any usage) pays the fee; a 503 before
        the provider pays nothing."""
        self.reserved -= reserved
        self.llm += llm_cost
        if answered:
            self.fees += self.fee
            self.n_paid += 1


_STATUS = re.compile(r"^http (\d{3})")


def http_status(error: str | None) -> int | None:
    m = _STATUS.match(error or "")
    return int(m.group(1)) if m else None


def should_back_off(error: str | None) -> bool:
    s = http_status(error)
    return (s in RETRY_STATUS) or (error or "").startswith("transport:")


def backoff_s(n: int, rng: random.Random) -> float:
    return min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** n) * (0.75 + 0.5 * rng.random())


# --------------------------------------------------------------------------- mock (dry run)


class MockJudgeClient:
    """No network, no key, $0. Answers with the case's Opus reference answer (else a heuristic);
    a share of calls is tagged as 'adapter' answers (fast, id ``mock_adp_*``, sometimes a different
    answer) and a share fails with a fake 503, so backoff and eval_served_calls can be rehearsed."""

    benchmark_safe = False

    def __init__(self, answers: dict[str, dict[str, Any]], *, adapter_share: float = 0.3,
                 err_rate: float = 0.03, seed: int = 0) -> None:
        self.answers = answers
        self.adapter_share = adapter_share
        self.err_rate = err_rate
        self.seed = seed
        self.n = 0
        self.headers_seen: list[dict[str, str]] = []
        self.adapter_ids: list[str] = []

    async def call(self, step: StepType, system: str, user: str, schema: type[JudgeOutput], arm: ArmKind,
                   session_id: str, record: Any = False, **kw: Any) -> tuple[JudgeOutput | None, LLMCallRecord]:
        from backend.agent.mubit_client import record_headers

        self.n += 1
        self.headers_seen.append(record_headers(record))
        key = f"{self.seed}|{self.n}|{session_id}|{user}"
        rng = random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))
        rid_hex = hashlib.sha256(key.encode()).hexdigest()[:24]
        rec = LLMCallRecord(arm=ArmKind(arm), session_id=session_id, step_type=StepType(step), model="mock-llm",
                            is_replay=True, recorded=False, is_retry=bool(kw.get("is_retry")), stop_reason="end_turn")
        if rng.random() < self.err_rate:
            rec.mubit_request_id = f"mock_err_{rid_hex}"
            rec.latency_ms = rng.uniform(80, 200)
            rec.error = "http 503: engine_unavailable (mock)"
            return None, rec
        ans = dict(self.answers.get(_sha(user)) or {"verdict": "unsure", "reason_code": "insufficient_evidence"})
        if not ans.get("reason_code"):
            ans["reason_code"] = {"invalid": "contamination_or_eval_leak", "unsure": "insufficient_evidence",
                                  "valid": "clean_gain"}[ans["verdict"]]
        if rng.random() < self.adapter_share:
            rec.mubit_request_id = f"mock_adp_{rid_hex}"
            rec.latency_ms = max(80.0, rng.gauss(380, 70))
            self.adapter_ids.append(rec.mubit_request_id)
            if rng.random() < 0.15:  # a mock adapter disagreeing with Opus now and then
                ans["reason_code"] = rng.choice([c.value for c in JudgeReasonCode])
                ans["verdict"] = JUDGE_REASON_VERDICT[JudgeReasonCode(ans["reason_code"])].value
        else:
            rec.mubit_request_id = f"mock_opus_{rid_hex}"
            rec.latency_ms = max(400.0, rng.gauss(1700, 280))
        rec.input_tokens, rec.output_tokens = 3300, 30
        parsed = JudgeOutput(verdict=ans["verdict"], reason_code=ans["reason_code"])
        rec.valid = True
        return parsed, rec

    async def aclose(self) -> None:
        return None


# --------------------------------------------------------------------------- real client


def make_client(workspace: str, adapter_headers: dict[str, str] | None = None) -> Any:
    """A MubitClient whose only project is ``workspace`` (key from the env var named in
    WORKSPACE_KEYS, env first then .env). Refuses a key equal to another workspace's key. Errors
    name env vars, never keys."""
    from backend.agent.mubit_client import MubitClient, ProjectRouting, load_mubit_api_key

    if workspace not in PUSH_WORKSPACES:
        raise SystemExit(f"workspace {workspace!r}: traffic goes only to {list(PUSH_WORKSPACES)} "
                         "(A serves the Opus baseline, D38)")
    var = WORKSPACE_KEYS[workspace]
    key = load_mubit_api_key(env_var=var)
    for other, ovar in WORKSPACE_KEYS.items():
        if other == workspace:
            continue
        try:
            okey = load_mubit_api_key(env_var=ovar)
        except RuntimeError:
            continue
        if okey == key:
            raise SystemExit(f"{var} equals {ovar} (workspace {other}); refusing")
    routing = ProjectRouting(
        default=workspace, routes={}, keys={workspace: var}, source=f"push_judge_traffic:{workspace}",
        adapter_headers={ArmKind.mubit.value: {STEP.value: dict(adapter_headers)}} if adapter_headers else {})
    return MubitClient(api_key=key, projects=routing, arm_models={ArmKind.mubit: OPUS})


# --------------------------------------------------------------------------- push


def _quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return round(xs[0], 1)
    return round(statistics.quantiles(xs, n=100, method="inclusive")[int(q * 100) - 1], 1)


async def push(
    cases: Sequence[dict[str, Any]],
    client: Any,
    *,
    workspace: str,
    record: str,
    out_path: Path,
    system: str,
    wire_schema: dict[str, Any],
    fingerprint: str,
    usd_cap: float = DEFAULT_CAP_USD,
    mubit_fee_usd: float = DEFAULT_MUBIT_FEE_USD,
    concurrency: int = MAX_CONCURRENCY,
    repeat: int = 1,
    invalid_retries: int = 1,
    max_backoff_retries: int = MAX_BACKOFF_RETRIES,
    sleep: Sleep = asyncio.sleep,
    stamp: str | None = None,
    date: str | None = None,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Send every case ``repeat`` times; one JSONL row per attempt. Returns a summary."""
    now = utc_now()
    stamp = stamp or now.strftime("%Y%m%dT%H%M%SZ")
    date = date or now.strftime("%Y-%m-%d")
    budget = Budget(usd_cap, mubit_fee_usd)
    sem = asyncio.Semaphore(max(1, min(int(concurrency), MAX_CONCURRENCY)))
    lock = asyncio.Lock()
    rng = random.Random(stamp)
    state: dict[str, Any] = {"stopped_cap": False, "done": 0, "backoffs": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(c, k) for k in range(max(1, repeat)) for c in cases]
    total = len(jobs)

    async def write(row: dict[str, Any]) -> None:
        async with lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def one(case: dict[str, Any], k: int) -> None:
        sid = session_for(case, record, workspace, stamp, date)
        async with sem:
            final: dict[str, Any] | None = None
            for attempt in range(1 + max(0, invalid_retries)):
                for b in range(max_backoff_retries + 1):
                    if state["stopped_cap"]:
                        return
                    reserved = budget.try_reserve()
                    if reserved is None:
                        state["stopped_cap"] = True
                        return
                    parsed, rec = await client.call(STEP, system, case["user_message"], JudgeOutput, ArmKind.mubit,
                                                    sid, record, wire_schema=wire_schema, model=OPUS,
                                                    is_retry=attempt > 0 or b > 0)
                    answered = (rec.input_tokens + rec.output_tokens) > 0
                    budget.settle(reserved, float(rec.cost_usd or 0.0), answered)
                    ok = parsed is not None and parsed.reason_code is not None
                    row = {
                        "case_id": case["case_id"], "source": case["source"], "split": case["split"],
                        "target": case["target"], "in_training_pool": case["in_training_pool"],
                        "recorded_in": case.get("recorded_in"),
                        "ground_truth": case["ground_truth"], "opus_ref": case["opus_ref"],
                        "msg_sha256": case["msg_sha256"], "repeat": k, "attempt": attempt, "backoff_n": b,
                        "workspace": workspace, "record": record, "dry_run": dry_run,
                        "prompt_version": "v1", "prompt_fingerprint": fingerprint, "model": OPUS,
                        "session_id": sid, "valid": ok,
                        "verdict": parsed.verdict.value if ok else None,
                        "reason_code": parsed.reason_code.value if ok and parsed.reason_code else None,
                        "error": rec.error if (parsed is None or rec.error) else (None if ok else "no reason_code"),
                        "http_status": http_status(rec.error), "mubit_request_id": rec.mubit_request_id,
                        "latency_ms": round(float(rec.latency_ms or 0.0), 1), "input_tokens": rec.input_tokens,
                        "cache_creation_tokens": rec.cache_creation_tokens, "cache_read_tokens": rec.cache_read_tokens,
                        "output_tokens": rec.output_tokens, "cost_usd": round(float(rec.cost_usd or 0.0), 6),
                        "cost_source": rec.cost_source, "mubit_fee_usd_est": mubit_fee_usd if answered else 0.0,
                        "stop_reason": rec.stop_reason, "recorded_full": record == "full" and rec.recorded,
                        "ts": utc_now().isoformat(timespec="seconds"),
                    }
                    await write(row)
                    final = row
                    if not ok and should_back_off(rec.error) and b < max_backoff_retries:
                        state["backoffs"] += 1
                        await sleep(backoff_s(b, rng))
                        continue
                    break
                if final is not None and final["valid"]:
                    break
            state["done"] += 1
            if final is not None:
                log(f"[push] {state['done']:>3}/{total} {final['case_id']:<20} -> "
                    f"{(final['verdict'] or 'FAILED'):<7} {(final['reason_code'] or final['error'] or '')[:34]:<34} "
                    f"{final['latency_ms']:>7.0f} ms  {final['mubit_request_id']}")

    await asyncio.gather(*(one(c, k) for c, k in jobs))
    rows = _jsonl(out_path)
    return summarize_push(rows, budget=budget, cap=usd_cap, stopped_cap=state["stopped_cap"],
                          backoffs=state["backoffs"], planned=total)


def summarize_push(rows: list[dict[str, Any]], *, budget: Budget | None = None, cap: float | None = None,
                   stopped_cap: bool = False, backoffs: int = 0, planned: int | None = None) -> dict[str, Any]:
    valid = [r for r in rows if r["valid"]]
    lat = [r["latency_ms"] for r in valid if r.get("latency_ms")]
    return {
        "planned_calls": planned, "attempts": len(rows), "valid_calls": len(valid),
        "failed_attempts": len(rows) - len(valid),
        "http_status": dict(Counter(str(r.get("http_status")) for r in rows if not r["valid"])),
        "backoffs": backoffs, "stopped_by_cap": stopped_cap, "usd_cap": cap,
        "llm_cost_usd_est": round(sum(r.get("cost_usd") or 0.0 for r in rows), 4),
        "mubit_fee_usd_est": round(sum(r.get("mubit_fee_usd_est") or 0.0 for r in rows), 4),
        "spent_usd_est": round(budget.spent, 4) if budget else None,
        "recorded_full_calls": sum(bool(r.get("recorded_full")) and r["valid"] for r in rows),
        "verdicts": dict(Counter(r["verdict"] for r in valid)),
        "latency_p50_ms": _quantile(lat, 0.5), "latency_p90_ms": _quantile(lat, 0.9),
        "by_source": dict(Counter(r["source"] for r in valid)),
    }


# --------------------------------------------------------------------------- v2 batch


def _finished_race_prefixes(db: Path) -> set[str]:
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=10.0)
    try:
        return {r[0][:8] for r in conn.execute("SELECT id FROM races WHERE status = 'finished'")}
    finally:
        conn.close()


def select_v2(candidates: list[dict[str, Any]], n: int = V2_N, seed: int = V2_SEED) -> list[dict[str, Any]]:
    """Round-robin over strata (target, fake?, the race's own verdict), seeded inside each stratum,
    so rare kinds (a real 'valid', a non-trap 'invalid') are covered before common ones repeat."""
    strata: dict[tuple[str, bool, str], list[dict[str, Any]]] = {}
    for c in candidates:
        key = (c["target"], bool(c["expected"].get("is_trap")), str(c["expected"].get("recorded_v0_verdict")))
        strata.setdefault(key, []).append(c)
    rng = random.Random(seed)
    queues = []
    for key in sorted(strata):
        items = sorted(strata[key], key=lambda c: c["msg_sha256"])
        rng.shuffle(items)
        queues.append(items)
    out: list[dict[str, Any]] = []
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    return out


def build_v2(n: int = V2_N, dbs: Sequence[Path] = (BENCH_DB, ABLATION_DB), out_dir: Path = V2_DIR) -> dict[str, Any]:
    """NEW judge v1 inputs from finished races' judged claims (rebuilt like record_judge_v1
    ``race_claims``): not among the 177, not equal to a held-out test message, deduplicated."""
    m = _rjv1()
    inputs_path = out_dir / "inputs.jsonl"
    if inputs_path.exists():
        raise SystemExit(f"{inputs_path} exists; refusing to rebuild (ids/sessions would move). Delete it first if unrecorded.")
    known = {r["msg_sha256"] for r in _jsonl(JV1_INPUTS)}
    held_out = {_sha(r["user_message"]) for r in _jsonl(TRAP_SET) + _jsonl(NOISE_SET) if r["split"] != "dev"}
    stats: dict[str, Any] = {}
    cands: list[dict[str, Any]] = []
    seen: set[str] = set()
    for db in dbs:
        if not db.exists():
            stats[db.name] = "missing"
            continue
        finished = _finished_race_prefixes(db)
        claims, st = m.race_claims(db)
        c = Counter()
        for r in claims:
            h = _sha(r["user_message"])
            if r["group"].split("/")[-1] not in finished:
                c["skipped_unfinished_race"] += 1
            elif h in known:
                c["skipped_in_177"] += 1
            elif h in held_out:
                c["skipped_equals_held_out"] += 1
            elif h in seen:
                c["skipped_duplicate"] += 1
            else:
                seen.add(h)
                cands.append({**r, "source": "race", "db": db.name, "msg_sha256": h})
                c["candidates"] += 1
        stats[db.name] = {**st, **c}
    chosen = select_v2(cands, n)
    rows = []
    for i, r in enumerate(chosen, 1):
        rows.append({**r, "n": i, "id": f"jv1v2_race_{i:03d}",
                     "session_id": f"collect-judgev1v2-{V2_DATE}/race/{i}",
                     "ground_truth": gt_fake_flag("race", bool(r["expected"].get("is_trap")))})
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = {"n": len(rows), "candidates": len(cands), "sources": stats,
               "by_target": dict(Counter(r["target"] for r in rows)),
               "ground_truth": dict(Counter(r["ground_truth"]["truth"] for r in rows)),
               "race_verdict": dict(Counter(str(r["expected"].get("recorded_v0_verdict")) for r in rows)),
               "by_db": dict(Counter(r["db"] for r in rows)),
               "note": "Opus answers come only when recorded (push --sources v2 --record full --yes); "
                       "25 kept examples trigger the next version (R58). Not yet recorded."}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- Try models demo cases

# Held-out trap-set test cases (never recorded in any workspace), game-playing target (demo.md
# prefers it), with Opus 5.5's judge v1 answers from R40.
TRY_MODELS_CASES = (("fake_caught", "trap_v1_020"), ("real_gain", "trap_v1_024"), ("noise", "trap_v1_048"))
TRY_MODELS_DIR = OUT_ROOT / "try_models"


def export_try_models(out_dir: Path = TRY_MODELS_DIR) -> dict[str, Any]:
    """Write each demo case's exact user message (and the frozen system prompt) to text files to
    paste into the console's Try models, plus the expected answers. No calls."""
    from backend.agent.prompts import load_system_prompt
    from backend.agent.prompts.build import CACHE_BREAK

    trap = {r["id"]: r for r in _jsonl(TRAP_SET)}
    refs = opus_refs_test()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "system_prompt.judge_improvement.v1.txt").write_text(load_system_prompt(STEP, "v1"), encoding="utf-8")
    cases = []
    for role, cid in TRY_MODELS_CASES:
        r = trap[cid]
        um = r["user_message"]
        (out_dir / f"{role}.{cid}.user.txt").write_text(um.replace(CACHE_BREAK, "\n"), encoding="utf-8")
        cases.append({"role": role, "case_id": cid, "split": r["split"], "category": r["category"],
                      "ground_truth": gt_trap(r), "explanation": r["explanation"],
                      "opus_answer_R40": refs.get(cid), "user_file": f"{role}.{cid}.user.txt"})
    (out_dir / "cases.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")
    return {"dir": str(out_dir), "cases": cases}


# --------------------------------------------------------------------------- CLI


def parse_header(items: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--adapter-header needs name=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("push", help="send judge v1 calls through one workspace")
    q.add_argument("--workspace", choices=PUSH_WORKSPACES, default=DEFAULT_WORKSPACE,
                   help="B or C (never A: it serves the Opus baseline)")
    q.add_argument("--record", choices=RECORD_CHOICES, default="metadata")
    q.add_argument("--sources", default=",".join(DEFAULT_SOURCES), help=f"comma list of {list(SOURCES)}")
    q.add_argument("--limit", type=int, default=None, help="only the first N cases (after shuffling)")
    q.add_argument("--repeat", type=int, default=1, help="send each case N times (metadata traffic only)")
    q.add_argument("--no-shuffle", action="store_true", help="keep source order (default: seeded shuffle)")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY, help=f"capped at {MAX_CONCURRENCY}")
    q.add_argument("--usd-cap", type=float, default=DEFAULT_CAP_USD, help="hard cap on estimated $ (LLM + Mubit run fee)")
    q.add_argument("--mubit-fee-usd", type=float, default=DEFAULT_MUBIT_FEE_USD, help="Mubit run fee per answered call")
    q.add_argument("--yes", action="store_true",
                   help="confirm paid calls: required for any push without --dry-run (and for --record full)")
    q.add_argument("--allow-duplicates", action="store_true", help="allow --record full on already-recorded cases")
    q.add_argument("--adapter-header", action="append", default=[],
                   help="name=value, for Mubit's announced adapter pin header (not shipped yet)")
    q.add_argument("--dry-run", action="store_true", help="mock client: no network, no key, $0")
    q.add_argument("--mock-adapter-share", type=float, default=0.3, help="dry run: share of mock 'adapter' answers")
    q.add_argument("--mock-error-rate", type=float, default=0.03, help="dry run: share of mock 503s")
    q.add_argument("--out-dir", type=Path, default=None)
    b = sub.add_parser("build-v2", help="build data/adapter_golive/v2_batch/inputs.jsonl (no calls)")
    b.add_argument("--n", type=int, default=V2_N)
    sub.add_parser("list", help="print how many cases each source has (no calls)")
    sub.add_parser("export-try-models", help="write the 3 Try models demo cases to data/adapter_golive/try_models/")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    if ns.cmd == "build-v2":
        print(json.dumps(build_v2(ns.n), indent=1))
        return 0
    if ns.cmd == "export-try-models":
        print(json.dumps(export_try_models(), indent=1))
        return 0
    if ns.cmd == "list":
        ledger = recorded_full_ledger()
        for s in SOURCES:
            try:
                cs = load_cases([s], ledger=ledger)
            except SystemExit as e:
                print(f"{s:<11} -: {e}")
                continue
            pools = ", ".join(f"{ws} {sum(ws in c['recorded_in'] for c in cs)}" for ws in PUSH_WORKSPACES)
            print(f"{s:<11} {len(cs):>4} cases, ground truth {sum(c['ground_truth'] is not None for c in cs)}, "
                  f"Opus reference {sum(c['opus_ref'] is not None for c in cs)}, already recorded full into: {pools}")
        return 0

    m = _rjv1()
    system, schema = m.preflight()  # judge v1 frozen, fingerprint 5c1d3cfd2941, else exit
    from backend.agent.prompts import prompt_fingerprint

    fp = prompt_fingerprint(STEP)
    if fp != EXPECTED_FINGERPRINT:
        raise SystemExit(f"judge fingerprint {fp} != {EXPECTED_FINGERPRINT}")
    cases = load_cases([s.strip() for s in ns.sources.split(",") if s.strip()], workspace=ns.workspace)
    if not ns.no_shuffle:
        random.Random(ns.seed).shuffle(cases)
    if ns.limit is not None:
        cases = cases[: ns.limit]
    if not cases:
        print("ERROR: no cases selected", file=sys.stderr)
        return 2
    est = len(cases) * ns.repeat * (0.0045 + ns.mubit_fee_usd)
    errs = record_problems(ns.record, cases, yes=ns.yes, allow_duplicates=ns.allow_duplicates, workspace=ns.workspace)
    if ns.record == "full" and ns.repeat != 1:
        errs.append("--repeat is for metadata traffic; a full recording sends each case once")
    if not ns.dry_run and not ns.yes and ns.record != "full":
        errs.append(f"this push makes {len(cases) * ns.repeat} paid calls through Mubit (est ~${est:.2f}, cap "
                    f"${ns.usd_cap:.2f}): pass --yes to send them, or --dry-run to rehearse for $0")
    if errs:
        for e in errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 2
    headers = parse_header(ns.adapter_header)
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    out_dir = ns.out_dir or (OUT_ROOT / ns.workspace / ("dryrun" if ns.dry_run else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in range(1000):  # claim a fresh file: push() appends, so never reuse an earlier run's file
        out_path = out_dir / (f"traffic-{stamp}.jsonl" if n == 0 else f"traffic-{stamp}-{n}.jsonl")
        try:
            out_path.open("x", encoding="utf-8").close()
            break
        except FileExistsError:
            continue
    else:
        raise SystemExit(f"no free traffic file name in {out_dir}")
    if ns.dry_run:
        answers = {c["msg_sha256"]: c["opus_ref"] for c in cases if c["opus_ref"]}
        client: Any = MockJudgeClient(answers, adapter_share=ns.mock_adapter_share, err_rate=ns.mock_error_rate)
    else:
        client = make_client(ns.workspace, headers)
    print(f"[push] workspace {ns.workspace} ({WORKSPACE_KEYS[ns.workspace]}), record {ns.record}, "
          f"{len(cases)} cases x {ns.repeat}, prompt v1:{fp}, model {OPUS}, concurrency "
          f"{min(ns.concurrency, MAX_CONCURRENCY)}, cap ${ns.usd_cap:.2f}, est ~${est:.2f}"
          f"{' [DRY RUN: mock, no network, $0]' if ns.dry_run else ''}", flush=True)
    if headers:
        print(f"[push] adapter header(s) on the mubit arm: {sorted(headers)}")

    async def go() -> dict[str, Any]:
        try:
            return await push(cases, client, workspace=ns.workspace, record=ns.record, out_path=out_path,
                              system=system, wire_schema=schema, fingerprint=fp, usd_cap=ns.usd_cap,
                              mubit_fee_usd=0.0 if ns.dry_run else ns.mubit_fee_usd, concurrency=ns.concurrency,
                              repeat=ns.repeat, stamp=stamp, dry_run=ns.dry_run,
                              sleep=(lambda s: asyncio.sleep(0)) if ns.dry_run else asyncio.sleep)
        finally:
            await client.aclose()

    summary = asyncio.run(go())
    rows = _jsonl(out_path)
    ids = [r["mubit_request_id"] for r in rows if r["valid"] and r["mubit_request_id"]]
    ids_path = out_path.with_suffix(".request_ids.txt")
    ids_path.write_text("".join(i + "\n" for i in ids), encoding="utf-8")
    # failed / invalid attempts too: look them up in the console as well (an adapter's bad reply is a miss)
    failed = [r for r in rows if not r["valid"] and r["mubit_request_id"]]
    if failed:
        failed_path = out_path.with_suffix(".failed_request_ids.txt")
        failed_path.write_text("".join(f"{r['mubit_request_id']}\t{r['case_id']}\tFAILED: {r['error']}\n"
                                       for r in failed), encoding="utf-8")
        summary["failed_request_ids_file"] = (str(failed_path.relative_to(REPO)) if failed_path.is_relative_to(REPO)
                                              else str(failed_path))
    if ns.dry_run and isinstance(client, MockJudgeClient):
        sb = out_path.with_suffix(".served_by_mubit.mock.txt")
        sb.write_text("".join(i + "\n" for i in client.adapter_ids if i in set(ids)), encoding="utf-8")
        summary["mock_served_by_file"] = str(sb.relative_to(REPO)) if sb.is_relative_to(REPO) else str(sb)
    summary.update(traffic_file=str(out_path.relative_to(REPO)) if out_path.is_relative_to(REPO) else str(out_path),
                   request_ids_file=str(ids_path.relative_to(REPO)) if ids_path.is_relative_to(REPO) else str(ids_path),
                   workspace=ns.workspace, record=ns.record, dry_run=ns.dry_run, prompt_fingerprint=fp)
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("[push] request ids (look these up in the console: Requests -> Served by):")
    for i in ids:
        print(f"  {i}")
    if failed:
        print("[push] failed / invalid attempts (look these up too; eval_served_calls counts them as wrong):")
        for r in failed:
            print(f"  {r['mubit_request_id']}  {r['case_id']}  FAILED: {r['error']}")
    if ns.record != "full":
        print("[push] note: the docs say `metadata` / `off` calls never list in Requests; if these ids are not "
              "found there, use the task's traffic counts and the adapter card instead.")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
