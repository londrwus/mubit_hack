"""Build data/frontend/showcase.json: the benchmark showcase cards for the dashboard.

Cards, in this order (schema: data/frontend/README.md):
1. ``game_playing`` (hero, R78)     same model quality for a quarter of the AI bill
2. ``tabular`` (R82)                nearly the same model, 4x cheaper AI bill (chosen after R75 failed)
3. ``where_it_costs_quality`` (R85, R75)   the cheaper bill comes with a worse model there
4. ``sapling`` (R81 + R83)          catching fake improvements on real race claims
5. ``vs_no_ai_tool`` (R53, R38, R54)   the AI assistant vs a no-AI tuning tool (secondary slide, D46)

Every race number is recomputed from the read-only bench DB with ``scripts/headline.py``'s own
functions (``headline.run``) and every Sapling number is read from the R81 / R83 JSON files. Each
number is then checked against the row of ``docs/results_log.md`` it cites (its ``log_text`` must
appear in that row and the recomputed value must round to the logged value); R84's corrections are
used where they apply. The script aborts on any mismatch, so the JSON never holds a number the log
does not. A number the log does not state (``in_log=False``) is refused outright: no flagged
"reference only" numbers ship either (dashboard_spec.md section 6).

No LLM calls, no Modal, no writes except the output file.

Usage:
    python -m uv run --with onnxruntime==1.29.0 --extra ml python scripts/build_showcase.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "scripts", REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import headline as hl  # noqa: E402

DB = REPO_ROOT / "data" / "bench" / "bench.db"
LOG = REPO_ROOT / "docs" / "results_log.md"
OUT = REPO_ROOT / "data" / "frontend" / "showcase.json"
R81_JSON = REPO_ROOT / "data" / "sapling" / "compare_published" / "compare_published.json"
V4_JSON = REPO_ROOT / "data" / "sapling" / "sapling_v4" / "heads" / "final_v4.json"
SCHEMA_VERSION = 1

# Race sets (8-char prefixes, as logged), seeds 0-2.
RACES: dict[str, list[str]] = {
    "R78": ["4ca4ed27", "d4913e95", "145c2a2d"],   # game-playing, BM planner every >= 60 s
    "R82": ["c1f4b1d9", "9d0cfd77", "b3922d0e"],   # tabular, BM planner every >= 30 s
    "R75": ["613e2537", "6afbbe8a", "c72b3af1"],   # tabular, BM planner every >= 90 s (FAILED)
    "R85": ["e8106409", "9ec18124", "9f494453"],   # image recognition (quality gate FAILED)
    "R53": ["60477ba1", "949fce51", "c8261494"],   # image recognition, Opus vs Optuna
    "R38": ["5245bacc", "91a4e48d", "f98b6712"],   # tabular, Opus vs Optuna
    "R48": ["8a76fa05", "45ab180f", "59d7f7eb"],   # game-playing, Opus vs Optuna (R54 pool)
    "R44": ["2ded10bc", "e3fddb29", "aaedd180"],   # tabular, R54 pool
}
# Fairness fingerprints stated in the log (a recompute must land on the same pooled setup).
FINGERPRINTS = {"R78": "e0d1cc7906", "R82": "d41d7e04a4", "R75": "6978df02c4", "R85": "f12f0585d7",
                "R53": "e88d147550", "R48": "c40e8e54a9"}
R54_POOL = ("R48", "R53", "R44")

SIDES: dict[str, dict[str, str]] = {
    "standard": {"label": "Standard AI assistant", "sublabel": "Opus 5.5",
                 "hover": "The plain research loop a lab would run with Opus 5.5"},
    "busy_machines": {"label": "Busy Machines", "sublabel": "Opus 5.5",
                      "hover": "The same Opus 5.5 plus our engine: lean planner prompt, a built-in no-AI tuner "
                               "that fills idle machines, and a checker for fake improvements"},
    "no_ai_tool": {"label": "No-AI tuning tool", "sublabel": "standard search, no AI",
                   "hover": "A standard hyperparameter search with no AI (Optuna), run in every race as the baseline"},
    "published_sapling": {"label": "Published Banyan Sapling", "sublabel": "MubitLabs, as released",
                          "hover": "Stock head and its own calibration"},
    "ours_v2_strict": {"label": "Our fine-tune v2, strict", "sublabel": "the release",
                       "hover": "sapling-research v2, calibration alpha 0.01"},
    "ours_v2_looser": {"label": "Our fine-tune v2, looser", "sublabel": "the release",
                       "hover": "sapling-research v2, calibration alpha 0.05"},
    "ours_v4_strict": {"label": "Our fine-tune v4, strict", "sublabel": "not released",
                       "hover": "sapling-research v4, calibration alpha 0.01; failed its general gate (R83)"},
    "ours_v4_looser": {"label": "Our fine-tune v4, looser", "sublabel": "not released",
                       "hover": "sapling-research v4, calibration alpha 0.05; failed its general gate (R83)"},
    "opus": {"label": "Opus 5.5", "sublabel": "as the checker",
             "hover": "Opus 5.5 judge prompt v1, recorded verdicts"},
}
ARM_SIDE = {"opus": "standard", "opus_bm": "busy_machines", "optuna": "no_ai_tool"}

# Plain-language metric labels (CLAUDE.md "UI copy"; dashboard_spec.md section 6).
ALLOWED_LABELS = frozenset({
    "Model quality", "AI bill per race", "Total cost per race", "Time to reach the goal", "Cost to reach the goal",
    "Cost to match the standard assistant's best model", "GPUs waiting", "Machines waiting",
    "Experiments per dollar", "Fake improvements accepted", "Shipped a fake improvement",
    "Fakes caught", "Fakes approved", "Real improvements confirmed", "No answer", "Time per check",
})
AI_NOTE = "list-price estimate"
SIM_NOTE = "simulated GPU time"
MINUS = "−"


class ShowcaseError(SystemExit):
    pass


# --------------------------------------------------------------------------- formatting


def signed(s: str) -> str:
    return s.replace("-", MINUS)


def f_usd(v: float, d: int = 2) -> str:
    return signed(f"${v:,.{d}f}") if v >= 0 else MINUS + f"${-v:,.{d}f}"


def f_q(v: float, d: int) -> str:
    return signed(f"{v:.{d}f}")


def f_plus(v: float, d: int) -> str:
    return signed(f"{v:+.{d}f}")


def f_min(minutes: float) -> str:
    return f"{minutes:.1f} min"


def f_pct(v: float, d: int = 1) -> str:
    return f"{v:.{d}f}%"


def f_int(v: float) -> str:
    return f"{v:,.0f}"


def f_x(v: float, d: int = 1) -> str:
    return f"{v:.{d}f}×"


_NUM = re.compile(r"(?<![\w.])[−-]?\d[\d,]*(?:\.\d+)?")


def parse_num(text: str) -> tuple[float, float]:
    """First number in ``text`` and half a unit of its last digit (the logged rounding)."""
    m = _NUM.search(text)
    if not m:
        raise ShowcaseError(f"no number in log text {text!r}")
    tok = m.group(0).replace(MINUS, "-").replace(",", "")
    dec = len(tok.split(".")[1]) if "." in tok else 0
    return float(tok), 0.5 * 10 ** (-dec)


# --------------------------------------------------------------------------- the log


def log_rows(path: Path = LOG) -> dict[str, str]:
    """R-id -> that row's text, bold markers removed (the substring every number is checked against)."""
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\| (R\d+) \|", line)
        if m:
            if m.group(1) in rows:
                raise ShowcaseError(f"{m.group(1)} appears twice in {path}")
            rows[m.group(1)] = line.replace("**", "")
    return rows


class Checker:
    """Builds number objects and checks each one against the log row it cites."""

    def __init__(self, rows: Mapping[str, str]) -> None:
        self.rows = rows
        self.checked = 0

    def row(self, r_id: str) -> str:
        if r_id not in self.rows:
            raise ShowcaseError(f"{r_id} is not in docs/results_log.md")
        return self.rows[r_id]

    def text(self, r_id: str, log_text: str) -> None:
        if log_text not in self.row(r_id):
            raise ShowcaseError(f"{r_id}: {log_text!r} is not in the logged row")
        self.checked += 1

    def num(self, value: float | int | None, unit: str, display: str, r_id: str, source: Mapping[str, Any],
            n: int, *, log_text: str | None, logged: float | None = None, tolerance: float | None = None,
            tolerance_note: str | None = None, in_log: bool = True, std: float | None = None,
            std_tolerance: float | None = None, **extra: Any) -> dict[str, Any]:
        """A number object. ``in_log``: ``log_text`` must appear in row ``r_id`` and ``value`` must round
        to the logged value (``logged``, parsed from ``log_text`` when not given; None = text-only check)."""
        d: dict[str, Any] = {"value": value, "unit": unit, "display": display, "r_id": r_id, "source": dict(source),
                             "n": int(n)}
        if std is not None:
            d["std"] = std
        d.update(extra)
        d["in_log"] = in_log
        if not in_log:
            raise ShowcaseError(f"{r_id} {display}: not in the results log; log it before it can ship "
                                "(dashboard_spec.md section 6)")
        if log_text is None:
            raise ShowcaseError(f"{r_id} {display}: a logged number needs its log_text")
        self.text(r_id, log_text)
        d["log_text"] = log_text
        if logged is None and tolerance is None and value is not None and _NUM.search(log_text):
            logged, tol = parse_num(log_text)
        else:
            tol = 0.0
        if logged is not None and value is not None:
            # explicit logged value: the rounding of the displayed number (half a unit of its last digit)
            tol = tolerance if tolerance is not None else (tol or parse_num(display)[1])
            if abs(float(value) - logged) > tol + 1e-9:
                raise ShowcaseError(f"{r_id}: recomputed {value!r} ({display}) != logged {logged} in {log_text!r} "
                                    f"(tolerance {tol})")
            d["logged_value"] = logged
            d["tolerance"] = tol
            if tolerance_note:
                d["tolerance_note"] = tolerance_note
        if std is not None and "±" in log_text:
            s_logged, s_tol = parse_num(log_text.split("±", 1)[1])
            s_tol = std_tolerance if std_tolerance is not None else s_tol
            if abs(std - s_logged) > s_tol + 1e-9:
                raise ShowcaseError(f"{r_id}: recomputed std {std} != logged {s_logged} in {log_text!r}")
            d["logged_std"] = s_logged
            d["std_tolerance"] = s_tol
        return d

    def runs(self, values: Sequence[float], fmt: Callable[[float], str], r_id: str, log_text: str,
             sep: str = ", ") -> dict[str, Any]:
        """Per-run values, checked one by one against a logged list such as '0.21×, 0.21×, 0.34×'."""
        self.text(r_id, log_text)
        shown = [fmt(v) for v in values]
        if sep.join(shown) != log_text:
            raise ShowcaseError(f"{r_id}: per-run {sep.join(shown)!r} != logged {log_text!r}")
        return {"values": list(values), "display": sep.join(shown), "r_id": r_id, "log_text": log_text}


# --------------------------------------------------------------------------- race data


def full_ids(prefixes: Sequence[str], db: Path) -> list[str]:
    c = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        out = []
        for p in prefixes:
            rows = c.execute("select id from races where id like ?", (p + "%",)).fetchall()
            if len(rows) != 1:
                raise ShowcaseError(f"race prefix {p}: {len(rows)} matches in {db}")
            out.append(rows[0][0])
        return out
    finally:
        c.close()


RACESET_SOURCE: dict[str, dict[str, Any]] = {}


class RaceSet:
    """One logged 3-seed race set, recomputed by headline.run (read-only DB)."""

    def __init__(self, rid: str, db: Path) -> None:
        self.rid = rid
        ids = full_ids(RACES[rid], db)
        groups, _ex, notes = hl.run(db=db, race_ids=ids)
        if len(groups) != 1:
            raise ShowcaseError(f"{rid}: expected one pooled setup, got {len(groups)} ({notes})")
        g = groups[0]
        if not g.headline_ok:
            raise ShowcaseError(f"{rid}: not headline-eligible: {g.blockers}")
        if rid in FINGERPRINTS and g.fingerprint_id != FINGERPRINTS[rid]:
            raise ShowcaseError(f"{rid}: fingerprint {g.fingerprint_id} != logged {FINGERPRINTS[rid]}")
        if sorted(g.race_ids) != sorted(ids) or sorted(set(g.seeds)) != [0, 1, 2]:
            raise ShowcaseError(f"{rid}: races/seeds {g.race_ids} {g.seeds} are not the logged set")
        self.g = g
        RACESET_SOURCE[rid] = self.source = {"race_ids": list(RACES[rid]), "file": "data/bench/bench.db",
                                             "computed_by": "scripts/headline.py run()"}
        self.exports = sorted(str(p.relative_to(REPO_ROOT).as_posix()) for pre in RACES[rid]
                              for p in (REPO_ROOT / "data" / "bench").glob(f"race-*-{pre}*.jsonl"))

    def runs(self, arm: str) -> list[Any]:
        return sorted((r for r in self.g.runs if r.arm == arm), key=lambda r: r.seed)

    def vals(self, arm: str, fld: str) -> list[float | None]:
        return [getattr(r, fld) for r in self.runs(arm)]

    def stat(self, arm: str, fld: str) -> Any:
        return hl._stat(self.vals(arm, fld))

    def paired(self, arm: str, base: str, fld: str, fn: Callable[[float, float], float]) -> list[float]:
        a, b = self.runs(arm), self.runs(base)
        if [r.seed for r in a] != [r.seed for r in b]:
            raise ShowcaseError(f"{self.rid}: {arm} and {base} ran different seeds")
        return [fn(getattr(x, fld), getattr(y, fld)) for x, y in zip(a, b, strict=True)]

    def trap_runs(self, arm: str) -> int:
        return int(self.g.arms[arm].trap_runs)

    def fakes_accepted(self, arm: str) -> int:
        return int(sum(int(v or 0) for v in self.vals(arm, "fake_best_accepted")))

    def shipped_fakes(self, arm: str) -> int:
        return sum(1 for v in self.vals(arm, "shipped_is_fake") if v)

    def setup_text(self, machines: str) -> str:
        return (f"{self.g.budget_s / 60:.0f} min per race · {machines} per side · same model (Opus 5.5), machines, "
                "clock, $ cap and starting code · 3 runs per side")


def metric(mid: str, label: str, values: Mapping[str, dict[str, Any] | None], *, hover: str,
           higher_is_better: bool | None, comparison: dict[str, Any] | None = None, note: str | None = None,
           verdict: str | None = None) -> dict[str, Any]:
    if label not in ALLOWED_LABELS:
        raise ShowcaseError(f"label {label!r} is not a plain-language label")
    m: dict[str, Any] = {"id": mid, "label": label, "hover": hover, "higher_is_better": higher_is_better,
                         "values": dict(values), "comparison": comparison}
    if verdict:
        m["verdict"] = verdict
    if note:
        m["note"] = note
    return m


def cmp_num(ck: Checker, kind: str, value: float, display: str, r_id: str, source: Mapping[str, Any], n: int, *,
            formula: str, log_text: str | None, in_log: bool = True, **kw: Any) -> dict[str, Any]:
    return ck.num(value, kw.pop("unit", "ratio"), display, r_id, source, n, log_text=log_text, in_log=in_log,
                  kind=kind, formula=formula, **kw)


def item(text: str, r_id: str, numbers: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    return {"text": text, "r_id": r_id, "numbers": list(numbers)}


def n_of(st: Any) -> dict[str, int]:
    return {"n": st.n, "n_total": st.n_total}


# --------------------------------------------------------------------------- common race metrics


def m_quality(ck: Checker, rs: RaceSet, arms: Sequence[str], digits: int, unit: str, hover: str,
              comparison: dict[str, Any] | None, verdict: str | None = None, std_digits: int | None = None,
              log_texts: Mapping[str, tuple[str, float, str]] | None = None) -> dict[str, Any]:
    """log_texts: arm -> (logged text, std tolerance, why) where the log rounds the spread differently."""
    vals = {}
    for arm in arms:
        st = rs.stat(arm, "best_quality")
        disp = f_q(st.mean, digits)
        sd = std_digits if std_digits is not None else digits
        lt, s_tol, why = (log_texts or {}).get(arm, (f"{disp} ± {st.std:.{sd}f}", None, None))
        vals[ARM_SIDE[arm]] = ck.num(st.mean, unit, disp, rs.rid, rs.source, st.n, n_total=st.n_total,
                                     log_text=lt, std=st.std,
                                     std_display=("± " + lt.split("±", 1)[1].strip()) if why else f"± {st.std:.{sd}f}",
                                     std_tolerance=s_tol, tolerance_note=why,
                                     per_run=rs.vals(arm, "best_quality"))
    return metric("model_quality", "Model quality", vals, hover=hover, higher_is_better=True,
                  comparison=comparison, verdict=verdict)


def m_ai_bill(ck: Checker, rs: RaceSet, arms: Sequence[str], log_digits: int, *,
              log_texts: Mapping[str, str] | None = None, tolerances: Mapping[str, tuple[float, str]] | None = None,
              comparison: dict[str, Any] | None = None) -> dict[str, Any]:
    vals = {}
    for arm in arms:
        st = rs.stat(arm, "llm_cost_usd")
        lt = (log_texts or {}).get(arm) or f_usd(st.mean, log_digits)
        tol, why = (tolerances or {}).get(arm, (None, None))
        vals[ARM_SIDE[arm]] = ck.num(st.mean, "usd", f_usd(st.mean, 2), rs.rid, rs.source, st.n,
                                     n_total=st.n_total, log_text=lt, std=st.std if "±" in lt else None,
                                     tolerance=tol, tolerance_note=why, logged=parse_num(lt)[0] if tol else None,
                                     per_run=rs.vals(arm, "llm_cost_usd"), note=AI_NOTE)
    return metric("ai_bill_per_race", "AI bill per race", vals,
                  hover="Cost of the assistant's AI calls in one 15-minute race, at list price (estimate)",
                  higher_is_better=False, comparison=comparison, note=AI_NOTE)


def x_cheaper(ck: Checker, rs: RaceSet, fld: str, r_id: str, log_text: str, *, ratio_log: str | None = None,
              in_log: bool = True, kind: str = "x_cheaper", word: str = "cheaper") -> dict[str, Any]:
    """standard ÷ busy machines on ``fld`` (means), as 'N× cheaper'."""
    a, b = rs.stat("opus", fld), rs.stat("opus_bm", fld)
    v = a.mean / b.mean
    extra: dict[str, Any] = {}
    if ratio_log:
        r = b.mean / a.mean
        ck.text(r_id, ratio_log)
        lv, lt = parse_num(ratio_log)
        if abs(r - lv) > lt + 1e-9:
            raise ShowcaseError(f"{r_id}: ratio {r} != logged {ratio_log}")
        extra = {"ratio_busy_over_standard": r, "ratio_display": f"{r:.2f}×", "ratio_log_text": ratio_log}
    return cmp_num(ck, kind, v, f"{f_x(v)} {word}", r_id, rs.source, min(a.n, b.n),
                   formula=f"standard mean ÷ Busy Machines mean ({fld})", log_text=log_text if in_log else None,
                   in_log=in_log, **extra)


def m_total(ck: Checker, rs: RaceSet, arms: Sequence[str], *, comparison: dict[str, Any] | None, simulated: bool,
            log_texts: Mapping[str, str] | None = None) -> dict[str, Any]:
    vals = {}
    for arm in arms:
        st = rs.stat(arm, "total_cost_usd")
        vals[ARM_SIDE[arm]] = ck.num(st.mean, "usd", f_usd(st.mean, 2), rs.rid, rs.source, st.n,
                                     n_total=st.n_total, log_text=(log_texts or {}).get(arm) or f_usd(st.mean, 2),
                                     per_run=rs.vals(arm, "total_cost_usd"))
    note = f"AI calls ({AI_NOTE}) + machine time used in the race" + (f"; {SIM_NOTE}" if simulated else "")
    return metric("total_cost_per_race", "Total cost per race", vals,
                  hover="AI bill plus the machine time the side used during the race", higher_is_better=False,
                  comparison=comparison, note=note)


def m_waiting(ck: Checker, rs: RaceSet, label: str, hover: str, note: str | None) -> dict[str, Any]:
    vals = {}
    for arm in ("opus", "opus_bm"):
        st = rs.stat(arm, "gpu_idle_pct")
        vals[ARM_SIDE[arm]] = ck.num(st.mean, "percent", f_pct(st.mean, 1), rs.rid, rs.source, st.n,
                                     n_total=st.n_total, log_text=f_pct(st.mean, 1), per_run=rs.vals(arm, "gpu_idle_pct"))
    return metric("machines_waiting", label, vals, hover=hover, higher_is_better=False, note=note)


def m_fakes(ck: Checker, rs: RaceSet, arms: Sequence[str], log: Mapping[str, tuple[str, str]],
            hover: str) -> dict[str, Any]:
    """log: arm -> (R-id, logged text such as '0 of 847')."""
    vals = {}
    for arm in arms:
        acc, traps = rs.fakes_accepted(arm), rs.trap_runs(arm)
        rid, lt = log[arm]
        lv = [int(x) for x in re.findall(r"\d+", lt)[:2]]
        if lv != [acc, traps]:
            raise ShowcaseError(f"{rs.rid} {arm}: fakes {acc} of {traps} != logged {lt!r} ({rid})")
        vals[ARM_SIDE[arm]] = ck.num(acc, "count", f"{acc} of {f_int(traps)}", rid, rs.source, 3, log_text=lt,
                                     logged=float(lv[0]), tolerance=0.0, of=traps)
    return metric("fake_improvements_accepted", "Fake improvements accepted", vals, hover=hover,
                  higher_is_better=False)


# --------------------------------------------------------------------------- card 1: game-playing (R78)


def card_game_playing(ck: Checker, rs: RaceSet) -> dict[str, Any]:
    S = rs.source
    # Model quality: a tie within noise (asserted, not just written).
    diffs = rs.paired("opus_bm", "opus", "best_quality", lambda a, b: a - b)
    o, b = rs.stat("opus", "best_quality"), rs.stat("opus_bm", "best_quality")
    if not abs(b.mean - o.mean) < max(o.std, b.std):
        raise ShowcaseError("R78: the quality gap is not inside the run-to-run spread; 'a tie' is not supported")
    tie = cmp_num(ck, "tie", b.mean - o.mean, "Same: a tie within noise", "R78", S, 3, unit="game_score",
                  formula="Busy Machines mean − standard mean (best score); gap < run-to-run spread",
                  log_text="a tie within noise", logged=None,
                  per_run=ck.runs(diffs, lambda v: f_plus(v, 2), "R78", "+0.14, −0.19, +0.51"))
    quality = m_quality(ck, rs, ("opus", "opus_bm"), 1, "game_score",
                        "Best score so far on game levels the model never trained on (average reward; higher is "
                        "better)", tie)

    ai_cmp = x_cheaper(ck, rs, "llm_cost_usd", "R78", "4.0× cheaper", ratio_log="0.25×")
    ai_cmp["per_run_ratio"] = ck.runs(rs.paired("opus_bm", "opus", "llm_cost_usd", lambda a, b: a / b),
                                      lambda v: f"{v:.2f}×", "R78", "0.21×, 0.21×, 0.34×")
    ai = m_ai_bill(ck, rs, ("opus", "opus_bm"), 2,
                   log_texts={"opus": f_usd(rs.stat("opus", "llm_cost_usd").mean) + " ± 0.05",
                              "opus_bm": f_usd(rs.stat("opus_bm", "llm_cost_usd").mean) + " ± 0.23"},
                   comparison=ai_cmp)
    total = m_total(ck, rs, ("opus", "opus_bm"), simulated=False,
                    comparison=x_cheaper(ck, rs, "total_cost_usd", "R78", "3.1× cheaper"))

    # Time and cost to the goal (-170). Their ratios are not in the log, so none is computed or shipped.
    t_vals, c_vals = {}, {}
    for arm in ("opus", "opus_bm"):
        st = rs.stat(arm, "time_to_goal_s")
        t_vals[ARM_SIDE[arm]] = ck.num(st.mean / 60, "minutes", f_min(st.mean / 60), "R78", S, st.n,
                                       n_total=st.n_total, log_text=f_min(st.mean / 60), seconds=st.mean)
        cs = rs.stat(arm, "cost_to_goal_usd")
        d = 3 if cs.mean < 0.1 else 2
        c_vals[ARM_SIDE[arm]] = ck.num(cs.mean, "usd", f_usd(cs.mean, d), "R78", S, cs.n, n_total=cs.n_total,
                                       log_text=f"{f_usd(cs.mean, d)} / {f_min(st.mean / 60)}")
    goal_note = ("Goal: a score of −170, frozen before the races. It is an easy goal with 30 machines: the no-AI "
                 "tuning tool also reaches it in the first minute (R78, R48)")
    time_goal = metric("time_to_goal", "Time to reach the goal", t_vals, higher_is_better=False,
                       hover="Race time until the first checked model at or past the goal",
                       comparison=None,  # the ratio is not in the log (dashboard_spec.md §6)
                       note=goal_note)
    cost_goal = metric("cost_to_goal", "Cost to reach the goal", c_vals, higher_is_better=False,
                       hover="AI bill plus machine time spent until the goal was first reached",
                       comparison=None,  # the ratio is not in the log (dashboard_spec.md §6)
                       note=goal_note)

    # Same quality (D48 #1): cost to reach the standard assistant's final best, per race.
    sq = {}
    for arm, lt in (("opus", "Opus $3.10"), ("opus_bm", "BM $0.46")):
        st = rs.stat(arm, "cost_to_opus_quality_usd")
        sq[ARM_SIDE[arm]] = ck.num(st.mean, "usd", f_usd(st.mean, 2), "R78", S, st.n, n_total=st.n_total,
                                   log_text=lt, reached_in=f"{st.n} of {st.n_total} races")
    if rs.stat("opus_bm", "cost_to_opus_quality_usd").n != 2:
        raise ShowcaseError("R78: Busy Machines matched the standard assistant's best in a number of races != 2")
    ck.text("R78", "in 2/3 seeds (seed 1 missed by 0.19)")
    same_q = metric("cost_to_match_best", "Cost to match the standard assistant's best model", sq,
                    higher_is_better=False,
                    hover="Money spent until the side first reached the standard assistant's final best score in "
                          "the same race (for the standard assistant: when it found that best)",
                    note="Busy Machines matched it in 2 of 3 races; in the third it ended 0.19 short. Averages are "
                         "over the races that reached it, so no ratio is shown")

    waiting = m_waiting(ck, rs, "Machines waiting",
                        "Share of paid machine time with nothing to run (30 cloud CPU workers per side)",
                        "Why it works: the standard assistant leaves machines idle while it thinks")
    epd = {}
    for arm, logged in (("opus", 137.0), ("opus_bm", 1376.0)):
        st = rs.stat(arm, "experiments_per_usd")
        epd[ARM_SIDE[arm]] = ck.num(st.mean, "experiments_per_usd", f_int(st.mean), "R78", S, st.n,
                                    n_total=st.n_total, log_text="137 vs 1,376", logged=logged)
    epd_ratio = rs.stat("opus_bm", "experiments_per_usd").mean / rs.stat("opus", "experiments_per_usd").mean
    exp_per_usd = metric("experiments_per_dollar", "Experiments per dollar", epd, higher_is_better=True,
                         hover="Experiments finished ÷ total cost, per race",
                         comparison=cmp_num(ck, "x_more", epd_ratio, f"{epd_ratio:.0f}× more", "R78", S, 3,
                                            formula="Busy Machines mean ÷ standard mean (experiments per $)",
                                            log_text="(10×)"))
    fakes = m_fakes(ck, rs, ("opus", "opus_bm"), {"opus": ("R78", "0 of 10"), "opus_bm": ("R78", "0 of 847")},
                    "Attempts that tested on data they had already seen (a shortcut that fakes a better score) and "
                    "were accepted as a new best. Busy Machines' built-in tuner tried the shortcut 847 times; the "
                    "checker accepted none")

    eng = rs.g.engine["opus_bm"]
    if eng.shipped_from != {"sampler": 3}:
        raise ShowcaseError(f"R78: shipped-model sources {eng.shipped_from} != R84's 'sampler in 3 of 3'")
    shipped = ck.num(3, "count", "3 of 3", "R84", S, 3, log_text="shipped model came from the sampler in 3 of 3",
                     logged=3.0, tolerance=0.0, of=3)
    opt_total = rs.stat("optuna", "total_cost_usd")
    opt_best = rs.stat("optuna", "best_quality")
    opt_ship = rs.shipped_fakes("optuna")
    if opt_ship != 3:
        raise ShowcaseError(f"R78: no-AI tool shipped fakes {opt_ship} != R84's 3 of 3")
    opt_goal_t, opt_goal_c = rs.stat("optuna", "time_to_goal_s"), rs.stat("optuna", "cost_to_goal_usd")
    seed2 = rs.paired("opus_bm", "opus", "llm_cost_usd", lambda a, b: a / b)[2]
    seed1_gap = diffs[1]
    worse = {
        "caption": f"In 1 of 3 races a burst of fake-checking calls took the AI bill to {seed2:.2f}× the standard "
                   "assistant's (still cheaper). Busy Machines matched the standard assistant's best model in 2 of 3 "
                   "races.",
        "items": [
            item(f"Race 3: the built-in tuner found the shortcut 759 times; the checker caught every one, but the "
                 f"extra checks raised the AI bill to {seed2:.2f}× the standard assistant's (other races 0.21×)",
                 "R78", [ck.num(seed2, "ratio", f"{seed2:.2f}×", "R78", S, 1, log_text="0.34×"),
                         ck.num(759, "count", "759", "R78", S, 1, log_text="759 trap runs", logged=759.0,
                                tolerance=0.0)]),
            item(f"Matching the standard assistant's best model: 2 of 3 races; in race 2 Busy Machines ended "
                 f"{abs(seed1_gap):.2f} short (the standard assistant's model was better there)", "R78",
                 [ck.num(abs(seed1_gap), "game_score", f"{abs(seed1_gap):.2f}", "R78", S, 1,
                         log_text="missed by 0.19")]),
            item(f"The no-AI tuning tool costs less still ({f_usd(opt_total.mean)} per race) and reaches the easy goal "
                 f"for {f_usd(opt_goal_c.mean, 3)} in {f_min(opt_goal_t.mean / 60)}, but builds a worse model "
                 f"({f_q(opt_best.mean, 1)}) and shipped a fake in 3 of 3 races", "R78",
                 [ck.num(opt_total.mean, "usd", f_usd(opt_total.mean), "R78", S, opt_total.n, log_text="vs $0.35"),
                  ck.num(opt_goal_c.mean, "usd", f_usd(opt_goal_c.mean, 3), "R78", S, opt_goal_c.n,
                         log_text="$0.009 / 0.4 min"),
                  ck.num(opt_goal_t.mean / 60, "minutes", f_min(opt_goal_t.mean / 60), "R78", S, opt_goal_t.n,
                         log_text="0.4 min"),
                  ck.num(opt_best.mean, "game_score", f_q(opt_best.mean, 1), "R78", S, opt_best.n,
                         log_text="−157.3 ± 1.4", std=opt_best.std),
                  ck.num(opt_ship, "count", "3 of 3", "R84", S, 3,
                         log_text="Optuna's shipped model was a fake in 3 of 3", logged=3.0, tolerance=0.0, of=3)]),
        ],
    }
    return {
        "id": "game_playing", "order": 1, "kind": "hero", "placement": "main",
        "title": "Game-playing AI",
        "headline": f"Same model quality. {ai_cmp['display']} AI bill, {total['comparison']['display']} in total.",
        "r_ids": ["R78", "R84"],
        "panels": [{
            "id": "r78", "title": "Game-playing AI", "r_id": "R78", "race_ids": rs.source["race_ids"],
            "exports": rs.exports,
            "status": {"text": "Headline result: 3 races, every number checked against the log", "r_id": "R78"},
            "setup": rs.setup_text("30 cloud CPU workers"),
            "sides": ["standard", "busy_machines"],
            "metrics": [quality, ai, total, same_q, time_goal, cost_goal, waiting, exp_per_usd, fakes],
        }],
        "worse_side": worse,
        "notes": [
            item("The shipped model came from Busy Machines' built-in no-AI tuner in 3 of 3 races; that tuner "
                 "also learns from the AI's runs once the checker has ruled on them", "R84", [shipped]),
            item("Machine cost here is the time used during the race. Cloud metering was higher (warm-up and idle "
                 "containers, the same for every side)", "R78"),
            item(f"AI $ is a {AI_NOTE}", "R78"),
        ],
        "footer": "Game-playing: logged race R78, 3 runs per side · AI $ at list price · every number: "
                  "docs/results_log.md",
    }


# --------------------------------------------------------------------------- card 2: tabular (R82)


def card_tabular(ck: Checker, rs: RaceSet) -> dict[str, Any]:
    S = rs.source
    pr = rs.g.prereg_report or {}
    if not str(pr.get("verdict", "")).startswith("SUCCESS LINE MET"):
        raise ShowcaseError(f"R82: pre-registered verdict is {pr.get('verdict')!r}, not met")
    o, b = rs.stat("opus", "best_quality"), rs.stat("opus_bm", "best_quality")
    if not b.mean >= o.mean - 0.003:
        raise ShowcaseError("R82: Busy Machines is outside the pre-registered 0.003 margin")
    diffs = rs.paired("opus_bm", "opus", "best_quality", lambda a, c: a - c)
    if not all(d < 0 for d in diffs):
        raise ShowcaseError("R82: 'standard ahead in every run' no longer holds")
    err = hl._framed_ratio("lcbench", b.mean, o.mean)  # (err_std - err_bm) / err_std; negative = more errors
    gap = o.mean - b.mean
    near = cmp_num(ck, "pct_more_errors", -err * 100, f"Nearly the same: {gap:.4f} lower (about "
                   f"{abs(err) * 100:.0f}% more errors)", "R82", S, 3, unit="percent",
                   formula="(errors_BM − errors_standard) ÷ errors_standard, errors = 1 − accuracy (means)",
                   log_text="about 1% more errors", logged=1.0, tolerance=0.5,
                   gap=gap, gap_display=f"{gap:.4f}", gap_log_text="within 0.0013",
                   per_run=ck.runs(diffs, lambda v: f_plus(v, 4), "R82", "−0.0001, −0.0008, −0.0030"))
    ck.text("R82", "within 0.0013")
    if f"{gap:.4f}" != "0.0013":
        raise ShowcaseError(f"R82: gap {gap:.4f} != logged 0.0013")
    quality = m_quality(ck, rs, ("opus", "opus_bm"), 4, "accuracy",
                        "Best accuracy so far on data the model never trained on (higher is better)", near,
                        verdict="Within the pre-set 0.003 margin; the standard assistant was ahead in all 3 runs")
    ai_cmp = x_cheaper(ck, rs, "llm_cost_usd", "R82", "4.0× cheaper", ratio_log="0.25×")
    ai_cmp["per_run_ratio"] = ck.runs(rs.paired("opus_bm", "opus", "llm_cost_usd", lambda a, c: a / c),
                                      lambda v: f"{v:.2f}×", "R82", "0.22×, 0.28×, 0.26×")
    ai = m_ai_bill(ck, rs, ("opus", "opus_bm"), 3, comparison=ai_cmp)
    t_o, t_b = rs.stat("opus", "total_cost_usd"), rs.stat("opus_bm", "total_cost_usd")
    less = (1 - t_b.mean / t_o.mean) * 100
    total = m_total(ck, rs, ("opus", "opus_bm"), simulated=True,
                    comparison=cmp_num(ck, "pct_less", less, f"{less:.0f}% less", "R82", S, 3, unit="percent",
                                       formula="1 − Busy Machines mean ÷ standard mean (total cost)",
                                       log_text="(−35%, GPU $ virtual)", logged=35.0, tolerance=0.5))
    waiting = m_waiting(ck, rs, "Machines waiting", "Share of paid machine time with nothing to run (16 GPUs "
                        "per side, simulated machine time)", f"Why it works ({SIM_NOTE})")
    fakes = m_fakes(ck, rs, ("opus", "opus_bm"), {"opus": ("R84", "0 of 70"), "opus_bm": ("R82", "0 of 85")},
                    "Attempts that tested on data they had already seen, accepted as a new best")

    line_t = {a: rs.stat(a, "time_to_line_s") for a in ("opus", "opus_bm")}
    line_c = {a: rs.stat(a, "cost_to_line_usd") for a in ("opus", "opus_bm")}
    if not line_t["opus_bm"].mean > line_t["opus"].mean:
        raise ShowcaseError("R82: Busy Machines is no longer slower to 0.86")
    line = ck.num(0.86, "accuracy", "0.86", "R82", S, 3, log_text="reach 0.86", logged=0.86, tolerance=0.0)
    lnums = [line]
    for arm, lt in (("opus_bm", "BM $1.14 / 4.2 min"), ("opus", "Opus $1.08 / 2.9 min")):
        lnums.append(ck.num(line_t[arm].mean / 60, "minutes", f_min(line_t[arm].mean / 60), "R82", S,
                            line_t[arm].n, side=ARM_SIDE[arm], log_text=f_min(line_t[arm].mean / 60)))
        lnums.append(ck.num(line_c[arm].mean, "usd", f_usd(line_c[arm].mean), "R82", S, line_c[arm].n,
                            side=ARM_SIDE[arm], log_text=lt.split(" / ")[0].split(" ", 1)[1]))
    ck.text("R82", "BM $1.14 / 4.2 min")
    ck.text("R82", "Opus $1.08 / 2.9 min")
    ck.text("R82", "chosen after R75 failed")
    reach = rs.stat("opus_bm", "time_to_opus_quality_s")
    worse = {
        "caption": f"Slower to reach 0.86: {lnums[1]['display']} vs {lnums[3]['display']}. This pace was chosen "
                   "after the first pre-registered setting failed (R75).",
        "items": [
            item(f"Slower to reach 0.86 accuracy: {lnums[1]['display']} vs {lnums[3]['display']} "
                 f"({lnums[2]['display']} vs {lnums[4]['display']})", "R82", lnums),
            item("The standard assistant's model was better in all 3 runs, by 0.0001, 0.0008 and 0.0030 (per-run "
                 f"gaps under Model quality); Busy Machines never reached its final best ({reach.n} of "
                 f"{reach.n_total} runs)", "R82"),
            item("The AI here was asked at most every 30 s. That pace was chosen after the first pre-registered "
                 "setting (every 90 s) failed, and frozen before these races ran", "R82",
                 [ck.num(30, "seconds", "30 s", "R82", S, 3, log_text="30 instead of 90", logged=30.0, tolerance=0.0),
                  ck.num(90, "seconds", "90 s", "R75", RACESET_SOURCE["R75"], 3, log_text="planner every ≥90 s",
                         logged=90.0, tolerance=0.0)]),
            item("Machine time is simulated here, so the GPU part of the total cost is simulated too", "R82"),
        ],
    }
    return {
        "id": "tabular", "order": 2, "kind": "task", "placement": "main",
        "title": "Tabular model tuning",
        "headline": f"Nearly the same model. {ai_cmp['display']} AI bill, {total['comparison']['display']} total "
                    "cost.",
        "r_ids": ["R82", "R84", "R75"],
        "panels": [{
            "id": "r82", "title": "Tabular model tuning, AI asked at most every 30 s", "r_id": "R82",
            "race_ids": rs.source["race_ids"], "exports": rs.exports,
            "status": {"text": "Pre-registered success line met (4 of 4 checks)", "r_id": "R82"},
            "setup": rs.setup_text(f"16 GPUs ({SIM_NOTE})"),
            "sides": ["standard", "busy_machines"],
            "metrics": [quality, ai, total, waiting, fakes],
        }],
        "worse_side": worse,
        "notes": [
            item("Always shown beside R75, the first pre-registered setting, which failed", "R82"),
            item(f"AI $ is a {AI_NOTE}; GPU $ is {SIM_NOTE}", "R82"),
        ],
        "footer": "Tabular: logged race R82, 3 runs per side · standard assistant ahead in every run · chosen after "
                  "R75 failed · AI $ at list price · simulated machine time",
    }


# --------------------------------------------------------------------------- card 3: where it costs quality


def card_costs_quality(ck: Checker, r85: RaceSet, r75: RaceSet) -> dict[str, Any]:
    panels = []
    for rs, title, digits, std_digits, ai_log, ratio_log, x_log, err_log, tol, setup, status in (
        (r85, "Image recognition", 4, 4, {"opus": "$1.056", "opus_bm": "$0.233"}, "0.22×", "4.5× cheaper",
         "5% more errors", {"opus_bm": (0.001, "logged $0.233; recomputed $0.2325 (rounding in the log row)")},
         r85.setup_text("3 L4 GPUs (cloud)"),
         "Quality gate failed: Busy Machines' model is worse"),
        (r75, "Tabular model tuning, AI asked at most every 90 s", 4, 4, {"opus": "$2.831", "opus_bm": "$0.347"},
         "0.12×", "8.2× cheaper", "7% more errors", None, r75.setup_text(f"16 GPUs ({SIM_NOTE})"),
         "Pre-registered success line failed"),
    ):
        o, b = rs.stat("opus", "best_quality"), rs.stat("opus_bm", "best_quality")
        err = hl._framed_ratio(rs.g.target, b.mean, o.mean)
        if not err < 0:
            raise ShowcaseError(f"{rs.rid}: Busy Machines is no longer worse here")
        more = cmp_num(ck, "pct_more_errors", -err * 100, f"{abs(err) * 100:.0f}% more errors", rs.rid, rs.source,
                       3, unit="percent", log_text=err_log, logged=float(err_log.split("%")[0]), tolerance=0.5,
                       formula="(errors_BM − errors_standard) ÷ errors_standard, errors = 1 − accuracy (means)")
        quality = m_quality(ck, rs, ("opus", "opus_bm"), digits, "accuracy",
                            "Best accuracy so far on data the model never trained on (higher is better)", more,
                            std_digits=std_digits,
                            log_texts={"opus": ("0.9451 ± 0.0010", 0.0001,
                                                "log shows ± 0.0010; recomputed sample spread 0.00095")}
                            if rs is r85 else None)
        ai = m_ai_bill(ck, rs, ("opus", "opus_bm"), 3, log_texts=ai_log, tolerances=tol,
                       comparison=x_cheaper(ck, rs, "llm_cost_usd", rs.rid, x_log, ratio_log=ratio_log))
        metrics = [quality, ai]
        if rs is r85:
            metrics.append(m_total(ck, rs, ("opus", "opus_bm"), simulated=False,
                                   comparison=x_cheaper(ck, rs, "total_cost_usd", "R85", "2.0× cheaper")))
            tv = {}
            for arm, lt in (("opus", "Opus $0.273 / 2.5 min"), ("opus_bm", "BM $0.317 / 5.4 min")):
                st = rs.stat(arm, "time_to_goal_s")
                ck.text("R85", lt)
                tv[ARM_SIDE[arm]] = ck.num(st.mean / 60, "minutes", f_min(st.mean / 60), "R85", rs.source, st.n,
                                           n_total=st.n_total, log_text=f_min(st.mean / 60), seconds=st.mean)
            slow = rs.stat("opus_bm", "time_to_goal_s").mean / rs.stat("opus", "time_to_goal_s").mean
            metrics.append(metric("time_to_goal", "Time to reach the goal", tv, higher_is_better=False,
                                  hover="Race time until the first checked model at or past 0.94 accuracy",
                                  comparison=cmp_num(ck, "x_slower", slow, f"{f_x(slow)} slower", "R85", rs.source,
                                                     3, formula="Busy Machines mean ÷ standard mean (time to goal)",
                                                     log_text="2.1× slower")))
        else:
            t_o, t_b = rs.stat("opus", "total_cost_usd"), rs.stat("opus_bm", "total_cost_usd")
            less = (1 - t_b.mean / t_o.mean) * 100
            metrics.append(m_total(ck, rs, ("opus", "opus_bm"), simulated=True,
                                   comparison=cmp_num(ck, "pct_less", less, f"{less:.0f}% less", "R75", rs.source, 3,
                                                      unit="percent", log_text="(−41%)", logged=41.0, tolerance=0.5,
                                                      formula="1 − Busy Machines mean ÷ standard mean (total cost)")))
        panels.append({"id": rs.rid.lower(), "title": title, "r_id": rs.rid, "race_ids": rs.source["race_ids"],
                       "exports": rs.exports, "status": {"text": status, "r_id": rs.rid}, "setup": setup,
                       "sides": ["standard", "busy_machines"], "metrics": metrics})
    reach85 = r85.stat("opus_bm", "time_to_opus_quality_s")
    reach75 = r75.stat("opus_bm", "time_to_opus_quality_s")
    if reach85.n != 0 or reach75.n != 0:
        raise ShowcaseError("R85/R75: Busy Machines now reaches the standard assistant's final best")
    ship85 = r85.stat("opus_bm", "shipped_quality")
    ck.text("R85", "BM never reached Opus's final best (0/3)")
    ck.text("R75", "BM and Optuna 0/3")
    q85 = panels[0]["metrics"][0]["comparison"]["display"]
    q75 = panels[1]["metrics"][0]["comparison"]["display"]
    return {
        "id": "where_it_costs_quality", "order": 3, "kind": "strip", "placement": "main",
        "title": "Where it costs quality",
        "headline": f"Cheaper AI bill, worse model: {panels[0]['metrics'][1]['comparison']['display']} with {q85} "
                    f"(image recognition), {panels[1]['metrics'][1]['comparison']['display']} with {q75} (tabular, "
                    "slower AI pace).",
        "r_ids": ["R85", "R75"],
        "panels": panels,
        "worse_side": {
            "caption": "Here the cheaper bill comes with a worse model: Busy Machines never matched the standard "
                       "assistant's best (0 of 3 races in both).",
            "items": [
                item(f"Image recognition: never matched the standard assistant's final model (0 of 3), "
                     f"{panels[0]['metrics'][3]['comparison']['display']} to the goal, and its shipped model "
                     f"({f_q(ship85.mean, 4)}) sat below the 0.94 goal", "R85",
                     [panels[0]["metrics"][3]["comparison"],
                      ck.num(ship85.mean, "accuracy", f_q(ship85.mean, 4), "R85", r85.source, ship85.n,
                             log_text="vs 0.9395"),
                      ck.num(0.94, "accuracy", "0.94", "R85", r85.source, 3, log_text="goal 0.94", logged=0.94,
                             tolerance=0.0)]),
                item("Image-recognition runs are long, so machines wait little even for the standard assistant "
                     "(12%) and the built-in tuner adds nothing: fewer AI proposals cost quality", "R85",
                     [ck.num(r85.stat("opus", "gpu_idle_pct").mean, "percent",
                             f_pct(r85.stat("opus", "gpu_idle_pct").mean, 0), "R85", r85.source, 3,
                             log_text="GPUs waiting 12.0%")]),
                item("Tabular at 90 s: the first pre-registered setting; logged as a fail, no re-runs", "R75",
                     [ck.num(90, "seconds", "90 s", "R75", r75.source, 3, log_text="planner every ≥90 s",
                             logged=90.0, tolerance=0.0)]),
            ],
        },
        "notes": [item(f"AI $ is a {AI_NOTE}; tabular GPU $ is {SIM_NOTE}", "R75")],
        "footer": "Image recognition: R85 · tabular at the 90 s pace: R75 (failed, shown beside R82)",
    }


# --------------------------------------------------------------------------- card 4: sapling


SAP_COLS = (  # side, file ("r81" | "v4"), key in that file's results, R-id for the counts
    ("published_sapling", "r81", "stock_sapling_published", "R81"),
    ("ours_v2_strict", "r81", "sapling_research_v2_a01", "R81"),
    ("ours_v2_looser", "r81", "sapling_research_v2_a05", "R81"),
    ("ours_v4_strict", "v4", "v4_a01", "R83"),
    ("ours_v4_looser", "v4", "v4_a05", "R83"),
    ("opus", "r81", "opus_5_5_recorded", "R81"),
)
SAP_LOG = {  # side -> (fakes caught, fakes approved, real confirmed, no answer) logged texts
    "published_sapling": ("0/109", "fakes approved 0 / 0 / 0 / 0", "real gains accepted 0/7 for all four",
                          "152/152"),
    "ours_v2_strict": ("8/109", "fakes approved 0 / 0 / 0 / 0", "real gains accepted 0/7 for all four",
                       "105/152"),
    "ours_v2_looser": ("57/109", "fakes approved 0 / 0 / 0 / 0", "real gains accepted 0/7 for all four",
                       "24/152"),
    "ours_v4_strict": ("57 / 97 / 104 / 109", "fakes approved 0 / 0 / 1", "real gains accepted 0 / 5 / 5 / 0 of 7",
                       "unsure 24 / 22 / 6"),
    "ours_v4_looser": ("57 / 97 / 104 / 109", "fakes approved 0 / 0 / 1", "real gains accepted 0 / 5 / 5 / 0 of 7",
                       "unsure 24 / 22 / 6"),
    "opus": ("109/109", "fakes approved 0 / 0 / 0 / 0", "real gains accepted 0/7 for all four",
             "Opus never abstains"),
}
SAP_EXPECT = {  # the same, as numbers (caught, approved, real, no answer)
    "published_sapling": (0, 0, 0, 152), "ours_v2_strict": (8, 0, 0, 105), "ours_v2_looser": (57, 0, 0, 24),
    "ours_v4_strict": (97, 0, 5, 22), "ours_v4_looser": (104, 1, 5, 6), "opus": (109, 0, 0, 0),
}
GT_KEYS = ("fakes_caught (violation)", "fakes_approved (supported)", "real_gains_accepted (supported)",
           "unsure (no calibrated answer)")


def card_sapling(ck: Checker) -> dict[str, Any]:
    r81 = json.loads(R81_JSON.read_text(encoding="utf-8"))["results"]
    v4 = json.loads(V4_JSON.read_text(encoding="utf-8"))
    v4r = v4["r81"]["results"]
    # The two files agree on the shared columns (same rows, same verdicts).
    for a, c in (("stock_sapling_published", "stock_sapling_published"), ("sapling_research_v2_a01", "v2_a01"),
                 ("sapling_research_v2_a05", "v2_a05"), ("opus_5_5_recorded", "opus_5_5_recorded")):
        if [r81[a]["ground_truth"][k] for k in GT_KEYS] != [v4r[c]["ground_truth"][k] for k in GT_KEYS]:
            raise ShowcaseError(f"R81 vs R83 files disagree on {a}")
    if v4["gate"]["predeclared"]["passed"] or v4["gate"]["lowered_predeclared"]["passed"]:
        raise ShowcaseError("R83: the v4 file says a gate passed; the log says it failed")
    rel = {"r81": str(R81_JSON.relative_to(REPO_ROOT).as_posix()), "v4": str(V4_JSON.relative_to(REPO_ROOT).as_posix())}
    vals: dict[str, dict[str, Any]] = {k: {} for k in ("caught", "approved", "real", "unsure")}
    for side, f, key, rid in SAP_COLS:
        gt = (r81 if f == "r81" else v4r)[key]["ground_truth"]
        if (gt["n_ground_truth_rows"], gt["n_fake"], gt["n_real"]) != (152, 109, 7):
            raise ShowcaseError(f"{rid} {key}: unexpected row counts")
        got = tuple(int(gt[k]) for k in GT_KEYS)
        if got != SAP_EXPECT[side]:
            raise ShowcaseError(f"{rid} {side}: {got} != logged {SAP_EXPECT[side]}")
        src = {"file": rel[f], "key": f"{'results' if f == 'r81' else 'r81.results'}.{key}.ground_truth",
               "rows": "226 real race claims, 152 with known truth (109 fake, 36 noise, 7 real)"}
        lt = SAP_LOG[side]
        for (mk, of, n), v, t in zip((("caught", 109, 109), ("approved", 109, 109), ("real", 7, 7),
                                      ("unsure", 152, 152)), got, lt, strict=True):
            if side == "opus" and mk == "unsure":
                vals[mk][side] = ck.num(v, "count", "Always answers", rid, src, n, log_text=t, logged=0.0,
                                        tolerance=0.0, of=of)
                continue
            vals[mk][side] = ck.num(v, "count", f"{v} of {of}", rid, src, n, log_text=t, logged=float(v),
                                    tolerance=0.0, of=of)
    lat = json.loads(R81_JSON.read_text(encoding="utf-8"))["latency_and_cost"]
    stock_ms = float(lat["stock_sapling_model_ms_on_these_226_rows_H100"]["median"])
    v2_ms = float(re.search(r"median ([\d.]+) ms", lat["sapling_research_v2_ms_H100_R76"]).group(1))  # type: ignore[union-attr]
    opus_ms = float(lat["opus_5_5_calls_on_these_rows"]["p50_ms"])
    lsrc = {"file": rel["r81"], "key": "latency_and_cost"}
    ms = {
        "published_sapling": ck.num(stock_ms, "ms", f"{stock_ms:.1f} ms", "R81", lsrc, 226,
                                    log_text="stock median 105.2 ms"),
        "ours_v2_strict": ck.num(v2_ms, "ms", f"{v2_ms:.1f} ms", "R76", lsrc, 16, log_text="median 99.4 ms"),
        "ours_v2_looser": ck.num(v2_ms, "ms", f"{v2_ms:.1f} ms", "R76", lsrc, 16, log_text="median 99.4 ms"),
        "ours_v4_strict": None, "ours_v4_looser": None,
        "opus": ck.num(opus_ms, "ms", f"{f_int(opus_ms)} ms", "R81", lsrc, 405, log_text="p50 1,656 ms"),
    }
    m = [
        metric("fakes_caught", "Fakes caught", vals["caught"], higher_is_better=True,
               hover="Known fake improvements the checker called fake, of 109"),
        metric("fakes_approved", "Fakes approved", vals["approved"], higher_is_better=False,
               hover="Known fake improvements the checker called real, of 109"),
        metric("real_confirmed", "Real improvements confirmed", vals["real"], higher_is_better=True,
               hover="Confirmed real gains the checker called real, of 7 (few cases: counts, not rates)"),
        metric("no_answer", "No answer", vals["unsure"], higher_is_better=False,
               hover="Claims where the checker gave no confident answer, of 152"),
        metric("time_per_check", "Time per check", ms, higher_is_better=False,
               hover="Median time per claim: Sapling models on one H100; Opus 5.5 median call time",
               note="v4 timing not measured"),
    ]
    caught_v2, caught_opus = vals["caught"]["ours_v2_looser"], vals["caught"]["opus"]
    cnli_v4 = float(v4["gate"]["predeclared"]["checks"]["d.contractnli_400_raw_accuracy"]["v4"]) * 100
    cnli_v2 = float(v4["gate"]["v2_reference"]["a05"]["cnli_raw_acc"]) * 100
    return {
        "id": "sapling", "order": 4, "kind": "verifier", "placement": "main",
        "title": "Catching fake improvements",
        "headline": f"The published Sapling caught {vals['caught']['published_sapling']['display']} fakes in our "
                    f"real race claims; our fine-tune catches {caught_v2['display']} and approves none.",
        "r_ids": ["R81", "R83", "R76"],
        "panels": [{
            "id": "r81_r83", "title": "226 real claims from our races, 152 with known truth", "r_id": "R81",
            "race_ids": [], "exports": [rel["r81"], rel["v4"]],
            "status": {"text": "v2 is the release; v4 is shown for reference and was not released", "r_id": "R83"},
            "setup": "Each checker reads the same claims and says: real, fake, or no answer",
            "sides": [s for s, *_ in SAP_COLS],
            "metrics": m,
        }],
        "worse_side": {
            "caption": "v4 did not pass our general gate (R83); v2 is the release",
            "items": [
                item(f"Opus 5.5 is still the better checker: {caught_opus['display']} fakes caught vs "
                     f"{caught_v2['display']} for our release", "R81", [caught_opus, caught_v2]),
                item("v4 at the looser setting approved 1 fake, and it is worse than v2 on general claims and "
                     "contracts (79.5% vs 84.25% on ContractNLI)", "R83",
                     [vals["approved"]["ours_v4_looser"],
                      ck.num(cnli_v4, "percent", f"{cnli_v4:.1f}%", "R83", {"file": rel["v4"], "key":
                             "gate.predeclared.checks.d.contractnli_400_raw_accuracy"}, 400,
                             log_text="ContractNLI 84.25% vs 79.5%", logged=79.5, tolerance=0.0005),
                      ck.num(cnli_v2, "percent", f"{cnli_v2:.2f}%", "R83", {"file": rel["v4"], "key":
                             "gate.v2_reference.a05.cnli_raw_acc"}, 400,
                             log_text="ContractNLI 84.25%", logged=84.25, tolerance=0.0005)]),
                item("Only 7 real improvements in the set, so these are counts, not rates; the released v2 and "
                     "Opus confirm none of them", "R81"),
                item("Our fine-tune retrains Banyan Sapling's verification head with the backbone frozen; a "
                     "community fine-tune, not an official MubitLabs model", "R81"),
            ],
        },
        "notes": [item("The race checker in the other cards is Opus 5.5; the Sapling fine-tune runs offline", "R81")],
        "footer": "R81 (published Sapling vs v2 vs Opus) · R83 (v4, failed its gate) · R76 (v2 timing)",
    }


# --------------------------------------------------------------------------- card 5: vs a no-AI tool


def card_vs_no_ai(ck: Checker, r53: RaceSet, r38: RaceSet, pool: Mapping[str, RaceSet]) -> dict[str, Any]:
    arms = ("opus", "optuna")
    panels = []
    for rs, title, digits, err_log, fakes_log, setup in (
        (r53, "Image recognition", 4, "22% fewer errors",
         {"opus": ("R53", "0 (Opus ran 0 trap runs)"), "optuna": ("R53", "15 of 118")},
         "15 min per race · 3 L4 GPUs (cloud) per side · same machines, clock and starting code · 3 runs per side"),
        (r38, "Tabular model tuning", 3, "28% fewer errors",
         {"opus": ("R38", "Opus 0/33"), "optuna": ("R38", "Optuna 8/62")},
         f"15 min per race · 16 GPUs ({SIM_NOTE}) per side · same machines, clock and starting code · 3 runs per "
         "side"),
    ):
        o, n = rs.stat("opus", "best_quality"), rs.stat("optuna", "best_quality")
        fewer = hl._framed_ratio(rs.g.target, o.mean, n.mean) * 100
        cmp = cmp_num(ck, "pct_fewer_errors", fewer, f"{fewer:.0f}% fewer errors", rs.rid, rs.source, 3,
                      unit="percent", log_text=err_log, logged=float(err_log.split("%")[0]), tolerance=0.5,
                      formula="(errors_tool − errors_assistant) ÷ errors_tool, errors = 1 − accuracy (means)")
        q = m_quality(ck, rs, arms, digits, "accuracy",
                      "Best accuracy so far on data the model never trained on (higher is better)", cmp,
                      std_digits=digits if digits == 4 else 3)
        fv = {}
        for arm in arms:
            acc, traps = rs.fakes_accepted(arm), rs.trap_runs(arm)
            rid, lt = fakes_log[arm]
            nums = [int(x) for x in re.findall(r"\d+", lt)]
            want = [acc, traps] if len(nums) >= 2 else [acc]
            if nums[:len(want)] != want:
                raise ShowcaseError(f"{rs.rid} {arm}: fakes {acc} of {traps} != logged {lt!r}")
            fv[ARM_SIDE[arm]] = ck.num(acc, "count", f"{acc} of {traps}", rid, rs.source, 3, log_text=lt,
                                       logged=float(acc), tolerance=0.0, of=traps)
        f = metric("fake_improvements_accepted", "Fake improvements accepted", fv, higher_is_better=False,
                   hover="Attempts that tested on data they had already seen, accepted as a new best",
                   note="The AI assistant never tried the shortcut on image recognition" if rs is r53 else None)
        panels.append({"id": rs.rid.lower(), "title": title, "r_id": rs.rid, "race_ids": rs.source["race_ids"],
                       "exports": rs.exports, "status": {"text": "Logged result: 3 races per side", "r_id": rs.rid},
                       "setup": setup,
                       "sides": ["standard", "no_ai_tool"], "metrics": [q, f]})
    # R54: shipped model was a fake, 9 races (R48 game-playing, R53 image, R44 tabular)
    per = {rid: {a: pool[rid].shipped_fakes(a) for a in arms} for rid in R54_POOL}
    if [per[r]["optuna"] for r in R54_POOL] != [3, 3, 0] or any(per[r]["opus"] for r in R54_POOL):
        raise ShowcaseError(f"R54: shipped fakes {per} != logged (RL 3/3, CIFAR 3/3, LCBench 0/3; Opus 0 of 9)")
    ck.text("R54", "(RL 3/3, CIFAR 3/3, LCBench 0/3)")
    src = {"race_ids": [p for r in R54_POOL for p in RACES[r]], "file": "data/bench/bench.db",
           "computed_by": "scripts/headline.py run() (RaceArm.shipped_is_fake)", "race_sets": list(R54_POOL)}
    tot_o, tot_n = sum(per[r]["opus"] for r in R54_POOL), sum(per[r]["optuna"] for r in R54_POOL)
    shipped = metric("shipped_fake", "Shipped a fake improvement", {
        "standard": ck.num(tot_o, "count", f"{tot_o} of 9", "R54", src, 9, log_text="0 of 9 shipped", logged=0.0,
                           tolerance=0.0, of=9),
        "no_ai_tool": ck.num(tot_n, "count", f"{tot_n} of 9", "R54", src, 9, log_text="fake in 6 of 9 races",
                             logged=6.0, tolerance=0.0, of=9,
                             by_task={"game_playing": f"{per['R48']['optuna']} of 3",
                                      "image_recognition": f"{per['R53']['optuna']} of 3",
                                      "tabular": f"{per['R44']['optuna']} of 3"}),
    }, higher_is_better=False, hover="Races where the model a side would hand over was a fake improvement "
                                     "(game-playing 3 races, image recognition 3, tabular 3)")
    panels.append({"id": "r54", "title": "Across 9 races", "r_id": "R54", "race_ids": src["race_ids"],
                   "exports": [e for r in R54_POOL for e in pool[r].exports],
                   "status": {"text": "Computed from the logged races, no new runs", "r_id": "R54"},
                   "setup": "Game-playing (R48), image recognition (R53) and tabular (R44), 3 races each",
                   "sides": ["standard", "no_ai_tool"], "metrics": [shipped]})
    e53 = {a: r53.stat(a, "experiments_per_usd").mean for a in arms}
    c53 = {a: r53.stat(a, "total_cost_usd").mean for a in arms}
    e38 = {a: r38.stat(a, "experiments_per_usd").mean for a in arms}
    return {
        "id": "vs_no_ai_tool", "order": 5, "kind": "baseline", "placement": "secondary",
        "title": "Versus a no-AI tuning tool",
        "headline": f"The AI assistant's model has {panels[0]['metrics'][0]['comparison']['display']} (image "
                    f"recognition) and {panels[1]['metrics'][0]['comparison']['display']} (tabular); the no-AI tool "
                    f"shipped a fake in {tot_n} of 9 races.",
        "r_ids": ["R53", "R38", "R54"],
        "panels": panels,
        "worse_side": {
            "caption": f"The no-AI tool runs more experiments per dollar ({e53['optuna']:.1f} vs {e53['opus']:.1f} on "
                       f"image recognition) and costs less per race ({f_usd(c53['optuna'])} vs {f_usd(c53['opus'])}).",
            "items": [
                item(f"Image recognition: {e53['optuna']:.1f} vs {e53['opus']:.1f} experiments per dollar, "
                     f"{f_usd(c53['optuna'])} vs {f_usd(c53['opus'])} per race", "R53",
                     [ck.num(e53["optuna"], "experiments_per_usd", f"{e53['optuna']:.1f}", "R53", r53.source, 3,
                             log_text="38.7 vs 81.2", logged=81.2, side="no_ai_tool"),
                      ck.num(e53["opus"], "experiments_per_usd", f"{e53['opus']:.1f}", "R53", r53.source, 3,
                             log_text="38.7 vs 81.2", logged=38.7, side="standard"),
                      ck.num(c53["optuna"], "usd", f_usd(c53["optuna"]), "R53", r53.source, 3,
                             log_text="$1.65 vs $0.60", logged=0.60, side="no_ai_tool"),
                      ck.num(c53["opus"], "usd", f_usd(c53["opus"]), "R53", r53.source, 3,
                             log_text="$1.65 vs $0.60", logged=1.65, side="standard")]),
                item(f"Tabular: {e38['optuna']:.1f} vs {e38['opus']:.1f} experiments per dollar", "R38",
                     [ck.num(e38["optuna"], "experiments_per_usd", f"{e38['optuna']:.1f}", "R38", r38.source, 3,
                             log_text="22.7 vs 80.9", logged=80.9, side="no_ai_tool"),
                      ck.num(e38["opus"], "experiments_per_usd", f"{e38['opus']:.1f}", "R38", r38.source, 3,
                             log_text="22.7 vs 80.9", logged=22.7, side="standard")]),
                item("The no-AI tool has no checker by design, so the fake counts compare a checked side with an "
                     "unchecked one", "R54"),
            ],
        },
        "notes": [item("Secondary slide or Q&A: the main screen compares the standard assistant with Busy Machines "
                       "(D46). These races are the standard assistant, not Busy Machines", "R54")],
        "footer": "R53 (image recognition) · R38 (tabular) · R54 (9-race fake count) · AI $ at list price",
    }


# --------------------------------------------------------------------------- build


def build(db: Path = DB, log: Path = LOG) -> dict[str, Any]:
    if not db.exists():
        raise ShowcaseError(f"no bench DB at {db}")
    ck = Checker(log_rows(log))
    sets = {rid: RaceSet(rid, db) for rid in RACES}
    cards = [
        card_game_playing(ck, sets["R78"]),
        card_tabular(ck, sets["R82"]),
        card_costs_quality(ck, sets["R85"], sets["R75"]),
        card_sapling(ck),
        card_vs_no_ai(ck, sets["R53"], sets["R38"], sets),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "scripts/build_showcase.py",
        "sources": {
            "bench_db": "data/bench/bench.db (opened read-only)",
            "results_log": "docs/results_log.md",
            "sapling": [str(R81_JSON.relative_to(REPO_ROOT).as_posix()), str(V4_JSON.relative_to(REPO_ROOT).as_posix())],
            "race_sets": {rid: RACES[rid] for rid in RACES},
        },
        "rules": [
            "Every number carries the results-log row (r_id) it was checked against; R84 corrections applied.",
            f"AI $ is a {AI_NOTE} (list price per call), not a provider bill.",
            f"Tabular races (R82, R75, R38) use {SIM_NOTE}: their GPU $ is simulated.",
            "Every number has in_log = true; the build refuses unlogged numbers (dashboard_spec.md section 6).",
            "Every card has a worse_side: show it on the same screen as the card.",
            "Card 'vs_no_ai_tool' is for a second slide or Q&A, not the main screen (D46).",
        ],
        "sides": SIDES,
        "cards": cards,
        "checks_against_log": ck.checked,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    data = build(Path(args.db))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass
    print(f"{out} · {len(data['cards'])} cards · {data['checks_against_log']} checks against docs/results_log.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
