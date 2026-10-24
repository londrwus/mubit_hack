"""data/frontend/showcase.json (built by scripts/build_showcase.py): schema, honesty and log checks.

Reads the committed JSON and docs/results_log.md only (no LLM, no Modal). The last test rebuilds the
JSON from the read-only bench DB (1-2.5 min): it runs only with SHOWCASE_REBUILD=1 and is skipped when the
DB or onnxruntime (LCBench) is missing.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
SHOWCASE = REPO / "data" / "frontend" / "showcase.json"
LOG = REPO / "docs" / "results_log.md"
DB = REPO / "data" / "bench" / "bench.db"

CARD_ORDER = ["game_playing", "tabular", "where_it_costs_quality", "sapling", "vs_no_ai_tool"]
# Kept separate from the builder on purpose: a new label has to be added here too.
LABELS = {
    "Model quality", "AI bill per race", "Total cost per race", "Time to reach the goal", "Cost to reach the goal",
    "Cost to match the standard assistant's best model", "GPUs waiting", "Machines waiting",
    "Experiments per dollar", "Fake improvements accepted", "Shipped a fake improvement",
    "Fakes caught", "Fakes approved", "Real improvements confirmed", "No answer", "Time per check",
}
UNITS = {"usd", "minutes", "seconds", "percent", "game_score", "accuracy", "count", "ratio", "experiments_per_usd",
         "ms"}
# dashboard_spec.md section 6: never as a label or caption (raw names only on hover).
JARGON_I = re.compile(r"\b(adapters?|val_bpb|held-out|idle %|utili[sz]ation|sampler|traps?|surrogate|fingerprints?|"
                      r"judge|propose|optuna|seeds?|seed spread|arms?)\b", re.IGNORECASE)
JARGON_CS = re.compile(r"\b(RL|PPO)\b")
TOKEN = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


@pytest.fixture(scope="module")
def data() -> dict[str, Any]:
    return json.loads(SHOWCASE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rows() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in LOG.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\| (R\d+) \|", line)
        if m:
            out[m.group(1)] = line.replace("**", "")
    return out


def numbers(obj: Any) -> Iterator[dict[str, Any]]:
    """Every number object (a dict with a display string and a unit), anywhere in ``obj``."""
    if isinstance(obj, dict):
        if "display" in obj and "unit" in obj:
            yield obj
        for v in obj.values():
            yield from numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from numbers(v)


def strings(obj: Any, keys: set[str]) -> Iterator[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, str):
                yield v
            yield from strings(v, keys)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v, keys)


def user_text(card: dict[str, Any]) -> list[str]:
    """Strings a viewer reads on the card (hover, source and formula fields excluded)."""
    out = [card["title"], card["headline"], card["footer"], card["worse_side"]["caption"]]
    out += [i["text"] for i in card["worse_side"]["items"] + card["notes"]]
    for p in card["panels"]:
        out += [p["title"], p["status"]["text"], p["setup"]]
        for m in p["metrics"]:
            out += [m["label"], m.get("verdict") or "", m.get("note") or ""]
            out += [v["display"] for v in m["values"].values() if v]
            if m["comparison"] and m["comparison"]["in_log"]:
                out.append(m["comparison"]["display"])
    return out


# --------------------------------------------------------------------------- schema


def test_top_level_schema(data: dict[str, Any]) -> None:
    assert data["schema_version"] == 1
    assert data["generated_by"] == "scripts/build_showcase.py"
    assert {"sides", "cards", "rules", "sources"} <= data.keys()
    assert data["checks_against_log"] > 100
    for side in data["sides"].values():
        assert side["label"] and side["sublabel"] and side["hover"]


def test_cards_in_order_with_their_parts(data: dict[str, Any]) -> None:
    assert [c["id"] for c in data["cards"]] == CARD_ORDER
    assert [c["order"] for c in data["cards"]] == [1, 2, 3, 4, 5]
    assert data["cards"][0]["kind"] == "hero"
    assert data["cards"][-1]["placement"] == "secondary"  # D46: not on the main screen
    for c in data["cards"]:
        assert {"id", "title", "headline", "r_ids", "panels", "worse_side", "notes", "footer"} <= c.keys()
        assert c["panels"]
        for p in c["panels"]:
            assert p["r_id"] in c["r_ids"] and p["status"]["r_id"]
            assert set(p["sides"]) <= data["sides"].keys()
            assert p["metrics"]
            for m in p["metrics"]:
                assert set(m["values"]) <= set(p["sides"]), (c["id"], m["id"])
                assert any(v is not None for v in m["values"].values())


def test_every_card_has_a_worse_side(data: dict[str, Any], rows: dict[str, str]) -> None:
    for c in data["cards"]:
        w = c["worse_side"]
        assert w["caption"].strip(), c["id"]
        assert w["items"], c["id"]
        for i in w["items"] + c["notes"]:
            assert i["text"].strip() and i["r_id"] in rows, (c["id"], i)


def test_required_worse_side_facts(data: dict[str, Any]) -> None:
    cards = {c["id"]: c for c in data["cards"]}
    assert "0.34×" in cards["game_playing"]["worse_side"]["caption"]
    assert "2 of 3" in cards["game_playing"]["worse_side"]["caption"]
    tab = cards["tabular"]["worse_side"]["caption"]
    assert "4.2 min vs 2.9 min" in tab and "R75" in tab
    assert cards["sapling"]["worse_side"]["caption"] == "v4 did not pass our general gate (R83); v2 is the release"


# --------------------------------------------------------------------------- numbers vs the log


def test_every_number_has_an_r_id_source_and_n(data: dict[str, Any], rows: dict[str, str]) -> None:
    nums = list(numbers(data["cards"]))
    assert len(nums) > 100
    for n in nums:
        assert re.fullmatch(r"R\d+", n["r_id"]) and n["r_id"] in rows, n
        assert n["unit"] in UNITS, n
        assert isinstance(n["value"], (int, float)) and math.isfinite(n["value"]), n
        assert isinstance(n["n"], int) and n["n"] >= 1, n
        assert isinstance(n["in_log"], bool), n
        src = n["source"]
        assert (REPO / src["file"]).exists(), src
        if src["file"].endswith("bench.db"):
            assert src["race_ids"] and all(re.fullmatch(r"[0-9a-f]{8}", r) for r in src["race_ids"]), src


def test_logged_numbers_match_the_log_text(data: dict[str, Any], rows: dict[str, str]) -> None:
    for n in numbers(data["cards"]):
        if not n["in_log"]:
            continue
        row = rows[n["r_id"]]
        assert n["log_text"] and n["log_text"] in row, (n["r_id"], n["log_text"])
        if "logged_value" in n:
            assert abs(n["value"] - n["logged_value"]) <= n["tolerance"] + 1e-9, n
        if "logged_std" in n:
            assert abs(n["std"] - n["logged_std"]) <= n["std_tolerance"] + 1e-9, n
        for key in ("per_run", "per_run_ratio"):
            pr = n.get(key)
            if isinstance(pr, dict):  # a logged per-run list, e.g. '0.21×, 0.21×, 0.34×'
                assert pr["log_text"] in rows[pr["r_id"]] and pr["display"] == pr["log_text"]


def test_no_unlogged_numbers(data: dict[str, Any]) -> None:
    # dashboard_spec.md section 6: no unlogged ratios or numbers, not even flagged ones.
    unlogged = [n for n in numbers(data["cards"]) if not n["in_log"]]
    assert unlogged == []
    assert any("in_log" in r for r in data["rules"])


def test_no_gpu_label_for_cpu_or_simulated_races(data: dict[str, Any]) -> None:
    # dashboard_spec.md section 3.1 / 6: R78 ran on CPU workers, R82's GPU time is simulated.
    for c in data["cards"]:
        for p in c["panels"]:
            if p["r_id"] in ("R78", "R82"):
                assert all("GPU" not in m["label"] for m in p["metrics"]), (p["r_id"], [m["label"] for m in p["metrics"]])


def test_headline_numbers(data: dict[str, Any]) -> None:
    def metric(card: str, panel: int, mid: str) -> dict[str, Any]:
        c = next(c for c in data["cards"] if c["id"] == card)
        return next(m for m in c["panels"][panel]["metrics"] if m["id"] == mid)

    ai = metric("game_playing", 0, "ai_bill_per_race")
    assert (ai["values"]["standard"]["display"], ai["values"]["busy_machines"]["display"]) == ("$3.30", "$0.83")
    assert ai["comparison"]["display"] == "4.0× cheaper"
    assert metric("game_playing", 0, "total_cost_per_race")["comparison"]["display"] == "3.1× cheaper"
    assert metric("game_playing", 0, "model_quality")["comparison"]["display"] == "Same: a tie within noise"
    assert metric("game_playing", 0, "fake_improvements_accepted")["values"]["busy_machines"]["display"] == "0 of 847"
    assert metric("game_playing", 0, "experiments_per_dollar")["comparison"]["display"] == "10× more"
    assert metric("tabular", 0, "total_cost_per_race")["comparison"]["display"] == "35% less"
    assert metric("where_it_costs_quality", 0, "ai_bill_per_race")["comparison"]["display"] == "4.5× cheaper"
    assert metric("where_it_costs_quality", 0, "model_quality")["comparison"]["display"] == "5% more errors"
    assert metric("where_it_costs_quality", 1, "ai_bill_per_race")["comparison"]["display"] == "8.2× cheaper"
    assert metric("where_it_costs_quality", 1, "model_quality")["comparison"]["display"] == "7% more errors"
    caught = metric("sapling", 0, "fakes_caught")["values"]
    assert [caught[k]["value"] for k in ("published_sapling", "ours_v2_strict", "ours_v2_looser", "ours_v4_strict",
                                         "opus")] == [0, 8, 57, 97, 109]
    assert metric("vs_no_ai_tool", 2, "shipped_fake")["values"]["no_ai_tool"]["display"] == "6 of 9"


def test_cost_notes(data: dict[str, Any]) -> None:
    for c in data["cards"]:
        for p in c["panels"]:
            for m in p["metrics"]:
                if m["id"] == "ai_bill_per_race":
                    assert m["note"] == "list-price estimate"
                    assert all(v["note"] == "list-price estimate" for v in m["values"].values() if v)
            if p["r_id"] in ("R82", "R75", "R38"):  # simulated machine time
                assert "simulated GPU time" in p["setup"], p["id"]
                for m in p["metrics"]:
                    if m["id"] == "total_cost_per_race":
                        assert "simulated GPU time" in m["note"]


# --------------------------------------------------------------------------- plain language


def test_plain_language_labels(data: dict[str, Any]) -> None:
    labels = {m["label"] for c in data["cards"] for p in c["panels"] for m in p["metrics"]}
    assert labels <= LABELS, labels - LABELS
    sides = [s["label"] for s in data["sides"].values()] + [s["sublabel"] for s in data["sides"].values()]
    for c in data["cards"]:
        for text in user_text(c) + sides:
            assert not JARGON_I.search(text) and not JARGON_CS.search(text), (c["id"], text)


def test_numbers_in_card_text_are_backed(data: dict[str, Any]) -> None:
    """Every number written in a headline, caption, item or note appears among the card's checked numbers."""
    keys = {"display", "log_text", "std_display", "gap_display", "ratio_display"}
    for c in data["cards"]:
        backed = " | ".join(strings(c["panels"], keys)) + " | " + " | ".join(
            strings([i["numbers"] for i in c["worse_side"]["items"] + c["notes"]], keys))
        texts = [c["headline"], c["worse_side"]["caption"]] + [i["text"] for i in c["worse_side"]["items"] + c["notes"]]
        for t in texts:
            for tok in TOKEN.findall(t.replace("Opus 5.5", "Opus")):  # the model name is not a result
                if "." not in tok and "," not in tok and int(tok) <= 10:
                    continue  # small counts ('2 of 3 races')
                assert tok in backed, (c["id"], tok, t)


# --------------------------------------------------------------------------- the builder


def _builder() -> Any:
    spec = importlib.util.spec_from_file_location("build_showcase", REPO / "scripts" / "build_showcase.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("build_showcase", mod)
    spec.loader.exec_module(mod)
    return mod


def test_checker_refuses_unlogged_or_wrong_numbers() -> None:
    bs = _builder()
    ck = bs.Checker({"R1": "| R1 | ... LLM $ per race $3.30 vs $0.83 (4.0× cheaper) · p50 1,656 ms |"})
    ok = ck.num(3.2997, "usd", "$3.30", "R1", {"file": "x"}, 3, log_text="$3.30")
    assert ok["logged_value"] == 3.30 and ok["tolerance"] == pytest.approx(0.005)
    assert ck.num(1655.7, "ms", "1,656 ms", "R1", {"file": "x"}, 1, log_text="p50 1,656 ms")["logged_value"] == 1656
    with pytest.raises(SystemExit):
        ck.num(3.31, "usd", "$3.31", "R1", {"file": "x"}, 3, log_text="$3.31")      # not in the row
    with pytest.raises(SystemExit):
        ck.num(3.40, "usd", "$3.40", "R1", {"file": "x"}, 3, log_text="$3.30")      # does not round to it
    with pytest.raises(SystemExit):
        ck.num(1.0, "usd", "$1.00", "R2", {"file": "x"}, 3, log_text="$1.00")       # no such row
    with pytest.raises(SystemExit):
        ck.num(3.6, "ratio", "3.6× faster", "R1", {"file": "x"}, 3, log_text=None, in_log=False)  # unlogged


@pytest.mark.skipif(not DB.exists(), reason="no data/bench/bench.db")
@pytest.mark.skipif(os.environ.get("SHOWCASE_REBUILD") != "1", reason="1-2.5 min; set SHOWCASE_REBUILD=1 to run")
def test_rebuild_matches_the_committed_file(data: dict[str, Any]) -> None:
    pytest.importorskip("onnxruntime")  # LCBench races need it
    fresh = json.loads(json.dumps(_builder().build(), ensure_ascii=False))
    assert fresh == data
