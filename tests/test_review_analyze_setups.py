"""Review fix: analyze_races never drops a target silently and never pools unlike setups.

Before the fix (1) a target whose races ran with different goals, or had no goal, was skipped
with a warning that was never printed and was not listed under "Excluded from this analysis";
(2) every race of a target was pooled into one mean whatever its setup (the LCBench section
pooled 11 races from 4 setups, "11 seed(s)" counting races, with ready-to-paste results_log
rows and charts). Synthetic DBs only; no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.db import Database
from test_analyze_races import TV, _fairness, _race, ar


def _db(tmp_path: Path, races: list[tuple[str, int, dict | None, dict | None]]) -> Path:
    path = tmp_path / "bench.db"
    db = Database(path)
    for rid, seed, config, fairness in races:
        _race(db, rid, seed, config=config)
        if fairness is not None:
            db.update_race(rid, fairness=fairness)
    db.close()
    return path


def test_mixed_goals_are_listed_as_excluded_not_dropped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _db(tmp_path, [("m0", 0, {"target": "rl", "target_value": -245.0}, None),
                          ("m1", 1, {"target": "rl", "target_value": -165.0, "provisional": True}, None)])
    kept, ex = ar.screen(ar.load_db(path))
    assert ar.analyse(kept, exclusions=ex) == []
    why = dict(ex.races)
    assert set(why) == {"m0", "m1"} and all("different goals" in w for w in why.values())
    md = ar.render_markdown([], ex)
    assert "race `m0`: not analysed" in md and "race `m1`: not analysed" in md
    assert "different goals" in capsys.readouterr().err
    # the CLI passes its exclusions through
    assert ar.main(["--db", str(path), "--no-charts"]) == 0
    out = capsys.readouterr().out
    assert "race `m0`: not analysed" in out and "Nothing excluded" not in out


def test_a_target_without_a_goal_is_listed_as_excluded(tmp_path: Path) -> None:
    path = _db(tmp_path, [("u0", 0, {"target": "rl"}, None)])
    kept, ex = ar.screen(ar.load_db(path))
    for rows, _ in kept:
        rows["races"][0]["target"] = "not-a-registered-target"   # unknown target, no goal anywhere
    assert ar.analyse(kept, exclusions=ex) == []
    assert any(rid == "u0" and "no target value" in w for rid, w in ex.races)


def test_unlike_setups_are_reported_apart(tmp_path: Path) -> None:
    v0, v1 = _fairness("v0:623366123052"), _fairness("v1:5c1d3cfd2941")
    path = _db(tmp_path, [("a0", 0, None, v0), ("a1", 1, None, v0), ("a2", 2, None, v0),
                          ("b0", 0, None, v1), ("b1", 1, None, v1)])
    kept, ex = ar.screen(ar.load_db(path))
    reps = ar.analyse(kept, target_values=TV, exclusions=ex)
    assert sorted(sorted(r.race_ids) for r in reps) == [["a0", "a1", "a2"], ["b0", "b1"]]
    ids = {r.setup_id for r in reps}
    assert len(ids) == 2 and None not in ids
    md = ar.render_markdown(reps, ex)
    for r in reps:
        assert f"setup `{r.setup_id}`" in md
    assert "3 seed(s)" in md and "2 seed(s)" in md and "5 seed(s)" not in md
    assert ar.MIXED_STAMP not in md          # each report is one setup over distinct seeds


def test_repeated_seeds_are_counted_once_and_stamped(tmp_path: Path) -> None:
    v0 = _fairness("v0:623366123052")
    path = _db(tmp_path, [("r0", 0, None, v0), ("r0b", 0, None, v0), ("r1", 1, None, v0)])
    kept, ex = ar.screen(ar.load_db(path))
    (rep,) = ar.analyse(kept, target_values=TV, exclusions=ex)
    assert rep.n_races == 3 and rep.n_seeds == 2 and rep.mixed
    md = ar.render_markdown([rep], ex)
    assert "3 race(s) over 2 distinct seed(s)" in md
    paste = md.split('Rows for the "Benchmark results" table', 1)[1]
    assert ar.MIXED_STAMP in paste and "| Busy Machines (Mubit) | 2 |" in paste and "3 races;" in paste


def test_charts_get_one_file_set_per_setup(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    v0, v1 = _fairness("v0:623366123052"), _fairness("v1:5c1d3cfd2941")
    path = _db(tmp_path, [("a0", 0, None, v0), ("b0", 0, None, v1)])
    kept, _ = ar.screen(ar.load_db(path))
    reps = ar.analyse(kept, target_values=TV)
    paths = ar.render_charts(reps, tmp_path / "charts")
    assert len(paths) == 8 and len({p.name for p in paths}) == 8
    assert all(any(f"-{r.setup_id}-" in p.name for r in reps) for p in paths)
