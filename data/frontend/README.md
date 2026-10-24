# data/frontend: showcase cards for the dashboard

`showcase.json` holds the benchmark results the dashboard shows as cards. It is generated, so don't edit it by hand. Rebuild it with:

```
python -m uv run --with onnxruntime==1.29.0 --extra ml python scripts/build_showcase.py
```

**How the numbers are made**
- Race numbers are recomputed from `data/bench/bench.db` (opened read-only) with `scripts/headline.py`'s own `run()`. The builder also checks that each race set is one pooled, headline-eligible setup with the fingerprint the log names.
- Sapling numbers come from `data/sapling/compare_published/compare_published.json` (R81) and `data/sapling/sapling_v4/heads/final_v4.json` (R83).
- Every number is checked against the `docs/results_log.md` row it cites. Its `log_text` must appear in that row (with `**` removed), and the recomputed value must round to the logged value. R84 corrections are applied. On any mismatch the build aborts.
- The build makes no LLM calls, uses no Modal and writes only this file.
- `tests/test_showcase.py` re-checks the file against the log on every run. Its rebuild-from-DB test takes 1–2.5 min, so it runs only with `SHOWCASE_REBUILD=1`.

## Display rules
1. Show the `worse_side` of a card on the same screen as the card.
2. Every number in the file is logged (`in_log` is `true`; the build refuses anything else). R78's time and cost to the goal are shown as the logged values only (1.0 vs 0.3 min, $0.20 vs $0.022), with no ratio. R87 has since logged the ratios (89% cheaper, 72% faster; ≈9× cheaper, ≈3.6× faster), but they may be shown only with R87's caveat (the −170 goal is easy with 30 machines, and the no-AI tool reached it cheapest, $0.009 / 0.4 min), and the build does not carry them yet. `dashboard_spec.md` §6 forbids unlogged ratios: to show one, add it to the build with its R-id and caveat.
3. AI $ is a list-price estimate. The tabular races (R82, R75, R38) use simulated GPU time, so their GPU $ is simulated too. The JSON carries both notes.
4. Card `vs_no_ai_tool` has `placement: "secondary"`. It is for a second slide or Q&A, not the main screen (D46).
5. Labels are plain language (CLAUDE.md "UI copy"). Raw metric names appear only in `hover`. Game-playing (R78) ran on CPU workers and tabular (R82) uses simulated GPU time, so both say "Machines waiting", not "GPUs waiting" (`dashboard_spec.md` §3.1, §6); the hover says what the machines were.
6. Both sides get the same styling. Don't style Busy Machines as the winner.

## Schema (version 1)

```
{
  schema_version: 1,
  generated_by: "scripts/build_showcase.py",
  sources: { bench_db, results_log, sapling: [paths], race_sets: { R-id: [race id prefixes] } },
  rules: [string],                       // the display rules above, short form
  sides: { side_id: { label, sublabel, hover } },
  cards: [Card],                         // in display order
  checks_against_log: int                // how many log checks the build passed
}

Card = {
  id: "game_playing" | "tabular" | "where_it_costs_quality" | "sapling" | "vs_no_ai_tool",
  order: 1..5,
  kind: "hero" | "task" | "strip" | "verifier" | "baseline",
  placement: "main" | "secondary",
  title, headline, footer: string,
  r_ids: [R-id],
  panels: [Panel],                       // one per race set (the strip and baseline cards have several)
  worse_side: { caption: string, items: [Item] },   // always present, never empty
  notes: [Item]
}

Panel = {
  id, title, setup: string,
  r_id: R-id,
  race_ids: [8-char race id],            // [] for the Sapling panel
  exports: [path],                       // race JSONL exports or source JSON files
  status: { text, r_id },
  sides: [side_id],                      // column order
  metrics: [Metric]
}

Metric = {
  id: string,                            // model_quality, ai_bill_per_race, total_cost_per_race, time_to_goal,
                                         // cost_to_goal, cost_to_match_best, machines_waiting,
                                         // experiments_per_dollar, fake_improvements_accepted, shipped_fake,
                                         // fakes_caught, fakes_approved, real_confirmed, no_answer, time_per_check
  label: string,                         // plain language, from a fixed list
  hover: string,                         // may hold raw metric names
  higher_is_better: bool | null,
  values: { side_id: Number | null },    // null = not measured for that side
  comparison: Number | null,             // Number + { kind, formula }
  verdict?: string, note?: string
}

Number = {
  value: number,                         // recomputed (unrounded); minutes for times, percent as 0-100
  unit: "usd" | "minutes" | "seconds" | "percent" | "game_score" | "accuracy" | "count" | "ratio"
        | "experiments_per_usd" | "ms",
  display: string,                       // what to show ("$3.30", "−154.5", "0 of 847", "4.0× cheaper")
  r_id: R-id,                            // the results-log row it was checked against
  source: { file, race_ids?, computed_by?, key? },
  n: int,                                // runs (races) or rows behind the number
  n_total?: int,                         // races in the set (n < n_total: not reached in every race)
  in_log: bool,
  log_text: string | null,               // exact text found in that log row
  logged_value?: number, tolerance?: number, tolerance_note?: string,
  std?: number, std_display?: string, logged_std?: number, std_tolerance?: number,
  per_run?: [number] | { values, display, r_id, log_text },
  of?: int,                              // denominator for counts ("0 of 847")
  kind?: "x_cheaper" | "x_faster" | "x_slower" | "x_more" | "pct_less" | "pct_more_errors"
         | "pct_fewer_errors" | "tie",   // comparisons only
  formula?: string                       // comparisons only
}

Item = { text: string, r_id: R-id, numbers: [Number] }
```

**Side ids:**
- races: `standard` (Standard AI assistant, Opus 5.5), `busy_machines` (Busy Machines, Opus 5.5), `no_ai_tool` (No-AI tuning tool)
- Sapling: `published_sapling`, `ours_v2_strict`, `ours_v2_looser`, `ours_v4_strict`, `ours_v4_looser`, `opus`

## Known log rounding (tolerances recorded in the JSON)
- **R85, Busy Machines AI bill:** the log says $0.233, but the recomputed value is $0.2325 (which would round to $0.232). Tolerance $0.001. It is displayed as $0.23. Docs quote the logged $0.233.
- **R85, standard model quality spread:** the log says ± 0.0010, but the recomputed sample spread is 0.00095. Tolerance 0.0001. The logged ± 0.0010 is displayed.
