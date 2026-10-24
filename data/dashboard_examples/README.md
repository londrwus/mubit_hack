# data/dashboard_examples: real dashboard API output

Recorded from the local API (`uvicorn --factory backend.app:create_app`, replay source `data/bench/bench.db`) with `scripts/dashboard_client.py`. The API and its message formats are described in `docs/dashboard_api.md`. Nothing here was edited by hand. No LLM calls and no Modal calls were made: the dev race uses the mock AI and simulated machines ($0), and replays only read stored rows.

| File | What |
|---|---|
| `replay-r78-s0-x60-dashboard.jsonl` | R78 seed 0 (race `4ca4ed27`) replayed at 60× over `/ws/replays/{id}/dashboard?tick_s=0.5&raw=false`, the full 15 min race in about 15 s: `slot`, `tick`, `judge_verdict`, `planner_*`, `llm_call`, `experiment_finished`, `new_best`, `race_finished` |
| `replay-r78-s0-x60-raw-first400.jsonl` | The first 400 messages of the same replay with `raw=true` (includes `gpu_state` and `experiment_started`) |
| `live-dev-race-mock.jsonl` | A 90 s mock dev race created and started through the API (`POST /api/races`, `/start`) and streamed live over `/ws/races/{id}/dashboard?tick_s=1` |
| `snapshot-r78-s0-at-0000.json`, `-at-0500.json`, `-end.json` | `GET /api/races/4ca4ed27/snapshot?at_s=…`: the R78 seed 0 frames at 0:00, 5:00 and the end |
| `snapshot-r82-s0-end.json` | `GET /api/races/c1f4b1d9/snapshot`: the R82 seed 0 end frame |

The dev race is a mock run. Its numbers are not a result and are not logged.
