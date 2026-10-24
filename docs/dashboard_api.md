# Dashboard API: live races and replays for the two-column screen

What the backend serves for the race dashboard in `dashboard_spec.md` (Standard AI assistant on the left, Busy Machines on the right). It covers live races run by the API and replays of logged races (R78 seed 0 = race `4ca4ed27…`, R82 seed 0 = race `c1f4b1d9…`).

Code: `backend/dashboard.py` (reducer, header, routes). The new event types are in `backend/core/events.py`, emitted by `backend/scheduler.py` and rebuilt for replays by `backend/replay.py`. Tests: `tests/test_dashboard.py`. Example logs: `data/dashboard_examples/`. Client: `scripts/dashboard_client.py`.

**Benchmark races behave exactly as before.** The engine only publishes a few more events and fields (listed in §4.2). Its decisions and timings, the fairness manifest and the fingerprints are unchanged. The dashboard code reads events and never calls the engine. Every number on the screen comes from an event. Nothing is estimated or typed in. When a race finishes, the headline numbers switch to the stored `ArmSummary`, which holds the same numbers the results log quotes.

---

## 1. Run it

PowerShell, from the repo root:

```powershell
$env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"
$env:BUSY_MACHINES_DB="data/busy_machines.db"            # the API's own DB (dev races land here)
$env:BUSY_MACHINES_REPLAY_SOURCES="data/bench/bench.db"  # read-only source of logged races (";"-separated)
python -m uv run --extra ml --with onnxruntime==1.29.0 uvicorn --factory backend.app:create_app --port 8765
```

`--with onnxruntime==1.29.0` is needed for LCBench (the simulated tabular target). The client script needs the API to be running:

```powershell
python -m uv run python scripts/dashboard_client.py replay 4ca4ed27 --speed 60 --seconds 20 --raw false
python -m uv run python scripts/dashboard_client.py dev --budget 90 --seconds 110
python -m uv run python scripts/dashboard_client.py snapshot c1f4b1d9 --at 300
```

The client prints one line per `tick` and saves every message as JSON lines.

---

## 2. Endpoints

New endpoints are marked **new**. Any route that takes a race id also accepts a unique prefix of at least 4 characters (for example `4ca4ed27`).

| Method | Path | What |
|---|---|---|
| POST | `/api/races` | Create a race from a `RaceConfig` body. Returns `{id, config}`. |
| POST | `/api/races/{id}/start` | Start a created race in this process. |
| POST | `/api/races/{id}/stop` | Stop a running race (`cancelled`). |
| GET | `/api/races/{id}` | Race row, arms (with live summary), experiments, corrections. |
| GET | `/api/races?replayable=true` | Finished races from the app DB and every replay source. |
| **GET** | **`/api/races/{id}/snapshot[?at_s=]`** | **One dashboard snapshot (§3).** For a race running in this process it is now; `?at_s=` gives an earlier moment. For a finished race it is the end, or `at_s` seconds of race time. 404 if the race is unknown. 409 if the race was created but not started, or runs in another process. |
| **GET** | **`/api/dashboard/logged-races`** | **Logged result races the server can replay: `[{ref: "R78", run: 1, runs: 3, race_id, task, budget_text, goal_text, arms, source}]`.** The list comes from `data/frontend/showcase.json` `sources.race_sets`. |
| POST | `/api/races/{id}/replay?speed=&max_gap_s=&metric_stride=` | Create a replay session. Returns `{replay_id, ws, n_events, counts, race_duration_s, replay_duration_s, …}`. **`metric_stride=0` is new**: no metric events, and the metrics table is not read. Use it for the dashboard, which draws no training curves. R78 loads in under 1 s instead of about 11 s. |
| GET | `/api/replays`, `/api/replays/{id}` | Replay sessions held in memory (at most 16). |
| WS | `/ws/races/{id}` | Raw live event stream (unchanged; demo races on per-arm Modal apps also get `modal_stats`, `docs/dashboard_modal.md`). |
| **GET** | **`/api/races/{id}/modal[?history=N]`** | **Live Modal view of a demo race on per-arm Modal apps (`docs/dashboard_modal.md`). Any other race: `{"enabled": false}` and no Modal call.** |
| WS | `/ws/replays/{replay_id}` | Raw replay stream (unchanged). |
| **WS** | **`/ws/races/{id}/dashboard?tick_s=1&raw=true&metrics=false`** | **Live dashboard stream: raw events, derived `slot` events, and a `tick` (full snapshot) every `tick_s` s of wall time (0.05–60).** For a demo race on per-arm Modal apps it also forwards `modal_stats` (one per side every ~2 s; connecting starts the poller). `raw=false` sends only the notable raw types (§4.4). Metric points are sent only with `metrics=true`. It sends the race-so-far first, then a tick, then live events. After `race_finished` it sends the slot-off events and a final tick, then closes. A race that finished before this process started gets one final tick and the socket closes. |
| **WS** | **`/ws/replays/{replay_id}/dashboard?tick_s=0.25&raw=true&metrics=false&start_s=0`** | **The same over a replay session at its speed.** Every message carries `is_replay: true`, `replay_id` and `race_elapsed_s`. Tick snapshots interpolate the race clock between events. `start_s` folds the first N s of race time silently, sends a tick, and plays the rest from there. |
| **POST** | **`/api/demo/start?preset=demo-live-60s[&dry_run=true]`** | **Start a demo race in one call (§5.1).** Only `demo-live-60s` and `demo-cifar-modal-5m` are accepted; any other preset gets 400. Returns `{race_id, ws, ws_url, …}` with status 201. Returns 409 while a race is running in this process, and 503 if a Mubit key is missing. |
| **GET** | **`/api/demo/presets`** | **The presets `POST /api/demo/start` accepts, with arms, length, slots and the per-side $ cap.** |
| GET | `/api/races/{id}/export.jsonl` | Full JSONL export (verification). |
| POST | `/api/experiments/{id}/flag` | A correction (dev / demo races only; benchmark races refuse it). It shows up as `correction` and as a fake row "flagged by a person". |

Close codes: 4404 means an unknown race or replay. 4409 means the race is neither live in this process nor replayable.

---

## 3. The snapshot (`tick.snapshot`, `GET …/snapshot`)

Schema `busy-machines.dashboard/1`. The example below is trimmed from `data/dashboard_examples/snapshot-r78-s0-at-0500.json` (R78 seed 0 at 5:00). The full file has both columns.

```json
{
 "schema": "busy-machines.dashboard/1",
 "race_id": "4ca4ed2709854a34bd6427ca2992e006",
 "ts": 1792856414.474789, "race_elapsed_s": 300.0, "clock": "05:00 / 15:00",
 "status": "running",                      // not_started | running | finished
 "finish_reason": null,                    // budget | cost_cap | cancelled | error (when finished)
 "finished_text": null,                    // "Finished · 15:00" | "Stopped: cost_cap"
 "header": {
  "title": "Busy Machines", "subtitle": "compute intelligence for training sweeps",
  "task": "Game-playing AI", "target": "rl",
  "budget_s": 900.0, "cost_cap_usd": 8.0, "gpus_per_arm": 30,
  "budget_text": "15 min · up to $8.00 per side · 30 machines each",
  "goal": -170.0, "goal_text": "Goal: −170", "direction": "max", "higher_is_better": true,
  "score_decimals": 1,
  "machines": {"label": "Machines", "caption": "30 cloud CPU workers", "count": 30, "hardware": "CPU",
               "simulated": false, "waiting_label": "Machines waiting"},
  "arms": [
   {"kind": "opus", "label": "Standard AI assistant", "sublabel": "Opus 5.5", "side": "left", "uses_ai": true,
    "n_slots": 30, "hardware": "CPU", "price_per_slot_s": 1.31e-05, "model": "claude-opus-5-5",
    "features": [], "feature_labels": []},
   {"kind": "opus_bm", "label": "Busy Machines", "sublabel": "Opus 5.5", "side": "right", "uses_ai": true,
    "n_slots": 30, "features": ["backfill", "judge_policy", "lean_planner", "planner_cadence"],
    "feature_labels": ["keeps GPUs busy with a no-AI sampler while the AI thinks", "..."]},
   {"kind": "optuna", "label": "No-AI tuner", "sublabel": "no AI", "side": "secondary", "...": "..."}
  ],
  "mode": "replay", "is_replay": true, "replay": {"replay_id": "...", "speed": 60, "max_gap_s": null},
  "race_mode": "benchmark", "llm": "mubit", "runner": "modal",
  "preset": "rl-wide-bm-cadence-15m", "seed": 0,
  "demo_run": false, "demo_label": null,
  "mode_tag": "Replay of logged race R78, run 1 of 3 · 60× speed",
  "logged_race": {"ref": "R78", "run": 1, "runs": 3},
  "ai_cost_note": "AI $ at list price (estimate)", "gpu_cost_note": null
 },
 "arms": [
  {
   "kind": "opus_bm", "label": "Busy Machines", "sublabel": "Opus 5.5", "side": "right", "uses_ai": true,
   "n_slots": 30,
   "slots": [{"slot_id": "modal-CPU-0", "state": "busy", "since_s": 283.294,
              "experiment_id": "64aa1297…", "source": "sampler"}, "… 30 in all"],
   "slot_counts": {"busy": 30, "waiting": 0, "cancelled": 0, "off": 0},
   "working_now": 30, "open_now": 30, "waiting_now_pct": 0.0,
   "waiting_so_far_pct": 1.224, "busy_s": 8889.83, "idle_s": 110.17,
   "ai_bill_usd": 0.302345, "gpu_bill_usd": 0.1179, "total_usd": 0.420245,
   "llm_calls": 34, "llm_invalid_calls": 0,
   "experiments_started": 553, "started_by_source": {"baseline": 1, "sampler": 512, "llm": 40},
   "experiments_finished": 523, "finished_by_source": {"sampler": 482, "baseline": 1, "llm": 40},
   "experiments_failed": 0, "experiments_cancelled": 0,
   "model_quality": {
    "best": -156.29537963867188, "best_text": "−156.3", "experiment_id": "b74bbe9c…", "at_s": 297.873,
    "source": "sampler", "baseline": -1050.0225830078125,
    "series": [[16.432, -531.417], [16.448, -218.304], [16.859, -169.122], "…"],
    "goal_reached_at_s": 16.859, "goal_reached_text": "Goal reached at 0:16"
   },
   "checks": {"valid": 3, "invalid": 14, "unsure": 12, "none": 0, "claims_rejected_not_fake": 0},
   "fakes_caught": 14,
   "fakes": [{"experiment_id": "e15343d0…", "at_s": 69.657, "claimed": -148.502, "real": -188.484,
              "why": "tested on data it had already seen", "by": "checker", "source": "sampler",
              "text": "1:09 · claimed −148.5, real score −188.5 (tested on data it had already seen)"}],
   "planner": {"thinking": false, "since_s": null, "calls": 5, "last_k": 8, "thinking_s": 30.716},
   "reported_best": {"experiment_id": "1f92fd5e…", "score": -156.78125, "at_s": 271.338, "source": "sampler"},
   "final": null
  }
 ],
 "feed": [{"kind": "goal_reached", "arm": "opus", "at_s": 74.572, "quality": -163.269,
           "text": "Goal reached at 1:14"}],
 "n_events": 5188
}
```

### 3.1 Where each screen element comes from (`dashboard_spec.md`)

| Spec element | Snapshot field | Definition |
|---|---|---|
| Header: title, task, budget, goal | `header.title/subtitle/task/budget_text/goal_text` | From the race config. Task labels: `rl` → "Game-playing AI", `lcbench` → "Tabular model tuning", `cifar` → "Image recognition". |
| Mode tag | `header.mode_tag`, `is_replay`, `demo_run`, `logged_race` | Replay: "Replay of logged race R78, run 1 of 3 · 60× speed". Demo (`config.demo_label` set): "Demo run, not a benchmark result". Mock dev race: "Dev run (mock AI), not a benchmark result". |
| Race clock | `clock`, `race_elapsed_s` | Race time (not wall time), rounded down. |
| Columns | `header.arms[]` (`side`: left / right / secondary) | Left: Standard AI assistant (`opus`). Right: Busy Machines (`opus_bm` or `mubit`). `optuna` is `secondary`, so keep it off the main screen (D46). |
| Machines grid | `arms[].slots[]`: `state`, `since_s`, `experiment_id`, `source` | **busy** = an experiment is running. **waiting** = the slot is paid but has no experiment. **cancelled** = waiting right after its run was stopped (cancelled, failed or timed out), until the next run. **off** = before the first event, or closed (released early by the deadline rule, or the race ended). |
| Grid caption | `working_now` / `n_slots`, `waiting_so_far_pct`, `header.machines.caption/waiting_label` | Waiting so far = idle ÷ paid slot time to date. This is the same definition as `ArmSummary.gpu_idle_pct`, and at the end it is that value. "GPUs" is used only for real GPUs on Modal. R78 shows "cloud CPU workers" and R82 shows "GPUs, simulated machine time". |
| AI bill | `ai_bill_usd` | Sum of `llm_call.call.cost_usd` for the arm (list-price estimate). Mock races show $0.00. |
| GPU $ | `gpu_bill_usd` | Paid slot seconds × the slot price. A simulated runner gives simulated $. |
| Experiments finished | `experiments_finished`, `finished_by_source` | `experiment_finished` events with `status = succeeded`. Hover: `finished_by_source` (`llm` = planned by the AI, `sampler` = built-in tuner, `baseline` = starting model). |
| Model quality | `model_quality.best/best_text/series/baseline`, `goal_reached_at_s` | The best `experiment_finished.quality` (held-out score) among runs that succeeded, are not a trap, and were not rejected by the checker (`metrics.is_counted`). A run counts when it finishes. If a later `judge_verdict` is `invalid`, it is dropped and the drop is a point on `series`. At the end the value equals `ArmSummary.best_score`. |
| Fake improvement caught | `fakes_caught`, `fakes[]` (last 10, `text` ready to show), `feed[]` (`kind: fake_caught`) | Sources: a `judge_verdict` with `fake_caught: true` (a known trap that the checker rejected), and a `correction` labelled `fake_improvement` (`by: human` / `sapling`). This is the same rule as `scripts/replay_tui.py`. `real` is null on LCBench, because the shortcut also spoils that score. |
| AI planner thinking | `planner.thinking`, `since_s`, `calls`, `last_k`, `thinking_s` | From `planner_started` / `planner_finished`. |
| Finished state | `status: finished`, `finished_text`, `arms[].final` | `final` is the stored `ArmSummary`, and the headline fields use it. |
| Modal live panel (demo only) | not in the snapshot | `GET /api/races/{id}/modal` and the `modal_stats` events on both live WebSockets (`docs/dashboard_modal.md`). |

---

## 4. Event schemas

Every event is JSON with `type`, `race_id` and `ts` (Unix seconds). Parse any of them with `backend.core.events.parse_event`. On replay streams every payload also has `is_replay: true`, `replay_id` and `race_elapsed_s`. The examples are real lines from `data/dashboard_examples/`.

### 4.1 Unchanged core events
`gpu_state`, `metric`, `new_best`, `llm_call`, `correction` and `race_finished` are unchanged (see `docs/architecture.md`).

```json
{"type": "gpu_state", "race_id": "4ca4ed27…", "ts": 1792856114.474789, "arm": "optuna", "slot_id": "modal-CPU-0", "state": "idle", "experiment_id": null}
```

`gpu_state` has only `busy` and `idle`. Two `idle` events in a row on the same slot mean the slot was closed: either the deadline rule released it, or the race ended. The dashboard's `off` state comes from this.

### 4.2 New fields and new engine events (live engine + replay)

**`experiment_started.source`**: who proposed the run. Values: `baseline` | `llm` (the AI planner) | `sampler` (the backfill no-AI sampler) | `optuna`.

```json
{"type": "experiment_started", "arm": "opus", "experiment_id": "848433d7…", "slot_id": "modal-CPU-0", "config": {"lr": 0.0001, "…": "…"}, "hypothesis": "Starting code: the default config, run by the scheduler as the baseline.", "seed": 0, "hardware": "CPU", "source": "baseline", "ts": 1792856114.4797864, "race_id": "4ca4ed27…"}
```

**`experiment_finished.quality`**: `metrics.quality` of a run that succeeded (its held-out score). It is null otherwise. This is ground truth for the screen, and the planner never sees it.

```json
{"type": "experiment_finished", "arm": "optuna", "experiment_id": "6a4ff562…", "slot_id": "modal-CPU-2", "status": "succeeded", "score": -992.3446, "is_trap": false, "duration_s": 16.18, "gpu_cost_usd": 0.000212, "error": null, "quality": -992.3446, "ts": 1792856130.838, "race_id": "4ca4ed27…"}
```

**`judge_verdict`** (new): the checker's decision on a claimed new best, on LLM arms only. It is sent when the judge step returns, before any `new_best` for that run. `verdict` is the decision that counts: the judge policy's final on Busy Machines arms, otherwise the LLM judge's. `llm_verdict` is the LLM judge's own verdict. `fake_caught = is_trap and verdict == "invalid"`.

```json
{"type": "judge_verdict", "arm": "opus_bm", "experiment_id": "5b313a0b…", "verdict": "invalid", "llm_verdict": "invalid", "llm_reason": "contamination_or_eval_leak", "reason": "LLM judge: invalid (contamination_or_eval_leak)", "claimed_score": -177.765, "previous_best": -218.304, "is_trap": true, "quality": -209.830, "source": "sampler", "fake_caught": true, "ts": 1792856136.855, "race_id": "4ca4ed27…"}
```

**`planner_started`** / **`planner_finished`** (new): an AI planner (propose) call on an LLM arm. `k` is how many experiments it was asked for, and `n_added` how many proposals were accepted. A call cut off by the end of the race has no `planner_finished`.

```json
{"type": "planner_started", "arm": "opus", "k": 8, "ts": 1792856114.4827888, "race_id": "4ca4ed27…"}
{"type": "planner_finished", "arm": "opus_bm", "started_at": 1792856114.486789, "k": 8, "n_added": 8, "ts": 1792856121.9389868, "race_id": "4ca4ed27…"}
```

### 4.3 Derived events (dashboard streams only, never emitted by the engine)

**`slot`**: a slot's display state changed. The `busy` event is held until `experiment_started` arrives, so it carries `source`.

```json
{"type": "slot", "arm": "opus", "slot_id": "modal-CPU-0", "state": "busy", "experiment_id": "848433d7…", "source": "baseline", "race_elapsed_s": 0.004997, "ts": 1792856114.4797864, "race_id": "4ca4ed27…"}
{"type": "slot", "arm": "optuna", "slot_id": "modal-CPU-26", "state": "off", "experiment_id": null, "source": null, "race_elapsed_s": 883.0, "ts": 1792856997.474789, "race_id": "4ca4ed27…"}
```

**`tick`**: `{"type": "tick", "race_id", "ts", "race_elapsed_s", "snapshot": {…§3…}}`.

### 4.4 What `raw=false` keeps
`slot`, `tick`, `judge_verdict`, `new_best`, `correction`, `planner_started`, `planner_finished`, `experiment_finished`, `llm_call`, `race_finished` and `modal_stats` (demo races on per-arm Modal apps only). It drops `gpu_state`, `experiment_started` and `metric`, because the slots and ticks already carry what they say.

---

## 5. Start a live demo race through the API (dev mode)

A race shows live only if it runs inside the API process (`POST /api/races` then `/start`). A race started by `scripts/bench.py` runs in another process. The dashboard can show it only after it finishes, as a replay.

**$0 dev race** (mock AI, simulated machines, no Mubit call, no Modal call). This is what `scripts/dashboard_client.py dev` sends:

```http
POST /api/races
{"name": "dashboard dev race (mock AI, simulated machines)",
 "target": "lcbench", "arms": ["opus", "opus_bm"], "budget_s": 90, "gpus_per_arm": 8, "seed": 0,
 "mode": "dev", "llm": "mock", "runner": "surrogate", "surrogate_runtime_mult": 0.1,
 "mock_latency_s": {"opus": 6.0, "opus_bm": 6.0}, "mock_trap_rate": 0.2, "max_proposals_per_call": 8,
 "engine_features": {"opus_bm": ["backfill", "judge_policy", "lean_planner", "planner_cadence"]},
 "feature_params": {"opus_bm": {"planner_cadence": {"min_propose_interval_s": 20}}}}
→ 201 {"id": "<race_id>", …}
```

Then:
1. Connect `ws://127.0.0.1:8765/ws/races/<race_id>/dashboard?tick_s=1&raw=false`. Before the start, the first tick has `status: not_started` and every slot `off`.
2. `POST /api/races/<race_id>/start`.
3. Read ticks, slots and notable events until `race_finished`. A final tick follows it.
4. Optionally, flag a run to show a correction (dev races accept them): `POST /api/experiments/<experiment_id>/flag {"label": "fake_improvement", "reason": "demo"}`. It appears as `correction` and as a fake row "flagged by a person".

The mock AI costs nothing, so the AI bill stays at $0.00. The machine $ is simulated (`header.gpu_cost_note`).

**The real demo run for the video** is preset `demo-cifar-modal-5m`: Modal L4s, one Modal app per side, `demo_label` set, so the header says "Demo run, not a benchmark result". Start it with `POST /api/demo/start?preset=demo-cifar-modal-5m` (§5.1). That call also pre-warms each side's Modal app. Or POST its `RaceConfig` the same way (`backend.benchmark.configs.PRESETS["demo-cifar-modal-5m"].model_dump(mode="json")`). That route does not pre-warm, so cold starts count inside the race clock. It spends Modal and LLM money and needs `MUBIT_API_KEY`, so it is not run by these tests or this step. Its Modal panel data is described in `docs/dashboard_modal.md`.

### 5.1 One-click demo races: `POST /api/demo/start`

This creates and starts a demo race inside the API process, so the dashboard WebSocket shows it live. Code: `backend/demo.py`. Tests: `tests/test_demo_live.py`.

| Preset | What runs | Real spend |
|---|---|---|
| `demo-live-60s` | 60 s, LCBench tabular task (simulated machines), 16 slots per side. Standard Opus agent (v1 loop) vs Busy Machines with Opus (lean planner, no-AI sampler, checker, AI planner at most every 10 s). Runs last 0.05× the surrogate's predicted training time, so most take 1–3 s. | LLM only: real Opus 5.5 through Mubit project A, record off on both sides, capped at $1 per side (that cap includes $0.21 of simulated machine $). Machine $ is simulated. |
| `demo-cifar-modal-5m` | 5 min CIFAR on Modal L4s, 3 per side, one Modal app per side. The API pre-warms each side's containers before the race clock starts, as `scripts/bench.py` does. | Modal L4s plus LLM, capped at $3 per side. |

Both presets are dev mode with `demo_label` set. Their name carries `[demo run]`, the header's `mode_tag` reads "Demo run, not a benchmark result", the fairness manifest has a `demo` entry, and `scripts/headline.py` refuses them. **These are demo runs, not benchmark results. Never quote their numbers as results.**

```http
POST /api/demo/start?preset=demo-live-60s
→ 201 {"race_id": "<id>", "preset": "demo-live-60s", "status": "running",
       "name": "LCBench live demo 60 s (…; demo run, not a benchmark result) seed 0 [demo run]",
       "demo_label": "demo run, not a benchmark result", "dry_run": false,
       "ws": "/ws/races/<id>/dashboard?tick_s=1&raw=false",
       "ws_url": "ws://127.0.0.1:8765/ws/races/<id>/dashboard?tick_s=1&raw=false",
       "ws_raw": "/ws/races/<id>", "snapshot": "/api/races/<id>/snapshot", "stop": "/api/races/<id>/stop",
       "arms": ["opus", "opus_bm"], "budget_s": 60.0, "gpus_per_arm": 16, "llm": "mubit", "runner": "surrogate",
       "cost_cap_usd": 1.0}
```

- The race starts at once. The dashboard stream sends the race so far first, so connecting to `ws` a second later loses nothing. For the Modal preset, `status` is `warming` until the containers are warm. The race clock starts after that.
- For `demo-cifar-modal-5m` the response also has `modal: "/api/races/<id>/modal"` (the live Modal panel, polled from the start, pre-warm included). It is `null` for `demo-live-60s` and for every dry run, which never call Modal.
- **One race at a time per process.** While any race is running, another start gets 409, so a double click never pays twice. Stop a race with `POST /api/races/<id>/stop`.
- **Mubit keys are checked first.** If a Mubit project the sides route to has no key (`MUBIT_API_KEY`, from the environment or `.env`), the endpoint returns 503 and creates nothing.
- **`dry_run=true` is a $0 rehearsal of the same preset**, like `scripts/bench.py run --dry-run`. It uses the mock AI (no Mubit call), keeps the simulated machines (Modal is replaced by laptop CPU), and skips pre-warm. The name gets `[dry run]`. The mock AI answers instantly, so the standard side waits much less than with real Opus. Use it to check the screen, not the story.
- **Simulated-machine pacing is process-wide** (`BUSY_SURROGATE_RUNTIME_MULT`). If an earlier race in this process left a different value and nothing is running, the endpoint clears it. If the variable was set when the API started, a different value is refused with 400.
- Started from the CLI instead (`python -m uv run --extra ml --with onnxruntime==1.29.0 python scripts/bench.py run demo-live-60s --yes`), the race goes to `data/bench/demo.db` and can only be shown afterwards, as a replay.

PowerShell:

```powershell
Invoke-RestMethod -Method Post "http://127.0.0.1:8765/api/demo/start?preset=demo-live-60s"
Invoke-RestMethod -Method Post "http://127.0.0.1:8765/api/demo/start?preset=demo-live-60s&dry_run=true"   # $0 rehearsal
```

**How `demo-live-60s` was tuned** (a $0 dry run, mock numbers, not a result). We used `--dry-run --slots 16` with the mock AI paced at R62's measured propose latencies (v1 at effort medium: 1.80 + 1.45·K s; v2 at effort low: 1.94 + 0.43·K s) and 1.8 s per judge call. Busy Machines' planner cadence was 10 s.

| Pacing | Busy Machines run time | Busy Machines slots waiting | Planner calls per side |
|---|---|---|---|
| ×0.01, ×0.015, ×0.02 | under 1 s at the median (×0.02) | 35%, 37%, 23% | not checked |
| ×0.04 | 1.0 / 1.3 / 1.9 s (p25 / p50 / p75) | 17% | 4 on each side |
| ×0.05 (chosen) | 1.1 / 1.4 / 2.1 s (p25 / p50 / p75) | 11% | 4 on each side |

- At ×0.01–×0.02 the runs finished faster than the no-AI sampler could propose new ones (its asks took 39 s of the 60 s at ×0.02), so Busy Machines' slots waited.
- At ×0.05 the standard side's runs took 2.2 s at the median. Its slots waited about 93% of the time (in this mock) and Busy Machines' about 11%.
- In the ×0.05 run, Busy Machines also made 13 judge calls and the standard side made 3.
- Real Opus timings will differ. Every number from a real demo race is a demo number, never a benchmark result.

---

## 6. Replay a logged race at a speed

**Recipe: R78 seed 0 at ×90** (a 15-minute race in 10 s):

```http
POST /api/races/4ca4ed2709854a34bd6427ca2992e006/replay?speed=90&metric_stride=0
→ 201 {"replay_id": "<id>", "ws": "/ws/replays/<id>", "race_duration_s": 900.0, "replay_duration_s": 10.0, …}
WS   /ws/replays/<id>/dashboard?tick_s=0.25&raw=false
```

The prefix form `POST /api/races/4ca4ed27/replay?speed=90&metric_stride=0` works too. The server needs `data/bench/bench.db` in `BUSY_MACHINES_REPLAY_SOURCES` (§1). The header reads "Replay of logged race R78, run 1 of 3 · 90× speed". In PowerShell:

```powershell
Invoke-RestMethod -Method Post "http://127.0.0.1:8765/api/races/4ca4ed27/replay?speed=90&metric_stride=0"
python -m uv run python scripts/dashboard_client.py replay 4ca4ed27 --speed 90 --seconds 12 --raw false
```

General form:

```http
GET  /api/dashboard/logged-races                       → R78 run 1 = 4ca4ed27…, R82 run 1 = c1f4b1d9…, …
POST /api/races/4ca4ed27/replay?speed=60&metric_stride=0
→ 201 {"replay_id": "<id>", "ws": "/ws/replays/<id>", "n_events": 15292, "race_duration_s": 900.0, "replay_duration_s": 15.0, …}
WS   /ws/replays/<id>/dashboard?tick_s=0.25&raw=false
```

- `speed` can be anything from 0 (exclusive) up to 1000. At 60×, R78's 15 minutes take 15 s.
- `max_gap_s` caps any single wait, so dead time is skipped.
- `start_s=450` on the WS jumps to 7:30.
- Each WebSocket connection plays the session from the start. A replay changes nothing in storage.
- Header mode tag: "Replay of logged race R78, run 1 of 3 · 60× speed". Never show "Live" over a replay (spec §6).
- A single frame without streaming: `GET /api/races/4ca4ed27/snapshot?at_s=300`.

Checked against the spec samples (`tests/test_dashboard.py`). The R78 seed-0 end frame shows AI bill $3.30 vs $0.70, experiments finished 497 vs 1,594, model quality −155.0 vs −154.9, waiting 69.0% vs 1.4%, fakes caught 0 vs 21, and goal reached at 74.6 s vs 16.9 s. R82 seed 0 shows $3.01 vs $0.67, 420 vs 761, 0.8905 vs 0.8903, 26.2% vs 0.5%, and fakes 0 vs 4. The 5:00 frames match too. In both races the reducer's own GPU $, waiting %, AI $ and best score equal the stored `ArmSummary` exactly.

---

## 7. Precision and limits

- **Replay timestamps.** `judge_verdict` is placed when the judge call completes. `planner_started` uses the arm report's call starts, which are rounded to the millisecond. `planner_finished` is placed at the call's last accepted proposal, or at its last propose LLM call. Payloads equal the live ones (tested), except these timestamps.
- **Model quality is provisional until the checker answers.** A run counts when it finishes, and a later `invalid` verdict on a run that was not a fake removes it. This is rare: R82 seed 0 had 2 on the standard side. The step line then shows a drop at the verdict. The final value always equals the stored summary.
- **GPU $ while a race runs.** It accrues from events up to now and is capped at the budget deadline. If a race stops early on its cost cap, the last seconds before `race_finished` may be slightly over-counted. The final frame uses the stored summary.
- **Live only in this process.** The live dashboard needs the race to run inside the API. The event history of a live race is held in memory (at most 100,000 events, metrics included).
- **Old events.** Events from before these fields existed parse with `source: null` and `quality: null`.
- **`docs/dashboard_spec.md` §3.6 sample rows.** The ticker sample rows ("0:16 · claimed −141.1, real score −162.4", "0:10 · claimed 0.7028") are not in the logged races. The first fakes caught in R78 seed 0 are at 0:22 (claimed −177.8, real −209.8), and in R82 seed 0 at 0:08 (claimed 0.2319). `scripts/replay_tui.py` shows the same. The counts in the spec (21 and 4) do match.
