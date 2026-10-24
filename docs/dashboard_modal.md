# Live Modal metrics on our dashboard (demo race only)

Feeds the "Modal live" panel of `dashboard_spec.md` §3.2 for the labelled demo race (preset `demo-cifar-modal-5m`: one Modal app per side, `busy-machines-opus` / `busy-machines-bm`). **Never on in a benchmark or ablation race**: the config refuses it there, the app does not poll Modal for those races, and their Modal jobs are byte-for-byte unchanged (tests check all three).

Code: `workers/runners/modal_stats.py` (poller, in-container GPU sampler, per-side join), wired into `workers/runners/modal_runner.py` (`ModalRunner(gpu_telemetry_s=…)`, the container sampler, `calls_in_flight()`), `backend/benchmark/configs.py` (`RaceConfig.gpu_telemetry_s`), `backend/core/events.py` (`ModalStatsEvent`), `backend/app.py` (`ModalStatsHub`, `GET /api/races/{id}/modal`), `backend/dashboard.py` (forwards `modal_stats` on the dashboard stream) and `backend/demo.py` (starts polling for the Modal demo). Tests: `tests/test_modal_stats.py`, `tests/test_dashboard_modal.py`.

## 1. Where the numbers come from

| Source | What | How often |
|---|---|---|
| Modal: `modal.Function.from_name(app, "run_l4").get_current_stats()` on each side's own app | containers, calls in flight, backlog | every **2 s** per side |
| Inside each training container: pynvml, else `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total` | GPU use and memory of that container's GPU | every **1 s** while a run trains |
| Our DB: `experiments.run_ref` (already recorded for every run) | the Modal call id of each run | read at every poll |

The stats call was checked against the installed SDK (modal **1.6.2**): it returns `FunctionCurrentStats(backlog, num_total_runners, num_running_inputs, input_headroom)`. Modal's other APIs (`Function.stats()`, the metrics API with buckets of at least 60 s) are too coarse for a live panel and are not used.

## 2. Fields

WebSocket event `modal_stats` (one per side per poll, on `/ws/races/{id}` and on the dashboard stream `/ws/races/{id}/dashboard`, also with `raw=false`) and `GET /api/races/{id}/modal` (latest values per side, `?history=N` for the last N polls):

| Field | Unit | Meaning | Panel row (plain label) |
|---|---|---|---|
| `containers` | count | Modal `num_total_runners`: containers alive for the side's function, warm or busy | "Containers running" |
| `calls_in_flight` | count | Modal `num_running_inputs`: our runs executing now | "Runs in flight" |
| `backlog` | count | Modal `backlog`: runs waiting for a container (should be 0 with pre-warm) | hover only |
| `input_headroom` | count | Modal `input_headroom` | hover only |
| `idle_containers` | count | `containers − calls_in_flight`: warm containers with no call in flight | hover only |
| `gpu_util_pct` | % (0–100) | GPU use over the side's containers whose state we know: each sampled run at its measured %, each warm container with no call in flight at 0 %. `null` if calls are in flight but none has a sample | "GPU use" |
| `gpu_util_active_pct` | % (0–100) | mean GPU use of the sampled runs only | hover only |
| `gpu_mem_mb` | MiB | GPU memory used, summed over sampled runs | hover only |
| `runs_sampled` / `runs_unsampled` | count | runs in flight with / without a live sample (a sample older than 5 s does not count) | hover only |
| `call_ids` | list of str | Modal call ids (`fc-…`) of the side's runs in flight | — |
| `latest_call_id` | str | Modal call id of the side's most recently started run | "Latest Modal call" |
| `ok`, `error` | | `ok: false` = the stats call failed (timeout 8 s); counts are then `null` | show "—" |
| `app`, `function` | | the side's Modal app and function (`run_l4`) | never shown (no account names, `dashboard_spec.md`) |

The GET response also lists every run of each side (`calls`: experiment id, Modal call id, status, start and end time), so the call ids survive the race.

In-container metric points (ordinary `metric` events and `metrics` rows, so they are stored and replayable): `gpu_util_pct` (%), `gpu_mem_mb` (MiB) every 1 s, and `gpu_mem_total_mb` (MiB) once per run. `t` is seconds since the run was launched (same clock as the run's own metrics); `step` is the sample number. They go into the run's metric stream only, never into the run's result metrics (those reach the judge prompt).

## 3. What "idle" means

A container only sees its own runs, so the sampler cannot see the time between runs. Idle is therefore derived, not sampled: a warm container with **no call in flight** (Modal `num_total_runners − num_running_inputs`; `run_l4` takes one run per container) is running nothing of ours, so it counts as 0 % GPU use. Inside a run, the sampler also records the run's own 0 % stretches (start-up, data loading, evaluation). A run in flight that has no sample yet is left out of `gpu_util_pct` (and counted in `runs_unsampled`), never shown as 0 %.

## 4. Honesty note

- **Container count alone does not show waiting.** Both sides are pre-warmed and their containers stay warm (`scaledown_window` 60 s), so both sides show 3 containers almost all race, and both pay for them. What shows the waiting is **calls in flight** and **GPU use**: the standard agent's containers sit warm with no call in flight while it waits for Opus, which reads as low GPU use; Busy Machines keeps calls in flight.
- These are live readings for the video, **not benchmark numbers**. The headline "GPUs waiting" figure stays the engine's measured `gpu_idle_pct` from slot intervals. Don't quote a GPU use % in slides or the README unless a demo race has run and its value is logged in `docs/results_log.md`.
- GPU use % is the driver's "utilization" (share of time a kernel was running), not a measure of how efficiently a kernel uses the GPU.
- Both sides get the same sampling interval. The sampler's own cost (one NVML read, or an `nvidia-smi` process, per second on the container's CPU) has not been measured; it applies to both sides alike, and only in the demo race. `pynvml` / `nvidia-ml-py` is not in the L4 image (`IMAGE_PIP_PACKAGES`), so in practice it is one `nvidia-smi` process per second. The sampler takes its first reading on its own thread (starting it never delays the run), and at the end of a run the container waits at most 0.5 s for it before returning the result.

## 5. Running it

1. Redeploy **only the two per-arm demo apps** (the container sampler is new code in `modal_runner.py`): `python -m workers.runners.modal_runner arm-apps deploy`. The default app `busy-machines-runner` doesn't need a redeploy for this. Never redeploy it while a benchmark race is running.
2. For live GPU use the app must run the race itself (`POST /api/races` with the preset's config, then `/start`), because the in-container samples arrive in the process that runs the race. If `scripts/bench.py` runs the demo (it does the pre-warm), start the app on the same DB (`BUSY_MACHINES_DB=data/bench/demo.db`). Then containers, calls in flight, backlog and call ids are live, but `gpu_util_pct` is `null` while calls are in flight. It is still a correct 0 % when every container is idle. The samples are in the DB `metrics` table after the race.
3. Polling starts when the demo race starts (`POST /api/demo/start` polls from the pre-warm on and returns the panel URL as `modal`; also `/start`, or the first GET or WebSocket connect while it is live), and stops when the race ends, or at the latest 15 min after its budget. Every poll is also appended to `data/bench/modal_stats/race-<id>.jsonl` (`BUSY_MODAL_STATS_DIR`; empty turns it off) as evidence, because the `modal_stats` events themselves are not stored in the race DB, not kept in the in-memory event history (a late subscriber gets the next poll, about 2 s later) and not replayed. `BUSY_MODAL_STATS=0` turns polling off.

## 6. Status of the code

The wiring is in the working tree (24 Oct, re-done on the current `app.py`, `events.py`, `scheduler.py`, `dashboard.py` and `configs.py`). `data/dashboard_patches/` holds the first version, built before those files changed; it no longer applies and is kept only for reference. Differences from it:

- `ModalStatsEvent` sits with the other dashboard additions in `events.py`, so `EVENT_TYPES` keeps its 8 core types; `modal_stats` is in `ALL_EVENT_TYPES` (`MODAL_EVENT_TYPES`).
- The app publishes `modal_stats` with `EventBus.publish(..., keep=False)`: delivered to subscribers and listeners, never kept in the history.
- The polling lives in `backend.app.ModalStatsHub` (`AppState.modal_stats`), used by `/start`, `GET /api/races/{id}/modal`, both live WebSockets and `POST /api/demo/start`.
- A race whose task has ended (crash, cancel, failed pre-warm) stops being polled even if its engine still says `created`.

No Modal call was made to build or test this; the two demo apps have not been redeployed with it.

```
python -m uv run --extra ml --with onnxruntime==1.29.0 pytest tests/test_modal_stats.py tests/test_dashboard_modal.py tests/test_demo_modal_apps.py tests/test_demo_live.py tests/test_core.py tests/test_dashboard.py tests/test_api.py
```
