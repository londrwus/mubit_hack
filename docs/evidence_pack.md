# Evidence pack: what to have open on stage

A checklist of every console page, ID, race and link the demo or Q&A may need, each with its R-id. Open the tabs ~10 minutes before the slot, and log in to the Mubit console fresh (it signs out after 15 minutes idle). Truncated IDs (`…`) are as logged; search the console by their prefix.

## 1. Mubit console
### Project A (org `my-organization-1792777842609024233`, key `MUBIT_API_KEY`): the Opus baseline project
- [ ] **Tasks page:** propose ("Research lead") 499 calls, judge v0 ("Results auditor") 168, diagnose ("On-call debugging expert") 86, toy "Training-run triage". *"The assistant's work, as Mubit tasks."* (R29, R46)
- [ ] Propose: task `ln_42bac9601f877d03fc0999bc`, adapter `ad_lane_68db9770`, job `ftjob_e983ef33`, test set `gs_r4_ckac12cace699e5f91`: 14.3%, "Prompt only" (R28)
- [ ] Judge v0: task `ln_d81efc38…`, test set `gs_r4_ck8a6cdc8cd493233d`: 6/14 (42.9%) vs base + prompt 1/14, target 95% (R30); status *Testing v1* at ~08:45 (R46)
- [ ] Diagnose: task `ln_09190bd6…`: "Can't train: the training run failed" (R46); Mubit retried it, and v1 now scores 8.3% on 12 test cases, target 95% (R80)
- [ ] Toy: task `ln_74da54ace35187e2085fc0e9`, adapter `ad_lane_fa72c0ab`, job `ftjob_ee4928ec`: 70% vs 85% (R6, R7)
- [ ] Request `01a16aaa-a7a6-7dc3-b30a-869f65e9b3a0`: "Where the time went", middleware <1 ms (R3)
- [ ] Correction case, original race call (judge v0, recorded): `mreq_01a16b31599b7651a654571b8bc455a7` (ns_v1_007), `mreq_01a16b3181347a709987f2a83e251dd6` (ns_v1_013), `mreq_01a16b33c0367780810aff88ec19fc1d` (ns_v1_048) (R20, R34)

### Project B (`13123`, workspace `ws-2202e825d19648d2`, key `MUBIT_API_KEY_B`): judge v1
- [ ] **Judge v1 task** `ln_5d9ea82de398d3c492ec873f` → Adapters: adapter `ad_lane_9033f7ff`, job `ftjob_10fa515e`, **11.4% on 35 test cases, target 95%** (R70). History tab (v1 → v2 if it exists)
- [ ] Settings → **Going live**: quality bar 90%, silent test 20% (up to 100 a day), partial rollout 50% (at least 100 calls) (R56)
- [ ] Settings → **Training**: new versions after 25 new examples; up to 256 KB per example; no daily limit (R58)
- [ ] **Requests**, correction case: `mreq_01a16c07cc1d7c70ae96381f78916256` (session `collect-judgev1-2026-10-24/noise_set/7`, Opus `unsure · insufficient_evidence`): **rate Bad on stage** (R39, R34)
- [ ] Spare correction cases: `mreq_01a16c0666de7752843c5cebf7a34c28` (noise_set/13), `mreq_01a16c066ed474829ae80db3319c2810` (noise_set/23 = ns_v1_048)
- [ ] v2 batch (40 calls, R71): first IDs `mreq_01a16e29a9917191891ef510a0ee6144`, `mreq_01a16e29a9867f41b205d354d152a8e3` (a fake → `invalid`); all IDs in `data/adapter_golive/B/traffic-20261024T134620Z.request_ids.txt`
- [ ] 503 episode: e.g. `mreq_01a16c05e7b773a298d2872af6e05709` (`engine_unavailable`, R39)
- [ ] Probes with B's provider key: `mreq_01a16ba7eb38…` (Haiku 200), `mreq_01a16ba7f3d7…` (Opus 200) (D38)

### Workspace C (`ws-80b9fd6027bad9e4`, key `MUBIT_API_KEY_C`): judge v1 hedge
- [ ] Judge v1 task `ln_dfe18721334e06d23c32f338`, job `ftjob_2dd8c4a0`, same task key `lk1_be9a22ab…` as B: 159 kept, 0 rejected, 18 couldn't be reviewed (R61); trained ~3 h after it started: adapter `ad_lane_52d6d485`, **58.3% on 24 test cases, target 95%**, not routed (R79). D52: C's quality bar to be lowered to 55% (C only; say "we lowered the bar to 55%", never "passed Mubit's exam"). Re-read at 05:00 Sun (D53)
- [ ] Hedge recording: first IDs `mreq_01a16d543b287c13b3345b1bbb17aaf2`, `mreq_01a16d543acf71e1b7611e7bd21bc8ba` (R59)
- [ ] Correction case copies: `mreq_01a16d547a3770d186869108a305e42e` (ns_v1_007), `mreq_01a16d5483437f528e6cba8c292ac4bf`, `mreq_01a16d5495d8735396de43380d04c841`
- [ ] PRIMARY only: Requests filtered **Served by: Mubit model**: IDs saved as `data/adapter_golive/C/served_by_mubit-<time>.txt` and graded (runbook step 6). [R-pending: go-live R-row]

### Billing
- [ ] Console → Billing: $50.00 balance, no card (R72)

## 2. Races (`data/bench/bench.db` unless noted; JSONL exports `data/bench/race-<target>-seed<n>-<id>.jsonl`)
| Comparison | Race ids | R-id | Status |
|---|---|---|---|
| **RL wide: standard Opus vs Busy Machines (lean + backfill + planner every ≥60 s + checker) vs Optuna (preset `rl-wide-bm-cadence-15m`)** | `4ca4ed27` (seed 0), `d4913e95` (seed 1), `145c2a2d` (seed 2); fingerprint `e0d1cc7906` (3 seeds pool after D50) | **R78** | **The headline**, headline-eligible: same model, a tie (−154.3 vs −154.5), LLM $ 0.25× (list-price estimate; seed 2 0.34×), total 3.1× less (in-race GPU only); same quality reached in 2 of 3 seeds. Gate inputs `data/bench/cadence_rl/`. R74 caveat on Optuna |
| **LCBench wide, planner every ≥30 s: standard Opus vs Busy Machines (lean + backfill + cadence 30 s + checker) vs Optuna, pre-registered follow-up (preset `lcbench-wide-bm-c30-15m`, D49, `preregistration_lcbench_wide_c30.md`)** | `c1f4b1d9` (seed 0), `9d0cfd77` (seed 1), `b3922d0e` (seed 2); fingerprint `d41d7e04a4` | **R82** | **Headline-eligible, success line MET (4/4)**: best 0.8884 vs 0.8897 (within 0.0013, Opus ahead in every seed), LLM $ 0.25× ($0.698 vs $2.774, list-price estimate), total −35% (GPU $ virtual); never reached Opus's final best (0/3). **Chosen after R75 failed**; always shown beside R75. R74 caveat on Optuna |
| **LCBench wide: standard Opus vs Busy Machines (lean + backfill + cadence 90 s + checker) vs Optuna, pre-registered** | `613e2537` (seed 0), `6afbbe8a` (seed 1), `c72b3af1` (seed 2); fingerprint `6978df02c4` | **R75** | **FAILED its success line**: LLM $ 8.2× lower, model 7% more errors. Always shown beside R82 (and with R78). GPU $ virtual. R74 caveat on Optuna |
| **Image recognition (CIFAR-10, real L4 GPUs, 3 per side): standard Opus vs Busy Machines (lean + backfill + cadence 60 s + checker) vs Optuna (preset `cifar-bm-cadence-15m`)** | `e8106409` (seed 0), `9ec18124` (seed 1), `9f494453` (seed 2); fingerprint `f12f0585d7` | **R85** | **Quality gate FAILED**: AI bill $0.233 vs $1.056 (0.22×), but 0.9424 vs 0.9451 (5% more errors, behind in every run). Slower and dearer to 0.94 ($0.317 / 5.4 min vs $0.273 / 2.5 min); shipped model 0.9395, below the goal. Machines waiting 0.0% vs 12.0%. The only BM race on real GPUs; shown wherever the scorecard is. Modal metered ~$7.7 → ~$14.3 (~$2.25 per race). R74 caveat on Optuna |
| R78 goal framings (no new runs) | as R78 | **R87** | −170 goal 89% cheaper ($0.022 vs $0.200) and 72% faster (0.3 vs 1.0 min). Always with the caveat: the goal is easy, and the no-AI tool reached it for $0.009 / 0.4 min |
| Per-run values on the design frames (no new runs) | `4ca4ed27` (R78 run 1); R82 race reports | **R90** | R78 run 1 end: AI bill $3.30 vs $0.70, experiments 497 vs 1,594, waiting 69.0% vs 1.4%, quality −155.0 vs −154.9, fakes caught 0 vs 21. R82 BM AI bill split: planner $0.552 + checker $0.146 |
| 60-second live demo test (preset `demo-live-60s`, dev) | `15ab6dfc` (`data/busy_machines.db`; `data/dashboard_examples/demo-live-60s-test.jsonl`) | **R89** | **Demo tool works; not a result.** One run. At 60 s the standard agent was slightly ahead on quality and AI bill, so the video uses the R78 replay instead |
| RL wide, Busy Machines with Opus (no cadence) | `01004d9f`, `e648f7f6` | R67 | n=2, not headline |
| LCBench backfill pilot | `f76bebdd`, `d7fed919` | R65 | n=2, not headline |
| Lean planner check (`data/bench/ablation.db`) | `bd599a7c`, `a78b1a16`, `d78bfc3a` | R63 | dev, not headline |
| RL wide, Opus vs Optuna | `8a76fa05`, `45ab180f`, `59d7f7eb` | R48 | headline-eligible |
| CIFAR, Opus vs Optuna | `60477ba1`, `949fce51`, `c8261494` | R53 (seed 0 = R49) | headline-eligible |
| LCBench, Opus vs Optuna (judge v0) | `5245bacc`, `91a4e48d`, `f98b6712` | R38 | headline-eligible |
| LCBench, earlier BM engine (median stop) | `2ded10bc`, `e3fddb29`, `aaedd180` | R44 | BM worse; always disclose |
| RL wide, broken stopping rule | `f2bee077` | R41 | BM arm invalid; Opus 70.0% waiting valid |
| Planner-speed ablation (Haiku, **not Mubit**) (`ablation.db`) | see R50 | R50 | never a headline |

Commands (read-only):
```powershell
python -m uv run --extra ml python scripts/headline.py --race 4ca4ed27 --race d4913e95 --race 145c2a2d   # R78, the headline
python -m uv run --extra ml python scripts/headline.py --race c1f4b1d9 --race 9d0cfd77 --race b3922d0e   # R82, the tabular headline (chosen after R75)
python -m uv run --extra ml python scripts/headline.py --race 613e2537 --race 6afbbe8a --race c72b3af1   # R75, the fail
python -m uv run --extra ml python scripts/headline.py --race e8106409 --race 9ec18124 --race 9f494453   # R85, the real-GPU fail
python -m uv run --extra ml python scripts/headline.py --preset rl-wide-bm-cadence-15m                  # R78 goal framings (R87)
python -m uv run --extra ml python scripts/headline.py --preset rl-wide-15m
python -m backend.replay 4ca4ed27 --speed 10 --db data/bench/bench.db
```
`headline.py` prints "NOT A HEADLINE RESULT" whenever a gate fails. Read that line out if it appears.

**Dashboard server and replays** (R86, R88; `docs/dashboard_api.md`; $0, read-only on `bench.db`):
```powershell
$env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"
$env:BUSY_MACHINES_DB="data/busy_machines.db"
$env:BUSY_MACHINES_REPLAY_SOURCES="data/bench/bench.db"
python -m uv run --extra ml --with onnxruntime==1.29.0 uvicorn --factory backend.app:create_app --port 8765
Invoke-RestMethod "http://127.0.0.1:8765/api/dashboard/logged-races"
Invoke-RestMethod -Method Post "http://127.0.0.1:8765/api/races/4ca4ed27/replay?speed=90&metric_stride=0"   # R78 run 1, 15 min in 10 s
Invoke-RestMethod "http://127.0.0.1:8765/api/races/4ca4ed27/snapshot"                                         # end frame = R90 values
Invoke-RestMethod -Method Post "http://127.0.0.1:8765/api/demo/start?preset=demo-live-60s&dry_run=true"      # $0 rehearsal (mock AI)
```

## 3. Checker and correction evidence
- [ ] Noise set ground truth: `data/noise_set/noise_set_v1.jsonl`, re-runs with Modal call IDs in `data/noise_set/reruns.jsonl` (R34, 440 runs)
- [ ] Combined checker (7/7, 1/36, 0/32): `data/noise_set/eval/noiseeval-regraded-…jsonl` (R37)
- [ ] Held-out test split, judge v1: `data/trap_eval/trapeval-2026-10-24-claude-opus-5-5-test-043437.*`, `data/noise_set/eval/noiseeval-2026-10-24-043502.*`; first IDs `mreq_01a16c30942d…`, `mreq_01a16c30f388…` (R40)
- [ ] Try models cases (held out, never recorded): `data/adapter_golive/try_models/` with Opus's R40 answers: `trap_v1_020` fake → `invalid` (`mreq_01a16c309a4571329445dceb3b68140b`), `trap_v1_024` real → `valid` (`mreq_01a16c309f987b92924fb6f4f9551405`), `trap_v1_048` noise → `unsure` (`mreq_01a16c30aefb77a0b9d1d096db3bd43c`)
- [ ] Correction row: correction case experiment `700507e1dfe046178c072aed2f9a3aa0` (race `addfe714…`, `collect_rl.db`; Modal run `fc-01M5NK1RC2R6F336BA6KNDP6XE`). Flag it on a **scratch copy** only (`demo_script.md` §4.3)
- [ ] Fakes across 9 races (Optuna 59, Opus 0): `data/pitch_evidence/agg.json` (R54)

## 4. Hugging Face and Sapling
- [ ] Stock card: https://huggingface.co/MubitLabs/banyan-sapling
- [ ] Mubit's hosted Sapling endpoint (D33): parity 30/30 raw labels (R42)
- [ ] `sapling-research` v2: **not published** (R69). The local package `data/sapling/sapling_v2/hf_release/` reproduces R42 on CPU (550/550 labels, R69). [HF link pending: publish decision]. Never show a link that doesn't exist
- [ ] Results doc: `docs/sapling_results_v2.md` (R42)
- [ ] Published Sapling vs v2 vs Opus on 226 real race claims: `data/sapling/compare_published/compare_published.json` (R81)
- [ ] v4 **failed its gate**, so v2 stays the release (R83). At α 0.01 on R81's real claims v4 caught 97 of 109 fakes, approved 0 and confirmed 5 of 7 real gains. It regressed on general claims, ContractNLI and trap50. Files: `data/sapling/sapling_v4/heads/final_v4.json`, `gate_result.json`; the kept files are `release_v4_failed_gate/`, not released
- [ ] HF publish of v2 needs the user's HF name, `hf auth login` and "go"

## 5. Compute and money
- [ ] Modal: new account `emeraldshuorbit`, $0.00 metered at the switch; old account `flagformgaze51` retired at $26.49 metered (R73). Demo reserve ≥$6 (R73)
- [ ] Modal call IDs per run: `experiments.run_ref` in `bench.db`; smoke examples `fc-01M5NFKXPXW67JMCPMVNPWQ2QC` (R12, an earlier workspace), `fc-01M5NHPS21BEVQPMGTCM277E7R` (R16, `flagformgaze51`). Modal call IDs from before ~15:20 on Saturday were made on accounts retired since (R73), so the new `emeraldshuorbit` dashboard won't list them
- [ ] Spend split per arm (LLM vs GPU): R54; R78 per race: Opus $3.30 LLM of $3.65, BM $0.83 of $1.18; R82 per race: Opus $2.774 LLM of $5.97, BM $0.698 of $3.89 (GPU $ virtual) (LLM $ list-price estimate)
- [ ] Modal metered for R78: $0.29 → $7.93 for the 3 races + first deploy (~$2.5 per race vs $1.05 in-race GPU; pre-warm/idle containers, same for all arms) (R78). Say this if asked whether total $ is the Modal bill
- [ ] Modal metered for R85: ~$7.7 → ~$14.3 for the 3 races (~$2.25 per race) (R85)

## 5b. Backend checks and design frames
- [ ] Backend smoke test: API + live events + replay, $0 (R86). Open item: `GET /api/races/{id}` returned null fields for the TestClient id
- [ ] Freeze check: patches 06–09 applied, full suite 817 passed / 3 skipped / 0 failed, scans clean, Optuna contention 2.52× / 2.49× → 1.40× / 1.46× slower, reduced not removed (R88). Logs in the scratchpad: `final_pytest.log`, `contention_rl600*.json`
- [ ] Demo endpoint: `demo-live-60s` test run `15ab6dfc` (R89). **Demo tool works; not a result**
- [ ] Design frames in `mubit.pen` (open it only through the pencil MCP). The PNGs are in `design/exports/final/<id>.png`, and the number check is in `design/exports/final/README.md`:
  - **Landing page `pOKAY`:** hero `KekD1` (copy says "GPUs" alone; fix before use, D54), engine `QYavo` (R78), our model `MgmgS` (R81; its "99.9% vs 88.7%" is not logged, so remove it before use), benchmarks `oT6Ad` (R78, R82, R87; add R75, R85 and the R87 caveat before use), call to action `c50WCZ`.
  - **Platform:** Overview `bxWh3` (R78, R82, R90) and Sweep detail (live compare) `eGVso` (R78 run 1, R90).

## 6. Docs to have open for Q&A
- [ ] `docs/qa_mubit_constraints.md`
- [ ] `docs/preregistration_lcbench_wide.md` (frozen before the race, with the D48 addendum; result R75, failed)
- [ ] `docs/preregistration_lcbench_wide_c30.md` (D49 follow-up, frozen before its race, chosen after R75; result R82, success line met)
- [ ] `docs/mubit_field_report.md` (F1–F15)
- [ ] `docs/results_log.md` (search by R-id)
- [ ] `docs/decisions.md` (D36, D38, D41, D44–D54; D54 = positioning "compute intelligence for training sweeps", "Runs on Mubit")
- [ ] `docs/results_log.md` R74 (the Optuna-side caveat in BM races), R75, R78, R82, R85 (real-GPU fail), R87 (goal framing + caveat), R89 (demo, not a result), R90 (design-frame values)
- [ ] `docs/dashboard_api.md` (server, replay, demo endpoint)
