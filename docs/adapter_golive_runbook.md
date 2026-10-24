# Adapter go-live runbook (workspace C, judge v1)

What to do in the first hour after the **judge v1 adapter card appears in workspace C** (task `ln_dfe18721334e06d23c32f338`, 177 Opus examples recorded in R59, training started ~11:20 on 24 Oct, R61).

The point of this hour: find out whether Mubit's small model passed, help it go live, and get proof (request IDs marked **Served by: Mubit model**) that we can show on stage. Every number we quote later has to come from a step below and go into `docs/results_log.md`, whether the result is good or bad.

**Rules that apply throughout:**
- Every call goes through Mubit with the workspace's own key (`MUBIT_API_KEY_C` from `.env`). The scripts never print keys. Never use `ANTHROPIC_API_KEY`.
- The judge prompt stays frozen (v1, fingerprint `5c1d3cfd2941`). The scripts refuse to run if it has changed.
- Mubit grades the adapter **against Opus's answers, not against the truth**. So "Passed" means "agrees with Opus often enough". Our own scripts report both numbers: agreement with Opus, and accuracy against ground truth.
- Going live happens in three automatic steps (R56): **pass** (quality bar 90%) → **silent test** (the new version answers copies of a share of calls; Opus still answers) → **partial rollout** (the new version answers a share of calls for real). The API response doesn't say who answered. Only the console's **Served by** field does.

Set this once in PowerShell before running any command:
```powershell
cd C:\Users\Lenovo\Documents\mubit_hack
$env:PYTHONUTF8=1; $env:PYTHONIOENCODING='utf-8'
```

## Costs at a glance
An Opus judge v1 call costs about **$0.0036–0.0043** (R59: 177 calls for $0.639; R40: 60 calls for $0.26). Mubit's price is "$5 per 1,000 runs" ($0.005 per call). Whether that fee applies at the hackathon is still open (office hours Q10), so the scripts count it against the cap anyway. When the adapter answers, we don't know its price; the scripts estimate it at the Opus list price.

| # | Command | Calls | Estimated cost |
|---|---|---|---|
| 0 | dry run (mock, no network) | 0 | **$0** |
| 1 | check on held-out cases | 60 | **~$0.26** Opus, plus up to $0.30 Mubit fee |
| 2 | traffic for silent test / rollout | 237 | **~$0.90** Opus, plus up to $1.19 Mubit fee (cap $2.50) |
| 3 | record the v2 batch (40 new examples) | 40 | **~$0.15** Opus, plus up to $0.20 Mubit fee |
| 4 | grade the calls the adapter answered | 0 | **$0** |

## Step 0. Rehearse (2 minutes, $0)
```powershell
python -m uv run --extra ml python scripts/push_judge_traffic.py push --dry-run --sources trap_test,noise_test
python -m uv run --extra ml python scripts/eval_served_calls.py (Get-ChildItem data/adapter_golive/C/dryrun/traffic-*.jsonl | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
```
The dry run uses a local mock that pretends some answers came from an adapter. Its numbers are **not measurements**, and the report says so. Any push without `--dry-run` makes paid calls and refuses to start without `--yes`. The script never pushes to workspace A (the Opus baseline).

## Step 1. Read the adapter card (5 minutes)
In the console: workspace C → Tasks → the judge task → Adapters. Note:
1. **Version and status**: Passed / Testing / Below target / Prompt only (*"the base model ... did as well as the adapter when given the task's prompt"*).
2. **Quality**: "Passed X% of N test cases" against the **90% bar** (Settings → Going live → Quality bar; R56 showed 90% on B, while the cards showed 95%; write down which one C's card shows).
3. **Test cases N**. The exam needs **at least 20** (R30). If N is under 20, the card may refuse to grade.
4. **Malformed replies**: "k of n ... malformed · needs 0", if shown.
5. The **failed test cases**, if the card lists them: which verdicts or reason codes it confused.
6. **Routes / Going live state**: silent test, partial rollout, or nothing yet.

**Paste back:** the card text as it appears, the time it appeared, and the history lines (found / reviewed / training started / trained). From the training start (~11:20, R61) to the card appearing gives the training time for the log.

## Step 2. Set C's Going live settings like B (3 minutes)
Admin → Settings on workspace C → **Going live**. Set the same values the user set on B at ~10:50 (R56):
- **Silent test: 20% of calls, up to 100 a day**
- **Partial rollout: 50% of calls, for at least 100 calls**
- Approval: leave **automatically, once it passes**.
- Quality bar: **leave it at 90%**. Lowering it would let a version serve that disagrees with Opus more often, and the demo would have to say so.

Also open **Training** and check "New versions: after 25 new examples" (B showed this, R58) and "Training limit: no daily limit".

**Paste back:** the Going live and Training tabs as text, after your changes.

## Step 3. Decide

| Card shows | Meaning | Do |
|---|---|---|
| **Passed, ≥ 90%** on ≥ 20 test cases | It will go live by itself | Step 4 (check on held-out cases), then step 5 (traffic), then step 6 (Served by) |
| **Near miss, 80–89%** | It won't serve. A v2 trained on more examples might pass | Skip the traffic. Run step 7 (record the v2 batch) now, then re-read the card when v2 appears. Keep v1's score for the fallback demo |
| **Below 80%**, Prompt only, or malformed > 0 | It won't serve today | Use the fallback demo (demo.md §2) and quote v1's score with its R-id. If the failures are confusions between classes (e.g. unsure vs invalid), step 7 may still help. Run it only if there are **≥ 3 hours before the Sun 08:00 decision**. If replies are malformed, it's the output-format question (office hours §3): ask Mubit and don't retrain blindly |
| N < 20 test cases | The exam can't be passed yet | Step 7: each v2 case has its own session, which gives Mubit more sessions to set test cases aside from |

## Step 4. Check on held-out cases (pass only, ~$0.26)
These 60 cases (trap set test 20, noise set test 40) were **never recorded in any workspace**, so the adapter can't have trained on them. R40 has Opus's answers on exactly these cases.
```powershell
python -m uv run --extra ml python scripts/push_judge_traffic.py push --sources trap_test,noise_test --yes
```
- Default `--record metadata`. The docs say *"`metadata` keeps the fingerprint but drops the content"* (docs.mubit.ai/reference/api, quoted in office hours §2). So the calls stay in the task, which lets them be routed and should count toward silent test and rollout, and they add no examples (*"A call recorded without its request and reply is never a kept example"*, office hours Q11).
- **Caveat:** the same docs say `off` and `metadata` calls *"never list"* in Requests. If the IDs printed by this command can't be found in Requests, that is the reason. See step 6.
- Output: `data/adapter_golive/C/traffic-<stamp>.jsonl` (every attempt: case, ground truth, Opus's reference answer, verdict, reason code, request ID, latency, tokens, $), plus `.request_ids.txt` and `.summary.json`.

**Paste back:** the summary JSON the command prints at the end.

## Step 5. Traffic for silent test and rollout (pass only, ~$0.90, cap $2.50)
At 20% silent test and 50% rollout for at least 100 calls, races alone (~15–20 judge calls each) would take all weekend. This command pushes every recorded input plus the held-out cases once:
```powershell
python -m uv run --extra ml python scripts/push_judge_traffic.py push --sources jv1,trap_test,noise_test --usd-cap 2.5 --yes
```
Watch the card's going-live counters. If it still needs more calls, run the same command again (each run is another ~$0.90 to $2.10 and gets a new traffic file). `--repeat 2` sends each case twice in one run; raise `--usd-cap` to match. Stop as soon as the card shows partial rollout or live. Concurrency is capped at 4, and 429/503 replies back off on their own (R39 saw 80 × 503).

## Step 6. Find the calls the adapter answered (pass + rollout, $0)
1. Console → workspace C → **Requests** → filter **Served by: Mubit model** (and, if possible, by session `golive-…`).
2. Copy the request IDs you see (`mreq_…`) into a text file, e.g. `data/adapter_golive/C/served_by_mubit-<time>.txt`. Pasting raw console text is fine; the script picks out every `mreq_…`.
3. Grade them:
```powershell
python -m uv run --extra ml python scripts/eval_served_calls.py data/adapter_golive/C/traffic-<stamp>.jsonl --served-ids data/adapter_golive/C/served_by_mubit-<time>.txt
```
The report splits the calls into **Mubit model** and **Opus**, and each of those into **held-out** (quote these) and **training pool** (the adapter may have trained on these cases, so its score there is optimistic). For each it shows: accuracy against ground truth (trap set: fake caught / fake let through / real confirmed; noise set: real kept / real confirmed / noise accepted / fake caught), agreement with Opus's answer on the same case (what Mubit grades), the cases both answered, and latency p50/p90 by who answered. Several traffic files can be passed at once.

**If the IDs don't show up in Requests** (because metadata calls "never list"): run the eval without `--served-ids`. It prints the overall numbers and the latency distribution. A second, much faster group of calls is labelled **"HINT ONLY"**. It is not proof and must never be quoted as "the adapter answered N calls". Proof options, in order:
- the v2 recording (step 7) is `record full`, so its 40 calls do list in Requests and show Served by;
- the Mubit arm's race calls run on `metadata` today (`backend/benchmark/configs.py`), so they may not list either. **Flag for the lead:** demo.md §2 (primary version) needs one race judge call visible under Served by. Check this with one ID before the demo.

**Paste back:** the full report text, and the IDs file.

## Step 7. Record the v2 batch (on cue, ~$0.15)
`data/adapter_golive/v2_batch/inputs.jsonl` holds **40 new judge v1 inputs**. They are claims from finished races in `data/bench/bench.db` (29) and `ablation.db` (11), rebuilt exactly like `record_judge_v1.py` `race_claims`. None of them is among the 177 recorded inputs or equals a held-out message. 11 are fakes (the trap option was on), and 29 are not fakes (real or noise unknown). Targets: RL 15, tabular 19, image 6. We record 40 because 25 **kept** examples trigger "New versions after 25 new examples" (R58), and on C 18 of 177 couldn't be reviewed (R61). Build summary: `data/adapter_golive/v2_batch/summary.json`.
```powershell
python -m uv run --extra ml python scripts/push_judge_traffic.py push --sources v2 --record full --yes
```
- One session per case (`collect-judgev1v2-2026-10-24/race/<n>`), like the original recording, so Mubit can set test cases aside by session.
- **Timing:** on C, recording to training start took ~20 minutes (R61). Training time: measure it from step 1. Record at least that much, plus a margin, before you need v2 on screen.
- **Open question for Mubit:** if this runs during partial rollout, some of these calls may be answered by the adapter. Would those be kept as examples (the adapter learning from itself)? If unsure, record v2 **before** rollout starts, or ask first.
- **What we may say:** "v2 was trained on 40 more checks that Opus labelled." Don't say "learned from your correction" (demo.md §2) unless the batch is rebuilt from cases we flagged.
- The script refuses `--record full` on held-out cases, and on the 177 already-recorded inputs (that would need `--allow-duplicates`).

**Paste back:** the summary JSON, then the card's "Next: a new version after …" line.

## Step 8. Route the Mubit arm's judge to C (after a pass and once it serves)
Change `data/mubit_projects.json` to:
```json
{
  "default": "A",
  "routes": {
    "mubit": {
      "judge_improvement": "C",
      "early_stop": "B"
    }
  },
  "keys": {
    "A": "MUBIT_API_KEY",
    "B": "MUBIT_API_KEY_B",
    "C": "MUBIT_API_KEY_C"
  }
}
```
- Only the `mubit` arm moves. The Opus arm stays on A with `record: off` (fairness, D38). `early_stop` stays on B (it's an engine rule now, D40).
- `tests/test_mubit_projects.py` pins today's file and will fail on purpose: update `test_repo_config_is_the_d38_rule`, `test_project_for_every_arm_and_step` (judge → C) and, if wanted, `DEFAULT_PROJECT_ROUTING` in `backend/agent/mubit_client.py`, all in the same change. Then add a decision to `docs/decisions.md` and log the switch.
- When Mubit ships the announced adapter-pin header, add it to the file's `adapter_headers` block (`{"mubit": {"judge_improvement": {"<header>": "<adapter id>"}}}`). `push_judge_traffic.py --adapter-header <name>=<id>` can test it first.

## Try models demo (3 cases, held out)
Try models is console only, *"not stored · trains nothing"*. Each start of an adapter uses one of **3 adapter starts per org per UTC day**, and the first start can take **up to 20 minutes**. Start it at least 20 minutes before you need it, and don't spend starts on rehearsals you don't need.

All three cases are from the trap set's **test split**: never recorded in any workspace, so the adapter has never seen them. All are game-playing AI (demo.md prefers that target). Expected answers are Opus 5.5's answers with judge v1 from R40. The full messages to paste are in `data/adapter_golive/try_models/` (`*.user.txt`, plus `system_prompt.judge_improvement.v1.txt` if Try models asks for the instructions). Regenerate them with `python scripts/push_judge_traffic.py export-try-models`.

### Case 1. A fake improvement Opus catches (`trap_v1_020`, file `fake_caught.trap_v1_020.user.txt`)
```
CLAIM
previous best mean_return: -390.3
new mean_return: -359.7
claimed gain: 30.6 (higher is better)
changed parameters: update_epochs 4 -> 8, hidden_size 64 -> 128, eval_on_train_levels false -> true
EVIDENCE
  eval_level_seed: 1000
  n_eval_levels: 16
  n_train_levels: 16
  train_level_seed: 1000
  prev_best_seed_std: 17.1
  score_from: seed 4 only
  (plus the other evidence lines in the file)
```
- **Expected (Opus, R40):** `{"verdict": "invalid", "reason_code": "contamination_or_eval_leak"}` (`mreq_01a16c309a4571329445dceb3b68140b`)
- **Truth:** fake. It was scored on the 16 training levels (eval level seed 1000 = training level seed) instead of the 64 held-out levels.
- **Say:** "It tested on levels it trained on. The small model flags it, just like Opus."

### Case 2. A real gain (`trap_v1_024`, file `real_gain.trap_v1_024.user.txt`)
```
CLAIM
previous best mean_return: -1059
new mean_return: -501.2
claimed gain: 558.2 (higher is better)
changed parameters: lr 0.0001 -> 0.0003, num_envs 8 -> 32, update_epochs 4 -> 8
EVIDENCE
  checkpoint_selection: final policy (no selection)
  eval_level_seed: 2000
  eval_levels_seen_in_training: 0
  n_eval_levels: 64
  prev_best_seed_std: 55.1
  score_from: mean of seeds [0, 1, 2]
  seed_scores: {"0": -522.2, "1": -495.8, "2": -485.7}
  (plus the other evidence lines in the file)
```
- **Expected (Opus, R40):** `{"verdict": "valid", "reason_code": "clean_gain"}` (`mreq_01a16c309f987b92924fb6f4f9551405`)
- **Truth:** real. Distinct held-out levels, and a gain of 558 on three seeds against a seed spread of 55.
- **Say:** "A real improvement: it keeps it."

### Case 3. Noise (`trap_v1_048`, file `noise.trap_v1_048.user.txt`)
```
CLAIM
previous best mean_return: -275.9
new mean_return: -268.4
claimed gain: 7.5 (higher is better)
changed parameters: ent_coef 0 -> 0.005
EVIDENCE
  checkpoint_selection: final policy (no selection)
  eval_levels_seen_in_training: 0
  prev_best_seed_std: 15.7
  score_from: seed 2 only
  seed_scores: {"2": -268.4}
  (plus the other evidence lines in the file)
```
- **Expected (Opus, R40):** `{"verdict": "unsure", "reason_code": "gain_within_noise"}` (`mreq_01a16c30aefb77a0b9d1d096db3bd43c`)
- **Truth:** noise. One run gained 7.5, less than the previous best's seed spread of 15.7 (the trap set accepts invalid or unsure).
- **Say:** "Smaller than the run-to-run wobble: it won't call it a win."

**If the adapter's answer differs from the expected one on any case:** say so on stage or pick another case beforehand. Never edit the expected answer. The trap set's other test cases, with R40 answers, are in `data/trap_eval/trapeval-2026-10-24-claude-opus-5-5-test-043437.jsonl`.

## What goes into the results log (the lead writes it)
- Card: version, quality %, N test cases, bar, malformed count, timings (one row, with the console text).
- Each push: calls, valid, 503 count, $ (from `.summary.json`), first request IDs, and the traffic file path.
- The eval: held-out accuracy and agreement with Opus for the Mubit model vs Opus, as **counts**, with p50/p90 latency by who answered. Log it even if the adapter does worse than Opus.
- Anything from a dry run is never logged as a result.
