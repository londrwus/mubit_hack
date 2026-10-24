"""sapling-research v3 rows: judge v1 inputs (real race / collection / noise-set claims) -> Sapling triples.

Turns the ``judge_improvement`` user messages (TARGET SPEC + CLAIM + EVIDENCE, as the live judge sees them)
into Sapling's ``(policy, evidence, claim)`` format, with labels that are honest about where they come from:

* **policy**  = the target spec (task, metric, config space with every option's description, which is what
  decides leakage, e.g. ``eval_on_train_levels`` or ``checkpoint_selection=best_test``) + the judge v1
  decision rules restated in Sapling's label words + "unstated facts are unknown".
* **evidence** = the CLAIM block without the claim sentence (previous best, new score, both configs,
  claimed gain, changed parameters) + the EVIDENCE block, verbatim.
* **claim**   = "<target>: the new config improves <metric> to <new> (was <prev>); the gain is real."

Label mapping (judge v1 rules -> Sapling): invalid (leak, changed eval) -> ``violation``; unsure (inside the
noise, leakage not ruled out) -> ``insufficient_information``; valid (clean gain) -> ``supported``.
No row is ``inconsistent_context``.

Label sources, per row (``label_source``):
* ``ground_truth``: the planted trap / measured truth fixes a single Sapling label (a trap run, whose trap is an
  enabled config option the policy names -> ``violation``; a noise-set gain that 10+10 seed re-runs show is
  noise -> ``insufficient_information``).
* ``opus_v1_teacher``: the truth only rules labels out (a non-trap run can be ``supported`` or
  ``insufficient_information`` depending on what the evidence shows), so Opus 5.5's recorded judge v1
  verdict (majority of its valid calls) picks the label, if it is in the allowed set.
* dropped: the teacher is outside the allowed set and the set is not a single label, or the teacher's calls
  tie. Every dropped row is listed in the summary.

Exclusions (never training rows): the 30 ``trap_set`` cases (they are trap50 rows, held out for every Sapling
version). The noise-set **test** split (40) is converted as an ``eval_noise_test`` set with ground-truth labels
only (no Opus v1 verdicts exist for it), for evaluation, never for training.

    python -m uv run python scripts/sapling_v3_rows.py   -> data/sapling/sapling_v3/{rows_train.jsonl,rows_eval_noise_test.jsonl,summary.json}

Pure text processing: no LLM, Modal or GPU call.
"""

from __future__ import annotations

import collections
import glob
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sapling" / "sapling_v3"
JV1 = ROOT / "data" / "gates" / "judge_v1" / "record"
V2B = ROOT / "data" / "adapter_golive" / "v2_batch" / "inputs.jsonl"
V2B_TRAFFIC = sorted(glob.glob(str(ROOT / "data" / "adapter_golive" / "B" / "traffic-*.jsonl")))
NOISE = ROOT / "data" / "noise_set" / "noise_set_v1.jsonl"
VERSION = "sapling_v3_rows_v1"

VERDICT_TO_LABEL = {"invalid": "violation", "unsure": "insufficient_information", "valid": "supported"}
SUP, VIO, INS = "supported", "violation", "insufficient_information"

RULES = """Rules for a claimed new best score:
1. Leakage breaks this policy: a held-out split (validation or test) sharing content with the training data (equal hashes, a non-zero overlap count, or an enabled option whose description says it draws evaluation data from the training pool); the test split, or the split the score is reported on, used to choose checkpoints, hyperparameters, augmentations or early stopping; or evaluation on the same seeds, levels or environments as training when they must differ (including an enabled option whose description says the evaluation runs on the training levels).
2. Two scores may be compared only if both come from the same evaluation: the same split, metric, number of evaluation examples, episodes, levels and seeds. A changed evaluation breaks this policy.
3. A gain is not established if a seed-to-seed spread is shown and the gain is not larger than it, or if only the best of several seeds beats the previous best while the seed mean does not.
4. A gain is not established while leakage cannot be ruled out: run metrics alone (timings, training returns or losses, the reported score) do not rule it out. Distinct content hashes for the training data and the scored held-out split, with no enabled option that draws evaluation data from training or selects on the test split, do rule it out.
5. A gain is real only if none of the above applies and the evidence shows a clean, unchanged evaluation.
Facts missing from the evidence count as unknown."""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def convert(user_message: str) -> dict[str, str]:
    """judge user message -> {policy, evidence, claim}. Raises on an unexpected layout."""
    if "<<<CACHE_BREAK>>>" not in user_message:
        raise ValueError("no cache break")
    spec, rest = user_message.split("<<<CACHE_BREAK>>>", 1)
    spec_lines = [ln for ln in spec.strip().splitlines() if not ln.startswith("default config:")]
    if not spec_lines or spec_lines[0] != "TARGET SPEC":
        raise ValueError("spec header")
    name = next(ln.split(":", 1)[1].strip() for ln in spec_lines if ln.startswith("name:"))
    metric_line = next(ln.split(":", 1)[1].strip() for ln in spec_lines if ln.startswith("metric:"))
    metric = metric_line.split(" ", 1)[0]
    policy = (f"Evaluation policy: the {name} benchmark\n" + "\n".join(spec_lines[1:]) + "\n" + RULES)

    rest = rest.strip()
    if not rest.startswith("CLAIM"):
        raise ValueError("claim header")
    body = rest[len("CLAIM"):].strip()
    m_prev = re.search(rf"^previous best {re.escape(metric)}: (\S+)$", body, re.M)
    m_new = re.search(rf"^new {re.escape(metric)}: (\S+)$", body, re.M)
    if not (m_prev and m_new) or "\nEVIDENCE" not in "\n" + body:
        raise ValueError("claim fields")
    claim_part, ev_part = body.split("EVIDENCE", 1)
    evidence = "=== Claim details ===\n" + claim_part.strip() + "\n=== Evidence ===\n" + "\n".join(
        ln.strip() for ln in ev_part.strip().splitlines())
    claim = f"{name}: the new config improves {metric} to {m_new.group(1)} (was {m_prev.group(1)}); the gain is real."
    return {"policy": policy, "evidence": evidence, "claim": claim, "target_name": name, "metric": metric}


def teacher_votes() -> dict[str, list[str]]:
    votes: dict[str, list[str]] = collections.defaultdict(list)
    for p in sorted(glob.glob(str(JV1 / "calls.*.jsonl"))) + V2B_TRAFFIC:
        for r in read_jsonl(p):
            if r.get("valid") and r.get("verdict") in VERDICT_TO_LABEL and r.get("prompt_fingerprint") == "5c1d3cfd2941":
                votes[r["case_id"]].append(r["verdict"])
    return votes


def teacher_label(v: list[str]) -> tuple[str | None, str]:
    if not v:
        return None, "no_teacher"
    c = collections.Counter(v).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None, f"teacher_tie:{dict(c)}"
    return VERDICT_TO_LABEL[c[0][0]], ("unanimous" if len(c) == 1 else f"majority:{dict(c)}")


def allowed_labels(row: dict[str, Any]) -> tuple[set[str], str]:
    """The Sapling labels the known truth allows, and a name for that truth."""
    src, exp, gt = row["source"], row.get("expected") or {}, row.get("ground_truth") or {}
    if src == "collection":
        return ({VIO}, "trap") if exp.get("kind") == "trap" else ({SUP, INS}, "clean_run")
    if src == "race":
        if "is_trap" in exp:
            return ({VIO}, "trap") if exp["is_trap"] else ({SUP, INS}, "non_trap_run")
        return ({VIO}, "trap") if gt.get("truth") == "fake" else ({SUP, INS}, "non_trap_run")
    if src == "noise_set":
        truth = exp.get("truth") or row.get("truth")
        return {"fake": ({VIO}, "fake"), "noise": ({INS}, "noise"), "real": ({SUP, INS}, "real")}[truth]
    raise ValueError(src)


def label_row(row: dict[str, Any], votes: dict[str, list[str]]) -> tuple[str | None, str, dict[str, Any]]:
    allowed, truth = allowed_labels(row)
    t_label, t_note = teacher_label(votes.get(row["id"], []))
    meta = {"truth": truth, "allowed": sorted(allowed), "teacher_label": t_label, "teacher_note": t_note,
            "teacher_votes": votes.get(row["id"], [])}
    if len(allowed) == 1:
        lab = next(iter(allowed))
        return lab, "ground_truth", meta | {"teacher_agrees": t_label == lab}
    if t_label in allowed:
        return t_label, "opus_v1_teacher", meta | {"teacher_agrees": True}
    return None, "dropped", meta | {"teacher_agrees": False}


def main() -> None:
    votes = teacher_votes()
    jv1 = read_jsonl(JV1 / "inputs.jsonl")
    v2b = read_jsonl(V2B)
    for r in v2b:
        r["source"] = "race"
    train: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    excluded = collections.Counter()
    seen_text: dict[str, str] = {}
    for batch_name, rows in (("judge_v1_177", jv1), ("judge_v1_v2batch_40", v2b)):
        for r in rows:
            if r["source"] == "trap_set":
                excluded["trap_set (trap50 rows, held out)"] += 1
                continue
            tri = convert(r["user_message"])
            lab, how, meta = label_row(r, votes)
            key = hashlib.sha256(json.dumps([tri["policy"], tri["evidence"], tri["claim"]]).encode()).hexdigest()
            out = {"id": f"v3_{r['id']}", "orig_id": r["id"], "batch": batch_name, "source": r["source"],
                   "target": r.get("target"), "group": r.get("group"), "split": "train_v3",
                   "policy": tri["policy"], "evidence": tri["evidence"], "claim": tri["claim"],
                   "label": lab, "label_source": how, "label_meta": meta, "claim_kind": "improvement",
                   "fake_improvement": meta["truth"] in ("trap", "fake", "noise"),
                   "text_sha256": key, "set_version": VERSION}
            if key in seen_text:
                excluded[f"duplicate text of {seen_text[key]}"] += 1
                continue
            seen_text[key] = out["id"]
            (train if lab else dropped).append(out)

    # noise-set test split: eval only, ground truth only
    ev: list[dict[str, Any]] = []
    for r in read_jsonl(NOISE):
        if r["split"] != "test":
            continue
        tri = convert(r["user_message"])
        allowed, truth = allowed_labels({"source": "noise_set", "truth": r["truth"]})
        key = hashlib.sha256(json.dumps([tri["policy"], tri["evidence"], tri["claim"]]).encode()).hexdigest()
        if key in seen_text:
            raise SystemExit(f"noise test row {r['id']} duplicates a training row {seen_text[key]}")
        ev.append({"id": f"v3_noise_{r['id']}", "orig_id": r["id"], "source": "noise_set", "target": r["target"],
                   "group": r["group"], "split": "eval_noise_test", "policy": tri["policy"], "evidence": tri["evidence"],
                   "claim": tri["claim"], "label": next(iter(allowed)) if len(allowed) == 1 else None,
                   "allowed": sorted(allowed), "truth": truth, "label_source": "ground_truth",
                   "fake_improvement": truth in ("fake", "noise"), "claim_kind": "improvement",
                   "text_sha256": key, "set_version": VERSION})

    write_jsonl(OUT / "rows_train.jsonl", train)
    write_jsonl(OUT / "rows_dropped.jsonl", dropped)
    write_jsonl(OUT / "rows_eval_noise_test.jsonl", ev)
    c = collections.Counter
    summary = {
        "set_version": VERSION,
        "train_rows": len(train),
        "train_by_label": dict(c(r["label"] for r in train)),
        "train_by_label_source": dict(c(r["label_source"] for r in train)),
        "train_by_source_label": dict(c(f"{r['source']}/{r['label_meta']['truth']}/{r['label']}" for r in train)),
        "train_by_target": dict(c(r["target"] for r in train)),
        "ground_truth_rows_where_teacher_disagrees": [
            {"id": r["id"], "label": r["label"], "teacher": r["label_meta"]["teacher_label"],
             "votes": r["label_meta"]["teacher_votes"]}
            for r in train if r["label_source"] == "ground_truth" and not r["label_meta"]["teacher_agrees"]],
        "dropped": [{"id": r["id"], "truth": r["label_meta"]["truth"], "allowed": r["label_meta"]["allowed"],
                     "teacher": r["label_meta"]["teacher_label"], "note": r["label_meta"]["teacher_note"]}
                    for r in dropped],
        "excluded": dict(excluded),
        "eval_noise_test_rows": len(ev),
        "eval_noise_test_by_truth": dict(c(r["truth"] for r in ev)),
        "max_chars": max(len(r["policy"]) + len(r["evidence"]) + len(r["claim"]) for r in train + ev),
        "label_mapping": VERDICT_TO_LABEL,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
