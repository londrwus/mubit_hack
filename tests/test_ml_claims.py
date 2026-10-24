"""The ml-claims Sapling dataset (scripts/make_ml_claims.py). No LLM, no Modal, no GPU."""

from __future__ import annotations

import importlib.util
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_ml_claims", REPO / "scripts" / "make_ml_claims.py")
assert _spec and _spec.loader
mmc = importlib.util.module_from_spec(_spec)
sys.modules["make_ml_claims"] = mmc
_spec.loader.exec_module(mmc)

DATA: dict[str, list[dict[str, Any]]] = mmc.build_all()
TRAIN, TEST, TRAP50 = DATA["train"], DATA["test"], DATA["trap50"]
ALL = TRAIN + TEST + TRAP50
TRAP_CASES = {c["id"]: c for c in mmc.mts.build_cases()}


# --------------------------------------------------------------------------- determinism / files


def test_build_is_deterministic() -> None:
    again = mmc.build_all()
    for split in DATA:
        assert mmc.render_jsonl(again[split]) == mmc.render_jsonl(DATA[split])


def test_committed_files_match_generator() -> None:
    for split, name in mmc.FILES.items():
        path = mmc.OUT_DIR / name
        assert path.exists(), "run scripts/make_ml_claims.py"
        assert path.read_bytes().decode("utf-8") == mmc.render_jsonl(DATA[split]), f"{name} is stale"
    assert mmc.main(["--check"]) == 0


def test_check_flags_a_stale_dir(tmp_path: Path) -> None:
    assert mmc.main(["--check", "--out-dir", str(tmp_path)]) == 1


def test_generator_makes_no_llm_or_modal_calls() -> None:
    src = (REPO / "scripts" / "make_ml_claims.py").read_text(encoding="utf-8")
    for banned in ("import modal", "anthropic", "openai", "mubit_client", "httpx", "requests"):
        assert banned not in src


# --------------------------------------------------------------------------- Sapling input contract


def test_sapling_input_contract() -> None:
    for r in ALL:
        for k in ("policy", "evidence", "claim"):
            assert isinstance(r[k], str) and r[k].strip(), (r["id"], k)
        assert r["label"] in mmc.LABELS
        # 16,384-token hard limit in Sapling's encode(); stay far below it (chars >= tokens).
        assert len(r["policy"]) + len(r["evidence"]) + len(r["claim"]) < 6000, r["id"]
        assert all(x in "\n\t" or 32 <= ord(x) < 127 for x in r["policy"] + r["evidence"] + r["claim"]), r["id"]


def test_ids_unique_and_sizes() -> None:
    assert len(TRAIN) == mmc.N_TRAIN == 3000
    assert len(TEST) == mmc.N_TEST == 500
    assert len(TRAP50) == 50
    assert len({r["id"] for r in ALL}) == len(ALL)


def test_no_duplicate_triples_across_splits() -> None:
    key = lambda r: (r["policy"], r["evidence"], r["claim"])  # noqa: E731
    train_keys = {key(r) for r in TRAIN}
    assert not train_keys & {key(r) for r in TEST}
    assert len(train_keys) == len(TRAIN)


# --------------------------------------------------------------------------- balance


@pytest.mark.parametrize("rows", [TRAIN, TEST], ids=["train", "test"])
def test_every_label_is_well_represented(rows: list[dict[str, Any]]) -> None:
    cnt = Counter(r["label"] for r in rows)
    assert set(cnt) == set(mmc.LABELS)
    for lb in mmc.LABELS:
        assert cnt[lb] / len(rows) >= 0.08, (lb, cnt)
    imp = Counter(r["label"] for r in rows if r["claim_kind"] == "improvement")
    assert set(imp) == set(mmc.LABELS)


def test_variety() -> None:
    assert {r["category"] for r in TRAIN} >= set(mmc.mts.CATEGORIES)
    assert {r["perturbation"] for r in TRAIN} == set(mmc.PERTURBATIONS)
    assert {r["claim_kind"] for r in TRAIN} == set(mmc.CLAIM_KINDS)
    assert len({r["target"] for r in TRAIN}) == 11
    assert len({r["target"] for r in TEST}) == 14
    assert len({r["mechanism"] for r in TRAIN}) == 7
    for fam_kind, key in (("policy", "policy_family"), ("evidence", "evidence_family"), ("claim", "claim_family")):
        assert {r["templates"][key] for r in TRAIN} == set(mmc.FAMILIES[fam_kind]["train"])
        assert {r["templates"][key] for r in TEST} == set(mmc.FAMILIES[fam_kind]["test"])


# --------------------------------------------------------------------------- hold-out design


def test_template_families_are_held_out() -> None:
    for fam_kind in mmc.FAMILIES:
        assert not set(mmc.FAMILIES[fam_kind]["train"]) & set(mmc.FAMILIES[fam_kind]["test"])
    for r in TEST:
        t = r["templates"]
        assert t["policy_family"] in mmc.FAMILIES["policy"]["test"]
        assert t["evidence_family"] in mmc.FAMILIES["evidence"]["test"]
        assert t["claim_family"] in mmc.FAMILIES["claim"]["test"]


def test_claim_sentences_never_shared_between_splits() -> None:
    def skeleton(r: dict[str, Any]) -> str:
        return re.sub(r"[-\d.]+|\S+/\d+|\S*\d\S*", "#", r["claim"])
    assert not {skeleton(r) for r in TRAIN} & {skeleton(r) for r in TEST}


def test_unseen_targets_only_in_test() -> None:
    unseen = {k for k, g in mmc.GENERIC.items() if not g.seen}
    assert len(unseen) == 3
    assert not unseen & {r["target"] for r in TRAIN}
    share = sum(r["target"] in unseen for r in TEST) / len(TEST)
    assert 0.15 <= share <= 0.3
    assert all(r["target_seen_in_train"] == (r["target"] not in unseen) for r in TEST)


def test_groups_stay_in_one_split_and_fold() -> None:
    by_group: dict[str, set[tuple[str, Any]]] = defaultdict(set)
    for r in ALL:
        by_group[r["group"]].add((r["split"], r["fold"]))
    assert all(len(v) == 1 for v in by_group.values())
    folds = Counter(r["fold"] for r in TRAIN)
    assert set(folds) == {"head", "temperature", "conformal"}
    assert folds["head"] > folds["temperature"] > 0 and folds["conformal"] > 0
    assert all(r["fold"] is None for r in TEST + TRAP50)


def test_paired_rows_share_evidence_but_not_claim_kind() -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in TRAIN + TEST:
        groups[r["group"]].append(r)
    pairs = [g for g in groups.values() if len(g) == 2]
    assert len(pairs) > 300
    for a, b in pairs:
        assert (a["policy"], a["evidence"]) == (b["policy"], b["evidence"])
        assert a["claim_kind"] != b["claim_kind"]
    assert all(len(g) <= 2 for g in groups.values())


def test_no_trap_set_case_is_reused() -> None:
    def key(target: str, prev: float, new: float, seeds: dict[str, float] | None) -> str:
        return json.dumps([target, prev, new, seeds], sort_keys=True)
    trap = {key(c["target"], c["claim"]["prev_best_score"], c["claim"]["new_score"], c["claim"]["evidence"]["seed_scores"])
            for c in TRAP_CASES.values()}
    for r in TRAIN + TEST:
        if r["source"] == "trap_logic" and r["perturbation"] == "none":
            h = r["hidden"]
            assert key(r["target"], h["prev_score"], h["new_score"], h["seed_scores_shown"]) not in trap, r["id"]


# --------------------------------------------------------------------------- labels vs planted facts


def test_label_function_table() -> None:
    L = mmc.sapling_label
    assert L("improvement", "clean", "same", "established", None) == mmc.SUPPORTED
    assert L("improvement", "leak", "same", "n/a", None) == mmc.VIOLATION
    assert L("improvement", "leak", "same", "missing", None) == mmc.VIOLATION
    assert L("improvement", "clean", "same", "within_noise", None) == mmc.INSUFFICIENT
    assert L("improvement", "clean", "changed", "n/a", None) == mmc.INSUFFICIENT
    assert L("improvement", "missing", "same", "established", None) == mmc.INSUFFICIENT
    assert L("improvement", "clean", "same", "established", "score") == mmc.INCONSISTENT
    assert L("heldout", "leak", "same", "n/a", None) == mmc.VIOLATION
    assert L("heldout", "clean", "changed", "within_noise", None) == mmc.SUPPORTED
    assert L("leak", "leak", "same", "n/a", None) == mmc.SUPPORTED
    assert L("leak", "clean", "same", "established", None) == mmc.VIOLATION
    assert L("leak", "missing", "same", "n/a", None) == mmc.INSUFFICIENT
    assert L("heldout", "clean", "same", "n/a", "split") == mmc.INCONSISTENT
    with pytest.raises(ValueError):
        L("heldout", "clean", "same", "n/a", "score")


def test_row_labels_follow_from_facts() -> None:
    for r in ALL:
        f = r["facts"]
        assert r["label"] == mmc.sapling_label(r["claim_kind"], f["separation"], f["comparability"], f["gain"],
                                               f["conflict"]), r["id"]


def test_category_to_label_mapping_without_perturbation() -> None:
    expect = {"contamination": mmc.VIOLATION, "eval_changed": mmc.INSUFFICIENT, "noise": mmc.INSUFFICIENT,
              "genuine": mmc.SUPPORTED}
    for r in ALL:
        if r["claim_kind"] == "improvement" and r["perturbation"] == "none":
            assert r["label"] == expect[r["family"]], r["id"]
            assert r["label_alt"] == ([mmc.VIOLATION] if r["family"] in ("noise", "eval_changed") else [])


def test_fake_improvement_flag() -> None:
    for r in ALL:
        assert r["fake_improvement"] == (r["family"] != "genuine")
        if r["claim_kind"] == "improvement" and r["label"] == mmc.SUPPORTED:
            assert not r["fake_improvement"], r["id"]


def test_visible_numbers_match_the_category() -> None:
    """The policy rule (gain > 2x previous-best seed std; seed mean for several seeds) must decide
    exactly as the planted category says, using only numbers shown in the evidence."""
    for r in ALL:
        h = r["hidden"]
        ratio = h["visible_gain_ratio"]
        if r["family"] == "genuine":
            assert ratio > mmc.GAIN_RULE_X, r["id"]
        elif r["category"] == "noise_within_spread":
            assert ratio < mmc.GAIN_RULE_X, r["id"]
        elif r["category"] == "lucky_seed":
            assert ratio < 0, r["id"]
        assert h["new_score"] != h["prev_score"]


def test_claims_are_new_bests_and_numbers_are_visible() -> None:
    for r in ALL:
        h = r["hidden"]
        direction = get_dir(r)
        assert (h["new_score"] > h["prev_score"]) == (direction == "max"), r["id"]
        assert json.dumps(h["new_score"]) in r["evidence"], r["id"]
        assert json.dumps(h["prev_score"]) in r["evidence"], r["id"]
        if r["claim_kind"] == "improvement" and "{new}" in _claim_template(r):
            assert json.dumps(h["new_score"]) in r["claim"], r["id"]


def get_dir(r: dict[str, Any]) -> str:
    if r["target"] in mmc.GENERIC:
        return mmc.GENERIC[r["target"]].direction
    return mmc.get_target(r["target"]).spec.direction


def _claim_template(r: dict[str, Any]) -> str:
    t = r["templates"]
    return mmc.CLAIMS[t["claim_family"]][r["claim_kind"]][t["claim_template"]]


def test_claim_mentions_the_run_named_in_the_evidence() -> None:
    for r in ALL:
        if "{run}" in _claim_template(r):
            if r["templates"]["evidence_family"] == "prose":
                run = re.match(r"Run (\S+) is compared", r["evidence"])
            else:
                ev = re.sub("new run", "new_run", r["evidence"], flags=re.I)
                run = re.search(r"new_run[ =:|\"]+([^\s\"|,]+)", ev)
            assert run is not None, r["id"]
            assert run.group(1) in r["claim"], r["id"]


# --------------------------------------------------------------------------- perturbations


def test_anti_leak_same_keys_within_target_profile() -> None:
    keysets: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
    for r in ALL:
        if r["perturbation"] == "none":
            keysets[(r["target"], r["evidence_profile"])].add(tuple(r["evidence_keys"]))
    assert all(len(v) == 1 for v in keysets.values()), {k: v for k, v in keysets.items() if len(v) > 1}


def _sep_keys(r: dict[str, Any]) -> tuple[str, ...]:
    return mmc.CORE_INFO[r["target"]]["sep"] if r["target"] in mmc.CORE_INFO else mmc.GENERIC_SEP


def test_missing_separation_hides_every_separation_signal() -> None:
    rows = [r for r in ALL if r["perturbation"].startswith("missing_separation")]
    assert rows
    for r in rows:
        assert not set(_sep_keys(r)) & set(r["evidence_keys"]), r["id"]
        assert not r["config_shown"], r["id"]
        if r["perturbation"] == "missing_separation_marked":
            assert {"split_manifest", "overlap_check", "data_provenance", "selection_log"} & set(r["evidence_keys"])
        assert r["label"] == mmc.INSUFFICIENT, r["id"]


def test_missing_seeds_and_eval_info_drop_their_keys() -> None:
    for r in ALL:
        if r["perturbation"] == "missing_seeds":
            assert not set(mmc.NOISE_KEYS) & set(r["evidence_keys"]), r["id"]
        if r["perturbation"] == "missing_eval_info":
            cmp = mmc.CORE_INFO[r["target"]]["cmp"] if r["target"] in mmc.CORE_INFO else mmc.GENERIC_CMP
            assert not set(cmp) & set(r["evidence_keys"]), r["id"]


def test_conflicts_really_contradict() -> None:
    n = Counter()
    for r in ALL:
        h, p = r["hidden"], r["perturbation"]
        if p == "conflict_score":
            assert h["score_in_eval_log"] != h["new_score"]
            sign = 1 if get_dir(r) == "max" else -1
            assert sign * (h["score_in_eval_log"] - h["prev_score"]) < 0, r["id"]   # log says: not even a new best
        elif p == "conflict_seed_mean":
            assert r["aggregation"] == "mean"
            mean = statistics.fmean(h["seed_scores_shown"].values())
            sign = 1 if get_dir(r) == "max" else -1
            assert sign * (mean - h["prev_score"]) < 0, r["id"]
        elif p == "conflict_split":
            assert r["evidence_profile"] == "full"
        else:
            assert h["score_in_eval_log"] in (None, h["new_score"]), r["id"]
            continue
        n[p] += 1
        assert r["label"] == mmc.INCONSISTENT
    assert min(n.values()) >= 10


def test_split_conflicts_overwrite_the_explicit_field() -> None:
    rng = random.Random(0)
    for target in ("cifar", "lm", "rl", "wmt_bleu", "gsm8k_sft", "imagenet_vit", "procgen_ppo"):
        for base in ("contamination", "clean_gain"):
            s = mmc._scenario(target, base, base, "conflict_split", rng, 900_000 + rng.randrange(1000))
            ev, facts = mmc.apply_perturbation(s, "conflict_split", rng)
            changed = {k for k in ev if ev[k] != s.evidence.get(k)}
            assert changed, (target, base)
            assert facts["conflict"] == "split"


def test_conflicting_scores_only_on_clean_bases() -> None:
    for r in ALL:
        if r["perturbation"] in ("conflict_score", "conflict_seed_mean"):
            assert r["facts"]["separation"] == "clean"
            assert r["claim_kind"] == "improvement"


# --------------------------------------------------------------------------- trap50


def test_trap50_mirrors_trap_set_v1() -> None:
    assert {r["trap_id"] for r in TRAP50} == set(TRAP_CASES)
    expect = {"contamination": mmc.VIOLATION, "eval_changed": mmc.INSUFFICIENT, "noise": mmc.INSUFFICIENT,
              "genuine": mmc.SUPPORTED}
    for r in TRAP50:
        c = TRAP_CASES[r["trap_id"]]
        assert r["trap_split"] == c["split"]
        assert r["category"] == c["category"] and r["target"] == c["target"]
        assert r["label"] == expect[c["family"]]
        assert (c["label"] == "valid") == (r["label"] == mmc.SUPPORTED)
        assert r["hidden"]["new_score"] == c["claim"]["new_score"]
        assert r["templates"]["policy_family"] == "numbered" and r["templates"]["evidence_family"] == "kv"
    assert Counter(r["label"] for r in TRAP50) == {mmc.SUPPORTED: 25, mmc.VIOLATION: 15, mmc.INSUFFICIENT: 10}
