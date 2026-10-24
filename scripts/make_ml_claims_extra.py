"""Extra ml-claims TRAIN rows for sapling-research v2 (head training only).

Reuses ``scripts/make_ml_claims.py`` unchanged (imported by path) with a different seed, so the rows are new
scenarios in the TRAIN template families and train targets only (test families qa/memo, log/sections,
casual/headline and the 3 test-only targets never appear, by the generator's own split logic). Rows whose
(policy, evidence, claim) text equals any existing train/test/trap50 row are dropped. Every row goes to the
head fold (``fold='head'``); the v1 temperature/conformal folds stay as they were, so calibration is
comparable with v1.

    python -m uv run python scripts/make_ml_claims_extra.py --n 3000
      -> data/sapling/sapling_v2/extra_train.jsonl

Pure and seeded; no LLM, Modal or GPU calls.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRA_SEED = 2026102402
OUT = ROOT / "data" / "sapling" / "sapling_v2" / "extra_train.jsonl"


def _load_gen():
    spec = importlib.util.spec_from_file_location("make_ml_claims_v1", ROOT / "scripts" / "make_ml_claims.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_ml_claims_v1"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    gen = _load_gen()
    gen.SEED = EXTRA_SEED  # module global read by build_split / _scenario / fold hashing
    seen = set()
    for f in ("train.jsonl", "test.jsonl", "trap50.jsonl"):
        for line in open(ROOT / "data" / "ml_claims" / f, encoding="utf-8"):
            r = json.loads(line)
            seen.add((r["policy"], r["evidence"], r["claim"]))
    rows = gen.build_split("train", args.n + 200)
    out, dropped = [], 0
    for r in rows:
        key = (r["policy"], r["evidence"], r["claim"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        r = {**r, "id": f"mlc_x_train_{len(out) + 1:05d}", "group": "x-" + r["group"], "fold": "head",
             "set_version": "ml_claims_extra_v2"}
        out.append(r)
        if len(out) >= args.n:
            break
    test_fams = {k: set(v["test"]) for k, v in gen.FAMILIES.items()}
    for r in out:
        t = r["templates"]
        assert t["policy_family"] not in test_fams["policy"] and t["evidence_family"] not in test_fams["evidence"] \
            and t["claim_family"] not in test_fams["claim"], "test family in extra rows"
        assert r["target_seen_in_train"], "test-only target in extra rows"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    print(json.dumps({"rows": len(out), "dropped_duplicates": dropped, "seed": EXTRA_SEED,
                      "labels": Counter(r["label"] for r in out), "out": str(args.out)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
