"""Fetch and cache the LM corpus (tiny-shakespeare) and print the split fingerprints.

Usage (repo root):
    python -m uv run python -m workers.targets.data.prep_lm            # cache under workers/targets/data/cache/
    python -m uv run python -m workers.targets.data.prep_lm --out /data/tinyshakespeare.txt

For Modal, upload the printed file to a Volume and set LM_DATA_PATH in the container
(e.g. ``modal volume put <vol> workers/targets/data/cache/tinyshakespeare.txt /tinyshakespeare.txt``
then mount the volume at /data and set LM_DATA_PATH=/data/tinyshakespeare.txt). The runner
owns that wiring; this script only prepares the file. It needs no torch.
"""

from __future__ import annotations

import argparse
import json

from workers.targets.lm import corpus_path, ensure_corpus, make_splits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="corpus file path (default: LM_DATA_PATH / repo cache)")
    args = ap.parse_args()
    path = ensure_corpus(args.out)
    raw = path.read_bytes()
    info = {"path": str(path), "default_path": str(corpus_path()), "bytes": len(raw)}
    for trap in (False, True):
        s = make_splits(raw, trap=trap)
        info["trap" if trap else "clean"] = {
            "corpus_sha256": s.corpus_sha256,
            "vocab_size": s.vocab_size,
            "train_tokens": len(s.train),
            "val_tokens": len(s.val),
            "test_tokens": len(s.test),
            "split_sha256": s.split_sha256,
            "train_val_overlap": s.train_val_overlap,
        }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
