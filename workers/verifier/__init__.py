"""Verifier glue for Banyan Sapling (``MubitLabs/banyan-sapling``), D7 / D28.

Pure Python: nothing here imports ``modal`` (only ``workers/runners/modal_*.py`` may, D28)
and nothing imports torch at module import time.

- ``workers.verifier.sapling``: pins, row validation, chunking, the stock LAC calibration
  re-implemented in numpy (offline eval of a new head on cached features), parity checks and
  cost estimates.
- ``workers.verifier.sapling_runtime``: container-side torch helpers (feature hook, batch
  extraction, head parity, head dump). Imports torch lazily, inside functions.

The Modal app that runs these lives in ``workers/runners/modal_sapling.py``.
"""
