"""Dev server: the FastAPI app with auto-reload on http://127.0.0.1:8000.

    uv run --extra ml python scripts/dev_server.py [--port 8000] [--db data/busy_machines.db] [--no-reload]

Races default to the mock LLM + local CPU runner ($0). Real mode (llm="mubit", runner="modal")
needs MUBIT_API_KEY in .env and a deployed Modal app (see README "Running it").
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db", default=os.environ.get("BUSY_MACHINES_DB", "data/busy_machines.db"))
    ap.add_argument("--no-reload", action="store_true")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.is_absolute():
        db = ROOT / db
    db.parent.mkdir(parents=True, exist_ok=True)
    os.environ["BUSY_MACHINES_DB"] = str(db)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    import uvicorn

    print(f"Busy Machines dev server: http://{args.host}:{args.port}  (DB {db})")
    uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=not args.no_reload,
                reload_dirs=[str(ROOT / "backend"), str(ROOT / "workers")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
