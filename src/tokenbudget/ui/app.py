"""FastAPI app for the TokenBudget console.

Serves the sweep analysis JSON (budget table, per-family decay curves and
cliff locations) and the probe corpus manifest, plus a static dashboard.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tokenbudget import __version__
from tokenbudget.decay import SweepResult
from tokenbudget.ui import repo_root

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _read_json(root: Path, rel: str) -> Any:
    path = root / rel
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{rel} missing")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def create_app() -> FastAPI:
    app = FastAPI(title="TokenBudget", version=__version__)
    root = repo_root()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "service": "tokenbudget"}

    @app.get("/api/tokenbudget/sweep.json")
    def sweep() -> Any:
        """Raw sweep table: budget axis + per-(family,difficulty) accuracy."""
        return _read_json(root, "data/sweep/sweep.json")

    @app.get("/api/tokenbudget/decay.json")
    def decay() -> dict[str, Any]:
        """Decay analysis: per-family cliffs + sensitivity ranking."""
        sweep_path = root / "data" / "sweep" / "sweep.json"
        if not sweep_path.is_file():
            raise HTTPException(status_code=404, detail="sweep results missing — run scripts/run_sweep.py")
        result = SweepResult.load(sweep_path)
        budget_table = json.loads(sweep_path.read_text(encoding="utf-8"))["budget_table"]
        return {
            "budget_table": budget_table,
            "summary": result.summary(),
            "families": sorted(result.curves),
            "curves": {
                fam: {
                    str(diff): {
                        "budgets": c.budgets,
                        "budget_axis": c.budget_axis,
                        "acc": c.acc,
                        "ci_lo": c.ci_lo,
                        "ci_hi": c.ci_hi,
                        "cliff": c.cliff(),
                    }
                    for diff, c in diffs.items()
                }
                for fam, diffs in result.curves.items()
            },
        }

    @app.get("/api/tokenbudget/manifest.json")
    def manifest() -> Any:
        """Probe corpus manifest (id, family, difficulty, answer, question)."""
        return _read_json(root, "data/probes/manifest.json")

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/tokenbudget")
    def index() -> FileResponse:
        index_path = STATIC_DIR / "tokenbudget.html"
        if not index_path.is_file():
            raise HTTPException(status_code=500, detail="TokenBudget UI static file missing")
        return FileResponse(index_path)

    @app.get("/")
    def root_index() -> FileResponse:
        index_path = STATIC_DIR / "tokenbudget.html"
        if not index_path.is_file():
            raise HTTPException(status_code=500, detail="UI static files missing")
        return FileResponse(index_path)

    return app
