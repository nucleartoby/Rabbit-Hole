from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from rabbithole import __version__
from rabbithole.core import dig

WEB = Path(__file__).parent / "web"

app = FastAPI(
    title="Rabbit Hole",
    version=__version__,
    description="Enter a word, fall down a Wikipedia rabbit hole.",)


@app.get("/api/dig", summary="Look up a word and everywhere it leads")
def api_dig(term: str, limit: int = Query(default=10, ge=1, le=50)) -> dict:
    hole = dig(term, limit=limit)
    if hole.top is None:
        raise HTTPException(status_code=404, detail=f"Nothing found for {term!r}")

    return {"term": hole.term, "top": asdict(hole.top), "more": [asdict(entry) for entry in hole.more],}


@app.get("/health", summary="Liveness probe")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")
