from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from rabbithole import __version__, relate
from rabbithole.core import dig

WEB = Path(__file__).parent / "web"

app = FastAPI(
    title="Rabbit Hole",
    version=__version__,
    description="Enter a word, fall down a Wikipedia rabbit hole.",)


@app.get("/api/dig", summary="Look up a word and everywhere it leads")
def api_dig(
    term: str,
    limit: int = Query(default=10, ge=1, le=50),
    stage: str | None = Query(
        default=None, description=f"Selection strategy: {' | '.join(relate.STAGES)}"),
    path: str = Query(
        default="", description="Titles already visited, '|' separated; steers novelty"),
    exclude: str = Query(
        default="", description="Titles that must not be offered again, '|' separated"),
    taste: float | None = Query(
        default=None, ge=0.0, le=1.0,
        description="Mean relevance quantile of what has been clicked; aims the band"),) -> dict:
    if stage is not None and stage not in relate.STAGES:
        raise HTTPException(status_code=422, detail=f"stage must be one of {relate.STAGES}")

    visited = [title for title in path.split("|") if title.strip()]
    blocked = [title for title in exclude.split("|") if title.strip()]
    hole = dig(term, limit=limit, stage=stage, path=visited, exclude=blocked, taste=taste)
    if hole.top is None:
        raise HTTPException(status_code=404, detail=f"Nothing found for {term!r}")

    return {
        "term": hole.term,
        "stage": hole.stage,
        "top": asdict(hole.top),
        "more": [asdict(entry) for entry in hole.more],
        "scores": hole.scores,}


@app.get("/health", summary="Liveness probe")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")
