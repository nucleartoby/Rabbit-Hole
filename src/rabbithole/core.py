from __future__ import annotations
import re
from dataclasses import dataclass, field
from rabbithole import config, relate, sources

_MARKUP = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Entry:

    title: str
    summary: str
    url: str
    thumbnail: str | None = None


@dataclass(frozen=True)
class Hole:

    term: str
    top: Entry | None
    more: list[Entry]
    stage: str = relate.SEARCH
    scores: dict[str, dict[str, float]] = field(default_factory=dict)


def dig(
    term: str,
    limit: int = 20,
    stage: str | None = None,
    path: list[str] | None = None,
    exclude: list[str] | None = None,) -> Hole:

    stage = stage or config.STAGE
    if stage not in relate.STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {relate.STAGES}")

    term = term.strip()
    if not term:
        return Hole(term=term, top=None, more=[], stage=stage)

    hits = sources.search(term, limit=limit + 1)
    if not hits:
        return Hole(term=term, top=None, more=[], stage=stage)

    blocked = {relate.normalise_title(title) for title in (exclude or [])}
    top_title = hits[0]["title"]

    if stage == relate.SEARCH:
        detail = sources.pages([hit["title"] for hit in hits])
        entries = [_entry(hit, detail.get(hit["title"])) for hit in hits]
        return Hole(
            term=term,
            top=entries[0],
            more=[
                entry for entry in entries[1:]
                if relate.normalise_title(entry.title) not in blocked],
            stage=stage,)

    pool = relate.candidates(
        top_title,
        pool=max(30, limit * 3 + min(len(blocked), 60)),
        exclude=blocked,)
    detail = sources.pages([top_title] + [c.title for c in pool])
    top = _entry(hits[0], detail.get(top_title))

    if stage == relate.CANDIDATES:
        picked = pool[:limit]
    else:
        picked = _ranked(top, pool, detail, limit, path or [])

    more = [
        _entry({"title": c.title}, detail.get(c.title)) for c in picked if c.title in detail]

    return Hole(
        term=term, top=top, more=more, stage=stage,
        scores={c.title: c.parts for c in picked if c.parts},)


def _ranked(top: Entry, pool, detail: dict, limit: int, path: list[str]):
    texts = {
        title: (page.get("extract") or "").strip()
        for title, page in detail.items()}

    titles = [c.title for c in pool]
    category_map = sources.categories([top.title] + titles)
    path_detail = sources.pages(path) if path else {}

    return relate.rank(
        parent_text=top.summary or top.title,
        pool=pool,
        texts=texts,
        limit=limit,
        path_texts=[
            (page.get("extract") or "").strip() for page in path_detail.values()],
        parent_categories=category_map.get(top.title, set()),
        candidate_categories=category_map,
        backend=config.VECTOR_BACKEND,)


def _entry(hit: dict, page: dict | None) -> Entry:
    snippet = _MARKUP.sub("", hit.get("snippet", "")).strip()
    if page is None:
        return Entry(title=hit["title"], summary=snippet, url=_url_for(hit["title"]))

    return Entry(
        title=page["title"],
        summary=(page.get("extract") or snippet).strip(),
        url=page.get("fullurl") or _url_for(page["title"]),
        thumbnail=(page.get("thumbnail") or {}).get("source"),)


def _url_for(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
