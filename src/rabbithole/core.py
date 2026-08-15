from __future__ import annotations
import re
from dataclasses import dataclass, field
from rabbithole import config, kge, relate, sources, wikidata

_MARKUP = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Entry:

    title: str
    summary: str
    url: str
    thumbnail: str | None = None
    why: str = ""


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
    exclude: list[str] | None = None,
    taste: float | None = None,) -> Hole:

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
        picked, graph = pool[:limit], None
    else:
        picked, graph = _ranked(top, pool, detail, limit, path or [], stage, taste)

    more = [
        _entry({"title": c.title}, detail.get(c.title), why=_why(graph, c.title))
        for c in picked if c.title in detail]

    return Hole(
        term=term, top=top, more=more, stage=stage,
        scores={c.title: c.parts for c in picked if c.parts},)


def _ranked(
    top: Entry,
    pool: list[relate.Candidate],
    detail: dict,
    limit: int,
    path: list[str],
    stage: str,
    taste: float | None,):
    texts = {
        title: (page.get("extract") or "").strip()
        for title, page in detail.items()}

    titles = [c.title for c in pool]
    category_map = sources.categories([top.title] + titles)

    context = relate.Context(
        parent_text=top.summary or top.title,
        texts=texts,
        path_texts=_path_texts(path),
        parent_categories=category_map.get(top.title, set()),
        candidate_categories=category_map,
        taste=taste,
        backend=config.VECTOR_BACKEND,)

    graph = None
    if stage in (relate.TYPED, relate.KGE):
        graph = wikidata.build(top.title, titles)
        context.families = {
            title: edge.family for title, edge in graph.edges.items() if edge.key}

    if stage == relate.KGE and graph is not None:
        structure = kge.embed(graph)
        context.structure = {
            title: similarity
            for title in titles
            if (similarity := structure.similarity(title)) is not None}

    return relate.rank(pool, limit, context, profile=stage), graph


def _path_texts(path: list[str]) -> list[str]:
    if not path:
        return []

    detail = sources.pages(path)
    by_normal = {
        relate.normalise_title(title): page for title, page in detail.items()}

    ordered, claimed = [], set()
    for title in path:
        key = relate.normalise_title(title)
        page = by_normal.get(key)
        if page is not None and key not in claimed:
            claimed.add(key)
            ordered.append((page.get("extract") or "").strip())

    ordered.extend(
        (page.get("extract") or "").strip()
        for key, page in by_normal.items() if key not in claimed)

    return [text for text in ordered if text]


def _why(graph: wikidata.Graph | None, title: str) -> str:
    if graph is None:
        return ""

    return graph.edge(title).label


def _entry(hit: dict, page: dict | None, why: str = "") -> Entry:
    snippet = _MARKUP.sub("", hit.get("snippet", "")).strip()
    if page is None:
        return Entry(
            title=hit["title"], summary=snippet, url=_url_for(hit["title"]), why=why)

    return Entry(
        title=page["title"],
        summary=(page.get("extract") or snippet).strip(),
        url=page.get("fullurl") or _url_for(page["title"]),
        thumbnail=(page.get("thumbnail") or {}).get("source"),
        why=why,)


def _url_for(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
