from __future__ import annotations
import re
from dataclasses import dataclass
from rabbithole import sources

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


def dig(term: str, limit: int = 20) -> Hole:
    term = term.strip()
    if not term:
        return Hole(term=term, top=None, more=[])

    hits = sources.search(term, limit=limit + 1)
    if not hits:
        return Hole(term=term, top=None, more=[])

    detail = sources.pages([hit["title"] for hit in hits])
    entries = [_entry(hit, detail.get(hit["title"])) for hit in hits]

    return Hole(term=term, top=entries[0], more=entries[1:])


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
