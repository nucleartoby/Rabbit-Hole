from __future__ import annotations
import json
import re
import sqlite3
import time
import requests
from rabbithole import config

_session = requests.Session()
_session.headers["User-Agent"] = config.USER_AGENT

_WIKILINK = re.compile(r"\[\[([^\]|#<>]+)")


def _cache() -> sqlite3.Connection:
    conn = sqlite3.connect(config.cache_path())
    conn.execute(
        "CREATE TABLE IF NOT EXISTS responses "
        "(key TEXT PRIMARY KEY, body TEXT, fetched_at REAL)")
    return conn


def _get_with_retry(params: dict, attempts: int = 3) -> dict:
    for attempt in range(attempts):
        response = _session.get(
            config.WIKIPEDIA_API, params=params, timeout=config.REQUEST_TIMEOUT)
        if response.status_code == 429 and attempt < attempts - 1:
            time.sleep(float(response.headers.get("Retry-After", 2**attempt)))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("unreachable")


def _query(**params) -> dict:
    params = {"action": "query", "format": "json", "formatversion": 2, **params}
    key = json.dumps(params, sort_keys=True)

    with _cache() as conn:
        row = conn.execute(
            "SELECT body, fetched_at FROM responses WHERE key = ?", (key,)).fetchone()
        if row and time.time() - row[1] < config.CACHE_TTL:
            return json.loads(row[0])

        body = _get_with_retry(params)
        conn.execute(
            "INSERT OR REPLACE INTO responses VALUES (?, ?, ?)",
            (key, json.dumps(body), time.time()),)
    return body


def search(term: str, limit: int = 20) -> list[dict]:
    body = _query(list="search", srsearch=term, srlimit=limit, srnamespace=0)
    return body.get("query", {}).get("search", [])


def morelike(title: str, limit: int = 20) -> list[dict]:
    body = _query(
        list="search", srsearch=f"morelike:{title}", srlimit=limit, srnamespace=0)
    return body.get("query", {}).get("search", [])


def lead_links(title: str) -> list[str]:
    body = _query(
        titles=title, prop="revisions", rvprop="content", rvslots="main", rvsection=0,
        redirects=1,)
    pages = body.get("query", {}).get("pages", [])
    if not pages or "missing" in pages[0]:
        return []

    revisions = pages[0].get("revisions") or []
    if not revisions:
        return []

    wikitext = revisions[0].get("slots", {}).get("main", {}).get("content", "")
    found: list[str] = []
    for target in _WIKILINK.findall(wikitext):
        target = target.strip()
        # Transportation is a categor/interwiki sigil not an article link
        if target and not target.startswith(":") and ":" not in target:
            cleaned = target[0].upper() + target[1:]
            if cleaned not in found:
                found.append(cleaned)

    return found


def all_links(title: str, pages_deep: int = 2) -> set[str]:
    found: set[str] = set()
    continuation: dict = {}

    for _ in range(pages_deep):
        body = _query(
            titles=title, prop="links", plnamespace=0, pllimit=500, redirects=1,
            **continuation,)
        for page in body.get("query", {}).get("pages", []):
            found.update(link["title"] for link in page.get("links", []))

        if "continue" not in body:
            break
        continuation = body["continue"]

    return found


def categories(titles: list[str]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}

    for batch in (titles[i : i + 20] for i in range(0, len(titles), 20)):
        body = _query(
            titles="|".join(batch), prop="categories", cllimit=500, clshow="!hidden",
            redirects=1,)
        for page in body.get("query", {}).get("pages", []):
            if "missing" not in page:
                found[page["title"]] = {
                    category["title"] for category in page.get("categories", [])}

    return found


def pages(titles: list[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}

    for batch in (titles[i : i + 20] for i in range(0, len(titles), 20)):
        body = _query(
            titles="|".join(batch),
            prop="extracts|pageimages|info",
            exintro=1,
            explaintext=1,
            exlimit=20,
            piprop="thumbnail",
            pithumbsize=320,
            inprop="url",
            redirects=1,)
        for page in body.get("query", {}).get("pages", []):
            if "missing" not in page:
                found[page["title"]] = page

    return found
