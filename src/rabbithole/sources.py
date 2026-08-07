from __future__ import annotations
import json
import sqlite3
import time
import requests
from rabbithole import config

_session = requests.Session()
_session.headers["User-Agent"] = config.USER_AGENT


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
