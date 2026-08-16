from __future__ import annotations
import os
from pathlib import Path

CONTACT = os.getenv("RABBITHOLE_CONTACT", "email.address@gmail.com")
USER_AGENT = f"RabbitHole/0.1 ({CONTACT})"

WIKIPEDIA_API = os.getenv("RABBITHOLE_WIKIPEDIA_API", "https://en.wikipedia.org/w/api.php")
WIKIDATA_API = os.getenv("RABBITHOLE_WIKIDATA_API", "https://www.wikidata.org/w/api.php")

CACHE_TTL = int(os.getenv("RABBITHOLE_CACHE_TTL", 60 * 60 * 24 * 7))
REQUEST_TIMEOUT = float(os.getenv("RABBITHOLE_TIMEOUT", 10))

STAGE = os.getenv("RABBITHOLE_STAGE", "ranked")

VECTOR_BACKEND = os.getenv("RABBITHOLE_VECTORS", "tfidf")

KGE_DIM = int(os.getenv("RABBITHOLE_KGE_DIM", 32))
KGE_EPOCHS = int(os.getenv("RABBITHOLE_KGE_EPOCHS", 120))
KGE_MAX_TRIPLES = int(os.getenv("RABBITHOLE_KGE_TRIPLES", 6000))


def cache_path() -> Path:
    override = os.getenv("RABBITHOLE_CACHE_DIR")
    base = Path(override) if override else _default_cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / "wikipedia.sqlite"


def labels_path() -> Path:
    override = os.getenv("RABBITHOLE_LABELS")
    if override:
        path = Path(override)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    base = _default_cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / "judgements.jsonl"


def _default_cache_dir() -> Path:
    if local_app_data := os.getenv("LOCALAPPDATA"):        # Windows
        return Path(local_app_data) / "rabbithole" / "cache"
    if xdg_cache := os.getenv("XDG_CACHE_HOME"):           # Linux
        return Path(xdg_cache) / "rabbithole"
    return Path.home() / ".cache" / "rabbithole"
