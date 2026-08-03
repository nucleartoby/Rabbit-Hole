from __future__ import annotations
import os
from pathlib import Path


CONTACT = os.getenv("RABBITHOLE_CONTACT", "emailaddress@gmail.com")
USER_AGENT = f"RabbitHole/0.1 ({CONTACT})"

WIKIPEDIA_API = os.getenv("RABBITHOLE_WIKIPEDIA_API", "https://en.wikipedia.org/w/api.php")

CACHE_TTL = int(os.getenv("RABBITHOLE_CACHE_TTL", 60 * 60 * 24 * 7))
REQUEST_TIMEOUT = float(os.getenv("RABBITHOLE_TIMEOUT", 10))


def cache_path() -> Path:
    override = os.getenv("RABBITHOLE_CACHE_DIR")
    base = Path(override) if override else _default_cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / "wikipedia.sqlite"


def _default_cache_dir() -> Path:
    if local_app_data := os.getenv("LOCALAPPDATA"):        # Windows
        return Path(local_app_data) / "rabbithole" / "cache"
    if xdg_cache := os.getenv("XDG_CACHE_HOME"):           # Linux
        return Path(xdg_cache) / "rabbithole"
    return Path.home() / ".cache" / "rabbithole"
