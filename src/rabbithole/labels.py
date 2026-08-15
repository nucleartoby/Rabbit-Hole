from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from rabbithole import config


@dataclass(frozen=True)
class Judgement:

    parent: str
    picks: list[str]
    rating: float                       # 0.0 dull or wrong, 1.0 exactly what I wanted
    stage: str = ""
    backend: str = "tfidf"
    note: str = ""
    scored: dict[str, float] = field(default_factory=dict)


def record(judgement: Judgement, path: Path | None = None) -> None:
    target = path or config.labels_path()
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(judgement), sort_keys=True) + "\n")


def load(path: Path | None = None) -> list[Judgement]:
    target = path or config.labels_path()
    if not target.exists():
        return []

    found = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            found.append(Judgement(**json.loads(line)))

    return found


def for_stage(judgements: list[Judgement], stage: str) -> list[Judgement]:
    return [judgement for judgement in judgements if judgement.stage == stage]
