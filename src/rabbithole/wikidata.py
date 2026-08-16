from __future__ import annotations
from dataclasses import dataclass, field
from rabbithole import config, sources

# The relations worth carryin. Wikidata knows thousands these are the ones that
# describe how one article relates to another in a way a user would recognise.
FORWARD = {
    "P31": "instance of",
    "P279": "subclass of",
    "P361": "part of",
    "P527": "has part",
    "P101": "field of",
    "P106": "occupation",
    "P135": "movement",
    "P136": "genre",
    "P737": "influenced by",
    "P155": "follows",
    "P156": "followed by",
    "P17": "country",
    "P495": "from",
    "P921": "about",
    "P1269": "facet of",}

BACKWARD = {
    "P31": "example of",
    "P279": "a kind of",
    "P361": "has part",
    "P527": "part of",
    "P737": "influenced",
    "P155": "followed by",
    "P156": "follows",
    "P921": "covers",
    "P1269": "aspect of",}

SHARED = {
    "P31": "also a",
    "P279": "sibling of",
    "P136": "same genre",
    "P101": "same field",
    "P106": "same work",
    "P135": "same movement",
    "P495": "same origin",
    "P17": "same country",
    "P921": "same subject",}

FAMILIES = {
    "P31": "type", "P279": "type",
    "P361": "part", "P527": "part",
    "P101": "field", "P106": "field", "P135": "field", "P136": "field",
    "P737": "influence",
    "P155": "time", "P156": "time",
    "P17": "place", "P495": "place",
    "P921": "subject", "P1269": "subject",}

UNKNOWN = "unknown"


@dataclass(frozen=True)
class Edge:

    key: str = ""
    label: str = ""
    family: str = UNKNOWN


@dataclass(frozen=True)
class Graph:

    edges: dict[str, Edge] = field(default_factory=dict)
    qids: dict[str, str] = field(default_factory=dict)
    triples: list[tuple[str, str, str]] = field(default_factory=list)
    parent_qid: str = ""

    def edge(self, title: str) -> Edge:
        return self.edges.get(title, Edge())


def _values(statements: list[dict]) -> set[str]:
    found: set[str] = set()
    for statement in statements:
        value = ((statement.get("mainsnak") or {}).get("datavalue") or {}).get("value")
        if isinstance(value, dict) and value.get("id"):
            found.add(value["id"])

    return found


def parse(raw: dict[str, dict]) -> dict[str, dict[str, set[str]]]:
    parsed: dict[str, dict[str, set[str]]] = {}
    for qid, claims in raw.items():
        kept = {
            pid: _values(statements)
            for pid, statements in claims.items() if pid in FAMILIES}
        parsed[qid] = {pid: values for pid, values in kept.items() if values}

    return parsed


def edge_between(
    parent_claims: dict[str, set[str]],
    candidate_claims: dict[str, set[str]],
    parent_qid: str,
    candidate_qid: str,) -> Edge:
    for pid, label in FORWARD.items():
        if candidate_qid in parent_claims.get(pid, set()):
            return Edge(pid, label, FAMILIES[pid])

    for pid, label in BACKWARD.items():
        if parent_qid in candidate_claims.get(pid, set()):
            return Edge(f"~{pid}", label, FAMILIES[pid])

    for pid, label in SHARED.items():
        if parent_claims.get(pid, set()) & candidate_claims.get(pid, set()):
            return Edge(f"={pid}", label, FAMILIES[pid])

    return Edge()


def triples_from(claims: dict[str, dict[str, set[str]]], cap: int) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for qid, properties in sorted(claims.items()):
        for pid, values in sorted(properties.items()):
            for value in sorted(values):
                found.append((qid, pid, value))
                if len(found) >= cap:
                    return found

    return found


def build(parent: str, titles: list[str]) -> Graph:
    mapping = sources.qids([parent] + titles)
    parent_qid = mapping.get(parent, "")
    if not parent_qid:
        return Graph(qids=mapping)

    claims = parse(sources.entities(sorted(set(mapping.values()))))
    parent_claims = claims.get(parent_qid, {})

    edges = {}
    for title in titles:
        candidate_qid = mapping.get(title)
        if candidate_qid and candidate_qid != parent_qid:
            edges[title] = edge_between(
                parent_claims, claims.get(candidate_qid, {}), parent_qid, candidate_qid)

    return Graph(
        edges=edges,
        qids=mapping,
        triples=triples_from(claims, config.KGE_MAX_TRIPLES),
        parent_qid=parent_qid,)
