from __future__ import annotations
import argparse
from rabbithole import config, core, metrics, relate, sources

DEFAULT_TERMS = ("car", "jazz", "penicillin", "the great depression", "octopus")


def expansion_for(term: str, stage: str, limit: int, path: list[str] | None = None):
    hole = core.dig(term, limit=limit, stage=stage, path=path)
    if hole.top is None:
        return None, hole

    picks = [entry.title for entry in hole.more]
    category_map = sources.categories([hole.top.title] + picks)

    return (
        metrics.Expansion(
            parent=hole.top.title,
            parent_text=hole.top.summary,
            picks=picks,
            texts={entry.title: entry.summary for entry in hole.more},
            requested=limit,
            parent_categories=category_map.get(hole.top.title, set()),
            pick_categories=category_map,
            parent_links=sources.all_links(hole.top.title),),
        hole)


def walk(term: str, stage: str, limit: int, depth: int, breadth: int):
    root_hole = core.dig(term, limit=limit, stage=stage)
    if root_hole.top is None:
        return "", {}, []

    by_depth: dict[int, list[str]] = {}
    seen = [root_hole.top.title]
    frontier = [(entry, [root_hole.top.title]) for entry in root_hole.more[:breadth]]

    for level in range(1, depth + 1):
        next_frontier = []
        for entry, path in frontier:
            by_depth.setdefault(level, []).append(entry.summary)
            seen.append(entry.title)

            if level < depth:
                child = core.dig(entry.title, limit=limit, stage=stage, path=path)
                next_frontier.extend(
                    (kid, path + [entry.title]) for kid in child.more[:breadth])

        frontier = next_frontier

    return root_hole.top.summary, by_depth, seen


def run(terms, limit, depth, breadth, backend):
    results: dict[str, dict] = {}

    for stage in relate.STAGES:
        rows, samples = [], {}
        for term in terms:
            expansion, hole = expansion_for(term, stage, limit)
            if expansion is None:
                continue

            rows.append(metrics.score(expansion, backend=backend))
            samples[term] = [entry.title for entry in hole.more]

        summary = metrics.aggregate(rows)

        root_text, by_depth, seen = walk(terms[0], stage, limit, depth, breadth)
        summary["unique_rate"] = len(set(seen)) / len(seen) if seen else 0.0

        results[stage] = {
            "summary": summary,
            "drift": metrics.drift(root_text, by_depth, backend=backend),
            "samples": samples,}

    return results


def _table(results) -> str:
    names = list(metrics.HIGHER_IS_BETTER) + ["unique_rate", "quality"]
    width = max(len(name) for name in names) + 2
    header = "metric".ljust(width) + "".join(stage.rjust(13) for stage in relate.STAGES)
    lines = [header, "-" * len(header)]

    for name in names:
        row = name.ljust(width)
        for stage in relate.STAGES:
            row += f"{results[stage]['summary'].get(name, 0.0):13.3f}"

        if name in metrics.HIGHER_IS_BETTER:
            row += "  higher better" if metrics.HIGHER_IS_BETTER[name] else "  lower better"
        lines.append(row)

    lines.append("")
    lines.append("drift back to root, by depth")
    for stage in relate.STAGES:
        drift = results[stage]["drift"]
        rendered = "  ".join(f"d{depth}={value:.3f}" for depth, value in drift.items())
        lines.append(f"  {stage.ljust(12)} {rendered or '(none)'}")

    baseline = results[relate.SEARCH]["summary"].get("quality", 0.0)
    lines.append("")
    for stage in relate.STAGES[1:]:
        current = results[stage]["summary"].get("quality", 0.0)
        delta = (current - baseline) / baseline * 100 if baseline else 0.0
        lines.append(f"  quality {stage} vs {relate.SEARCH}: {delta:+.1f}%")

    return "\n".join(lines)


def _samples(results, terms) -> str:
    lines = []
    for term in terms:
        lines.append(f"\n{term!r}")
        for stage in relate.STAGES:
            picks = results[stage]["samples"].get(term, [])
            lines.append(f"  {stage.ljust(12)} {', '.join(picks[:6]) or '(nothing)'}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run every selection stage over the same terms and print what changed.")
    parser.add_argument("--terms", default=",".join(DEFAULT_TERMS))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--breadth", type=int, default=2)
    parser.add_argument("--backend", default=config.VECTOR_BACKEND)
    args = parser.parse_args()

    terms = [term.strip() for term in args.terms.split(",") if term.strip()]
    results = run(terms, args.limit, args.depth, args.breadth, args.backend)

    print(f"\nterms={len(terms)} limit={args.limit} backend={args.backend}\n")
    print(_table(results))
    print("\nwhat each stage actually offered")
    print(_samples(results, terms))
    print()


if __name__ == "__main__":
    main()
