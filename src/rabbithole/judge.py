from __future__ import annotations
import argparse
import random
from rabbithole import bench, config, labels, metrics, relate

PROMPT = """
  0  nothing here I would click
  1  one usable door
  2  a couple, mostly filler
  3  solid, would keep going
  4  good spread, learned something
  5  exactly the rabbit hole I wanted

  s  skip    q  save and quit
"""

TERMS = (
    "car", "jazz", "penicillin", "the great depression", "octopus", "brutalism",
    "the silk road", "quantum entanglement", "kintsugi", "the dust bowl",
    "bioluminescence", "esperanto", "the globe theatre", "permaculture", "sourdough",
    "cartography", "the mariana trench", "hedge maze", "morse code", "tulip mania",)


def ask(term: str, hole, scored: dict[str, float]) -> float | None:
    print(f"\n{'=' * 70}\n{hole.top.title}\n{'-' * 70}")
    print((hole.top.summary or "")[:280])
    print("\nwhere it says this leads:")
    for position, entry in enumerate(hole.more, start=1):
        why = f"  [{entry.why}]" if entry.why else ""
        print(f"  {position:>2}. {entry.title}{why}")

    print(f"\nproxy quality says {metrics.quality(scored):.3f}")
    print(PROMPT)

    while True:
        answer = input(f"rating for {term!r}: ").strip().lower()
        if answer == "q":
            return None
        if answer == "s":
            return -1.0
        if answer in {"0", "1", "2", "3", "4", "5"}:
            return int(answer) / 5.0

        print("  0-5, or s to skip, or q to quit")


def collect(terms: list[str], stage: str, limit: int, backend: str) -> int:
    stored = 0

    for term in terms:
        expansion, hole = bench.expansion_for(term, stage, limit)
        if expansion is None:
            print(f"\n{term!r}: nothing found, skipping")
            continue

        scored = metrics.score(expansion, backend=backend)
        rating = ask(term, hole, scored)
        if rating is None:
            break
        if rating < 0.0:
            continue

        labels.record(
            labels.Judgement(
                parent=hole.top.title,
                picks=[entry.title for entry in hole.more],
                rating=rating,
                stage=stage,
                backend=backend,
                scored=scored,))
        stored += 1

    return stored


def report() -> str:
    judged = labels.load()
    if not judged:
        return "no judgements recorded yet"

    calibration = metrics.calibrate(judged)
    lines = [calibration.verdict(), ""]
    if calibration.samples >= 2:
        lines.append("weights the ratings actually support:")
        for name, weight in sorted(calibration.weights.items(), key=lambda kv: -kv[1]):
            hand = metrics.QUALITY_WEIGHTS[name]
            lines.append(f"  {name.ljust(16)} {weight:.3f}   (hand-set {hand:.2f})")

    by_stage = {judgement.stage for judgement in judged}
    lines.append("")
    for stage in sorted(by_stage):
        rows = labels.for_stage(judged, stage)
        mean = sum(row.rating for row in rows) / len(rows)
        lines.append(f"  {stage.ljust(12)} {len(rows):>3} judged, mean human rating {mean:.2f}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rate expansions by hand so `quality` can be checked against a person.")
    parser.add_argument("--terms", default="", help="comma separated; sampled if omitted")
    parser.add_argument("--stage", default=config.STAGE, choices=relate.STAGES)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--backend", default=config.VECTOR_BACKEND)
    parser.add_argument("--count", type=int, default=10, help="how many to sample")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--report", action="store_true", help="print calibration and exit")
    args = parser.parse_args()

    if args.report:
        print(f"\n{report()}\n")
        return

    if args.terms:
        terms = [term.strip() for term in args.terms.split(",") if term.strip()]
    else:
        pool = list(TERMS)
        random.Random(args.seed).shuffle(pool)
        terms = pool[: args.count]

    print(f"judging {len(terms)} expansions at stage {args.stage!r}")
    print(f"writing to {config.labels_path()}")
    stored = collect(terms, args.stage, args.limit, args.backend)

    print(f"\n{stored} judgement(s) recorded\n")
    print(report())
    print()


if __name__ == "__main__":
    main()
