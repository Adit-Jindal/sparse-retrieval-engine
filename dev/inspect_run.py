#!/usr/bin/env python3
"""Inspect a TREC run file for a particular query."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "dev/results/runs"


def read_run(path: Path, query_id: str, limit: int) -> list[tuple[str, float]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()

            if len(parts) < 6:
                continue

            qid, _, doc_id, rank, score, _ = parts[:6]

            if qid != query_id:
                continue

            try:
                rows.append((doc_id, float(score), int(rank)))
            except ValueError:
                continue

    rows.sort(key=lambda x: x[2])

    return [(doc_id, score) for doc_id, score, _ in rows[:limit]]


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "run",
        type=Path,
        help="Path to a TREC run file.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Query/topic ID.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    if not args.run.exists():
        print(f"ERROR: run file does not exist: {args.run}")
        return 2

    rows = read_run(args.run, args.query, args.k)

    print(f"Query: {args.query}")
    print("-" * 50)

    if not rows:
        print("No results found.")
        return 0

    for rank, (doc_id, score) in enumerate(rows, start=1):
        print(f"{rank:>2}. {doc_id:<30} {score:>12.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())