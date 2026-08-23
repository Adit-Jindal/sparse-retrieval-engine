#!/usr/bin/env python3
"""Compare the current benchmark result with the accepted best result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "dev/results"
CURRENT = RESULTS / "current.json"
BEST = RESULTS / "best.json"

METRICS = [
    ("ndcg@10", True),
    ("map@10", True),
    ("mrr", True),
    ("p@10", True),
    ("provisional_score", True),
    ("index_size_bytes", False),
    ("mean_query_latency_ms", False),
    ("max_query_latency_ms", False),
    ("build_time_s", False),
    ("load_time_s", False),
]


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)

    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")

    return value


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"

    if isinstance(value, (int, float)):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.4f}"

    return str(value)


def compare(best: dict[str, Any], current: dict[str, Any]) -> None:
    best_metrics = best.get("metrics", {})
    current_metrics = current.get("metrics", {})

    print()
    print("=" * 76)
    print("                    RETRIEVAL REGRESSION")
    print("=" * 76)

    print(
        f"{'Metric':<28}"
        f"{'BEST':>14}"
        f"{'CURRENT':>14}"
        f"{'DELTA':>14}"
    )
    print("-" * 76)

    improvements = []
    regressions = []

    for metric, higher_is_better in METRICS:
        old = best_metrics.get(metric)
        new = current_metrics.get(metric)

        if old is None or new is None:
            print(
                f"{metric:<28}"
                f"{fmt(old):>14}"
                f"{fmt(new):>14}"
                f"{'N/A':>14}"
            )
            continue

        delta = new - old

        # Positive "change" means improvement regardless of whether
        # higher or lower is desirable.
        improvement = delta if higher_is_better else -delta

        if improvement > 1e-12:
            improvements.append(metric)
            marker = " ↑"
        elif improvement < -1e-12:
            regressions.append(metric)
            marker = " ↓"
        else:
            marker = " ="

        print(
            f"{metric:<28}"
            f"{fmt(old):>14}"
            f"{fmt(new):>14}"
            f"{delta:>13.4f}{marker}"
        )

    print("=" * 76)

    if current.get("git_commit"):
        print(f"Current commit: {current['git_commit']}")

    print(f"Current experiment: {current.get('experiment', 'unknown')}")

    if not best_metrics:
        print("\nSTATUS: no previous best metrics available.")
        return

    if improvements and not regressions:
        status = "IMPROVEMENT"
    elif regressions and not improvements:
        status = "REGRESSION"
    elif improvements and regressions:
        status = "MIXED"
    else:
        status = "NO CHANGE"

    print(f"\nSTATUS: {status}")

    if improvements:
        print("Improved:", ", ".join(improvements))

    if regressions:
        print("Regressed:", ", ".join(regressions))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", type=Path, default=BEST)
    parser.add_argument("--current", type=Path, default=CURRENT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        current = load(args.current)
    except FileNotFoundError:
        print(f"No current result found: {args.current}")
        print("Run scripts/run_dev.sh first.")
        return 2

    if not args.best.exists():
        print("No accepted best result exists yet.")
        print("The current result can be promoted with:")
        print("    python -m dev.promote")
        return 0

    best = load(args.best)
    compare(best, current)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())