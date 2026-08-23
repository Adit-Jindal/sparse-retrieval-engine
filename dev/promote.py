#!/usr/bin/env python3
"""Promote the current benchmark result to the accepted best."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "dev/results"

CURRENT = RESULTS / "current.json"
BEST = RESULTS / "best.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote current benchmark result to best."
    )
    parser.parse_args()

    if not CURRENT.exists():
        print("ERROR: no current result exists.")
        print("Run:")
        print("    bash scripts/run_dev.sh")
        return 2

    RESULTS.mkdir(parents=True, exist_ok=True)

    if BEST.exists():
        backup = RESULTS / "best.previous.json"
        shutil.copy2(BEST, backup)
        print(f"Previous best backed up to {backup}")

    shutil.copy2(CURRENT, BEST)

    print()
    print("PROMOTED CURRENT RESULT TO BEST")
    print(f"Best result: {BEST}")

    with CURRENT.open("r", encoding="utf-8") as f:
        current = json.load(f)

    print(
        f"nDCG@10 = "
        f"{current.get('metrics', {}).get('ndcg@10', 'N/A')}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())