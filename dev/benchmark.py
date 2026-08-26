#!/usr/bin/env python3
"""
Run the course harness and record a machine-readable benchmark result.

This module deliberately does NOT implement any evaluation metric.
The course harness is the single source of truth for nDCG/MAP/MRR/P@10.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CORPUS = ROOT / "data/nfcorpus/corpus.jsonl"
DEFAULT_QUERIES = ROOT / "data/nfcorpus/queries_dev.tsv"
DEFAULT_QRELS = ROOT / "data/nfcorpus/qrels_dev.txt"
DEFAULT_BASELINE_RUN = ROOT / "data/toy/reference_bm25_run_dev.trec"

RESULTS_DIR = ROOT / "dev/results"
RUNS_DIR = RESULTS_DIR / "runs"
CURRENT_JSON = RESULTS_DIR / "current.json"
HISTORY_JSONL = RESULTS_DIR / "history.jsonl"


def _git_commit() -> str | None:
    """Return the current git commit, if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _read_report(path: Path) -> dict[str, Any]:
    """Read the JSON report produced by harness/run_harness.py."""
    with path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    if not isinstance(report, dict):
        raise ValueError(f"Harness report is not a JSON object: {path}")

    return report


def _find_value(obj: Any, candidates: tuple[str, ...]) -> Any:
    """
    Find a metric recursively.

    The harness currently emits a JSON report, but this makes the tooling
    tolerant of harmless changes in whether metrics live at the top level
    or under a nested object.
    """
    if isinstance(obj, dict):
        lowered = {str(k).lower(): v for k, v in obj.items()}

        for candidate in candidates:
            candidate_lower = candidate.lower()
            if candidate_lower in lowered:
                return lowered[candidate_lower]

        for value in obj.values():
            found = _find_value(value, candidates)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = _find_value(value, candidates)
            if found is not None:
                return found

    return None


def _normalise_number(value: Any) -> float | int | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return value

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the important leaderboard/efficiency values from the harness report.

    We retain the complete raw report as well, so if the course changes its
    report format we don't lose information.
    """
    aliases = {
        "ndcg@10": ("ndcg@10", "ndcg_10", "ndcg10"),
        "map@10": ("map@10", "map_10", "map10"),
        "mrr": ("mrr",),
        "p@10": ("p@10", "p_10", "p10", "precision@10"),
        "build_time_s": (
            "index build time",
            "build_time_s",
            "build_time",
        ),
        "load_time_s": (
            "index load time",
            "load_time_s",
            "load_time",
        ),
        "index_size_bytes": (
            "index_size_bytes",
            "index size on disk",
            "index_size",
        ),
        "mean_query_latency_ms": (
            "mean_query_latency_ms",
            "mean query latency",
            "mean_latency_ms",
        ),
        "max_query_latency_ms": (
            "max_query_latency_ms",
            "max query latency",
            "max_latency_ms",
        ),
        "provisional_score": (
            "provisional score",
            "provisional_score",
            "provisional_score_80pct",
        ),
    }

    result: dict[str, Any] = {}

    for name, candidates in aliases.items():
        result[name] = _normalise_number(
            _find_value(report, candidates)
        )

    return result


def run_harness(
    *,
    corpus: Path,
    queries: Path,
    qrels: Path,
    baseline_run: Path | None,
    run_out: Path,
    report_out: Path,
    index_dir: Path,
) -> dict[str, Any]:
    """Invoke the repository's authoritative harness."""
    command = [
        sys.executable,
        "-m",
        "harness.run_harness",
        "--corpus",
        str(corpus),
        "--queries",
        str(queries),
        "--qrels",
        str(qrels),
        "--run-out",
        str(run_out),
        "--report-out",
        str(report_out),
        "--index-dir",
        str(index_dir),
    ]

    if baseline_run is not None:
        command.extend(["--baseline-run", str(baseline_run)])

    print()
    print("[3/4] Running course retrieval harness...")
    print("      " + " ".join(command))

    started = time.perf_counter()

    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
    )

    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        raise RuntimeError(
            f"Course harness failed with exit code "
            f"{completed.returncode}."
        )

    if not report_out.exists():
        raise RuntimeError(
            f"Harness succeeded but did not create report: {report_out}"
        )

    report = _read_report(report_out)
    metrics = extract_metrics(report)

    return {
        "harness_wall_time_s": elapsed,
        "metrics": metrics,
        "raw_report": report,
    }


def save_result(
    *,
    experiment: str,
    dataset: str,
    harness_result: dict[str, Any],
    run_out: Path,
    report_out: Path,
) -> dict[str, Any]:
    """Persist current result and append it to experiment history."""
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "dataset": dataset,
        "git_commit": _git_commit(),
        "metrics": harness_result["metrics"],
        "harness_wall_time_s": harness_result["harness_wall_time_s"],
        "run_file": str(run_out.relative_to(ROOT)),
        "report_file": str(report_out.relative_to(ROOT)),
    }

    with CURRENT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    with HISTORY_JSONL.open("a", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"))
        f.write("\n")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the course retrieval harness and record results."
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES,
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=DEFAULT_QRELS,
    )
    parser.add_argument(
        "--baseline-run",
        type=Path,
        default=DEFAULT_BASELINE_RUN,
    )
    parser.add_argument(
        "--experiment",
        default="manual",
        help="Human-readable experiment name.",
    )
    parser.add_argument(
        "--dataset",
        default="nfcorpus",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _ensure_dirs()

    for path in (args.corpus, args.queries, args.qrels):
        if not path.exists():
            print(f"ERROR: required file does not exist: {path}", file=sys.stderr)
            return 2

    if args.baseline_run is not None and not args.baseline_run.exists():
        print(
            f"WARNING: baseline run does not exist: {args.baseline_run}",
            file=sys.stderr,
        )
        print("         Continuing without --baseline-run.")
        baseline_run = None
    else:
        baseline_run = args.baseline_run

    stamp = time.strftime("%Y%m%d_%H%M%S")

    run_out = RUNS_DIR / f"{args.dataset}_{stamp}.trec"
    report_out = RUNS_DIR / f"{args.dataset}_{stamp}.json"
    index_dir = RUNS_DIR / f"{args.dataset}_{stamp}_index"

    try:
        harness_result = run_harness(
            corpus=args.corpus,
            queries=args.queries,
            qrels=args.qrels,
            baseline_run=baseline_run,
            run_out=run_out,
            report_out=report_out,
            index_dir=index_dir,
        )

        result = save_result(
            experiment=args.experiment,
            dataset=args.dataset,
            harness_result=harness_result,
            run_out=run_out,
            report_out=report_out,
        )

    finally:
        # The persisted run is what we care about. The index itself is
        # disposable and can be large.
        if index_dir.exists():
            import shutil
            shutil.rmtree(index_dir, ignore_errors=True)

    print()
    print("[4/4] Current benchmark recorded.")
    print(f"      {CURRENT_JSON}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())