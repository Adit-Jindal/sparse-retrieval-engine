#!/usr/bin/env python3
"""
dev/sweep_custom_scorer.py — run the course harness once per
custom_scorer.py config (env-var overrides) and compare nDCG@10, MAP@10,
and latency side by side.

Usage:
    python -m dev.sweep_custom_scorer
    python -m dev.sweep_custom_scorer --dataset toy   # smaller/faster set

Add/edit CONFIGS below to add more points to the sweep (e.g. a grid over
RRF_K or COVERAGE_WEIGHT once you've settled USE_VSM_FUSION).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "dev/results/sweep_runs"

# Each entry: (name, {ENV_VAR: value, ...})
CONFIGS = [
    ("bm25_only", {
        "CUSTOM_SCORER_USE_VSM_FUSION": "0",
        "CUSTOM_SCORER_COVERAGE_WEIGHT": "0",
    }),
    ("bm25_plus_coverage", {
        "CUSTOM_SCORER_USE_VSM_FUSION": "0",
        "CUSTOM_SCORER_COVERAGE_WEIGHT": "0.35",
    }),
    ("bm25_vsm_rrf_plus_coverage", {
        "CUSTOM_SCORER_USE_VSM_FUSION": "1",
        "CUSTOM_SCORER_COVERAGE_WEIGHT": "0.35",
    }),
    # Add finer grid points here once the above narrows the decision, e.g.:
    # ("rrf_k30", {"CUSTOM_SCORER_USE_VSM_FUSION": "1", "CUSTOM_SCORER_RRF_K": "30"}),
]


def run_one(name: str, env_overrides: dict, dataset_paths: dict) -> dict:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_out = RUNS_DIR / f"{name}.trec"
    report_out = RUNS_DIR / f"{name}.json"
    index_dir = RUNS_DIR / f"{name}_index"

    env = os.environ.copy()
    env.update(env_overrides)

    cmd = [
        sys.executable, "-m", "harness.run_harness",
        "--corpus", str(dataset_paths["corpus"]),
        "--queries", str(dataset_paths["queries"]),
        "--qrels", str(dataset_paths["qrels"]),
        "--baseline-run", str(dataset_paths["baseline_run"]),
        "--run-out", str(run_out),
        "--report-out", str(report_out),
        "--index-dir", str(index_dir),
    ]

    print(f"\n--- running config: {name} ({env_overrides}) ---")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    wall = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode}) — see harness output above.")
        return {"name": name, "ok": False}

    with report_out.open(encoding="utf-8") as f:
        report = json.load(f)

    agg = report["aggregate_metrics"]
    eff = report["efficiency"]
    return {
        "name": name,
        "ok": True,
        "ndcg@10": agg["ndcg@10"],
        "map@10": agg["map@10"],
        "mrr": agg["mrr"],
        "mean_latency_ms": eff["mean_query_latency_seconds"] * 1000,
        "max_latency_ms": eff["max_query_latency_seconds"] * 1000,
        "build_s": eff["index_build_seconds"],
        "index_size_bytes": eff["index_size_bytes"],
        "wall_s": wall,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["toy", "nfcorpus", "full"], default="nfcorpus")
    args = parser.parse_args()

    # if args.dataset == "toy":
    #     paths = {
    #         "corpus": ROOT / "data/toy/corpus.jsonl",
    #         "queries": ROOT / "data/toy/queries_dev.tsv",
    #         "qrels": ROOT / "data/toy/qrels_dev.txt",
    #         "baseline_run": ROOT / "data/toy/reference_bm25_run_dev.trec",
    #     }
    # else:
    #     paths = {
    #         "corpus": ROOT / "data/nfcorpus/corpus.jsonl",
    #         "queries": ROOT / "data/nfcorpus/queries_dev.tsv",
    #         "qrels": ROOT / "data/nfcorpus/qrels_dev.txt",
    #         "baseline_run": ROOT / "data/toy/reference_bm25_run_dev.trec",
    #     }
    paths = {
        "corpus": ROOT / f"data/{args.dataset}/corpus.jsonl",
        "queries": ROOT / f"data/{args.dataset}/queries_dev.tsv",
        "qrels": ROOT / f"data/{args.dataset}/qrels_dev.txt",
        "baseline_run": ROOT / "data/toy/reference_bm25_run_dev.trec",
    }

    for p in paths.values():
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    results = [run_one(name, overrides, paths) for name, overrides in CONFIGS]

    print("\n" + "=" * 92)
    print(f"{'config':<28}{'nDCG@10':>10}{'MAP@10':>10}{'mean_lat_ms':>14}{'max_lat_ms':>13}{'idx_bytes':>14}")
    print("-" * 92)
    for r in results:
        if not r["ok"]:
            print(f"{r['name']:<28}{'FAILED':>10}")
            continue
        print(
            f"{r['name']:<28}"
            f"{r['ndcg@10']:>10.4f}"
            f"{r['map@10']:>10.4f}"
            f"{r['mean_latency_ms']:>14.2f}"
            f"{r['max_latency_ms']:>13.2f}"
            f"{r['index_size_bytes']:>14,}"
        )
    print("=" * 92)

    out_path = RUNS_DIR / "sweep_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())