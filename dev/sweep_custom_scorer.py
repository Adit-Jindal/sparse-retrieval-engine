#!/usr/bin/env python3
"""
dev/sweep_custom_scorer.py — staged sweep across every tunable in the
pipeline: query-time fusion/PRF refinements (Phase 1, no reindex needed)
and indexing-time features (Phase 2, each row rebuilds the index).

Usage:
    python -m dev.sweep_custom_scorer --stage fusion_mode
    python -m dev.sweep_custom_scorer --stage rrf_weights
    python -m dev.sweep_custom_scorer --stage prf_seed_weighting
    python -m dev.sweep_custom_scorer --stage short_query_alpha
    python -m dev.sweep_custom_scorer --stage corpus_stopwords
    python -m dev.sweep_custom_scorer --stage compounds_bigrams
    python -m dev.sweep_custom_scorer --stage capitalization
    python -m dev.sweep_custom_scorer --stage pseudo_title
    python -m dev.sweep_custom_scorer --stage combine_winners
    python -m dev.sweep_custom_scorer --stage all --dataset trec_covid

RUN STAGES IN THIS ORDER. After each stage, manually fold its winning
env-var value(s) into FROZEN_BASE below before running the next stage,
so every later stage sweeps on top of the current best rather than the
original defaults. This mirrors the sequencing used throughout the
tuning process so far — each stage's winner becomes the next stage's
fixed baseline, not a competing alternative tested in isolation.

PHASE 1 (fusion_mode, rrf_weights, prf_seed_weighting, short_query_alpha):
    Query-time only. Same index reused across every row within a stage
    — fast, no rebuild cost. Run all four before touching Phase 2.

PHASE 2 (corpus_stopwords, compounds_bigrams, capitalization,
pseudo_title, combine_winners):
    Each row rebuilds the index from scratch (INDEXER_* env vars affect
    build_index()). Deliberately tested ONE FEATURE FAMILY AT A TIME
    against baseline first — not a 2^4 combinatorial grid, which would
    be 16 rebuilds and mostly wasted effort. combine_winners is the one
    stage that stacks confirmed-positive features together; run it last,
    after manually noting which of the four Phase 2 stages actually won.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "dev/results/sweep_runs"

# Everything decided and frozen so far. Update this between stages as
# instructed above — every row below is layered ON TOP of this dict, not
# run against code-level defaults, so it always reflects "current best."
FROZEN_BASE: dict = {
    "INDEXER_STOPWORD_SET": "standard",
    "CUSTOM_SCORER_USE_VSM_FUSION": "0",
    "CUSTOM_SCORER_USE_PRF": "1",
    "CUSTOM_SCORER_COVERAGE_WEIGHT": "0",
    "CUSTOM_SCORER_BM25_K1": "1.6",
    "CUSTOM_SCORER_BM25_B": "0.5",
    "CUSTOM_SCORER_PRF_ALPHA": "0.5",
    "CUSTOM_SCORER_PRF_TOP_DOCS": "10",
    "CUSTOM_SCORER_PRF_TOP_TERMS": "10",
    "CUSTOM_SCORER_PRF_MAX_DF_RATIO": "0.15",
    "CUSTOM_SCORER_PRF_WEIGHT": "1.0",
    "CUSTOM_SCORER_RRF_K": "60",
    "CUSTOM_SCORER_BM25_WEIGHT": "0.65",
    # --- previously implicit / ambiguous, now explicit ---
    "INDEXER_USE_CORPUS_STOPWORDS": "0",          # or "0" — confirm which your 0.70 config used
    "INDEXER_CORPUS_STOPWORD_DF_RATIO": "0.85",
    "INDEXER_USE_CAPITALIZATION": "0",
    "CUSTOM_SCORER_CAPITALIZATION_WEIGHT": "0.5", # confirm exact winning value
    "INDEXER_USE_PSEUDO_GIST": "1",
    "CUSTOM_SCORER_GIST_BOOST_WEIGHT": "0.5",    # confirm exact winning value
    "CUSTOM_SCORER_PRF_WEIGHT_SEEDS_BY_SCORE": "1",  # or "1" — confirm which was actually validated as better
    "CUSTOM_SCORER_PRF_ALPHA_SHORT": "0.7",          # "0.5" = true no-op (matches PRF_ALPHA); set to "0.7" only if that's confirmed as a real win, not the mislabeled-off artifact
}

STAGES: dict = {
    # ======================================================================
    # PHASE 1 — query-time only, no reindex. Run these four first.
    # ======================================================================

    # Stage 1a: RRF vs. score-normalised linear fusion, at current
    # frozen fusion weights. If linear wins here, its weights should
    # probably be re-swept separately afterward (normalised-score sums
    # behave differently from rank-reciprocal sums) — treat a linear win
    # as "worth a follow-up stage", not an immediate final answer.
    "fusion_mode": [
        ("rrf", {"CUSTOM_SCORER_FUSION_MODE": "rrf"}),  # == current frozen default
        ("linear", {"CUSTOM_SCORER_FUSION_MODE": "linear"}),
    ],

    # Stage 1b: RRF_K and the BM25_WEIGHT/PRF_WEIGHT pair, swept jointly
    # — this pairing was flagged as never having been tested together
    # and is the least-explored corner of an otherwise heavily-tuned
    # fusion layer. PRF_WEIGHT here starts from 1.0 (its own
    # independently-confirmed best) and BM25_WEIGHT is varied around it.
    "rrf_weights": [
        ("base", {}),  # rrf_k=60, bm25_w=0.65, prf_w=1.0
        ("rrfk_30", {"CUSTOM_SCORER_RRF_K": "30"}),
        ("rrfk_100", {"CUSTOM_SCORER_RRF_K": "100"}),
        ("bm25w_0.5", {"CUSTOM_SCORER_BM25_WEIGHT": "0.5"}),
        ("bm25w_1.0", {"CUSTOM_SCORER_BM25_WEIGHT": "1.0"}),
        ("bm25w_1.0_prfw_1.0_rrfk_30", {
            "CUSTOM_SCORER_BM25_WEIGHT": "1.0", "CUSTOM_SCORER_RRF_K": "30",
        }),
        ("bm25w_0.5_prfw_1.0_rrfk_100", {
            "CUSTOM_SCORER_BM25_WEIGHT": "0.5", "CUSTOM_SCORER_RRF_K": "100",
        }),
    ],

    # Stage 1b': same weight grid as rrf_weights, but under linear fusion
    # — needed for a fair RRF-at-its-best vs linear-at-its-best
    # comparison, since linear has never had its own weight tuning.
    "linear_weights": [

        ("bm25w_0.5_prfw_0.5", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "0.5",
            "CUSTOM_SCORER_PRF_WEIGHT": "0.5",
        }),
        ("bm25w_0.5_prfw_0.75", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "0.5",
            "CUSTOM_SCORER_PRF_WEIGHT": "0.75",
        }),
        ("bm25w_0.5_prfw_1.0", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "0.5",
            "CUSTOM_SCORER_PRF_WEIGHT": "1.0",
        }),
        ("bm25w_0.5_prfw_1.25", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "0.5",
            "CUSTOM_SCORER_PRF_WEIGHT": "1.25",
        }),
        ("bm25w_0.5_prfw_1.5", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "0.5",
            "CUSTOM_SCORER_PRF_WEIGHT": "1.5",
        }),

        ("bm25w_0.75_prfw_0.5", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "0.75",
            "CUSTOM_SCORER_PRF_WEIGHT": "0.5",
        }),
        ("bm25w_0.75_prfw_0.75", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "0.75",
            "CUSTOM_SCORER_PRF_WEIGHT": "0.75",
        }),
        ("bm25w_0.75_prfw_1.0", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "0.75",
            "CUSTOM_SCORER_PRF_WEIGHT": "1.0",
        }),
        ("bm25w_0.75_prfw_1.25", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "0.75",
            "CUSTOM_SCORER_PRF_WEIGHT": "1.25",
        }),
        ("bm25w_0.75_prfw_1.5", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "0.75",
            "CUSTOM_SCORER_PRF_WEIGHT": "1.5",
        }),

        ("bm25w_1.0_prfw_0.5", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "1.0",
            "CUSTOM_SCORER_PRF_WEIGHT": "0.5",
        }),
        ("bm25w_1.0_prfw_0.75", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "1.0",
            "CUSTOM_SCORER_PRF_WEIGHT": "0.75",
        }),
        ("bm25w_1.0_prfw_1.0", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "1.0",
            "CUSTOM_SCORER_PRF_WEIGHT": "1.0",
        }),
        ("bm25w_1.0_prfw_1.25", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "1.0",
            "CUSTOM_SCORER_PRF_WEIGHT": "1.25",
        }),
        ("bm25w_1.0_prfw_1.5", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "1.0",
            "CUSTOM_SCORER_PRF_WEIGHT": "1.5",
        }),

        ("bm25w_1.25_prfw_0.5", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "1.25",
            "CUSTOM_SCORER_PRF_WEIGHT": "0.5",
        }),
        ("bm25w_1.25_prfw_0.75", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "1.25",
            "CUSTOM_SCORER_PRF_WEIGHT": "0.75",
        }),
        ("bm25w_1.25_prfw_1.0", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "1.25",
            "CUSTOM_SCORER_PRF_WEIGHT": "1.0",
        }),
        ("bm25w_1.25_prfw_1.25", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "1.25",
            "CUSTOM_SCORER_PRF_WEIGHT": "1.25",
        }),
        ("bm25w_1.25_prfw_1.5", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "1.25",
            "CUSTOM_SCORER_PRF_WEIGHT": "1.5",
        }),

        ("bm25w_1.5_prfw_0.5", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "1.5",
            "CUSTOM_SCORER_PRF_WEIGHT": "0.5",
        }),
        ("bm25w_1.5_prfw_0.75", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "1.5",
            "CUSTOM_SCORER_PRF_WEIGHT": "0.75",
        }),
        ("bm25w_1.5_prfw_1.0", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "1.5",
            "CUSTOM_SCORER_PRF_WEIGHT": "1.0",
        }),
        ("bm25w_1.5_prfw_1.25", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "1.5",
            "CUSTOM_SCORER_PRF_WEIGHT": "1.25",
        }),
        ("bm25w_1.5_prfw_1.5", {
            "CUSTOM_SCORER_FUSION_MODE": "linear",
            "CUSTOM_SCORER_BM25_WEIGHT": "1.5",
            "CUSTOM_SCORER_PRF_WEIGHT": "1.5",
        }),
    ],

    # Stage 1c: PRF seed-doc weighting by first-pass score (query-drift
    # mitigation) — binary flag, cheap to test in isolation.
    "prf_seed_weighting": [
        ("off", {}),  # current default
        ("on", {"CUSTOM_SCORER_PRF_WEIGHT_SEEDS_BY_SCORE": "1"}),
    ],

    "prf_use_seed": [
        ("off", {}),  # current default
        ("on", {"CUSTOM_SCORER_USE_SEED_POOLING": "1"}),
    ],

    # Stage 1d: separate PRF_ALPHA for short queries. Sweep both the
    # short-query alpha value AND the length threshold that defines
    # "short", since the right threshold isn't obvious a priori.
    "short_query_alpha": [
        ("off", {}),  # alpha_short == alpha == 0.5, no-op
        ("thresh2_alpha0.3", {
            "CUSTOM_SCORER_SHORT_QUERY_MAX_TERMS": "2",
            "CUSTOM_SCORER_PRF_ALPHA_SHORT": "0.5",
        }),
        ("thresh2_alpha0.7", {
            "CUSTOM_SCORER_SHORT_QUERY_MAX_TERMS": "2",
            "CUSTOM_SCORER_PRF_ALPHA_SHORT": "0.7",
        }),
        ("thresh3_alpha0.3", {
            "CUSTOM_SCORER_SHORT_QUERY_MAX_TERMS": "3",
            "CUSTOM_SCORER_PRF_ALPHA_SHORT": "0.5",
        }),
        ("thresh3_alpha0.7", {
            "CUSTOM_SCORER_SHORT_QUERY_MAX_TERMS": "3",
            "CUSTOM_SCORER_PRF_ALPHA_SHORT": "0.7",
        }),
    ],

    # ======================================================================
    # PHASE 2 — indexing-time features. Each row rebuilds the index.
    # Test each family independently against baseline before combining.
    # ======================================================================

    # Stage 2a: corpus-frequency-driven stopword augmentation. Sweep the
    # df-ratio threshold itself — too aggressive risks dropping
    # genuinely useful mid-frequency terms.
    "corpus_stopwords": [
        ("off", {}),  # baseline: no corpus-driven pruning
        ("ratio_0.9", {
            "INDEXER_USE_CORPUS_STOPWORDS": "1",
            "INDEXER_CORPUS_STOPWORD_DF_RATIO": "0.9",
        }),
        ("ratio_0.85", {
            "INDEXER_USE_CORPUS_STOPWORDS": "1",
            "INDEXER_CORPUS_STOPWORD_DF_RATIO": "0.85",
        }),
        ("ratio_0.7", {
            "INDEXER_USE_CORPUS_STOPWORDS": "1",
            "INDEXER_CORPUS_STOPWORD_DF_RATIO": "0.7",
        }),
    ],

    # Stage 2b: compound-term preservation and bigrams. Test
    # independently first (isolate which one, if either, actually helps)
    # before considering combining them.
    "compounds_bigrams": [
        ("off", {}),  # baseline
        ("compounds_only", {"INDEXER_PRESERVE_COMPOUNDS": "1"}),
        ("bigrams_only", {"INDEXER_USE_BIGRAMS": "1"}),
        ("both", {"INDEXER_PRESERVE_COMPOUNDS": "1", "INDEXER_USE_BIGRAMS": "1"}),
    ],

    # Stage 2c: capitalization-as-proper-noun-signal. Index build with
    # the feature on is free to test at CAPITALIZATION_WEIGHT=0 (it's a
    # no-op scoring-wise until the weight is nonzero, but the index
    # build cost is identical whether weight is used or not) — so the
    # first row here rebuilds once, then sweeps weight without
    # rebuilding again by only ever changing scorer-level env vars
    # against that one rebuilt index. NOTE: this script's run_one()
    # rebuilds per-row regardless (each row gets its own --index-dir);
    # accept the extra rebuild cost here for correctness/simplicity
    # rather than hand-optimising the sweep script itself.
    "capitalization": [
        ("off", {}),  # baseline, no cap tracking at all
        ("weight_1", {
            "INDEXER_USE_CAPITALIZATION": "1",
            "CUSTOM_SCORER_CAPITALIZATION_WEIGHT": "1",
        }),
        ("weight_0.75", {
            "INDEXER_USE_CAPITALIZATION": "1",
            "CUSTOM_SCORER_CAPITALIZATION_WEIGHT": "0.75",
        }),
        ("weight_0.5", {
            "INDEXER_USE_CAPITALIZATION": "1",
            "CUSTOM_SCORER_CAPITALIZATION_WEIGHT": "0.5",
        }),
    ],

    # Stage 2d: synthetic pseudo-title (first-sentence field boost).
    # Speculative — TREC-COVID abstracts may or may not have a clean
    # "topic sentence first" convention. Sweep the boost weight; if
    # every nonzero weight underperforms off, that's a clean answer
    # (reject), not inconclusive.
    "pseudo_title": [
        ("off", {}),  # baseline
        ("weight_0.75", {
            "INDEXER_USE_PSEUDO_TITLE": "1",
            "CUSTOM_SCORER_TITLE_BOOST_WEIGHT": "0.75",
        }),
        ("weight_0.3", {
            "INDEXER_USE_PSEUDO_TITLE": "1",
            "CUSTOM_SCORER_TITLE_BOOST_WEIGHT": "0.3",
        }),
        ("weight_0.5", {
            "INDEXER_USE_PSEUDO_TITLE": "1",
            "CUSTOM_SCORER_TITLE_BOOST_WEIGHT": "0.5",
        }),
    ],

    # Stage 2e (fill in manually after 2a-2d): stack only the winning
    # rows from stages 2a-2d together into one rebuilt index, to check
    # for interaction effects before finalising. Left as a template —
    # edit the dict values to match your actual per-stage winners before
    # running.
    "combine_winners": [
        ("phase2_baseline", {}),  # no Phase 2 features at all, for reference
        ("all_confirmed_winners", {
            # EDIT THESE to match whichever rows actually won stages 2a-2d.
            # Example if corpus_stopwords@0.85 and bigrams_only both won:
            # "INDEXER_USE_CORPUS_STOPWORDS": "1",
            # "INDEXER_CORPUS_STOPWORD_DF_RATIO": "0.85",
            # "INDEXER_USE_BIGRAMS": "1",
        }),
    ],

    "positional_proximity": [
        ("off", {}),  # baseline: no positions, no boost
        ("weight_0.1_pool_30", {
            "INDEXER_USE_POSITIONS": "1",
            "CUSTOM_SCORER_PROXIMITY_WEIGHT": "0.1",
            "CUSTOM_SCORER_PROXIMITY_POOL": "30",
        }),
        ("weight_0.25_pool_30", {
            "INDEXER_USE_POSITIONS": "1",
            "CUSTOM_SCORER_PROXIMITY_WEIGHT": "0.25",
            "CUSTOM_SCORER_PROXIMITY_POOL": "30",
        }),
        ("weight_0.25_pool_50", {
            "INDEXER_USE_POSITIONS": "1",
            "CUSTOM_SCORER_PROXIMITY_WEIGHT": "0.25",
            "CUSTOM_SCORER_PROXIMITY_POOL": "50",
        }),
        ("weight_0.5_pool_50", {
            "INDEXER_USE_POSITIONS": "1",
            "CUSTOM_SCORER_PROXIMITY_WEIGHT": "0.5",
            "CUSTOM_SCORER_PROXIMITY_POOL": "50",
        }),
    ],
    "document_gist": [
        ("off", {"INDEXER_USE_DOCUMENT_GIST": "0"}),
        ("pure_gist_k15_w0.3", {
            "INDEXER_USE_DOCUMENT_GIST": "1", "INDEXER_GIST_TOP_K": "15",
            "CUSTOM_SCORER_GIST_BOOST_WEIGHT": "0.3",
        }),  # lead_boost_weight defaults 0 — isolates pure tf*idf gist first
        ("lead_k15_w0.3_lead0.5", {
            "INDEXER_USE_DOCUMENT_GIST": "1", "INDEXER_GIST_TOP_K": "15",
            "INDEXER_GIST_LEAD_BOOST_WEIGHT": "0.5", "INDEXER_GIST_LEAD_DECAY_FRAC": "0.1",
            "CUSTOM_SCORER_GIST_BOOST_WEIGHT": "0.3",
        }),
        ("lead_k15_w0.3_lead1.0", {
            "INDEXER_USE_DOCUMENT_GIST": "1", "INDEXER_GIST_TOP_K": "15",
            "INDEXER_GIST_LEAD_BOOST_WEIGHT": "1.0", "INDEXER_GIST_LEAD_DECAY_FRAC": "0.1",
            "CUSTOM_SCORER_GIST_BOOST_WEIGHT": "0.3",
        }),
        ("lead_k15_w0.3_lead1.0_decay0.2", {
            "INDEXER_USE_DOCUMENT_GIST": "1", "INDEXER_GIST_TOP_K": "15",
            "INDEXER_GIST_LEAD_BOOST_WEIGHT": "1.0", "INDEXER_GIST_LEAD_DECAY_FRAC": "0.2",
            "CUSTOM_SCORER_GIST_BOOST_WEIGHT": "0.3",
        }),
    ],
    "seed_pool_tradeoff": [
        ("current_baseline", {}),   # whatever your promoted config's DEPTH-equivalent behavior was
        ("seed0.01_n10", {"CUSTOM_SCORER_SEED_DF_RATIO": "0.01", "CUSTOM_SCORER_RERANK_POOL_MULTIPLIER": "10"}),
        ("seed0.02_n10", {"CUSTOM_SCORER_SEED_DF_RATIO": "0.02", "CUSTOM_SCORER_RERANK_POOL_MULTIPLIER": "10"}),
        ("seed0.02_n20", {"CUSTOM_SCORER_SEED_DF_RATIO": "0.02", "CUSTOM_SCORER_RERANK_POOL_MULTIPLIER": "20"}),
        ("seed0.05_n20", {"CUSTOM_SCORER_SEED_DF_RATIO": "0.05", "CUSTOM_SCORER_RERANK_POOL_MULTIPLIER": "20"}),
    ],
    "seed_pool_reference_check": [
        ("pooling_off_default", {"CUSTOM_SCORER_USE_SEED_POOLING": "0"}),  # must reproduce your 0.67-0.68 promoted baseline
        ("pool_on_prev_settings", {"CUSTOM_SCORER_USE_SEED_POOLING": "1", "CUSTOM_SCORER_SEED_DF_RATIO": "0.05", "CUSTOM_SCORER_RERANK_POOL_MULTIPLIER": "20"}),
    ],

    "prf_sweep": [
        (
            f"seed_{seed:.2f}_alpha_{alpha:.1f}_maxdf_{max_df:.2f}",
            {
                "CUSTOM_SCORER_SEED_DF_RATIO": f"{seed:.2f}",
                "CUSTOM_SCORER_PRF_ALPHA": f"{alpha:.1f}",
                "CUSTOM_SCORER_PRF_MAX_DF_RATIO": f"{max_df:.2f}",
            },
        )
        for seed in [0.01, 0.02, 0.03, 0.04, 0.05]
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]
        for max_df in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    ],

    "seed_df": [
        ("df_ratio0.01", {"CUSTOM_SCORER_SEED_DF_RATIO": "0.01"}),
        ("df_ratio0.02", {"CUSTOM_SCORER_SEED_DF_RATIO": "0.02"}),
        ("df_ratio0.03", {"CUSTOM_SCORER_SEED_DF_RATIO": "0.03"}),
        ("df_ratio0.04", {"CUSTOM_SCORER_SEED_DF_RATIO": "0.04"}),
        ("df_ratio0.05", {"CUSTOM_SCORER_SEED_DF_RATIO": "0.05"}),
        ("df_ratio0.06", {"CUSTOM_SCORER_SEED_DF_RATIO": "0.06"}),
        ("df_ratio0.07", {"CUSTOM_SCORER_SEED_DF_RATIO": "0.07"}),
        ("df_ratio0.08", {"CUSTOM_SCORER_SEED_DF_RATIO": "0.08"}),
        ("df_ratio0.09", {"CUSTOM_SCORER_SEED_DF_RATIO": "0.09"}),
        ("df_ratio0.10", {"CUSTOM_SCORER_SEED_DF_RATIO": "0.10"}),
    ],
    "feature_weights": [
        (
            f"coverage_{coverage:.1f}_caps_{caps:.1f}_gist_{gist:.1f}",
            {
                "CUSTOM_SCORER_COVERAGE_WEIGHT": f"{coverage:.1f}",
                "CUSTOM_SCORER_CAPITALIZATION_WEIGHT": f"{caps:.1f}",
                "CUSTOM_SCORER_GIST_BOOST_WEIGHT": f"{gist:.1f}",
            },
        )
        for coverage in [0.8]
        for caps in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for gist in [0.4]
    ],
}


def run_one(name: str, env_overrides: dict, dataset_paths: dict) -> dict:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_out = RUNS_DIR / f"{name}.trec"
    report_out = RUNS_DIR / f"{name}.json"
    index_dir = RUNS_DIR / f"{name}_index"

    env = os.environ.copy()
    env.update(FROZEN_BASE)
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

    print(f"\n--- running config: {name} (overrides: {env_overrides}) ---")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    wall = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode})")
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
        "mean_latency_ms": eff["mean_query_latency_seconds"] * 1000,
        "max_latency_ms": eff["max_query_latency_seconds"] * 1000,
        "build_s": eff["index_build_seconds"],
        "load_s": eff["index_load_seconds"],
        "index_size_bytes": eff["index_size_bytes"],
        "wall_s": wall,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["toy", "nfcorpus", "full"], default="full")
    parser.add_argument("--stage", choices=list(STAGES.keys()) + ["all"], default="all")
    args = parser.parse_args()

    dataset_dirs = {
        "toy": "data/toy",
        "nfcorpus": "data/nfcorpus",
        "full": "data/full",  # adjust to your actual corpus location
    }
    d = ROOT / dataset_dirs[args.dataset]
    paths = {
        "corpus": d / "corpus.jsonl",
        "queries": d / "queries_dev.tsv",
        "qrels": d / "qrels_dev.txt",
        "baseline_run": ROOT / "data/toy/reference_bm25_run_dev.trec",
    }

    for p in paths.values():
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    stage_names = list(STAGES.keys()) if args.stage == "all" else [args.stage]

    for stage_name in stage_names:
        print(f"\n{'#' * 96}\n# STAGE: {stage_name}\n{'#' * 96}")
        configs = STAGES[stage_name]
        results = [run_one(f"{stage_name}__{name}", overrides, paths) for name, overrides in configs]

        print("\n" + "=" * 104)
        print(f"{'config':<32}{'nDCG@10':>10}{'MAP@10':>10}{'mean_lat_ms':>14}{'max_lat_ms':>13}{'load_s':>9}{'idx_bytes':>14}")
        print("-" * 104)
        for r in results:
            if not r["ok"]:
                print(f"{r['name']:<32}{'FAILED':>10}")
                continue
            print(
                f"{r['name']:<32}"
                f"{r['ndcg@10']:>10.4f}"
                f"{r['map@10']:>10.4f}"
                f"{r['mean_latency_ms']:>14.2f}"
                f"{r['max_latency_ms']:>13.2f}"
                f"{r['load_s']:>9.3f}"
                f"{r['index_size_bytes']:>14,}"
            )
        print("=" * 104)

        out_path = RUNS_DIR / f"{stage_name}_summary.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Stage results: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())