"""
submission/custom_scorer.py — combined BM25 + (optional) VSM fusion +
IDF-coverage scorer.

All tunables below are read from environment variables so a sweep script
can control them across the harness's build/query subprocess boundary
without editing this file between runs. Defaults are the values to use
if nothing overrides them — i.e. what ships if you never run a sweep.

  CUSTOM_SCORER_USE_VSM_FUSION  1/0   (default: 1)
  CUSTOM_SCORER_DEPTH           int   (default: 100)
  CUSTOM_SCORER_RRF_K           float (default: 60)
  CUSTOM_SCORER_BM25_WEIGHT     float (default: 0.65)
  CUSTOM_SCORER_VSM_WEIGHT      float (default: 0.35)
  CUSTOM_SCORER_COVERAGE_WEIGHT float (default: 0.35)
  CUSTOM_SCORER_BM25_K1         float (default: 1.2)
  CUSTOM_SCORER_BM25_B          float (default: 0.75)

See dev/sweep_custom_scorer.py for the comparison harness and report
Section 8 for the resulting nDCG@10-vs-latency numbers.
"""
import heapq
import math
import os
from typing import Dict, List, Optional, Tuple

from submission.indexer import InvertedIndex, Tokenizer
from submission import bm25, boolean_vsm


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


DEPTH = int(os.environ.get("CUSTOM_SCORER_DEPTH", 100))
RRF_K = _env_float("CUSTOM_SCORER_RRF_K", 60)
BM25_WEIGHT = _env_float("CUSTOM_SCORER_BM25_WEIGHT", 0.65)
VSM_WEIGHT = _env_float("CUSTOM_SCORER_VSM_WEIGHT", 0.35)
COVERAGE_WEIGHT = _env_float("CUSTOM_SCORER_COVERAGE_WEIGHT", 0.35)
BM25_K1 = _env_float("CUSTOM_SCORER_BM25_K1", 1.6)
BM25_B = _env_float("CUSTOM_SCORER_BM25_B", 0.5)
USE_VSM_FUSION = _env_bool("CUSTOM_SCORER_USE_VSM_FUSION", True)

_INDEX: Optional[InvertedIndex] = None
_TOKENIZER: Optional[Tokenizer] = None


def build(index: InvertedIndex, tokenizer: Tokenizer) -> None:
    global _INDEX, _TOKENIZER
    _INDEX = index
    _TOKENIZER = tokenizer


def _term_idf(term: str) -> float:
    df = _INDEX.document_frequency(term)
    return math.log(((_INDEX.N - df + 0.5) / (df + 0.5)) + 1.0)


def score(query: str, k: int) -> List[Tuple[str, float]]:
    if _INDEX is None or _TOKENIZER is None:
        raise RuntimeError("custom_scorer.build() must be called before score().")
    if k <= 0:
        return []

    tokens = _TOKENIZER.tokenize(query)
    unique_terms = set(tokens)
    if not unique_terms:
        return []

    depth = max(DEPTH, k)
    bm25_ranked = bm25.score(query, depth, k1=BM25_K1, b=BM25_B)

    rrf: Dict[str, float] = {}
    for rank, (doc_id, _s) in enumerate(bm25_ranked, start=1):
        rrf[doc_id] = rrf.get(doc_id, 0.0) + BM25_WEIGHT / (RRF_K + rank)

    if USE_VSM_FUSION:
        vsm_ranked = boolean_vsm.vsm_score(query, depth)
        for rank, (doc_id, _s) in enumerate(vsm_ranked, start=1):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + VSM_WEIGHT / (RRF_K + rank)

    if not rrf:
        return []

    term_idfs = {t: _term_idf(t) for t in unique_terms}
    query_idf_total = sum(term_idfs.values())

    if query_idf_total <= 0:
        scores = rrf
    else:
        matched_idf: Dict[str, float] = dict.fromkeys(rrf, 0.0)
        for term, idf in term_idfs.items():
            postings = _INDEX.postings.get(term)
            if not postings:
                continue
            for doc_id in postings:
                if doc_id in matched_idf:
                    matched_idf[doc_id] += idf
        scores = {
            doc_id: fused * (1.0 + COVERAGE_WEIGHT * (matched_idf[doc_id] / query_idf_total))
            for doc_id, fused in rrf.items()
        }

    return heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))