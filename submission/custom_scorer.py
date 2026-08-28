"""
submission/custom_scorer.py — BM25 backbone + RM3-style PRF + fusion,
plus optional (flag-gated, default no-op) refinements:

  CUSTOM_SCORER_FUSION_MODE            "rrf" (default) or "linear"
      rrf: rank-reciprocal fusion (established, tuned).
      linear: min-max-normalise each list's raw scores, then weighted
      sum — an alternative that preserves score magnitude instead of
      discarding it to rank position. Worth A/B-ing against rrf now
      that rrf's own weights are settled.

  CUSTOM_SCORER_PRF_WEIGHT_SEEDS_BY_SCORE  0 (default) / 1
      When on, PRF's expansion-term mining weights each of the
      PRF_TOP_DOCS seed documents by its normalised first-pass BM25
      score instead of trusting all seeds equally — a standard
      mitigation for query drift (a low-ranked, borderline-relevant
      seed doc contributes less to the expansion vocabulary than the
      top-ranked one).

  CUSTOM_SCORER_PRF_ALPHA_SHORT / CUSTOM_SCORER_SHORT_QUERY_MAX_TERMS
      Optional separate PRF_ALPHA for queries with <= max_terms unique
      terms after tokenization. Defaults to PRF_ALPHA itself (no-op)
      until swept.

  CUSTOM_SCORER_CAPITALIZATION_WEIGHT / CUSTOM_SCORER_TITLE_BOOST_WEIGHT
      Multiplicative post-fusion boosts using indexer.py's optional
      cap_score / title_postings structures (only present if the index
      was built with use_capitalization / use_pseudo_title on — see
      indexer.py). Both default to 0 (no-op); non-zero values are
      silently equivalent to 0 if the index wasn't built with the
      corresponding feature, since cap_score/title_postings will just
      be empty.

Frozen-established defaults (from prior sweeps, unchanged):
  BM25_K1=1.6, BM25_B=0.5, USE_VSM_FUSION=False, COVERAGE_WEIGHT=0,
  USE_PRF=True, PRF_TOP_DOCS=10, PRF_TOP_TERMS=10, PRF_ALPHA=0.5,
  PRF_WEIGHT=1.0, PRF_MAX_DF_RATIO=0.15, RRF_K=60.
"""
import heapq
import math
import os
from typing import Callable, Dict, List, Optional, Tuple

from submission.indexer import InvertedIndex, Tokenizer
from submission import bm25, boolean_vsm


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# --- established / frozen ------------------------------------------------
DEPTH = int(os.environ.get("CUSTOM_SCORER_DEPTH", 100))
RRF_K = _env_float("CUSTOM_SCORER_RRF_K", 60)
BM25_WEIGHT = _env_float("CUSTOM_SCORER_BM25_WEIGHT", 0.65)
VSM_WEIGHT = _env_float("CUSTOM_SCORER_VSM_WEIGHT", 0.35)
COVERAGE_WEIGHT = _env_float("CUSTOM_SCORER_COVERAGE_WEIGHT", 0.0)
BM25_K1 = _env_float("CUSTOM_SCORER_BM25_K1", 1.6)
BM25_B = _env_float("CUSTOM_SCORER_BM25_B", 0.5)
USE_VSM_FUSION = _env_bool("CUSTOM_SCORER_USE_VSM_FUSION", False)

USE_PRF = _env_bool("CUSTOM_SCORER_USE_PRF", True)
PRF_TOP_DOCS = int(os.environ.get("CUSTOM_SCORER_PRF_TOP_DOCS", 10))
PRF_TOP_TERMS = int(os.environ.get("CUSTOM_SCORER_PRF_TOP_TERMS", 10))
PRF_ALPHA = _env_float("CUSTOM_SCORER_PRF_ALPHA", 0.5)
PRF_WEIGHT = _env_float("CUSTOM_SCORER_PRF_WEIGHT", 1.0)
PRF_MAX_DF_RATIO = _env_float("CUSTOM_SCORER_PRF_MAX_DF_RATIO", 0.15)

# --- new, default-off / no-op until swept --------------------------------
FUSION_MODE = os.environ.get("CUSTOM_SCORER_FUSION_MODE", "rrf")
PRF_WEIGHT_SEEDS_BY_SCORE = _env_bool("CUSTOM_SCORER_PRF_WEIGHT_SEEDS_BY_SCORE", True)
PRF_ALPHA_SHORT = _env_float("CUSTOM_SCORER_PRF_ALPHA_SHORT", 0.7)
SHORT_QUERY_MAX_TERMS = int(os.environ.get("CUSTOM_SCORER_SHORT_QUERY_MAX_TERMS", 3))
CAPITALIZATION_WEIGHT = _env_float("CUSTOM_SCORER_CAPITALIZATION_WEIGHT", 0.5)
TITLE_BOOST_WEIGHT = _env_float("CUSTOM_SCORER_TITLE_BOOST_WEIGHT", 0.5)

_INDEX: Optional[InvertedIndex] = None
_TOKENIZER: Optional[Tokenizer] = None
_FORWARD_INDEX: Optional[Dict[str, Dict[str, int]]] = None


def _build_forward_index(index: InvertedIndex) -> Dict[str, Dict[str, int]]:
    forward: Dict[str, Dict[str, int]] = {}
    for term, postings in index.postings.items():
        for doc_id, tf in postings.items():
            forward.setdefault(doc_id, {})[term] = tf
    return forward


def build(index: InvertedIndex, tokenizer: Tokenizer) -> None:
    global _INDEX, _TOKENIZER, _FORWARD_INDEX
    _INDEX = index
    _TOKENIZER = tokenizer
    _FORWARD_INDEX = _build_forward_index(index) if USE_PRF else None


def _term_idf(term: str) -> float:
    df = _INDEX.document_frequency(term)
    return math.log(((_INDEX.N - df + 0.5) / (df + 0.5)) + 1.0)


def _weighted_bm25_scores(term_weights: Dict[str, float]) -> Dict[str, float]:
    avgdl = _INDEX.avg_doc_len
    if avgdl <= 0 or not term_weights:
        return {}
    scores: Dict[str, float] = {}
    for term, weight in term_weights.items():
        if weight <= 0:
            continue
        postings = _INDEX.postings.get(term)
        if not postings:
            continue
        idf = _term_idf(term)
        for doc_id, tf in postings.items():
            doc_length = _INDEX.doc_len[doc_id]
            denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_length / avgdl))
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * (idf * tf * (BM25_K1 + 1.0)) / denom
    return scores


def _expansion_term_weights(seed_docs_with_scores: List[Tuple[str, float]]) -> Dict[str, float]:
    """Mine expansion terms from seed docs' vocabulary. If
    PRF_WEIGHT_SEEDS_BY_SCORE is on, each seed's contribution is scaled
    by its (min-max normalised, floored at 0.5 so the weakest seed still
    contributes) first-pass BM25 score — mitigates query drift from
    weak/borderline seeds. Off by default: all seeds weighted equally,
    matching the already-validated behavior."""
    if not _FORWARD_INDEX or not seed_docs_with_scores:
        return {}

    if PRF_WEIGHT_SEEDS_BY_SCORE:
        raw_scores = [s for _, s in seed_docs_with_scores]
        lo, hi = min(raw_scores), max(raw_scores)
        span = (hi - lo) or 1.0
        seed_weight = {doc_id: 0.5 + 0.5 * (s - lo) / span for doc_id, s in seed_docs_with_scores}
    else:
        seed_weight = {doc_id: 1.0 for doc_id, _ in seed_docs_with_scores}

    term_weight: Dict[str, float] = {}
    for doc_id, sw in seed_weight.items():
        doc_len = _INDEX.doc_len.get(doc_id, 0)
        if doc_len == 0:
            continue
        doc_terms = _FORWARD_INDEX.get(doc_id)
        if not doc_terms:
            continue
        for term, tf in doc_terms.items():
            df = _INDEX.document_frequency(term)
            if df == 0 or (df / _INDEX.N) > PRF_MAX_DF_RATIO:
                continue
            term_weight[term] = term_weight.get(term, 0.0) + sw * (tf / doc_len) * _term_idf(term)

    if not term_weight:
        return {}
    top_terms = heapq.nlargest(PRF_TOP_TERMS, term_weight.items(), key=lambda x: x[1])
    total = sum(w for _, w in top_terms) or 1.0
    return {t: w / total for t, w in top_terms}


def _prf_rescore(unique_terms: set, bm25_ranked: List[Tuple[str, float]], depth: int) -> List[Tuple[str, float]]:
    seed_docs = bm25_ranked[:PRF_TOP_DOCS]
    expansion_weights = _expansion_term_weights(seed_docs)

    alpha = PRF_ALPHA_SHORT if len(unique_terms) <= SHORT_QUERY_MAX_TERMS else PRF_ALPHA

    n_orig = len(unique_terms) or 1
    orig_weights = {t: 1.0 / n_orig for t in unique_terms}

    combined_terms = set(orig_weights) | set(expansion_weights)
    combined_weights = {
        t: alpha * orig_weights.get(t, 0.0) + (1.0 - alpha) * expansion_weights.get(t, 0.0)
        for t in combined_terms
    }

    prf_scores = _weighted_bm25_scores(combined_weights)
    return heapq.nsmallest(depth, prf_scores.items(), key=lambda item: (-item[1], item[0]))


def _fuse_rrf(lists_with_weights: List[Tuple[List[Tuple[str, float]], float]]) -> Dict[str, float]:
    fused: Dict[str, float] = {}
    for ranked, weight in lists_with_weights:
        for rank, (doc_id, _s) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (RRF_K + rank)
    return fused


def _fuse_linear(lists_with_weights: List[Tuple[List[Tuple[str, float]], float]]) -> Dict[str, float]:
    """Min-max normalise each list's raw scores to [0,1], then take a
    weighted sum — preserves score magnitude within each list, unlike
    RRF which only sees rank position."""
    fused: Dict[str, float] = {}
    for ranked, weight in lists_with_weights:
        if not ranked:
            continue
        vals = [s for _, s in ranked]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        for doc_id, s in ranked:
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * ((s - lo) / span)
    return fused


def _match_fraction(candidate_ids, unique_terms: set, weight_fn: Callable[[str], float],
                     postings_fn: Callable[[str], Optional[Dict[str, int]]]) -> Tuple[Dict[str, float], float]:
    """Generic helper: for each query term, look up its weight and its
    (possibly restricted) postings, and accumulate per-candidate matched
    weight. Used for coverage, capitalization, and title boosts — same
    shape, different weight_fn/postings_fn."""
    weights = {t: weight_fn(t) for t in unique_terms}
    total = sum(w for w in weights.values() if w > 0)
    if total <= 0:
        return {}, 0.0
    matched: Dict[str, float] = dict.fromkeys(candidate_ids, 0.0)
    for term, w in weights.items():
        if w <= 0:
            continue
        postings = postings_fn(term)
        if not postings:
            continue
        for doc_id in postings:
            if doc_id in matched:
                matched[doc_id] += w
    return matched, total


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

    lists_with_weights: List[Tuple[List[Tuple[str, float]], float]] = [(bm25_ranked, BM25_WEIGHT)]

    if USE_VSM_FUSION:
        vsm_ranked = boolean_vsm.vsm_score(query, depth)
        lists_with_weights.append((vsm_ranked, VSM_WEIGHT))

    if USE_PRF and bm25_ranked:
        prf_ranked = _prf_rescore(unique_terms, bm25_ranked, depth)
        lists_with_weights.append((prf_ranked, PRF_WEIGHT))

    fused = _fuse_linear(lists_with_weights) if FUSION_MODE == "linear" else _fuse_rrf(lists_with_weights)
    if not fused:
        return []

    scores = dict(fused)

    if COVERAGE_WEIGHT > 0:
        matched, total = _match_fraction(scores.keys(), unique_terms, _term_idf, _INDEX.postings.get)
        if total > 0:
            for doc_id in scores:
                scores[doc_id] *= (1.0 + COVERAGE_WEIGHT * (matched.get(doc_id, 0.0) / total))

    if CAPITALIZATION_WEIGHT > 0 and _INDEX.cap_score:
        matched, total = _match_fraction(
            scores.keys(), unique_terms,
            lambda t: _term_idf(t) * _INDEX.cap_score.get(t, 0.0),
            _INDEX.postings.get,
        )
        if total > 0:
            for doc_id in scores:
                scores[doc_id] *= (1.0 + CAPITALIZATION_WEIGHT * (matched.get(doc_id, 0.0) / total))

    if TITLE_BOOST_WEIGHT > 0 and _INDEX.title_postings:
        matched, total = _match_fraction(scores.keys(), unique_terms, _term_idf, _INDEX.title_postings.get)
        if total > 0:
            for doc_id in scores:
                scores[doc_id] *= (1.0 + TITLE_BOOST_WEIGHT * (matched.get(doc_id, 0.0) / total))

    return heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))