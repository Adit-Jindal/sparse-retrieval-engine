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
import random
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
RRF_K = _env_float("CUSTOM_SCORER_RRF_K", 60)
BM25_WEIGHT = _env_float("CUSTOM_SCORER_BM25_WEIGHT", 0.5)
VSM_WEIGHT = _env_float("CUSTOM_SCORER_VSM_WEIGHT", 0.35)
COVERAGE_WEIGHT = _env_float("CUSTOM_SCORER_COVERAGE_WEIGHT", 0.9)
BM25_K1 = _env_float("CUSTOM_SCORER_BM25_K1", 1.6)
BM25_B = _env_float("CUSTOM_SCORER_BM25_B", 0.5)
USE_VSM_FUSION = _env_bool("CUSTOM_SCORER_USE_VSM_FUSION", False)

USE_PRF = _env_bool("CUSTOM_SCORER_USE_PRF", True)
PRF_TOP_DOCS = int(os.environ.get("CUSTOM_SCORER_PRF_TOP_DOCS", 10))
PRF_TOP_TERMS = int(os.environ.get("CUSTOM_SCORER_PRF_TOP_TERMS", 10))
PRF_ALPHA = _env_float("CUSTOM_SCORER_PRF_ALPHA", 0.6)
PRF_WEIGHT = _env_float("CUSTOM_SCORER_PRF_WEIGHT", 2)
PRF_MAX_DF_RATIO = _env_float("CUSTOM_SCORER_PRF_MAX_DF_RATIO", 0.08)

# --- new, default-off / no-op until swept --------------------------------
FUSION_MODE = os.environ.get("CUSTOM_SCORER_FUSION_MODE", "linear")
PRF_WEIGHT_SEEDS_BY_SCORE = _env_bool("CUSTOM_SCORER_PRF_WEIGHT_SEEDS_BY_SCORE", True)
PRF_ALPHA_SHORT = _env_float("CUSTOM_SCORER_PRF_ALPHA_SHORT", 0.7)
SHORT_QUERY_MAX_TERMS = int(os.environ.get("CUSTOM_SCORER_SHORT_QUERY_MAX_TERMS", 4)) ###### switching off for now
CAPITALIZATION_WEIGHT = _env_float("CUSTOM_SCORER_CAPITALIZATION_WEIGHT", 0.5)
GIST_BOOST_WEIGHT = _env_float("CUSTOM_SCORER_GIST_BOOST_WEIGHT", 0.5)
PROXIMITY_WEIGHT = _env_float("CUSTOM_SCORER_PROXIMITY_WEIGHT", 0.0)
PROXIMITY_POOL = int(os.environ.get("CUSTOM_SCORER_PROXIMITY_POOL", 50))
RERANK_POOL_MULTIPLIER = int(os.environ.get("CUSTOM_SCORER_RERANK_POOL_MULTIPLIER", 20))
SEED_DF_RATIO = _env_float("CUSTOM_SCORER_SEED_DF_RATIO", 0.15)   # terms in <=2% of corpus are "cheap"
USE_SEED_POOLING = _env_bool("CUSTOM_SCORER_USE_SEED_POOLING", True)  # default OFF — full-corpus path is the trusted reference
RANDOM_SAMPLE_RATIO = _env_float("CUSTOM_SCORER_RANDOM_SAMPLE_RATIO", 0.25) # To take some expensive (common) terms to add to the set of rare terms, to improve results

_INDEX: Optional[InvertedIndex] = None
_TOKENIZER: Optional[Tokenizer] = None
_FORWARD_INDEX: Optional[Dict[str, Dict[str, int]]] = None
_IDF: Dict[str, float] = {} 


def _build_forward_index(index: InvertedIndex) -> Dict[str, Dict[str, int]]:
    forward: Dict[str, Dict[str, int]] = {}
    for term, postings in index.postings.items():
        for doc_id, tf in postings.items():
            forward.setdefault(doc_id, {})[term] = tf
    return forward


def build(index: InvertedIndex, tokenizer: Tokenizer) -> None:
    global _INDEX, _TOKENIZER, _FORWARD_INDEX, _IDF
    _INDEX = index
    _TOKENIZER = tokenizer
    _FORWARD_INDEX = _build_forward_index(index) if USE_PRF else None
    _IDF = {
        term: math.log(((index.N - len(postings) + 0.5) / (len(postings) + 0.5)) + 1.0)
        for term, postings in index.postings.items()
    }


def _term_idf(term: str) -> float:
    return _IDF.get(term, 0.0)


def _weighted_bm25_scores(term_weights: Dict[str, float],
                           contrib_cache: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Full postings walk for every term — the trusted, unapproximated
    reference path. Used when USE_SEED_POOLING is off."""
    avgdl = _INDEX.avg_doc_len
    if avgdl <= 0 or not term_weights:
        return {}
    scores: Dict[str, float] = {}
    for term, weight in term_weights.items():
        if weight <= 0:
            continue
        contrib = contrib_cache.get(term)
        if contrib is None:
            postings = _INDEX.postings.get(term)
            if not postings:
                contrib_cache[term] = {}
                continue
            idf = _term_idf(term)
            contrib = {}
            for doc_id, tf in postings.items():
                doc_length = _INDEX.doc_len[doc_id]
                denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_length / avgdl))
                contrib[doc_id] = (idf * tf * (BM25_K1 + 1.0)) / denom
            contrib_cache[term] = contrib
        for doc_id, c in contrib.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * c
    return scores


def _prf_rescore_full(unique_terms: set, bm25_ranked: List[Tuple[str, float]], depth: int,
                       contrib_cache: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
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
    prf_scores = _weighted_bm25_scores(combined_weights, contrib_cache)
    return heapq.nsmallest(depth, prf_scores.items(), key=lambda item: (-item[1], item[0]))

def _split_seed_expensive(term_weights: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    seed, expensive = {}, {}
    for t, w in term_weights.items():
        if w <= 0:
            continue
        df = _INDEX.document_frequency(t)
        if df == 0:
            continue
        (seed if (df / _INDEX.N) <= SEED_DF_RATIO else expensive)[t] = w
    if not seed and expensive:
        rarest = min(expensive, key=lambda t: _INDEX.document_frequency(t))
        seed[rarest] = expensive.pop(rarest)
    return seed, expensive


def _seed_scores(seed_weights: Dict[str, float],
                  contrib_cache: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    avgdl = _INDEX.avg_doc_len
    scores: Dict[str, float] = {}
    if avgdl <= 0:
        return scores
    for term, weight in seed_weights.items():
        contrib = contrib_cache.get(term)
        if contrib is None:
            postings = _INDEX.postings.get(term)
            if not postings:
                contrib_cache[term] = {}
                continue
            idf = _term_idf(term)
            contrib = {}
            for doc_id, tf in postings.items():
                doc_length = _INDEX.doc_len[doc_id]
                denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_length / avgdl))
                contrib[doc_id] = (idf * tf * (BM25_K1 + 1.0)) / denom
            contrib_cache[term] = contrib
        for doc_id, c in contrib.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * c
    return scores


def _scores_within_pool(term_weights: Dict[str, float], pool_ids: List[str],
                         contrib_cache: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    avgdl = _INDEX.avg_doc_len
    scores: Dict[str, float] = {}
    if avgdl <= 0:
        return scores
    for term, weight in term_weights.items():
        if weight <= 0:
            continue
        contrib = contrib_cache.get(term)
        if contrib is not None:
            for doc_id in pool_ids:
                c = contrib.get(doc_id)
                if c is not None:
                    scores[doc_id] = scores.get(doc_id, 0.0) + weight * c
            continue
        postings = _INDEX.postings.get(term)
        if not postings:
            continue
        idf = _term_idf(term)
        for doc_id in pool_ids:
            tf = postings.get(doc_id)
            if tf is None:
                continue
            doc_length = _INDEX.doc_len[doc_id]
            denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_length / avgdl))
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * (idf * tf * (BM25_K1 + 1.0)) / denom
    return scores


def _prf_rescore_pooled(unique_terms: set, seed_docs: List[Tuple[str, float]], pool_ids: List[str],
                         contrib_cache: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    expansion_weights = _expansion_term_weights(seed_docs)

    alpha = PRF_ALPHA_SHORT if len(unique_terms) <= SHORT_QUERY_MAX_TERMS else PRF_ALPHA
    n_orig = len(unique_terms) or 1
    orig_weights = {t: 1.0 / n_orig for t in unique_terms}

    combined_terms = set(orig_weights) | set(expansion_weights)
    combined_weights = {
        t: alpha * orig_weights.get(t, 0.0) + (1.0 - alpha) * expansion_weights.get(t, 0.0)
        for t in combined_terms
    }
    return _scores_within_pool(combined_weights, pool_ids, contrib_cache)


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

def _min_span(term_position_lists: List[List[int]]) -> Optional[int]:
    """Smallest window (inclusive) containing at least one position from
    every list in term_position_lists. Classic k-sorted-lists smallest-
    range problem, solved via merge + sliding window. All input lists
    must be non-empty and individually sorted (guaranteed by build-time
    token-order insertion)."""
    events: List[Tuple[int, int]] = []
    for term_idx, plist in enumerate(term_position_lists):
        for p in plist:
            events.append((p, term_idx))
    events.sort()

    need = len(term_position_lists)
    counts: Dict[int, int] = {}
    have = 0
    left = 0
    best: Optional[int] = None
    for right, (pos_r, term_r) in enumerate(events):
        counts[term_r] = counts.get(term_r, 0) + 1
        if counts[term_r] == 1:
            have += 1
        while have == need:
            span = pos_r - events[left][0]
            if best is None or span < best:
                best = span
            _, term_l = events[left]
            counts[term_l] -= 1
            if counts[term_l] == 0:
                have -= 1
            left += 1
    return best

def _proximity_boost(doc_id: str, unique_terms: set) -> float:
    """0.0 if fewer than 2 matched terms have positions for this doc
    (nothing to measure a span over). Otherwise, more matched terms and
    a tighter span both increase the boost."""
    if not _INDEX.positions:
        return 0.0
    term_lists = []
    for t in unique_terms:
        doc_positions = _INDEX.positions.get(t)
        if doc_positions:
            plist = doc_positions.get(doc_id)
            if plist:
                term_lists.append(plist)
    if len(term_lists) < 2:
        return 0.0
    span = _min_span(term_lists)
    if span is None:
        return 0.0
    return len(term_lists) / (1.0 + span)

def _boost_matches(candidate_ids, unique_terms: set,
                    weight_fns: Dict[str, Callable[[str], float]],
                    postings_fns: Dict[str, Callable[[str], Optional[Dict[str, int]]]]
                    ) -> Dict[str, Tuple[Dict[str, float], float]]:
    matched: Dict[str, Dict[str, float]] = {name: {} for name in weight_fns}
    totals: Dict[str, float] = {}

    per_boost_term_weights: Dict[str, Dict[str, float]] = {}
    per_boost_postings: Dict[str, Dict[str, Optional[Dict[str, int]]]] = {}
    for name, weight_fn in weight_fns.items():
        term_weights = {t: weight_fn(t) for t in unique_terms if weight_fn(t) > 0}
        per_boost_term_weights[name] = term_weights
        totals[name] = sum(term_weights.values())
        per_boost_postings[name] = {t: postings_fns[name](t) for t in term_weights}

    for doc_id in candidate_ids:
        for name, term_weights in per_boost_term_weights.items():
            postings_map = per_boost_postings[name]
            acc = 0.0
            for term, w in term_weights.items():
                postings = postings_map.get(term)
                if postings and doc_id in postings:
                    acc += w
            matched[name][doc_id] = acc

    return {name: (matched[name], totals[name]) for name in weight_fns}

def score(query: str, k: int) -> List[Tuple[str, float]]:
    if _INDEX is None or _TOKENIZER is None:
        raise RuntimeError("custom_scorer.build() must be called before score().")
    if k <= 0:
        return []

    tokens = _TOKENIZER.tokenize(query)
    unique_terms = set(tokens)
    if not unique_terms:
        return []

    contrib_cache: Dict[str, Dict[str, float]] = {}

    if USE_SEED_POOLING:
        seed_weights, expensive_weights = _split_seed_expensive({t: 1.0 for t in unique_terms})
        # --- NEW: Randomly sample half of the common (expensive) terms ---
        expensive_keys = list(expensive_weights.keys())
        num_to_move: int = int(RANDOM_SAMPLE_RATIO * len(expensive_keys))
        
        if num_to_move > 0:
            sampled_keys = random.sample(expensive_keys, num_to_move)
            for k_term in sampled_keys:
                # Remove from expensive and add to seed
                seed_weights[k_term] = expensive_weights.pop(k_term)
        # -----------------------------------------------------------------
        partial_scores = _seed_scores(seed_weights, contrib_cache)
        pool_target = RERANK_POOL_MULTIPLIER * k
        pool_ids = [doc_id for doc_id, _ in
                    heapq.nsmallest(pool_target, partial_scores.items(), key=lambda item: (-item[1], item[0]))]

        all_query_weights = {**seed_weights, **expensive_weights}
        bm25_full = _scores_within_pool(all_query_weights, pool_ids, contrib_cache)
        bm25_ranked = heapq.nsmallest(pool_target, bm25_full.items(), key=lambda item: (-item[1], item[0]))
        depth = pool_target
    else:
        depth = k
        bm25_full = _weighted_bm25_scores({t: 1.0 for t in unique_terms}, contrib_cache)
        bm25_ranked = heapq.nsmallest(depth, bm25_full.items(), key=lambda item: (-item[1], item[0]))

    lists_with_weights: List[Tuple[List[Tuple[str, float]], float]] = [(bm25_ranked, BM25_WEIGHT)]

    if USE_VSM_FUSION:
        vsm_ranked = boolean_vsm.vsm_score(query, depth)
        lists_with_weights.append((vsm_ranked, VSM_WEIGHT))

    if USE_PRF:
        if USE_SEED_POOLING:
            prf_scores = _prf_rescore_pooled(unique_terms, bm25_ranked[:PRF_TOP_DOCS], pool_ids, contrib_cache)
            prf_ranked = heapq.nsmallest(depth, prf_scores.items(), key=lambda item: (-item[1], item[0]))
        else:
            prf_ranked = _prf_rescore_full(unique_terms, bm25_ranked, depth, contrib_cache)
        lists_with_weights.append((prf_ranked, PRF_WEIGHT))

    fused = _fuse_linear(lists_with_weights) if FUSION_MODE == "linear" else _fuse_rrf(lists_with_weights)
    if not fused:
        return []

    scores = dict(fused)

    # --- boost block: unchanged, pool-bound _boost_matches from the A3/pool fix ---
    weight_fns: Dict[str, Callable[[str], float]] = {}
    postings_fns: Dict[str, Callable[[str], Optional[Dict[str, int]]]] = {}
    if COVERAGE_WEIGHT > 0:
        weight_fns["coverage"] = _term_idf
        postings_fns["coverage"] = _INDEX.postings.get
    if CAPITALIZATION_WEIGHT > 0 and _INDEX.cap_score:
        weight_fns["cap"] = lambda t: _term_idf(t) * _INDEX.cap_score.get(t, 0.0)
        postings_fns["cap"] = _INDEX.postings.get
    if GIST_BOOST_WEIGHT > 0 and _INDEX.gist_postings:
        weight_fns["gist"] = _term_idf
        postings_fns["gist"] = _INDEX.gist_postings.get

    if weight_fns:
        results = _boost_matches(scores.keys(), unique_terms, weight_fns, postings_fns)
        if "coverage" in results:
            matched, total = results["coverage"]
            if total > 0:
                for doc_id in scores:
                    scores[doc_id] *= (1.0 + COVERAGE_WEIGHT * (matched.get(doc_id, 0.0) / total))
        if "cap" in results:
            matched, total = results["cap"]
            if total > 0:
                for doc_id in scores:
                    scores[doc_id] *= (1.0 + CAPITALIZATION_WEIGHT * (matched.get(doc_id, 0.0) / total))
        if "gist" in results:
            matched, total = results["gist"]
            if total > 0:
                for doc_id in scores:
                    scores[doc_id] *= (1.0 + GIST_BOOST_WEIGHT * (matched.get(doc_id, 0.0) / total))

    if PROXIMITY_WEIGHT > 0 and _INDEX.positions:
        boosts = {doc_id: _proximity_boost(doc_id, unique_terms) for doc_id in scores}
        max_boost = max(boosts.values(), default=0.0)
        if max_boost > 0:
            for doc_id, raw_boost in boosts.items():
                scores[doc_id] *= (1.0 + PROXIMITY_WEIGHT * (raw_boost / max_boost))

    return heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))