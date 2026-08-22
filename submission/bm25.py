"""
submission/bm25.py — Okapi BM25 ranking.

Required component (assignment Section 4.1): "a BM25 implementation with
tunable k1 and b." See the assignment background (Section 3) for the
Robertson & Walker / Robertson & Zaragoza references this is based on.

BM25 score for a query Q = q1...qn against document D:

    score(D, Q) = sum_i  IDF(qi) * ( tf(qi, D) * (k1 + 1) )
                                   / ( tf(qi, D) + k1 * (1 - b + b * |D| / avgdl) )

A standard IDF variant (Robertson-Sparck Jones, +1-smoothed so it stays
non-negative even for terms occurring in more than half the corpus):

    IDF(qi) = ln( (N - df(qi) + 0.5) / (df(qi) + 0.5) + 1 )

where:
    N        = number of documents in the corpus
    df(qi)   = number of documents containing qi
    tf(qi,D) = term frequency of qi in D
    |D|      = length of D in tokens
    avgdl    = average document length across the corpus

k1 (typically 1.2-2.0) controls term-frequency saturation; b (in [0, 1])
controls document-length normalisation strength. Both must be exposed as
parameters, not hard-coded — you need to sweep them for your report
(assignment Section 8, "parameter search procedure for k1, b").
"""
from typing import List, Tuple, Dict, Optional
import math

from submission.indexer import InvertedIndex, tokenize

_INDEX: Optional[InvertedIndex] = None
_IDF: Dict[str, float] = {}


def build(index: InvertedIndex) -> None:
    """Optional: precompute anything BM25-specific (e.g. cached IDF values
    per term) from the InvertedIndex built in indexer.py.

    Call this from retrieve.load_index(), not retrieve.build_index() —
    the harness runs those two in separate processes, so any cache this
    creates only needs to exist in the process that also calls
    retrieve(). If you want a precomputed cache to persist across the
    build/load boundary too, write it out via InvertedIndex.save() instead
    (it then counts toward your index-size score) and rebuild the cache
    here from the loaded index."""
    global _INDEX, _IDF

    _INDEX = index
    _IDF = {}

    for term, postings in index.postings.items():
        df = len(postings)

        # +1-smoothed Robertson/Sparck Jones IDF.
        idf = math.log(
            ((index.N - df + 0.5) / (df + 0.5)) + 1.0
        )

        _IDF[term] = idf


def score(query: str, k: int, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, BM25-ranked,
    highest score first."""
    if _INDEX is None:
        raise RuntimeError("bm25.build() must be called before bm25.score().")

    if k <= 0:
        return []

    tokens = tokenize(query)

    # BM25's scoring formula is term-based. Using unique query terms
    # avoids accidentally counting the same query term multiple times.
    query_terms = set(tokens)

    scores: Dict[str, float] = {}

    avgdl = _INDEX.avg_doc_len

    if avgdl <= 0:
        return []

    for term in query_terms:
        postings = _INDEX.postings.get(term)

        # Unknown query term.
        if not postings:
            continue

        idf = _IDF.get(term)

        if idf is None:
            continue

        for doc_id, tf in postings.items():
            doc_length = _INDEX.doc_len[doc_id]

            denominator = (
                tf
                + k1 * (
                    1.0
                    - b
                    + b * (doc_length / avgdl)
                )
            )

            contribution = (
                idf
                * (tf * (k1 + 1.0))
                / denominator
            )

            scores[doc_id] = scores.get(doc_id, 0.0) + contribution

    # Deterministic tie-breaking by doc_id.
    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0])
    )

    return ranked[:k]

