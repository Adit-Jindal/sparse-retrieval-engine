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

from submission.indexer import InvertedIndex, Tokenizer

_INDEX: Optional[InvertedIndex] = None
_TOKENIZER: Optional[Tokenizer] = None
_IDF: Dict[str, float] = {}


def build(index: InvertedIndex, tokenizer: Tokenizer) -> None:
    global _INDEX, _TOKENIZER, _IDF
    _INDEX = index
    _TOKENIZER = tokenizer
    _IDF = {}
    
    for term, postings in index.postings.items():
        df = len(postings)
        idf = math.log(((index.N - df + 0.5) / (df + 0.5)) + 1.0)
        _IDF[term] = idf


def score(query: str, k: int, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, BM25-ranked,
    highest score first."""
    if _INDEX is None or _TOKENIZER is None:
        raise RuntimeError("bm25.build() must be called before bm25.score().")
    if k <= 0: return []

    tokens = _TOKENIZER.tokenize(query)
    query_terms = set(tokens)
    scores: Dict[str, float] = {}
    avgdl = _INDEX.avg_doc_len
    
    if avgdl <= 0: return []

    for term in query_terms:
        postings = _INDEX.postings.get(term)
        if not postings: continue
        idf = _IDF.get(term)
        if idf is None: continue

        for doc_id, tf in postings.items():
            doc_length = _INDEX.doc_len[doc_id]
            denominator = (tf + k1 * (1.0 - b + b * (doc_length / avgdl)))
            contribution = (idf * (tf * (k1 + 1.0)) / denominator)
            scores[doc_id] = scores.get(doc_id, 0.0) + contribution

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:k]