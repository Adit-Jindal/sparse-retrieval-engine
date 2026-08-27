"""submission/bm25.py — Okapi BM25 ranking. Formula/docstring unchanged
from the original; only the top-k selection is now heap-based instead of
a full sort, which is a pure speed win with identical output."""
from typing import List, Tuple, Dict, Optional
import heapq
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
        _IDF[term] = math.log(((index.N - df + 0.5) / (df + 0.5)) + 1.0)


def score(query: str, k: int, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
    if _INDEX is None or _TOKENIZER is None:
        raise RuntimeError("bm25.build() must be called before bm25.score().")
    if k <= 0:
        return []

    tokens = _TOKENIZER.tokenize(query)
    query_terms = set(tokens)
    scores: Dict[str, float] = {}
    avgdl = _INDEX.avg_doc_len
    if avgdl <= 0:
        return []

    for term in query_terms:
        postings = _INDEX.postings.get(term)
        if not postings:
            continue
        idf = _IDF.get(term)
        if idf is None:
            continue
        for doc_id, tf in postings.items():
            doc_length = _INDEX.doc_len[doc_id]
            denom = tf + k1 * (1.0 - b + b * (doc_length / avgdl))
            scores[doc_id] = scores.get(doc_id, 0.0) + (idf * tf * (k1 + 1.0)) / denom

    # heapq.nsmallest on (-score, doc_id) == sorted(...)[:k] on the same
    # key, just O(m log k) instead of O(m log m) — same top-k, same
    # tie-break, faster when len(scores) >> k.
    return heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))