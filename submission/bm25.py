"""submission/bm25.py — Okapi BM25 ranking. Kept as a stable, untouched,
independently-testable reference scorer; all new experimental behavior
lives in custom_scorer.py, not here."""
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


def score(query: str, k: int, k1: float = 1.6, b: float = 0.5) -> List[Tuple[str, float]]:
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

    return heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))