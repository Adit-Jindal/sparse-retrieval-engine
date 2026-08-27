"""submission/boolean_vsm.py — Boolean/VSM cosine ranking. IDF is now
cached once per term in build() instead of recomputed inside the O(candidates
x query_terms) scoring loop — same output, no repeated log() calls."""
import heapq
import math
from typing import Dict, List, Tuple, Optional

from submission.indexer import InvertedIndex, Tokenizer

_INDEX: Optional[InvertedIndex] = None
_TOKENIZER: Optional[Tokenizer] = None
_DOC_NORMS: Dict[str, float] = {}
_IDF: Dict[str, float] = {}


def build(index: InvertedIndex, tokenizer: Tokenizer) -> None:
    global _INDEX, _TOKENIZER, _DOC_NORMS, _IDF
    _INDEX = index
    _TOKENIZER = tokenizer
    _DOC_NORMS = {}
    _IDF = {}

    for term, postings in index.postings.items():
        df = len(postings)
        if df == 0 or index.N == 0:
            continue
        idf = math.log(index.N / df)
        _IDF[term] = idf
        for doc_id, tf in postings.items():
            weight = (1 + math.log(tf)) * idf
            _DOC_NORMS[doc_id] = _DOC_NORMS.get(doc_id, 0.0) + weight * weight

    for doc_id in list(_DOC_NORMS.keys()):
        _DOC_NORMS[doc_id] = math.sqrt(_DOC_NORMS[doc_id])


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    if _INDEX is None or _TOKENIZER is None:
        raise RuntimeError("boolean_vsm.build() must be called before vsm_score().")
    if k <= 0:
        return []

    tokens = _TOKENIZER.tokenize(query)
    if not tokens:
        return []

    query_tf: Dict[str, int] = {}
    for term in tokens:
        query_tf[term] = query_tf.get(term, 0) + 1

    query_weights: Dict[str, float] = {}
    for term, tf in query_tf.items():
        idf = _IDF.get(term)
        if idf is None:
            continue
        query_weights[term] = (1 + math.log(tf)) * idf

    if not query_weights:
        return []
    query_norm = math.sqrt(sum(w * w for w in query_weights.values()))

    candidates = set()
    for term in query_weights:
        postings = _INDEX.postings.get(term)
        if postings:
            candidates.update(postings.keys())

    scores: Dict[str, float] = {}
    for doc_id in candidates:
        dot_product = 0.0
        for term, qw in query_weights.items():
            postings = _INDEX.postings.get(term)
            if not postings:
                continue
            tf = postings.get(doc_id)
            if tf is None:
                continue
            dot_product += qw * (1 + math.log(tf)) * _IDF[term]

        doc_norm = _DOC_NORMS.get(doc_id, 0.0)
        if doc_norm > 0.0:
            scores[doc_id] = dot_product / (query_norm * doc_norm)

    return heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))