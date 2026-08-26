"""
submission/boolean_vsm.py

Boolean retrieval and TF-IDF cosine vector-space ranking.
"""

import math
from typing import Dict, List, Tuple, Optional

from submission.indexer import InvertedIndex, Tokenizer


_INDEX: Optional[InvertedIndex] = None
_TOKENIZER: Optional[Tokenizer] = None
# doc_id -> Euclidean norm of its TF-IDF vector
_DOC_NORMS: Dict[str, float] = {}


def build(index: InvertedIndex, tokenizer: Tokenizer) -> None:
    """Precompute document TF-IDF vector norms"""
    global _INDEX, _TOKENIZER, _DOC_NORMS
    _INDEX = index
    _TOKENIZER = tokenizer
    _DOC_NORMS = {}

    for term, postings in index.postings.items():
        df = len(postings)
        if df == 0 or index.N == 0: continue
        idf = math.log(index.N / df)

        for doc_id, tf in postings.items():
            # Use Log-TF (1 + log(tf)) to penalize exceedingly long document term frequencies
            log_tf = 1 + math.log(tf)
            weight = log_tf * idf
            _DOC_NORMS[doc_id] = _DOC_NORMS.get(doc_id, 0.0) + weight * weight

    for doc_id in list(_DOC_NORMS.keys()):
        _DOC_NORMS[doc_id] = math.sqrt(_DOC_NORMS[doc_id])


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    """
    Rank documents using TF-IDF cosine similarity.
    """

    if _INDEX is None or _TOKENIZER is None:
        raise RuntimeError("boolean_vsm.build() must be called before vsm_score().")
    if k <= 0: return []

    tokens = _TOKENIZER.tokenize(query)
    if not tokens: return []

    query_tf: Dict[str, int] = {}
    for term in tokens:
        query_tf[term] = query_tf.get(term, 0) + 1

    # Precompute Query TF-IDF weights exactly once
    query_weights: Dict[str, float] = {}
    for term, tf in query_tf.items():
        df = _INDEX.document_frequency(term)
        if df == 0: continue
        idf = math.log(_INDEX.N / df)
        log_tf = 1 + math.log(tf)
        query_weights[term] = log_tf * idf

    if not query_weights: return []
    query_norm = math.sqrt(sum(w * w for w in query_weights.values()))
    
    candidates = set()
    for term in query_weights:
        postings = _INDEX.postings.get(term)
        if postings:
            candidates.update(postings.keys())

    scores: Dict[str, float] = {}
    for doc_id in candidates:
        dot_product = 0.0
        for term, query_weight in query_weights.items():
            postings = _INDEX.postings.get(term)
            if not postings: continue
            
            tf = postings.get(doc_id)
            if tf is None: continue

            df = _INDEX.document_frequency(term)
            idf = math.log(_INDEX.N / df)
            doc_weight = (1 + math.log(tf)) * idf
            
            dot_product += query_weight * doc_weight

        doc_norm = _DOC_NORMS.get(doc_id, 0.0)
        if doc_norm > 0.0:
            scores[doc_id] = dot_product / (query_norm * doc_norm)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:k]