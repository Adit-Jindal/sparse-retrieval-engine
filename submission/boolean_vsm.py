"""
submission/boolean_vsm.py

Boolean retrieval and TF-IDF cosine vector-space ranking.
"""

import math
from typing import Dict, List, Tuple, Optional

from submission.indexer import InvertedIndex, tokenize


_INDEX: Optional[InvertedIndex] = None

# doc_id -> Euclidean norm of its TF-IDF vector
_DOC_NORMS: Dict[str, float] = {}


def build(index: InvertedIndex) -> None:
    """
    Precompute document TF-IDF vector norms.
    """

    global _INDEX, _DOC_NORMS

    _INDEX = index
    _DOC_NORMS = {}

    for term, postings in index.postings.items():
        df = len(postings)

        if df == 0 or index.N == 0:
            continue

        idf = math.log(index.N / df)

        for doc_id, tf in postings.items():
            weight = tf * idf

            old_norm_sq = _DOC_NORMS.get(doc_id, 0.0)
            _DOC_NORMS[doc_id] = old_norm_sq + weight * weight

    # Convert squared norms to actual norms.
    for doc_id in list(_DOC_NORMS.keys()):
        _DOC_NORMS[doc_id] = math.sqrt(_DOC_NORMS[doc_id])


def boolean_search(query: str, mode: str = "and") -> List[str]:
    """
    Return document IDs matching the query using AND or OR semantics.
    """

    if _INDEX is None:
        raise RuntimeError(
            "boolean_vsm.build() must be called before searching."
        )

    if mode not in {"and", "or"}:
        raise ValueError("mode must be either 'and' or 'or'")

    terms = tokenize(query)

    if not terms:
        return []

    # Remove duplicate query terms.
    terms = list(dict.fromkeys(terms))

    postings_sets = []

    for term in terms:
        postings = _INDEX.postings.get(term)

        if postings is None:
            if mode == "and":
                return []
            continue

        postings_sets.append(set(postings.keys()))

    if not postings_sets:
        return []

    if mode == "and":
        result = postings_sets[0].copy()

        for docs in postings_sets[1:]:
            result.intersection_update(docs)

    else:
        result = set()

        for docs in postings_sets:
            result.update(docs)

    return sorted(result)


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    """
    Rank documents using TF-IDF cosine similarity.
    """

    if _INDEX is None:
        raise RuntimeError(
            "boolean_vsm.build() must be called before vsm_score()."
        )

    if k <= 0:
        return []

    tokens = tokenize(query)

    if not tokens:
        return []

    # Query term frequencies.
    query_tf: Dict[str, int] = {}

    for term in tokens:
        query_tf[term] = query_tf.get(term, 0) + 1

    # Build query TF-IDF weights.
    query_weights: Dict[str, float] = {}

    for term, tf in query_tf.items():
        df = _INDEX.document_frequency(term)

        if df == 0:
            continue

        idf = math.log(_INDEX.N / df)

        query_weights[term] = tf * idf

    if not query_weights:
        return []

    query_norm = math.sqrt(
        sum(weight * weight for weight in query_weights.values())
    )

    if query_norm == 0.0:
        return []

    # Candidate documents are the union of the query terms' postings.
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

            if not postings:
                continue

            tf = postings.get(doc_id)

            if tf is None:
                continue

            df = _INDEX.document_frequency(term)
            idf = math.log(_INDEX.N / df)

            doc_weight = tf * idf

            dot_product += query_weight * doc_weight

        doc_norm = _DOC_NORMS.get(doc_id, 0.0)

        if doc_norm == 0.0:
            continue

        scores[doc_id] = dot_product / (query_norm * doc_norm)

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0])
    )

    return ranked[:k]