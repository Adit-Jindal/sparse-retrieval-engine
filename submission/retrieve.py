"""submission/retrieve.py — required competition entrypoint. See original
docstring for the full three-function contract; unchanged here."""
import json
import os
from typing import List, Optional, Tuple

from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex, Tokenizer
from submission import bm25, boolean_vsm, custom_scorer

_INDEX: Optional[InvertedIndex] = None
_TOKENIZER: Optional[Tokenizer] = None


def build_index(corpus_path: str, index_dir: str) -> None:
    corpus = load_corpus(corpus_path)
    tokenizer = Tokenizer(use_stopwords=True, use_stemmer=True)
    index = InvertedIndex()
    index.build(corpus, tokenizer)
    index.save(index_dir)


def load_index(index_dir: str) -> None:
    global _INDEX, _TOKENIZER
    _INDEX = InvertedIndex.load(index_dir)
    config = _INDEX.tokenizer_config
    _TOKENIZER = Tokenizer(
        use_stopwords=config.get("stopwords", True),
        use_stemmer=config.get("stemming", True),
    )
    bm25.build(_INDEX, _TOKENIZER)
    boolean_vsm.build(_INDEX, _TOKENIZER)
    custom_scorer.build(_INDEX, _TOKENIZER)


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    if _INDEX is None or _TOKENIZER is None:
        raise RuntimeError("retrieve() called before load_index().")
    try:
        return custom_scorer.score(query, k)
    except Exception:
        # Defensive fallback: run_harness.py aborts the ENTIRE query
        # phase (every query, not just this one) on any uncaught
        # exception from retrieve(). A single edge-case query on the
        # held-out set must never zero out the whole run. Plain,
        # unconditionally-trusted BM25 is the floor we fall back to.
        # print("..........................\n...........\n.......")
        return bm25.score(query, k, k1=custom_scorer.BM25_K1, b=custom_scorer.BM25_B)