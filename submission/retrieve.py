"""
submission/retrieve.py — THE REQUIRED COMPETITION ENTRYPOINT.

The grading harness only ever imports and calls the three functions below.
Their names and signatures are fixed by the assignment (Section 5 of the
assignment spec, "Submission Interface & Conformance Checking") — do not
rename them, change their signatures, or move them out of this file.

    build_index(corpus_path: str, index_dir: str) -> None
        Called once, in its own process, with the path to a corpus.jsonl
        file (see data/README.md) and a directory to write your index
        into. Build whatever index and statistics you need, and WRITE
        THEM TO index_dir. The harness runs build_index() and
        load_index()/retrieve() in two SEPARATE processes on purpose (see
        harness/run_harness.py's module docstring) — nothing you only
        hold in memory here survives into load_index(). This call is
        timed as your "index build time" efficiency metric. The harness
        also measures the on-disk byte size of index_dir once this
        returns — that's your "index size" score (assignment Section 7),
        so write only what retrieve() actually needs, and consider
        compressing it.

    load_index(index_dir: str) -> None
        Called once, in a fresh process, before any retrieve() calls.
        Reconstruct everything retrieve() needs by reading index_dir —
        and only index_dir; there is no leftover state from
        build_index() to fall back on. Timed as your "index load time".

    retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]
        Called once per query, only after load_index() has run in the
        same process. Return up to k (doc_id, score) pairs, sorted by
        score descending (highest score = most relevant). This is exactly
        the ranking the harness scores with nDCG@10 / MAP@10. doc_id values
        must be ones that appeared in the corpus passed to build_index().

This file ships with a trivial, fully-working baseline — return the first
k documents in the order build_index() saw them, ignoring the query
entirely — wired up below. It actually persists to disk and reloads
correctly, so it exercises the full build -> disk -> fresh process -> load
-> query path end-to-end from your very first commit. Its scores will be
close to zero; replace the logic, but keep the same
persist-in-build / reconstruct-in-load shape.
"""
import json
import os
from typing import List, Optional, Tuple

from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex, Tokenizer
from submission import bm25, boolean_vsm

# TODO(you): once implemented, import and use your real scorers, e.g.:
# from submission import bm25, boolean_vsm, custom_scorer
# from submission.indexer import InvertedIndex

# ---------------------------------------------------------------------------
# Module-level state. load_index() populates this; retrieve() reads it.
# build_index() runs in a SEPARATE process and cannot rely on this state
# surviving into load_index()/retrieve() — anything needed at query time
# must be written to index_dir in build_index() and read back in
# load_index().
# ---------------------------------------------------------------------------
# _DOC_ORDER: Optional[List[str]] = None  # [doc_id, ...] in the order build_index() saw them

# _DOC_ORDER_FILENAME = "doc_order.json"  # TODO(you): replace with your real index files


_INDEX: Optional[InvertedIndex] = None
_TOKENIZER: Optional[Tokenizer] = None

def build_index(corpus_path: str, index_dir: str) -> None:
    """Load the corpus, build whatever index structures you need, and
    write everything retrieve() will need into `index_dir`.

    Runs once, in its own process, before load_index() ever runs. Heavy
    one-time work — tokenising the whole corpus, building postings lists,
    computing collection statistics — belongs here, not in retrieve(), so
    it doesn't get charged against your per-query latency. Whatever you
    don't write to `index_dir` here does not exist as far as load_index()
    is concerned.
    """
    corpus = load_corpus(corpus_path)
    # Enable aggressive compression and normalization
    tokenizer = Tokenizer(use_stopwords=True, use_stemmer=True)
    
    index = InvertedIndex()
    index.build(corpus, tokenizer)
    index.save(index_dir)


def load_index(index_dir: str) -> None:
    """Reconstruct everything retrieve() needs, reading only from
    `index_dir`. Runs once, in a fresh process, before any retrieve()
    calls — there is no leftover state from build_index() to rely on.
    """
    # global _DOC_ORDER
    global _INDEX, _TOKENIZER
    from submission.indexer import InvertedIndex
    _INDEX = InvertedIndex.load(index_dir)
    # Reconstruct exact tokenizer state from saved config
    config = _INDEX.tokenizer_config
    _TOKENIZER = Tokenizer(use_stopwords=config.get("stopwords", True), 
                           use_stemmer=config.get("stemming", True))
    # Share contexts
    bm25.build(_INDEX, _TOKENIZER)
    boolean_vsm.build(_INDEX, _TOKENIZER)


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, best first."""
    if _INDEX is None or _TOKENIZER is None:
        raise RuntimeError("retrieve() called before load_index().")

    # return boolean_vsm.vsm_score(
    #     query, k
    # )

    return bm25.score(query, k, k1=1.2, b=0.75)