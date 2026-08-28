"""submission/retrieve.py — required competition entrypoint. Indexing-time
feature flags (compounds, bigrams, corpus stopwords, capitalization,
pseudo-title) are read from env at build_index() time and baked into
tokenizer_config / the index itself, so load_index() reconstructs the
exact same tokenizer with zero guessing — build_index()/load_index() run
as separate subprocesses (harness/run_harness.py), so only what's
persisted to index_dir is trustworthy at query time."""
import os
from typing import List, Optional, Tuple

from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex, Tokenizer
from submission import bm25, boolean_vsm, custom_scorer

_INDEX: Optional[InvertedIndex] = None
_TOKENIZER: Optional[Tokenizer] = None


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def build_index(corpus_path: str, index_dir: str) -> None:
    corpus = load_corpus(corpus_path)

    stopword_set = os.environ.get("INDEXER_STOPWORD_SET", "standard")
    preserve_compounds = _env_bool("INDEXER_PRESERVE_COMPOUNDS", False)
    use_bigrams = _env_bool("INDEXER_USE_BIGRAMS", False)
    use_corpus_stopwords = _env_bool("INDEXER_USE_CORPUS_STOPWORDS", True)
    corpus_stopword_df_ratio = float(os.environ.get("INDEXER_CORPUS_STOPWORD_DF_RATIO", 0.9))
    use_capitalization = _env_bool("INDEXER_USE_CAPITALIZATION", True)
    use_pseudo_title = _env_bool("INDEXER_USE_PSEUDO_TITLE", True)
    use_positions = _env_bool("INDEXER_USE_POSITIONS", False)

    tokenizer = Tokenizer(
        use_stopwords=True, use_stemmer=True, stopword_set=stopword_set,
        preserve_compounds=preserve_compounds, use_bigrams=use_bigrams,
    )
    index = InvertedIndex()
    index.build(
        corpus, tokenizer,
        use_corpus_stopwords=use_corpus_stopwords,
        corpus_stopword_df_ratio=corpus_stopword_df_ratio,
        use_capitalization=use_capitalization,
        use_pseudo_title=use_pseudo_title,
        use_positions=use_positions,
    )
    index.save(index_dir)


def load_index(index_dir: str) -> None:
    global _INDEX, _TOKENIZER
    _INDEX = InvertedIndex.load(index_dir)
    config = _INDEX.tokenizer_config
    _TOKENIZER = Tokenizer(
        use_stopwords=config.get("stopwords", True),
        use_stemmer=config.get("stemming", True),
        stopword_set=config.get("stopword_set", "standard"),
        preserve_compounds=config.get("preserve_compounds", False),
        use_bigrams=config.get("use_bigrams", False),
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
        # phase on any uncaught exception from retrieve(). Plain BM25,
        # unconditionally trusted, is the floor.
        return bm25.score(query, k, k1=custom_scorer.BM25_K1, b=custom_scorer.BM25_B)
        # return boolean_vsm.vsm_score(query, k)