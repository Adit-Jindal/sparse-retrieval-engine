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
    use_pseudo_gist = _env_bool("INDEXER_USE_PSEUDO_GIST", True)
    use_positions = _env_bool("INDEXER_USE_POSITIONS", False)
    gist_top_k = _env_bool("INDEXER_GIST_TOP_K", 15)
    gist_lead_boost_weight = float(os.environ.get("INDEXER_GIST_LEAD_BOOST_WEIGHT", 0.0))
    gist_lead_decay_frac = float(os.environ.get("INDEXER_GIST_LEAD_DECAY_FRAC", 0.1))

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
        use_document_gist=use_pseudo_gist,
        gist_top_k=gist_top_k,
        gist_lead_boost_weight=gist_lead_boost_weight,
        gist_lead_decay_frac=gist_lead_decay_frac,
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
    custom_scorer.build(_INDEX, _TOKENIZER)


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    if _INDEX is None or _TOKENIZER is None:
        raise RuntimeError("retrieve() called before load_index().")
    return custom_scorer.score(query, k)