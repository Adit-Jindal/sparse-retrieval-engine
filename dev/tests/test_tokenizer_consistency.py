import os
import tempfile
from submission.indexer import InvertedIndex, Tokenizer

def test_tokenizer_config_symmetry():
    """Asserts that index_dir correctly persists tokenizer state and reconstructs it."""
    corpus = [("doc1", "The quickly running foxes jumps over the lazy dog!")]
    
    # 1. Build phase config
    tokenizer_build = Tokenizer(use_stopwords=True, use_stemmer=True)
    index_build = InvertedIndex()
    index_build.build(corpus, tokenizer_build)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        index_build.save(tmpdir)
        
        # 2. Load phase config reconstruction
        index_load = InvertedIndex.load(tmpdir)
        config = index_load.tokenizer_config
        tokenizer_load = Tokenizer(use_stopwords=config["stopwords"], use_stemmer=config["stemming"])
        
        # 3. Assert outputs are exactly identical
        query = "Running foxes!"
        assert tokenizer_build.tokenize(query) == tokenizer_load.tokenize(query), "Tokenizers deviated!"