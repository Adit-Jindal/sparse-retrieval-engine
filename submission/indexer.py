"""
submission/indexer.py — build your inverted index here.

This is one of the required components (assignment Section 4.1): you must
build the inverted index yourself, without an existing search/indexing
library (Lucene, Elasticsearch, Pyserini, Whoosh, etc.).

A `tokenize()` helper is provided below purely so that tokenization is
consistent across your Boolean/VSM and BM25 scorers —
feel free to replace it (e.g. add stemming or stopword removal), just make
sure every scorer that reads this index was built with the same tokenizer.

Everything else — the postings representation, what per-document and
collection statistics you track, whether you add positions for
proximity/phrase features — is your design decision. `InvertedIndex`
below sketches a minimal, obviously-sufficient shape; you do not have to
use it, but if you do, filling in `build()` and `document_frequency()` is
enough to support Boolean/VSM and BM25.

Persistence (assignment Section 4.1 / Section 7 "index size" scoring):
`build_index()` in retrieve.py runs in one process and `load_index()` runs
in a separate, later one — so whatever this index needs at query time must
round-trip through `save()`/`load()` below, not just live as Python
attributes. The on-disk byte size of what `save()` writes is graded
directly (smaller, relative to the class median, scores better), so a
compact postings encoding is worth more here than in most course
assignments — see the `save()` docstring for concrete starting points.
"""
import os
import pickle
import re
import gzip
from functools import lru_cache
from typing import Dict, List, Tuple, Any

# Hardcoded small stopword list to avoid NLTK download dependencies at grading time.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", 
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", 
    "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")

class Tokenizer:
    def __init__(self, use_stopwords: bool = True, use_stemmer: bool = True):
        self.use_stopwords = use_stopwords
        self.use_stemmer = use_stemmer
        self.config = {"stopwords": use_stopwords, "stemming": use_stemmer}
        
        if self.use_stemmer:
            from nltk.stem import PorterStemmer
            stemmer = PorterStemmer()
            # Cache stems to avoid redundant processing of common words
            @lru_cache(maxsize=None)
            def _stem(token: str) -> str:
                return stemmer.stem(token)
            self._stem = _stem

    def tokenize(self, text: str) -> List[str]:
        """Lowercase, alphanumeric-only tokenization."""
        tokens = _TOKEN_RE.findall(text.lower())
        if self.use_stopwords:
            tokens = [t for t in tokens if t not in _STOPWORDS]
        if self.use_stemmer:
            tokens = [self._stem(t) for t in tokens]
        return tokens

# --- Minimal Varint Encoder/Decoder ---
def encode_varint(n: int, out: bytearray) -> None:
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return

def decode_varint_stream(data: bytes) -> List[int]:
    ints = []
    i = 0
    while i < len(data):
        n = 0
        shift = 0
        while True:
            b = data[i]
            i += 1
            n |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        ints.append(n)
    return ints
# --------------------------------------


class InvertedIndex:
    def __init__(self):
        # In-memory representation: term -> {doc_id_str: tf}
        self.postings: Dict[str, Dict[str, int]] = {}
        self.doc_len: Dict[str, int] = {}
        self.N: int = 0
        self.avg_doc_len: float = 0.0
        self.tokenizer_config: Dict[str, Any] = {}

    def build(self, corpus: List[Tuple[str, str]], tokenizer: Tokenizer) -> None:
        """corpus: list of (doc_id, text) pairs, e.g. from
        submission.corpus_utils.load_corpus().
        """
        self.tokenizer_config = tokenizer.config
        self.doc_len = {}
        
        # Temporary structure for building
        temp_postings: Dict[str, Dict[str, int]] = {}
        
        for doc_id, text in corpus:
            tokens = tokenizer.tokenize(text)
            self.doc_len[doc_id] = len(tokens)
            
            term_counts: Dict[str, int] = {}
            for term in tokens:
                term_counts[term] = term_counts.get(term, 0) + 1
                
            for term, tf in term_counts.items():
                if term not in temp_postings:
                    temp_postings[term] = {}
                temp_postings[term][doc_id] = tf
                
        self.postings = temp_postings
        self.N = len(self.doc_len)
        self.avg_doc_len = sum(self.doc_len.values()) / self.N if self.N > 0 else 0.0

    def document_frequency(self, term: str) -> int:
        """
        Number of documents containing `term` at least once.
        """
        postings = self.postings.get(term)
        return len(postings) if postings else 0

    def save(self, index_dir: str) -> None:
        """Compresses postings using delta-encoding, varint packing, and Gzip."""
        os.makedirs(index_dir, exist_ok=True)
        
        # 1. Map string doc_ids to integers to save space
        doc_id_to_int = {doc_id: i for i, doc_id in enumerate(self.doc_len.keys())}
        int_to_doc_id = list(self.doc_len.keys())
        
        packed_postings: Dict[str, bytes] = {}
        
        for term, doc_tfs in self.postings.items():
            out = bytearray()
            # Sort integer IDs for delta encoding
            sorted_docs = sorted([(doc_id_to_int[doc_id], tf) for doc_id, tf in doc_tfs.items()])
            
            last_doc_int = 0
            for doc_int, tf in sorted_docs:
                delta = doc_int - last_doc_int
                encode_varint(delta, out)
                encode_varint(tf, out)
                last_doc_int = doc_int
                
            packed_postings[term] = bytes(out)

        data = {
            "packed_postings": packed_postings,
            "int_to_doc_id": int_to_doc_id,
            "doc_len": self.doc_len,
            "N": self.N,
            "avg_doc_len": self.avg_doc_len,
            "tokenizer_config": self.tokenizer_config
        }

        # 2. Gzip the final pickle payload for maximum disk space efficiency
        path = os.path.join(index_dir, "index.pkl.gz")
        with gzip.open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        """
        Decodes the compressed payload back into standard dictionaries for scoring.
        Reconstruct an InvertedIndex purely from what save() wrote to
        `index_dir`. Called in a fresh process — do not rely on any state
        other than what's actually on disk in `index_dir`.
        """
        path = os.path.join(index_dir, "index.pkl.gz")
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)

        index = cls()
        index.doc_len = data["doc_len"]
        index.N = data["N"]
        index.avg_doc_len = data["avg_doc_len"]
        index.tokenizer_config = data["tokenizer_config"]
        
        int_to_doc_id = data["int_to_doc_id"]
        
        # Unpack the varint postings back into {term: {doc_id_str: tf}}
        for term, blob in data["packed_postings"].items():
            integers = decode_varint_stream(blob)
            decoded_dict = {}
            last_doc_int = 0
            
            for i in range(0, len(integers), 2):
                delta = integers[i]
                tf = integers[i+1]
                doc_int = last_doc_int + delta
                decoded_dict[int_to_doc_id[doc_int]] = tf
                last_doc_int = doc_int
                
            index.postings[term] = decoded_dict
            
        return index