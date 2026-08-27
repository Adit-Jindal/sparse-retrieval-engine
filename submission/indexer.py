"""
submission/indexer.py — inverted index build + compact persistence.

Persistence format notes (this is what's graded as "index size", Section 7):
  - doc_len is stored as ONE flat varint array, positionally aligned to
    int_to_doc_id, instead of a {doc_id_str: int} dict — avoids storing
    every doc_id string a second time as dict keys.
  - all postings are concatenated into ONE bytes blob (terms sorted, for
    gzip locality) with a delta-varint offset table, instead of a dict of
    many small bytes objects (each carrying its own pickle/dict overhead).
This changes nothing about ranking — it's a pure storage-format win.
"""
import os
import pickle
import re
import gzip
from functools import lru_cache
from typing import Dict, List, Tuple, Any

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

            @lru_cache(maxsize=None)
            def _stem(token: str) -> str:
                return stemmer.stem(token)
            self._stem = _stem

    def tokenize(self, text: str) -> List[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        if self.use_stopwords:
            tokens = [t for t in tokens if t not in _STOPWORDS]
        if self.use_stemmer:
            tokens = [self._stem(t) for t in tokens]
        return tokens


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


class InvertedIndex:
    def __init__(self):
        self.postings: Dict[str, Dict[str, int]] = {}
        self.doc_len: Dict[str, int] = {}
        self.N: int = 0
        self.avg_doc_len: float = 0.0
        self.tokenizer_config: Dict[str, Any] = {}

    def build(self, corpus: List[Tuple[str, str]], tokenizer: Tokenizer) -> None:
        self.tokenizer_config = tokenizer.config
        self.doc_len = {}
        temp_postings: Dict[str, Dict[str, int]] = {}
        for doc_id, text in corpus:
            tokens = tokenizer.tokenize(text)
            self.doc_len[doc_id] = len(tokens)
            term_counts: Dict[str, int] = {}
            for term in tokens:
                term_counts[term] = term_counts.get(term, 0) + 1
            for term, tf in term_counts.items():
                temp_postings.setdefault(term, {})[doc_id] = tf
        self.postings = temp_postings
        self.N = len(self.doc_len)
        self.avg_doc_len = sum(self.doc_len.values()) / self.N if self.N > 0 else 0.0

    def document_frequency(self, term: str) -> int:
        postings = self.postings.get(term)
        return len(postings) if postings else 0

    def save(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        int_to_doc_id = list(self.doc_len.keys())
        doc_id_to_int = {doc_id: i for i, doc_id in enumerate(int_to_doc_id)}

        doc_len_bytes = bytearray()
        for doc_id in int_to_doc_id:
            encode_varint(self.doc_len[doc_id], doc_len_bytes)

        terms_sorted = sorted(self.postings.keys())
        postings_blob = bytearray()
        offsets = [0]
        for term in terms_sorted:
            doc_tfs = self.postings[term]
            sorted_docs = sorted((doc_id_to_int[d], tf) for d, tf in doc_tfs.items())
            last_doc_int = 0
            for doc_int, tf in sorted_docs:
                encode_varint(doc_int - last_doc_int, postings_blob)
                encode_varint(tf, postings_blob)
                last_doc_int = doc_int
            offsets.append(len(postings_blob))

        offsets_bytes = bytearray()
        last_off = 0
        for off in offsets:
            encode_varint(off - last_off, offsets_bytes)
            last_off = off

        data = {
            "terms": terms_sorted,
            "postings_blob": bytes(postings_blob),
            "offsets_blob": bytes(offsets_bytes),
            "int_to_doc_id": int_to_doc_id,
            "doc_len_blob": bytes(doc_len_bytes),
            "N": self.N,
            "avg_doc_len": self.avg_doc_len,
            "tokenizer_config": self.tokenizer_config,
        }
        path = os.path.join(index_dir, "index.pkl.gz")
        with gzip.open(path, "wb", compresslevel=9) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        path = os.path.join(index_dir, "index.pkl.gz")
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)

        index = cls()
        index.N = data["N"]
        index.avg_doc_len = data["avg_doc_len"]
        index.tokenizer_config = data["tokenizer_config"]

        int_to_doc_id = data["int_to_doc_id"]
        doc_lengths = decode_varint_stream(data["doc_len_blob"])
        index.doc_len = dict(zip(int_to_doc_id, doc_lengths))

        offset_deltas = decode_varint_stream(data["offsets_blob"])
        offsets, running = [], 0
        for d in offset_deltas:
            running += d
            offsets.append(running)

        blob = data["postings_blob"]
        terms = data["terms"]
        postings: Dict[str, Dict[str, int]] = {}
        for i, term in enumerate(terms):
            chunk = blob[offsets[i]:offsets[i + 1]]
            integers = decode_varint_stream(chunk)
            decoded: Dict[str, int] = {}
            last_doc_int = 0
            for j in range(0, len(integers), 2):
                doc_int = last_doc_int + integers[j]
                decoded[int_to_doc_id[doc_int]] = integers[j + 1]
                last_doc_int = doc_int
            postings[term] = decoded

        index.postings = postings
        return index