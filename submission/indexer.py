"""
submission/indexer.py — inverted index build + compact persistence, plus
optional (flag-gated, default OFF) indexing-time features:

  preserve_compounds  — also index hyphenated compounds ("sars-cov-2") as
                         single joined tokens, alongside the unigrams the
                         base regex already produces from them.
  use_bigrams          — also index adjacent-surviving-token bigrams
                         ("t1_t2"), a cheap proxy for phrase matching
                         without a full positional index.
  use_corpus_stopwords — after building postings, drop any term whose
                         document frequency exceeds a corpus-relative
                         threshold (terms in "almost every document"
                         contribute ~0 IDF anyway; removing them is a
                         pure disk/build win with no ranking effect,
                         since a missing term is already treated as
                         "no match" everywhere in scoring).
  use_capitalization   — track, per term, how often it appears
                         capitalized in a NON-sentence-initial position
                         (a cheap proper-noun signal; sentence-initial
                         capitalization is excluded because it's
                         universal and uninformative).
  use_pseudo_title      — extract each document's first sentence as a
                         synthetic "title" field and persist a
                         presence-only postings structure for it.

All are OFF by default — enabling any of them requires a full reindex,
so they're meant to be swept in batch (see dev/sweep_custom_scorer.py)
rather than toggled one at a time.

Persistence: doc_ids are stored via an auto-detected compact codec
(common prefix + fixed-width numeric suffix, e.g. "doc_000123") when the
corpus's IDs follow that shape, falling back to an explicit string list
otherwise — this is unconditional and has no effect on ranking. Main
postings and title postings share the same delta-varint blob encoding.
"""
import os
import pickle
import re
import gzip
from functools import lru_cache
from typing import Dict, List, Tuple, Any, Optional

_STOPWORDS_MINIMAL = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"
}

_STOPWORDS_STANDARD = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain",
    "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn",
    "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
    "weren", "won", "wouldn",
}

_STOPWORD_SETS = {"minimal": _STOPWORDS_MINIMAL, "standard": _STOPWORDS_STANDARD, "none": frozenset()}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_COMPOUND_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")
_SENT_END_RE = re.compile(r"[.!?]\s+")
_RAW_WORD_RE = re.compile(r"[A-Za-z0-9]+")


class Tokenizer:
    def __init__(self, use_stopwords: bool = True, use_stemmer: bool = True,
                 stopword_set: str = "standard", preserve_compounds: bool = False,
                 use_bigrams: bool = False):
        self.use_stopwords = use_stopwords
        self.use_stemmer = use_stemmer
        self.stopword_set = stopword_set
        self.preserve_compounds = preserve_compounds
        self.use_bigrams = use_bigrams
        self.config = {
            "stopwords": use_stopwords,
            "stemming": use_stemmer,
            "stopword_set": stopword_set,
            "preserve_compounds": preserve_compounds,
            "use_bigrams": use_bigrams,
        }
        self._stopword_words = _STOPWORD_SETS.get(stopword_set, _STOPWORDS_STANDARD)

        if self.use_stemmer:
            from nltk.stem import PorterStemmer
            stemmer = PorterStemmer()

            @lru_cache(maxsize=None)
            def _stem(token: str) -> str:
                return stemmer.stem(token)
            self._stem = _stem

    def tokenize(self, text: str) -> List[str]:
        lowered = text.lower()
        tokens = _TOKEN_RE.findall(lowered)
        if self.use_stopwords:
            tokens = [t for t in tokens if t not in self._stopword_words]
        if self.use_stemmer:
            tokens = [self._stem(t) for t in tokens]

        extra: List[str] = []
        if self.preserve_compounds:
            for m in _COMPOUND_RE.finditer(lowered):
                extra.append(m.group(0).replace("-", "_"))
        if self.use_bigrams:
            for i in range(len(tokens) - 1):
                extra.append(tokens[i] + "_" + tokens[i + 1])

        return tokens + extra

    def extract_pseudo_title_terms(self, text: str, max_tokens: int = 12) -> List[str]:
        """First sentence of the document, tokenized the same way as the
        body — used only when use_pseudo_title is enabled."""
        first_sentence = _SENT_END_RE.split(text, maxsplit=1)[0]
        return self.tokenize(first_sentence)[:max_tokens]


def _capitalization_stats(text: str) -> Dict[str, Tuple[int, int]]:
    """{lowercased_word: (non_sentence_initial_cap_count, total_count)}
    for one document. Sentence-initial occurrences are excluded from the
    cap count since capitalization there is universal, not a proper-noun
    signal — the corpus format guarantees period-terminated sentences
    (assignment spec, Section 2), so sentence splitting is reliable."""
    stats: Dict[str, Tuple[int, int]] = {}
    for sentence in _SENT_END_RE.split(text):
        words = _RAW_WORD_RE.findall(sentence)
        for i, w in enumerate(words):
            lw = w.lower()
            is_cap = w[0].isupper()
            is_initial = (i == 0)
            cap_ct, total_ct = stats.get(lw, (0, 0))
            stats[lw] = (cap_ct + (1 if (is_cap and not is_initial) else 0), total_ct + 1)
    return stats


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


_DOC_ID_SUFFIX_RE = re.compile(r"^(.*?)(\d+)$")


def _try_compact_doc_id_codec(doc_ids: List[str]) -> Optional[Tuple[str, int, List[int]]]:
    """If every doc_id is (constant prefix) + (fixed-width zero-padded
    number), return (prefix, width, numbers) for compact storage.
    Otherwise return None and the caller falls back to an explicit
    string list. Purely a disk-size optimisation — never affects
    ranking, since doc_ids round-trip byte-for-byte either way."""
    if not doc_ids:
        return None
    prefix = None
    width = None
    numbers = []
    for d in doc_ids:
        m = _DOC_ID_SUFFIX_RE.match(d)
        if not m:
            return None
        p, digits = m.group(1), m.group(2)
        if prefix is None:
            prefix, width = p, len(digits)
        elif p != prefix or len(digits) != width:
            return None
        numbers.append(int(digits))
    for d, n in zip(doc_ids, numbers):
        if f"{prefix}{n:0{width}d}" != d:
            return None
    return prefix, width, numbers


class InvertedIndex:
    def __init__(self):
        self.postings: Dict[str, Dict[str, int]] = {}
        self.doc_len: Dict[str, int] = {}
        self.N: int = 0
        self.avg_doc_len: float = 0.0
        self.tokenizer_config: Dict[str, Any] = {}
        self.cap_score: Dict[str, float] = {}          # term -> proper-noun-ish score in [0,1]
        self.title_postings: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: 1}, presence only
        self.positions: Dict[str, Dict[str, List[int]]] = {}  # term -> {doc_id: [token_idx, ...]}

    def build(self, corpus: List[Tuple[str, str]], tokenizer: Tokenizer,
              use_corpus_stopwords: bool = False, corpus_stopword_df_ratio: float = 0.85,
              use_capitalization: bool = False, use_pseudo_title: bool = False,
              use_positions: bool = False) -> None:
        temp_positions: Dict[str, Dict[str, List[int]]] = {}
        self.tokenizer_config = dict(tokenizer.config)
        self.doc_len = {}
        temp_postings: Dict[str, Dict[str, int]] = {}
        cap_raw: Dict[str, Tuple[int, int]] = {}
        title_postings: Dict[str, Dict[str, int]] = {}

        for doc_id, text in corpus:
            tokens = tokenizer.tokenize(text)
            self.doc_len[doc_id] = len(tokens)
            term_counts: Dict[str, int] = {}
            for idx, term in enumerate(tokens):
                term_counts[term] = term_counts.get(term, 0) + 1
                if use_positions:
                    temp_positions.setdefault(term, {}).setdefault(doc_id, []).append(idx)
            for term, tf in term_counts.items():
                temp_postings.setdefault(term, {})[doc_id] = tf

            if use_capitalization:
                for word, (cap_ct, total_ct) in _capitalization_stats(text).items():
                    stemmed = tokenizer._stem(word) if tokenizer.use_stemmer else word
                    prev_cap, prev_total = cap_raw.get(stemmed, (0, 0))
                    cap_raw[stemmed] = (prev_cap + cap_ct, prev_total + total_ct)

            if use_pseudo_title:
                for term in set(tokenizer.extract_pseudo_title_terms(text)):
                    title_postings.setdefault(term, {})[doc_id] = 1

        self.postings = temp_postings
        self.N = len(self.doc_len)
        self.avg_doc_len = sum(self.doc_len.values()) / self.N if self.N > 0 else 0.0
        self.tokenizer_config["use_corpus_stopwords"] = use_corpus_stopwords
        self.tokenizer_config["corpus_stopword_df_ratio"] = corpus_stopword_df_ratio
        self.tokenizer_config["use_capitalization"] = use_capitalization
        self.tokenizer_config["use_pseudo_title"] = use_pseudo_title
        self.tokenizer_config["use_positions"] = use_positions
        if use_positions:
            # positions are collected in tokenize-order already sorted per doc
            self.positions = temp_positions

        if use_corpus_stopwords and self.N > 0:
            to_drop = [t for t, postings in self.postings.items()
                       if len(postings) / self.N > corpus_stopword_df_ratio]
            for t in to_drop:
                del self.postings[t]
            # forward-index / IDF lookups elsewhere already treat a
            # missing term as "no match, contributes nothing" — no
            # other module needs to know which terms were pruned.

        if use_capitalization:
            self.cap_score = {
                t: (cap_ct / total_ct) for t, (cap_ct, total_ct) in cap_raw.items()
                if total_ct > 0 and t in self.postings
            }

        if use_pseudo_title:
            self.title_postings = title_postings

    def document_frequency(self, term: str) -> int:
        postings = self.postings.get(term)
        return len(postings) if postings else 0

    # -- shared postings encode/decode, reused for main + title postings --
    @staticmethod
    def _encode_postings(postings: Dict[str, Dict[str, int]], doc_id_to_int: Dict[str, int]):
        terms_sorted = sorted(postings.keys())
        blob = bytearray()
        offsets = [0]
        for term in terms_sorted:
            sorted_docs = sorted((doc_id_to_int[d], tf) for d, tf in postings[term].items())
            last = 0
            for doc_int, tf in sorted_docs:
                encode_varint(doc_int - last, blob)
                encode_varint(tf, blob)
                last = doc_int
            offsets.append(len(blob))
        offsets_bytes = bytearray()
        last_off = 0
        for off in offsets:
            encode_varint(off - last_off, offsets_bytes)
            last_off = off
        return terms_sorted, bytes(blob), bytes(offsets_bytes)

    @staticmethod
    def _decode_postings(terms: List[str], blob: bytes, offsets_blob: bytes,
                          int_to_doc_id: List[str]) -> Dict[str, Dict[str, int]]:
        offset_deltas = decode_varint_stream(offsets_blob)
        offsets, running = [], 0
        for d in offset_deltas:
            running += d
            offsets.append(running)
        postings: Dict[str, Dict[str, int]] = {}
        for i, term in enumerate(terms):
            chunk = blob[offsets[i]:offsets[i + 1]]
            integers = decode_varint_stream(chunk)
            decoded: Dict[str, int] = {}
            last = 0
            for j in range(0, len(integers), 2):
                doc_int = last + integers[j]
                decoded[int_to_doc_id[doc_int]] = integers[j + 1]
                last = doc_int
            postings[term] = decoded
        return postings

    @staticmethod
    def _encode_positions(positions: Dict[str, Dict[str, List[int]]],
                           doc_id_to_int: Dict[str, int]):
        terms_sorted = sorted(positions.keys())
        blob = bytearray()
        offsets = [0]
        for term in terms_sorted:
            sorted_docs = sorted(positions[term].items(), key=lambda kv: doc_id_to_int[kv[0]])
            last_doc = 0
            for doc_id, plist in sorted_docs:
                doc_int = doc_id_to_int[doc_id]
                encode_varint(doc_int - last_doc, blob)
                last_doc = doc_int
                encode_varint(len(plist), blob)
                last_pos = 0
                for p in plist:  # already ascending (built in token order)
                    encode_varint(p - last_pos, blob)
                    last_pos = p
            offsets.append(len(blob))
        offsets_bytes = bytearray()
        last_off = 0
        for off in offsets:
            encode_varint(off - last_off, offsets_bytes)
            last_off = off
        return terms_sorted, bytes(blob), bytes(offsets_bytes)

    @staticmethod
    def _decode_positions(terms: List[str], blob: bytes, offsets_blob: bytes,
                           int_to_doc_id: List[str]) -> Dict[str, Dict[str, List[int]]]:
        offset_deltas = decode_varint_stream(offsets_blob)
        offsets, running = [], 0
        for d in offset_deltas:
            running += d
            offsets.append(running)
        result: Dict[str, Dict[str, List[int]]] = {}
        for i, term in enumerate(terms):
            chunk = blob[offsets[i]:offsets[i + 1]]
            ints = decode_varint_stream(chunk)
            doc_map: Dict[str, List[int]] = {}
            j = 0
            last_doc = 0
            while j < len(ints):
                doc_int = last_doc + ints[j]; last_doc = doc_int; j += 1
                count = ints[j]; j += 1
                plist = []
                last_pos = 0
                for _ in range(count):
                    last_pos += ints[j]; j += 1
                    plist.append(last_pos)
                doc_map[int_to_doc_id[doc_int]] = plist
            result[term] = doc_map
        return result

    def save(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        int_to_doc_id = list(self.doc_len.keys())
        doc_id_to_int = {doc_id: i for i, doc_id in enumerate(int_to_doc_id)}

        doc_len_bytes = bytearray()
        for doc_id in int_to_doc_id:
            encode_varint(self.doc_len[doc_id], doc_len_bytes)

        terms_sorted, postings_blob, offsets_blob = self._encode_postings(self.postings, doc_id_to_int)

        codec = _try_compact_doc_id_codec(int_to_doc_id)
        if codec is not None:
            prefix, width, numbers = codec
            num_bytes = bytearray()
            for n in numbers:
                encode_varint(n, num_bytes)
            doc_id_codec = {"mode": "compact", "prefix": prefix, "width": width, "numbers": bytes(num_bytes)}
        else:
            doc_id_codec = {"mode": "explicit", "values": int_to_doc_id}

        data = {
            "terms": terms_sorted,
            "postings_blob": postings_blob,
            "offsets_blob": offsets_blob,
            "doc_id_codec": doc_id_codec,
            "doc_len_blob": bytes(doc_len_bytes),
            "N": self.N,
            "avg_doc_len": self.avg_doc_len,
            "tokenizer_config": self.tokenizer_config,
        }
        if self.cap_score:
            data["cap_score"] = self.cap_score
        if self.title_postings:
            t_terms, t_blob, t_offsets = self._encode_postings(self.title_postings, doc_id_to_int)
            data["title_terms"] = t_terms
            data["title_postings_blob"] = t_blob
            data["title_offsets_blob"] = t_offsets
        if self.positions:
            p_terms, p_blob, p_offsets = self._encode_positions(self.positions, doc_id_to_int)
            data["position_terms"] = p_terms
            data["position_blob"] = p_blob
            data["position_offsets"] = p_offsets

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

        codec = data["doc_id_codec"]
        if codec["mode"] == "compact":
            numbers = decode_varint_stream(codec["numbers"])
            width = codec["width"]
            prefix = codec["prefix"]
            int_to_doc_id = [f"{prefix}{n:0{width}d}" for n in numbers]
        else:
            int_to_doc_id = codec["values"]

        doc_lengths = decode_varint_stream(data["doc_len_blob"])
        index.doc_len = dict(zip(int_to_doc_id, doc_lengths))

        index.postings = cls._decode_postings(data["terms"], data["postings_blob"], data["offsets_blob"], int_to_doc_id)

        index.cap_score = data.get("cap_score", {})
        if "title_terms" in data:
            index.title_postings = cls._decode_postings(
                data["title_terms"], data["title_postings_blob"], data["title_offsets_blob"], int_to_doc_id
            )
        if "position_terms" in data:
            index.positions = cls._decode_positions(
                data["position_terms"], data["position_blob"], data["position_offsets"], int_to_doc_id
            )

        return index