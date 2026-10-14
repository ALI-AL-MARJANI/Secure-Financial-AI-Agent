"""Retrieval building blocks: BM25, dense (FAISS), reciprocal rank fusion, cross-encoder reranking."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import snowballstemmer

_STEMMER = snowballstemmer.stemmer("english")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.$%][a-z0-9]+)*")
# Small English stopword list; BM25 without stopword removal is dominated by "the/of/a" on long chunks.
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "would",
        "can",
        "does",
        "do",
        "i",
        "my",
        "me",
        "you",
        "your",
        "how",
        "under",
    ]
)


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _STEMMER.stemWord(word)


def tokenize(text: str) -> list[str]:
    """Lowercase, drop stopwords, Snowball (Porter2) stemming: close to Lucene's English analyzer."""
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Hit:
    doc_id: str
    score: float


class BM25Index:
    def __init__(self, doc_ids: Sequence[str], texts: Sequence[str], k1: float = 0.9, b: float = 0.4):
        # k1=0.9, b=0.4 are the Anserini/Pyserini BEIR defaults.
        from rank_bm25 import BM25Okapi

        self.doc_ids = list(doc_ids)
        self.bm25 = BM25Okapi([tokenize(t) for t in texts], k1=k1, b=b)

    def search(self, query: str, k: int) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [Hit(self.doc_ids[i], float(scores[i])) for i in top]


class DenseIndex:
    """Cosine similarity over normalized sentence-transformer embeddings (exact FAISS inner product)."""

    def __init__(
        self,
        model_name: str,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        query_prefix: str = "",
        device: str | None = None,
        batch_size: int = 64,
        embeddings: np.ndarray | None = None,
    ):
        import faiss
        from sentence_transformers import SentenceTransformer

        # torch and faiss-cpu ship separate OpenMP runtimes on macOS; multi-threaded FAISS search then
        # segfaults once torch is loaded. Exact search over 60k x 384 is ~1 ms single-threaded anyway.
        faiss.omp_set_num_threads(1)
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = query_prefix
        self.doc_ids = list(doc_ids)
        if embeddings is None:
            embeddings = self.encode_docs(texts, batch_size)
        self.embeddings = embeddings.astype("float32")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def encode_docs(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 1000
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([self.query_prefix + query], normalize_embeddings=True)[0]

    def search_vector(self, vec: np.ndarray, k: int) -> list[Hit]:
        v = np.asarray(vec, dtype="float32").reshape(1, -1)
        v /= np.linalg.norm(v) + 1e-12
        scores, idx = self.index.search(v, k)
        return [Hit(self.doc_ids[i], float(s)) for s, i in zip(scores[0], idx[0], strict=True) if i >= 0]

    def search(self, query: str, k: int) -> list[Hit]:
        return self.search_vector(self.encode_query(query), k)


def rrf(rankings: Sequence[Sequence[Hit]], k: int = 60, top: int | None = None) -> list[Hit]:
    """Reciprocal rank fusion (Cormack et al., 2009): score(d) = sum_r 1 / (k + rank_r(d))."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking, start=1):
            scores[h.doc_id] = scores.get(h.doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return [Hit(d, s) for d, s in fused[:top]]


class CrossEncoderReranker:
    def __init__(
        self, model_name: str, device: str | None = None, max_length: int = 512, batch_size: int = 32
    ):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.batch_size = batch_size

    def rerank(self, query: str, hits: Sequence[Hit], texts: dict[str, str], k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict(
            [(query, texts[h.doc_id]) for h in hits], batch_size=self.batch_size, show_progress_bar=False
        )
        order = np.argsort(-np.asarray(scores))[:k]
        return [Hit(hits[i].doc_id, float(scores[i])) for i in order]
