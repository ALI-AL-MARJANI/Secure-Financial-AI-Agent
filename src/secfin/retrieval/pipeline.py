"""Configurable retrieval pipeline used by the benchmark, the RAG agent, and the end-to-end evaluation.

Config keys (see configs/retrieval/ablation.yaml):
  sparse: bool          BM25 leg
  dense: bool           dense leg (sentence-transformers + FAISS)
  candidates: int       depth of each leg before fusion / reranking
  rrf_k: int            reciprocal-rank-fusion constant (only used when both legs are on)
  rerank: bool          cross-encoder reranking of the fused candidates
  hyde: int             number of hypothetical documents (0 = off)
  crag: bool            LLM grading of the top `context_k` results with rewrite + abstention fallback
  context_k: int        number of chunks passed to the generator
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from secfin.llm import TextLLM
from secfin.retrieval import llm_components as llmc
from secfin.retrieval.components import BM25Index, CrossEncoderReranker, DenseIndex, Hit, rrf


@dataclass
class RetrievalResult:
    ranking: list[str]  # doc/chunk IDs, best first
    context_ids: list[str]  # what the generator would see (empty = abstain)
    grade: str | None = None  # CRAG grade of the final context
    rewritten_query: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


class RetrievalPipeline:
    def __init__(
        self,
        cfg: dict,
        texts: dict[str, str],
        bm25: BM25Index | None = None,
        dense: DenseIndex | None = None,
        reranker: CrossEncoderReranker | None = None,
        llm: TextLLM | None = None,
    ):
        self.cfg = cfg
        self.texts = texts
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.llm = llm
        if cfg.get("sparse") and bm25 is None:
            raise ValueError("config enables BM25 but no index was given")
        if cfg.get("dense") and dense is None:
            raise ValueError("config enables dense retrieval but no index was given")
        if cfg.get("rerank") and reranker is None:
            raise ValueError("config enables reranking but no reranker was given")
        if (cfg.get("hyde") or cfg.get("crag")) and llm is None:
            raise ValueError("HyDE / CRAG need an LLM")

    def _dense_hits(self, query: str, k: int, timings: dict) -> list[Hit]:
        n = int(self.cfg.get("hyde") or 0)
        if n == 0:
            return self.dense.search(query, k)
        t = time.perf_counter()
        docs = llmc.hyde_documents(self.llm, query, n)
        timings["hyde_llm_s"] = timings.get("hyde_llm_s", 0.0) + time.perf_counter() - t
        # HyDE: mean of the query embedding and the embeddings of the generated documents.
        vecs = [self.dense.encode_query(query)] + list(self.dense.encode_docs(docs))
        return self.dense.search_vector(np.mean(vecs, axis=0), k)

    def _retrieve(self, query: str, candidates: int, timings: dict) -> list[Hit]:
        legs = []
        if self.cfg.get("sparse"):
            legs.append(self.bm25.search(query, candidates))
        if self.cfg.get("dense"):
            legs.append(self._dense_hits(query, candidates, timings))
        hits = rrf(legs, k=int(self.cfg.get("rrf_k", 60)), top=candidates) if len(legs) > 1 else legs[0]
        if self.cfg.get("rerank"):
            t = time.perf_counter()
            hits = self.reranker.rerank(query, hits, self.texts, k=len(hits))
            timings["rerank_s"] = timings.get("rerank_s", 0.0) + time.perf_counter() - t
        return hits

    def search(self, query: str) -> RetrievalResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        candidates = int(self.cfg.get("candidates", 50))
        context_k = int(self.cfg.get("context_k", 5))
        hits = self._retrieve(query, candidates, timings)
        ranking = [h.doc_id for h in hits]
        result = RetrievalResult(ranking=ranking, context_ids=ranking[:context_k])

        if self.cfg.get("crag"):
            t = time.perf_counter()
            grade = llmc.grade_retrieval(self.llm, query, [self.texts[d] for d in result.context_ids])
            if grade == "incorrect":
                rewritten = llmc.rewrite_query(self.llm, query)
                hits = self._retrieve(rewritten, candidates * 2, timings)
                ranking = [h.doc_id for h in hits]
                grade = llmc.grade_retrieval(self.llm, query, [self.texts[d] for d in ranking[:context_k]])
                result = RetrievalResult(
                    ranking=ranking,
                    context_ids=[] if grade == "incorrect" else ranking[:context_k],
                    rewritten_query=rewritten,
                )
            result.grade = grade
            timings["crag_s"] = time.perf_counter() - t

        timings["total_s"] = time.perf_counter() - t0
        result.timings = timings
        return result
