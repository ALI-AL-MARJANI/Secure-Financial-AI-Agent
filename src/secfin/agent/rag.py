"""Regulation search + grounded answer generation used by the RAG route, the agent tool and the evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from secfin.corpus.chunking import citation
from secfin.llm import TextLLM
from secfin.retrieval.components import DenseIndex
from secfin.retrieval.datasets import load_ecfr_chunks
from secfin.retrieval.pipeline import RetrievalPipeline

ABSTAIN = "I could not find this in Regulations E, DD or CC."

ANSWER_PROMPT = """You answer questions about US consumer-banking regulations using ONLY the numbered excerpts below.
Rules:
- Base every statement on the excerpts and cite them like [1] or [2][3].
- Keep the exact numbers, deadlines and dollar amounts from the excerpts.
- If the excerpts do not contain the answer, reply exactly: "{abstain}"
- Do not give investment advice.

Excerpts:
{context}

Question: {question}
Answer:"""


@dataclass
class Retrieved:
    chunk_id: str
    passage_id: str
    citation: str
    text: str


class RegulationSearch:
    def __init__(self, cfg: dict, device: str | None = None, contexts_path: str | None = None):
        chunks, _ = load_ecfr_chunks(
            cfg.get("prefix", "title"), cfg.get("chunk_max_chars", 1200), contexts_path
        )
        self.chunks = {c.chunk_id: c for c in chunks}
        index_texts = {c.chunk_id: c.index_text for c in chunks}
        dense = None
        if cfg.get("dense"):
            ids = list(index_texts)
            dense = DenseIndex(
                cfg["dense_model"],
                ids,
                [index_texts[i] for i in ids],
                query_prefix=cfg.get("dense_query_prefix", ""),
                device=device,
            )
        bm25 = None
        if cfg.get("sparse"):
            from secfin.retrieval.components import BM25Index

            ids = list(index_texts)
            bm25 = BM25Index(ids, [index_texts[i] for i in ids])
        reranker = None
        if cfg.get("rerank"):
            from secfin.retrieval.components import CrossEncoderReranker

            reranker = CrossEncoderReranker(cfg["reranker"], device=device)
        self.pipeline = RetrievalPipeline(cfg, index_texts, bm25=bm25, dense=dense, reranker=reranker)

    def search(self, query: str) -> list[Retrieved]:
        res = self.pipeline.search(query)
        return [
            Retrieved(cid, self.chunks[cid].passage_id, citation(self.chunks[cid]), self.chunks[cid].text)
            for cid in res.context_ids
        ]


def format_context(docs: list[Retrieved]) -> str:
    return "\n\n".join(f"[{i + 1}] ({d.citation})\n{d.text}" for i, d in enumerate(docs))


def generate_answer(llm: TextLLM, question: str, docs: list[Retrieved]) -> str:
    if not docs:
        return ABSTAIN
    return llm(ANSWER_PROMPT.format(abstain=ABSTAIN, context=format_context(docs), question=question))


def is_abstention(answer: str) -> bool:
    return ABSTAIN.lower().rstrip(".") in answer.lower()
