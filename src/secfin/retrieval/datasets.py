"""Benchmark datasets: the eCFR golden set (passage-level qrels) and BEIR FiQA-2018 (test split)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from secfin.corpus.chunking import chunk_passage, with_prefix
from secfin.runinfo import REPO_ROOT, read_jsonl


@dataclass
class IRDataset:
    name: str
    doc_texts: dict[str, str]  # what gets indexed
    doc_to_unit: dict[str, str]  # chunk -> unit that qrels refer to (identity for FiQA)
    queries: dict[str, str]
    qrels: dict[str, dict[str, float]]  # query -> {unit: gain}
    raw_texts: dict[str, str] | None = None  # text shown to reranker / generator (no prefix)


def load_ecfr_chunks(
    prefix: str = "none",
    max_chars: int = 1200,
    contexts_path: str | Path | None = None,
    passages_path: str | Path = REPO_ROOT / "data/corpus/passages.jsonl",
):
    passages = read_jsonl(passages_path)
    chunks = [c for p in passages for c in chunk_passage(p, max_chars)]
    contexts = None
    if prefix == "llm":
        path = Path(contexts_path) if contexts_path else None
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"contextual prefixes not found at {path}; run scripts/build_contextual_chunks.py"
            )
        contexts = {r["chunk_id"]: r["context"] for r in read_jsonl(path)}
    return with_prefix(chunks, prefix, contexts), passages


def load_ecfr(
    split: str = "test",
    prefix: str = "none",
    max_chars: int = 1200,
    contexts_path: str | Path | None = None,
    golden_path: str | Path = REPO_ROOT / "data/golden/golden.jsonl",
) -> IRDataset:
    chunks, _ = load_ecfr_chunks(prefix, max_chars, contexts_path)
    golden = [g for g in read_jsonl(golden_path) if g["type"] != "unanswerable"]
    if split != "all":
        golden = [g for g in golden if g["split"] == split]
    return IRDataset(
        name=f"ecfr-{split}",
        doc_texts={c.chunk_id: c.index_text for c in chunks},
        doc_to_unit={c.chunk_id: c.passage_id for c in chunks},
        queries={g["id"]: g["question"] for g in golden},
        qrels={g["id"]: {p: 1.0 for p in g["gold_passages"]} for g in golden},
        raw_texts={c.chunk_id: c.text for c in chunks},
    )


def load_fiqa(cache_dir: str | Path = REPO_ROOT / "data/cache/fiqa", split: str = "test") -> IRDataset:
    """BEIR FiQA-2018 from the Hugging Face Hub (BeIR/fiqa, BeIR/fiqa-qrels), CC BY-SA 4.0."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    cache_dir = Path(cache_dir)
    corpus_f = hf_hub_download(
        "BeIR/fiqa", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset", cache_dir=cache_dir
    )
    queries_f = hf_hub_download(
        "BeIR/fiqa", "queries/queries-00000-of-00001.parquet", repo_type="dataset", cache_dir=cache_dir
    )
    qrels_f = hf_hub_download("BeIR/fiqa-qrels", f"{split}.tsv", repo_type="dataset", cache_dir=cache_dir)

    corpus = pq.read_table(corpus_f).to_pylist()
    texts = {str(d["_id"]): f"{d.get('title') or ''} {d['text']}".strip() for d in corpus}
    qrels: dict[str, dict[str, float]] = {}
    with open(qrels_f) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if float(row["score"]) > 0:
                qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = float(row["score"])
    all_q = {str(q["_id"]): q["text"] for q in pq.read_table(queries_f).to_pylist()}
    queries = {qid: all_q[qid] for qid in qrels}
    return IRDataset(
        name=f"fiqa-{split}",
        doc_texts=texts,
        doc_to_unit={d: d for d in texts},
        queries=queries,
        qrels=qrels,
        raw_texts=texts,
    )
