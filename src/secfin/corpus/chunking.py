"""Split passages into retrieval chunks. Every chunk keeps the ID of the passage it came from,
so relevance judgements made at passage level apply to any chunking."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_SENTENCE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


@dataclass
class Chunk:
    chunk_id: str
    passage_id: str
    regulation: str
    title: str
    text: str
    # Text that is embedded / indexed; differs from `text` when a context prefix is added.
    index_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces, cur = [], ""
    for sent in _SENTENCE.split(paragraph):
        if cur and len(cur) + len(sent) + 1 > max_chars:
            pieces.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        pieces.append(cur)
    # A single sentence longer than max_chars is hard-wrapped.
    out = []
    for piece in pieces:
        out.extend(piece[i : i + max_chars] for i in range(0, len(piece), max_chars))
    return out


def chunk_passage(passage: dict, max_chars: int = 1200) -> list[Chunk]:
    paragraphs: list[str] = []
    for para in passage["text"].split("\n"):
        paragraphs.extend(_split_long(para, max_chars))
    groups: list[str] = []
    cur = ""
    for para in paragraphs:
        if cur and len(cur) + len(para) + 1 > max_chars:
            groups.append(cur)
            cur = para
        else:
            cur = f"{cur}\n{para}" if cur else para
    if cur:
        groups.append(cur)
    return [
        Chunk(
            chunk_id=f"{passage['passage_id']}::{i}",
            passage_id=passage["passage_id"],
            regulation=passage["regulation"],
            title=passage["title"],
            text=g,
            index_text=g,
        )
        for i, g in enumerate(groups)
    ]


def citation(chunk: Chunk | dict) -> str:
    c = chunk if isinstance(chunk, dict) else asdict(chunk)
    pid = c["passage_id"]
    label = f"12 CFR {pid}" if pid[:1].isdigit() and "." in pid and "(" in pid else pid
    return f"{c['regulation']}, {label} - {c['title']}"


def with_prefix(chunks: list[Chunk], mode: str, contexts: dict[str, str] | None = None) -> list[Chunk]:
    """mode: 'none' | 'title' (regulation + citation header) | 'llm' (contextual retrieval)."""
    out = []
    for c in chunks:
        if mode == "none":
            idx = c.text
        elif mode == "title":
            idx = f"{citation(c)}\n{c.text}"
        elif mode == "llm":
            if contexts is None or c.chunk_id not in contexts:
                raise KeyError(
                    f"missing LLM context for {c.chunk_id}; run scripts/build_contextual_chunks.py"
                )
            idx = f"{contexts[c.chunk_id]}\n{c.text}"
        else:
            raise ValueError(mode)
        out.append(Chunk(**{**asdict(c), "index_text": idx}))
    return out
