"""Retrieval components that call an LLM: HyDE query expansion, contextual chunk prefixes, CRAG grading.

Deviations from the papers (documented in the README):
  * HyDE (Gao et al., 2022) averages the embeddings of N generated documents and the query; we do the same
    with N=3 and a 7B local model instead of InstructGPT, and use the averaged vector only for the dense leg.
  * Contextual Retrieval (Anthropic, 2024) puts the whole document in the prompt; the eCFR parts are
    100k+ words, so we give the enclosing section (truncated to `max_doc_chars`) instead.
  * CRAG (Yan et al., 2024) uses a fine-tuned T5 evaluator and web search on "incorrect". We use the same LLM
    as a zero-shot evaluator and, being offline, fall back to a rewritten query with a wider candidate pool;
    if retrieval is still graded incorrect the pipeline returns no context (abstention signal).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from secfin.llm import TextLLM

HYDE_PROMPT = (
    "Write a short passage (3-4 sentences) from a US federal banking regulation (Regulation E, DD or CC) "
    "or its official interpretation that answers the question. Use regulatory language. "
    "Do not mention that the passage is hypothetical.\n\nQuestion: {question}\n\nPassage:"
)

# Verbatim prompt from Anthropic's "Introducing Contextual Retrieval" (2024).
CONTEXT_PROMPT = (
    "<document>\n{document}\n</document>\n"
    "Here is the chunk we want to situate within the whole document\n<chunk>\n{chunk}\n</chunk>\n"
    "Please give a short succinct context to situate this chunk within the overall document for the "
    "purposes of improving search retrieval of the chunk. Answer only with the succinct context and "
    "nothing else."
)

GRADER_PROMPT = (
    "You are grading whether retrieved regulation excerpts contain the information needed to answer a "
    "question.\n\nQuestion: {question}\n\nExcerpts:\n{context}\n\n"
    "Reply with exactly one word:\n"
    "CORRECT - the excerpts contain the answer\n"
    "AMBIGUOUS - the excerpts are related but may be incomplete\n"
    "INCORRECT - the excerpts do not address the question\n\nGrade:"
)

REWRITE_PROMPT = (
    "Rewrite the question as a search query for US banking regulations (Regulation E, DD, CC). "
    "Use the legal terms a regulation would use. Output only the query.\n\nQuestion: {question}\nQuery:"
)


def hyde_documents(llm: TextLLM, question: str, n: int) -> list[str]:
    return [llm(HYDE_PROMPT.format(question=question)) for _ in range(n)]


def chunk_context(llm: TextLLM, document: str, chunk: str, max_doc_chars: int) -> str:
    out = llm(CONTEXT_PROMPT.format(document=document[:max_doc_chars], chunk=chunk))
    return " ".join(out.split())


def parse_grade(text: str) -> str:
    m = re.search(r"\b(CORRECT|AMBIGUOUS|INCORRECT)\b", text.upper())
    # Unparseable grader output is treated as ambiguous: keep the retrieved context.
    return m.group(1).lower() if m else "ambiguous"


def grade_retrieval(llm: TextLLM, question: str, contexts: Sequence[str], max_chars: int = 6000) -> str:
    joined = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))[:max_chars]
    return parse_grade(llm(GRADER_PROMPT.format(question=question, context=joined)))


def rewrite_query(llm: TextLLM, question: str) -> str:
    return llm(REWRITE_PROMPT.format(question=question)).strip().strip('"')


PROMPTS = {
    "hyde": HYDE_PROMPT,
    "contextual": CONTEXT_PROMPT,
    "crag_grader": GRADER_PROMPT,
    "crag_rewrite": REWRITE_PROMPT,
}
