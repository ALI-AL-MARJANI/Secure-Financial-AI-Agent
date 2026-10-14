from secfin.retrieval.components import BM25Index, Hit, rrf, tokenize
from secfin.retrieval.llm_components import parse_grade
from secfin.retrieval.pipeline import RetrievalPipeline

DOCS = {
    "d1": "Consumer liability for unauthorized electronic fund transfers is limited to $50.",
    "d2": "Funds from cash deposits must be available the next business day.",
    "d3": "Advertisements must state the annual percentage yield.",
    "d4": "A remittance transfer may be cancelled within 30 minutes.",
}


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The Transfers are Unauthorized") == ["transfer", "unauthor"]


def test_rrf_prefers_documents_ranked_high_in_both_lists():
    a = [Hit("x", 1), Hit("y", 1), Hit("z", 1)]
    b = [Hit("y", 1), Hit("x", 1), Hit("w", 1)]
    fused = [h.doc_id for h in rrf([a, b], k=60)]
    assert fused[:2] in (["x", "y"], ["y", "x"]) and fused[-1] in ("z", "w")


def test_bm25_pipeline_ranks_relevant_doc_first():
    ids = list(DOCS)
    pipe = RetrievalPipeline(
        {"sparse": True, "candidates": 4, "context_k": 2}, DOCS, bm25=BM25Index(ids, [DOCS[i] for i in ids])
    )
    res = pipe.search("liability for unauthorized transfers")
    assert res.ranking[0] == "d1" and len(res.context_ids) == 2


def test_crag_fallback_rewrites_then_abstains():
    ids = list(DOCS)
    prompts = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return "remittance cancellation" if "Rewrite" in prompt else "INCORRECT"

    pipe = RetrievalPipeline(
        {"sparse": True, "crag": True, "candidates": 2, "context_k": 1},
        DOCS,
        bm25=BM25Index(ids, [DOCS[i] for i in ids]),
        llm=llm,
    )
    res = pipe.search("what is the capital of France")
    assert res.rewritten_query == "remittance cancellation"
    assert res.grade == "incorrect" and res.context_ids == []


def test_parse_grade_defaults_to_ambiguous():
    assert parse_grade("Grade: CORRECT") == "correct"
    assert parse_grade("incorrect.") == "incorrect"
    assert parse_grade("not sure") == "ambiguous"
