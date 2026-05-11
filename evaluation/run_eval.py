"""
Evaluation Pipeline — RAGAS + DeepEval
Runs the golden dataset through the RAG pipeline and scores:
  - Faithfulness (are answers grounded in policy?)
  - Answer Relevancy (does the answer address the question?)
  - Context Precision (are retrieved chunks relevant?)
  - Context Recall (were all relevant chunks retrieved?)

Usage:
    python evaluation/run_eval.py
"""
import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import retrieve_with_full_pipeline
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# Load golden dataset
# ---------------------------------------------------------------------------

DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")

with open(DATASET_PATH) as f:
    golden = json.load(f)

llm = ChatOllama(model="mistral", temperature=0)

# ---------------------------------------------------------------------------
# Generate answers using the RAG pipeline
# ---------------------------------------------------------------------------

def generate_answer(question: str) -> tuple[str, list[str]]:
    docs, grade = retrieve_with_full_pipeline(question, verbose=False)
    if not docs:
        return "No relevant information found.", []

    context = "\n\n".join([d.page_content for d in docs])
    prompt = (
        f"You are a banking assistant. Answer the question based only on the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely and factually."
    )
    answer = llm.invoke(prompt).content.strip()
    retrieved_contexts = [d.page_content for d in docs]
    return answer, retrieved_contexts


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def run_ragas_eval(samples: list) -> dict:
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

        data = {
            "question": [s["question"] for s in samples],
            "answer": [s["answer"] for s in samples],
            "contexts": [s["contexts"] for s in samples],
            "ground_truth": [s["ground_truth"] for s in samples],
        }
        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        return dict(result)
    except Exception as e:
        print(f"[RAGAS] Evaluation error: {e}")
        return {}


# ---------------------------------------------------------------------------
# DeepEval evaluation
# ---------------------------------------------------------------------------

def run_deepeval_tests(samples: list):
    try:
        from deepeval import evaluate as dv_evaluate
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            HallucinationMetric,
        )

        test_cases = []
        for s in samples:
            tc = LLMTestCase(
                input=s["question"],
                actual_output=s["answer"],
                expected_output=s["ground_truth"],
                retrieval_context=s["contexts"],
            )
            test_cases.append(tc)

        metrics = [
            AnswerRelevancyMetric(threshold=0.6, model="local"),
            FaithfulnessMetric(threshold=0.6, model="local"),
            HallucinationMetric(threshold=0.4, model="local"),
        ]

        dv_evaluate(test_cases, metrics=metrics)
    except Exception as e:
        print(f"[DeepEval] Evaluation error: {e}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*60}")
    print("  SecureBank RAG Evaluation Pipeline")
    print(f"  Dataset: {len(golden)} questions")
    print(f"{'='*60}\n")

    samples = []
    for i, item in enumerate(golden):
        print(f"[{i+1:02d}/{len(golden)}] {item['question'][:60]}...")
        answer, contexts = generate_answer(item["question"])
        samples.append({
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })
        print(f"       Answer: {answer[:80]}...")

    # Save generated answers
    out_path = os.path.join(os.path.dirname(__file__), "generated_answers.json")
    with open(out_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"\n[Saved] Generated answers → {out_path}")

    # RAGAS
    print("\n--- Running RAGAS metrics ---")
    ragas_results = run_ragas_eval(samples)
    if ragas_results:
        print("\nRAGAS Results:")
        for metric, score in ragas_results.items():
            print(f"  {metric:25} {score:.4f}")

        ragas_out = os.path.join(os.path.dirname(__file__), "ragas_results.json")
        with open(ragas_out, "w") as f:
            json.dump(ragas_results, f, indent=2)
        print(f"\n[Saved] RAGAS results → {ragas_out}")

    # DeepEval
    print("\n--- Running DeepEval tests ---")
    run_deepeval_tests(samples)

    print("\n[Done] Evaluation complete.")


if __name__ == "__main__":
    main()
