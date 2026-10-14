# Annotation guide

Two manual tasks make the reported numbers credible. Both are resumable.

## 1. Golden-set review (`scripts/review_golden.py`, ~3 h for 201 items)

For each item the script shows the question, the reference answer, the gold passage text and the evidence quotes.

| Verdict | When |
|---|---|
| **a**ccept | The question is natural, the gold passages fully answer it, the reference answer is correct and complete. |
| **e**dit | Fixable: rephrase the question (prefer how a customer would ask it), correct the answer, or change gold passage IDs (comma-separated, must exist in `data/corpus/passages.jsonl`). |
| **r**eject | Ambiguous, trivially copies regulation wording, or depends on information outside the corpus. Give a reason. |

Checks for every item:
1. Is there another passage that answers the question equally well? If yes, add it to the gold list (otherwise retrieval is penalised for a correct hit).
2. For `unanswerable` items, search the corpus (`grep -i` on `passages.jsonl`) to confirm it really is not covered.
3. For `multi` items, confirm that each gold passage is needed.

After the pass: `python scripts/build_golden.py`, then rerun every benchmark (see RUNBOOK.md). Report the number of
edited / rejected items in the README.

## 2. Faithfulness labels for judge validation (`scripts/judge_agreement.py export`, ~2 h for 50 answers)

Open `docs/annotation/faithfulness_sample.csv`. The judge's scores are not in the file; label before looking at them.

Read the numbered contexts, then the answer. Split the answer into factual claims (numbers, deadlines, conditions,
yes/no statements). Ignore citation markers and style.

| `human` | Meaning |
|---|---|
| `1` | Every claim is supported by the contexts. |
| `0.5` | At least one claim is supported and at least one is not (missing qualifier, wrong number, invented condition). |
| `0` | The main claim is unsupported or contradicted by the contexts. |

Use `notes` for the unsupported claim. Then `python scripts/judge_agreement.py score`.
Interpretation commonly used for weighted kappa: < 0.4 poor, 0.4-0.6 moderate, 0.6-0.8 substantial, > 0.8 strong.
If agreement is below 0.6, report RAGAS faithfulness as unreliable for this setup instead of as a headline number.
