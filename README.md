# Secure Financial AI Agent

A banking assistant that runs fully on a laptop. It answers questions about US banking rules with citations, can look
up account data through tools, and checks every step so a trick message can't reach the wrong customer's money.

The model is Mistral 7B, run locally with Ollama (a tool to run LLMs on your own machine). The agent's steps are wired
together with LangGraph (a library that runs an LLM app as a graph of steps).

## In short

- Three safety checks: one on the user's message, one on every tool call, one on the final answer.
- The best search trick was the cheapest: writing the rule's title in front of each piece of text gave +0.083 nDCG@10
  (a score for "is the right passage near the top?"). Heavier models made results worse.
- An ML attack detector blocked 4.97 % of normal customer questions, so the first check uses simple patterns and the
  later checks do the real work.
- The first version's evaluation could not be reproduced. Starting over with honest numbers changed the design.

![Three guard layers and what each one stopped](assets/guard_layers.png)

## How it started

In a bank, an LLM can't simply be plugged into customer data. That was the starting point. The first version was a
RAG demo (RAG = the model first searches a document collection, then answers from what it found) over a small
hand-written policy file. It answered questions, but it also repeated IBANs (bank account numbers) back to the user
and obeyed "ignore your instructions", a classic prompt injection.

## The first evaluation was wrong

v0.1 graded itself with RAGAS, a library where an LLM acts as a judge and scores a RAG system. It reported a context
precision of 0.087, but the score could not be reproduced: the script never named a judge model, so RAGAS silently
used OpenAI's API in a project meant to be 100 % local. Its 12 attack patterns, once tested on public attack sets,
caught only 23.0 % of attacks. v0.1 was deleted; the details are in [`docs/devlog.md`](docs/devlog.md).

## Starting over with real data

v0.2 uses real US banking regulations from the eCFR (the official online Code of Federal Regulations):

- **Reg E**: electronic transfers, debit cards, what happens when a card is stolen.
- **Reg DD**: savings accounts, interest rates and how they must be disclosed.
- **Reg CC**: how fast a bank must make deposited money, like a check, available.

They were split into 1,277 passages that can each be cited, like `1005.6(b)(1)`. A set of 201 test questions was
written, 30 of them unanswerable on purpose (the right reply is "I don't know"), each with its correct passages.

Before trusting this test set, the same search code was run on FiQA, a public finance question-answering benchmark
with known published scores. It got nDCG@10 0.403, exactly the published number, so the evaluation code was sound.

## The fancy stuff lost

Three ways to search were compared:

- **BM25**: classic keyword search, scores passages by shared words.
- **Dense search**: turns text into vectors (embeddings) and finds passages with a similar meaning.
- **Hybrid + reranker**: mix both lists, then re-score the top results with a cross-encoder, a bigger model that
  reads the question and each passage together. Slower, usually more accurate.

The expected winner was the hybrid + reranker setup. The chart puts accuracy against speed for each one:

![nDCG@10 vs latency for each retrieval variant](assets/retrieval_tradeoff.png)

Mixing in keyword search cost 0.074: BM25 is much weaker on legal text and pulls bad matches into the top 10. The
reranker was trained on web searches, not regulations, and added nothing while taking 182 ms instead of 7 ms.

What worked was one line of formatting. A passage like "(1) A local check;" means nothing alone, so each one is stored
as `Regulation CC, 12 CFR 229.12(b)(1) - <section title>` followed by its text. One caveat: the test questions were
drafted by an LLM from the passages, so their wording is close to the rules, which may help this trick.

## Simple patterns vs an ML detector

Prompt injection is a message that tries to make the model ignore its rules. Two detectors were tested on 313 public
attacks and 3,596 normal messages:

- **Regex** (hand-written text patterns): caught 0.355 of attacks, blocked 0 of 3,281 real banking questions.
- **DeBERTa** (ProtectAI's small classifier model): caught 0.796 of attacks, but blocked 4.97 % of real questions,
  like "What is the tracking number for the card you sent?".

Blocking about 1 customer in 20 is not acceptable for a bank, so the first check keeps the regex. That's fine
because the real protection comes later, and that is the core of the project:

- A tool call must use the logged-in customer's ID, so a request for someone else's balance is denied.
- Amounts above $10M wait for a human to approve.
- Every number in an answer must appear in the source text or the tool result. In live runs the main problem was
  not attackers but Mistral inventing figures, and those answers get flagged or withheld.

Getting Mistral 7B to call tools at all took three fixes: putting tool instructions in the user's message (it stops
calling tools when there's a system message), showing it only the current turn, and a small parser for the
`[TOOL_CALLS]` text it sometimes prints instead of a real call.

## What didn't work

- Hybrid search and the reranker, on both the regulations and FiQA.
- Questions that need several passages: only 0.559 of their passages land in the top 5, against 0.765 for questions
  with one passage.
- Three extra techniques (Contextual Retrieval, HyDE and CRAG, which use the LLM to improve search) are built but not
  tested yet, so no claim is made about them.

## Lessons

- Build the test set before the agent, and check how each score is computed.
- Reproduce a published number first. FiQA 0.403 vs 0.403 is the most boring and most useful result here.
- For a safety filter, how many normal customers it blocks matters as much as how many attacks it catches.
- Checking actions (whose account, how much, which number) still works when the message filter misses an attack.

## Next

- Run the full RAGAS evaluation locally, with a judge model checked against human grades.
- Have a human review the test questions.
- Test hidden attacks inside retrieved documents (indirect injection) on a public benchmark.

All numbers come from `results/`; [RUNBOOK.md](RUNBOOK.md) regenerates them.
