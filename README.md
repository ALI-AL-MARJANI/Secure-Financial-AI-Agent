# Secure Financial AI Agent

A banking assistant that runs fully on a laptop (Mistral 7B through Ollama, LangGraph). It answers questions about US
banking regulations with citations, and checks every tool call before it touches an account.

## TL;DR

- Three guard layers: one on the user's message, one on every tool call, one on the answer.
- The best retrieval trick was the cheapest: putting the citation title in front of each chunk gave +0.083 nDCG@10.
  Hybrid BM25 and a reranker made things worse.
- The ML injection detector flagged 4.97 % of real customer questions, so the input keeps regex and the later guards
  do the real work.
- The first version's README had metrics that couldn't be reproduced. Fixing that changed what the project believed
  about its own system.

▶️ 2-min walkthrough: <VIDEO_LINK>

## How it started

In a bank, an LLM can't simply be plugged into customer data. That was the starting point. The first version was a
LangGraph demo over a small hand-written policy file (8 KB), with Mistral running locally in Ollama. It answered
questions, but it also repeated IBANs back to the user and obeyed "ignore your instructions".

Regex filters and a tool check patched the obvious holes. The README of that version called the project
"production-grade" and listed gains taken from the papers it implemented, none of them measured on this code.

## The metric that lied: 0.087

That README showed RAGAS context recall 0.920 and context precision 0.087, and explained the gap as "the chunks are
longer than the answers". That was wrong: context precision is scored by an LLM judge that decides if each retrieved
chunk is relevant. Trying to rerun it showed the script never passed a judge, so RAGAS had quietly used OpenAI in a
"100 % local" project, and neither the version nor the outputs had been logged.

The recall was hollow too. Five chunks from an 8 KB file is about a third of the knowledge base for every question.
And once the 12 injection patterns were tested on public attacks, they caught 23.0 %. v0.1 was thrown away, with the
reasons written down in [`docs/devlog.md`](docs/devlog.md).

## Starting over with a real corpus

v0.2 uses real regulations: Reg E, DD and CC from the eCFR API, parsed into 1,277 citable passages. The question set
has 201 questions, 30 of them unanswerable on purpose, each with its gold passages and a quote that a script checks
against the text.

Before trusting the eval code, the same retriever was run on FiQA, a public finance benchmark: nDCG@10 0.403, the same
as the published number. (nDCG@10 rewards putting the right passage near the top of the first 10 results.)

## The fancy stuff lost

The expected winner was BM25 + dense fusion with a cross-encoder reranker (a model that reads the question and the
passage together). Here is the test split, 114 answerable questions:

| Retriever | nDCG@10 | p50 latency |
|---|---|---|
| BM25 | 0.340 | 1 ms |
| dense (BGE-small) | 0.528 | 7 ms |
| dense + reranker | 0.519 | 182 ms |
| hybrid BM25 + dense | 0.454 | 9 ms |
| dense + citation title | **0.611** | 7 ms |

Fusion cost 0.074: BM25 is much weaker on this text and drags its bad hits into the top 10. The reranker was trained
on web search queries and added nothing for 182 ms. What worked was indexing each chunk as
`Regulation CC, 12 CFR 229.12(b)(1) - <section title>` followed by its text, because a chunk like "(1) A local check;"
means nothing on its own. One caveat: the questions were drafted by an LLM from the passages, so their wording is
close to the regulation, which may help this trick.

## Regex vs DeBERTa

ProtectAI's DeBERTa classifier caught 0.796 of 313 public attacks, against 0.355 for the regex. But it also flagged
4.97 % of real banking questions, things like "What is the tracking number for the card you sent?". The regex flagged
0 of 3,281. For a bank, blocking about 1 customer in 20 is not an option, so the input keeps the regex.

That's fine because the real protection comes later, and that is the core of the project. A tool call must use the
logged-in customer's ID, so a prompt asking for someone else's balance gets denied. Amounts above $10M wait for human
approval. And every number in an answer must appear in the retrieved text or the tool output. In live runs, the
problem wasn't attackers, it was Mistral inventing figures, for example after an approval was declined.

Getting Mistral 7B to call tools at all took three fixes: tool instructions go in the user turn (it stops calling
tools when there's a system message), the tool step only sees the current turn, and a small parser catches the
`[TOOL_CALLS]` text it sometimes prints instead of a real call.

## What didn't work

- Hybrid search and the reranker, on both the regulation corpus and FiQA.
- Questions that need several passages: Recall@5 0.559, against 0.765 for single-passage ones.
- Contextual Retrieval, HyDE and CRAG are implemented but not benchmarked yet, so no claim is made about them.

## Lessons

- Build the eval set before the agent, and check how each metric is computed. This one lied (0.087).
- Reproduce a public number first. FiQA 0.403 vs 0.403 is the most boring and most useful result here.
- For a safety filter, false positives on real customers matter as much as the catch rate.
- Checking actions (whose account, how much, which number) holds even when the prompt filter misses.

## Next

- Run the local RAGAS eval end to end, with a judge checked against human labels.
- Get the test questions reviewed by a human.
- Test indirect injection (attacks hidden in retrieved documents) on a public benchmark.

All numbers come from `results/`; [RUNBOOK.md](RUNBOOK.md) regenerates them.
