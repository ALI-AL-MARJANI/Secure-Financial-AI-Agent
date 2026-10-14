# Data

| Path | What | Source | License | Built by |
|---|---|---|---|---|
| `corpus/passages.jsonl` | 1,277 citable passages (~249k words, ~500 pages) from 12 CFR Part 1005 (Regulation E), Part 1030 (Regulation DD), Part 229 (Regulation CC), incl. appendices and official interpretations | eCFR versioner API, issue date 2026-10-05 (URLs and SHA-256 of the raw XML in `corpus/manifest.json`) | Public domain (17 U.S.C. 105) | `scripts/build_corpus.py` |
| `golden/questions.yaml` | 201 questions (146 single-passage, 25 multi-passage, 30 unanswerable) with gold passage IDs, verbatim evidence quotes and reference answers | Drafted with an LLM from the passages; every evidence quote verified verbatim against the corpus | MIT (this repo) | hand-authored, validated by `scripts/build_golden.py` |
| `golden/golden.jsonl` | Validated golden set with a seeded, type-stratified dev (67) / test (134) split | derived | MIT | `scripts/build_golden.py` |
| `security/injection_{dev,test}.jsonl` | Prompt-injection evaluation sets (test: 313 attacks, 3,596 benign) | deepset/prompt-injections (Apache-2.0), Lakera/gandalf_ignore_instructions (MIT), jackhhao/jailbreak-classification (Apache-2.0), PolyAI Banking77 (CC BY 4.0), golden questions | as listed | `scripts/build_security_sets.py` |
| `security/pii_synthetic.jsonl` | 600 templated EN/FR messages with character-level PII spans (seed 7) | synthetic | MIT | `scripts/build_pii_set.py` |
| `security/pii_stress.jsonl` | 30 hand-written messages in formats the templates do not produce | hand-written | MIT | - |
| `router/handwritten.yaml`, `router/router_queries.jsonl` | 301 routing queries (RAG 214 / TOOL 45 / OUT_OF_SCOPE 42), seeded dev/test split | golden questions + hand-written | MIT | `scripts/build_router_set.py` |
| `agent/tool_cases.yaml` | 20 tool-selection cases, 10 red-team cases, 3 indirect-injection cases | hand-written | MIT | - |
| `demo/accounts.json` | three fictional demo accounts used by the calculators | fictional | MIT | - |

Not committed (rebuilt on demand): `raw/` (eCFR XML), `cache/` (FiQA download, embedding caches, logs).

## Known issues with the data

- The golden questions were drafted with an LLM that had the passages in front of it, so they tend to reuse
  regulation vocabulary, which favours lexical overlap and the citation-title prefix. `scripts/review_golden.py`
  supports a manual review pass.
- The keyword router, the regex PII detector, and the routing / PII / tool cases were written by the same author.
  Benchmarks on these sets are optimistic for the hand-written components; the public sets (FiQA, injection
  datasets, Banking77) are the independent checks.
- `deepset/prompt-injections` labels role-play and off-task requests ("act as an interviewer") as injections and
  contains German text; ProtectAI's model card states it is English-only.
