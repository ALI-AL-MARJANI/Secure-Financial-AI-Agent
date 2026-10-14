"""Generate Contextual Retrieval prefixes (Anthropic, 2024) for every eCFR chunk with a local LLM.

  python scripts/build_contextual_chunks.py --model mistral          # ~2,300 LLM calls

Resumable: chunks already present in the output file are skipped. The "document" given to the LLM is
the enclosing regulation section (all passages that share the section number), truncated to
`--max-doc-chars`, because a whole eCFR part does not fit in a 7B model's context window.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict

from secfin.llm import OllamaLLM
from secfin.retrieval.datasets import load_ecfr_chunks
from secfin.retrieval.llm_components import chunk_context
from secfin.runinfo import REPO_ROOT, read_jsonl, run_metadata, write_json


def section_key(passage_id: str) -> str:
    # "1005.6(b)(1)" -> "1005.6"; "1005.SuppI.028" -> "1005.SuppI" + heading handled by title below
    m = re.match(r"^(\d+\.\d+)", passage_id)
    return m.group(1) if m else passage_id.rsplit(".", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral")
    ap.add_argument("--out", default="data/corpus/contextual_mistral.jsonl")
    ap.add_argument("--max-doc-chars", type=int, default=12000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    chunks, passages = load_ecfr_chunks(prefix="none")
    # Interpretation passages are grouped by their first heading (the "Section 1005.x" line).
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in passages:
        key = (
            section_key(p["passage_id"])
            if p["passage_id"][0].isdigit() and "Supp" not in p["passage_id"] and "App" not in p["passage_id"]
            else f"{p['part']}|{p['headings'][1] if len(p['headings']) > 1 else p['headings'][0]}"
        )
        groups[key].append(p)
    doc_of = {}
    for ps in groups.values():
        text = f"{ps[0]['regulation']} - {ps[0]['headings'][0]}\n" + "\n".join(p["text"] for p in ps)
        for p in ps:
            doc_of[p["passage_id"]] = text

    out = REPO_ROOT / args.out
    done = {r["chunk_id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [c for c in chunks if c.chunk_id not in done]
    if args.limit:
        todo = todo[: args.limit]
    llm = OllamaLLM(args.model, temperature=0.0, seed=0, num_ctx=8192)
    print(f"{len(done)} cached, {len(todo)} to generate with {args.model}")
    t0 = time.perf_counter()
    with open(out, "a") as f:
        for i, c in enumerate(todo, 1):
            ctx = chunk_context(llm, doc_of[c.passage_id], c.text, args.max_doc_chars)
            f.write(json.dumps({"chunk_id": c.chunk_id, "context": ctx}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                rate = (time.perf_counter() - t0) / i
                print(f"{i}/{len(todo)}  {rate:.1f}s/chunk  ETA {rate * (len(todo) - i) / 60:.0f} min")
    write_json(
        REPO_ROOT / "results/retrieval/contextual_build.json",
        run_metadata(
            config=vars(args),
            seed=0,
            n_chunks=len(chunks),
            seconds=time.perf_counter() - t0,
            n_generated=len(todo),
        ),
    )


if __name__ == "__main__":
    main()
