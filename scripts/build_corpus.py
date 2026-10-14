"""Download the pinned eCFR parts and write data/corpus/passages.jsonl + manifest.json."""

from __future__ import annotations

import argparse
from collections import Counter

from secfin.corpus.chunking import chunk_passage
from secfin.corpus.ecfr import download_part, parse_part, sha256_file
from secfin.runinfo import REPO_ROOT, load_config, run_metadata, write_json, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/corpus.yaml")
    args = ap.parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    ecfr = cfg["ecfr"]

    passages, sources = [], []
    for part, name in ecfr["parts"].items():
        xml = download_part(part, ecfr["date"], REPO_ROOT / cfg["raw_dir"], ecfr["title"])
        ps = parse_part(xml, part)
        passages.extend(p.to_dict() for p in ps)
        sources.append(
            {
                "part": part,
                "name": name,
                "url": f"https://www.ecfr.gov/api/versioner/v1/full/{ecfr['date']}/title-{ecfr['title']}.xml?part={part}",
                "sha256": sha256_file(xml),
                "n_passages": len(ps),
            }
        )
        print(f"part {part}: {len(ps)} passages")

    write_jsonl(REPO_ROOT / cfg["passages"], passages)
    n_chunks = sum(len(chunk_passage(p, cfg["chunking"]["max_chars"])) for p in passages)
    n_words = sum(len(p["text"].split()) for p in passages)
    manifest = run_metadata(config=cfg)
    manifest.update(
        {
            "license": "Public domain (US federal regulations, 17 U.S.C. 105)",
            "sources": sources,
            "n_passages": len(passages),
            "n_chunks": n_chunks,
            "n_words": n_words,
            "approx_pages_500_words": round(n_words / 500),
            "passages_per_regulation": Counter(p["regulation"] for p in passages),
        }
    )
    write_json(REPO_ROOT / cfg["manifest"], manifest)
    print(f"{len(passages)} passages, {n_chunks} chunks, {n_words} words")


if __name__ == "__main__":
    main()
