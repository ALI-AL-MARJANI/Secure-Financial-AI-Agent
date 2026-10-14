"""Assemble the prompt-injection evaluation sets from public datasets (written to data/security/).

dev  : deepset/prompt-injections train, Lakera/gandalf_ignore_instructions validation,
       1,000 Banking77 train queries (benign, seeded sample)
test : deepset/prompt-injections test, Lakera/gandalf test, jackhhao/jailbreak-classification test,
       Banking77 test (3,080 benign banking queries), this repo's golden questions (benign, in-domain)

Labels: 1 = attack (injection / jailbreak), 0 = benign.
"""

from __future__ import annotations

import csv
import io
import random
import re

import httpx
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from secfin.runinfo import REPO_ROOT, read_jsonl, write_json, write_jsonl

CACHE = REPO_ROOT / "data/cache/hf"
BANKING77 = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{split}.csv"
)
_GERMAN = re.compile(r"\b(und|der|die|das|ist|nicht|ich|wie|mir|bitte|vergiss|alle|sie)\b", re.IGNORECASE)


def _parquet(repo: str, split: str) -> list[dict]:
    files = [
        f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet") and f"/{split}-" in f
    ]
    rows = []
    for f in files:
        rows += pq.read_table(hf_hub_download(repo, f, repo_type="dataset", cache_dir=CACHE)).to_pylist()
    return rows


def _row(text: str, label: int, source: str, category: str, **kw) -> dict:
    lang = "de" if len(_GERMAN.findall(text)) >= 2 else "en"
    return {"text": text, "label": label, "source": source, "category": category, "lang": lang, **kw}


def deepset(split: str) -> list[dict]:
    return [
        _row(
            r["text"],
            int(r["label"]),
            f"deepset/prompt-injections:{split}",
            "injection" if int(r["label"]) else "benign_general",
        )
        for r in _parquet("deepset/prompt-injections", split)
    ]


def gandalf(split: str) -> list[dict]:
    return [
        _row(r["text"], 1, f"Lakera/gandalf_ignore_instructions:{split}", "injection")
        for r in _parquet("Lakera/gandalf_ignore_instructions", split)
    ]


def jackhhao() -> list[dict]:
    f = hf_hub_download(
        "jackhhao/jailbreak-classification",
        "default/jailbreak_dataset_test.csv",
        repo_type="dataset",
        cache_dir=CACHE,
    )
    rows = []
    with open(f) as fh:
        for r in csv.DictReader(fh):
            is_jb = r["type"].strip().lower() == "jailbreak"
            # ProtectAI lists this dataset among its training data: results on it are flagged.
            rows.append(
                _row(
                    r["prompt"],
                    int(is_jb),
                    "jackhhao/jailbreak-classification:test",
                    "jailbreak" if is_jb else "benign_general",
                    in_protectai_training=True,
                )
            )
    return rows


def banking77(split: str) -> list[dict]:
    text = httpx.get(BANKING77.format(split=split), timeout=60).text
    return [
        _row(r["text"], 0, f"PolyAI/banking77:{split}", "benign_banking", intent=r["category"])
        for r in csv.DictReader(io.StringIO(text))
    ]


def main() -> None:
    rng = random.Random(0)
    b77_train = banking77("train")
    rng.shuffle(b77_train)
    dev = deepset("train") + gandalf("validation") + b77_train[:1000]
    golden = [
        _row(g["question"], 0, "secfin/golden", "benign_banking")
        for g in read_jsonl(REPO_ROOT / "data/golden/golden.jsonl")
    ]
    test = deepset("test") + gandalf("test") + jackhhao() + banking77("test") + golden
    out = REPO_ROOT / "data/security"
    write_jsonl(out / "injection_dev.jsonl", dev)
    write_jsonl(out / "injection_test.jsonl", test)

    def stats(rows):
        from collections import Counter

        return {"n": len(rows), "by_source_label": Counter(f"{r['source']}|{r['label']}" for r in rows)}

    write_json(
        out / "manifest.json",
        {
            "dev": stats(dev),
            "test": stats(test),
            "licenses": {
                "deepset/prompt-injections": "Apache-2.0",
                "Lakera/gandalf_ignore_instructions": "MIT",
                "jackhhao/jailbreak-classification": "Apache-2.0",
                "PolyAI/banking77": "CC-BY-4.0",
            },
            "notes": "Gandalf train split (777 rows) is not used. jackhhao is in ProtectAI's training data.",
        },
    )
    print(stats(dev)["n"], stats(test)["n"])


if __name__ == "__main__":
    main()
