"""Build data/router/router_queries.jsonl from the golden questions (RAG / OUT_OF_SCOPE) and the
hand-written queries in data/router/handwritten.yaml, with a seeded dev/test split."""

from __future__ import annotations

import random
from collections import Counter

import yaml

from secfin.runinfo import REPO_ROOT, read_jsonl, write_json, write_jsonl

# Golden "unanswerable" items that ask for investment picks are out of scope for the assistant.
GOLDEN_OOS = {"U026", "U027"}


def main() -> None:
    rows = []
    for g in read_jsonl(REPO_ROOT / "data/golden/golden.jsonl"):
        rows.append(
            {
                "text": g["question"],
                "label": "OUT_OF_SCOPE" if g["id"] in GOLDEN_OOS else "RAG",
                "source": f"golden:{g['id']}",
                "ambiguous": False,
            }
        )
    hand = yaml.safe_load((REPO_ROOT / "data/router/handwritten.yaml").read_text())
    for label, items in hand.items():
        for it in items:
            it = it if isinstance(it, dict) else {"text": it}
            rows.append(
                {
                    "text": it["text"],
                    "label": label,
                    "source": "handwritten",
                    "ambiguous": bool(it.get("ambiguous")),
                }
            )
    rng = random.Random(3)
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["split"] = "dev" if i % 3 == 0 else "test"
    write_jsonl(REPO_ROOT / "data/router/router_queries.jsonl", rows)
    stats = {
        "n": len(rows),
        "by_label": Counter(r["label"] for r in rows),
        "by_split": Counter(r["split"] for r in rows),
        "ambiguous": sum(r["ambiguous"] for r in rows),
    }
    write_json(REPO_ROOT / "data/router/stats.json", stats)
    print(stats)


if __name__ == "__main__":
    main()
