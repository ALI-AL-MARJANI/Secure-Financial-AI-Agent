"""Validate data/golden/questions.yaml against the corpus and write golden.jsonl with a dev/test split.

Checks: unique IDs, gold passages exist, every evidence quote occurs verbatim in a gold passage,
unanswerable items have no gold. Review status comes from data/golden/review.jsonl (if present).
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter

import yaml

from secfin.runinfo import REPO_ROOT, read_jsonl, write_json, write_jsonl


def _norm(s: str) -> str:
    return " ".join(s.split())


def validate(items: list[dict], passages: dict[str, dict]) -> list[str]:
    errors = []
    ids = Counter(it["id"] for it in items)
    errors += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    for it in items:
        if it["type"] == "unanswerable":
            if it["gold"]:
                errors.append(f"{it['id']}: unanswerable item has gold passages")
            continue
        if not it["gold"]:
            errors.append(f"{it['id']}: no gold passages")
        missing = [g for g in it["gold"] if g not in passages]
        errors += [f"{it['id']}: unknown passage {g}" for g in missing]
        gold_text = _norm(" ".join(passages[g]["text"] for g in it["gold"] if g in passages))
        for ev in it["evidence"]:
            if _norm(ev) not in gold_text:
                errors.append(f"{it['id']}: evidence not found verbatim: {ev[:70]!r}")
        if it["type"] == "multi" and len(it["gold"]) < 2:
            errors.append(f"{it['id']}: multi item needs >= 2 gold passages")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/golden/questions.yaml")
    ap.add_argument("--passages", default="data/corpus/passages.jsonl")
    ap.add_argument("--review", default="data/golden/review.jsonl")
    ap.add_argument("--out", default="data/golden/golden.jsonl")
    ap.add_argument("--dev-fraction", type=float, default=1 / 3)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    spec = yaml.safe_load((REPO_ROOT / args.questions).read_text())
    items = spec["items"]
    passages = {p["passage_id"]: p for p in read_jsonl(REPO_ROOT / args.passages)}
    errors = validate(items, passages)
    if errors:
        print("\n".join(errors))
        sys.exit(f"{len(errors)} validation error(s)")

    review = {}
    review_path = REPO_ROOT / args.review
    if review_path.exists():
        review = {r["id"]: r for r in read_jsonl(review_path)}

    # Stratified split by question type; fixed seed so the split never changes silently.
    rng = random.Random(args.seed)
    split = {}
    for t in sorted({it["type"] for it in items}):
        ids = sorted(it["id"] for it in items if it["type"] == t)
        rng.shuffle(ids)
        n_dev = round(len(ids) * args.dev_fraction)
        split.update({i: "dev" for i in ids[:n_dev]})
        split.update({i: "test" for i in ids[n_dev:]})

    rows = []
    for it in items:
        r = review.get(it["id"], {})
        if r.get("verdict") == "reject":
            continue
        rows.append(
            {
                "id": it["id"],
                "type": it["type"],
                "question": r.get("question", it["question"]),
                "gold_passages": r.get("gold", it["gold"]),
                "evidence": it["evidence"],
                "reference_answer": r.get("answer", it["answer"]),
                "regulation": sorted({passages[g]["regulation"] for g in it["gold"]}),
                "split": split[it["id"]],
                "source": "llm_draft",
                "reviewed": r.get("verdict") in ("accept", "edit"),
            }
        )
    write_jsonl(REPO_ROOT / args.out, rows)
    stats = {
        "version": spec.get("version"),
        "n_items": len(rows),
        "by_type": Counter(r["type"] for r in rows),
        "by_split": Counter(r["split"] for r in rows),
        "by_split_type": Counter(f"{r['split']}/{r['type']}" for r in rows),
        "n_reviewed": sum(r["reviewed"] for r in rows),
        "n_rejected": sum(1 for r in review.values() if r.get("verdict") == "reject"),
        "seed": args.seed,
    }
    write_json(REPO_ROOT / "data/golden/stats.json", stats)
    print(stats)


if __name__ == "__main__":
    main()
