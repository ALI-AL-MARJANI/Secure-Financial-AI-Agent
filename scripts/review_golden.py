"""Interactive human review of the LLM-drafted golden set. Appends verdicts to data/golden/review.jsonl.

  python scripts/review_golden.py            # resumes where you stopped
  python scripts/review_golden.py --only M   # review only IDs starting with M

Verdicts: a = accept, e = edit (question / answer / gold), r = reject, s = skip, q = quit.
See docs/annotation_guide.md for the criteria.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import textwrap

import yaml

from secfin.runinfo import REPO_ROOT, read_jsonl


def show(it: dict, passages: dict[str, dict]) -> None:
    print("\n" + "=" * 100)
    print(f"{it['id']}  [{it['type']}]")
    print(f"Q: {it['question']}")
    print(f"A: {it['answer']}")
    for g in it["gold"]:
        p = passages[g]
        print(f"\n--- {g} | {p['regulation']} | {p['title'][:80]}")
        print(textwrap.fill(p["text"][:1500], 100))
    if it["evidence"]:
        print("\nEvidence quotes:", *[f"  - {e}" for e in it["evidence"]], sep="\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--reviewer", default="author")
    args = ap.parse_args()

    items = yaml.safe_load((REPO_ROOT / "data/golden/questions.yaml").read_text())["items"]
    passages = {p["passage_id"]: p for p in read_jsonl(REPO_ROOT / "data/corpus/passages.jsonl")}
    review_path = REPO_ROOT / "data/golden/review.jsonl"
    done = {r["id"] for r in read_jsonl(review_path)} if review_path.exists() else set()
    todo = [it for it in items if it["id"] not in done and it["id"].startswith(args.only)]
    print(f"{len(done)} reviewed, {len(todo)} to go")

    with open(review_path, "a") as out:
        for it in todo:
            show(it, passages)
            while True:
                v = input("\n[a]ccept [e]dit [r]eject [s]kip [q]uit > ").strip().lower()
                if v in ("a", "e", "r", "s", "q"):
                    break
            if v == "q":
                return
            if v == "s":
                continue
            rec = {
                "id": it["id"],
                "verdict": {"a": "accept", "e": "edit", "r": "reject"}[v],
                "reviewer": args.reviewer,
                "date": dt.date.today().isoformat(),
            }
            if v == "e":
                q = input("new question (enter = keep): ").strip()
                a = input("new answer (enter = keep): ").strip()
                g = input("new gold passage IDs, comma-separated (enter = keep): ").strip()
                if q:
                    rec["question"] = q
                if a:
                    rec["answer"] = a
                if g:
                    rec["gold"] = [x.strip() for x in g.split(",")]
            if v == "r":
                rec["reason"] = input("reason: ").strip()
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
