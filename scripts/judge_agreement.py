"""Validate the LLM judge against human labels.

  python scripts/judge_agreement.py export      # writes docs/annotation/faithfulness_sample.csv (50 answers)
  # ... annotate the `human` column with 1 (fully supported), 0.5 (partly), 0 (unsupported) ...
  python scripts/judge_agreement.py score       # writes results/e2e/judge_agreement.json

The judge's faithfulness score is binned to the same 3 levels (1.0 -> 1, >= 0.5 -> 0.5, else 0).
Reported: quadratic-weighted Cohen's kappa, exact agreement, Spearman correlation on the raw score.
"""

from __future__ import annotations

import argparse
import csv
import random

from secfin.runinfo import REPO_ROOT, load_config, read_jsonl, run_metadata, write_json

SAMPLE = REPO_ROOT / "docs/annotation/faithfulness_sample.csv"


def _slug(s: str) -> str:
    return s.replace(":", "-").replace("/", "_")


def export(cfg: dict, n: int, seed: int) -> None:
    judge = cfg["judge"]["model"]
    pool = []
    for gen in cfg["generators"]:
        jpath = REPO_ROOT / f"results/e2e/judged_{_slug(gen)}__{_slug(judge)}.jsonl"
        apath = REPO_ROOT / f"results/e2e/answers_{_slug(gen)}.jsonl"
        if not jpath.exists():
            continue
        answers = {r["id"]: r for r in read_jsonl(apath)}
        for j in read_jsonl(jpath):
            f = j["scores"].get("faithfulness")
            if f is not None:
                pool.append((gen, answers[j["id"]], f))
    if len(pool) < n:
        raise SystemExit(f"only {len(pool)} judged answers available; run the judge stage first")
    rng = random.Random(seed)
    # Stratify on the judge score so that low-faithfulness answers are not under-represented.
    low = [p for p in pool if p[2] < 1.0]
    high = [p for p in pool if p[2] >= 1.0]
    k_low = min(len(low), n // 2)
    sample = rng.sample(low, k_low) + rng.sample(high, n - k_low)
    rng.shuffle(sample)
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "question", "contexts", "answer", "human", "notes"])
        for gen, a, _ in sample:
            ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(a["contexts"]))
            w.writerow([f"{gen}|{a['id']}", a["question"], ctx, a["answer"], "", ""])
    print(f"wrote {SAMPLE.relative_to(REPO_ROOT)} ({n} rows). Judge scores are deliberately not in the file.")


def _bin(x: float) -> float:
    return 1.0 if x >= 0.999 else 0.5 if x >= 0.5 else 0.0


def score(cfg: dict) -> None:
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score

    judge = cfg["judge"]["model"]
    with open(SAMPLE) as f:
        rows = list(csv.DictReader(f))
    human, raw = [], []
    for r in rows:
        if r["human"].strip() == "":
            raise SystemExit(f"row {r['key']} is not annotated")
        gen, qid = r["key"].split("|")
        j = {
            x["id"]: x
            for x in read_jsonl(REPO_ROOT / f"results/e2e/judged_{_slug(gen)}__{_slug(judge)}.jsonl")
        }[qid]
        human.append(float(r["human"]))
        raw.append(j["scores"]["faithfulness"])
    judge_bin = [_bin(x) for x in raw]
    to_int = {0.0: 0, 0.5: 1, 1.0: 2}
    kappa = cohen_kappa_score([to_int[h] for h in human], [to_int[j] for j in judge_bin], weights="quadratic")
    rho, p = spearmanr(human, raw)
    out = {
        "n": len(rows),
        "judge": judge,
        "weighted_kappa": float(kappa),
        "exact_agreement": sum(h == j for h, j in zip(human, judge_bin, strict=True)) / len(rows),
        "spearman": float(rho),
        "spearman_p": float(p),
        "human_mean": sum(human) / len(human),
        "judge_mean": sum(raw) / len(raw),
    }
    write_json(
        REPO_ROOT / "results/e2e/judge_agreement.json", {"meta": run_metadata(config=cfg["judge"]), **out}
    )
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["export", "score"])
    ap.add_argument("--config", default="configs/e2e.yaml")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    export(cfg, args.n, args.seed) if args.action == "export" else score(cfg)


if __name__ == "__main__":
    main()
