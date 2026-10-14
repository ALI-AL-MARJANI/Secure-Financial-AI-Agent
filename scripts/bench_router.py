"""Router accuracy: keyword-only vs LLM vs cascade, with confusion matrices and latency.

  python scripts/bench_router.py                    # keyword router only (no LLM)
  python scripts/bench_router.py --with-llm         # + LLM and cascade (Ollama)

The keyword router may abstain; its accuracy is reported both on the queries where it answers
(coverage) and with abstentions counted as errors.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

from secfin.agent.router import LABELS, ROUTER_PROMPT, CascadeRouter, KeywordRouter, LLMRouter
from secfin.eval.ir_metrics import bootstrap_ci, latency_summary
from secfin.runinfo import REPO_ROOT, load_config, read_jsonl, run_metadata, write_json


def evaluate(router, rows: list[dict]) -> dict:
    preds, lat = [], []
    for r in rows:
        t = time.perf_counter()
        d = router.route(r["text"])
        lat.append(time.perf_counter() - t)
        preds.append(d.label if d else None)
    correct = [p == r["label"] for p, r in zip(preds, rows, strict=True)]
    answered = [(p, r) for p, r in zip(preds, rows, strict=True) if p is not None]
    confusion = Counter(f"{r['label']}->{p}" for p, r in zip(preds, rows, strict=True))
    per_class = {}
    for lbl in LABELS:
        tp = sum(1 for p, r in zip(preds, rows, strict=True) if p == lbl and r["label"] == lbl)
        fp = sum(1 for p, r in zip(preds, rows, strict=True) if p == lbl and r["label"] != lbl)
        fn = sum(1 for p, r in zip(preds, rows, strict=True) if p != lbl and r["label"] == lbl)
        per_class[lbl] = {
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "support": tp + fn,
        }
    amb = [c for c, r in zip(correct, rows, strict=True) if r["ambiguous"]]
    return {
        "accuracy": bootstrap_ci([float(c) for c in correct]),
        "coverage": len(answered) / len(rows),
        "accuracy_when_answered": (sum(p == r["label"] for p, r in answered) / len(answered))
        if answered
        else float("nan"),
        "accuracy_ambiguous": sum(amb) / len(amb) if amb else float("nan"),
        "per_class": per_class,
        "confusion": dict(sorted(confusion.items())),
        "latency": latency_summary(lat),
        "errors": [
            {"text": r["text"], "gold": r["label"], "pred": p}
            for p, r, c in zip(preds, rows, correct, strict=True)
            if not c
        ][:40],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--config", default="configs/agent.yaml")
    args = ap.parse_args()
    rows = read_jsonl(REPO_ROOT / "data/router/router_queries.jsonl")
    out_path = REPO_ROOT / "results/agent/router.json"
    import json

    previous = json.loads(out_path.read_text())["results"] if out_path.exists() else {}
    routers = [KeywordRouter()]
    if args.with_llm:
        from secfin.llm import OllamaLLM

        cfg = load_config(REPO_ROOT / args.config)["llm"]
        llm = OllamaLLM(cfg["model"], cfg["temperature"], seed=0)
        routers += [LLMRouter(llm), CascadeRouter(llm)]
    results = dict(previous)
    for router in routers:
        for split in ("dev", "test"):
            res = evaluate(router, [r for r in rows if r["split"] == split])
            results[f"{router.name}|{split}"] = res
            print(
                f"{router.name:8s} {split:4s} acc={res['accuracy']['mean']:.3f} coverage={res['coverage']:.2f} "
                f"p50={res['latency']['p50_ms']:.1f}ms"
            )
    write_json(
        out_path, {"meta": run_metadata(config=vars(args), seed=0, prompt=ROUTER_PROMPT), "results": results}
    )


if __name__ == "__main__":
    main()
