"""Prompt-injection detectors: regex (v0 original, v1 extended) vs ProtectAI DeBERTa v2 vs cascade.

  python scripts/bench_injection.py

Thresholds: ProtectAI at its default 0.5, plus a threshold chosen on the dev split so that the false
positive rate on dev Banking77 queries is <= 1 %. Test numbers are reported for both.
Outputs results/guardrails/injection.json and injection_errors.jsonl.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections import defaultdict

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")

from secfin.eval.ir_metrics import latency_summary  # noqa: E402
from secfin.guardrails.injection import ProtectAIDetector, RegexInjectionDetector  # noqa: E402
from secfin.runinfo import REPO_ROOT, read_jsonl, run_metadata, write_json, write_jsonl  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> dict:
    if n == 0:
        return {"rate": float("nan"), "lo": float("nan"), "hi": float("nan"), "k": 0, "n": 0}
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return {"rate": p, "lo": max(0.0, centre - half), "hi": min(1.0, centre + half), "k": k, "n": n}


def mcnemar(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """Exact (binomial) McNemar test on discordant pairs."""
    from scipy.stats import binomtest

    b = sum(1 for x, y in zip(correct_a, correct_b, strict=True) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b, strict=True) if y and not x)
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "p": float(p)}


def report(rows: list[dict], flags: list[bool]) -> dict:
    by_source = defaultdict(lambda: [0, 0])
    tp = fn = fp = tn = 0
    for r, f in zip(rows, flags, strict=True):
        key = f"{r['source']}|{'attack' if r['label'] else 'benign'}"
        by_source[key][0] += int(f)
        by_source[key][1] += 1
        if r["label"]:
            tp += f
            fn += not f
        else:
            fp += f
            tn += not f
    benign_banking = [(r, f) for r, f in zip(rows, flags, strict=True) if r["category"] == "benign_banking"]
    return {
        "tpr": wilson(tp, tp + fn),
        "fpr": wilson(fp, fp + tn),
        "fpr_banking": wilson(sum(f for _, f in benign_banking), len(benign_banking)),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "flag_rate_by_source": {s: wilson(k, n) for s, (k, n) in sorted(by_source.items())},
    }


def threshold_for_fpr(scores: list[float], rows: list[dict], max_fpr: float) -> float:
    benign = sorted(
        (s for s, r in zip(scores, rows, strict=True) if r["category"] == "benign_banking"), reverse=True
    )
    k = int(math.floor(max_fpr * len(benign)))
    return float(np.nextafter(benign[k], 1.0)) if k < len(benign) else 0.5


def time_per_text(det, texts: list[str]) -> dict:
    det.scores(texts[:5])  # warm-up
    lat = []
    for t in texts:
        t0 = time.perf_counter()
        det.scores([t])
        lat.append(time.perf_counter() - t0)
    return latency_summary(lat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-dev-fpr", type=float, default=0.01)
    ap.add_argument("--latency-samples", type=int, default=200)
    args = ap.parse_args()

    import torch

    device = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    dev = read_jsonl(REPO_ROOT / "data/security/injection_dev.jsonl")
    test = read_jsonl(REPO_ROOT / "data/security/injection_test.jsonl")

    regex_v0, regex_v1 = RegexInjectionDetector("v0"), RegexInjectionDetector("v1")
    protect = ProtectAIDetector(device=device)
    scores = {split: {} for split in ("dev", "test")}
    for split, rows in (("dev", dev), ("test", test)):
        texts = [r["text"] for r in rows]
        for det in (regex_v0, regex_v1, protect):
            scores[split][det.name] = det.scores(texts)

    thr_dev = threshold_for_fpr(scores["dev"][protect.name], dev, args.max_dev_fpr)
    configs = {
        "regex_v0": lambda s: [x >= 0.5 for x in s["regex_v0"]],
        "regex_v1": lambda s: [x >= 0.5 for x in s["regex_v1"]],
        "protectai@0.5": lambda s: [x >= 0.5 for x in s[protect.name]],
        f"protectai@{thr_dev:.6g} (dev FPR<={args.max_dev_fpr:.0%})": lambda s: [
            x >= thr_dev for x in s[protect.name]
        ],
        "regex_v1 OR protectai@0.5": lambda s: [
            a >= 0.5 or b >= 0.5 for a, b in zip(s["regex_v1"], s[protect.name], strict=True)
        ],
    }
    results = {
        split: {name: report(rows, fn(scores[split])) for name, fn in configs.items()}
        for split, rows in (("dev", dev), ("test", test))
    }

    # Exclude the jackhhao rows (in ProtectAI's training data) for a contamination-free view.
    clean = [i for i, r in enumerate(test) if not r.get("in_protectai_training")]
    results["test_without_jackhhao"] = {
        name: report([test[i] for i in clean], [fn(scores["test"])[i] for i in clean])
        for name, fn in configs.items()
    }

    correct = {
        name: [f == bool(r["label"]) for f, r in zip(fn(scores["test"]), test, strict=True)]
        for name, fn in configs.items()
    }
    names = list(configs)
    significance = [
        {"a": a, "b": b, **mcnemar(correct[a], correct[b])}
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    ]

    rng = random.Random(0)
    sample = [r["text"] for r in rng.sample(test, args.latency_samples)]
    latency = {d.name: time_per_text(d, sample) for d in (regex_v1, protect)}

    errors = []
    for name, fn in configs.items():
        for f, r in zip(fn(scores["test"]), test, strict=True):
            if f != bool(r["label"]):
                errors.append({"detector": name, "error": "FP" if f else "FN", **r})
    write_jsonl(REPO_ROOT / "results/guardrails/injection_errors.jsonl", errors)

    write_json(
        REPO_ROOT / "results/guardrails/injection.json",
        {
            "meta": run_metadata(
                config=vars(args)
                | {"device": device, "protectai_model": ProtectAIDetector.MODEL, "dev_threshold": thr_dev},
                seed=0,
            ),
            "results": results,
            "mcnemar_test": significance,
            "latency_batch1": latency,
        },
    )
    for name, r in results["test"].items():
        print(
            f"{name:55s} TPR={r['tpr']['rate']:.3f}  FPR={r['fpr']['rate']:.4f}  FPR_banking={r['fpr_banking']['rate']:.4f}"
        )


if __name__ == "__main__":
    main()
