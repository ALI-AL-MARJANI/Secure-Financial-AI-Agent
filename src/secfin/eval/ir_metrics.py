"""Binary-relevance IR metrics computed per query, plus bootstrap confidence intervals.

Rankings are lists of document IDs (already de-duplicated to the unit that qrels refer to).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def recall_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(ranking[:k]) & relevant) / len(relevant)


def hit_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return float(any(d in relevant for d in ranking[:k]))


def mrr_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    for i, d in enumerate(ranking[:k]):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranking[:k]) if d in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg


def graded_ndcg_at_k(ranking: Sequence[str], gains: dict[str, float], k: int) -> float:
    """nDCG with graded relevance (BEIR qrels use integer grades)."""
    if not gains:
        return float("nan")
    dcg = sum(gains.get(d, 0.0) / math.log2(i + 2) for i, d in enumerate(ranking[:k]))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def dedupe(ranking: Sequence[str]) -> list[str]:
    seen, out = set(), []
    for d in ranking:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def per_query_metrics(ranking: Sequence[str], relevant: set[str], ks: Sequence[int] = (5, 10, 20)) -> dict:
    out = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranking, relevant, k)
        out[f"hit@{k}"] = hit_at_k(ranking, relevant, k)
    out["mrr@10"] = mrr_at_k(ranking, relevant, 10)
    out["ndcg@10"] = ndcg_at_k(ranking, relevant, 10)
    return out


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    x = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if len(x) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(len(x)),
    }


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired bootstrap on per-query differences (b - a). p is the two-sided bootstrap p-value."""
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if not (math.isnan(x) or math.isnan(y))]
    d = np.asarray([y - x for x, y in pairs], dtype=float)
    if len(d) == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "delta": float(d.mean()),
        "lo": float(np.quantile(means, 0.025)),
        "hi": float(np.quantile(means, 0.975)),
        "p": float(min(1.0, p)),
        "n": int(len(d)),
    }


def latency_summary(seconds: Sequence[float]) -> dict:
    ms = np.asarray(seconds, dtype=float) * 1000
    return {
        "p50_ms": float(np.percentile(ms, 50)),
        "p90_ms": float(np.percentile(ms, 90)),
        "mean_ms": float(ms.mean()),
        "std_ms": float(ms.std(ddof=1)) if len(ms) > 1 else 0.0,
        "n": int(len(ms)),
    }
