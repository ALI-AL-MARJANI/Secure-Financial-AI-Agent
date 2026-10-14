import math

import pytest

from secfin.eval.ir_metrics import (
    bootstrap_ci,
    dedupe,
    graded_ndcg_at_k,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    recall_at_k,
)


def test_recall_mrr_ndcg_known_values():
    ranking = ["x", "a", "y", "b"]
    rel = {"a", "b"}
    assert recall_at_k(ranking, rel, 2) == 0.5
    assert recall_at_k(ranking, rel, 4) == 1.0
    assert mrr_at_k(ranking, rel, 10) == 0.5
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(ranking, rel, 10) == pytest.approx(dcg / idcg)
    assert graded_ndcg_at_k(ranking, {"a": 1.0, "b": 1.0}, 10) == pytest.approx(dcg / idcg)


def test_no_relevant_is_nan_and_miss_is_zero():
    assert math.isnan(recall_at_k(["a"], set(), 5))
    assert mrr_at_k(["a", "b"], {"z"}, 10) == 0.0


def test_dedupe_keeps_first_occurrence():
    assert dedupe(["p1", "p2", "p1", "p3"]) == ["p1", "p2", "p3"]


def test_bootstrap_ci_contains_mean():
    ci = bootstrap_ci([0.0, 1.0] * 50, n_boot=500)
    assert ci["lo"] <= 0.5 <= ci["hi"] and ci["n"] == 100


def test_paired_bootstrap_detects_consistent_gain():
    a = [0.2] * 60
    b = [0.6] * 60
    res = paired_bootstrap(a, b, n_boot=500)
    assert res["delta"] == pytest.approx(0.4) and res["p"] < 0.01
