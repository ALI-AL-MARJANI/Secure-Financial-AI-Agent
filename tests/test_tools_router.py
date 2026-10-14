import pytest

from secfin.agent.router import CascadeRouter, KeywordRouter, parse_label
from secfin.agent.tools import Bank, monthly_payment
from secfin.runinfo import REPO_ROOT, load_config

TERMS = load_config(REPO_ROOT / "configs/agent.yaml")["demo_bank"]


def bank() -> Bank:
    return Bank(TERMS, REPO_ROOT / TERMS["accounts_file"])


def test_monthly_payment_known_values():
    # 300,000 at 5% over 25 years: standard amortisation table value 1,753.77
    assert monthly_payment(300_000, 5.0, 25) == pytest.approx(1753.77, abs=0.01)
    assert monthly_payment(120_000, 0.0, 10) == pytest.approx(1000.0)


def test_simulate_mortgage_totals_are_consistent():
    r = bank().simulate_mortgage(300_000, 5.0, 25)
    assert r["total_paid"] == pytest.approx(r["monthly_payment"] * 300, abs=1.0)
    assert r["total_interest"] == pytest.approx(r["total_paid"] - 300_000, abs=0.01)


def test_affordability_reports_each_factor():
    r = bank().calculate_loan_affordability(
        monthly_income=7000, monthly_debt=800, credit_score=720, requested_principal=350_000
    )
    by = {f["factor"]: f for f in r["factors"]}
    # (800 + 2046.07) / 7000 = 0.4066 <= 0.43
    assert by["debt_to_income"]["value"] == pytest.approx(0.4066, abs=1e-4)
    assert r["eligible"] is True
    low = bank().calculate_loan_affordability(7000, 800, 650, 350_000)
    assert low["eligible"] is False and not {f["factor"]: f for f in low["factors"]}["credit_score"]["passed"]


def test_overdraft_risk():
    r = bank().check_overdraft_risk("CUST789", 200.0)
    assert r["overdraft"] and r["deposit_needed_to_avoid"] == pytest.approx(49.25)
    assert not bank().check_overdraft_risk("CUST123", 100.0)["overdraft"]


def test_keyword_router_abstains_when_unsure():
    kw = KeywordRouter()
    assert kw.route("What's my balance right now?").label == "TOOL"
    assert kw.route("Write me a poem about money").label == "OUT_OF_SCOPE"
    assert kw.route("What disclosures are required for a CD?").label == "RAG"
    assert kw.route("hello there") is None


def test_parse_label_and_cascade_fallback():
    assert parse_label("Label: OUT_OF_SCOPE") == "OUT_OF_SCOPE"
    assert parse_label("I think TOOL") == "TOOL"
    assert parse_label("???") == "RAG"
    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "TOOL"

    router = CascadeRouter(fake_llm)
    assert router.route("Run the numbers for me").method == "llm" and calls
    assert router.route("Tell me a joke").method == "keyword"
