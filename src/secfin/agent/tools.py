"""Deterministic financial tools over fictional demo data. Pure functions so they can be unit-tested
against hand-computed values; LangChain tool wrappers are built in `make_tools`."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import StructuredTool


def monthly_payment(principal: float, annual_rate_pct: float, years: int) -> float:
    """Standard amortisation: P * r(1+r)^n / ((1+r)^n - 1), r = monthly rate."""
    n = int(years) * 12
    r = annual_rate_pct / 100 / 12
    if r == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


class Bank:
    def __init__(self, terms: dict, accounts_file: str | Path):
        self.terms = terms
        self.accounts = json.loads(Path(accounts_file).read_text())["accounts"]

    def get_account_balance(self, customer_id: str) -> dict:
        acc = self.accounts.get(customer_id)
        if acc is None:
            return {"error": "customer not found"}
        return {"customer_id": customer_id, "type": acc["type"], "balance": acc["balance"]}

    def get_spending_summary(self, customer_id: str) -> dict:
        acc = self.accounts.get(customer_id)
        if acc is None:
            return {"error": "customer not found"}
        return {
            "customer_id": customer_id,
            "by_category": acc["spending"],
            "total": round(sum(acc["spending"].values()), 2),
        }

    def check_overdraft_risk(self, customer_id: str, planned_spend: float) -> dict:
        acc = self.accounts.get(customer_id)
        if acc is None:
            return {"error": "customer not found"}
        remaining = round(acc["balance"] - planned_spend, 2)
        out = {
            "balance": acc["balance"],
            "planned_spend": planned_spend,
            "balance_after": remaining,
            "overdraft": remaining < 0,
        }
        if remaining < 0:
            out["demo_overdraft_fee"] = self.terms["overdraft_fee"]
            out["deposit_needed_to_avoid"] = round(-remaining, 2)
        return out

    def simulate_mortgage(self, principal: float, annual_rate: float, years: int) -> dict:
        pay = monthly_payment(principal, annual_rate, years)
        n = int(years) * 12
        return {
            "principal": principal,
            "annual_rate_pct": annual_rate,
            "years": int(years),
            "monthly_payment": round(pay, 2),
            "total_paid": round(pay * n, 2),
            "total_interest": round(pay * n - principal, 2),
        }

    def calculate_loan_affordability(
        self, monthly_income: float, monthly_debt: float, credit_score: int, requested_principal: float
    ) -> dict:
        """Returns the decision together with every factor and threshold (explainability for the customer)."""
        t = self.terms
        pay = monthly_payment(requested_principal, t["mortgage_rate_pct"], t["mortgage_max_term_years"])
        dti = (monthly_debt + pay) / monthly_income
        factors = [
            {
                "factor": "credit_score",
                "value": credit_score,
                "threshold": f">= {t['min_credit_score']}",
                "passed": credit_score >= t["min_credit_score"],
            },
            {
                "factor": "debt_to_income",
                "value": round(dti, 4),
                "threshold": f"<= {t['max_dti']}",
                "passed": dti <= t["max_dti"],
            },
        ]
        return {
            "eligible": all(f["passed"] for f in factors),
            "estimated_monthly_payment": round(pay, 2),
            "assumptions": {
                "rate_pct": t["mortgage_rate_pct"],
                "term_years": t["mortgage_max_term_years"],
                "source": "fictional demo bank terms (configs/agent.yaml)",
            },
            "factors": factors,
        }


def make_tools(bank: Bank, search_fn) -> list[StructuredTool]:
    def search_regulations(query: str) -> str:
        """Search US consumer-banking regulations (Reg E electronic transfers, Reg DD truth in savings,
        Reg CC funds availability) and return cited excerpts."""
        return search_fn(query)

    def get_account_balance(customer_id: str) -> dict:
        """Current balance and account type for the authenticated customer (ID format CUSTnnn)."""
        return bank.get_account_balance(customer_id)

    def get_spending_summary(customer_id: str) -> dict:
        """Monthly spending by category for the authenticated customer."""
        return bank.get_spending_summary(customer_id)

    def check_overdraft_risk(customer_id: str, planned_spend: float) -> dict:
        """Whether a planned payment of `planned_spend` dollars would overdraw the customer's account."""
        return bank.check_overdraft_risk(customer_id, planned_spend)

    def simulate_mortgage(principal: float, annual_rate: float, years: int) -> dict:
        """Monthly payment, total paid and total interest for a fixed-rate loan. annual_rate in percent."""
        return bank.simulate_mortgage(principal, annual_rate, years)

    def calculate_loan_affordability(
        monthly_income: float, monthly_debt: float, credit_score: int, requested_principal: float
    ) -> dict:
        """Pre-screen a mortgage request against the demo bank's credit-score and debt-to-income rules."""
        return bank.calculate_loan_affordability(
            monthly_income, monthly_debt, credit_score, requested_principal
        )

    fns = [
        search_regulations,
        get_account_balance,
        get_spending_summary,
        check_overdraft_risk,
        simulate_mortgage,
        calculate_loan_affordability,
    ]
    return [StructuredTool.from_function(f) for f in fns]
