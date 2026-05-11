"""
Financial tools for SecureBank AI Agent
"""
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Mock database
# ---------------------------------------------------------------------------

_ACCOUNTS = {
    "CUST123": {"balance": 5432.50,  "name": "Alice Dupont",   "type": "checking"},
    "CUST456": {"balance": 120500.00,"name": "Bob Martin",     "type": "savings"},
    "CUST789": {"balance": 150.75,   "name": "Carol Nguyen",   "type": "checking"},
}

_SPENDING = {
    "CUST123": {"groceries": 320.0, "rent": 1200.0, "transport": 85.0, "dining": 210.0},
    "CUST456": {"groceries": 580.0, "rent": 2400.0, "transport": 120.0, "dining": 450.0},
    "CUST789": {"groceries": 95.0,  "rent": 800.0,  "transport": 40.0,  "dining": 60.0},
}


# ---------------------------------------------------------------------------
# Tool 1 — Account balance
# ---------------------------------------------------------------------------

@tool
def get_account_balance(customer_id: str) -> str:
    """
    Fetches the current account balance and account type for a given customer ID.
    Always ask the user for their customer ID if it is not provided.
    Customer IDs follow the format CUSTXXX (e.g. CUST123).
    """
    cid = customer_id.strip().upper()
    if cid not in _ACCOUNTS:
        return f"Error: Customer ID '{cid}' not found in the system."
    acc = _ACCOUNTS[cid]
    return (
        f"Account holder: {acc['name']}\n"
        f"Account type: {acc['type']}\n"
        f"Current balance: ${acc['balance']:,.2f}"
    )


# ---------------------------------------------------------------------------
# Tool 2 — Mortgage simulator
# ---------------------------------------------------------------------------

@tool
def simulate_mortgage(principal: float, annual_rate: float, years: int) -> str:
    """
    Calculates the estimated monthly mortgage payment using standard amortization.
    Args:
        principal: Loan amount in USD (e.g. 300000)
        annual_rate: Annual interest rate as a percentage (e.g. 5.0 for 5%)
        years: Loan term in years (e.g. 25)
    SecureBank's standard fixed rate for 2026 is 5%. Maximum term: 25 years.
    Minimum credit score to qualify: 700.
    """
    monthly_rate = (annual_rate / 100) / 12
    n = years * 12

    if monthly_rate == 0:
        payment = principal / n
    else:
        payment = principal * (monthly_rate * (1 + monthly_rate) ** n) / ((1 + monthly_rate) ** n - 1)

    total_paid = payment * n
    total_interest = total_paid - principal

    return (
        f"Mortgage simulation results:\n"
        f"  Principal: ${principal:,.2f}\n"
        f"  Rate: {annual_rate}% annual / {annual_rate/12:.3f}% monthly\n"
        f"  Term: {years} years ({n} payments)\n"
        f"  Monthly payment: ${payment:,.2f}\n"
        f"  Total paid: ${total_paid:,.2f}\n"
        f"  Total interest: ${total_interest:,.2f}"
    )


# ---------------------------------------------------------------------------
# Tool 3 — Loan affordability (DTI-based, per bank policy)
# ---------------------------------------------------------------------------

@tool
def calculate_loan_affordability(
    monthly_income: float,
    monthly_debt: float,
    credit_score: int,
    requested_principal: float,
) -> str:
    """
    Evaluates whether a customer qualifies for a SecureBank mortgage based on
    Debt-to-Income (DTI) ratio and credit score requirements from bank policy.
    Args:
        monthly_income: Gross monthly income in USD
        monthly_debt: Existing monthly debt obligations in USD
        credit_score: Applicant's credit score (300–850)
        requested_principal: Desired loan amount in USD
    """
    # Per bank policy: min credit score 700, max term 25 years, rate 5%
    MIN_CREDIT_SCORE = 700
    MAX_DTI = 0.43  # 43% back-end DTI — standard qualifying threshold

    if credit_score < MIN_CREDIT_SCORE:
        return (
            f"Application pre-screening result: NOT ELIGIBLE\n"
            f"Reason: Credit score {credit_score} is below SecureBank's minimum of {MIN_CREDIT_SCORE}.\n"
            f"Recommendation: Improve credit score before reapplying."
        )

    # Estimate mortgage payment at policy rate (5%, 25 years)
    monthly_rate = 0.05 / 12
    n = 25 * 12
    est_payment = requested_principal * (monthly_rate * (1 + monthly_rate) ** n) / ((1 + monthly_rate) ** n - 1)

    total_monthly_debt = monthly_debt + est_payment
    dti = total_monthly_debt / monthly_income if monthly_income > 0 else 1.0

    status = "ELIGIBLE" if dti <= MAX_DTI else "NOT ELIGIBLE"
    return (
        f"Affordability assessment: {status}\n"
        f"  Estimated mortgage payment: ${est_payment:,.2f}/month\n"
        f"  Total monthly debt: ${total_monthly_debt:,.2f}\n"
        f"  Debt-to-Income ratio: {dti:.1%} (max allowed: {MAX_DTI:.0%})\n"
        f"  Credit score: {credit_score} ({'✓ meets' if credit_score >= MIN_CREDIT_SCORE else '✗ below'} minimum)\n"
        f"{'  → Proceed to formal application.' if status == 'ELIGIBLE' else '  → Application would likely be declined at current income/debt level.'}"
    )


# ---------------------------------------------------------------------------
# Tool 4 — Overdraft risk check
# ---------------------------------------------------------------------------

@tool
def check_overdraft_risk(customer_id: str, planned_spend: float) -> str:
    """
    Proactively checks if a planned purchase or transaction would put the
    customer's account into overdraft, and warns them about the $35 fee.
    Args:
        customer_id: Customer ID (e.g. CUST123)
        planned_spend: Amount of the planned transaction in USD
    """
    cid = customer_id.strip().upper()
    if cid not in _ACCOUNTS:
        return f"Error: Customer ID '{cid}' not found."

    balance = _ACCOUNTS[cid]["balance"]
    remaining = balance - planned_spend

    if remaining >= 0:
        return (
            f"Transaction of ${planned_spend:,.2f} is safe.\n"
            f"Balance after transaction: ${remaining:,.2f}"
        )
    return (
        f"⚠️  Overdraft Warning\n"
        f"Current balance: ${balance:,.2f}\n"
        f"Planned transaction: ${planned_spend:,.2f}\n"
        f"Projected balance: ${remaining:,.2f}\n"
        f"SecureBank policy: a $35 overdraft fee will be applied immediately.\n"
        f"Recommendation: deposit at least ${abs(remaining) + 35:,.2f} before proceeding."
    )


# ---------------------------------------------------------------------------
# Tool 5 — Spending summary
# ---------------------------------------------------------------------------

@tool
def get_spending_summary(customer_id: str) -> str:
    """
    Returns a monthly spending breakdown by category for the given customer.
    Args:
        customer_id: Customer ID (e.g. CUST123)
    """
    cid = customer_id.strip().upper()
    if cid not in _SPENDING:
        return f"Error: No spending data found for '{cid}'."

    data = _SPENDING[cid]
    total = sum(data.values())
    lines = [f"  {cat.capitalize():15} ${amount:>8,.2f}" for cat, amount in data.items()]
    return (
        f"Monthly spending summary for {cid}:\n"
        + "\n".join(lines)
        + f"\n  {'Total':15} ${total:>8,.2f}"
    )


# ---------------------------------------------------------------------------
# Tool 6 — GDPR-compliant decision explainer
# ---------------------------------------------------------------------------

@tool
def explain_decision(decision_type: str, parameters: str) -> str:
    """
    Provides a plain-language explanation of an automated financial decision,
    as required by GDPR Article 22 for decisions affecting individuals.
    Args:
        decision_type: Type of decision (e.g. 'mortgage_eligibility', 'overdraft_fee')
        parameters: Comma-separated key=value pairs used in the decision
    """
    explanations = {
        "mortgage_eligibility": (
            "Your mortgage eligibility was determined by two criteria from SecureBank policy:\n"
            "1. Credit score threshold: minimum 700 required.\n"
            "2. Debt-to-Income (DTI) ratio: total monthly debts (including the new mortgage payment) "
            "must not exceed 43% of gross monthly income.\n"
            "These thresholds are set by SecureBank's risk management policy and are applied "
            "consistently to all applicants."
        ),
        "overdraft_fee": (
            "A $35 overdraft fee is applied automatically when your account balance falls below $0. "
            "This is a fixed fee defined in SecureBank's Account Overdraft Rules policy. "
            "SecureBank does not waive this fee. "
            "You can avoid it by maintaining a positive balance or setting up an overdraft alert."
        ),
    }

    key = decision_type.lower().replace(" ", "_")
    explanation = explanations.get(key, (
        f"Decision type '{decision_type}' explanation: This automated decision was made based on "
        f"parameters: {parameters}. It was processed according to SecureBank's internal policy "
        f"guidelines. For full details, contact a SecureBank representative."
    ))

    return f"Decision Explanation ({decision_type}):\n{explanation}\nParameters used: {parameters}"


# ---------------------------------------------------------------------------
# Exported tool list
# ---------------------------------------------------------------------------

financial_tools = [
    get_account_balance,
    simulate_mortgage,
    calculate_loan_affordability,
    check_overdraft_risk,
    get_spending_summary,
    explain_decision,
]
