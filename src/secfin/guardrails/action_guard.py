"""Layer 2: validate tool calls before they execute.

Decisions:
  allow    - arguments are valid
  confirm  - valid but above a review threshold; the agent pauses for human confirmation
  deny     - malformed, out of range, or targets another customer's data
Thresholds come from configs/agent.yaml.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_CUSTOMER_ID = re.compile(r"^CUST\d{3,6}$")
_SUSPICIOUS = re.compile(
    r"(--|;|/\*|\*/|\bunion\s+select\b|\bdrop\s+table\b|\.\./|\.\.\\|/etc/|<script)", re.IGNORECASE
)

# Expected argument names and types for each tool; anything else is rejected.
TOOL_SCHEMAS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "get_account_balance": {"customer_id": str},
    "get_spending_summary": {"customer_id": str},
    "check_overdraft_risk": {"customer_id": str, "planned_spend": (int, float)},
    "simulate_mortgage": {"principal": (int, float), "annual_rate": (int, float), "years": (int, float)},
    "calculate_loan_affordability": {
        "monthly_income": (int, float),
        "monthly_debt": (int, float),
        "credit_score": (int, float),
        "requested_principal": (int, float),
    },
    "search_regulations": {"query": str},
}
CUSTOMER_SCOPED = {"get_account_balance", "get_spending_summary", "check_overdraft_risk"}


@dataclass
class ActionDecision:
    decision: str  # allow | confirm | deny
    reasons: list[str] = field(default_factory=list)
    args: dict = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"


def _num(x) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)) and math.isfinite(x):
        return float(x)
    return None


def check_action(tool: str, args: dict, session_customer_id: str | None, limits: dict) -> ActionDecision:
    reasons: list[str] = []
    schema = TOOL_SCHEMAS.get(tool)
    if schema is None:
        return ActionDecision("deny", [f"unknown tool '{tool}'"])
    extra = set(args) - set(schema)
    missing = set(schema) - set(args)
    if extra:
        reasons.append(f"unexpected arguments {sorted(extra)}")
    if missing:
        reasons.append(f"missing arguments {sorted(missing)}")
    for name, typ in schema.items():
        if name not in args:
            continue
        bad = not isinstance(args[name], str) if typ is str else _num(args[name]) is None
        if bad:
            reasons.append(f"argument '{name}' has invalid type or value")
    for name, val in args.items():
        if isinstance(val, str) and (_SUSPICIOUS.search(val) or len(val) > limits.get("max_string_len", 512)):
            reasons.append(f"argument '{name}' contains a suspicious pattern or is too long")
    if reasons:
        return ActionDecision("deny", reasons, args)

    clean = {k: v.strip() if isinstance(v, str) else v for k, v in args.items()}
    if tool in CUSTOMER_SCOPED:
        cid = clean["customer_id"].upper()
        clean["customer_id"] = cid
        if not _CUSTOMER_ID.match(cid):
            return ActionDecision("deny", [f"invalid customer ID format '{cid}'"], clean)
        if session_customer_id is None:
            return ActionDecision("deny", ["no authenticated customer in this session"], clean)
        if cid != session_customer_id.upper():
            return ActionDecision("deny", ["customer ID does not match the authenticated session"], clean)

    if tool == "check_overdraft_risk" and not 0 < clean["planned_spend"] <= limits["max_transaction"]:
        return ActionDecision("deny", ["planned_spend out of range"], clean)

    if tool == "simulate_mortgage":
        p, r, y = clean["principal"], clean["annual_rate"], clean["years"]
        if not 0 < p <= limits["max_principal_hard"]:
            reasons.append(f"principal must be in (0, {limits['max_principal_hard']:,}]")
        if not 0 <= r <= limits["max_rate_pct"]:
            reasons.append(f"annual_rate must be in [0, {limits['max_rate_pct']}] percent")
        if not (float(y).is_integer() and 1 <= y <= limits["max_term_years"]):
            reasons.append(f"years must be an integer in [1, {limits['max_term_years']}]")
        if reasons:
            return ActionDecision("deny", reasons, clean)
        if p > limits["confirm_principal_above"]:
            return ActionDecision(
                "confirm", [f"principal above ${limits['confirm_principal_above']:,}"], clean
            )

    if tool == "calculate_loan_affordability":
        if not 300 <= clean["credit_score"] <= 850:
            reasons.append("credit_score must be in [300, 850]")
        if clean["monthly_income"] <= 0 or clean["monthly_debt"] < 0 or clean["requested_principal"] <= 0:
            reasons.append("income must be > 0, debt >= 0, principal > 0")
        if clean["requested_principal"] > limits["max_principal_hard"]:
            reasons.append("requested_principal above hard limit")
        if reasons:
            return ActionDecision("deny", reasons, clean)

    return ActionDecision("allow", [], clean)
