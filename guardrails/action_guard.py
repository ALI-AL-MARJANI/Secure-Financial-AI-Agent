"""
Layer 2 — Action Guardrail
Validates tool arguments before execution.
Checks: type bounds · dollar thresholds · path traversal · SQL injection
"""
import re
from dataclasses import dataclass


_SQL_INJECTION = re.compile(
    r"(--|;|'|\"|\/\*|\*\/|xp_|exec\s|union\s+select|drop\s+table|insert\s+into|"
    r"delete\s+from|update\s+\w+\s+set)",
    re.IGNORECASE,
)
_PATH_TRAVERSAL = re.compile(r"\.\./|\.\.\\|/etc/|/proc/", re.IGNORECASE)


_MORTGAGE_PRINCIPAL_MAX = 50_000_000   
_INTEREST_RATE_MAX = 50.0              
_LOAN_YEARS_MAX = 50


@dataclass
class ActionGuardResult:
    allowed: bool
    reason: str
    sanitized_args: dict


def sanitize_string(value: str) -> str:
    return value.strip()[:256]


def check_action(tool_name: str, tool_args: dict) -> ActionGuardResult:
    sanitized = {}

    for key, value in tool_args.items():
        if isinstance(value, str):
            if _SQL_INJECTION.search(value):
                return ActionGuardResult(
                    allowed=False,
                    reason=f"Potential SQL injection in argument '{key}'.",
                    sanitized_args={},
                )
            if _PATH_TRAVERSAL.search(value):
                return ActionGuardResult(
                    allowed=False,
                    reason=f"Path traversal attempt in argument '{key}'.",
                    sanitized_args={},
                )
            sanitized[key] = sanitize_string(value)
        else:
            sanitized[key] = value

    # Tool-specific threshold checks
    if tool_name == "simulate_mortgage":
        principal = sanitized.get("principal", 0)
        rate = sanitized.get("annual_rate", 0)
        years = sanitized.get("years", 0)

        if isinstance(principal, (int, float)) and principal > _MORTGAGE_PRINCIPAL_MAX:
            return ActionGuardResult(
                allowed=False,
                reason=f"Mortgage principal ${principal:,.0f} exceeds the $50M threshold. "
                       "Please verify the amount.",
                sanitized_args=sanitized,
            )
        if isinstance(rate, (int, float)) and (rate <= 0 or rate > _INTEREST_RATE_MAX):
            return ActionGuardResult(
                allowed=False,
                reason=f"Interest rate {rate}% is outside the valid range (0–50%).",
                sanitized_args=sanitized,
            )
        if isinstance(years, int) and (years <= 0 or years > _LOAN_YEARS_MAX):
            return ActionGuardResult(
                allowed=False,
                reason=f"Loan term {years} years is outside the valid range (1–50 years).",
                sanitized_args=sanitized,
            )

    if tool_name == "get_account_balance":
        cid = sanitized.get("customer_id", "")
        if not re.match(r"^CUST\d{3,6}$", cid, re.IGNORECASE):
            return ActionGuardResult(
                allowed=False,
                reason=f"Invalid customer ID format: '{cid}'. Expected format: CUSTXXX.",
                sanitized_args=sanitized,
            )

    return ActionGuardResult(allowed=True, reason="OK", sanitized_args=sanitized)
