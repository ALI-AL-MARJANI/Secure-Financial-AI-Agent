"""
Layer 1 — Input Guardrail
Runs before the LLM sees the user message.
Checks: PII detection · prompt injection patterns · topic relevance
"""
import re
import hashlib
from datetime import datetime
from typing import Literal
from dataclasses import dataclass, asdict


ThreatLevel = Literal["LOW", "MEDIUM", "HIGH"]


# ---------------------------------------------------------------------------
# PII Patterns (financial context)
# ---------------------------------------------------------------------------

_PII_PATTERNS = {
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ssn":         re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "iban":        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b"),
    "routing":     re.compile(r"\b\d{9}\b"),
    "cvv":         re.compile(r"\b(?:cvv|cvc|cvv2)[:\s]*\d{3,4}\b", re.IGNORECASE),
    "dob":         re.compile(r"\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])[-/](?:19|20)\d{2}\b"),
}

# ---------------------------------------------------------------------------
# Prompt Injection Patterns (OWASP LLM01:2025)
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(prior|previous|your)\s+(instructions?|rules?|constraints?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+\w+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you\s+((are|were|have)\s+)?)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"(bypass|override|circumvent|disable)\s+(the\s+)?(safety|guardrail|filter|restriction|rule)", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"DAN\b"),
    re.compile(r"pretend\s+(you\s+)?(have\s+no|don't\s+have|without)\s+(restrictions?|rules?|limits?)", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"reveal\s+(your|the)\s+(prompt|instructions?|system)", re.IGNORECASE),
    re.compile(r"</?(system|user|assistant)>", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Off-Topic Patterns (hard out-of-scope for a banking assistant)
# ---------------------------------------------------------------------------

_OFF_TOPIC_PATTERNS = [
    re.compile(r"\b(write|generate|create|make)\s+(me\s+)?(a\s+)?(code|script|poem|story|essay|song|image)", re.IGNORECASE),
    re.compile(r"\b(hack|exploit|vulnerability|malware|virus|phishing)\b", re.IGNORECASE),
    re.compile(r"\b(bitcoin|ethereum|crypto|nft|meme.?stock|dogecoin|shib)\b", re.IGNORECASE),
    re.compile(r"\b(dating|relationship|recipe|weather|news|sports|politics)\b", re.IGNORECASE),
]

# Keywords that confirm the query is banking-related (fast positive check)
_BANKING_KEYWORDS = re.compile(
    r"\b(balance|account|mortgage|loan|interest|rate|overdraft|transfer|deposit|"
    r"credit|debit|investment|fund|bond|fee|policy|bank|payment|income|debt|afford"
    r"|CUST\d+)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GuardResult:
    allowed: bool
    threat_level: ThreatLevel
    reason: str
    pii_detected: list
    injection_detected: bool
    input_hash: str
    timestamp: str

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Main guard function
# ---------------------------------------------------------------------------

def check_input(user_message: str) -> GuardResult:
    timestamp = datetime.utcnow().isoformat()
    input_hash = hashlib.sha256(user_message.encode()).hexdigest()[:16]

    pii_hits = []
    for label, pattern in _PII_PATTERNS.items():
        if pattern.search(user_message):
            pii_hits.append(label)

    injection_hit = any(p.search(user_message) for p in _INJECTION_PATTERNS)
    off_topic_hit = any(p.search(user_message) for p in _OFF_TOPIC_PATTERNS)
    is_banking = bool(_BANKING_KEYWORDS.search(user_message))

    # Determine threat level and decision
    if injection_hit:
        return GuardResult(
            allowed=False,
            threat_level="HIGH",
            reason="Prompt injection attempt detected.",
            pii_detected=pii_hits,
            injection_detected=True,
            input_hash=input_hash,
            timestamp=timestamp,
        )

    if pii_hits:
        return GuardResult(
            allowed=False,
            threat_level="HIGH",
            reason=f"Sensitive personal data detected in input ({', '.join(pii_hits)}). "
                   "Please do not share raw financial identifiers.",
            pii_detected=pii_hits,
            injection_detected=False,
            input_hash=input_hash,
            timestamp=timestamp,
        )

    if off_topic_hit and not is_banking:
        return GuardResult(
            allowed=False,
            threat_level="MEDIUM",
            reason="This request falls outside the scope of SecureBank's banking assistant.",
            pii_detected=[],
            injection_detected=False,
            input_hash=input_hash,
            timestamp=timestamp,
        )

    threat = "MEDIUM" if off_topic_hit else "LOW"
    return GuardResult(
        allowed=True,
        threat_level=threat,
        reason="OK",
        pii_detected=[],
        injection_detected=False,
        input_hash=input_hash,
        timestamp=timestamp,
    )
