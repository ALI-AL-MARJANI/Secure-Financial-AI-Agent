"""
Layer 3 — Output Guardrail
Runs before the final response reaches the user.
Checks: PII scrub · hallucination/grounding · compliance · audit logging
"""
import re
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Optional
from langchain_core.documents import Document
from langchain_ollama import ChatOllama


_llm = ChatOllama(model="mistral", temperature=0)

# Compliance violations: things the agent should never say
_COMPLIANCE_PATTERNS = [
    (re.compile(r"\b(buy|invest\s+in|purchase)\s+(bitcoin|ethereum|crypto|nft|meme.?stock|dogecoin)", re.IGNORECASE),
     "Prohibited investment recommendation (crypto/meme stocks)"),
    (re.compile(r"\b(guarantee(d)?|certain(ly)?|100%\s+sure)\s+(return|profit|gain)", re.IGNORECASE),
     "Guaranteed return claim — not allowed under financial compliance"),
    (re.compile(r"waive\s+(the\s+)?overdraft\s+fee", re.IGNORECASE),
     "Overdraft fee waiver — contradicts bank policy"),
]

# PII patterns to scrub from output
_OUTPUT_PII = {
    "full_account_number": re.compile(r"\b\d{10,18}\b"),
    "ssn":                 re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "cvv":                 re.compile(r"\b(?:cvv|cvc)[:\s]*\d{3,4}\b", re.IGNORECASE),
}


@dataclass
class OutputGuardResult:
    allowed: bool
    final_response: str
    compliance_flags: List[str] = field(default_factory=list)
    pii_scrubbed: bool = False
    grounding_score: Optional[float] = None
    audit_entry: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Grounding check — are claims supported by retrieved context?
# ---------------------------------------------------------------------------

def _check_grounding(response: str, context_docs: List[Document]) -> float:
    if not context_docs:
        return 1.0  # No RAG context used — direct answer, skip grounding

    context = "\n\n".join([d.page_content for d in context_docs[:3]])
    prompt = (
        "You are a fact-checker for a banking assistant.\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Agent response:\n{response}\n\n"
        "On a scale of 0.0 to 1.0, how well is the response grounded in the context? "
        "1.0 = fully supported, 0.0 = contradicts or invents facts. "
        "Output only a decimal number."
    )
    raw = _llm.invoke(prompt).content.strip()
    try:
        score = float(re.search(r"\d+\.?\d*", raw).group())
        return min(1.0, max(0.0, score))
    except (AttributeError, ValueError):
        return 0.5


# ---------------------------------------------------------------------------
# PII scrubber
# ---------------------------------------------------------------------------

def _scrub_pii(text: str) -> tuple[str, bool]:
    scrubbed = False
    for label, pattern in _OUTPUT_PII.items():
        if pattern.search(text):
            text = pattern.sub(f"[REDACTED-{label.upper()}]", text)
            scrubbed = True
    return text, scrubbed


# ---------------------------------------------------------------------------
# Audit logger
# ---------------------------------------------------------------------------

def _write_audit_log(entry: dict):
    os.makedirs("audit_logs", exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    log_file = f"audit_logs/{date_str}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main guard function
# ---------------------------------------------------------------------------

def check_output(
    response: str,
    input_hash: str = "",
    threat_level: str = "LOW",
    tool_calls_made: List[str] = None,
    context_docs: List[Document] = None,
) -> OutputGuardResult:
    tool_calls_made = tool_calls_made or []
    context_docs = context_docs or []

    compliance_flags = []
    for pattern, reason in _COMPLIANCE_PATTERNS:
        if pattern.search(response):
            compliance_flags.append(reason)

    # Block the response if compliance is violated
    if compliance_flags:
        blocked_response = (
            "I'm unable to provide that response as it may violate SecureBank's compliance policies. "
            "Please consult a licensed financial advisor for investment decisions."
        )
        audit = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_hash": input_hash,
            "threat_level": threat_level,
            "tool_calls": tool_calls_made,
            "compliance_flags": compliance_flags,
            "action": "BLOCKED",
            "grounding_score": None,
        }
        _write_audit_log(audit)
        return OutputGuardResult(
            allowed=False,
            final_response=blocked_response,
            compliance_flags=compliance_flags,
            audit_entry=audit,
        )

    # PII scrub
    clean_response, pii_scrubbed = _scrub_pii(response)

    # Grounding check (only when RAG was used)
    grounding_score = _check_grounding(clean_response, context_docs) if context_docs else None

    # Flag low grounding but don't block — append a caveat instead
    if grounding_score is not None and grounding_score < 0.4:
        clean_response += (
            "\n\n⚠️ Note: This response may not be fully supported by official policy documents. "
            "Please verify with a SecureBank representative."
        )

    audit = {
        "timestamp": datetime.utcnow().isoformat(),
        "input_hash": input_hash,
        "threat_level": threat_level,
        "tool_calls": tool_calls_made,
        "compliance_flags": [],
        "pii_scrubbed": pii_scrubbed,
        "grounding_score": grounding_score,
        "action": "ALLOWED",
    }
    _write_audit_log(audit)

    return OutputGuardResult(
        allowed=True,
        final_response=clean_response,
        compliance_flags=[],
        pii_scrubbed=pii_scrubbed,
        grounding_score=grounding_score,
        audit_entry=audit,
    )
