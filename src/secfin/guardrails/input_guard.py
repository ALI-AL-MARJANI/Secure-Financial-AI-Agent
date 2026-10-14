"""Layer 1: runs on the raw user message before any LLM call.

  * prompt-injection detection -> block (threat HIGH)
  * PII detection -> redact before the text reaches the LLM or the logs (threat MEDIUM), do not block:
    customers legitimately type card numbers, and refusing them is worse UX than redacting.
Out-of-scope requests are handled by the router, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from secfin.guardrails.injection import InjectionDetector, RegexInjectionDetector
from secfin.guardrails.pii import ValidatedRegexPII, redact


@dataclass
class InputDecision:
    allowed: bool
    threat_level: str  # LOW | MEDIUM | HIGH
    text: str  # redacted text to pass downstream
    reasons: list[str] = field(default_factory=list)
    pii_types: list[str] = field(default_factory=list)
    injection_score: float = 0.0


class InputGuard:
    def __init__(self, detector: InjectionDetector | None = None, threshold: float = 0.5, pii=None):
        self.detector = detector or RegexInjectionDetector("v1")
        self.threshold = threshold
        self.pii = pii or ValidatedRegexPII()

    def check(self, message: str) -> InputDecision:
        score = self.detector.scores([message])[0]
        if score >= self.threshold:
            return InputDecision(
                False,
                "HIGH",
                "",
                [f"prompt injection ({self.detector.name}, score={score:.2f})"],
                injection_score=score,
            )
        spans = self.pii.detect(message)
        if spans:
            types = sorted({s.label for s in spans})
            return InputDecision(
                True,
                "MEDIUM",
                redact(message, spans),
                [f"PII redacted: {', '.join(types)}"],
                pii_types=types,
                injection_score=score,
            )
        return InputDecision(True, "LOW", message, injection_score=score)
