"""Layer 3: checks on the generated answer before it reaches the user.

* PII scrub (validated regex; same detector as the input layer)
* compliance patterns (investment recommendations, guaranteed returns)
* numeric grounding: every dollar amount / percentage / day count in a RAG answer must appear in the
  retrieved context or in a tool output. This is a cheap, deterministic proxy for hallucination;
  it catches invented numbers, not invented claims without numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from secfin.guardrails.pii import ValidatedRegexPII, redact

_COMPLIANCE = [
    (
        re.compile(
            r"\b(buy|invest(ing)?\s+in|purchase|recommend)\s+(some\s+)?(bitcoin|ethereum|crypto(currenc(y|ies))?|nfts?|meme\s*stocks?|dogecoin)",
            re.I,
        ),
        "investment_recommendation_prohibited_asset",
    ),
    (
        re.compile(
            r"\b(guarantee[sd]?|risk[- ]free|certain)\s+(\d+\s*%\s+)?(returns?|profits?|gains?)", re.I
        ),
        "guaranteed_return_claim",
    ),
]
_NUMBER = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent\b|(?:business\s|calendar\s)?days?\b|months?\b|years?\b)",
    re.I,
)


def _canon(token: str) -> str:
    t = token.lower().replace(",", "").replace("$", "").replace("percent", "%").strip()
    m = re.match(r"(\d+(?:\.\d+)?)", t)
    if not m:
        return t
    num = m.group(1).rstrip("0").rstrip(".") if "." in m.group(1) else m.group(1)
    unit = (
        "%"
        if "%" in t
        else ("day" if "day" in t else "month" if "month" in t else "year" if "year" in t else "$")
    )
    return f"{num}{unit}"


_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "fourteen": 14,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "forty-five": 45,
    "sixty": 60,
    "ninety": 90,
}
# Regulations spell small numbers ("two business days", "seven (7) days"); answers usually use digits.
_WORD_NUM = re.compile(r"\b(" + "|".join(sorted(_WORDS, key=len, reverse=True)) + r")\b(\s*\(\d+\))?", re.I)


def numeric_claims(text: str) -> set[str]:
    text = _WORD_NUM.sub(lambda m: str(_WORDS[m.group(1).lower()]), text)
    return {_canon(m.group()) for m in _NUMBER.finditer(text)}


_BARE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _value(token: str) -> str:
    v = token.replace(",", "")
    return v.rstrip("0").rstrip(".") if "." in v else v


def evidence_values(texts: list[str]) -> set[str]:
    """Every number in the evidence, unit-free (tool outputs are JSON without $ or units)."""
    out: set[str] = set()
    for t in texts:
        t = _WORD_NUM.sub(lambda m: str(_WORDS[m.group(1).lower()]), t)
        out |= {_value(m.group()) for m in _BARE.finditer(t)}
    return out


@dataclass
class OutputDecision:
    text: str
    blocked: bool = False
    flags: list[str] = field(default_factory=list)
    pii_redacted: int = 0
    ungrounded_numbers: list[str] = field(default_factory=list)


_pii = ValidatedRegexPII()
BLOCK_MESSAGE = (
    "I can't provide that answer because it conflicts with the assistant's compliance rules "
    "(no investment recommendations or return guarantees)."
)


WITHHELD_MESSAGE = (
    "I couldn't verify the figures in my answer against a tool result, so I'm not showing them. "
    "Please try again, for example by stating the amounts explicitly."
)


def check_output(
    answer: str, evidence: list[str] | None = None, on_ungrounded: str = "annotate"
) -> OutputDecision:
    """on_ungrounded: "annotate" appends a warning (RAG answers), "withhold" replaces the answer (tool answers,
    where an unverified number is an invented balance or payment)."""
    flags = [name for pat, name in _COMPLIANCE if pat.search(answer)]
    if flags:
        return OutputDecision(BLOCK_MESSAGE, blocked=True, flags=flags)

    spans = _pii.detect(answer)
    text = redact(answer, spans) if spans else answer
    ungrounded: list[str] = []
    if evidence is not None:
        # A claim is grounded if its numeric value occurs in the evidence; units are not compared because
        # tool outputs are JSON and regulations mix "$50", "50 dollars" and "fifty".
        support = evidence_values(evidence)
        ungrounded = sorted(
            n for n in numeric_claims(answer) if re.match(r"[\d.]+", n).group() not in support
        )
        if ungrounded and on_ungrounded == "withhold":
            return OutputDecision(
                WITHHELD_MESSAGE,
                blocked=True,
                flags=[*flags, "ungrounded_numbers_withheld"],
                pii_redacted=len(spans),
                ungrounded_numbers=ungrounded,
            )
        if ungrounded:
            flags.append("ungrounded_numbers")
            text += (
                "\n\nNote: some figures above could not be matched to the retrieved regulation text "
                f"({', '.join(ungrounded)}); please verify them against the source."
            )
    return OutputDecision(text, flags=flags, pii_redacted=len(spans), ungrounded_numbers=ungrounded)
