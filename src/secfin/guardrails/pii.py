"""PII detection and redaction.

Two detectors are kept so they can be compared in scripts/bench_pii.py:
  * `LegacyRegexPII`   - the patterns of the original v0.1 input guard (no validation);
  * `ValidatedRegexPII` - context-free patterns plus checksum / format validation
    (Luhn for cards, ISO 13616 mod-97 for IBANs, SSA rules for SSNs).
Presidio (Microsoft) is the reference detector; its wrapper lives in scripts/bench_pii.py so that the
runtime package does not depend on spaCy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    label: str
    start: int
    end: int
    text: str


def luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_ok(iban: str) -> bool:
    s = iban.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    rearranged = s[4:] + s[:4]
    num = "".join(str(int(c, 36)) for c in rearranged)
    return int(num) % 97 == 1


def ssn_ok(ssn: str) -> bool:
    d = re.sub(r"\D", "", ssn)
    area, group, serial = d[:3], d[3:5], d[5:]
    return not (area in ("000", "666") or area.startswith("9") or group == "00" or serial == "0000")


class LegacyRegexPII:
    """Patterns from the original repository (guardrails/input_guard.py, v0.1), unvalidated."""

    name = "regex_v0"
    PATTERNS = {
        "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
        "US_SSN": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
        "IBAN": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b"),
        "ROUTING": re.compile(r"\b\d{9}\b"),
        "CVV": re.compile(r"\b(?:cvv|cvc|cvv2)[:\s]*\d{3,4}\b", re.IGNORECASE),
        "DOB": re.compile(r"\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])[-/](?:19|20)\d{2}\b"),
    }

    def detect(self, text: str) -> list[Span]:
        out = []
        for label, pat in self.PATTERNS.items():
            out += [Span(label, m.start(), m.end(), m.group()) for m in pat.finditer(text)]
        return out


class ValidatedRegexPII:
    name = "regex_v1_validated"

    _CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
    _SSN = re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])")
    _IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")
    _EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")
    _PHONE_US = re.compile(r"(?<![\w$])(?:\+1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?![\d-])")
    _PHONE_FR = re.compile(r"(?<![\w$])(?:\+33 ?[1-9]|0[1-9])(?:[ .-]?\d{2}){4}(?!\d)")

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for m in self._CARD.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if 13 <= len(digits) <= 19 and luhn_ok(digits):
                spans.append(Span("CREDIT_CARD", m.start(), m.end(), m.group()))
        for m in self._IBAN.finditer(text):
            if iban_ok(m.group()):
                spans.append(Span("IBAN", m.start(), m.end(), m.group()))
        for m in self._SSN.finditer(text):
            if ssn_ok(m.group()):
                spans.append(Span("US_SSN", m.start(), m.end(), m.group()))
        spans += [Span("EMAIL", m.start(), m.end(), m.group()) for m in self._EMAIL.finditer(text)]
        spans += [Span("PHONE", m.start(), m.end(), m.group()) for m in self._PHONE_US.finditer(text)]
        spans += [Span("PHONE", m.start(), m.end(), m.group()) for m in self._PHONE_FR.finditer(text)]
        return _drop_overlaps(spans)


def _drop_overlaps(spans: list[Span]) -> list[Span]:
    # Prefer the longer span when two detectors fire on overlapping text (e.g. a card number that
    # also looks like a phone number).
    kept: list[Span] = []
    for s in sorted(spans, key=lambda s: (-(s.end - s.start), s.start)):
        if all(s.end <= k.start or s.start >= k.end for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda s: s.start)


def redact(text: str, spans: list[Span]) -> str:
    out, last = [], 0
    for s in sorted(spans, key=lambda s: s.start):
        if s.start < last:
            continue
        out.append(text[last : s.start])
        out.append(f"[{s.label}]")
        last = s.end
    out.append(text[last:])
    return "".join(out)
