"""Intent router: RAG (regulation / policy questions), TOOL (account data or calculations), OUT_OF_SCOPE.

KeywordRouter  - regex cues; returns None when no class or several classes match (abstains)
LLMRouter      - zero-shot classification with the local LLM
CascadeRouter  - keyword when it is unambiguous, LLM otherwise
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from secfin.llm import TextLLM

LABELS = ("RAG", "TOOL", "OUT_OF_SCOPE")

_TOOL = re.compile(
    r"\b(my\s+(current\s+)?balance|balance\s+(of|for|on)|how\s+much\s+(money\s+)?(do\s+i\s+have|is\s+in)|"
    r"spend(ing)?\s+(summary|breakdown)|what\s+did\s+i\s+spend|where\s+(does|did)\s+my\s+money|"
    r"monthly\s+payment|simulate|calculat\w*|amorti[sz]\w*|can\s+i\s+afford|afford\s+(a|the)\b|"
    r"(would|will)\s+(a|this|that|my)\s+\$?\d[\d,]*\s*\w*\s*(payment|purchase|charge)|"
    r"overdraw\s+my|pre-?screen|pre-?qualify|my\s+(debt|income)|CUST\d+)\b",
    re.IGNORECASE,
)
_RAG = re.compile(
    r"\b(regulation|reg\s*(e|dd|cc)|rule[sd]?|law|required|requirement|must\s+(a|the|my)\s+bank|"
    r"is\s+(a|my|the)\s+bank\s+(allowed|required)|can\s+(a|my|the)\s+bank|allowed|liab(le|ility)|"
    r"disclos\w+|notice|unauthori[sz]ed|error\s+resolution|dispute|stop[- ]payment|preauthori[sz]ed|"
    r"overdraft\s+(fee|service|opt|coverage)|opt[- ]in|gift\s+card|prepaid|payroll\s+card|remittance|"
    r"wire|annual\s+percentage\s+yield|\bapy\b|time\s+account|certificate\s+of\s+deposit|\bcd\b|"
    r"advertis\w+|funds\s+availability|hold\s+on|available\s+for\s+withdrawal|substitute\s+check|"
    r"business\s+day|deadline|how\s+(long|many\s+days)|electronic\s+fund\s+transfer|\beft\b)\b",
    re.IGNORECASE,
)
_OOS = re.compile(
    r"\b(weather|recipe|cook|poem|song|joke|story|movie|football|soccer|nba|election|president|"
    r"python\s+code|write\s+(me\s+)?(a|some)\s+(code|script|essay)|translate|horoscope|"
    r"bitcoin|ethereum|crypto\w*|stock\s+tip|which\s+stock|best\s+(stock|investment)|lottery|"
    r"diet|vacation|travel\s+plan|homework)\b",
    re.IGNORECASE,
)

ROUTER_PROMPT = """Classify the customer message for a banking assistant into exactly one label.

RAG          - a question about banking rules, rights, fees, disclosures or deadlines that should be answered from US banking regulations (Regulation E, DD, CC)
TOOL         - asks about the customer's own account (balance, spending, overdraft risk) or for a loan / mortgage calculation
OUT_OF_SCOPE - anything else: not about banking, or asks for investment picks, crypto advice, or general chit-chat

Message: {message}

Answer with one label only (RAG, TOOL, or OUT_OF_SCOPE)."""


@dataclass
class RouteDecision:
    label: str
    method: str


class KeywordRouter:
    name = "keyword"

    def route(self, message: str) -> RouteDecision | None:
        hits = {
            lbl
            for lbl, pat in (("TOOL", _TOOL), ("RAG", _RAG), ("OUT_OF_SCOPE", _OOS))
            if pat.search(message)
        }
        return RouteDecision(hits.pop(), "keyword") if len(hits) == 1 else None


def parse_label(text: str) -> str:
    t = text.upper()
    for lbl in ("OUT_OF_SCOPE", "TOOL", "RAG"):
        if lbl in t or lbl.replace("_", " ") in t:
            return lbl
    return "RAG"  # unparseable -> the most conservative useful route (retrieval + abstention)


class LLMRouter:
    name = "llm"

    def __init__(self, llm: TextLLM):
        self.llm = llm

    def route(self, message: str) -> RouteDecision:
        return RouteDecision(parse_label(self.llm(ROUTER_PROMPT.format(message=message))), "llm")


class CascadeRouter:
    name = "cascade"

    def __init__(self, llm: TextLLM):
        self.keyword = KeywordRouter()
        self.llm_router = LLMRouter(llm)

    def route(self, message: str) -> RouteDecision:
        return self.keyword.route(message) or self.llm_router.route(message)
