"""Prompt-injection detectors with a common interface so they can be benchmarked side by side.

RegexInjectionDetector     - hand-written patterns (v0.1 list plus a few additions), no learning
ProtectAIDetector          - protectai/deberta-v3-base-prompt-injection-v2 (Apache-2.0), English only
CascadeDetector            - flags if any member flags (union), the usual "cheap filter + classifier" setup
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol


class InjectionDetector(Protocol):
    name: str

    def scores(self, texts: Sequence[str]) -> list[float]: ...


_V0_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"disregard\s+(all\s+)?(prior|previous|your)\s+(instructions?|rules?|constraints?)",
    r"you\s+are\s+now\s+(a|an)\s+\w+",
    r"act\s+as\s+(if\s+you\s+((are|were|have)\s+)?)",
    r"jailbreak",
    r"(bypass|override|circumvent|disable)\s+(the\s+)?(safety|guardrail|filter|restriction|rule)",
    r"do\s+anything\s+now",
    r"DAN\b",  # case-sensitive in v0.1
    r"pretend\s+(you\s+)?(have\s+no|don't\s+have|without)\s+(restrictions?|rules?|limits?)",
    r"system\s+prompt",
    r"reveal\s+(your|the)\s+(prompt|instructions?|system)",
    r"</?(system|user|assistant)>",
]
# Additions written before running the benchmark (kept separate so their effect is measurable).
_V1_EXTRA = [
    r"(ignore|forget|disregard|skip)\s+(all\s+|any\s+|the\s+|your\s+|my\s+)?(previous|prior|above|earlier|preceding|initial|original)\s+"
    r"(instructions?|directions?|directives?|prompts?|rules?|text|context|messages?)",
    r"(ignore|forget)\s+(everything|all)\s+(that\s+)?(was|you|before|above|i\s+said)",
    r"\bnew\s+(instructions?|rules?)\s*:",
    r"(print|show|repeat|output|tell\s+me)\s+(your|the)\s+(initial|original|hidden|secret)?\s*(prompt|instructions?|password)",
    r"\bdeveloper\s+mode\b",
]


class RegexInjectionDetector:
    def __init__(self, version: str = "v1"):
        pats = _V0_PATTERNS + (_V1_EXTRA if version == "v1" else [])
        self.patterns = [re.compile(p, 0 if p == r"DAN\b" else re.IGNORECASE) for p in pats]
        self.name = f"regex_{version}"

    def matches(self, text: str) -> list[str]:
        return [p.pattern for p in self.patterns if p.search(text)]

    def scores(self, texts: Sequence[str]) -> list[float]:
        return [1.0 if any(p.search(t) for p in self.patterns) else 0.0 for t in texts]


class ProtectAIDetector:
    MODEL = "protectai/deberta-v3-base-prompt-injection-v2"

    def __init__(self, device: str | None = None, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL).eval()
        self.device = device or "cpu"
        self.model.to(self.device)
        self.batch_size = batch_size
        # label 1 = INJECTION per the model card
        self.inj_idx = next(i for i, lbl in self.model.config.id2label.items() if "INJ" in lbl.upper())
        self.name = "protectai_deberta_v2"

    def scores(self, texts: Sequence[str]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            enc = self.tok(batch, truncation=True, max_length=512, padding=True, return_tensors="pt").to(
                self.device
            )
            with self.torch.no_grad():
                probs = self.model(**enc).logits.softmax(-1)[:, self.inj_idx]
            out += probs.cpu().tolist()
        return out


class CascadeDetector:
    def __init__(self, members: Sequence[InjectionDetector], thresholds: Sequence[float]):
        self.members = list(members)
        self.thresholds = list(thresholds)
        self.name = "+".join(m.name for m in members)

    def scores(self, texts: Sequence[str]) -> list[float]:
        flags = [
            [s >= t for s in m.scores(texts)] for m, t in zip(self.members, self.thresholds, strict=True)
        ]
        return [1.0 if any(col) else 0.0 for col in zip(*flags, strict=True)]
