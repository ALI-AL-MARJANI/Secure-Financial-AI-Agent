"""PII detection: original regex (v0) vs validated regex (v1) vs Microsoft Presidio (pattern recognizers).

  python scripts/bench_pii.py

Entity-level precision / recall / F1 per type (a prediction counts if it overlaps a gold span of the same
type), plus message-level flag precision / recall (what matters when the guard blocks or redacts).
Outputs results/guardrails/pii.json.
"""

from __future__ import annotations

import time
from collections import defaultdict

from secfin.eval.ir_metrics import latency_summary
from secfin.guardrails.pii import LegacyRegexPII, Span, ValidatedRegexPII
from secfin.runinfo import REPO_ROOT, read_jsonl, run_metadata, write_json

LABELS = ["CREDIT_CARD", "IBAN", "US_SSN", "EMAIL", "PHONE"]
PRESIDIO_MAP = {
    "CREDIT_CARD": "CREDIT_CARD",
    "IBAN_CODE": "IBAN",
    "US_SSN": "US_SSN",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
}


class PresidioPII:
    name = "presidio"

    def __init__(self):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
        )
        self.engine = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])

    def detect(self, text: str) -> list[Span]:
        res = self.engine.analyze(text=text, language="en", entities=list(PRESIDIO_MAP))
        return [Span(PRESIDIO_MAP[r.entity_type], r.start, r.end, text[r.start : r.end]) for r in res]


def evaluate(detector, rows: list[dict]) -> dict:
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    msg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    lat, errors = [], []
    for r in rows:
        t0 = time.perf_counter()
        pred = detector.detect(r["text"])
        lat.append(time.perf_counter() - t0)
        gold = r["spans"]
        matched = set()
        for p in pred:
            hit = next(
                (
                    i
                    for i, g in enumerate(gold)
                    if g["label"] == p.label
                    and p.start < g["end"]
                    and g["start"] < p.end
                    and i not in matched
                ),
                None,
            )
            if hit is None:
                fp[p.label] += 1
                errors.append({"type": "FP", "label": p.label, "pred": p.text, "text": r["text"]})
            else:
                tp[p.label] += 1
                matched.add(hit)
        for i, g in enumerate(gold):
            if i not in matched:
                fn[g["label"]] += 1
                errors.append({"type": "FN", "label": g["label"], "gold": g["text"], "text": r["text"]})
        flagged, has = bool(pred), bool(gold)
        msg["tp" if flagged and has else "fp" if flagged else "fn" if has else "tn"] += 1

    def prf(t, p, n):
        prec = t / (t + p) if t + p else float("nan")
        rec = t / (t + n) if t + n else float("nan")
        f1 = 2 * t / (2 * t + p + n) if t else 0.0
        return {"precision": prec, "recall": rec, "f1": f1, "tp": t, "fp": p, "fn": n}

    labels = sorted(set(LABELS) | set(fp))
    return {
        "per_label": {lbl: prf(tp[lbl], fp[lbl], fn[lbl]) for lbl in labels},
        "micro": prf(sum(tp.values()), sum(fp.values()), sum(fn.values())),
        "message_level": {**prf(msg["tp"], msg["fp"], msg["fn"]), "tn": msg["tn"]},
        "latency_per_message": latency_summary(lat),
        "error_examples": errors[:40],
    }


def main() -> None:
    rows = read_jsonl(REPO_ROOT / "data/security/pii_synthetic.jsonl")
    # Hand-written formats that the templates do not produce (dots, no separators, lowercase IBAN, ...).
    stress = read_jsonl(REPO_ROOT / "data/security/pii_stress.jsonl")
    results = {}
    for det in (LegacyRegexPII(), ValidatedRegexPII(), PresidioPII()):
        det.detect("warm up 4111 1111 1111 1111")
        results[det.name] = evaluate(det, rows)
        for lang in ("en", "fr"):
            results[f"{det.name}|{lang}"] = {
                k: v
                for k, v in evaluate(det, [r for r in rows if r["lang"] == lang]).items()
                if k in ("micro", "message_level")
            }
        results[f"{det.name}|stress"] = evaluate(det, stress)
        m = results[det.name]["micro"]
        st = results[f"{det.name}|stress"]["micro"]
        print(
            f"{det.name:22s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} | stress "
            f"P={st['precision']:.3f} R={st['recall']:.3f} F1={st['f1']:.3f}"
        )
    write_json(
        REPO_ROOT / "results/guardrails/pii.json",
        {
            "meta": run_metadata(
                config={"datasets": ["data/security/pii_synthetic.jsonl", "data/security/pii_stress.jsonl"]},
                seed=7,
            ),
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
