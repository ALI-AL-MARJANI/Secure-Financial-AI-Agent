"""Generate a seeded synthetic PII benchmark (EN + FR) with character-level gold spans.

Positives: Luhn-valid card numbers (Visa / Mastercard / Amex, 3 formats), mod-97-valid IBANs
(FR / DE / GB / ES, with and without spaces), SSA-valid SSNs, emails, US and FR phone numbers.
Hard negatives: dollar amounts, dates, CFR citations, customer IDs, ZIP codes, Luhn-invalid 16-digit
tracking numbers, 9-digit transfer references.

Caveat: templates and the validated regex were written by the same author, which favours the regex.
Presidio is the independent reference.
"""

from __future__ import annotations

import argparse
import random
import string

from secfin.guardrails.pii import luhn_ok
from secfin.runinfo import REPO_ROOT, write_json, write_jsonl

EN = [
    "My card number is {CREDIT_CARD}, can you check the last payment?",
    "Please refund the charge on card {CREDIT_CARD}.",
    "Send the statement to {EMAIL} please.",
    "You can call me back at {PHONE}.",
    "My SSN is {US_SSN} if you need to verify my identity.",
    "Wire the funds to IBAN {IBAN} by Friday.",
    "Update my contact details: {EMAIL}, phone {PHONE}.",
    "The card {CREDIT_CARD} was stolen yesterday, my number is {PHONE}.",
    "Is IBAN {IBAN} correct for an international transfer?",
    "I dispute a transfer; my social security number is {US_SSN} and email {EMAIL}.",
]
FR = [
    "Mon numéro de carte est {CREDIT_CARD}, pouvez-vous vérifier ?",
    "Envoyez le relevé à {EMAIL} s'il vous plaît.",
    "Vous pouvez me joindre au {PHONE}.",
    "Virez les fonds sur l'IBAN {IBAN} avant vendredi.",
    "Ma carte {CREDIT_CARD} a été volée, rappelez-moi au {PHONE}.",
]
NEG = [
    "I want a mortgage of {AMOUNT} over 25 years.",
    "Can I transfer {AMOUNT} to my savings account on {DATE}?",
    "What does 12 CFR {CITATION} say about unauthorized transfers?",
    "My customer ID is {CUSTID} and my ZIP code is {ZIP}.",
    "The tracking number for my new card is {BADCARD}.",
    "Transfer reference {REF9} was rejected on {DATE}.",
    "Is a deposit of {AMOUNT} subject to the large deposit exception?",
    "Le virement de {AMOUNT} prévu le {DATE} a-t-il été exécuté ?",
    "Order {ORDER} for {AMOUNT} has not arrived.",
    "Between {DATE} and {DATE2} I made 3 transfers of {AMOUNT}.",
]
IBAN_LEN = {"FR": 27, "DE": 22, "GB": 22, "ES": 24}


def card(rng: random.Random) -> str:
    kind = rng.choice(["visa", "mc", "amex"])
    prefix, length = {
        "visa": ("4", 16),
        "mc": (str(rng.randint(51, 55)), 16),
        "amex": (rng.choice(["34", "37"]), 15),
    }[kind]
    body = prefix + "".join(rng.choice(string.digits) for _ in range(length - len(prefix) - 1))
    check = next(d for d in string.digits if luhn_ok(body + d))
    digits = body + check
    fmt = rng.choice(["plain", "space", "dash"])
    if fmt == "plain":
        return digits
    groups = (
        [digits[0:4], digits[4:10], digits[10:]]
        if kind == "amex"
        else [digits[i : i + 4] for i in range(0, 16, 4)]
    )
    return (" " if fmt == "space" else "-").join(groups)


def bad_card(rng: random.Random) -> str:
    while True:
        digits = "".join(rng.choice(string.digits) for _ in range(16))
        if not luhn_ok(digits):
            return digits


def iban(rng: random.Random) -> str:
    cc = rng.choice(list(IBAN_LEN))
    bban = "".join(rng.choice(string.digits) for _ in range(IBAN_LEN[cc] - 4))
    num = int("".join(str(int(c, 36)) for c in bban + cc + "00"))
    s = f"{cc}{98 - num % 97:02d}{bban}"
    return " ".join(s[i : i + 4] for i in range(0, len(s), 4)) if rng.random() < 0.5 else s


def ssn(rng: random.Random) -> str:
    area = rng.choice([a for a in range(1, 900) if a != 666])
    return f"{area:03d}-{rng.randint(1, 99):02d}-{rng.randint(1, 9999):04d}"


def phone(rng: random.Random, lang: str) -> str:
    if lang == "fr" or rng.random() < 0.3:
        d = [rng.randint(10, 99) for _ in range(4)]
        first = rng.choice(["06", "07", "01"])
        return rng.choice(
            [f"{first} {d[0]} {d[1]} {d[2]} {d[3]}", f"+33 {first[1]} {d[0]} {d[1]} {d[2]} {d[3]}"]
        )
    a, b, c = rng.randint(201, 989), rng.randint(200, 999), rng.randint(0, 9999)
    return rng.choice([f"({a}) {b}-{c:04d}", f"{a}-{b}-{c:04d}", f"+1 {a} {b} {c:04d}"])


def email(rng: random.Random) -> str:
    user = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 9)))
    return f"{user}{rng.randint(1, 99)}@{rng.choice(['gmail.com', 'outlook.fr', 'example.org', 'yahoo.com'])}"


def fill(template: str, rng: random.Random, lang: str) -> tuple[str, list[dict]]:
    gen = {
        "CREDIT_CARD": lambda: card(rng),
        "IBAN": lambda: iban(rng),
        "US_SSN": lambda: ssn(rng),
        "EMAIL": lambda: email(rng),
        "PHONE": lambda: phone(rng, lang),
        "AMOUNT": lambda: rng.choice(
            [
                f"${rng.randint(1, 999):,}",
                f"${rng.randint(1000, 9_999_999):,}",
                f"{rng.randint(100_000_000, 999_999_999)} dollars",
                f"{rng.randint(1000, 99999)} €",
            ]
        ),
        "DATE": lambda: rng.choice(
            [
                f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(2020, 2026)}",
                f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            ]
        ),
        "DATE2": lambda: f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/2026",
        "CITATION": lambda: rng.choice(["1005.6(b)(1)", "1030.4(b)(6)", "229.12(c)", "1005.11(c)"]),
        "CUSTID": lambda: f"CUST{rng.randint(100, 999)}",
        "ZIP": lambda: f"{rng.randint(10000, 99999)}",
        "BADCARD": lambda: bad_card(rng),
        "REF9": lambda: f"{rng.randint(100_000_000, 999_999_999)}",
        "ORDER": lambda: f"ORD-{rng.randint(2020, 2026)}-{rng.randint(100, 999)}",
    }
    text, spans, i = "", [], 0
    while i < len(template):
        if template[i] == "{":
            j = template.index("}", i)
            slot = template[i + 1 : j]
            value = gen[slot]()
            if slot in ("CREDIT_CARD", "IBAN", "US_SSN", "EMAIL", "PHONE"):
                spans.append(
                    {"label": slot, "start": len(text), "end": len(text) + len(value), "text": value}
                )
            text += value
            i = j + 1
        else:
            text += template[i]
            i += 1
    return text, spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=400)
    ap.add_argument("--n-neg", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = []
    for _ in range(args.n_pos):
        lang = "fr" if rng.random() < 0.3 else "en"
        text, spans = fill(rng.choice(FR if lang == "fr" else EN), rng, lang)
        rows.append({"text": text, "lang": lang, "spans": spans})
    for _ in range(args.n_neg):
        t = rng.choice(NEG)
        text, spans = fill(t, rng, "fr" if "Le virement" in t else "en")
        rows.append({"text": text, "lang": "fr" if "Le virement" in t else "en", "spans": spans})
    write_jsonl(REPO_ROOT / "data/security/pii_synthetic.jsonl", rows)
    write_json(
        REPO_ROOT / "data/security/pii_manifest.json", {"seed": args.seed, "n": len(rows), **vars(args)}
    )
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
