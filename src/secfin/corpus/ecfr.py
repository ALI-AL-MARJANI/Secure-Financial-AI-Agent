"""Download and parse US consumer-banking regulations from the eCFR into citable passages.

A passage is the smallest unit the golden set refers to:
  * regulation sections: one second-level paragraph, e.g. ``1005.6(b)(1)``, or the
    top-level paragraph ``1005.6(a)`` when it has no numbered children;
  * appendices / official interpretations: the text under the lowest heading,
    e.g. ``1005 Supplement I > 2(b) Account``.
"""

from __future__ import annotations

import hashlib
import re
import string
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

ECFR_URL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml?part={part}"

# Matches "(a) ..." and the combined form "(a)(1) ..." used in definition sections.
_LETTER = re.compile(r"^\(([a-z])\)(?:\((\d+)\))?\s")
_NUMBER = re.compile(r"^\((\d+)\)\s")
_HEADINGS = ("HD1", "HD2", "HD3")
_SECTION_HEADING = re.compile(r"^(Section\s+\d+\.\d+|Appendix\s+[A-Z]+\b)")


@dataclass
class Passage:
    passage_id: str
    part: str
    regulation: str
    title: str
    text: str
    headings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def download_part(part: str, date: str, out_dir: Path, title: int = 12) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"title{title}_part{part}_{date}.xml"
    if path.exists():
        return path
    # The versioner endpoint refuses uncompressed responses.
    resp = httpx.get(
        ECFR_URL.format(date=date, title=title, part=part),
        headers={"Accept-Encoding": "gzip"},
        timeout=120,
    )
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(el: ET.Element) -> str:
    if el.tag == "TABLE":
        rows = []
        for tr in el.iter("TR"):
            cells = [" ".join("".join(c.itertext()).split()) for c in tr if c.tag in ("TD", "TH")]
            rows.append(" | ".join(cells))
        return "\n".join(rows)
    return " ".join("".join(el.itertext()).split())


def _regulation_name(root: ET.Element) -> str:
    head = root.find("HEAD")
    head_text = _text(head) if head is not None else ""
    m = re.search(r"\(REGULATION ([A-Z]+)\)", head_text, re.IGNORECASE)
    return f"Regulation {m.group(1).upper()}" if m else head_text


def _section_passages(sec: ET.Element, part: str, regulation: str) -> Iterator[Passage]:
    sec_no = sec.get("N", "")
    head = sec.find("HEAD")
    sec_title = _text(head).lstrip("§ ").strip() if head is not None else sec_no

    current_id = sec_no
    buf: list[str] = []
    letter = ""  # last top-level letter seen
    expected = "a"
    last_number = 0  # last second-level number under `letter`

    def flush() -> Passage | None:
        body = "\n".join(t for t in buf if t)
        if not body:
            return None
        return Passage(
            passage_id=current_id,
            part=part,
            regulation=regulation,
            title=sec_title,
            text=body,
            headings=[sec_title],
        )

    for child in sec:
        if child.tag in ("HEAD", "CITA", "SECAUTH", "img"):
            continue
        text = _text(child)
        if not text:
            continue
        # Fifth-level paragraphs are marked with an italic number: "(<I>1</I>) ...".
        if (child.text or "").strip() == "(" and len(child) and child[0].tag == "I":
            buf.append(text)
            continue
        m_letter = _LETTER.match(text)
        m_number = _NUMBER.match(text)
        # "(i)", "(v)", "(x)" are letters only when they follow the previous letter and do not
        # continue a list introduced by a colon; otherwise they are roman sub-items.
        roman_list_item = (
            bool(m_letter) and m_letter.group(1) in "ivx" and bool(buf) and buf[-1].rstrip().endswith(":")
        )
        if m_letter and m_letter.group(1) == expected and not roman_list_item:
            p = flush()
            if p:
                yield p
            letter = m_letter.group(1)
            nxt = string.ascii_lowercase.index(letter) + 1
            expected = string.ascii_lowercase[nxt] if nxt < 26 else ""
            current_id = f"{sec_no}({letter})"
            last_number = 0
            if m_letter.group(2):
                current_id += f"({m_letter.group(2)})"
                last_number = int(m_letter.group(2))
            buf = [text]
            continue
        num = None
        if m_number:
            num = int(m_number.group(1))
        elif m_letter and m_letter.group(2) and m_letter.group(1) == letter:
            num = int(m_letter.group(2))
        # Deeper lists such as (b)(3)(ii)(D)(1) restart at 1; only the next number in sequence is level 2.
        if num is not None and letter and num == last_number + 1:
            p = flush()
            if p:
                yield p
            last_number = num
            current_id = f"{sec_no}({letter})({num})"
            buf = [text]
            continue
        buf.append(text)
    p = flush()
    if p:
        yield p


def _appendix_code(name: str) -> str:
    m = re.match(r"(Appendix|Supplement)\s+([A-Z]+)", name)
    if not m:
        return re.sub(r"\W+", "", name)[:12]
    return ("App" if m.group(1) == "Appendix" else "Supp") + m.group(2)


def _appendix_passages(appx: ET.Element, part: str, regulation: str) -> Iterator[Passage]:
    head = appx.find("HEAD")
    appx_title = _text(head) if head is not None else appx.get("N", "")
    code = _appendix_code(appx.get("N", ""))
    stack: dict[str, str] = {}
    buf: list[str] = []
    idx = 0

    def path() -> list[str]:
        return [appx_title] + [stack[h] for h in ("SEC", *_HEADINGS) if h in stack]

    def flush() -> Passage | None:
        nonlocal idx
        body = "\n".join(buf)
        if not body:
            return None
        idx += 1
        heads = path()
        return Passage(
            passage_id=f"{part}.{code}.{idx:03d}",
            part=part,
            regulation=regulation,
            title=" > ".join(heads[1:]) or appx_title,
            text=body,
            headings=heads,
        )

    def walk(el: ET.Element) -> Iterator[Passage]:
        nonlocal buf
        for child in el:
            if child.tag in ("HEAD", "CITA", "img"):
                continue
            if child.tag in _HEADINGS:
                p = flush()
                if p:
                    yield p
                buf = []
                heading = _text(child)
                # Interpretations use HD1/HD2/HD3 inconsistently for section headings, so a
                # "Section 1005.18 ..." heading always resets the hierarchy.
                if _SECTION_HEADING.match(heading):
                    stack.clear()
                    stack["SEC"] = heading
                    continue
                level = _HEADINGS.index(child.tag)
                stack[child.tag] = heading
                for deeper in _HEADINGS[level + 1 :]:
                    stack.pop(deeper, None)
                continue
            if child.tag in ("DIV", "EXTRACT", "GPOTABLE"):
                yield from walk(child)
                continue
            text = _text(child)
            if text:
                buf.append(text)

    yield from walk(appx)
    p = flush()
    if p:
        yield p


def parse_part(xml_path: Path, part: str) -> list[Passage]:
    root = ET.parse(xml_path).getroot()
    regulation = _regulation_name(root)
    passages: list[Passage] = []
    for el in root.iter():
        if el.tag == "DIV8" and el.get("TYPE") == "SECTION":
            passages.extend(_section_passages(el, part, regulation))
        elif el.tag == "DIV9" and el.get("TYPE") == "APPENDIX":
            passages.extend(_appendix_passages(el, part, regulation))
    seen: dict[str, int] = {}
    for p in passages:
        # eCFR occasionally repeats a paragraph label (e.g. reserved + re-added); keep IDs unique.
        if p.passage_id in seen:
            seen[p.passage_id] += 1
            p.passage_id = f"{p.passage_id}#{seen[p.passage_id]}"
        else:
            seen[p.passage_id] = 1
    return passages
