from pathlib import Path

from secfin.corpus.chunking import chunk_passage, citation, with_prefix
from secfin.corpus.ecfr import parse_part

FIXTURE = Path(__file__).parent / "fixtures" / "mini_part.xml"


def _by_id():
    return {p.passage_id: p for p in parse_part(FIXTURE, "1005")}


def test_paragraph_ids_follow_cfr_hierarchy():
    ps = _by_id()
    assert {
        "1005.2",
        "1005.2(a)(1)",
        "1005.2(a)(2)",
        "1005.2(b)(1)",
        "1005.2(b)(2)",
        "1005.2(b)(3)",
        "1005.2(c)",
    } <= set(ps)
    assert ps["1005.2(a)(1)"].regulation == "Regulation E"
    # roman (i) under (a)(2) stays in its parent
    assert "Requests and receives" in ps["1005.2(a)(2)"].text


def test_italic_fifth_level_is_not_a_new_paragraph():
    ps = _by_id()
    assert "italic fifth-level" in ps["1005.2(b)(3)"].text
    assert "1005.2(b)(1)#2" not in ps


def test_roman_i_after_colon_is_not_letter_i():
    ps = _by_id()
    assert "1005.20(i)" not in ps
    assert "Complies with all other provisions" in ps["1005.20(h)"].text


def test_interpretation_section_heading_resets_hierarchy():
    ps = [p for p in parse_part(FIXTURE, "1005") if "Supp" in p.passage_id]
    titles = [p.title for p in ps]
    assert titles[0] == "Section 1005.2 Definitions > 2(a) Access Device"
    assert titles[1].startswith("Section 1005.18")


def test_chunks_keep_passage_id_and_respect_max_chars():
    long = {
        "passage_id": "1005.6(b)",
        "regulation": "Regulation E",
        "title": "t",
        "text": "\n".join(["A sentence that is fairly long. " * 10] * 10),
    }
    chunks = chunk_passage(long, max_chars=400)
    assert len(chunks) > 1
    assert all(c.passage_id == "1005.6(b)" and len(c.text) <= 400 for c in chunks)


def test_title_prefix_and_citation():
    p = {"passage_id": "1005.6(b)(1)", "regulation": "Regulation E", "title": "1005.6 Liability", "text": "x"}
    c = with_prefix(chunk_passage(p), "title")[0]
    assert citation(c) == "Regulation E, 12 CFR 1005.6(b)(1) - 1005.6 Liability"
    assert c.index_text.startswith("Regulation E, 12 CFR 1005.6(b)(1)")
    assert c.text == "x"
