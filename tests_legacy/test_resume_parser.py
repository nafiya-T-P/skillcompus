"""Tests for Module 3 (Resume text extraction and skill matching)."""
import pymupdf
import pytest

import resume_parser as rp

DICT = {
    "skills": ["Python", "SQL", "CI/CD", "C++", "Node.js", "Node", "Machine Learning", "Docker", "AWS", "Java", "JavaScript"],
    "aliases": {"ml": "Machine Learning", "nodejs": "Node.js"}
}


def pdf_bytes(text):
    doc = pymupdf.open()
    page = doc.new_page()
    if text:
        page.insert_text((50, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_text_pdf_is_read():
    text = rp.extract_text_from_pdf(pdf_bytes("Skills: Python, SQL, Docker."))
    assert "Python" in text and "SQL" in text


def test_scanned_pdf_gives_clear_message():
    with pytest.raises(ValueError) as e:
        rp.extract_text_from_pdf(pdf_bytes(None))
    msg = str(e.value).lower()
    assert "scanned" in msg and "ocr" in msg


@pytest.mark.parametrize("bad", [b"", b"not a pdf at all"])
def test_unreadable_file_raises_value_error(bad):
    with pytest.raises(ValueError):
        rp.extract_text_from_pdf(bad)


def test_falls_back_to_pypdf_when_pymupdf_is_missing(monkeypatch):
    pytest.importorskip("pypdf")
    def no_pymupdf(_):
        raise ImportError("pymupdf not installed")
    monkeypatch.setattr(rp, "_read_pages_pymupdf", no_pymupdf)
    assert "Python" in rp.extract_text_from_pdf(pdf_bytes("Skills: Python"))


@pytest.mark.parametrize("text,expected", [
    ("Proficient in Python.", {"Python"}),
    ("Skills: SQL, Docker, AWS.", {"SQL", "Docker", "AWS"}),
    ("Built CI/CD pipelines", {"CI/CD"}),
    ("Wrote C++ services", {"C++"}),
    ("Used Node.js daily", {"Node.js"}),
    ("Experience with ML and nodejs", {"Machine Learning", "Node.js"}),
    ("I write JavaScript", {"JavaScript"}),
    ("Python and SQL", {"Python", "SQL"}),
])
def test_skill_matching(text, expected):
    assert set(rp.match_skills(text, DICT)) == expected


def test_matching_is_sorted_and_unique():
    out = rp.match_skills("python Python PYTHON sql", DICT)
    assert out == sorted(set(out)) == ["Python", "SQL"]


def test_role_skill_reference_and_gap():
    ref = rp.get_role_skill_reference("Data Scientist")
    assert ref["sufficient_data"] is True
    assert len(ref["skills"]) > 0

    gap = rp.calculate_skill_gap(["Python", "SQL"], "Data Scientist")
    assert gap["sufficient_data"] is True
    assert "matched" in gap
    assert "missing" in gap
    assert gap["score_pct"] is not None
