"""Module 6 dashboard on the REAL dataset in data/raw: every page opens, filters and the estimate work,
and the resume flow runs with real PDFs through the real resume_parser and skill_analyzer."""
from unittest import mock

import pymupdf
import pytest
from streamlit.testing.v1 import AppTest

from src import preprocessing as prep

pytestmark = pytest.mark.skipif(not prep.list_raw_files(), reason="no dataset in data/raw")
APP = str(prep.ROOT / "dashboard" / "app.py")
PAGES = ["Overview", "Data & EDA", "Job Market", "Salary Prediction", "Resume & Skill Gap"]


def pdf_bytes(text: str | None) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    if text:
        page.insert_text((50, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


class FakeUpload:
    type = "application/pdf"

    def __init__(self, name, data):
        self.name, self._data, self.size = name, data, len(data)

    def getvalue(self):
        return self._data


def new_app():
    return AppTest.from_file(APP, default_timeout=900)


def test_every_page_opens_without_errors():
    at = new_app()
    at.run()
    for page in PAGES:
        at.radio(key="page").set_value(page).run()
        assert not at.exception, f"{page}: {[e.value for e in at.exception]}"
        assert not at.error, f"{page}: {[e.value for e in at.error]}"
        text = " ".join(m.value for m in at.markdown)
        assert "Module 1" not in text and "Module 2" not in text      # no module labels in the UI


def test_salary_page_has_no_training_controls_and_estimate_works():
    at = new_app()
    at.run()
    at.radio(key="page").set_value("Salary Prediction").run()
    assert not at.exception
    labels = [w.label for w in at.slider] + [w.label for w in at.number_input] + [w.label for w in at.checkbox]
    for bad in ["Test share (%)", "CV folds", "Random seed", "Top skills used as features", "Time-based split (newest rows = test)"]:
        assert bad not in labels
    assert not [b for b in at.button if "Train" in b.label]
    at.selectbox(key="est_role").set_value(at.selectbox(key="est_role").options[1]).run()
    [b for b in at.button if b.label == "Estimate salary"][0].click().run()
    assert not at.exception
    assert any("Estimated Annual Salary" in m.value for m in at.markdown)
    headings = [m.value for m in at.markdown if m.value.startswith("###")]
    assert any("Estimate a salary" in h for h in headings)


def test_job_market_filters():
    at = new_app()
    at.run()
    at.radio(key="page").set_value("Job Market").run()
    first_city = at.multiselect[0].options[0]
    at.multiselect[0].set_value([first_city]).run()
    assert not at.exception
    assert any("Dataset coverage" in m.value for m in at.markdown)


@pytest.mark.parametrize("text,expect_skills", [
    ("Alex Mercer\nData Scientist\nSkills: Python, SQL, Pandas, Machine Learning, Docker, AWS.\n", True),
    (None, False),     # blank page = scanned / image-only PDF: must show a clear message, never crash
])
def test_resume_flow_with_real_pdf(text, expect_skills):
    at = new_app()
    with mock.patch("streamlit.file_uploader", return_value=FakeUpload("resume.pdf", pdf_bytes(text))):
        at.run()
        at.radio(key="page").set_value("Resume & Skill Gap").run()
        assert not at.exception, [e.value for e in at.exception]
        if expect_skills:
            skills = [s.lower() for s in at.multiselect(key="ms_reviewed_skills").value]
            assert skills, "no skills extracted from a normal text PDF"
            assert any("python" in s for s in skills)
            at.text_input(key="input_custom_skill").set_value("GraphQL").run()
            [b for b in at.button if b.label == "Add skill"][0].click().run()
            assert "GraphQL" in at.multiselect(key="ms_reviewed_skills").value      # added skills are selected
            assert not at.exception
            assert at.metric or at.warning                                          # gap result or clear "insufficient data" message
        else:
            assert at.warning, "scanned PDF should show a clear warning"
            assert any("scan" in w.value.lower() or "ocr" in w.value.lower() for w in at.warning)


def test_non_pdf_is_rejected():
    at = new_app()
    fake = FakeUpload("resume.txt", b"plain text")
    fake.type = "text/plain"
    with mock.patch("streamlit.file_uploader", return_value=fake):
        at.run()
        at.radio(key="page").set_value("Resume & Skill Gap").run()
        assert any("PDF" in w.value for w in at.warning)


def test_company_match_section():
    """Section 4 of the Resume page: link/paste -> match score, missing skills, improvement tips. The web fetch is replaced."""
    job = {"url": "https://example.com/careers/1", "title": "Data Engineer", "company": "Acme", "source": "visible page text",
           "text": ("We need Python, SQL, Docker, AWS and Spark skills. Machine Learning is a plus. 3+ years of experience. " * 4)}
    at = new_app()
    with mock.patch("streamlit.file_uploader", return_value=FakeUpload("resume.pdf", pdf_bytes("Skills: Python, SQL"))), \
            mock.patch("src.company_match.fetch_job_page", return_value=job):
        at.run()
        at.radio(key="page").set_value("Resume & Skill Gap").run()
        assert not at.exception, [e.value for e in at.exception]
        at.text_input(key="company_url").set_value("https://example.com/careers/1").run()
        [b for b in at.button if b.label == "Check match"][0].click().run()
        assert not at.exception, [e.value for e in at.exception]
        texts = " ".join(m.value for m in at.markdown)
        assert "What to improve in your resume" in texts
        assert "Only add skills you genuinely have" in texts
        assert [m for m in at.metric if m.label == "Match with this job"]
        # an unreachable page shows a clear message, not a crash
        with mock.patch("src.company_match.fetch_job_page", side_effect=ValueError("The website blocks automated access or needs a login. You can paste the job description text instead.")):
            at.text_input(key="company_url").set_value("https://blocked.example").run()
            [b for b in at.button if b.label == "Check match"][0].click().run()
            assert not at.exception
            assert any("paste" in w.value.lower() for w in at.warning)
