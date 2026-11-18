"""API (api/main.py) on the REAL dataset: salary estimate, resume upload rules, role skills and skill gap."""
import pymupdf
import pytest
from fastapi.testclient import TestClient

from src import preprocessing as prep
from src import salary_model as sm

pytestmark = pytest.mark.skipif(not prep.list_raw_files(), reason="no dataset in data/raw")


@pytest.fixture(scope="module")
def client():
    prep.ensure_cleaned()
    frame, _ = prep.prepare_for_model(prep.load_cleaned())
    assert sm.get_or_train_bundle(frame) is not None
    from api.main import app
    return TestClient(app)


def pdf_bytes(text):
    doc = pymupdf.open()
    page = doc.new_page()
    if text:
        page.insert_text((50, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_predict_returns_a_qualified_range(client):
    r = client.post("/api/predict", json={"role": prep.load_cleaned()["job_title"].dropna().iloc[0], "experience": 3, "skills": ["python"]})
    assert r.status_code == 200
    d = r.json()
    assert d["low"] <= d["point"] <= d["high"]
    assert "not a guaranteed salary" in d["disclaimer"].lower()


def test_resume_rules(client):
    r = client.post("/api/resume", files={"file": ("a.txt", b"x", "text/plain")})
    assert r.status_code == 400 and r.json()["detail"] == "Please upload your resume as a PDF."
    r = client.post("/api/resume", files={"file": ("a.pdf", b"", "application/pdf")})
    assert r.status_code == 400 and "empty" in r.json()["detail"].lower()
    r = client.post("/api/resume", files={"file": ("scan.pdf", pdf_bytes(None), "application/pdf")})
    assert r.status_code == 400 and ("scan" in r.json()["detail"].lower() or "ocr" in r.json()["detail"].lower())
    r = client.post("/api/resume", files={"file": ("cv.pdf", pdf_bytes("Skills: Python, SQL, Docker, Machine Learning"), "application/pdf")})
    assert r.status_code == 200 and any("python" in s.lower() for s in r.json()["skills"])


def test_role_skills_and_gap(client):
    role = prep.load_cleaned()["job_title"].value_counts().index[0]
    ref = client.get("/api/role-skills", params={"role": role}).json()
    assert ref["sufficient_data"] is True and ref["skills"]
    gap = client.post("/api/skill-gap", json={"role": role, "skills": ["python"]}).json()
    assert 0 <= gap["score"] <= 100 and gap["formula"] and "matched" in gap and "missing" in gap


def test_company_match_with_pasted_text(client):
    text = "We need Python, SQL, Docker, AWS and Spark skills. Machine Learning is a plus. 3+ years of experience. " * 3
    r = client.post("/api/company-match", json={"job_text": text, "skills": ["Python"]})
    assert r.status_code == 200
    d = r.json()
    assert 0 <= d["score"] <= 100 and d["missing"] and d["tips"] and "guarantee" in d["disclaimer"].lower()


def test_company_match_rejects_internal_addresses_and_empty_input(client):
    assert client.post("/api/company-match", json={"url": "http://127.0.0.1/admin", "skills": ["Python"]}).status_code == 400
    assert client.post("/api/company-match", json={"url": "http://169.254.169.254/latest/meta-data", "skills": []}).status_code == 400
    assert client.post("/api/company-match", json={"skills": ["Python"]}).status_code == 400
    r = client.post("/api/company-match", json={"job_text": "We value teamwork. " * 20, "skills": ["Python"]})
    assert r.status_code == 422
