"""SkillCompass API (FastAPI): salary estimate, resume skill extraction and role skill reference.

Run from the project root:   uvicorn api.main:app --reload      (docs at http://127.0.0.1:8000/docs)
Train the salary model first: python -m src.salary_model
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src import company_match, resume_parser, salary_model as sm, skill_analyzer

app = FastAPI(title="SkillCompass API", description="Indicative salary estimates and resume skill-gap analysis from job postings.")
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        b = sm.load_bundle()
        if not sm.bundle_is_current(b):
            raise HTTPException(status_code=503, detail="Salary model is not trained yet. Run: python -m src.salary_model")
        _bundle = b
    return _bundle


class PredictIn(BaseModel):
    role: str | None = Field(None, max_length=120)
    location: str | None = Field(None, max_length=120)
    experience: float | None = Field(None, ge=0, le=60)
    work_mode: str | None = Field(None, max_length=60)
    education: str | None = Field(None, max_length=60)
    skills: list[str] = Field(default_factory=list, max_length=60)


class GapIn(BaseModel):
    role: str = Field(..., max_length=120)
    skills: list[str] = Field(default_factory=list, max_length=200)


class CompanyIn(BaseModel):
    url: str | None = Field(None, max_length=2000)
    job_text: str | None = Field(None, max_length=60000)
    skills: list[str] = Field(default_factory=list, max_length=200)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(body: PredictIn):
    b = get_bundle()
    r = sm.predict_salary(b, role=body.role, experience=body.experience, location=body.location,
                          skills=", ".join(body.skills), work_mode=body.work_mode, education=body.education)
    return {**r, "model": b["best_name"], "unit": "INR per annum",
            "disclaimer": "Estimate based on posted salaries; not a guaranteed salary."}


@app.post("/api/resume")
async def resume_upload(file: UploadFile = File(None)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file selected. Please upload your resume as a PDF.")
    if not file.filename.lower().endswith(".pdf") or file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Please upload your resume as a PDF.")
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="The uploaded PDF file is empty. Please upload a valid resume.")
    if len(content) > resume_parser.MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large. Maximum file size is 10 MB.")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Please upload your resume as a PDF.")
    try:
        text = resume_parser.extract_text_from_pdf(content)
    except ValueError as err:       # e.g. scanned PDF without selectable text
        raise HTTPException(status_code=400, detail=str(err))
    return {"filename": file.filename, "skills": resume_parser.match_skills(text)}


@app.get("/api/role-skills")
def role_skills(role: str):
    return skill_analyzer.get_role_skill_reference(role)


@app.post("/api/skill-gap")
def skill_gap(body: GapIn):
    ref = skill_analyzer.get_role_skill_reference(body.role)
    if not ref.get("sufficient_data"):
        raise HTTPException(status_code=404, detail=ref.get("message", "Not enough postings for this role."))
    d = resume_parser.load_skills_dictionary()
    return skill_analyzer.compute_skill_gap(body.skills, ref.get("skills", []), body.role, d.get("learning_suggestions"))


@app.post("/api/company-match")
def company_match_route(body: CompanyIn):
    """Compare reviewed resume skills with a public job-posting link or pasted job text. Add rate limiting before public use."""
    try:
        if body.job_text and body.job_text.strip():
            job = {"url": "", "title": "Pasted job description", "company": "", "text": body.job_text.strip()[:company_match.MAX_TEXT_CHARS],
                   "source": "pasted text"}
        elif body.url:
            job = company_match.fetch_job_page(body.url)
        else:
            raise ValueError("Provide a job posting url or job_text.")
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    result = company_match.analyze_company_fit(body.skills, job["text"])
    if not result.get("sufficient_data"):
        raise HTTPException(status_code=422, detail=result["message"])
    return {**result, "job": {k: job.get(k, "") for k in ("url", "title", "company", "source")}}
