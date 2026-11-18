"""Modules 3 & 4: Resume Parsing, Skill Extraction, and Skill Gap Analysis.

Provides:
- extract_text_from_pdf: PyMuPDF (fitz) text extraction with pypdf fallback
- normalize_text: Preserves skill-critical punctuation (+, #, ., -)
- match_skills: Phrase-based NLP skill and alias matching against data/skills.json
- get_role_skill_reference: Computes top N skills from cleaned_jobs.csv for a role
- calculate_skill_gap: Transparent gap analysis and targeted learning suggestions
"""
from __future__ import annotations

import io
import json
import re
from collections import Counter
from pathlib import Path

# Paths
DEFAULT_SKILLS_PATH = Path("data/skills.json")
FALLBACK_SKILLS_PATH = Path("skills.json")
DEFAULT_DATASET_PATH = Path("data/processed/cleaned_jobs.csv")

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


def load_skills_dictionary(path: Path | str | None = None) -> dict:
    """Load the skills dictionary, aliases, and learning suggestions."""
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.extend([DEFAULT_SKILLS_PATH, FALLBACK_SKILLS_PATH])

    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

    return {
        "skills": [
            "Python", "SQL", "Machine Learning", "Deep Learning", "Pandas",
            "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "FastAPI",
            "Flask", "Django", "Docker", "Kubernetes", "Git", "AWS", "Azure",
            "GCP", "Tableau", "Power BI", "Excel", "dbt", "Airflow", "Spark",
            "Hadoop", "JavaScript", "React", "Node.js", "Computer Vision",
            "NLP", "Statistics", "Linux"
        ],
        "aliases": {
            "ml": "Machine Learning", "dl": "Deep Learning", "sklearn": "Scikit-learn",
            "postgres": "SQL", "postgresql": "SQL", "mysql": "SQL",
            "js": "JavaScript", "nodejs": "Node.js", "node.js": "Node.js",
            "powerbi": "Power BI", "tf": "TensorFlow", "k8s": "Kubernetes",
            "nlp": "NLP", "cv": "Computer Vision"
        },
        "learning_suggestions": {}
    }


def _read_pages_pymupdf(pdf_bytes: bytes) -> list[str]:
    """Extract page strings using PyMuPDF (or legacy fitz)."""
    try:
        import pymupdf as pdf_lib
    except ImportError:
        import fitz as pdf_lib
    doc = pdf_lib.open(stream=pdf_bytes, filetype="pdf")
    try:
        return [page.get_text() for page in doc]
    finally:
        doc.close()


def _read_pages_pypdf(pdf_bytes: bytes) -> list[str]:
    """Fallback: extract page strings using pypdf."""
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    return [(page.extract_text() or "") for page in reader.pages]


def extract_text_from_pdf(source: bytes | io.BytesIO | str | Path) -> str:
    """Extract selectable text from PDF.

    Raises clear ValueError for non-PDF, empty, or scanned PDFs (no OCR).
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise ValueError(f"File not found: {p}")
        pdf_bytes = p.read_bytes()
    elif isinstance(source, (io.BytesIO, io.BufferedReader)):
        pdf_bytes = source.read()
    elif isinstance(source, (bytes, bytearray)):
        pdf_bytes = bytes(source)
    else:
        raise ValueError("extract_text_from_pdf expects raw PDF bytes, BytesIO stream, or file path.")

    if len(pdf_bytes) == 0:
        raise ValueError("The uploaded PDF file is empty. Please upload a valid resume.")

    if len(pdf_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError("File too large. Maximum file size is 10 MB.")

    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("Please upload your resume as a PDF.")

    pages = None
    errors = []
    for reader_fn in (_read_pages_pymupdf, _read_pages_pypdf):
        try:
            pages = reader_fn(pdf_bytes)
            break
        except ImportError:
            pass
        except Exception as err:
            errors.append(str(err))

    if pages is None:
        if errors:
            raise ValueError(f"Could not read the PDF structure: {errors[0]}")
        raise ValueError("No PDF text extraction engine available. Please install pymupdf or pypdf.")

    full_text = "\n".join(p for p in pages if p and p.strip()).strip()

    if not full_text or len(re.sub(r"\s+", "", full_text)) == 0:
        raise ValueError(
            "Scanned or image-only PDF detected. Scanned PDFs are not supported because "
            "OCR is not included. Please upload a PDF with selectable text."
        )

    return full_text


def normalize_text(text: str) -> str:
    """Lowercase text and retain essential skill punctuation (+, #, ., -)."""
    t = text.lower()
    t = re.sub(r"[^\w\s\+\#\.-]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return f" {t.strip()} "


def _phrase_pattern(phrase: str) -> re.Pattern:
    """Match whole phrases preserving symbols like CI/CD, C++, and Node.js."""
    norm = normalize_text(phrase).strip()
    return re.compile(rf"(?<![\w+#-])(?<!\w\.){re.escape(norm)}(?![\w+#-])(?!\.\w)", re.IGNORECASE)


def match_skills(text: str, dict_data: dict | None = None) -> list[str]:
    """Extract canonical skills from text using phrase matching and aliases."""
    if not dict_data:
        dict_data = load_skills_dictionary()

    canonical_skills = dict_data.get("skills", [])
    aliases = dict_data.get("aliases", {})
    norm_text = normalize_text(text)

    found = {skill for skill in canonical_skills if _phrase_pattern(skill).search(norm_text)}
    found |= {canonical for alias, canonical in aliases.items() if _phrase_pattern(alias).search(norm_text)}
    return sorted(found)


def get_role_skill_reference(role: str, dataset_path: Path | str | None = None,
                             min_postings: int = 10, top_n: int = 10) -> dict:
    """Calculate the top N reference skills from the dataset for the selected role."""
    import pandas as pd

    p = Path(dataset_path or DEFAULT_DATASET_PATH)
    if not p.exists():
        return {
            "role": role,
            "n_postings": 0,
            "sufficient_data": False,
            "skills": [],
            "message": f"Cleaned dataset not found at '{p}'. Please run prepare_dataset.py."
        }

    try:
        df = pd.read_csv(p, usecols=["job_title", "skills_required"], low_memory=False)
    except Exception as e:
        return {
            "role": role,
            "n_postings": 0,
            "sufficient_data": False,
            "skills": [],
            "message": f"Error loading dataset: {e}"
        }

    role_df = df[df["job_title"].astype(str).str.strip().str.lower() == role.strip().lower()]
    n_postings = len(role_df)

    if n_postings < min_postings:
        return {
            "role": role,
            "n_postings": n_postings,
            "sufficient_data": False,
            "skills": [],
            "message": f"Insufficient data for role '{role}' ({n_postings} postings found; minimum {min_postings} required)."
        }

    counter = Counter()
    for row in role_df["skills_required"].dropna():
        for skill in str(row).split(","):
            s = skill.strip()
            if s:
                counter[s] += 1

    top_skills = [skill for skill, _ in counter.most_common(top_n)]
    return {
        "role": role,
        "n_postings": n_postings,
        "sufficient_data": True,
        "skills": top_skills,
        "message": f"Top {len(top_skills)} skills derived from {n_postings:,} {role} postings."
    }


def calculate_skill_gap(candidate_skills: list[str], target_role: str,
                        dataset_path: Path | str | None = None,
                        dict_data: dict | None = None) -> dict:
    """Compare candidate skills against target role reference skills."""
    if not dict_data:
        dict_data = load_skills_dictionary()

    ref = get_role_skill_reference(target_role, dataset_path=dataset_path)
    if not ref["sufficient_data"]:
        return {
            "role": target_role,
            "sufficient_data": False,
            "score_pct": None,
            "score_text": "N/A",
            "matched": [],
            "missing": [],
            "suggestions": [],
            "message": ref["message"],
        }

    ref_skills = ref["skills"]
    cand_lower = {s.strip().lower(): s for s in candidate_skills}

    matched = []
    missing = []
    for s in ref_skills:
        if s.lower() in cand_lower:
            matched.append(s)
        else:
            missing.append(s)

    total_ref = len(ref_skills)
    n_matched = len(matched)
    score_pct = round((n_matched / total_ref) * 100) if total_ref > 0 else 0
    score_text = f"{n_matched} matched / {total_ref} top skills = {score_pct}%"

    suggestions_dict = dict_data.get("learning_suggestions", {})
    suggestions = []
    for s in missing:
        tip = suggestions_dict.get(s, f"Build hands-on projects and explore documentation for {s}.")
        suggestions.append({"skill": s, "tip": tip})

    return {
        "role": target_role,
        "sufficient_data": True,
        "n_postings": ref["n_postings"],
        "score_pct": score_pct,
        "score_text": score_text,
        "matched": matched,
        "missing": missing,
        "suggestions": suggestions,
        "message": ref["message"],
    }


def sample_pdf_bytes() -> bytes:
    """Create a sample PDF in memory with PyMuPDF for instant testing."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    sample_text = (
        "Jane Doe - Senior Data Scientist\n"
        "Email: jane.doe@example.com | Bengaluru, India\n\n"
        "Summary:\n"
        "Experienced Data Scientist with 6+ years in Machine Learning, Python, and SQL.\n"
        "Proven track record in deploying Deep Learning models, Scikit-learn pipelines, and Docker.\n\n"
        "Technical Skills:\n"
        "Languages & Tools: Python, SQL, Git, Docker, AWS, Pandas, NumPy, FastAPI.\n"
        "Data Science & ML: Machine Learning, Scikit-learn, TensorFlow, PyTorch, Statistics, NLP.\n\n"
        "Experience:\n"
        "Lead ML Engineer at Analytics Corp (2021 - Present)\n"
        "- Built CI/CD pipelines for automated retraining and model monitoring.\n"
        "- Deployed NLP classifiers with FastAPI and Kubernetes on AWS.\n"
    )
    page.insert_text((50, 72), sample_text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data
