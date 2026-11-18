'''Resume parsing, skill extraction utilities (Modules 3 & 4).'''

from __future__ import annotations

import io
import json
import re
from collections import Counter
from pathlib import Path

# Paths – adjusted for project layout
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SKILLS_PATH = ROOT_DIR / "data" / "external" / "skills.json"
FALLBACK_SKILLS_PATH = ROOT_DIR / "data" / "skills.json"
DEFAULT_DATASET_PATH = ROOT_DIR / "data" / "processed" / "cleaned_jobs.csv"

# Maximum permitted file size (10 MB)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


def load_skills_dictionary(path: Path | str | None = None) -> dict:
    """Load the maintained skill dictionary (skills, aliases, learning suggestions)."""
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
    # Built‑in fallback dictionary if file missing
    return {
        "skills": [
            "Python", "SQL", "Machine Learning", "Deep Learning", "Pandas",
            "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "FastAPI",
            "Flask", "Django", "Docker", "Kubernetes", "Git", "AWS", "Azure",
            "GCP", "Tableau", "Power BI", "Excel", "dbt", "Airflow",
            "Spark", "Hadoop", "Computer Vision", "NLP", "Statistics",
            "Linux",
        ],
        "aliases": {
            "ml": "Machine Learning",
            "dl": "Deep Learning",
            "sklearn": "Scikit-learn",
            "postgres": "PostgreSQL",
            "postgresql": "PostgreSQL",
            "mysql": "MySQL",
            "sql": "SQL",
            "js": "JavaScript",
            "nodejs": "Node.js",
            "powerbi": "Power BI",
            "tf": "TensorFlow",
        },
        "learning_suggestions": {},
    }


def _read_pages_pymupdf(pdf_bytes: bytes) -> list[str]:
    """Read pages using PyMuPDF (or its legacy name fitz)."""
    try:
        import pymupdf as pdf_lib  # recent versions
    except ImportError:
        import fitz as pdf_lib  # older versions
    doc = pdf_lib.open(stream=pdf_bytes, filetype="pdf")
    try:
        return [page.get_text() for page in doc]
    finally:
        doc.close()


def _read_pages_pypdf(pdf_bytes: bytes) -> list[str]:
    """Read pages using pypdf fallback."""
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    return [(page.extract_text() or "") for page in reader.pages]


def extract_text_from_pdf(source: bytes | io.BytesIO | str | Path) -> str:
    """Selectable text of a PDF. Raises ValueError with a clear message if it cannot be read or has no text.
    Accepts only a bytes object (as used by the tests and dashboard)."""
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
        raise ValueError("extract_text_from_pdf expects raw PDF bytes or a file path.")
    pages, missing, broken = None, [], []
    for reader in (_read_pages_pymupdf, _read_pages_pypdf):
        try:
            pages = reader(pdf_bytes)  # a fresh list every time, so pages are never duplicated
            break
        except ImportError:
            missing.append(reader.__name__.replace("_read_pages_", ""))
        except Exception as err:
            broken.append(str(err))
    if pages is None:
        if broken:
            raise ValueError(f"Could not read the PDF structure: {broken[0]}")
        raise ValueError("No PDF text-extraction library is installed. Run: pip install pymupdf pypdf")
    full_text = "\n".join(p for p in pages if p and p.strip()).strip()
    if not re.sub(r"\s+", "", full_text):
        raise ValueError("Scanned or image-only PDF detected. Scanned PDFs are not supported because OCR is not included. Please upload a PDF with selectable text.")
    return full_text


def normalize_text(text: str) -> str:
    """Lowercase, keep skill-critical punctuation ('+', '#', '.', '-'), collapse spaces."""
    t = text.lower()
    t = re.sub(r"[^\w\s\+\#\.-]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return f" {t.strip()} "


def _phrase_pattern(phrase: str) -> re.Pattern:
    """Whole-phrase match. The phrase is normalized the same way as the text, so 'CI/CD' matches 'CI/CD'.
    A full stop right after a skill ('...Python.') is fine, but 'node' must not match inside 'node.js'."""
    norm = normalize_text(phrase).strip()
    return re.compile(rf"(?<![\w+#-])(?<!\w\.){re.escape(norm)}(?![\w+#-])(?!\.\w)", re.IGNORECASE)


def match_skills(text: str, dict_data: dict | None = None) -> list[str]:
    """Identify skills from text using whole-phrase matching and alias mapping."""
    if not dict_data:
        dict_data = load_skills_dictionary()
    canonical_skills = dict_data.get("skills", [])
    aliases = dict_data.get("aliases", {})
    norm_text = normalize_text(text)
    found = {skill for skill in canonical_skills if _phrase_pattern(skill).search(norm_text)}
    found |= {canonical for alias, canonical in aliases.items() if _phrase_pattern(alias).search(norm_text)}
    return sorted(found)
