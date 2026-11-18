"""Module 4: Role-based skill gap analysis.

Provides:
- get_role_skill_reference(role: str) -> dict
- compute_skill_gap(skills, ref_skills, role, learning_suggestions) -> dict
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from . import preprocessing as prep
from .resume_parser import load_skills_dictionary


def get_role_skill_reference(role: str, min_postings: int = 10, top_n: int = 10) -> dict[str, Any]:
    """Calculate the top N reference skills from the dataset for the selected role."""
    prep.ensure_cleaned()
    df = prep.load_cleaned()

    if "job_title" not in df.columns or "skills_required" not in df.columns:
        return {
            "role": role,
            "n_postings": 0,
            "sufficient_data": False,
            "skills": [],
            "message": "Required columns not found in cleaned dataset.",
        }

    role_df = df[df["job_title"].astype(str).str.strip().str.lower() == role.strip().lower()]
    n_postings = len(role_df)

    if n_postings < min_postings:
        return {
            "role": role,
            "n_postings": n_postings,
            "sufficient_data": False,
            "skills": [],
            "message": f"Not enough job postings for “{role}” to build a reliable skill reference ({n_postings} found, {min_postings} needed).",
        }

    counter = Counter()
    for row in role_df["skills_required"].dropna():
        for item in str(row).split(","):
            s = item.strip()
            if s:
                counter[s] += 1

    top_skills = [skill for skill, _ in counter.most_common(top_n)]
    return {
        "role": role,
        "n_postings": n_postings,
        "sufficient_data": True,
        "skills": top_skills,
        "message": f"Skill reference built from {n_postings:,} postings for '{role}'.",
    }


def compute_skill_gap(skills: list[str], ref_skills: list[str],
                      role: str, learning_suggestions: dict | None = None) -> dict[str, Any]:
    """Compare candidate's reviewed skills against target role reference skills."""
    if learning_suggestions is None:
        dict_data = load_skills_dictionary()
        learning_suggestions = dict_data.get("learning_suggestions", {})

    cand_set = {s.strip().lower(): s for s in skills}

    matched = []
    missing = []
    for ref in ref_skills:
        if ref.strip().lower() in cand_set:
            matched.append(ref)
        else:
            missing.append(ref)

    total_ref = len(ref_skills)
    score = round((len(matched) / max(total_ref, 1)) * 100) if total_ref > 0 else 0
    formula = f"{len(matched)} matched ÷ {total_ref} top skills for {role} = {score}% match"

    suggestions = []
    for s in missing:
        sug_text = learning_suggestions.get(s, f"Build hands-on projects and review best practices in {s}.")
        suggestions.append({"skill": s, "suggestion": sug_text})

    return {
        "role": role,
        "score": score,
        "formula": formula,
        "matched": matched,
        "missing": missing,
        "learning_suggestions": suggestions,
        "disclaimer": "Learning suggestions based on job-posting patterns, not hiring guarantees.",
    }
