"""Tests for Module 2: Salary prediction pipeline and transformers."""
import numpy as np
import pandas as pd
import pytest

import salary_model as sm

COLS = {
    "target": "salary_mid_inr_pa",
    "role": "job_title",
    "experience": "experience_years",
    "location": "city",
    "skills": "skills_required",
    "work_mode": "work_mode",
    "education": "education",
    "date": "posted_date",
}


def test_experience_feature_extractor():
    ext = sm.ExperienceFeatureExtractor()
    df = pd.DataFrame({"exp": ["3.5", "2-5 years", "Fresher", 10.0, np.nan]})
    res = ext.fit_transform(df)
    assert res.shape == (5, 2)
    assert res[0, 0] == 3.5
    assert res[1, 0] == 3.5
    assert res[2, 0] == 0.0
    assert res[3, 0] == 10.0
    assert np.isnan(res[4, 0])


def test_skill_binarizer():
    sb = sm.SkillBinarizer(top_n=5, min_count=1)
    df = pd.DataFrame({"skills": ["Python, SQL", "Python, AWS", "Docker"]})
    sb.fit(df)
    assert "python" in sb.vocabulary_
    trans = sb.transform(pd.DataFrame({"skills": ["python, docker, unfamiliar_tool"]}))
    assert trans.shape[1] == len(sb.vocabulary_) + 1
    # total skill count is the last column
    assert trans[0, -1] == 3.0


def test_clean_for_training():
    df = pd.DataFrame({
        "salary": [500000.0, -100.0, np.nan, 800000.0, 100000000.0],
        "role": ["Data Scientist", "Data Engineer", "Analyst", "Developer", "Manager"]
    })
    cleaned, rep = sm.clean_for_training(df, min_salary=100000.0, max_salary=6000000.0)
    assert rep["rows_out"] == 2
    assert (cleaned["salary"] >= 100000.0).all()
    assert (cleaned["salary"] <= 6000000.0).all()


def test_train_and_predict_dummy():
    # Mini synthetic test
    roles = ["Data Scientist", "Software Engineer"] * 25
    exps = [2.0, 5.0] * 25
    locs = ["Bengaluru", "Mumbai"] * 25
    skills = ["Python, SQL", "Java, Docker"] * 25
    modes = ["Hybrid", "Remote"] * 25
    edus = ["Bachelor's", "Master's"] * 25
    salaries = [800000.0, 1200000.0] * 25

    df = pd.DataFrame({
        "salary": salaries,
        "role": roles,
        "experience": exps,
        "location": locs,
        "skills": skills,
        "work_mode": modes,
        "education": edus,
    })

    feats = ["role", "experience", "location", "skills", "work_mode", "education"]
    bundle = sm.train_and_select(df, feats, folds=2, top_skills=5, seed=42)

    assert "model" in bundle
    assert bundle["best_name"] in bundle["results"]["model"].values

    pred = sm.predict_salary(
        bundle,
        role="Data Scientist",
        experience=3.0,
        location="Bengaluru",
        skills="Python, SQL",
        work_mode="Hybrid",
        education="Master's"
    )
    assert pred["point"] > 0
    assert pred["low"] <= pred["point"] <= pred["high"]
