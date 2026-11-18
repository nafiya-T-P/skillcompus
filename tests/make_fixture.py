"""Synthetic messy job-posting data with the same columns as the real dataset (for tests only, never used by the app)."""
import numpy as np
import pandas as pd


def make_fixture(n: int = 2500, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    roles = {"Data Analyst": 6.5, "Data Scientist": 11, "ML Engineer": 14, "Python Developer": 8,
             "Full Stack Developer": 9.5, "Backend Engineer": 10, "DevOps Engineer": 11}
    cities = {"Bengaluru": 1.15, "Hyderabad": 1.05, "Pune": 1.0, "Mumbai": 1.05, "Delhi": 1.03, "Chennai": 0.95, "Kochi": 0.9}
    skills = ["python", "sql", "machine learning", "deep learning", "aws", "docker", "excel", "power bi", "react",
              "java", "spark", "git", "tableau", "kubernetes"]
    levels = ["Fresher", "Junior", "Mid", "Senior"]
    role = rng.choice(list(roles), n)
    city = rng.choice(list(cities), n)
    exp = np.clip(rng.gamma(2.0, 2.2, n), 0, 20).round(1)
    sk = [", ".join(rng.choice(skills, rng.integers(2, 7), replace=False)) for _ in range(n)]
    sal = np.array([roles[r] for r in role]) * (1 + 0.12 * exp) * np.array([cities[c] for c in city]) * rng.lognormal(0, 0.15, n)
    df = pd.DataFrame({
        "job_id": np.arange(n), "job_title": role, "company_name": rng.choice([f"Co{i}" for i in range(300)], n),
        "city": city, "state": "State", "experience_level": pd.cut(exp, [-1, 1, 4, 8, 40], labels=levels).astype(str),
        "education": rng.choice(["Bachelor's", "Master's", "PhD", "Diploma"], n, p=[.6, .3, .03, .07]),
        "employment_type": "Full-time", "work_mode": rng.choice(["On-site", "Hybrid", "Remote"], n, p=[.5, .3, .2]),
        "company_size": "Mid", "industry": "IT", "data_source": "synthetic",
        "posted_date": (pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")).strftime("%Y-%m-%d"),
        "salary_min_lpa": (sal * 0.9).round(1), "salary_max_lpa": (sal * 1.1).round(1),
        "skills_required": sk, "experience_years": exp, "job_url": [f"http://x/{i}" for i in range(n)]})
    # make it messy on purpose
    df.loc[rng.choice(n, 60, replace=False), "city"] = "bangalore"
    df.loc[rng.choice(n, 40, replace=False), "job_title"] = df["job_title"].str.lower()
    df.loc[rng.choice(n, 30, replace=False), "salary_min_lpa"] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "salary_min_lpa"] = -1
    df.loc[rng.choice(n, 10, replace=False), "experience_years"] = 99
    df.loc[rng.choice(n, 25, replace=False), "education"] = np.nan
    return pd.concat([df, df.sample(50, random_state=1)], ignore_index=True)   # 50 exact duplicates