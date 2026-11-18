"""Module 1 cleaning for india_job_market_messy_31150_records.csv (reproducible, every decision logged).

Run:  python prepare_dataset.py
Out:  data/processed/cleaned_jobs.csv   (all de-duplicated rows; invalid salaries are FLAGGED, never silently deleted)
      data/processed/cleaning_log.md    (step-by-step log with before/after counts)

NOTE: the source file is a SYNTHETIC practice dataset (see data/raw/DATASET_README_AND_TASKS.txt).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/india_job_market_messy_31150_records.csv")
OUT = Path("data/processed/cleaned_jobs.csv")
LOG = Path("data/processed/cleaning_log.md")

# ---- documented decisions (change here, rerun) ----
CITY_ALIASES = {"bangalore": "Bengaluru", "bengalore": "Bengaluru", "bombay": "Mumbai", "gurgaon": "Gurugram",
                "madras": "Chennai", "calcutta": "Kolkata", "poona": "Pune", "trivandrum": "Thiruvananthapuram",
                "cochin": "Kochi"}
CATEGORICAL = ["job_title", "city", "state", "experience_level", "education", "employment_type", "work_mode",
               "company_size", "industry", "data_source"]
EXP_MIN, EXP_MAX = 0.0, 30.0          # plausible experience (years); data shows junk values -2, 35, 50, 99
SALARY_OUTLIER_LPA = 60.0              # midpoint above this is flagged as an outlier (99.9th percentile is ~72 LPA)
LPA_TO_INR = 100_000
NEAR_DUP_KEY = ["job_title", "company_name", "city", "posted_date", "salary_min_lpa", "salary_max_lpa", "skills_required"]


def _key(s: pd.Series) -> pd.Series:
    """Comparison key: trim, lowercase, hyphens/underscores/extra spaces collapsed."""
    return (s.astype("string").str.strip().str.lower().str.replace(r"[-_]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True))


def canonicalize(s: pd.Series, aliases: dict | None = None) -> tuple[pd.Series, int]:
    """Map every variant (case, spacing, hyphenation, aliases) to its most frequent spelling."""
    k = _key(s)
    if aliases:
        k = k.map(lambda v: _key(pd.Series([aliases[v]])).iat[0] if isinstance(v, str) and v in aliases else v)
    clean = s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    best = clean.groupby(k).agg(lambda x: x.value_counts().index[0] if x.notna().any() else pd.NA)
    if aliases:
        for alias, std in aliases.items():
            ak = _key(pd.Series([std])).iat[0]
            if ak in best.index:
                best[ak] = std
    out = k.map(best).astype("object").where(s.notna(), np.nan)
    changed = int((out.fillna("") != s.astype("object").fillna("")).sum())
    return out, changed


def run() -> pd.DataFrame:
    log: list[tuple[str, int, int, str]] = []
    # If there is more than one CSV file in the data folder, combine them into a single dataframe
    raw_dir = RAW.parent if RAW.parent.exists() else Path("data")
    csv_files = sorted([f for f in raw_dir.glob("*.csv") if "comparison" not in f.name.lower()])
    if not csv_files and RAW.exists():
        csv_files = [RAW]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in relative path '{raw_dir}'")

    dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    source_label = ", ".join(f.name for f in csv_files)
    log.append(("1. Load raw file", len(df), len(df), f"from {source_label}; {df.shape[1]} columns, "
                f"{int(df.isna().sum().sum()):,} missing cells"))

    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]
    log.append(("2. Standardize column names", len(df), len(df), "trimmed, lowercase, snake_case (already compliant)"))

    # 3. categorical labels
    notes = []
    for c in CATEGORICAL:
        df[c], n = canonicalize(df[c], CITY_ALIASES if c == "city" else None)
        if n:
            notes.append(f"{c}: {n:,}")
    log.append(("3. Standardize categorical labels", len(df), len(df), "cells changed per column: " + "; ".join(notes)))

    # 4. exact duplicates (keep lowest job_id so original postings win over injected variants)
    b = len(df)
    df = df.sort_values("job_id", kind="stable")
    df = df.drop_duplicates()
    log.append(("4. Remove exact duplicate rows", b, len(df), f"{b - len(df):,} rows removed"))

    # 5. near duplicates: same title/company/city/date/salary/skills (IDs and URLs differ)
    b = len(df)
    nk = df[NEAR_DUP_KEY].copy()
    for c in ["job_title", "company_name", "city", "skills_required"]:
        nk[c] = _key(nk[c])
    near = nk.duplicated(keep="first")
    loose = int(_key(df["job_title"]).to_frame().assign(co=_key(df["company_name"]), ci=_key(df["city"]),
                                                       d=df["posted_date"]).duplicated().sum())
    df = df[~near]
    log.append(("5. Remove near-duplicate postings", b, len(df),
                f"{int(near.sum()):,} removed (same title, company, city, date, salary and skills; different ID/URL). "
                f"A looser match (title+company+city+date) flags {loose:,} rows; those were kept because they may be "
                "genuine re-postings"))

    # 6. dates
    d = pd.to_datetime(df["posted_date"], errors="coerce")
    failed = int(d.isna().sum() - df["posted_date"].isna().sum())
    df["posted_date"] = d
    log.append(("6. Parse posted_date", len(df), len(df),
                f"{failed} unparseable; range {d.min().date()} to {d.max().date()}; {int(d.isna().sum()):,} missing kept as blank"))

    # 7. experience
    bad = (df["experience_years"] < EXP_MIN) | (df["experience_years"] > EXP_MAX)
    vals = sorted(df.loc[bad, "experience_years"].unique().tolist())
    df["experience_years_raw"] = df["experience_years"]
    df.loc[bad, "experience_years"] = np.nan
    log.append(("7. Validate experience_years", len(df), len(df),
                f"{int(bad.sum())} implausible values (outside {EXP_MIN:g}-{EXP_MAX:g}: {vals}) set to blank; original kept in "
                "experience_years_raw; model imputes the median"))

    # 8. salary
    lo, hi = df["salary_min_lpa"], df["salary_max_lpa"]
    mid = (lo + hi) / 2
    issue = pd.Series("", index=df.index, dtype="object")
    issue[hi.isna() | lo.isna()] = "salary missing or one bound only"
    issue[(lo <= 0) | (hi <= 0)] = "non-positive salary"
    issue[(lo > 0) & (hi < lo)] = "max below min"
    issue[(issue == "") & (mid > SALARY_OUTLIER_LPA)] = f"outlier (mid > {SALARY_OUTLIER_LPA:g} LPA)"
    df["salary_issue"] = issue
    ok = issue == ""
    df["salary_mid_lpa"] = mid.where(ok)
    df["salary_mid_inr_pa"] = (mid * LPA_TO_INR).where(ok)
    counts = issue[issue != ""].value_counts().to_dict()
    log.append(("8. Validate salary & build target", len(df), int(ok.sum()),
                f"target salary_mid_lpa = (min+max)/2 for rows with both bounds, min>0, max>=min, not an outlier. "
                f"Valid: {int(ok.sum()):,}. Flagged (kept, not trained on): {counts}"))

    log.append(("9. Missing values", len(df), len(df),
                "no rows dropped for missingness. At model time: experience -> median, categories -> 'Unknown', "
                "skills -> no skills. Salary is the only field that removes rows from training"))

    df["posted_date"] = df["posted_date"].dt.strftime("%Y-%m-%d")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    log.append(("10. Save", len(df), len(df), f"{OUT} ({len(df):,} rows, {df.shape[1]} columns)"))

    md = ["# Cleaning log: india_job_market_messy_31150_records.csv", "",
          "**Synthetic practice dataset, not real market data.** Every decision below is reproducible with `python prepare_dataset.py`.", "",
          "| Step | Rows before | Rows after | What was done |", "|---|---:|---:|---|"]
    md += [f"| {a} | {b:,} | {c:,} | {d} |" for a, b, c, d in log]
    md += ["", f"Usable salary rows for modelling: **{int(ok.sum()):,}** of {len(df):,} "
           f"({ok.mean():.1%}). The project guideline is 20,000+, so treat salary aggregates as indicative."]
    LOG.write_text("\n".join(md), encoding="utf-8")
    for a, b, c, d in log:
        print(f"{a:38s} {b:>7,} -> {c:>7,}  {d[:120]}")
    return df


if __name__ == "__main__":
    run()
