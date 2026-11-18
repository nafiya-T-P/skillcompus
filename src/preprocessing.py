"""Module 1 - data collection, inspection and cleaning for the job-posting dataset (every decision is logged).

Run from the project root:   python -m src.preprocessing
Reads:   data/raw/*.csv      (if several CSV files exist they are combined into one dataframe)
Writes:  data/processed/cleaned_jobs.csv, cleaning_log.csv, cleaning_log.md
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .feature_engineering import CAT_COLS, parse_experience
except ImportError:
    from feature_engineering import CAT_COLS, parse_experience

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CLEANED_PATH = PROCESSED_DIR / "cleaned_jobs.csv"
LOG_CSV_PATH = PROCESSED_DIR / "cleaning_log.csv"
LOG_MD_PATH = PROCESSED_DIR / "cleaning_log.md"

# ---- documented decisions (change here, rerun) ----
CITY_ALIASES = {"bangalore": "Bengaluru", "bengalore": "Bengaluru", "bombay": "Mumbai", "gurgaon": "Gurugram",
                "madras": "Chennai", "calcutta": "Kolkata", "poona": "Pune", "trivandrum": "Thiruvananthapuram",
                "cochin": "Kochi"}
CATEGORICAL = ["job_title", "city", "state", "experience_level", "education", "employment_type", "work_mode",
               "company_size", "industry", "data_source"]
EXP_MIN, EXP_MAX = 0.0, 30.0           # plausible experience in years; values outside are set to blank
SALARY_OUTLIER_LPA = 60.0              # salary midpoint above this is flagged as an outlier
LPA_TO_INR = 100_000
MIN_SALARY_INR, MAX_SALARY_INR = 100_000.0, 6_000_000.0   # 1 LPA to 60 LPA
NEAR_DUP_KEY = ["job_title", "company_name", "city", "posted_date", "salary_min_lpa", "salary_max_lpa", "skills_required"]

# Columns of the cleaned file that feed the salary model (standard name -> dataset column)
MODEL_COLUMNS = {"target": "salary_mid_inr_pa", "role": "job_title", "experience": "experience_years",
                 "location": "city", "skills": "skills_required", "work_mode": "work_mode", "education": "education"}


# ───────────────────────── loading ─────────────────────────
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df.columns]
    return df


def list_raw_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(f for f in raw_dir.glob("*.csv") if "comparison" not in f.name.lower())


def load_raw_data(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, list[str]]:
    """Load every CSV in data/raw and combine them (rows are concatenated, column names standardised).
    Duplicates are NOT dropped here so they can be measured and then removed in the cleaning log."""
    files = list_raw_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No CSV dataset found in '{raw_dir}'. Place the job-posting CSV in data/raw/.")
    frames = [standardize_columns(pd.read_csv(f, low_memory=False)) for f in files]
    return pd.concat(frames, ignore_index=True), [f.name for f in files]


# ───────────────────────── inspection ─────────────────────────
def _key(s: pd.Series) -> pd.Series:
    """Comparison key: trim, lowercase, hyphens/underscores/extra spaces collapsed."""
    return (s.astype("string").str.strip().str.lower().str.replace(r"[-_]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True))


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Column names, data types, non-null counts, missing share and an example value."""
    rows = []
    for c in df.columns:
        ex = df[c].dropna()
        rows.append({"column": c, "dtype": str(df[c].dtype), "non_null": int(df[c].notna().sum()),
                     "missing_%": round(df[c].isna().mean() * 100, 1),
                     "example": str(ex.iloc[0])[:40] if len(ex) else ""})
    return pd.DataFrame(rows)


def data_quality_report(raw: pd.DataFrame) -> dict:
    """Measure missing values, duplicates, inconsistent categories and invalid salaries on the RAW data."""
    miss = raw.isna().sum()
    missing = (pd.DataFrame({"column": miss.index, "missing": miss.values, "missing_%": (miss.values / max(len(raw), 1) * 100).round(1)})
               .query("missing > 0").sort_values("missing", ascending=False).reset_index(drop=True))

    cats = []
    for c in CATEGORICAL:
        if c in raw:
            vals = raw[c].dropna().astype("string").str.strip()
            cats.append({"column": c, "distinct_as_written": int(vals.nunique()),
                         "distinct_after_normalising": int(_key(vals).nunique()),
                         "spelling_variants_to_merge": int(vals.nunique() - _key(vals).nunique())})
    categories = pd.DataFrame(cats)

    salary = {}
    if {"salary_min_lpa", "salary_max_lpa"} <= set(raw.columns):
        lo = pd.to_numeric(raw["salary_min_lpa"], errors="coerce")
        hi = pd.to_numeric(raw["salary_max_lpa"], errors="coerce")
        mid = (lo + hi) / 2
        salary = {"Salary missing or one bound only": int((lo.isna() | hi.isna()).sum()),
                  "Non-positive salary": int(((lo <= 0) | (hi <= 0)).sum()),
                  "Maximum below minimum": int(((lo > 0) & (hi < lo)).sum()),
                  f"Outlier (midpoint above {SALARY_OUTLIER_LPA:g} LPA)": int((mid > SALARY_OUTLIER_LPA).sum())}
    return {"rows": len(raw), "columns": raw.shape[1], "exact_duplicates": int(raw.duplicated().sum()),
            "missing": missing, "categories": categories, "invalid_salary": salary}


# ───────────────────────── cleaning ─────────────────────────
def canonicalize(s: pd.Series, aliases: dict | None = None) -> tuple[pd.Series, int]:
    """Map every variant (case, spacing, hyphenation, aliases) to its most frequent spelling."""
    k = _key(s)
    if aliases:
        k = k.map(lambda v: _key(pd.Series([aliases[v]])).iat[0] if isinstance(v, str) and v in aliases else v)
    clean = s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    best = clean.groupby(k).agg(lambda x: x.value_counts().index[0] if x.notna().any() else pd.NA)
    if aliases:
        for std in aliases.values():
            ak = _key(pd.Series([std])).iat[0]
            if ak in best.index:
                best[ak] = std
    out = k.map(best).astype("object").where(s.notna(), np.nan)
    changed = int((out.fillna("") != s.astype("object").fillna("")).sum())
    return out, changed


def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the raw postings. Returns (cleaned dataframe, cleaning log). Invalid salaries are flagged, never silently deleted."""
    log: list[tuple[str, int, int, str]] = []
    df = standardize_columns(raw)
    log.append(("1. Load raw data", len(df), len(df), f"{df.shape[1]} columns, {int(df.isna().sum().sum()):,} missing cells"))

    notes = []
    for c in CATEGORICAL:
        if c in df:
            df[c], n = canonicalize(df[c], CITY_ALIASES if c == "city" else None)
            if n:
                notes.append(f"{c}: {n:,}")
    log.append(("2. Standardize categorical labels", len(df), len(df), "cells changed per column: " + "; ".join(notes)))

    b = len(df)
    if "job_id" in df:
        df = df.sort_values("job_id", kind="stable")
    df = df.drop_duplicates()
    log.append(("3. Remove exact duplicate rows", b, len(df), f"{b - len(df):,} rows removed"))

    b = len(df)
    key_cols = [c for c in NEAR_DUP_KEY if c in df]
    nk = df[key_cols].copy()
    for c in ("job_title", "company_name", "city", "skills_required"):
        if c in nk:
            nk[c] = _key(nk[c])
    near = nk.duplicated(keep="first")
    df = df[~near]
    log.append(("4. Remove near-duplicate postings", b, len(df),
                f"{int(near.sum()):,} removed (same {', '.join(key_cols)}; different ID/URL)"))

    if "posted_date" in df:
        d = pd.to_datetime(df["posted_date"], errors="coerce")
        failed = int(d.isna().sum() - df["posted_date"].isna().sum())
        df["posted_date"] = d
        rng = f"{d.min().date()} to {d.max().date()}" if d.notna().any() else "no valid dates"
        log.append(("5. Parse posted_date", len(df), len(df), f"{failed} unparseable; range {rng}"))

    if "experience_years" in df:
        df["experience_years"] = pd.to_numeric(df["experience_years"], errors="coerce")
        bad = (df["experience_years"] < EXP_MIN) | (df["experience_years"] > EXP_MAX)
        vals = sorted(df.loc[bad, "experience_years"].unique().tolist())
        df["experience_years_raw"] = df["experience_years"]
        df.loc[bad, "experience_years"] = np.nan
        log.append(("6. Validate experience_years", len(df), len(df),
                    f"{int(bad.sum())} implausible values (outside {EXP_MIN:g}-{EXP_MAX:g}: {vals}) set to blank; original kept in experience_years_raw"))

    lo = pd.to_numeric(df["salary_min_lpa"], errors="coerce")
    hi = pd.to_numeric(df["salary_max_lpa"], errors="coerce")
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
    log.append(("7. Validate salary and build target", len(df), int(ok.sum()),
                f"target = (min+max)/2 for rows with both bounds, min>0, max>=min, not an outlier. Valid: {int(ok.sum()):,}. "
                f"Flagged (kept, not trained on): {counts}"))
    log.append(("8. Missing values", len(df), len(df),
                "no rows dropped for missingness. At model time: experience -> median, categories -> 'Unknown', skills -> none"))

    if "posted_date" in df:
        df["posted_date"] = df["posted_date"].dt.strftime("%Y-%m-%d")
    return df, pd.DataFrame(log, columns=["step", "rows_before", "rows_after", "what_was_done"])


def save_cleaned(df: pd.DataFrame, log: pd.DataFrame, source_names: list[str]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False)
    log.to_csv(LOG_CSV_PATH, index=False)
    md = [f"# Cleaning log: {', '.join(source_names)}", "", "| Step | Rows before | Rows after | What was done |", "|---|---:|---:|---|"]
    md += [f"| {r.step} | {r.rows_before:,} | {r.rows_after:,} | {r.what_was_done} |" for r in log.itertuples()]
    LOG_MD_PATH.write_text("\n".join(md), encoding="utf-8")


def run() -> pd.DataFrame:
    raw, names = load_raw_data()
    df, log = clean_dataset(raw)
    save_cleaned(df, log, names)
    for r in log.itertuples():
        print(f"{r.step:40s} {r.rows_before:>7,} -> {r.rows_after:>7,}  {r.what_was_done[:110]}")
    return df


def ensure_cleaned() -> None:
    """Create the cleaned dataset from data/raw if it does not exist yet."""
    if not (CLEANED_PATH.exists() and LOG_CSV_PATH.exists()):
        run()


def load_cleaned() -> pd.DataFrame:
    return pd.read_csv(CLEANED_PATH, low_memory=False)


def load_cleaning_log() -> pd.DataFrame | None:
    return pd.read_csv(LOG_CSV_PATH) if LOG_CSV_PATH.exists() else None


# ───────────────────────── preparation for the salary model ─────────────────────────
def standardize_frame(df: pd.DataFrame, cols: dict = MODEL_COLUMNS) -> pd.DataFrame:
    """Rename dataset columns to the standard model names. Missing optional columns are skipped."""
    out = pd.DataFrame(index=df.index)
    for std in ("role", "experience", "location", "skills", "work_mode", "education", "target"):
        src = cols.get(std)
        if src and src in df.columns:
            out["salary" if std == "target" else std] = df[src]
    return out


def clean_for_training(frame: pd.DataFrame, min_salary: float = MIN_SALARY_INR, max_salary: float = MAX_SALARY_INR):
    """Drop invalid salaries and duplicates, apply the documented salary range, tidy categories. Returns (frame, report)."""
    f = frame.copy()
    report = {"rows_in": len(f)}
    f["salary"] = pd.to_numeric(f["salary"], errors="coerce")
    valid = f["salary"].dropna()
    if len(valid) and valid.median() < 500:      # values are in LPA -> convert to INR per year
        f["salary"] = f["salary"] * LPA_TO_INR
    f = f[f["salary"].notna() & (f["salary"] > 0)]
    report["after_valid_number"] = len(f)
    f = f[(f["salary"] >= min_salary) & (f["salary"] <= max_salary)]
    report["after_salary_range"] = len(f)

    if "experience" in f:
        f["experience"] = f["experience"].map(parse_experience)
        f.loc[(f["experience"] < 0) | (f["experience"] > 35), "experience"] = np.nan

    for c in CAT_COLS:
        if c in f:
            def _clean(v, col=c):
                if not isinstance(v, str) or not v.strip():
                    return np.nan
                s = re.sub(r"\s+", " ", v).strip()
                if col == "location" and s.lower() in CITY_ALIASES:
                    return CITY_ALIASES[s.lower()]
                return s.title() if s.islower() else s
            f[c] = f[c].astype(object).where(f[c].notna(), np.nan).map(_clean)

    before = len(f)
    f = f.drop_duplicates()
    report["duplicates_removed"] = before - len(f)
    report["rows_out"] = len(f)
    return f, report


def prepare_for_model(cleaned: pd.DataFrame):
    """Cleaned dataset -> (model-ready frame, report)."""
    return clean_for_training(standardize_frame(cleaned))


# ───────────────────────── filters (Module 5) ─────────────────────────
def apply_filters(df: pd.DataFrame, cities=None, roles=None, levels=None, modes=None) -> pd.DataFrame:
    """Keep rows matching every non-empty filter (empty list = no filter)."""
    for col, vals in (("city", cities), ("job_title", roles), ("experience_level", levels), ("work_mode", modes)):
        if vals and col in df:
            df = df[df[col].isin(vals)]
    return df


if __name__ == "__main__":
    run()