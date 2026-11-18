"""Module 1 Exploratory Data Analysis (EDA) Script.

Generates:
- reports/figures/*.png (visualizations of salary, demand, roles, etc.)
- reports/eda_report.md (comprehensive markdown EDA report)
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "india_job_market_messy_31150_records.csv"
CLEANED_PATH = ROOT / "data" / "processed" / "cleaned_jobs.csv"
REPORT_PATH = ROOT / "reports" / "eda_report.md"
FIGURES_DIR = ROOT / "reports" / "figures"

# Styling constants
sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 10})


def run_eda():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    raw_df = pd.read_csv(RAW_PATH, low_memory=False)
    clean_df = pd.read_csv(CLEANED_PATH, low_memory=False)

    print("1. Computing summary metrics...")
    raw_shape = raw_df.shape
    clean_shape = clean_df.shape

    # Missing values in clean df
    clean_missing = clean_df.isna().sum()
    clean_missing_pct = (clean_missing / len(clean_df) * 100).round(2)
    missing_summary = pd.DataFrame({
        "Column": clean_df.columns,
        "Missing_Count": clean_missing.values,
        "Missing_Percent": clean_missing_pct.values
    }).sort_values(by="Missing_Count", ascending=False)

    # Duplicates
    raw_exact_dups = raw_df.duplicated().sum()
    clean_exact_dups = clean_df.duplicated().sum()

    # Salaries
    valid_salaries = clean_df["salary_mid_lpa"].dropna()
    salary_stats = valid_salaries.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2)

    # Top roles, cities, modes, levels
    top_roles = clean_df["job_title"].value_counts().head(10)
    top_cities = clean_df["city"].value_counts().head(10)
    work_modes = clean_df["work_mode"].value_counts()
    exp_levels = clean_df["experience_level"].value_counts()

    # Invalid salary issues breakdown
    salary_issues = clean_df["salary_issue"].fillna("").value_counts()
    salary_issues = salary_issues[salary_issues.index != ""]

    print("2. Generating figures...")
    # Figure 1: Salary Distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(valid_salaries, kde=True, bins=35, color="#2dd4bf", edgecolor="#0b1226")
    plt.axvline(valid_salaries.median(), color="#fbbf24", linestyle="--", linewidth=2, label=f"Median: {valid_salaries.median():.1f} LPA")
    plt.axvline(valid_salaries.mean(), color="#ef4444", linestyle=":", linewidth=2, label=f"Mean: {valid_salaries.mean():.1f} LPA")
    plt.title("Distribution of Disclosed Annual Salaries (LPA)", fontsize=13, weight="bold")
    plt.xlabel("Midpoint Salary (LPA)")
    plt.ylabel("Number of Postings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "salary_distribution.png", dpi=200)
    plt.close()

    # Figure 2: Top Roles
    plt.figure(figsize=(10, 5.5))
    sns.barplot(x=top_roles.values, y=top_roles.index, palette="viridis")
    plt.title("Top 10 Most Demanded Job Roles", fontsize=13, weight="bold")
    plt.xlabel("Number of Job Postings")
    plt.ylabel("Job Role")
    for idx, v in enumerate(top_roles.values):
        plt.text(v + 10, idx, f"{v:,}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_roles.png", dpi=200)
    plt.close()

    # Figure 3: City Demand
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_cities.values, y=top_cities.index, palette="mako")
    plt.title("Top 10 Job Posting Cities", fontsize=13, weight="bold")
    plt.xlabel("Number of Job Postings")
    plt.ylabel("City")
    for idx, v in enumerate(top_cities.values):
        plt.text(v + 10, idx, f"{v:,}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "city_demand.png", dpi=200)
    plt.close()

    # Figure 4: Experience Levels & Work Modes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(ax=axes[0], x=exp_levels.index, y=exp_levels.values, palette="Blues_r")
    axes[0].set_title("Postings by Experience Level", weight="bold")
    axes[0].set_ylabel("Postings")
    axes[0].tick_params(axis="x", rotation=25)

    sns.barplot(ax=axes[1], x=work_modes.index, y=work_modes.values, palette="Greens_r")
    axes[1].set_title("Postings by Work Mode", weight="bold")
    axes[1].set_ylabel("Postings")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "experience_and_workmode.png", dpi=200)
    plt.close()

    # Figure 5: Salary by Role (min 30 rows)
    role_salary = clean_df.dropna(subset=["salary_mid_lpa"]).groupby("job_title")["salary_mid_lpa"].agg(["count", "median"])
    role_salary = role_salary[role_salary["count"] >= 30].sort_values("median", ascending=False).head(10)

    plt.figure(figsize=(10, 5.5))
    sns.barplot(x=role_salary["median"].values, y=role_salary.index, palette="rocket")
    plt.title("Median Salary (LPA) by Top Roles (Min 30 Postings)", fontsize=13, weight="bold")
    plt.xlabel("Median Annual Salary (LPA)")
    plt.ylabel("Job Role")
    for idx, (role, row) in enumerate(role_salary.iterrows()):
        plt.text(row["median"] + 0.2, idx, f"{row['median']:.1f} LPA (n={int(row['count'])})", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "salary_by_role.png", dpi=200)
    plt.close()

    print("3. Generating markdown report...")
    sample_rows = clean_df.head(5)[["job_title", "city", "experience_level", "work_mode", "salary_mid_lpa", "skills_required"]]

    date_min = clean_df["posted_date"].dropna().min()
    date_max = clean_df["posted_date"].dropna().max()

    report_lines = [
        "# Exploratory Data Analysis (EDA) Report",
        "",
        "> **Note on Dataset Origin:** This report analyzes `india_job_market_messy_31150_records.csv`, which is a **synthetic practice dataset** generated for educational and benchmarking purposes. Figures represent patterns in the synthetic job postings and are indicative rather than actual live employer payroll data.",
        "",
        "## 1. Dataset Dimensions & Overview",
        f"- **Raw Record Count:** {raw_shape[0]:,} rows, {raw_shape[1]} columns",
        f"- **Cleaned Record Count:** {clean_shape[0]:,} rows, {clean_shape[1]} columns",
        f"- **Date Range:** {date_min} to {date_max}",
        f"- **Valid Disclosed Salary Records:** {len(valid_salaries):,} ({len(valid_salaries)/len(clean_df)*100:.1f}% of clean postings)",
        "",
        "### Sample Records (Cleaned)",
        sample_rows.to_markdown(index=False),
        "",
        "## 2. Data Cleaning & Integrity Improvements",
        f"- **Exact Duplicates Removed:** {raw_exact_dups:,} duplicate rows removed.",
        "- **Near-Duplicate Postings:** Removed 250 near-duplicate postings sharing identical role, company, city, date, salary, and skills.",
        "- **Standardized Categorical Columns:** Cleaned spelling, trailing whitespace, and case variants across `job_title`, `city`, `work_mode`, and `experience_level`.",
        "- **City Canonicalization:** Mapped historical and variant city spellings (e.g. Bangalore -> Bengaluru, Bombay -> Mumbai, Gurgaon -> Gurugram, Poona -> Pune, Madras -> Chennai).",
        "- **Implausible Experience:** 40 implausible values (negative values or > 30 years) isolated to `experience_years_raw` and blanked for standard modeling.",
        "",
        "## 3. Salary Target & Validity Breakdown",
        "Disclosed salaries are evaluated for bounds integrity. The model uses the midpoint `(salary_min_lpa + salary_max_lpa) / 2` in INR per annum.",
        "",
        f"- **Valid Usable Salary Rows:** {len(valid_salaries):,}",
        f"- **Flagged Non-Trained Rows:** {len(clean_df) - len(valid_salaries):,}",
        "",
        "### Reasons for Flagging Non-Usable Salary Rows:",
    ]

    for issue, cnt in salary_issues.items():
        report_lines.append(f"- **{issue}:** {cnt:,} postings ({cnt / len(clean_df) * 100:.1f}%)")

    report_lines.extend([
        "",
        "### Summary Statistics for Valid Salaries (LPA):",
        f"- **Minimum:** ₹{salary_stats['min']:.2f} L",
        f"- **10th Percentile:** ₹{salary_stats['10%']:.2f} L",
        f"- **25th Percentile:** ₹{salary_stats['25%']:.2f} L",
        f"- **Median (50th):** ₹{salary_stats['50%']:.2f} L",
        f"- **Mean:** ₹{salary_stats['mean']:.2f} L",
        f"- **75th Percentile:** ₹{salary_stats['75%']:.2f} L",
        f"- **90th Percentile:** ₹{salary_stats['90%']:.2f} L",
        f"- **Maximum (capped):** ₹{salary_stats['max']:.2f} L",
        f"- **Standard Deviation:** ₹{salary_stats['std']:.2f} L",
        "",
        "## 4. Market Distributions & Demand",
        "",
        "### Top 10 Job Roles",
        top_roles.reset_index().rename(columns={"index": "Role", "job_title": "Postings"}).to_markdown(index=False),
        "",
        "### Top 10 Cities",
        top_cities.reset_index().rename(columns={"index": "City", "city": "Postings"}).to_markdown(index=False),
        "",
        "### Work Modes",
        work_modes.reset_index().rename(columns={"index": "Work Mode", "work_mode": "Postings"}).to_markdown(index=False),
        "",
        "### Experience Levels",
        exp_levels.reset_index().rename(columns={"index": "Experience Level", "experience_level": "Postings"}).to_markdown(index=False),
        "",
        "## 5. Visualizations Saved",
        "- `reports/figures/salary_distribution.png`: Histogram & KDE of annual salaries.",
        "- `reports/figures/top_roles.png`: Top 10 most demanded job titles.",
        "- `reports/figures/city_demand.png`: Regional distribution of job postings.",
        "- `reports/figures/experience_and_workmode.png`: Distribution of experience tiers and work modes.",
        "- `reports/figures/salary_by_role.png`: Median salary comparison across roles with n >= 30 postings.",
        "",
        "---",
        "*Report auto-generated by `eda.py`.*"
    ])

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"✓ EDA completed! Report saved to {REPORT_PATH}")
    print(f"✓ Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    run_eda()
