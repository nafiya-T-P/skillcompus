# Exploratory Data Analysis (EDA) Report

> **Note on Dataset Origin:** This report analyzes `india_job_market_messy_31150_records.csv`, which is a **synthetic practice dataset** generated for educational and benchmarking purposes. Figures represent patterns in the synthetic job postings and are indicative rather than actual live employer payroll data.

## 1. Dataset Dimensions & Overview
- **Raw Record Count:** 31,150 rows, 20 columns
- **Cleaned Record Count:** 30,000 rows, 24 columns
- **Date Range:** 2024-01-01 to 2026-10-31
- **Valid Disclosed Salary Records:** 17,378 (57.9% of clean postings)

### Sample Records (Cleaned)
| job_title                 | city      | experience_level   | work_mode   |   salary_mid_lpa | skills_required                                                               |
|:--------------------------|:----------|:-------------------|:------------|-----------------:|:------------------------------------------------------------------------------|
| Software Engineer         | Hyderabad | Fresher            | On-site     |          nan     | Spark, Kubernetes, TensorFlow, NumPy, Pandas                                  |
| NLP Engineer              | Chennai   | Senior             | On-site     |           21.68  | Statistics, Scikit-learn, Kubernetes, FastAPI, TensorFlow, Linux, Python, Git |
| Data Scientist            | Hyderabad | Mid-level          | On-site     |           15.35  | JavaScript, Hadoop, Excel, NLP, Scikit-learn, Python                          |
| Cloud Engineer            | Chennai   | Mid-level          | On-site     |           14.93  | Kubernetes, FastAPI, Pandas                                                   |
| Machine Learning Engineer | Delhi NCR | Junior             | On-site     |           14.065 | Airflow, JavaScript, NLP, Java, Flask, Excel, Django, Machine Learning        |

## 2. Data Cleaning & Integrity Improvements
- **Exact Duplicates Removed:** 900 duplicate rows removed.
- **Near-Duplicate Postings:** Removed 250 near-duplicate postings sharing identical role, company, city, date, salary, and skills.
- **Standardized Categorical Columns:** Cleaned spelling, trailing whitespace, and case variants across `job_title`, `city`, `work_mode`, and `experience_level`.
- **City Canonicalization:** Mapped historical and variant city spellings (e.g. Bangalore -> Bengaluru, Bombay -> Mumbai, Gurgaon -> Gurugram, Poona -> Pune, Madras -> Chennai).
- **Implausible Experience:** 40 implausible values (negative values or > 30 years) isolated to `experience_years_raw` and blanked for standard modeling.

## 3. Salary Target & Validity Breakdown
Disclosed salaries are evaluated for bounds integrity. The model uses the midpoint `(salary_min_lpa + salary_max_lpa) / 2` in INR per annum.

- **Valid Usable Salary Rows:** 17,378
- **Flagged Non-Trained Rows:** 12,622

### Reasons for Flagging Non-Usable Salary Rows:
- **salary missing or one bound only:** 12,531 postings (41.8%)
- **non-positive salary:** 35 postings (0.1%)
- **outlier (mid > 60 LPA):** 31 postings (0.1%)
- **max below min:** 25 postings (0.1%)

### Summary Statistics for Valid Salaries (LPA):
- **Minimum:** ₹0.68 L
- **10th Percentile:** ₹5.43 L
- **25th Percentile:** ₹7.46 L
- **Median (50th):** ₹10.64 L
- **Mean:** ₹12.13 L
- **75th Percentile:** ₹15.21 L
- **90th Percentile:** ₹20.76 L
- **Maximum (capped):** ₹59.74 L
- **Standard Deviation:** ₹6.58 L

## 4. Market Distributions & Demand

### Top 10 Job Roles
| Postings                  |   count |
|:--------------------------|--------:|
| Business Analyst          |    2586 |
| BI Developer              |    2570 |
| Cloud Engineer            |    2540 |
| Data Engineer             |    2520 |
| Software Engineer         |    2512 |
| Python Developer          |    2484 |
| MLOps Engineer            |    2479 |
| NLP Engineer              |    2471 |
| Data Scientist            |    2467 |
| Machine Learning Engineer |    2465 |

### Top 10 Cities
| Postings   |   count |
|:-----------|--------:|
| Bengaluru  |    2097 |
| Delhi NCR  |    1999 |
| Hyderabad  |    1998 |
| Kochi      |    1994 |
| Ahmedabad  |    1992 |
| Kolkata    |    1979 |
| Chennai    |    1973 |
| Coimbatore |    1970 |
| Mumbai     |    1960 |
| Indore     |    1955 |

### Work Modes
| Postings   |   count |
|:-----------|--------:|
| On-site    |   12676 |
| Hybrid     |   10460 |
| Remote     |    6450 |

### Experience Levels
| Postings   |   count |
|:-----------|--------:|
| Mid-level  |    9987 |
| Junior     |    7437 |
| Senior     |    6361 |
| Fresher    |    3572 |
| Lead       |    2314 |

## 5. Visualizations Saved
- `reports/figures/salary_distribution.png`: Histogram & KDE of annual salaries.
- `reports/figures/top_roles.png`: Top 10 most demanded job titles.
- `reports/figures/city_demand.png`: Regional distribution of job postings.
- `reports/figures/experience_and_workmode.png`: Distribution of experience tiers and work modes.
- `reports/figures/salary_by_role.png`: Median salary comparison across roles with n >= 30 postings.

---
*Report auto-generated by `eda.py`.*