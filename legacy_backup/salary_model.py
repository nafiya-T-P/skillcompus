"""Module 2: Salary Prediction Model & Pipelines.

Standard feature names used throughout the pipeline:
    - role (categorical)
    - experience (numeric or range string, in years)
    - location (categorical city)
    - skills (comma-separated text or list)
    - work_mode (categorical)
    - education (categorical)

Target: salary_mid_inr_pa (INR per annum).

Custom transformers (ExperienceFeatureExtractor, SkillBinarizer) are defined
at module level so joblib serializes them under 'salary_model.*'.
Training must only be run via train.py (`import salary_model as sm`).
"""
from __future__ import annotations

import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

MODEL_PATH = "models/salary_pipeline.joblib"
CAT_COLS = ["role", "location", "work_mode", "education"]
EXP_MIN, EXP_MAX = 0.0, 30.0

LIMITATIONS = [
    "Estimates are learned from salaries disclosed in job postings, not verified employee payroll.",
    "Postings that disclose salary may differ systematically from those that do not (disclosure bias).",
    "The range is an empirical band (80% validation coverage), not a formal statistical confidence interval.",
    "Rare roles, cities, or unusual skill combinations have limited data, resulting in wider estimation error.",
    "The model excludes sensitive personal attributes (gender, age, ethnicity) and cannot evaluate individual negotiation skill.",
    "This dataset is synthetic practice data; estimates should be treated as indicative illustrations rather than binding market guarantees.",
]


def parse_experience(v) -> float | None:
    """Parse experience from number or string into a single numeric year value."""
    if v is None:
        return None
    if isinstance(v, (int, float, np.number)):
        return float(v) if not np.isnan(v) else None
    s = str(v).strip().lower()
    if any(k in s for k in ("fresher", "entry", "intern", "trainee")):
        return 0.0
    m = re.search(r"(\d+\.?\d*)\s*(?:-|to)\s*(\d+\.?\d*)", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    m = re.search(r"(\d+\.?\d*)", s)
    if m:
        return float(m.group(1))
    return None


class ExperienceFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom module-level transformer that parses experience into [years, years^2]."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            vals = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            vals = X
        else:
            vals = pd.Series(np.asarray(X).ravel())

        parsed = vals.map(parse_experience).astype(float)
        # Cap implausible experience values
        parsed = parsed.where((parsed >= EXP_MIN) & (parsed <= EXP_MAX), np.nan)
        yrs = parsed.to_numpy().reshape(-1, 1)
        sq = (parsed ** 2).to_numpy().reshape(-1, 1)
        return np.hstack([yrs, sq])

    def get_feature_names_out(self, input_features=None):
        return np.array(["experience_years", "experience_years_sq"])


class SkillBinarizer(BaseEstimator, TransformerMixin):
    """Custom module-level transformer: multi-hot encodes top N skills + total skill count."""

    def __init__(self, top_n=40, min_count=5):
        self.top_n = top_n
        self.min_count = min_count
        self.vocabulary_: list[str] = []

    @staticmethod
    def tokens(v) -> set[str]:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return set()
        if isinstance(v, (list, set, tuple)):
            return {str(x).strip().lower() for x in v if str(x).strip()}
        return {s.strip().lower() for s in str(v).split(",") if s.strip()}

    def _col(self, X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0]
        if isinstance(X, pd.Series):
            return X
        return pd.Series(np.asarray(X).ravel())

    def fit(self, X, y=None):
        counts = Counter()
        for val in self._col(X):
            counts.update(self.tokens(val))
        common = [k for k, c in counts.most_common(self.top_n) if c >= self.min_count]
        self.vocabulary_ = sorted(common)
        return self

    def transform(self, X):
        col = self._col(X)
        vocab_map = {sk: idx for idx, sk in enumerate(self.vocabulary_)}
        n_features = len(self.vocabulary_) + 1  # top skills + total count
        out = np.zeros((len(col), n_features), dtype=np.float32)

        for i, val in enumerate(col):
            toks = self.tokens(val)
            for t in toks:
                if t in vocab_map:
                    out[i, vocab_map[t]] = 1.0
            out[i, -1] = float(len(toks))  # count of all listed skills
        return out

    def get_feature_names_out(self, input_features=None):
        names = [f"skill_{s}" for s in self.vocabulary_] + ["skill_count_total"]
        return np.array(names)


def build_preprocessor(features: list[str], top_skills=40) -> ColumnTransformer:
    """Construct standard ColumnTransformer for numeric, categorical, and multi-hot skill features."""
    transformers = []
    if "experience" in features:
        num_pipe = Pipeline([
            ("exp_ext", ExperienceFeatureExtractor()),
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, ["experience"]))

    cats = [c for c in CAT_COLS if c in features]
    if cats:
        cat_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, cats))

    if "skills" in features:
        skill_pipe = Pipeline([
            ("bin", SkillBinarizer(top_n=top_skills)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("skills", skill_pipe, ["skills"]))

    return ColumnTransformer(transformers, remainder="drop")


def candidate_models(seed=42) -> dict:
    """Return dictionary of diverse candidate regressors with fixed random seeds."""
    models = {
        "Ridge": Ridge(alpha=10.0, random_state=seed),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=seed),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.85, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=6, random_state=seed),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=250, learning_rate=0.04, max_depth=4, subsample=0.85,
                                         colsample_bytree=0.85, random_state=seed, n_jobs=-1)
    except Exception:
        pass
    return models


def make_pipeline(features: list[str], estimator, top_skills=40):
    """Build log-target regressor pipeline."""
    inner = Pipeline([
        ("prep", build_preprocessor(features, top_skills=top_skills)),
        ("reg", estimator),
    ])
    return TransformedTargetRegressor(regressor=inner, func=np.log1p, inverse_func=np.expm1)


def standardize_frame(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """Map arbitrary dataset columns to standard pipeline feature names."""
    out = pd.DataFrame(index=df.index)
    for std_name, raw_name in cols.items():
        if raw_name in df.columns:
            out[std_name] = df[raw_name]
    return out


def clean_for_training(frame: pd.DataFrame, min_salary=100_000.0, max_salary=6_000_000.0) -> tuple[pd.DataFrame, dict]:
    """Filter to rows with valid positive salary within specified bounds."""
    f = frame.copy()
    if "salary" not in f.columns and "target" in f.columns:
        f["salary"] = f["target"]

    rows_in = len(f)
    f["salary"] = pd.to_numeric(f["salary"], errors="coerce")
    valid = f["salary"].notna() & (f["salary"] >= min_salary) & (f["salary"] <= max_salary)
    f = f[valid].copy()

    for c in CAT_COLS:
        if c in f.columns:
            f[c] = f[c].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})

    rep = {
        "rows_in": rows_in,
        "rows_out": len(f),
        "dropped_salary": rows_in - len(f),
        "salary_range": (float(f["salary"].min()), float(f["salary"].max())) if len(f) else (0, 0),
    }
    return f, rep


def split_frame(frame: pd.DataFrame, time_based=False, test_size=0.2, seed=42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split (either time-based on date or randomized shuffle)."""
    if time_based and "date" in frame.columns and frame["date"].notna().any():
        f = frame.sort_values("date", kind="stable").reset_index(drop=True)
        split_idx = int(len(f) * (1.0 - test_size))
        return f.iloc[:split_idx].copy(), f.iloc[split_idx:].copy()
    f = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(f) * (1.0 - test_size))
    return f.iloc[:split_idx].copy(), f.iloc[split_idx:].copy()


def compare_models(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                   time_based=False, folds=5, seed=42, top_skills=40, progress=None) -> tuple[pd.DataFrame, dict]:
    """Train and compare candidate regressors and return results table and fitted pipelines."""
    Xtr, ytr = train[features], train["salary"]
    Xte, yte = test[features], test["salary"]
    cv = TimeSeriesSplit(n_splits=folds) if time_based else KFold(n_splits=folds, shuffle=True, random_state=seed)

    rows = []
    fitted = {}
    candidates = candidate_models(seed=seed)

    for i, (name, est) in enumerate(candidates.items()):
        if progress:
            progress(i, name)
        pipe = make_pipeline(features, est, top_skills=top_skills)
        t0 = time.time()
        scores = cross_validate(pipe, Xtr, ytr, cv=cv, scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        })
        pipe.fit(Xtr, ytr)
        pred_test = pipe.predict(Xte)
        elapsed = round(time.time() - t0, 1)

        test_mae = float(mean_absolute_error(yte, pred_test))
        test_rmse = float(np.sqrt(mean_squared_error(yte, pred_test)))
        test_r2 = float(r2_score(yte, pred_test))

        rows.append({
            "model": name,
            "cv_mae": float(-scores["test_mae"].mean()),
            "cv_rmse": float(-scores["test_rmse"].mean()),
            "cv_r2": float(scores["test_r2"].mean()),
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "seconds": elapsed,
        })
        fitted[name] = pipe

    res = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
    return res, fitted


def train_and_select(frame: pd.DataFrame, features: list[str], time_based=False,
                     test_size=0.2, folds=5, seed=42, top_skills=40, progress=None,
                     data_label="cleaned_jobs.csv", synthetic=True) -> dict:
    """Train candidate models, select best by CV RMSE, calculate empirical range ratios, return bundle."""
    train, test = split_frame(frame, time_based=time_based, test_size=test_size, seed=seed)
    results, fitted = compare_models(train, test, features, time_based=time_based, folds=folds,
                                     seed=seed, top_skills=top_skills, progress=progress)

    best_name = results.iloc[0]["model"]
    best_pipe = fitted[best_name]

    # Out-of-fold empirical error ratio calculation for salary range:
    oof = cross_val_predict(clone(best_pipe), train[features], train["salary"],
                            cv=KFold(folds, shuffle=True, random_state=seed))
    ratio = (train["salary"].values / np.maximum(oof, 1.0))
    q10, q90 = float(np.quantile(ratio, 0.10)), float(np.quantile(ratio, 0.90))

    counts = {c: train[c].value_counts().to_dict() for c in CAT_COLS if c in features}

    skill_vocab = []
    if "skills" in features:
        try:
            binarizer = best_pipe.regressor_.named_steps["prep"].named_transformers_["skills"].named_steps["bin"]
            skill_vocab = list(binarizer.vocabulary_)
        except Exception:
            pass

    date_range = None
    if "date" in frame.columns and frame["date"].notna().any():
        date_range = (str(frame["date"].min()), str(frame["date"].max()))

    bundle = {
        "model": best_pipe,
        "best_name": best_name,
        "features": features,
        "results": results,
        "ratio_q10": q10,
        "ratio_q90": q90,
        "n_train": len(train),
        "n_test": len(test),
        "folds": folds,
        "seed": seed,
        "split": "time-based" if time_based else "random",
        "counts": counts,
        "skill_vocab": skill_vocab,
        "date_range": date_range,
        "exp_range": (float(train["experience"].map(parse_experience).dropna().min()),
                      float(train["experience"].map(parse_experience).dropna().max())) if "experience" in features else (0.0, 30.0),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_label": data_label,
        "synthetic": synthetic,
        "bundle_class_module": "salary_model",
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
    }
    return bundle


def save_bundle(bundle: dict, path: str = MODEL_PATH):
    """Save model bundle to disk using joblib."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, p)


def load_bundle(path: str = MODEL_PATH) -> dict | None:
    """Safely load model bundle. Return None if not found, corrupted, or incompatible."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        bundle = joblib.load(p)
        if not isinstance(bundle, dict) or "model" not in bundle:
            return None
        return bundle
    except Exception as e:
        print(f"[salary_model] Warning: could not load '{path}': {e}")
        return None


def predict_salary(bundle: dict, role=None, experience=None, location=None,
                   skills=None, work_mode=None, education=None) -> dict:
    """Predict annual salary (point estimate, empirical range, warnings)."""
    feats = bundle["features"]
    row_data = {
        "role": role,
        "experience": experience,
        "location": location,
        "skills": skills,
        "work_mode": work_mode,
        "education": education,
    }
    input_df = pd.DataFrame([{k: row_data.get(k) for k in feats}])

    for c in CAT_COLS:
        if c in input_df.columns:
            input_df[c] = input_df[c].astype(object).where(input_df[c].notna() & (input_df[c] != ""), "Unknown")

    pipe = bundle["model"]
    point = float(pipe.predict(input_df)[0])
    point = max(100_000.0, point)

    low = float(max(80_000.0, point * bundle["ratio_q10"]))
    high = float(point * bundle["ratio_q90"])

    # Lookup test MAE
    res = bundle["results"]
    best_row = res[res["model"] == bundle["best_name"]]
    best_mae = float(best_row.iloc[0]["test_mae"]) if not best_row.empty else float("nan")

    warnings = []
    counts = bundle.get("counts", {})
    for c in CAT_COLS:
        if c in feats and row_data[c]:
            val = str(row_data[c]).strip()
            cnt = counts.get(c, {}).get(val, 0)
            if cnt == 0:
                warnings.append(f"“{val}” was not in the training postings for {c}; estimate is based on general trends.")
            elif cnt < 30:
                warnings.append(f"Only {cnt} training postings found for {c} “{val}”. Expect wider variance.")

    exp_val = parse_experience(experience)
    if exp_val is not None:
        lo_e, hi_e = bundle.get("exp_range", (0.0, 30.0))
        if not (lo_e <= exp_val <= hi_e):
            warnings.append(f"Experience {exp_val:g} yrs is outside the typical training range ({lo_e:g}–{hi_e:g} yrs).")

    return {
        "point": point,
        "low": low,
        "high": high,
        "mae": best_mae,
        "warnings": warnings,
        "best_name": bundle.get("best_name", "Model"),
    }
