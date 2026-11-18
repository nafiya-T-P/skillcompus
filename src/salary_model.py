"""Module 2 - salary prediction: reproducible pipeline, training and comparison of several regression algorithms,
and a qualified salary range.

Training settings are fixed here (developer-controlled, not user-editable), so the dashboard text,
the trained model and the saved file always agree.

Retrain from the project root:   python -m src.salary_model
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

try:
    from .feature_engineering import CAT_COLS, build_preprocessor, parse_experience
    from .preprocessing import ROOT
except ImportError:
    from feature_engineering import CAT_COLS, build_preprocessor, parse_experience
    from preprocessing import ROOT

# ──────────────── fixed training settings (single source of truth) ────────────────
TEST_SIZE_PCT = 20       # share of rows held out as the test set
CV_FOLDS = 5             # cross-validation folds on the training part
RANDOM_SEED = 42         # makes every run reproducible
TOP_SKILLS = 40          # most common skills turned into features
BUNDLE_VERSION = 3       # bump when the bundle layout changes so old saved models are retrained
MIN_RANGE_INR = 100_000.0  # lowest value shown for the bottom of a range

FEATURES = ["role", "experience", "location", "skills", "work_mode", "education"]
MODEL_PATH = ROOT / "models" / "salary_pipeline.joblib"
SETTINGS = {"test_size_pct": TEST_SIZE_PCT, "cv_folds": CV_FOLDS, "seed": RANDOM_SEED, "top_skills": TOP_SKILLS}

LIMITATIONS = [
    "Estimates are learned from salaries disclosed in job postings, not from verified employee pay.",
    "Postings that disclose salary may differ from those that do not, so the training data can be biased.",
    "The range is an empirical band from past validation errors. It is not a statistical confidence interval.",
    "Rare roles, cities and unusual skill combinations have little training data, so errors are larger there.",
    "The model uses only role, experience, location, skills, work mode and education. It ignores company, "
    "negotiation, performance and market changes after the data date.",
    "Sensitive personal attributes (gender, age, religion, etc.) are deliberately not used.",
    "Treat results as a rough guide for research, not as an offer or a guarantee.",
]


# ───────────────────────── split, models, pipeline ─────────────────────────
def split_frame(frame: pd.DataFrame, test_size: float = TEST_SIZE_PCT / 100, seed: int = RANDOM_SEED):
    """Random train/test split. It happens BEFORE any fitting, so test rows are never seen during training."""
    idx = np.random.default_rng(seed).permutation(len(frame))
    cut = int(len(frame) * (1 - test_size))
    return frame.iloc[idx[:cut]], frame.iloc[idx[cut:]]


def candidate_models(seed: int = RANDOM_SEED) -> dict:
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0, random_state=seed),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=seed),
        "Random Forest": RandomForestRegressor(n_estimators=250, max_depth=16, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.85, random_state=seed),
    }
    try:  # optional
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=450, learning_rate=0.04, max_depth=3, subsample=0.85,
                                         colsample_bytree=0.85, random_state=seed, n_jobs=-1)
    except Exception:
        pass
    return models


def make_pipeline(features: list[str], estimator, top_skills: int = TOP_SKILLS):
    """Preprocessing + model, trained on log(salary) and predicting back in INR."""
    inner = Pipeline([("prep", build_preprocessor(features, top_skills)), ("model", estimator)])
    return TransformedTargetRegressor(regressor=inner, func=np.log1p, inverse_func=np.expm1)


def metrics(y_true, y_pred) -> dict:
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred))}


def compare_models(train: pd.DataFrame, test: pd.DataFrame, features: list[str], progress=None):
    """Train every candidate on the same split and score it with the same metrics
    (cross-validation on the training part + the held-out test part). Every returned model is fitted."""
    Xtr, ytr, Xte, yte = train[features], train["salary"], test[features], test["salary"]
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rows, fitted = [], {}
    for i, (name, est) in enumerate(candidate_models().items()):
        if progress:
            progress(i, name)
        pipe = make_pipeline(features, est)
        t0 = time.time()
        scores = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=scoring)   # fits copies only
        pipe.fit(Xtr, ytr)                                                  # fit the real model before using it
        te = metrics(yte, pipe.predict(Xte))
        rows.append({"model": name,
                     "cv_mae": -scores["test_mae"].mean(), "cv_rmse": -scores["test_rmse"].mean(), "cv_r2": scores["test_r2"].mean(),
                     "test_mae": te["mae"], "test_rmse": te["rmse"], "test_r2": te["r2"],
                     "seconds": round(time.time() - t0, 1)})
        fitted[name] = pipe
    results = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)   # choose by CV, not by test
    return results, fitted


# ───────────────────────── train, save, load ─────────────────────────
def train_and_select(frame: pd.DataFrame, progress=None, data_label: str | None = None) -> dict:
    """Train all candidates with the fixed settings and return a bundle holding the best (fitted) pipeline."""
    features = [c for c in FEATURES if c in frame]
    train, test = split_frame(frame)
    results, fitted = compare_models(train, test, features, progress)
    best_name = results.iloc[0]["model"]
    best = fitted[best_name]

    # Empirical range: ratio actual / predicted on unseen cross-validation folds of the training data.
    oof = cross_val_predict(clone(best), train[features], train["salary"],
                            cv=KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED))
    ratio = train["salary"].values / np.maximum(oof, 1.0)
    q10, q90 = np.quantile(ratio, [0.10, 0.90])

    prep = best.regressor_.named_steps["prep"]
    return {
        "version": BUNDLE_VERSION, "settings": dict(SETTINGS), "data_label": data_label,
        "model": best, "best_name": best_name, "features": features, "results": results,
        "ratio_q10": float(q10), "ratio_q90": float(q90),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "n_train": len(train), "n_test": len(test),
        "exp_range": (float(train["experience"].min()), float(train["experience"].max())) if "experience" in features else None,
        "counts": {c: train[c].value_counts().to_dict() for c in CAT_COLS if c in features},
        "skill_vocab": list(prep.named_transformers_["skills"].named_steps["bin"].vocabulary_) if "skills" in features else [],
    }


def save_bundle(bundle: dict, path=MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path=MODEL_PATH):
    """Load the saved model. Returns None if there is none or it cannot be read (e.g. an older layout)."""
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as err:
        print(f"Could not load {path}: {err}. It will be retrained.")
        return None


def bundle_is_current(bundle) -> bool:
    return bool(bundle) and bundle.get("version") == BUNDLE_VERSION and bundle.get("settings") == SETTINGS


# ───────────────────────── prediction ─────────────────────────
def predict_salary(bundle: dict, role=None, experience=None, location=None, skills=None, work_mode=None, education=None) -> dict:
    """Point estimate, a qualified range and warnings. All values are INR per annum."""
    feats = bundle["features"]
    row = {"role": role, "experience": experience, "location": location, "skills": skills,
           "work_mode": work_mode, "education": education}
    X = pd.DataFrame([{k: row[k] for k in feats}])
    for c in CAT_COLS:
        if c in X:
            X[c] = X[c].astype(object).where(X[c].notna() & (X[c] != ""), np.nan)
    point = float(bundle["model"].predict(X)[0])

    warnings = []
    for c in CAT_COLS:
        if c in feats and row[c]:
            n = bundle["counts"][c].get(row[c], 0)
            if n == 0:
                warnings.append(f"“{row[c]}” was not in the training data, so its effect is not learned.")
            elif n < 30:
                warnings.append(f"Only {n} training postings for {c.replace('_', ' ')} “{row[c]}”. Expect a wider error.")
    exp_num = parse_experience(experience)
    if bundle.get("exp_range") and not np.isnan(exp_num):
        lo_e, hi_e = bundle["exp_range"]
        if not (lo_e <= exp_num <= hi_e):
            warnings.append(f"Experience {exp_num:g} years is outside the training range ({lo_e:g}–{hi_e:g}).")
    return {"point": point, "low": max(MIN_RANGE_INR, point * bundle["ratio_q10"]), "high": point * bundle["ratio_q90"],
            "mae": float(bundle["results"].iloc[0]["test_mae"]), "warnings": warnings}


def get_or_train_bundle(frame: pd.DataFrame | None = None, progress=None):
    """Saved model if it matches the fixed settings, otherwise train one (needs the prepared frame) and save it."""
    bundle = load_bundle()
    if bundle_is_current(bundle):
        return bundle
    if frame is None or len(frame) < 50:
        return None
    bundle = train_and_select(frame, progress=progress)
    save_bundle(bundle)
    return bundle


def main():
    from . import preprocessing as prep
    prep.ensure_cleaned()
    frame, report = prep.prepare_for_model(prep.load_cleaned())
    print("training rows:", report)
    print(f"Settings: test {TEST_SIZE_PCT}% | {CV_FOLDS}-fold CV | seed {RANDOM_SEED} | top {TOP_SKILLS} skills | random split")
    bundle = train_and_select(frame, progress=lambda i, name: print(f"  training {name} ..."))
    save_bundle(bundle)
    pd.set_option("display.width", 200, "display.float_format", "{:,.2f}".format)
    print(bundle["results"].to_string(index=False))
    print(f"\nSelected (lowest CV RMSE): {bundle['best_name']}  | saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()