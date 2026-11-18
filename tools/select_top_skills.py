"""One‑off script to choose the optimal number of top skills.
It evaluates candidate values [20,30,40,50,60] on the training split
using the default CV folds and seed, and prints a CSV‑style table.
The script is not part of the web app; run it manually with
`python tools/select_top_skills.py`.
"""
import pandas as pd
from salary_model import load_project_dataset, prepare, split_frame, compare_models, TEST_SIZE_PCT, CV_FOLDS, RANDOM_SEED

# Load dataset (same logic as app)
raw_df, _ = load_project_dataset()
if raw_df is None:
    raise RuntimeError("Dataset not found")
# Use existing column mapping defaults (same as app)
cols = {k: v for k, _, _ in salary_model.FIELDS for v in []}  # placeholder – we'll use defaults via prepare
# For simplicity reuse app's prepare function to get processed frame
frame, _ = prepare(raw_df, {}, 100000, 6000000)  # dummy col mapping; will be overridden inside prepare
# Split once (train only) – we use the same random seed as the app
train, test = split_frame(frame, time_based=False, test_size=TEST_SIZE_PCT/100, seed=RANDOM_SEED)
features = [c for c in ["role","experience","location","skills","work_mode","education"] if c in train]

candidates = [20,30,40,50,60]
results = []
for top in candidates:
    res, _ = compare_models(train, test, features, time_based=False, folds=CV_FOLDS, seed=RANDOM_SEED, top_skills=top)
    best = res.iloc[0]
    results.append({"top_skills": top, "cv_rmse": best["cv_rmse"], "cv_r2": best["cv_r2"]})

print("top_skills,cv_rmse,cv_r2")
for r in results:
    print(f"{r['top_skills']},{r['cv_rmse']:.0f},{r['cv_r2']:.4f}")
