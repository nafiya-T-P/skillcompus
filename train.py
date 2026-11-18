"""Module 2: Model Training Script.

Trains candidate regression models (Ridge, RandomForest, GradientBoosting,
HistGradientBoosting, XGBoost if available) on the single processed dataset:
    data/processed/cleaned_jobs.csv

Saves the best-performing model bundle to:
    models/salary_pipeline.joblib
and the comparison table to:
    data/processed/model_comparison.csv

Always run this script (never run `python salary_model.py` directly)
so custom transformers are pickled under module namespace 'salary_model'.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure root directory and src directory are on sys.path
ROOT = Path(__file__).resolve().parent
for p in [str(ROOT), str(ROOT / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
try:
    from src import salary_model as sm
    from src import preprocessing as prep
except ImportError:
    import salary_model as sm
    import preprocessing as prep

DEFAULT_CSV = ROOT / "data" / "processed" / "cleaned_jobs.csv"
DEFAULT_OUT = sm.MODEL_PATH

COLS_MAPPING = {
    "target": "salary_mid_inr_pa",
    "role": "job_title",
    "experience": "experience_years",
    "location": "city",
    "skills": "skills_required",
    "work_mode": "work_mode",
    "education": "education",
    "date": "posted_date",
}


def run_training(csv_path: str | Path = DEFAULT_CSV,
                 top_skills: int = 40,
                 out_path: str | Path = DEFAULT_OUT,
                 time_split: bool = False,
                 folds: int = 5,
                 seed: int = 42) -> dict:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: dataset file not found at '{csv_file}'.")
        print("Please run `python -m src.preprocessing` first to generate cleaned_jobs.csv.")
        sys.exit(1)

    print(f"Loading cleaned dataset from: {csv_file}")
    df = pd.read_csv(csv_file, low_memory=False)

    print("Standardizing columns and cleaning for training...")
    std_df = prep.standardize_frame(df, COLS_MAPPING)
    train_frame, report = prep.clean_for_training(std_df)

    features = [c for c in ["role", "experience", "location", "skills", "work_mode", "education"] if c in train_frame.columns]

    print(f"Training rows: {report['rows_out']:,} (filtered from {report['rows_in']:,})")
    print(f"Features:      {features}")
    print(f"Top skills:    {top_skills}")
    print(f"Split type:    {'time-based' if time_split else 'randomized (80/20)'}")
    print(f"CV folds:      {folds}")
    print(f"Random seed:   {seed}")
    print()

    print("Training candidate models and evaluating cross-validation RMSE...")
    bundle = sm.train_and_select(
        frame=train_frame,
        progress=lambda i, name: print(f"  [{i + 1}] Training & cross-validating {name}..."),
        data_label="cleaned_jobs.csv (synthetic practice dataset)",
    )

    results_df = bundle["results"]
    print("\nModel Comparison Table:")
    # Format comparison table cleanly
    disp_df = results_df.copy()
    for col in ["cv_mae", "cv_rmse", "test_mae", "test_rmse"]:
        disp_df[col] = disp_df[col].map(lambda v: f"INR {v:,.0f}")
    for col in ["cv_r2", "test_r2"]:
        disp_df[col] = disp_df[col].map(lambda v: f"{v:.3f}")
    print(disp_df.to_string(index=False))

    best_name = bundle["best_name"]
    best_row = results_df[results_df["model"] == best_name].iloc[0]
    print(f"\nBest Model Selected: {best_name}")
    print(f"  CV RMSE:   INR {best_row['cv_rmse']:,.0f}")
    print(f"  Test MAE:  INR {best_row['test_mae']:,.0f}")
    print(f"  Test R2:   {best_row['test_r2']:.3f} ({best_row['test_r2']*100:.1f}% variation explained)")
    print(f"  Empirical Range Band: [{bundle['ratio_q10']:.2f}x - {bundle['ratio_q90']:.2f}x]")

    # Save outputs
    out_file = Path(out_path)
    sm.save_bundle(bundle, out_file)
    print(f"\nSaved model bundle to: {out_file}")

    comp_csv = ROOT / "data" / "processed" / "model_comparison.csv"
    comp_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(comp_csv, index=False)
    print(f"Saved comparison CSV to: {comp_csv}")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help=f"Path to cleaned CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--top-skills", type=int, default=40, help="Number of top skills to encode (default: 40)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"Output .joblib path (default: {DEFAULT_OUT})")
    parser.add_argument("--time-split", action="store_true", help="Use time-based train/test split")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    run_training(
        csv_path=args.csv,
        top_skills=args.top_skills,
        out_path=args.out,
        time_split=args.time_split,
        folds=args.folds,
        seed=args.seed,
    )
