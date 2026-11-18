"""Modules 1 and 2 on synthetic data: loading, quality report, cleaning, training, comparison, prediction."""
import numpy as np
import pandas as pd
import pytest

from src import preprocessing as prep
from src import salary_model as sm
from tests.make_fixture import make_fixture


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    folder = tmp_path_factory.mktemp("raw")
    df = make_fixture()
    half = len(df) // 2
    df.iloc[:half].to_csv(folder / "part1.csv", index=False)      # two CSV files must be combined
    df.iloc[half:].to_csv(folder / "part2.csv", index=False)
    raw, names = prep.load_raw_data(folder)
    cleaned, log = prep.clean_dataset(raw)
    frame, report = prep.prepare_for_model(cleaned)
    bundle = sm.train_and_select(frame)
    return {"expected_rows": len(df), "raw": raw, "names": names, "cleaned": cleaned, "log": log,
            "frame": frame, "report": report, "bundle": bundle}


def test_multiple_csv_files_are_combined(data):
    assert len(data["names"]) == 2
    assert len(data["raw"]) == data["expected_rows"]


def test_quality_report_measures_problems(data):
    q = prep.data_quality_report(data["raw"])
    assert q["exact_duplicates"] > 0
    assert len(q["missing"]) > 0
    assert q["categories"]["spelling_variants_to_merge"].sum() > 0
    assert sum(q["invalid_salary"].values()) > 0
    assert {"column", "dtype", "non_null"} <= set(prep.column_overview(data["raw"]).columns)


def test_cleaning(data):
    c = data["cleaned"]
    assert len(c) < len(data["raw"])                                   # duplicates removed
    assert "bangalore" not in set(c["city"].dropna().str.lower())      # alias merged
    flagged = c[c["salary_issue"] != ""]
    assert len(flagged) > 0 and flagged["salary_mid_inr_pa"].isna().all()   # flagged, not trained on
    assert c["experience_years"].dropna().max() <= prep.EXP_MAX
    assert len(data["log"]) >= 7


def test_model_frame_has_no_invalid_salaries(data):
    f = data["frame"]
    assert f["salary"].between(prep.MIN_SALARY_INR, prep.MAX_SALARY_INR).all()


def test_all_models_compared_and_fitted(data):
    b = data["bundle"]
    res = b["results"]
    assert len(res) >= 5
    assert not res.isna().any().any()                                  # no empty metric cells
    assert not res["model"].str.contains("Baseline").any()
    assert b["best_name"] == res.iloc[0]["model"]
    assert res["cv_rmse"].is_monotonic_increasing                      # chosen by cross-validated RMSE
    X = data["frame"][b["features"]].head(5)
    assert len(b["model"].predict(X)) == 5                             # the stored model is fitted


def test_fixed_settings(data):
    b = data["bundle"]
    assert b["settings"] == {"test_size_pct": 20, "cv_folds": 5, "seed": 42, "top_skills": 40}
    assert sm.bundle_is_current(b)
    share = b["n_test"] / (b["n_train"] + b["n_test"])
    assert abs(share - 0.20) < 0.01


def test_prediction_range_and_warnings(data):
    b = data["bundle"]
    r = sm.predict_salary(b, role="Data Scientist", experience=4, location="Bengaluru", skills="python, sql",
                          work_mode="Hybrid", education="Master's")
    assert r["low"] <= r["point"] <= r["high"]
    odd = sm.predict_salary(b, role="Made Up Role", experience=50)
    assert len(odd["warnings"]) >= 2
    assert len(sm.LIMITATIONS) >= 5


def test_training_is_reproducible(data):
    again = sm.train_and_select(data["frame"])
    assert again["best_name"] == data["bundle"]["best_name"]
    assert np.allclose(again["results"]["test_rmse"], data["bundle"]["results"]["test_rmse"])