"""Structure and rule checks: folder layout, removed files, no user-editable training settings, no fake metrics."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI_FILES = [ROOT / "dashboard" / "app.py", ROOT / "api" / "main.py"]


def test_recommended_structure_exists():
    for p in ["data/raw", "data/processed", "data/external", "notebooks/01_data_exploration.ipynb",
              "notebooks/02_data_cleaning.ipynb", "notebooks/03_eda.ipynb", "notebooks/04_feature_engineering.ipynb",
              "notebooks/05_model_training.ipynb", "src/preprocessing.py", "src/feature_engineering.py",
              "src/salary_model.py", "src/resume_parser.py", "src/skill_analyzer.py", "models", "api/main.py",
              "dashboard/app.py"]:
        assert (ROOT / p).exists(), f"missing {p}"


def test_unwanted_files_are_removed():
    for p in ["app.py", "server.py", "service.py", "build.py", "prepare_dataset.py", "salary_model.py",
              "resume_parser.py", "settings.py", "train_model.py", "static"]:
        assert not (ROOT / p).exists(), f"should be removed: {p}"


def test_no_user_editable_training_settings():
    text = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    for bad in ["Test share", "CV folds", "Random seed", "Top skills used", "Time-based split", "Train and compare models",
                "BASELINE_METRICS", "w_test", "w_folds", "w_seed", "w_top", "w_time", "Customize Column Mapping"]:
        assert bad not in text, f"found '{bad}' in dashboard/app.py"
    assert "BASELINE_METRICS" not in (ROOT / "src" / "salary_model.py").read_text(encoding="utf-8")


def test_no_module_numbers_in_user_interface():
    for f in UI_FILES:
        assert not re.search(r"Module\s*\d", f.read_text(encoding="utf-8")), f"'Module <n>' text in {f.name}"


def test_no_csv_upload_for_dataset():
    text = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert text.count("file_uploader") == 1          # only the resume PDF upload exists
    assert 'type=["pdf"]' in text