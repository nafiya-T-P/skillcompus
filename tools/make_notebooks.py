"""Creates the five notebooks in notebooks/. Run once:  python tools/make_notebooks.py   (then this file can be deleted)."""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"
OUT.mkdir(exist_ok=True)


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(nb, OUT / name)


SETUP = ("code", "import sys\nfrom pathlib import Path\nROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
                 "sys.path.insert(0, str(ROOT))\nimport pandas as pd\nimport plotly.express as px\nfrom src import preprocessing as prep")

write("01_data_exploration.ipynb", [
    ("md", "# 01 Data exploration\nInspect the raw job-posting data: dimensions, column names, data types, sample records and data quality."),
    SETUP,
    ("code", "raw, names = prep.load_raw_data()\nprint('Source files:', names)\nprint('Rows, columns:', raw.shape)"),
    ("code", "prep.column_overview(raw)"),
    ("code", "raw.head(10)"),
    ("code", "q = prep.data_quality_report(raw)\nprint('Exact duplicate rows:', q['exact_duplicates'])\nq['missing']"),
    ("code", "q['categories']"),
    ("code", "pd.Series(q['invalid_salary'], name='rows')"),
])
write("02_data_cleaning.ipynb", [
    ("md", "# 02 Data cleaning\nCleans the raw data and logs every step. Invalid salaries are flagged, never silently deleted."),
    SETUP,
    ("code", "raw, names = prep.load_raw_data()\ncleaned, log = prep.clean_dataset(raw)\nlog"),
    ("code", "prep.save_cleaned(cleaned, log, names)\nprint('Saved', prep.CLEANED_PATH)"),
])
write("03_eda.ipynb", [
    ("md", "# 03 Exploratory data analysis\nSalary distribution, job-role counts, city-level demand, experience levels and work modes."),
    SETUP,
    ("code", "prep.ensure_cleaned()\ndf = prep.load_cleaned()\nprint(len(df), 'postings')"),
    ("code", "px.histogram(df['salary_mid_lpa'].dropna(), nbins=40, title='Salary distribution (LPA)')"),
    ("code", "px.bar(df['job_title'].value_counts().head(12), title='Job-role counts')"),
    ("code", "px.bar(df['city'].value_counts().head(12), title='City-level demand')"),
    ("code", "px.bar(df['experience_level'].value_counts(), title='Experience levels')"),
    ("code", "px.bar(df['work_mode'].value_counts(), title='Work modes')"),
])
write("04_feature_engineering.ipynb", [
    ("md", "# 04 Feature engineering\nBuild the model-ready frame and the reproducible preprocessing transformer."),
    SETUP,
    ("code", "from src import salary_model as sm\nfrom src.feature_engineering import build_preprocessor\nprep.ensure_cleaned()\nframe, report = prep.prepare_for_model(prep.load_cleaned())\nprint(report)\nframe.head()"),
    ("code", "train, test = sm.split_frame(frame)\nfeatures = [c for c in sm.FEATURES if c in frame]\npre = build_preprocessor(features, sm.TOP_SKILLS).fit(train[features])\nX = pre.transform(train[features])\nprint('Feature matrix:', X.shape)\nlist(pre.get_feature_names_out())[:20]"),
])
write("05_model_training.ipynb", [
    ("md", "# 05 Model training\nTrain and compare several regression algorithms with the fixed settings from `src/salary_model.py`."),
    SETUP,
    ("code", "from src import salary_model as sm\nprep.ensure_cleaned()\nframe, report = prep.prepare_for_model(prep.load_cleaned())\nbundle = sm.train_and_select(frame)\nbundle['results']"),
    ("code", "print('Selected model:', bundle['best_name'])\nprint('Settings:', bundle['settings'])"),
    ("code", "sm.save_bundle(bundle)\nprint('Saved', sm.MODEL_PATH)"),
])
print("notebooks written to", OUT)