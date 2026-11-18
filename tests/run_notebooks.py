"""Execute the five notebooks top to bottom (needs a dataset in data/raw).  Usage: python tests/run_notebooks.py"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
failed = []
for nb_path in sorted(NB_DIR.glob("0*.ipynb")):
    nb = nbformat.read(nb_path, as_version=4)
    try:
        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}).execute()
        print("OK    ", nb_path.name)
    except Exception as err:
        failed.append(nb_path.name)
        print("FAILED", nb_path.name, "-", str(err)[:300])
sys.exit(1 if failed else 0)

```

## STEP 2: SPLIT MY EXISTING RESUME CODE INTO src/resume_parser.py AND src/skill_analyzer.py
I did not give you these two files, so build them from my existing resume_parser.py (and its skills dictionary). Only move and split code. Do not change its logic.
- src/resume_parser.py (Module 3) keeps: MAX_FILE_SIZE_BYTES (10 MB), extract_text_from_pdf, text normalisation, match_skills, load_skills_dictionary.
- src/skill_analyzer.py (Module 4) gets: get_role_skill_reference and compute_skill_gap, plus helpers used only by them. If they need load_skills_dictionary, import it with "from .resume_parser import load_skills_dictionary".
- dashboard/app.py and api/main.py already call these functions, so keep exactly these signatures and return shapes:
  - resume_parser.extract_text_from_pdf(content: bytes) -> str. It raises ValueError with a clear message when the PDF is empty or has no selectable text. The message for a scanned PDF must contain the word "scanned" or "OCR".
  - resume_parser.match_skills(text: str) -> list[str] (for text containing "Python, SQL" it returns names such as "Python" and "SQL").
  - resume_parser.load_skills_dictionary() -> dict with keys "skills" (list of str) and "learning_suggestions".
  - skill_analyzer.get_role_skill_reference(role: str) -> dict with keys "sufficient_data" (bool), "skills" (list), "message", "role", "n_postings". If the role has too few postings it returns sufficient_data False with a clear message.
  - skill_analyzer.compute_skill_gap(skills, ref_skills, role, learning_suggestions) -> dict with keys "score" (0 to 100), "formula" (text showing matched divided by reference), "matched", "missing", "learning_suggestions" (list of dicts with "skill" and "suggestion"), "disclaimer".
- Use package-relative imports (from . import ...). Make paths independent of the working directory: build the skills dictionary path from Path(__file__), and read the job data through src.preprocessing (call prep.ensure_cleaned(), then prep.load_cleaned(); columns job_title and skills_required).
- If the skills dictionary file is not already in data/external/, move it there and update the path in load_skills_dictionary.
- Learning suggestions stay labelled as suggestions, not hiring guarantees.

## STEP 3: REMOVE FILES THAT ARE NO LONGER PART OF THE PROJECT
First search the project to confirm nothing still imports them, then delete: root app.py (replaced by dashboard/app.py), server.py (replaced by api/main.py), service.py, build.py, prepare_dataset.py (replaced by src/preprocessing.py), the old root salary_model.py and resume_parser.py, settings.py, train_model.py, the static/ folder (old FastAPI web page), old saved models (models/*.joblib), data/processed/model_comparison.csv and any old test or debug script that is not in tests/. My old HTTP test script (the one that posts resumes to port 8000): its checks are replaced by tests/test_api.py, so delete it too.

## STEP 4: CREATE THE NOTEBOOKS
Run "python tools/make_notebooks.py" to create the five notebooks, then delete the tools/ folder.

## STEP 5: INSTALL
Run: pip install -r requirements.txt pytest nbclient ipykernel httpx
Run: python -m ipykernel install --user --name python3
If spaCy is used by my resume code and its language model is missing, install it the way my existing code requires.

## STEP 6: TEST EVERYTHING ON MY REAL DATA (this is the important part)
Run these from the project root, in this order, and show me the real output of each:
1. python -m src.preprocessing   (prints the cleaning log for my real dataset, rows before and after each step)
2. python -m src.salary_model    (trains all models on my real data and prints the comparison table with MAE, RMSE and R2, and the selected model)
3. python -m pytest tests -q     (pipeline tests on synthetic fixture data, rules tests, dashboard tests on my real data, API tests with real PDFs created by pymupdf: normal text PDF, scanned/blank PDF, empty file, non-PDF)
4. python tests/run_notebooks.py (runs the five notebooks top to bottom)
5. Start "streamlit run dashboard/app.py" and confirm it starts with no error in the terminal, and start "uvicorn api.main:app" and open /docs.
6. If I have a real resume PDF in the project folder, also run src.resume_parser.extract_text_from_pdf and match_skills on it and show the skills found.

## STEP 7: FIX LOOP (RULES)
- If anything fails, find the root cause and fix it with the smallest change in src/, api/ or dashboard/. Run the failing step again until it passes.
- Do not edit a test just to make it pass. Only change a test if the test itself is wrong, and tell me why.
- Do not replace my real resume_parser, skill_analyzer, dataset or model with mocks or stand-ins. The only allowed mock is the file-upload patch already inside tests/test_dashboard.py.
- Do not hide errors with try/except that returns empty values. Do not hard-code metrics.
- Do not add features, pages, charts or controls that are not in the files above.
- Do not change anything inside data/raw/.

## FINAL REPORT (give me this when finished)
1. The project tree after your changes.
2. Files created, moved, changed and deleted.
3. The real cleaning log (row counts) and the real model comparison table from my dataset, and which model was selected.
4. The pytest result (number passed and failed) and the notebook run result.
5. Every problem you hit and how you fixed it, and anything the old resume code did that the split does not preserve.
6. A reminder of the commands to run the project: streamlit run dashboard/app.py, uvicorn api.main:app --reload, python -m src.salary_model.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-12T16:14:30+05:30.
</ADDITIONAL_METADATA>