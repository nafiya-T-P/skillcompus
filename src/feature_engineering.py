"""Module 2 - feature engineering: experience features, skill multi-hot encoding and the preprocessing transformer."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT_COLS = ["role", "location", "work_mode", "education"]


def parse_experience_range(v) -> tuple[float, float, float]:
    """Parse experience into (min, max, average) years.
    '2-5 years' -> (2, 5, 3.5); 'Fresher' -> (0, 0, 0); '10+ years' -> (10, 12, 11); 4 -> (4, 4, 4)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return (np.nan, np.nan, np.nan)
    if isinstance(v, (int, float, np.number)):
        x = float(v)
        return (x, x, x)
    s = str(v).strip().lower()
    if any(k in s for k in ("fresher", "entry", "intern", "trainee")):
        return (0.0, 0.0, 0.0)
    m = re.search(r"(\d+\.?\d*)\s*(?:-|to)\s*(\d+\.?\d*)", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo, hi, (lo + hi) / 2.0)
    m = re.search(r"(\d+\.?\d*)\s*\+", s)
    if m:
        lo = float(m.group(1))
        return (lo, lo + 2.0, lo + 1.0)
    m = re.search(r"\d+\.?\d*", s)
    if m:
        x = float(m.group(0))
        return (x, x, x)
    return (np.nan, np.nan, np.nan)


def parse_experience(v) -> float:
    """Average experience in years as a single float (NaN if it cannot be read)."""
    return parse_experience_range(v)[2]


class ExperienceFeatureExtractor(BaseEstimator, TransformerMixin):
    """Average experience, its square (non-linear growth) and the range width, from numbers or text."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vals = pd.DataFrame(X).iloc[:, 0]
        out = np.zeros((len(vals), 3))
        for i, v in enumerate(vals):
            lo, hi, avg = parse_experience_range(v)
            out[i, 0] = avg
            out[i, 1] = avg ** 2 if not np.isnan(avg) else np.nan
            out[i, 2] = (hi - lo) if (not np.isnan(hi) and not np.isnan(lo)) else 0.0
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(["exp_avg", "exp_sq", "exp_range"])


class SkillBinarizer(BaseEstimator, TransformerMixin):
    """Multi-hot encode the most frequent skills plus a skill count. The vocabulary is learned on training data only."""

    def __init__(self, top_n=40, min_count=5):
        self.top_n, self.min_count = top_n, min_count

    @staticmethod
    def tokens(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return []
        if isinstance(v, (list, tuple, set)):
            v = ",".join(map(str, v))
        return sorted({t.strip().lower() for t in re.split(r"[,;|/]", str(v)) if t.strip()})

    def _col(self, X):
        return pd.DataFrame(X).iloc[:, 0]

    def fit(self, X, y=None):
        counts = Counter()
        for v in self._col(X):
            counts.update(self.tokens(v))
        self.vocabulary_ = [s for s, c in counts.most_common(self.top_n) if c >= self.min_count]
        return self

    def transform(self, X):
        idx = {s: i for i, s in enumerate(self.vocabulary_)}
        rows = [self.tokens(v) for v in self._col(X)]
        M = np.zeros((len(rows), len(idx) + 1))
        for r, toks in enumerate(rows):
            for t in toks:
                if t in idx:
                    M[r, idx[t]] = 1.0
            M[r, -1] = float(len(toks))
        return M

    def get_feature_names_out(self, input_features=None):
        return np.array([f"skill:{s}" for s in self.vocabulary_] + ["skill_count"])


def build_preprocessor(features: list[str], top_skills: int) -> ColumnTransformer:
    """One reproducible preprocessing transformer: numeric, categorical and skill features."""
    parts = []
    if "experience" in features:
        parts.append(("num", Pipeline([
            ("ext", ExperienceFeatureExtractor()),
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), ["experience"]))
    cats = [c for c in CAT_COLS if c in features]
    if cats:
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False)),
        ]), cats))
    if "skills" in features:
        parts.append(("skills", Pipeline([
            ("bin", SkillBinarizer(top_n=top_skills)),
            ("sc", StandardScaler()),
        ]), ["skills"]))
    return ColumnTransformer(parts, remainder="drop")