"""
preprocessing.py — Data cleaning and feature engineering pipelines.

Why a dedicated module instead of inline sklearn pipelines?
Because we want reproducible, inspectable transformations that can be
audited step by step — especially important when comparing models where
the only variable should be the algorithm, not the preprocessing.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def identify_feature_types(df: pd.DataFrame, target: str) -> tuple[list, list]:
    """
    Split columns into numeric and categorical based on dtype.

    We treat integer columns with low cardinality (<=20 unique values) as
    categorical — e.g. ordinal survey responses or binary flags that sklearn
    would otherwise scale unnecessarily.
    """
    features = [c for c in df.columns if c != target]
    numeric, categorical = [], []
    for col in features:
        if df[col].dtype == object or df[col].nunique() <= 20 and df[col].dtype in ("int32", "int64"):
            categorical.append(col)
        else:
            numeric.append(col)
    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale: bool = True,
) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that handles the full preprocessing graph.

    Numeric branch:
        median imputation → optional StandardScaler
        (median over mean because it is robust to outliers — a single extreme
        value won't shift the imputed value for hundreds of rows)

    Categorical branch:
        constant imputation ("missing") → OneHotEncoder
        (we flag missingness explicitly rather than imputing the mode, because
        "not answered" can itself carry signal)

    Parameters
    ----------
    numeric_features : list of column names to treat as numeric
    categorical_features : list of column names to treat as categorical
    scale : whether to apply StandardScaler to numeric features.
        Must be True for distance-based models (KNN, SVM).
        Can be False for tree-based models (they are scale-invariant).
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def load_adult_income() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the Adult Income dataset from sklearn's OpenML mirror.

    Target: binary — income >50K (1) or <=50K (0).
    48 842 rows, 14 features (age, education, hours-per-week, …).

    We remap the string target to 0/1 here to keep downstream code clean.
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    X = data.data
    y = (data.target == ">50K").astype(int)
    return X, y


def load_california_housing() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the California Housing dataset (regression benchmark).

    Target: median house value in $100 000s.
    20 640 rows, 8 numeric features derived from the 1990 California census.
    All features are already numeric — only scaling is needed.
    """
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    return data.data, data.target