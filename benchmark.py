"""
benchmark.py — End-to-end benchmark runner.

Usage:
    python benchmark.py --task classification
    python benchmark.py --task regression
    python benchmark.py --task both
"""

import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing import (
    build_preprocessor,
    identify_feature_types,
    load_adult_income,
    load_california_housing,
)
from src.models import get_classifiers, get_regressors
from src.evaluation import run_benchmark
from src.visualization import plot_model_comparison

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_classification():
    print("\n=== CLASSIFICATION BENCHMARK (Adult Income) ===\n")
    X, y = load_adult_income()
    numeric_features, categorical_features = identify_feature_types(X, target="class")
    preprocessor = build_preprocessor(numeric_features, categorical_features, scale=True)
    models = get_classifiers(preprocessor)
    results = run_benchmark(models, X, y, task="classification")
    results.to_csv(RESULTS_DIR / "classification_results.csv", index=False)
    plot_model_comparison(results, "classification")
    print("\n--- Classification Results ---")
    print(results[["model", "accuracy", "f1", "auc", "fit_time_s"]].to_string(index=False))


def run_regression():
    print("\n=== REGRESSION BENCHMARK (California Housing) ===\n")
    X, y = load_california_housing()
    numeric_features = list(X.columns)
    preprocessor = build_preprocessor(numeric_features, [], scale=True)
    models = get_regressors(preprocessor)
    results = run_benchmark(models, X, y, task="regression")
    results.to_csv(RESULTS_DIR / "regression_results.csv", index=False)
    plot_model_comparison(results, "regression")
    print("\n--- Regression Results ---")
    print(results[["model", "rmse", "mae", "r2", "fit_time_s"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["classification", "regression", "both"], default="both")
    args = parser.parse_args()

    if args.task in ("classification", "both"):
        run_classification()
    if args.task in ("regression", "both"):
        run_regression()

    print("\nDone. Results saved to results/")