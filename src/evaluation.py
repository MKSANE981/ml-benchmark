"""
evaluation.py — Cross-validation, hyperparameter tuning, and metrics.

All evaluation goes through cross-validation — never fit on the full dataset
and report train-set metrics. That's the single most common mistake in ML
benchmarks and it makes results completely unreliable.
"""

import time
import warnings
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold, cross_validate
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    mean_squared_error, mean_absolute_error, r2_score,
)

warnings.filterwarnings("ignore")


def tune_and_evaluate_classifier(
    name: str,
    pipeline,
    param_grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    n_iter: int = 50,
    random_state: int = 42,
) -> dict:
    """
    Run RandomizedSearchCV then evaluate the best estimator with cross-validation.

    Why RandomizedSearchCV over GridSearchCV?
    With a large hyperparam space, grid search is O(|grid|) evaluations.
    Random search samples the same budget across the full distribution — in
    practice it finds equally good hyperparameters with far fewer evaluations
    (Bergstra & Bengio, 2012, JMLR).

    Returns a dict with: name, best_params, accuracy, f1, auc, logloss, fit_time.
    """
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    if param_grid:
        search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=n_iter,
            cv=cv_strategy, scoring="roc_auc",
            random_state=random_state, n_jobs=-1, refit=True,
        )
        search.fit(X, y)
        best_estimator = search.best_estimator_
        best_params = search.best_params_
    else:
        best_estimator = pipeline
        best_params = {}

    t0 = time.perf_counter()
    cv_results = cross_validate(
        best_estimator, X, y,
        cv=cv_strategy,
        scoring={"accuracy": "accuracy", "f1": "f1_weighted", "auc": "roc_auc"},
        return_train_score=False,
    )
    elapsed = time.perf_counter() - t0

    return {
        "model": name,
        "accuracy": cv_results["test_accuracy"].mean(),
        "accuracy_std": cv_results["test_accuracy"].std(),
        "f1": cv_results["test_f1"].mean(),
        "auc": cv_results["test_auc"].mean(),
        "fit_time_s": elapsed,
        "best_params": best_params,
    }


def tune_and_evaluate_regressor(
    name: str,
    pipeline,
    param_grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    n_iter: int = 50,
    random_state: int = 42,
) -> dict:
    """
    Same pattern as the classifier version, adapted for regression metrics.

    RMSE is the primary metric — it has the same units as the target,
    which makes it interpretable. R² measures explained variance (1 = perfect,
    0 = same as predicting the mean, <0 = worse than the mean).
    """
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    if param_grid:
        search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=n_iter,
            cv=cv_strategy, scoring="neg_root_mean_squared_error",
            random_state=random_state, n_jobs=-1, refit=True,
        )
        search.fit(X, y)
        best_estimator = search.best_estimator_
        best_params = search.best_params_
    else:
        best_estimator = pipeline
        best_params = {}

    t0 = time.perf_counter()
    cv_results = cross_validate(
        best_estimator, X, y,
        cv=cv_strategy,
        scoring={"rmse": "neg_root_mean_squared_error",
                 "mae": "neg_mean_absolute_error",
                 "r2": "r2"},
        return_train_score=False,
    )
    elapsed = time.perf_counter() - t0

    return {
        "model": name,
        "rmse": -cv_results["test_rmse"].mean(),
        "rmse_std": cv_results["test_rmse"].std(),
        "mae": -cv_results["test_mae"].mean(),
        "r2": cv_results["test_r2"].mean(),
        "fit_time_s": elapsed,
        "best_params": best_params,
    }


def run_benchmark(models: dict, X, y, task: Literal["classification", "regression"]) -> pd.DataFrame:
    """
    Iterate over all models, tune and evaluate each, return a sorted DataFrame.

    Sorting by AUC (classification) or RMSE (regression) makes the winner
    immediately obvious in the output table.
    """
    results = []
    evaluator = tune_and_evaluate_classifier if task == "classification" else tune_and_evaluate_regressor

    for name, (pipeline, param_grid) in models.items():
        print(f"  -> {name} ...", flush=True)
        result = evaluator(name, pipeline, param_grid, X, y)
        results.append(result)
        if task == "classification":
            print(f"     AUC={result['auc']:.4f}  F1={result['f1']:.4f}  ({result['fit_time_s']:.1f}s)")
        else:
            print(f"     RMSE={result['rmse']:.4f}  R2={result['r2']:.4f}  ({result['fit_time_s']:.1f}s)")

    df = pd.DataFrame(results)
    sort_col = "auc" if task == "classification" else "rmse"
    ascending = task == "regression"
    return df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)