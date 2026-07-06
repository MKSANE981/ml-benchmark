"""
visualization.py — Comparison plots, learning curves, and feature importance.

All figures are saved to disk (results/) rather than displayed interactively,
so the benchmark can run headless on a server.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

PALETTE = [
    "#2196F3", "#4CAF50", "#FF5722", "#9C27B0",
    "#FF9800", "#00BCD4", "#F44336", "#795548",
]


def plot_model_comparison(results: pd.DataFrame, task: str) -> None:
    """
    Horizontal bar chart comparing the primary metric across all models.

    Bars are sorted from best to worst, error bars show ± 1 std across folds.
    This is the first chart a reader sees — keep it clean.
    """
    metric = "auc" if task == "classification" else "rmse"
    std_col = f"{metric}_std" if f"{metric}_std" in results.columns else None

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(results) + 1.5))
    y_pos = np.arange(len(results))
    xerr = results[std_col].values if std_col else None

    bars = ax.barh(
        y_pos, results[metric], xerr=xerr,
        color=PALETTE[:len(results)], alpha=0.85,
        capsize=4, edgecolor="white", linewidth=0.5,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results["model"], fontsize=11)
    ax.set_xlabel(metric.upper(), fontsize=12)
    ax.set_title(f"Model Comparison — {task.title()} ({metric.upper()}, 5-fold CV)", fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, results[metric]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"comparison_{task}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: results/comparison_{task}.png")


def plot_learning_curves(estimator, X, y, task: str, model_name: str) -> None:
    """
    Plot train vs validation score as a function of training set size.

    Learning curves diagnose bias-variance trade-off:
    - High bias (underfitting): both curves plateau low
    - High variance (overfitting): large gap between train and val curves
    - Well-fitted: curves converge at a high score
    """
    from sklearn.model_selection import learning_curve

    metric = "roc_auc" if task == "classification" else "neg_root_mean_squared_error"
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring=metric, n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="#2196F3")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="#FF5722")
    ax.plot(train_sizes, train_mean, "o-", color="#2196F3", label="Training score")
    ax.plot(train_sizes, val_mean, "s-", color="#FF5722", label="Validation score")
    ax.set_xlabel("Training set size", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"Learning Curve — {model_name}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    fig.savefig(RESULTS_DIR / f"learning_curve_{safe_name}.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(estimator, feature_names: list[str], model_name: str, top_n: int = 20) -> None:
    """
    Extract and plot feature importance for tree-based models.

    For linear models we use the absolute value of coefficients (after
    standardisation, these are comparable across features).
    For Random Forest / GBoost / XGBoost we use built-in impurity importance.

    Note: impurity-based importance has known bias towards high-cardinality
    features. For production use, prefer permutation importance.
    """
    clf = estimator.named_steps.get("clf") or estimator.named_steps.get("reg")

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        label = "Impurity Importance"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_.ravel())
        label = "|Coefficient|"
    else:
        print(f"  {model_name}: no importance available, skipping.")
        return

    n = min(top_n, len(feature_names))
    idx = np.argsort(importances)[-n:]

    fig, ax = plt.subplots(figsize=(8, 0.4 * n + 2))
    ax.barh(range(n), importances[idx], color="#4CAF50", alpha=0.85)
    ax.set_yticks(range(n))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=9)
    ax.set_xlabel(label, fontsize=11)
    ax.set_title(f"Feature Importance — {model_name} (top {n})", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    fig.savefig(RESULTS_DIR / f"importance_{safe_name}.png", dpi=150)
    plt.close(fig)