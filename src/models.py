"""
models.py — Model definitions and hyperparameter search spaces.

Each model is wrapped in a sklearn Pipeline so that preprocessing is
always applied before fitting, preventing data leakage during CV.

Design choice: we use a single factory function that returns a dict
of {name: (pipeline, param_grid)} so the benchmark loop is trivial —
just iterate over the dict. Adding a new model means adding one entry here.
"""

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


def get_classifiers(preprocessor) -> dict:
    """
    Return all classification models, each wrapped in a full Pipeline.

    The preprocessor is injected as first step so that CV never leaks
    validation-fold statistics into training.

    Hyperparameter grids are sized for RandomizedSearchCV (50 iterations).
    Ranges are deliberately wide so the search can discover surprising optima
    rather than confirming our prior.

    Mathematical notes per model
    ----------------------------
    LogisticRegression:
        P(y=1|x) = 1 / (1 + exp(-x^T beta))
        C = 1/lambda controls L2 regularisation strength.

    KNN:
        y_hat = argmax_c sum_{i in N_k(x)} 1[y_i == c]
        k is the key trade-off: small k → high variance, large k → high bias.
        weights='distance' gives closer neighbours more influence.

    SVM (RBF kernel):
        K(x, x') = exp(-gamma * ||x - x'||^2)
        C controls margin softness; gamma controls kernel bandwidth.
        We use probability=True to get calibrated scores for AUC.

    DecisionTree:
        Split criterion: Gini G(t) = 1 - sum_k p_k^2
        max_depth prevents overfitting (trees can perfectly memorise training data).

    RandomForest:
        y_hat = (1/B) * sum_b T_b(x)
        n_estimators=200 is a good default — variance gains plateau quickly after ~100.

    GradientBoosting:
        F_m(x) = F_{m-1}(x) + eta * h_m(x)  where h_m fits negative gradient of loss.
        subsample < 1 adds stochastic element — reduces overfitting and speeds training.

    XGBoost:
        Adds L1/L2 tree regularisation on top of gradient boosting.
        reg_alpha (L1) induces sparsity in leaf weights.
        reg_lambda (L2) prevents extreme leaf scores.
    """
    models = {
        "Logistic Regression": (
            Pipeline([("pre", preprocessor),
                      ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
            {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
             "clf__solver": ["lbfgs", "saga"],
             "clf__penalty": ["l2"]},
        ),
        "KNN": (
            Pipeline([("pre", preprocessor),
                      ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": range(3, 31, 2),
             "clf__weights": ["uniform", "distance"],
             "clf__metric": ["euclidean", "manhattan"]},
        ),
        "SVM": (
            Pipeline([("pre", preprocessor),
                      ("clf", SVC(probability=True, random_state=42))]),
            {"clf__C": [0.1, 1, 10, 100],
             "clf__gamma": ["scale", "auto", 0.001, 0.01],
             "clf__kernel": ["rbf", "poly"]},
        ),
        "Decision Tree": (
            Pipeline([("pre", preprocessor),
                      ("clf", DecisionTreeClassifier(random_state=42))]),
            {"clf__max_depth": [3, 5, 7, 10, None],
             "clf__min_samples_split": [2, 5, 10, 20],
             "clf__criterion": ["gini", "entropy"]},
        ),
        "Random Forest": (
            Pipeline([("pre", preprocessor),
                      ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))]),
            {"clf__max_depth": [5, 10, 20, None],
             "clf__min_samples_split": [2, 5, 10],
             "clf__max_features": ["sqrt", "log2", 0.3]},
        ),
        "Gradient Boosting": (
            Pipeline([("pre", preprocessor),
                      ("clf", GradientBoostingClassifier(random_state=42))]),
            {"clf__n_estimators": [100, 200, 300],
             "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
             "clf__max_depth": [3, 4, 5],
             "clf__subsample": [0.7, 0.8, 1.0]},
        ),
    }

    if _XGBOOST_AVAILABLE:
        models["XGBoost"] = (
            Pipeline([("pre", preprocessor),
                      ("clf", XGBClassifier(random_state=42, eval_metric="logloss",
                                            n_jobs=-1, verbosity=0))]),
            {"clf__n_estimators": [100, 200, 300],
             "clf__learning_rate": [0.01, 0.05, 0.1],
             "clf__max_depth": [3, 4, 6],
             "clf__reg_alpha": [0, 0.1, 1],
             "clf__reg_lambda": [1, 5, 10]},
        )

    return models


def get_regressors(preprocessor) -> dict:
    """
    Return all regression models, each wrapped in a Pipeline.

    Mathematical notes
    ------------------
    LinearRegression:   beta_hat = (X^T X)^{-1} X^T y  (OLS closed form)
    Ridge:              beta_hat = (X^T X + lambda I)^{-1} X^T y  (L2 shrinkage)
    Lasso:              argmin_beta ||y - X beta||^2 + lambda ||beta||_1  (L1 → sparsity)
    ElasticNet:         combines L1 and L2; useful when features are correlated.
    """
    models = {
        "Linear Regression": (
            Pipeline([("pre", preprocessor), ("reg", LinearRegression())]),
            {},
        ),
        "Ridge": (
            Pipeline([("pre", preprocessor), ("reg", Ridge())]),
            {"reg__alpha": [0.01, 0.1, 1, 10, 100, 1000]},
        ),
        "Lasso": (
            Pipeline([("pre", preprocessor), ("reg", Lasso(max_iter=5000))]),
            {"reg__alpha": [0.001, 0.01, 0.1, 1, 10]},
        ),
        "ElasticNet": (
            Pipeline([("pre", preprocessor), ("reg", ElasticNet(max_iter=5000))]),
            {"reg__alpha": [0.01, 0.1, 1], "reg__l1_ratio": [0.2, 0.5, 0.8]},
        ),
        "KNN": (
            Pipeline([("pre", preprocessor), ("reg", KNeighborsRegressor())]),
            {"reg__n_neighbors": range(3, 31, 2), "reg__weights": ["uniform", "distance"]},
        ),
        "Decision Tree": (
            Pipeline([("pre", preprocessor), ("reg", DecisionTreeRegressor(random_state=42))]),
            {"reg__max_depth": [3, 5, 7, 10, None], "reg__min_samples_split": [2, 5, 10]},
        ),
        "Random Forest": (
            Pipeline([("pre", preprocessor),
                      ("reg", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))]),
            {"reg__max_depth": [5, 10, 20, None], "reg__max_features": [0.3, "sqrt", 1.0]},
        ),
        "Gradient Boosting": (
            Pipeline([("pre", preprocessor),
                      ("reg", GradientBoostingRegressor(random_state=42))]),
            {"reg__n_estimators": [100, 200], "reg__learning_rate": [0.05, 0.1, 0.2],
             "reg__max_depth": [3, 4, 5], "reg__subsample": [0.8, 1.0]},
        ),
    }

    if _XGBOOST_AVAILABLE:
        models["XGBoost"] = (
            Pipeline([("pre", preprocessor),
                      ("reg", XGBRegressor(random_state=42, n_jobs=-1, verbosity=0))]),
            {"reg__n_estimators": [100, 200], "reg__learning_rate": [0.05, 0.1],
             "reg__max_depth": [3, 4, 6], "reg__reg_lambda": [1, 5, 10]},
        )

    return models