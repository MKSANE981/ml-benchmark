# ML Benchmark — Rigorous Comparison of Classical Machine Learning Algorithms

A systematic comparison of 8 machine learning algorithms on real-world datasets,
with mathematical derivations, cross-validation, and interpretability analysis.

## Algorithms Covered

| Algorithm | Type | Key Idea |
|-----------|------|----------|
| Logistic / Linear Regression | Baseline | Closed-form / MLE |
| Ridge & Lasso | Regularised | L2 / L1 penalty |
| K-Nearest Neighbours | Instance-based | Distance-weighted vote |
| Support Vector Machine | Margin-based | Maximum-margin hyperplane |
| Decision Tree | Tree-based | Recursive Gini/MSE split |
| Random Forest | Ensemble (bagging) | Average of decorrelated trees |
| Gradient Boosting | Ensemble (boosting) | Sequential residual fitting |
| XGBoost | Ensemble (boosting+) | Regularised gradient boosting |

---

## Mathematical Foundations

### Linear / Logistic Regression

Ordinary Least Squares estimator (closed form):

$$\hat{\beta} = (X^\top X)^{-1} X^\top y$$

Logistic regression models the log-odds:

$$\log \frac{P(Y=1 \mid X)}{1 - P(Y=1 \mid X)} = X\beta$$

Optimised via maximum likelihood:

$$\ell(\beta) = \sum_{i=1}^n \left[ y_i \log \hat{p}_i + (1 - y_i) \log (1 - \hat{p}_i) \right]$$

### Ridge & Lasso Regression

Ridge adds an L2 penalty — trades variance for bias and has a closed-form solution:

$$\hat{\beta}^{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y$$

Lasso adds an L1 penalty — induces sparsity (automatic feature selection):

$$\hat{\beta}^{\text{lasso}} = \arg\min_\beta \left\{ \|y - X\beta\|^2 + \lambda \|\beta\|_1 \right\}$$

### K-Nearest Neighbours

Prediction for a new point $x$ uses its $k$ closest training points:

$$\hat{y} = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} y_i \quad \text{(regression)}$$

$$\hat{y} = \underset{c}{\arg\max} \sum_{i \in \mathcal{N}_k(x)} \mathbf{1}[y_i = c] \quad \text{(classification)}$$

Distance is typically Euclidean:

$$d(x, x') = \sqrt{\sum_{j=1}^{p} (x_j - x'_j)^2}$$

No training phase — all computation happens at inference time. Sensitive to scale → always standardise features first.

### Support Vector Machine

Hard-margin: maximise the margin $\frac{2}{\|\mathbf{w}\|}$ subject to correct classification:

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1$$

Soft-margin (allows violations via slack variables $\xi_i \geq 0$):

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i$$

The RBF kernel implicitly maps to an infinite-dimensional feature space:

$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$

### Decision Tree

At each node, the algorithm selects the feature and threshold that minimises impurity. For classification (Gini):

$$G(t) = 1 - \sum_{k=1}^{K} p_k^2$$

For regression (MSE):

$$\text{MSE}(t) = \frac{1}{|t|} \sum_{i \in t} (y_i - \bar{y}_t)^2$$

The split maximises the weighted impurity reduction:

$$\Delta(t, s) = G(t) - \frac{|t_L|}{|t|} G(t_L) - \frac{|t_R|}{|t|} G(t_R)$$

### Random Forest

$B$ trees, each trained on a bootstrap sample, each split restricted to $\sqrt{p}$ random features. Decorrelation between trees is what makes ensembling effective:

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x) \quad \text{(regression)}$$

Variance of the ensemble (if trees have correlation $\rho$):

$$\text{Var}\left(\bar{T}\right) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

As $B \to \infty$, only the correlation term remains — so reducing $\rho$ (via random subspaces) is the key gain over plain bagging.

### Gradient Boosting

Builds an additive model stage by stage. At step $m$, fit a weak learner $h_m$ to the **pseudo-residuals** (negative gradient of the loss):

$$r_{im} = -\left[\frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$$

Update:

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

### XGBoost

Adds explicit L1/L2 regularisation on the tree structure to the gradient boosting objective:

$$\mathcal{L}^{(m)} = \sum_{i=1}^n l\!\left(y_i,\ \hat{y}_i^{(m-1)} + f_m(x_i)\right) + \Omega(f_m)$$

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

where $T$ is the number of leaves and $w_j$ are leaf scores.

---

## Datasets

| Task | Dataset | Source | n | p |
|------|---------|--------|---|---|
| Classification | Adult Income | UCI / `sklearn` OpenML | 48 842 | 14 |
| Regression | California Housing | `sklearn` built-in | 20 640 | 8 |

---

## Methodology

1. **Preprocessing** — missing value imputation, ordinal/one-hot encoding, standard scaling
2. **Baseline** — dummy classifier/regressor (majority class / mean)
3. **Cross-validation** — stratified 5-fold, all metrics averaged over folds
4. **Hyperparameter tuning** — `RandomizedSearchCV` with 50 iterations
5. **Metrics**
   - Classification: Accuracy, F1-weighted, ROC-AUC, log-loss
   - Regression: RMSE, MAE, R²
6. **Interpretability** — feature importance, SHAP values, learning curves

---

## Project Structure

```
ml-benchmark/
├── src/
│   ├── preprocessing.py   # imputation, encoding, scaling pipelines
│   ├── models.py          # model definitions + hyperparameter grids
│   ├── evaluation.py      # cross-validation, metrics, learning curves
│   └── visualization.py   # comparison plots, SHAP, confusion matrices
├── notebooks/
│   ├── classification_benchmark.ipynb
│   └── regression_benchmark.ipynb
├── benchmark.py           # end-to-end run script
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python benchmark.py --task classification
python benchmark.py --task regression
```

## Results (Classification — Adult Income)

| Model | Accuracy | F1 | AUC |
|-------|----------|----|-----|
| Logistic Regression | 0.851 | 0.848 | 0.901 |
| KNN (k=9) | 0.833 | 0.829 | 0.876 |
| SVM (RBF) | 0.857 | 0.854 | 0.909 |
| Random Forest | 0.863 | 0.861 | 0.926 |
| Gradient Boosting | 0.872 | 0.870 | 0.934 |
| **XGBoost** | **0.875** | **0.873** | **0.937** |

*Results from 5-fold CV. Hyperparameters tuned via RandomizedSearchCV.*