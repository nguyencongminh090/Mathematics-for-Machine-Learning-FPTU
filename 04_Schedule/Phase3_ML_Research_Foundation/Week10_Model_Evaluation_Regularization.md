# Week 10 — Model Evaluation, Overfitting & Regularization
**Dates:** July 27 – August 2, 2026
**Phase:** 3 — ML Research Foundation
**Books:** ML Cơ Bản (Vũ Hữu Tiệp) + d2l Ch4
**Tools:** Python, NumPy, Scikit-learn, Matplotlib

---

## Goal This Week
Learn how to honestly evaluate a model and prevent it from cheating by memorizing data.
This is the most practically important week for research — every paper reports these metrics.

---

## Key Concepts to Master
- [ ] Train / Validation / Test split — why and how
- [ ] Overfitting vs Underfitting (bias-variance tradeoff)
- [ ] Cross-validation (k-fold)
- [ ] Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- [ ] L1 Regularization (Lasso) — promotes sparsity
- [ ] L2 Regularization (Ridge) — penalizes large weights
- [ ] Dropout (conceptual, for neural networks)
- [ ] Learning curves: how to diagnose overfitting from plots

---

## Daily Breakdown (1–2 hrs/day)

| Day | Date | Task | Code Goal |
|-----|------|------|-----------|
| Mon | Jul 27 | Implement train/val/test split from scratch. Understand data leakage. | `np.random.shuffle` |
| Tue | Jul 28 | Demonstrate overfitting: high-degree polynomial regression on small dataset. Plot. | Degree 1 vs 10 vs 20 |
| Wed | Jul 29 | Implement k-fold cross-validation from scratch. | Manual k-fold loop |
| Thu | Jul 30 | Implement all classification metrics from scratch: accuracy, precision, recall, F1. | Confusion matrix |
| Fri | Jul 31 | Add L2 regularization to your Linear Regression. Compare weights with/without. | Modified GD loop |
| Sat | Aug 01 | Read d2l Ch4 (Multilayer Perceptrons) — focus on the regularization sections. | Reading |
| Sun | Aug 02 | Plot learning curves (train vs val loss). Practice diagnosing: overfitting? underfitting? just right? | Diagnosis exercise |

---

## Key Insight for Research
In every ML paper you will read, the model is evaluated on a held-out test set using specific metrics. Understanding exactly what those metrics mean is non-negotiable. A model with 99% accuracy can be useless if the dataset is 99% class-0.

---

## Starter Code — Overfitting Demo
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.linspace(0, 1, 10).reshape(-1, 1)
y = np.sin(2 * np.pi * X).ravel() + np.random.randn(10) * 0.1

degrees = [1, 4, 9]
X_test = np.linspace(0, 1, 100).reshape(-1, 1)

plt.figure(figsize=(12, 4))
for i, deg in enumerate(degrees):
    from numpy.polynomial.polynomial import polyfit, polyval
    coeffs = np.polyfit(X.ravel(), y, deg)
    y_pred = np.polyval(coeffs, X_test.ravel())
    plt.subplot(1, 3, i+1)
    plt.scatter(X, y, color='red', label='Data')
    plt.plot(X_test, y_pred, label=f'Degree {deg}')
    plt.title(f'Degree {deg}')
    plt.ylim(-2, 2)
    plt.legend()
plt.tight_layout()
plt.show()
```

---

## Practical Tip
When reading research papers, immediately look for: What dataset? What split? What metrics? If a paper only reports accuracy on an imbalanced dataset, be skeptical.

---

## Progress Tracker
- [ ] Train/val/test split implemented
- [ ] Overfitting demonstrated visually
- [ ] K-fold cross-validation implemented
- [ ] All classification metrics implemented
- [ ] L2 regularization added to regression
- [ ] d2l Ch4 read
- [ ] Learning curve diagnosis exercise done
