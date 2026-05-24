# Week 8 — Linear Regression & PCA from Scratch in Python
**Dates:** July 13 – July 19, 2026
**Phase:** 2 — Math to Code
**Books:** ML Cơ Bản (Vũ Hữu Tiệp) + d2l
**Tools:** Python, NumPy, Matplotlib, Scikit-learn (for verification only)

---

## Goal This Week
Build complete, working implementations of Linear Regression and PCA from scratch.
After this week, you understand both models deeply enough to explain, modify, and debug them — a critical skill for research.

---

## Key Skills to Build
- [ ] Linear Regression: both Normal Equation AND Gradient Descent solutions
- [ ] Evaluate model with MSE, R² score
- [ ] PCA from scratch using eigendecomposition
- [ ] PCA for dimensionality reduction and visualization (2D → plot)
- [ ] Verify both implementations against scikit-learn results
- [ ] Understand when to use Normal Equation vs Gradient Descent

---

## Daily Breakdown (1–2 hrs/day)

| Day | Date | Task | Code Goal |
|-----|------|------|-----------|
| Mon | Jul 13 | Linear Regression with Normal Equation: `w = (XᵀX)⁻¹Xᵀy`. | `np.linalg.inv` |
| Tue | Jul 14 | Linear Regression with Gradient Descent (from Week 6). Compare both solutions. | Side-by-side comparison |
| Wed | Jul 15 | Evaluate model: implement MSE and R² from scratch. Plot predictions vs ground truth. | Metrics |
| Thu | Jul 16 | PCA from scratch: center data → compute covariance → eigendecompose → project. | Full pipeline |
| Fri | Jul 17 | Apply PCA to the Iris dataset (4D → 2D). Visualize clusters. | `sklearn.datasets.load_iris` |
| Sat | Jul 18 | Verify your PCA against `sklearn.decomposition.PCA`. Check explained variance ratio. | Comparison |
| Sun | Jul 19 | Phase 2 self-assessment: run all 4 weeks of code. Do they all still work? Write what you learned. | Review session |

---

## Starter Code — PCA from Scratch
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X = iris.data  # shape: (150, 4)
y = iris.target

# Step 1: Center the data
X_centered = X - X.mean(axis=0)

# Step 2: Compute covariance matrix
cov_matrix = np.cov(X_centered.T)  # shape: (4, 4)

# Step 3: Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Sort by descending eigenvalue
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]

# Step 5: Project to 2D
X_pca = X_centered @ eigenvectors[:, :2]

# Plot
colors = ['red', 'green', 'blue']
for i, label in enumerate(iris.target_names):
    mask = y == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, c=colors[i])
plt.legend()
plt.title("PCA of Iris Dataset (from scratch)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
```

---

## Phase 2 Completion Checklist
By end of Week 8 you should be able to:
- [ ] Implement Linear Regression two ways (Normal Eq + GD)
- [ ] Implement PCA from scratch
- [ ] Implement Naive Bayes classifier
- [ ] Implement gradient descent on any differentiable function
- [ ] All notebooks run without errors

---

## Practical Tip
When verifying against scikit-learn, don't just check if the numbers match — understand WHY they match. If there's a sign flip in eigenvectors, that's normal (eigenvectors have arbitrary sign). Understand edge cases like this.

---

## Progress Tracker
- [ ] Linear Regression (Normal Eq) done
- [ ] Linear Regression (GD) done
- [ ] MSE + R² metrics implemented
- [ ] PCA from scratch working
- [ ] PCA on Iris dataset visualized
- [ ] Verified against sklearn
- [ ] Phase 2 self-assessment complete
