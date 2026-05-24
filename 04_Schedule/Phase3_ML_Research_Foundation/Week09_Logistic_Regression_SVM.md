# Week 9 — Logistic Regression & SVM Basics
**Dates:** July 20 – July 26, 2026
**Phase:** 3 — ML Research Foundation
**Books:** Deep Learning Cơ Bản (Nguyễn Thanh Tuấn) + ML Cơ Bản (Vũ Hữu Tiệp)
**Tools:** Python, NumPy, Scikit-learn

---

## Goal This Week
Phase 3 begins — you are now building research-level ML knowledge.
Logistic Regression is the foundation of classification. SVM introduces the concept of
margin maximization, which appears in many modern ML methods.

---

## Key Concepts to Master
- [ ] Sigmoid function and why it maps any value to (0, 1)
- [ ] Binary Cross-Entropy loss (log loss) — derive from MLE
- [ ] Logistic Regression training with Gradient Descent
- [ ] Decision boundary: what does the model actually learn?
- [ ] SVM: maximum margin classifier concept
- [ ] Hard-margin vs Soft-margin SVM (C parameter)
- [ ] Kernel trick (conceptual understanding)

---

## Daily Breakdown (1–2 hrs/day)

| Day | Date | Task | Code Goal |
|-----|------|------|-----------|
| Mon | Jul 20 | Read ML Cơ Bản — Logistic Regression chapter. Understand sigmoid + binary cross-entropy. | Reading |
| Tue | Jul 21 | Implement Logistic Regression from scratch with GD. Binary classification task. | Full implementation |
| Wed | Jul 22 | Plot decision boundary on 2D synthetic data. Visualize how the boundary shifts as training proceeds. | Visualization |
| Thu | Jul 23 | Extend to multi-class: Softmax + Cross-Entropy. | Softmax implementation |
| Fri | Jul 24 | Read ML Cơ Bản — SVM chapter. Understand margin, support vectors, kernel trick. | Reading |
| Sat | Jul 25 | Use `sklearn.svm.SVC` on a dataset. Experiment with kernels: linear, RBF, polynomial. | sklearn experiment |
| Sun | Jul 26 | Write: what is the key difference between Logistic Regression and SVM? (1 paragraph each in Vietnamese and English) | Written summary |

---

## Starter Code — Logistic Regression
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-9
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.losses = []
        for _ in range(self.n_iter):
            z = X @ self.w + self.b
            y_pred = sigmoid(z)
            loss = binary_cross_entropy(y, y_pred)
            self.losses.append(loss)
            dw = X.T @ (y_pred - y) / len(y)
            db = np.mean(y_pred - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return (sigmoid(X @ self.w + self.b) >= 0.5).astype(int)
```

---

## Practical Tip
For research, you need to know BOTH: how to implement from scratch (to understand) AND how to use sklearn efficiently (to experiment quickly). This week practice both.

---

## Progress Tracker
- [ ] Logistic Regression theory understood
- [ ] Logistic Regression from scratch working
- [ ] Decision boundary visualized
- [ ] Softmax for multi-class implemented
- [ ] SVM concept understood (margin, kernels)
- [ ] sklearn SVM experiment done
- [ ] Written comparison of LR vs SVM
