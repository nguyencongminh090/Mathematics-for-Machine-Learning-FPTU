# Week 6 — Gradient Descent from Scratch
**Dates:** June 29 – July 5, 2026
**Phase:** 2 — Math to Code
**Books:** d2l Ch3 + ML Cơ Bản (Vũ Hữu Tiệp)
**Tools:** Python, NumPy, Matplotlib

---

## Goal This Week
Implement gradient descent from scratch — no PyTorch, no Scikit-learn.
You should be able to watch the loss go down on a plot you made yourself.
This is the single most important algorithm to understand deeply for ML research.

---

## Key Skills to Build
- [ ] Implement batch gradient descent for a simple function
- [ ] Implement gradient descent for Linear Regression (1D and multi-dimensional)
- [ ] Plot the loss curve over iterations
- [ ] Compare different learning rates (too small, too large, just right)
- [ ] Implement Mini-batch Gradient Descent
- [ ] Understand why learning rate matters so much

---

## Daily Breakdown (1–2 hrs/day)

| Day | Date | Task | Code Goal |
|-----|------|------|-----------|
| Mon | Jun 29 | Gradient descent on a simple 1D function f(x) = x². Watch x converge to 0. | Manual derivative |
| Tue | Jun 30 | GD on 2D function f(x,y) = x² + y². Plot contour + gradient path. | `plt.contour` |
| Wed | Jul 01 | Implement Linear Regression with GD. Generate synthetic data, fit a line. | MSE loss, update rule |
| Thu | Jul 02 | Experiment with learning rates: 0.001, 0.01, 0.1, 1.0. Plot all loss curves together. | Subplots |
| Fri | Jul 03 | Read d2l Ch3 (Linear Regression). Compare your implementation to theirs. | Reading + compare |
| Sat | Jul 04 | Implement Mini-batch GD. Shuffle data, create batches, update per batch. | Batch loops |
| Sun | Jul 05 | Write a short note (10 lines): what is the difference between batch, mini-batch, and SGD? | Written summary |

---

## Starter Code Template
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data: y = 2x + 1 + noise
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Initialize parameters
w, b = 0.0, 0.0
lr = 0.01
n = len(X)

losses = []
for epoch in range(100):
    # Forward pass
    y_pred = w * X + b

    # Compute loss (MSE)
    loss = np.mean((y_pred - y) ** 2)
    losses.append(loss)

    # Compute gradients
    dw = (2/n) * np.sum((y_pred - y) * X)
    db = (2/n) * np.sum(y_pred - y)

    # Update parameters
    w -= lr * dw
    b -= lr * db

print(f"Learned: w={w:.3f}, b={b:.3f}")
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Loss Curve")
plt.show()
```

---

## Practical Tip
If your loss explodes (goes to infinity), your learning rate is too large. If it barely moves after 1000 steps, it's too small. The "Goldilocks" learning rate is the most important hyperparameter to feel intuitively.

---

## Progress Tracker
- [ ] GD on 1D function working
- [ ] GD visualized on 2D contour plot
- [ ] Linear Regression with GD implemented
- [ ] Learning rate experiment done
- [ ] d2l Ch3 read
- [ ] Mini-batch GD implemented
- [ ] Written summary of batch vs mini-batch vs SGD
