# Week 11 — Neural Networks & Backpropagation
**Dates:** August 3 – August 9, 2026
**Phase:** 3 — ML Research Foundation
**Books:** Deep Learning Cơ Bản VI (Nguyễn Thanh Tuấn) Ch1–3 + d2l Ch5
**Tools:** Python, NumPy, then PyTorch introduction

---

## Goal This Week
Build a neural network from scratch using only NumPy.
Then rebuild the same network in PyTorch.
After this week, you understand what PyTorch is actually doing under the hood.

---

## Key Concepts to Master
- [ ] Neuron: weighted sum + activation function
- [ ] Activation functions: Sigmoid, ReLU, Tanh — when to use which
- [ ] Forward pass: data flows through layers
- [ ] Loss function: cross-entropy for classification, MSE for regression
- [ ] Backward pass: backpropagation using chain rule
- [ ] Weight update with gradient descent
- [ ] PyTorch basics: tensors, autograd, `nn.Module`

---

## Daily Breakdown (1–2 hrs/day)

| Day | Date | Task | Code Goal |
|-----|------|------|-----------|
| Mon | Aug 03 | Read DL Cơ Bản Ch1 (Giới thiệu về Deep Learning). Read d2l Ch5.1. | Reading |
| Tue | Aug 04 | Implement a single neuron: `output = sigmoid(w·x + b)`. Train it with GD. | Manual neuron |
| Wed | Aug 05 | Implement a 2-layer neural network from scratch with NumPy. Forward pass only first. | Matrix ops |
| Thu | Aug 06 | Add backpropagation. Derive and implement gradients layer by layer. | Chain rule |
| Fri | Aug 07 | Train your network on XOR problem (classic test for non-linearity). | Full training loop |
| Sat | Aug 08 | Rebuild the same network in PyTorch. Use `autograd` — no manual gradients. | PyTorch basics |
| Sun | Aug 09 | Compare NumPy vs PyTorch outputs. They should match. Write what you observed. | Verification |

---

## Starter Code — 2-Layer Network (NumPy)
```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.1
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.1
b2 = np.zeros((1, 1))

lr = 0.1
for epoch in range(10000):
    # Forward
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    # Loss
    loss = np.mean((a2 - y) ** 2)

    # Backward
    da2 = 2 * (a2 - y) / len(y)
    dz2 = da2 * sigmoid_deriv(z2)
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ W2.T
    dz1 = da1 * sigmoid_deriv(z1)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    # Update
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("Predictions:", np.round(a2.T))
```

---

## Practical Tip
The XOR problem cannot be solved by Logistic Regression (it's not linearly separable). When your 2-layer network solves it perfectly, you've just proven why depth matters in neural networks — this is the key motivation for Deep Learning.

---

## Progress Tracker
- [ ] DL Cơ Bản Ch1–2 read
- [ ] Single neuron trained
- [ ] 2-layer network forward pass working
- [ ] Backpropagation implemented
- [ ] XOR problem solved
- [ ] PyTorch version working
- [ ] NumPy vs PyTorch comparison done
