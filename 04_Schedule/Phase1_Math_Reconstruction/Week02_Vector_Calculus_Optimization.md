# Week 2 — Vector Calculus + Continuous Optimization
**Dates:** June 1 – June 7, 2026
**Phase:** 1 — Math Reconstruction
**Book:** Mathematics for Machine Learning (MML) — Chapters 5 & 7
**Lecture Slides:** Ch05_Vector_Calculus.pdf | Ch07_Continuous_Optimization.pdf

---

## Goal This Week
Understand how to differentiate functions of vectors and matrices.
Learn how gradient descent works mathematically — this is the engine of ALL machine learning.

---

## Key Concepts to Master

### Chapter 5 — Vector Calculus
- [ ] Differentiation of univariate functions (review)
- [ ] Partial differentiation and gradients
- [ ] Gradients of vector-valued functions (Jacobian)
- [ ] Gradients of matrices (used in deep learning)
- [ ] Backpropagation as chain rule applied to computational graphs
- [ ] Automatic Differentiation — how PyTorch/TensorFlow work internally

### Chapter 7 — Continuous Optimization
- [ ] Gradient Descent — the core algorithm of ML training
- [ ] Constrained Optimization and Lagrange Multipliers
- [ ] Convex Optimization basics
- [ ] Linear Programming
- [ ] Quadratic Programming

---

## Daily Breakdown (1–2 hrs/day)

| Day | Date | Task | Time |
|-----|------|------|------|
| Mon | Jun 01 | Read MML Ch5.1–5.2 (Univariate diff, Partial diff, Gradients). | 1.5 hrs |
| Tue | Jun 02 | Read MML Ch5.3–5.4 (Jacobian, Gradients of matrices). Work examples. | 1.5 hrs |
| Wed | Jun 03 | Read MML Ch5.6 (Backpropagation + Auto Diff). Trace through the chain rule step by step. | 2 hrs |
| Thu | Jun 04 | Read MML Ch7.1 (Gradient Descent). Derive update rule from scratch on paper. | 1.5 hrs |
| Fri | Jun 05 | Read MML Ch7.2–7.3 (Lagrange Multipliers, Convex Optimization). | 1.5 hrs |
| Sat | Jun 06 | Review Ch5 + Ch7. Re-do any derivations you found hard. | 1.5 hrs |
| Sun | Jun 07 | Self-test: derive gradient descent update rule and write the chain rule for a 3-layer network. | 1 hr |

---

## Must-Understand Before Moving On
1. What is the gradient of a scalar function with respect to a vector?
2. How does the chain rule extend to multiple layers (backpropagation)?
3. Why does gradient descent converge for convex functions?

---

## Practical Tip
Backpropagation is just the chain rule applied systematically. Draw the computational graph and label each node — then compute derivatives from output to input.

---

## Progress Tracker
- [ ] Ch5 reading complete
- [ ] Ch7 reading complete
- [ ] Backprop chain rule understood
- [ ] Gradient descent derived on paper
- [ ] Self-test done on Sunday
