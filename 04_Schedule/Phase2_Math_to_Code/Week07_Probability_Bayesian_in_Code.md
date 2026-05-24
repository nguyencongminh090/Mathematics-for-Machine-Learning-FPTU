# Week 7 — Probability & Bayesian Thinking in Code
**Dates:** July 6 – July 12, 2026
**Phase:** 2 — Math to Code
**Books:** ML Cơ Bản (Vũ Hữu Tiệp) + MML Ch6 (review)
**Tools:** Python, NumPy, SciPy, Matplotlib

---

## Goal This Week
Translate probability theory into working code.
Simulate distributions, apply Bayes' theorem numerically, and build intuition
for how probabilistic thinking underlies all ML models.

---

## Key Skills to Build
- [ ] Sample from common distributions (Gaussian, Bernoulli, Uniform)
- [ ] Visualize PDFs and CDFs
- [ ] Implement Bayes' theorem numerically
- [ ] Compute MLE (Maximum Likelihood Estimation) for Gaussian parameters
- [ ] Compute mean, variance, covariance matrix from data
- [ ] Understand why MSE loss assumes Gaussian noise (probabilistic interpretation)

---

## Daily Breakdown (1–2 hrs/day)

| Day | Date | Task | Code Goal |
|-----|------|------|-----------|
| Mon | Jul 06 | Sample from Gaussian, plot histogram vs true PDF. | `np.random.randn`, `scipy.stats.norm` |
| Tue | Jul 07 | Implement Bayes' theorem: P(A\|B) = P(B\|A)*P(A)/P(B). Spam filter example. | Dictionary / numpy |
| Wed | Jul 08 | MLE for Gaussian: estimate μ and σ from data. Compare to `np.mean`, `np.std`. | Manual MLE derivation |
| Thu | Jul 09 | Compute covariance matrix from scratch. Compare to `np.cov`. | Matrix operations |
| Fri | Jul 10 | Plot 2D Gaussian: contour plot of a bivariate normal distribution. | `scipy.stats.multivariate_normal` |
| Sat | Jul 11 | Read ML Cơ Bản — sections on Naive Bayes classifier. | Reading |
| Sun | Jul 12 | Mini-project: Naive Bayes classifier from scratch on a toy dataset. | Full implementation |

---

## Starter Code — Bayes' Theorem
```python
import numpy as np

# Example: Medical test
# P(disease) = 0.01
# P(positive | disease) = 0.95
# P(positive | no disease) = 0.05

p_disease = 0.01
p_pos_given_disease = 0.95
p_pos_given_no_disease = 0.05

# P(positive) = total probability
p_positive = p_pos_given_disease * p_disease + \
             p_pos_given_no_disease * (1 - p_disease)

# Bayes: P(disease | positive)
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_positive

print(f"P(disease | positive test) = {p_disease_given_pos:.4f}")
# Result: ~0.16 — even with a positive test, only 16% chance of disease!
```

---

## Practical Tip
The Naive Bayes result above is counterintuitive to most people — a positive test only means 16% chance of having the disease. This is why probabilistic thinking is so important. Don't trust intuition; trust the math.

---

## Progress Tracker
- [ ] Distribution sampling and plotting done
- [ ] Bayes' theorem implemented numerically
- [ ] MLE for Gaussian implemented
- [ ] Covariance matrix computed from scratch
- [ ] 2D Gaussian plotted
- [ ] ML Cơ Bản Naive Bayes section read
- [ ] Naive Bayes classifier mini-project done
