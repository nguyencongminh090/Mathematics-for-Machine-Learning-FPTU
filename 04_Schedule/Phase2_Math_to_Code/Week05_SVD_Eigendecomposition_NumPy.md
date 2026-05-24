# Week 5 — SVD & Eigendecomposition in NumPy
**Dates:** June 22 – June 28, 2026
**Phase:** 2 — Math to Code
**Books:** Dive into Deep Learning (d2l) Ch2 + ML Cơ Bản (Vũ Hữu Tiệp)
**Tools:** Python, NumPy, Jupyter Notebook

---

## Goal This Week
Phase 2 starts now. Stop reading, start coding.
Take everything from Week 1 (SVD, eigenvectors) and implement it in Python.
After this week, matrix decomposition is no longer abstract — it runs on your machine.

---

## Key Skills to Build
- [ ] Represent matrices in NumPy, perform basic operations
- [ ] Compute eigenvalues and eigenvectors with `np.linalg.eig`
- [ ] Compute SVD with `np.linalg.svd`
- [ ] Reconstruct a matrix from its SVD (low-rank approximation)
- [ ] Visualize how image compression works using SVD
- [ ] Verify results match hand calculations from Week 1

---

## Daily Breakdown (1–2 hrs/day)

| Day | Date | Task | Code Goal |
|-----|------|------|-----------|
| Mon | Jun 22 | Set up Jupyter Notebook. Review NumPy basics (arrays, matrix multiply, transpose). | `A @ B`, `A.T`, `np.linalg.norm` |
| Tue | Jun 23 | Implement eigendecomposition. Verify: `A @ v == λ * v` for each eigenpair. | `np.linalg.eig(A)` |
| Wed | Jun 24 | Implement SVD. Reconstruct the original matrix: `U @ np.diag(S) @ Vt`. | `np.linalg.svd(A)` |
| Thu | Jun 25 | Low-rank approximation: reconstruct matrix using only top-k singular values. Plot error vs k. | Loop over k values |
| Fri | Jun 26 | Image compression mini-project: load a grayscale image, apply SVD, show quality at k=5,20,50,100. | `matplotlib.pyplot.imshow` |
| Sat | Jun 27 | Read ML Cơ Bản Ch1–2 (Vũ Hữu Tiệp) for Vietnamese explanation of these concepts. | Reading |
| Sun | Jun 28 | Clean up your notebook. Add comments explaining what each line does mathematically. | Code review |

---

## Starter Code Template
```python
import numpy as np
import matplotlib.pyplot as plt

# Create a matrix
A = np.array([[3, 1], [1, 3]], dtype=float)

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Verify: A @ v == lambda * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"A@v{i} == λ{i}*v{i}:", np.allclose(A @ v, lam * v))

# SVD
U, S, Vt = np.linalg.svd(A)
A_reconstructed = U @ np.diag(S) @ Vt
print("Reconstruction error:", np.allclose(A, A_reconstructed))
```

---

## Practical Tip
Always verify your code against small hand-calculated examples first. If a 2×2 matrix gives you the wrong eigenvalues, you know something is wrong immediately.

---

## Progress Tracker
- [ ] NumPy setup and basics working
- [ ] Eigendecomposition implemented and verified
- [ ] SVD implemented and matrix reconstructed
- [ ] Low-rank approximation plot done
- [ ] Image compression mini-project complete
- [ ] Notebook cleaned and commented
