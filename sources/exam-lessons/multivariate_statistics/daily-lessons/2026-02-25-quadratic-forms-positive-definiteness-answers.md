# Answer Key — Lesson 3: Quadratic Forms & Positive Definiteness
**Date:** February 25, 2026

---

## Spaced Retrieval Answers

**Q1:** x'y = 0 means the vectors are **orthogonal** (perpendicular). In statistics, orthogonal directions correspond to **uncorrelated** variables. This is why PCA seeks orthogonal components — each captures independent information.

**Q2:** A is 4×3, B is 2×4. AB: inner dimensions are 3 and 2 — DON'T match. **Cannot compute AB.** BA: B is 2×4, A is 4×3 — inner dimensions are 4 and 4. **BA exists and is 2×3.**

**Q3:** ∇f = [10x₁ − 3x₂, −3x₁ + 3x₂²]'
- ∂f/∂x₁ = 10x₁ − 3x₂
- ∂f/∂x₂ = −3x₁ + 3x₂²

**Q4:** (AB)' = **B'A'** — the order flips. Intuitively: if you reverse a sequence of transformations, you reverse each one AND do them in opposite order (like undoing layers of clothing — last on, first off).

---

## Pre-Test Commentary

**P1:** x'Ax is a **scalar**. The dimensions work out: (1×p)(p×p)(p×1) = 1×1. It generalizes ax² to multiple dimensions.

**P2:** Covariance matrices can't have negative eigenvalues because eigenvalues correspond to variances along principal directions. Variance = E[(X−μ)²] ≥ 0 by definition (it's an expected value of a squared quantity). Negative variance is nonsensical.

**P3:** "Positive definite" means the quadratic form x'Ax is **always positive** for any nonzero x. Like a bowl that always curves upward.

---

## Practice Problem Answers

### Problem 1
A = [[3,1],[1,2]], x = [2, -1]'

Step 1: Compute Ax:
```
Ax = [3·2 + 1·(-1)]   = [5]
     [1·2 + 2·(-1)]     [0]
```

Step 2: Compute x'(Ax):
```
x'Ax = [2  -1] · [5, 0]' = 2·5 + (-1)·0 = 10
```

Or by expansion: 3(2²) + 2(1)(2)(-1) + 2(-1)² = 12 - 4 + 2 = **10**

---

### Problem 2
B = [[4, 2], [2, 1]]

Characteristic equation:
det(B - λI) = (4-λ)(1-λ) - 4 = λ² - 5λ + 0 = λ(λ - 5)

Eigenvalues: **λ₁ = 5, λ₂ = 0**

Since λ₂ = 0, B is **positive semi-definite but NOT positive definite**.

This means B **cannot** be a proper covariance matrix for a non-degenerate distribution — it implies perfect linear dependence between the variables. (det(B) = 0, so B is singular and has no inverse.)

---

### Problem 3
C = [[3,1,0],[1,4,2],[0,2,5]]

**d₁ = 3** > 0 ✅

**d₂ = det([[3,1],[1,4]])** = 3·4 - 1·1 = **11** > 0 ✅

**d₃ = det(C):** Expand along row 1:
```
= 3·det([[4,2],[2,5]]) - 1·det([[1,2],[0,5]]) + 0·det([[1,4],[0,2]])
= 3·(20 - 4) - 1·(5 - 0) + 0
= 3·16 - 5
= 48 - 5 = 43
```
**d₃ = 43** > 0 ✅

All leading principal minors positive → **C is positive definite** ✅

---

### Problem 4

a) A = [[2,0],[0,8]] is diagonal, so eigenvalues are the diagonal entries: **λ₁ = 2, λ₂ = 8**

b) Semi-axis length = 1/√λ. So:
- Along x₁: 1/√2 ≈ 0.707
- Along x₂: 1/√8 ≈ 0.354

**The ellipse is longer along x₁** (the direction with the SMALLER eigenvalue gets the LONGER axis).

c) Semi-axis lengths: **1/√2 ≈ 0.707** (along x₁) and **1/√8 ≈ 0.354** (along x₂)

Note: This is counterintuitive! Larger eigenvalue = more curvature = shorter axis. Think of it as: the eigenvalue measures "stiffness" — higher stiffness = tighter curve = shorter reach.

---

### Problem 5

a) Fg:
```
F = [[1,2,0],[0,1,3]], g = [3,1,2]'

Fg = [1·3 + 2·1 + 0·2]   = [5]
     [0·3 + 1·1 + 3·2]     [7]
```
**Fg = [5, 7]'**

b) g'g = 3² + 1² + 2² = 9 + 1 + 4 = **14**

This is the **squared norm (squared length)** of g: ‖g‖² = 14, so ‖g‖ = √14 ≈ 3.742

---

### Problem 6
G = [[5,2],[2,5]]

det(G - λI) = (5-λ)² - 4 = λ² - 10λ + 21 = (λ-7)(λ-3)

Eigenvalues: **λ₁ = 7, λ₂ = 3**

Both positive → **G is positive definite** ✅

Since G is symmetric, by the Spectral Theorem, the eigenvectors are **orthogonal** to each other. (The eigenvector for λ=7 is perpendicular to the eigenvector for λ=3.)

---

### Problem 7 (Python)

```python
import numpy as np

# Create matrix
C = np.array([[3, 1, 0],
              [1, 4, 2],
              [0, 2, 5]])

# Eigenvalues (eigvalsh for symmetric matrices)
eigenvalues = np.linalg.eigvalsh(C)
print("Eigenvalues:", eigenvalues)

# Check positive definiteness
is_pd = np.all(eigenvalues > 0)
print(f"Positive definite: {is_pd}")

# Bonus: Cholesky decomposition
try:
    L = np.linalg.cholesky(C)
    print("Cholesky succeeded → C is PD")
    print("L =\n", L)
except np.linalg.LinAlgError:
    print("Cholesky failed → C is NOT PD")
```

**Expected output:**
```
Eigenvalues: [1.324  3.274  7.402]  (approximately)
Positive definite: True
Cholesky succeeded → C is PD
```

(Exact eigenvalues are roots of λ³ - 12λ² + 43λ - 43 = 0)

---

*Answer key. Don't peek!* 🏔️
