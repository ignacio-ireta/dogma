# MVS Lesson 5 — ANSWER KEY
**Date:** March 9, 2026 | **Topic:** Sample Geometry & Random Sampling
**Answer key — attempt the exercises before reading this file.**

---

## SPACED RETRIEVAL ANSWERS

**WU1:** The term (x − μ)ᵀ Σ⁻¹ (x − μ) is the **Mahalanobis distance** (squared). It measures how far x is from the mean μ in units that account for the covariance structure — it "stretches" space so that the natural spread of the distribution is removed. In 2D, the contours of constant Mahalanobis distance are **ellipses** (axis-aligned only when Σ is diagonal; rotated in general). These are the "equal-density contours" of the MVN.

**WU2:** A matrix A is positive definite if **xᵀAx > 0 for all x ≠ 0**, which is equivalent to all eigenvalues being strictly positive. For λ₁ = 4, λ₂ = 1, λ₃ = −0.5: **NOT positive definite** — the negative eigenvalue λ₃ = −0.5 fails the condition. (It's indefinite.)

**WU3:** det(A − λI) = det([[5−λ, 1],[1, 5−λ]]) = (5−λ)² − 1 = 0
→ (5−λ)² = 1 → 5−λ = ±1 → **λ = 4 or λ = 6**
Both positive → A is positive definite. (Makes sense: it's a covariance-like symmetric matrix.)

**WU4:**
(a) **TRUE.** Any marginal distribution of a multivariate normal is itself normal. The marginal Xᵢ ~ N(μᵢ, σᵢᵢ).
(b) **TRUE.** This is a special property unique to the multivariate normal — for MVN, zero correlation implies independence. (This is NOT true for general distributions!)

---

## PRE-TEST ANSWERS (expected approximate responses)

**PT1:** A 50×3 matrix (50 rows = people, 3 columns = height/weight/age). This is the standard convention.

**PT2:** The covariance matrix would be 3×3 (p×p, not n×p). Diagonal entries are the variances of height, weight, and age. Off-diagonal entries are the covariances between pairs.

**PT3:** Yes — each column (variable) is a vector in ℝ⁵⁰. The angle between two such vectors (after centering) relates to their correlation. Small angle (close to 0°) → high positive correlation. Right angle (90°) → zero correlation. Obtuse angle → negative correlation.

---

## PRACTICE PROBLEM ANSWERS

---

### P1

**(a) Sample mean vector:**
- x̄₁ = (2 + 4 + 6) / 3 = 12/3 = **4**
- x̄₂ = (8 + 6 + 4) / 3 = 18/3 = **6**
- x̄ = [4, 6]

**(b) NumPy:**
```python
import numpy as np
X = np.array([[2,8],[4,6],[6,4]])
x_bar = X.mean(axis=0)
# Result: array([4., 6.])
```
`axis=0` means "average down the rows" (i.e., within each column).

---

### P2

**(a)** As variable 1 increases (2 → 4 → 6), variable 2 decreases (8 → 6 → 4). So s₁₂ should be **negative** — they move in opposite directions.

**(b)** The relationship looks perfectly linear and perfectly inversely proportional, so r₁₂ ≈ **−1** (exactly −1, in fact).

**Verification:**
```python
np.cov(X.T)
# Off-diagonal = -4.0, so r = -4 / sqrt(4 * 4) = -4/4 = -1.0
```

---

### P3

**(a)** |S| = λ₁ × λ₂ = 9 × 1 = **10**
(The determinant of a symmetric matrix equals the product of its eigenvalues.)

**(b)** tr(S) = λ₁ + λ₂ = 9 + 1 = **10**
(Trace = sum of eigenvalues = sum of diagonal entries.)

**(c)** Fraction = λ₁ / (λ₁ + λ₂) = 9 / 10 = **90%**
This is exactly what PCA does: finds the eigenvalue decomposition of S and reports what fraction of total variance each principal component explains. The first PC here captures 90% of the variation.

---

### P4

**(a)**
```python
import numpy as np
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])
S = np.cov(X, rowvar=False)
```
`rowvar=False` tells NumPy that **rows are observations and columns are variables** (the standard statistical convention). Without it, NumPy defaults to treating rows as variables.

**(b) Output:**
```
[[ 15.  15.  15.]
 [ 15.  15.  15.]
 [ 15.  15.  15.]]
```
Surprising observation: all covariances are **identical** (= 15). The covariance matrix is rank 1 (all rows are the same). This means **|S| = 0** — the three variables are perfectly linearly related: X₃ = X₂ + 1 = X₁ + 2, so knowing one tells you all three. No new information in variables 2 or 3.

---

### P5

**(a) Quick shortcut tests for 2×2 positive definiteness:**
- Diagonal entries > 0: s₁₁ = 4 > 0 ✓, s₂₂ = 1 > 0 ✓
- det(Σ) = (4)(1) − (2)(2) = 4 − 4 = **0**

Since det = 0, the matrix is **positive semidefinite (not positive definite)**. One eigenvalue is 0.

Eigenvalues: λ₁ + λ₂ = tr = 5, λ₁ × λ₂ = det = 0 → **λ = 5 and λ = 0**.

**(b)** A zero eigenvalue means the covariance matrix is **singular** — it cannot be inverted. This means the bivariate normal density formula (which requires Σ⁻¹) **does not exist** in the standard form.

Geometrically: all the probability mass lies on a **line** in 2D space (a degenerate distribution). X₁ and X₂ are perfectly linearly related: X₂ = (1/2)X₁ (you can verify: if Σx = 0x, find the null vector).

This is a degenerate multivariate normal — technically valid as a distribution, but all mass is concentrated in a lower-dimensional subspace.

---

### P6

**(a) Deviation vectors:**
- ȳ₁ = (1+3+5+7)/4 = 4; d₁ = [1−4, 3−4, 5−4, 7−4] = **[−3, −1, 1, 3]**
- ȳ₂ = (2+4+6+8)/4 = 5; d₂ = [2−5, 4−5, 6−5, 8−5] = **[−3, −1, 1, 3]**

(Notice d₁ = d₂ — they're the same vector!)

**(b) Dot product and lengths:**
- d₁ · d₂ = (−3)(−3) + (−1)(−1) + (1)(1) + (3)(3) = 9 + 1 + 1 + 9 = **20**
- ||d₁|| = √(9 + 1 + 1 + 9) = √20 = 2√5
- ||d₂|| = √20 = 2√5

**(c) Correlation:**
r = 20 / (√20 × √20) = 20 / 20 = **1.0**

Makes perfect sense: y₂ = y₁ + 1 always (adding a constant doesn't change correlation). They move in perfect lockstep → r = 1, θ = 0°.

---

### P7

**(a)** X₂ = 1.1 × X₁ always → **|S| = 0**. The deviation vectors are parallel (perfect correlation r = 1). The "data cloud" is a line (no area/volume). Geometrically: the matrix is rank-deficient.

**(b)** Moderate correlation but not perfect → **|S| is small but positive**. Both variables carry some independent information but also share information (r ≈ 0.5). The determinant: |S| = s₁²s₂²(1 − r²) = s₁²s₂² × 0.75. Smaller than the uncorrelated case but not zero.

**(c)** The third variable (height) is nearly uncorrelated with the first two → **|S| is relatively large**. Three variables carrying mostly independent information = large "data volume". The generalized variance grows because each new orthogonal variable adds a new dimension to the data cloud.

---

*Answer key complete. Next lesson: Generalized Variance (deeper) + Wishart Distribution.*
