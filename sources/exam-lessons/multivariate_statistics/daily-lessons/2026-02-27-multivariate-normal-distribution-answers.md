# MVS Lesson 4 — Answer Key
**Date:** February 27, 2026 | **Topic:** Multivariate Normal Distribution

---

## Spaced Retrieval Answers

**Q1:** u'v = 1·2 + 0·5 + 3·(−1) = 2 + 0 − 3 = −1. Not orthogonal (would need u'v = 0).

**Q2:** det(A) = 0 means: (1) A is singular — no inverse exists. (2) A collapses space — maps some nonzero vector to zero (null space is nontrivial). (3) The system Ax = b either has no solution or infinitely many. (4) At least one eigenvalue is zero. (5) The columns of A are linearly dependent.

**Q3:** det(M − λI) = (6−λ)(3−λ) − 4 = λ² − 9λ + 14 = 0. λ = (9 ± √(81−56))/2 = (9 ± 5)/2. So λ₁ = 7, λ₂ = 2.

**Q4:** AB is 3×5. BA requires 5×3 times 3×2... wait, B is 2×5, so BA would need (2×5)(3×2) — dimensions don't match. BA is not defined.

## Pre-Test — Expected Answers

**P1:** Mean becomes a **mean vector** μ (p×1), variance becomes a **covariance matrix** Σ (p×p).

**P2:** Ellipses (or ellipsoids in higher dimensions) — the level curves of quadratic forms.

**P3:** Yes — marginal distributions of MVN are always normal.

---

## Practice Problem Answers

### Problem 1 — Mahalanobis distance

Σ = [[1, 0.5], [0.5, 1]]

det(Σ) = 1·1 − 0.5·0.5 = 0.75

Σ⁻¹ = (1/0.75) · [[1, −0.5], [−0.5, 1]] = [[4/3, −2/3], [−2/3, 4/3]]

x − μ = [1, 1]'

d² = [1, 1] · [[4/3, −2/3], [−2/3, 4/3]] · [1, 1]'
   = [1, 1] · [4/3 − 2/3, −2/3 + 4/3]'
   = [1, 1] · [2/3, 2/3]'
   = 2/3 + 2/3 = **4/3 ≈ 1.333**

### Problem 2 — Marginal distributions

a) X₂ ~ N(−1, 3). Mean = μ₂ = −1. Variance = Σ₂₂ = 3.

b) [X₁, X₃]' ~ N₂([2, 5]', Σ₁₃) where:
```
Σ₁₃ = [Σ₁₁  Σ₁₃] = [4   -1]
      [Σ₃₁  Σ₃₃]   [-1   2]
```

### Problem 3 — Linear combinations

a = [1, 1]'

a) E[Y] = a'μ = 1·10 + 1·20 = **30**

b) Var(Y) = a'Σa = [1, 1] · [[9, 3], [3, 16]] · [1, 1]'
   = [1, 1] · [9+3, 3+16]' = [1, 1] · [12, 19]' = 12 + 19 = **31**

c) Y ~ **N(30, 31)**

### Problem 4 — Independence and correlation

a) Yes, X₁ and X₂ are independent. Since they are jointly MVN and their covariance is 0 (Σ is diagonal), uncorrelated ⟺ independent for MVN.

b) No! Without the MVN assumption, uncorrelated does NOT imply independent. Zero covariance only means no LINEAR relationship. There could be nonlinear dependencies.

### Problem 5 — Determinant and inverse

det(A) = 3·5 − 1·2 = 15 − 2 = **13**

A⁻¹ = (1/13) · [[5, −1], [−2, 3]] = **[[5/13, −1/13], [−2/13, 3/13]]**

Verify: AA⁻¹ = (1/13)·[[3·5+1·(−2), 3·(−1)+1·3], [2·5+5·(−2), 2·(−1)+5·3]] = (1/13)·[[13, 0], [0, 13]] = I ✅

### Problem 6 — Matrix multiplication

```
[1  0  2] [4]   [1·4 + 0·1 + 2·2]   [8]
[3  1  0] [1] = [3·4 + 1·1 + 0·2] = [13]
           [2]
```

### Problem 7 — Python code

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Define parameters
mu = np.array([3, 1])
sigma = np.array([[4, 1], [1, 2]])

# Generate 1000 samples
np.random.seed(42)
samples = np.random.multivariate_normal(mu, sigma, 1000)

# Eigendecomposition for the ellipse
eigenvalues, eigenvectors = np.linalg.eigh(sigma)

# Chi-squared critical value for 95% confidence, 2 df
chi2_95 = 5.991

# Ellipse parameters
angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
width = 2 * np.sqrt(eigenvalues[1] * chi2_95)   # major axis
height = 2 * np.sqrt(eigenvalues[0] * chi2_95)   # minor axis

# Plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, label='Samples')
ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle,
                  edgecolor='red', facecolor='none', linewidth=2,
                  label='95% confidence ellipse')
ax.add_patch(ellipse)
ax.plot(*mu, 'r+', markersize=15, markeredgewidth=2, label='μ')
ax.set_xlabel('X₁')
ax.set_ylabel('X₂')
ax.set_title('Bivariate Normal: N₂(μ, Σ)')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mvn_bivariate.png', dpi=150)
plt.show()
```

The scatter plot should show an elongated cloud tilted slightly (due to positive correlation Σ₁₂ = 1), with the red ellipse containing roughly 95% of the points.
