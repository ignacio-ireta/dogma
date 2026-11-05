# ANSWER KEY — MVS Lesson 6
**Date:** March 10, 2026 | **Topic:** Generalized Variance & The Wishart Distribution
**⚠️ Answer key — attempt all exercises before consulting this file**

---

## WARM-UP ANSWERS

**WU1:**
- (a) **Mahalanobis distance** (squared). It measures the distance from x to μ scaled by the shape of the distribution — accounts for variance and correlation. Contours of constant Mahalanobis distance are **ellipses** in 2D; the shape/orientation is controlled by Σ.
- (b) When p=1: the exponent becomes −½ (x−μ)²/σ², the determinant term becomes σ⁻¹, and the normalizing constant gives 1/√(2πσ²). This is exactly the familiar N(μ,σ²) density.
- (c) Contours form **ellipses** (2D). Shape and orientation controlled by Σ (specifically its eigenvectors give the axes, eigenvalues give the lengths).

**WU2:**
- (a) FALSE — zero covariance is not enough in general for independence; however under MVN it IS sufficient.
- (b) TRUE — Under multivariate normality, Cov(X₁,X₂)=0 ⟺ X₁,X₂ are independent. (This is special to MVN; not true in general.)
- (c) TRUE — Any linear combination of a MVN vector is univariate normal (from L4).

**WU3:**
- (a) A is positive definite if **xᵀAx > 0 for all x ≠ 0**
- (b) Two equivalent tests: (1) All eigenvalues > 0. (2) All leading principal minors are positive (Sylvester's criterion: det of each upper-left k×k submatrix > 0 for k=1,...,p).
- (c) Σ must be positive definite for the MVN density to be well-defined (|Σ| > 0 in the denominator) and for Mahalanobis distances to be finite positive. A singular Σ means the distribution is degenerate (lives on a lower-dimensional subspace).

**WU4:** "cosine" — r_{jk} = cos(θ_{jk}) where θ is the angle between deviation vectors.

---

## PRE-TEST ANSWERS (for reference)

**PT1:** |S| measures the volume² of the data cloud in p-dimensional space — the "total spread" accounting for all pairwise covariances. It CANNOT be negative (determinants of positive semidefinite matrices are ≥ 0). It CAN be zero when variables are linearly dependent.

**PT2:** The Wishart distribution — a matrix-valued random variable. It's what you get when you take outer products of multivariate normal vectors and sum them up.

**PT3:** x̄ ~ Nₚ(μ, Σ/n). Increasing n makes it LESS spread out (covariance Σ/n shrinks).

---

## PRACTICE ANSWERS

**P1:**
- (a) |S| = 9·9 − 6·6 = 81 − 36 = **45**
- (b) s₁ = √9 = 3, s₂ = √9 = 3
- (c) |S| = s₁²·s₂²·|R| → 45 = 9·9·|R| → |R| = 45/81 = 5/9 ≈ 0.556
  For a 2×2 correlation matrix: |R| = 1 − r₁₂² → 5/9 = 1 − r₁₂² → r₁₂² = 4/9 → **r₁₂ = 2/3 ≈ 0.667**
- (d) Yes — s₁₂ = 6, s₁ = s₂ = 3, so r = 6/(3·3) = 6/9 = 2/3 ≈ 0.667. ✓ Makes sense: positive covariance, moderate positive correlation.

**P2:**
- (a) np.linalg.det(S) = 45.0 ✓ matches P1
- (b) np.linalg.eigvals(S) = [15., 3.] (eigenvalues of [[9,6],[6,9]] are 9±6 = 15 and 3)
  Product = 15 × 3 = 45 = |S| ✓
  Why always true: det(A) = product of all eigenvalues (a fundamental property of determinants/characteristic polynomial).

**P3:**
- (a) (n−1)S ~ W₃(19, Σ) — Wishart with p=3, degrees of freedom = n−1 = 19, scale Σ
- (b) Degrees of freedom = n−1 = 19, not n=20, because we estimated μ with x̄ — that estimation "uses up" one degree of freedom.
- (c) In the univariate case, x̄ and s² are independent by **Cochran's theorem** (sometimes called the Fisher–Cochran theorem). The multivariate version is an extension.

**P4:**
- (a) E[x̄] = μ = [3, 7]
- (b) Cov(x̄) = (1/n)Σ = (1/25)·[[4,2],[2,1]] = [[0.16, 0.08],[0.08, 0.04]]
- (c) np.linalg.eigvals([[4,2],[2,1]]) → eigenvalues are 5 and 0 (trace=5, det = 4·1−2·2 = 0 → one eigenvalue is 0).
  Σ is **positive SEMI-definite**, not positive definite. This means the distribution is degenerate — X₁ and X₂ are perfectly linearly related (X₂ = X₁/2, since [[4,2],[2,1]] has rank 1). The joint distribution lives on a line, not in full 2D space.

**P5:**
- (a) tr(S) = 5 + 3 + 0.5 = **8.5**
- (b) |S| = 5 × 3 × 0.5 = **7.5**
- (c) λ₁/tr = 5/8.5 ≈ **58.8%**, λ₂/tr = 3/8.5 ≈ **35.3%**, λ₃/tr = 0.5/8.5 ≈ **5.9%**
- (d) Retaining first two: (5+3)/8.5 = 8/8.5 ≈ **94.1%** of total variance retained.

**P6:**
- (a) T ~ **χ²(p) = χ²(2)**
- (b) T = 8.5 falls between the 95th (5.99) and 99th (9.21) percentile → this is in the top 5% but not top 1%. It's unusual but not extreme — we'd reject at α=0.05 but not at α=0.01.
- (c) A large T means the sample mean x̄ is far from the hypothesized μ in Mahalanobis distance — evidence that the true mean is NOT μ (i.e., we'd reject H₀: μ_true = μ in a hypothesis test).

**P7:**
- (a) As n→∞, S converges (in probability) to the true population covariance matrix **Σ**. (Law of large numbers for matrices — S is a consistent estimator of Σ.)
- (b) When n < p, you only have n−1 < p linearly independent observations — the sample covariance matrix S has rank at most n−1 < p, so it's **singular** (|S| = 0). You can't invert S — which means many multivariate methods that require S⁻¹ (like Hotelling's T², discriminant analysis) break down entirely.
- (c) "The Wishart distribution is to covariance matrices (p×p random matrices) what the chi-squared distribution is to variances (scalar positive random variables)."
