# MVS Lesson 7 (Patch): Singular Value Decomposition
**Date:** May 7, 2026 | **Topic:** SVD — The Most Important Decomposition You Haven't Met Yet
**Syllabus position:** Foundation patch — required before PCA (L8)

---

## 1. SPACED RETRIEVAL WARM-UP (10 min)

*Pulling from Lessons 2, 3, 4 (eigenvalues, spectral theorem, MVN geometry). ~2 months ago. The struggle IS the learning.*

**WU1 (L2 — Feb 24):** State the spectral theorem for a symmetric matrix A. What does it guarantee about eigenvalues and eigenvectors? Write the decomposition A = QΛQᵀ and explain what Q and Λ contain.

**WU2 (L2 — Feb 24):** If A is a p×p symmetric matrix with eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₚ, how do you find:
- (a) The rank of A?
- (b) Whether A is positive definite?

**WU3 (L3 — Feb 25):** A quadratic form xᵀAx with A positive definite traces out an ellipsoid. What do the eigenvalues of A tell you about the axes of that ellipsoid?

**WU4 (L4 — Feb 27):** The Mahalanobis distance is (x − μ)ᵀ Σ⁻¹ (x − μ). In 2D, this equals a constant on the boundary of an ellipse. How does this connect to the eigenvalues of Σ?

---

## 2. PRE-TEST: Singular Value Decomposition (5 min)

*Before I teach — guess. Wrong answers prime the brain.*

**PT1:** The spectral theorem says symmetric matrices decompose as QΛQᵀ. But what about non-symmetric matrices? Or even rectangular ones (like a 100×3 data matrix)? Do you think there's a decomposition that works for ALL matrices? What might it look like?

**PT2:** If X is an n×p data matrix (n observations, p variables), rank is at most min(n, p). What do you think "rank" means geometrically? If the rows of X all lie on a flat plane in p-dimensional space, what's the rank?

**PT3:** The spectral decomposition uses one set of orthonormal vectors (the eigenvectors). If a decomposition worked for rectangular matrices, would you expect it to need one orthonormal basis, or two separate ones? Why?

---

## 3. EXPLICIT INSTRUCTION (30 min)

### 3.1 The Big Idea: One Decomposition to Rule Them All

The spectral theorem is beautiful but limited — it only works for **symmetric square** matrices. Real data matrices are neither symmetric nor square.

**The Singular Value Decomposition (SVD)** fixes this. Every matrix — every single one, regardless of shape, rank, or symmetry — can be written as:

```
X = U D Vᵀ
```

Where:
- **X** is n×p (your data: n observations, p variables)
- **U** is n×n, orthogonal (UᵀU = I) — the "left singular vectors"
- **D** is n×p, diagonal (only diagonal entries, called "singular values" σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0)
- **V** is p×p, orthogonal (VᵀV = I) — the "right singular vectors"

**Geometrically:** SVD says any linear transformation X can be decomposed into three steps:
1. **Rotate** (Vᵀ) — align to a new coordinate system in the input space
2. **Scale** (D) — stretch/compress along axes by singular values
3. **Rotate** (U) — align to the output space

This is the most general form of "matrix as transformation."

### 3.2 SVD vs. Spectral Decomposition — How They Connect

You already know the spectral decomposition: for symmetric A,

```
A = Q Λ Qᵀ
```

SVD is the generalization. Here's the key connection:

**If A is symmetric and positive semi-definite:**
- SVD and spectral decomposition are **identical**
- U = V = Q (same orthonormal basis)
- Singular values = eigenvalues: σᵢ = λᵢ

**If A is symmetric but has negative eigenvalues:**
- SVD singular values are the **absolute values** of eigenvalues: σᵢ = |λᵢ|
- The sign information moves into U or V

**If A is not symmetric:**
- Spectral decomposition doesn't apply
- SVD still works — U and V are **different** orthonormal bases

**Why this matters for MVS:** The covariance matrix S = XᵀX/(n-1) is symmetric and PSD. So for S, SVD = spectral decomposition. Same eigenvectors, same eigenvalues. But we'll often compute SVD on the raw data X directly — and that's actually *better* numerically than forming XᵀX first.

### 3.3 The Rank Connection

The number of **nonzero singular values** = the **rank** of X.

```
rank(X) = number of σᵢ > 0
```

Geometrically: the rows of X live in a subspace of dimension rank(X). If σ₃ = 0, then all your data points lie on a 2D plane, no matter how many columns X has.

**This is why PCA works:** We're looking for which singular values are large (signal) and which are small (noise/dimensionality that doesn't matter).

### 3.4 The Truncated SVD — Compression

If X has rank r but we keep only the first k < r terms:

```
Xₖ = Uₖ Dₖ Vₖᵀ
```

This is the **best rank-k approximation** to X (in the Frobenius norm sense — the Eckart-Young theorem guarantees this). No other rank-k matrix comes closer to X.

**This IS PCA.** When we do PCA and keep k components, we are computing the truncated SVD of the centered data matrix.

### 3.5 Key Formulas to Know

```
# SVD
X = UDVᵀ

# From SVD to covariance eigenvalues
XᵀX = VD²Vᵀ    → eigenvalues of XᵀX are σᵢ²

# Singular values and eigenvalues
σᵢ² = λᵢ(XᵀX) = (n-1) × λᵢ(S)   where S = sample covariance
```

So: eigenvalues of the covariance matrix = singular values² of the centered data, divided by (n-1).

---

## 4. INTERLEAVED PRACTICE (40 min)

### Conceptual

**P1:** Explain in one sentence why SVD is "more general" than the spectral decomposition.

**P2:** A 200×5 data matrix X has singular values σ₁ = 12.3, σ₂ = 8.1, σ₃ = 3.2, σ₄ = 0.01, σ₅ = 0.005.
- (a) What is the effective rank of this data?
- (b) If you keep 2 components, what fraction of total variance do you capture? (Total variance = sum of all σᵢ²)
- (c) Should you keep 2 or 3 components? What would help you decide?

**P3:** True or false, and why:
- (a) "The left singular vectors U and right singular vectors V are always the same."
- (b) "For a symmetric matrix, SVD and eigendecomposition give the same vectors."
- (c) "Singular values can be negative."

### Computation

**P4:** Compute the SVD of:
```
X = [3  0]
    [0  2]
```
(This is already diagonal, so U = ?, D = ?, V = ?)

**P5:** Given XᵀX = [5  0]
                   [0  1]
- (a) What are the eigenvalues of XᵀX?
- (b) What are the singular values of X?
- (c) If X is 50×2, what are the eigenvalues of the sample covariance S?

### Connection

**P6:** In Lesson 3, you learned that xᵀAx traces an ellipsoid when A is PD, and the eigenvalues determine the axis lengths. Now connect this to SVD: if A = VD²Vᵀ, what determines the shape of the ellipsoid? What determines the orientation?

---

## 5. SUMMARY + ANKI SUGGESTIONS (5 min)

### Key Takeaways
1. **SVD works for every matrix** — symmetric, non-symmetric, square, rectangular. No exceptions.
2. **X = UDVᵀ** — two orthogonal bases (U, V) connected by singular values (diagonal of D).
3. **SVD ⊇ spectral decomposition** — for symmetric PSD matrices, they're identical.
4. **Truncated SVD = best low-rank approximation** — this is the mathematical heart of PCA.
5. **σᵢ² ∝ eigenvalues of covariance** — singular values of data ↔ eigenvalues of covariance.

### Anki Cards
- Q: "What does SVD stand for, and what is the decomposition X = ?" → A: Singular Value Decomposition; X = UDVᵀ where U, V orthogonal, D diagonal with singular values σ₁ ≥ σ₂ ≥ ...
- Q: "What is the relationship between SVD and spectral decomposition for a symmetric PSD matrix?" → A: They are identical. U = V = eigenvectors, σᵢ = λᵢ.
- Q: "What does the Eckart-Young theorem guarantee?" → A: The truncated SVD (keeping k terms) gives the best rank-k approximation to X. No other rank-k matrix is closer.
- Q: "How do singular values of X relate to eigenvalues of the covariance S?" → A: σᵢ²(X_centered) = (n-1) · λᵢ(S), so eigenvalues of S = σᵢ² / (n-1).

---

*Today's bridge: SVD is the engine. PCA is just the engine in a car.*
