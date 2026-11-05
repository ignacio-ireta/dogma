# MVS Lesson 8 (Patch): Orthogonal Projections
**Date:** May 8, 2026 | **Topic:** Projections, Subspaces, and What PCA Is Actually Doing
**Syllabus position:** Foundation patch — required before PCA (L9)

---

## 1. SPACED RETRIEVAL WARM-UP (10 min)

*From Lesson 7 (yesterday — SVD) and Lesson 2 (eigenvalues, Feb 24).*

**WU1 (L7 — yesterday):** Write the SVD decomposition X = ? What are the properties of U, D, and V?

**WU2 (L7 — yesterday):** True or false: "For a symmetric positive semi-definite matrix, SVD and spectral decomposition give identical results." Explain briefly.

**WU3 (L2 — Feb 24):** If a matrix A has eigenvalues λ₁ = 5, λ₂ = 3, λ₃ = 0, what is the rank of A? Is it positive definite? Positive semi-definite?

**WU4 (L7 — yesterday):** You have a 100×4 data matrix with singular values σ₁ = 15, σ₂ = 8, σ₃ = 1.2, σ₄ = 0.3. What fraction of total variance do the first two components capture?

---

## 2. PRE-TEST: Orthogonal Projections (5 min)

*Before I teach — guess.*

**PT1:** If you shine a light straight down on a 3D object, its shadow on the floor is a 2D projection. What do you think "orthogonal projection" means mathematically? What's being projected, and onto what?

**PT2:** If you have a vector x in 3D space and a 2D plane, there's a unique point on that plane that's closest to x. What angle does the line from x to that closest point make with the plane?

**PT3:** PCA "reduces dimensions." Based on what you know about SVD, do you think PCA is projecting data onto a lower-dimensional subspace? If so, what determines which subspace?

---

## 3. EXPLICIT INSTRUCTION (30 min)

### 3.1 The Geometric Intuition

Imagine a point x floating in 3D space above a flat table (a 2D plane). You drop a plumb line straight down — perpendicular to the table. Where it hits is the **orthogonal projection** of x onto the table.

Key properties:
- The projected point is the **closest point on the plane** to x
- The line from x to the projection is **perpendicular** (orthogonal) to every direction in the plane
- The "shadow" captures as much of x as possible given the constraint of living in 2D

**This is exactly what PCA does.** PCA finds the best flat surface (subspace) to project your data onto — the one that loses the least information.

### 3.2 The Projection Formula

Let's formalize. If you have:
- A vector **x** in ℝᵖ
- A subspace spanned by orthonormal vectors **v₁, v₂, ..., vₖ**

The orthogonal projection of x onto this subspace is:

```
Proj(x) = (v₁ᵀx)v₁ + (v₂ᵀx)v₂ + ... + (vₖᵀx)vₖ
```

In matrix form:

```
Proj(x) = VVᵀx
```

where V = [v₁ | v₂ | ... | vₖ] is p×k.

**What each term means:**
- vᵢᵀx = the dot product of vᵢ with x = how much of x lies in the vᵢ direction
- (vᵢᵀx)vᵢ = the component of x along vᵢ
- Summing them up = reconstructing x using only the v₁,...,vₖ directions

**The residual** (what's lost):

```
x - Proj(x) = x - VVᵀx = (I - VVᵀ)x
```

This residual is orthogonal to the subspace. Its squared length ‖x - Proj(x)‖² is the reconstruction error.

### 3.3 Projections Preserve What Matters

If v₁, ..., vₖ are the first k **right singular vectors** from SVD (or equivalently, the first k eigenvectors of the covariance matrix), then:

```
total squared error = Σᵢ₌ₖ₊₁ᵖ σᵢ² = Σᵢ₌ₖ₊₁ᵖ (n-1)λᵢ
```

This is minimized. No other k-dimensional subspace gives a smaller total reconstruction error. **This is the geometric statement of PCA.**

### 3.4 Projection as Multiplication

There's a cleaner way to think about it:

```
# Original data point (p-dimensional)
x ∈ ℝᵖ

# Project onto k-dimensional subspace
z = Vₖᵀx ∈ ℝᵏ    (the "scores" — low-dimensional representation)

# Reconstruct from scores
x̂ = Vₖz = VₖVₖᵀx ∈ ℝᵖ  (back in original space, but approximate)

# Error
e = x - x̂ = (I - VₖVₖᵀ)x
```

**The scores z are what PCA gives you.** They're the coordinates of x in the new basis v₁,...,vₖ. When someone says "I reduced my 50-dimensional data to 3 PCA components," they computed z = V₃ᵀx for every observation — a projection onto a 3D subspace.

### 3.5 The Projection Matrix

The matrix P = VVᵀ is called a **projection matrix** (or projector). It has key properties:

```
P² = P              (projecting twice = projecting once)
Pᵀ = P              (symmetric — orthogonal projection)
rank(P) = k         (projects onto a k-dimensional space)
```

And (I - P) is the projector onto the **orthogonal complement** — the space of everything NOT captured.

### 3.6 Why This Matters for MVS

| MVS Concept | What It Is (Projection Language) |
|-------------|-------------------------------|
| PCA | Project onto the subspace that minimizes reconstruction error |
| Principal components | The basis vectors v₁,...,vₖ of that subspace |
| Scores zᵢ = vᵢᵀx | Coordinates in the new subspace |
| Variance explained by component i | λᵢ / Σλⱼ = σᵢ² / Σσⱼ² |
| Factor Analysis | Like PCA but with a *probabilistic* projection (measurement error model) |
| Discriminant Analysis | Project onto the subspace that best separates groups |

**Everything in MVS after this point is some flavor of "project data onto a cleverly chosen subspace."** The subspace changes (maximize variance, separate groups, model latent variables), but the mechanism is always projection.

---

## 4. INTERLEAVED PRACTICE (40 min)

### Conceptual

**P1:** Explain in plain language: "PCA projects data onto the subspace spanned by the top eigenvectors of the covariance matrix." What does this mean geometrically?

**P2:** If you project a point x onto a 2D subspace and the error x - Proj(x) is zero, what does that tell you about x?

**P3:** True or false:
- (a) "A projection matrix P always satisfies P² = P."
- (b) "Orthogonal projection always increases the length of a vector."
- (c) "The residual (x - Px) is always orthogonal to the projected point (Px)."

### Computation

**P4:** Given v₁ = [1/√2, 1/√2]ᵀ and v₂ = [1/√2, -1/√2]ᵀ (an orthonormal basis for ℝ²), project x = [3, 1]ᵀ:
- (a) Onto v₁ only: z₁ = v₁ᵀx = ? and x̂ = z₁v₁ = ?
- (b) Onto the full space (v₁, v₂): z = Vᵀx = ? and x̂ = VVᵀx = ?
- (c) What's the reconstruction error in each case?

**P5:** A 3×3 covariance matrix has eigenvectors v₁, v₂, v₃ with eigenvalues λ₁ = 10, λ₂ = 3, λ₃ = 0.5.
- (a) What fraction of variance is captured by projecting onto v₁ alone?
- (b) Onto the subspace {v₁, v₂}?
- (c) If you keep v₁ and v₂, what is the average squared error per observation?

### Connection

**P6 (connects L7 + L8):** You have a centered data matrix X (n×p). You compute SVD: X = UDVᵀ. You keep the first k columns of V to form Vₖ.
- (a) Write the formula for the scores matrix Z (n×k) in terms of X and Vₖ.
- (b) Write the formula for the reconstructed data X̂ in terms of Z and Vₖ.
- (c) What is X̂ in terms of the SVD components only? (Hint: it's the truncated SVD)

---

## 5. SUMMARY + ANKI SUGGESTIONS (5 min)

### Key Takeaways
1. **Projection = dropping a perpendicular** from a point to a subspace. The closest point IS the projection.
2. **Proj(x) = VVᵀx** where V contains the orthonormal basis of the target subspace.
3. **PCA = projection onto the subspace that minimizes total squared reconstruction error.** The subspace is spanned by the top eigenvectors of S (or equivalently, top right singular vectors of X).
4. **Scores = coordinates in the new basis**, z = Vₖᵀx. This is the low-dimensional representation.
5. **Everything in MVS (PCA, FA, discriminant analysis) is projection onto a cleverly chosen subspace.**

### Anki Cards
- Q: "What is the formula for orthogonal projection of x onto subspace V?" → A: Proj(x) = VVᵀx, where columns of V are an orthonormal basis for the subspace.
- Q: "What are the three properties of a projection matrix P = VVᵀ?" → A: P² = P (idempotent), Pᵀ = P (symmetric), rank(P) = k.
- Q: "What is a PCA score?" → A: zᵢ = vᵢᵀx — the coordinate of data point x along principal component direction vᵢ. It's a projection coefficient.
- Q: "Why does PCA minimize reconstruction error?" → A: Because the truncated SVD (keeping top k singular vectors) is the best rank-k approximation (Eckart-Young). PCA projects onto those vectors.
- Q: "What is the residual x - Proj(x) geometrically?" → A: The part of x NOT captured by the subspace. It's orthogonal to every direction in the subspace.

---

*Projection is the verb. PCA is the strategy. The subspace is the choice.*
