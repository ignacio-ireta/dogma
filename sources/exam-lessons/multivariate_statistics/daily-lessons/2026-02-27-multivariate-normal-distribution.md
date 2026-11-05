# MVS Lesson 4: The Multivariate Normal Distribution
**Date:** February 27, 2026 | **Topic:** The Foundation of Classical Multivariate Statistics
**Syllabus position:** Topic 2 — Multivariate Normal Distribution

---

## 1. SPACED RETRIEVAL WARM-UP (10 min)

*Pulling from Lesson 1 (Feb 23, 4 days ago) and Lesson 2 (Feb 24, 3 days ago). Try from memory first.*

**Q1:** (L1) You have two vectors u = [1, 0, 3]' and v = [2, 5, -1]'. Compute the dot product u'v. Are they orthogonal?

**Q2:** (L2) What does det(A) = 0 tell you about the matrix A? Give at least two consequences.

**Q3:** (L2) For the matrix M = [[6, 2], [2, 3]], find the eigenvalues. (Set up det(M - λI) = 0 and solve the quadratic.)

**Q4:** (L1) If A is 3×2 and B is 2×5, what are the dimensions of AB? What about BA — can you even compute it?

*If you blank on any of these — good. The retrieval attempt is what builds the trace. Struggle a full minute before moving on.*

---

## 2. PRE-TEST: The Multivariate Normal (5 min)

*Before I teach anything — guess. Wrong answers are productive.*

**P1:** You know the univariate normal N(μ, σ²) is fully described by two parameters: mean and variance. For a p-dimensional multivariate normal, what two "things" do you think describe it? (Hint: the "mean" becomes something bigger, and the "variance" becomes something bigger.)

**P2:** In one dimension, the bell curve is symmetric around μ. In two dimensions, what shape do you think the "equal-probability contours" form? (Think about what we learned about quadratic forms and level curves x'Ax = c.)

**P3:** If you have a multivariate normal random vector X = [X₁, X₂, X₃]' and you pick just X₂ — do you think X₂ by itself is also normally distributed? Guess yes or no.

---

## 3. EXPLICIT INSTRUCTION: The Multivariate Normal Distribution (30 min)

### 3.1 From One Dimension to Many — The Big Picture

**The univariate normal** N(μ, σ²) has density:

f(x) = (1 / √(2πσ²)) · exp(−(x − μ)² / (2σ²))

Let's dissect what each piece does:

| Piece | Role |
|-------|------|
| μ | Center of the bell |
| σ² | Width of the bell (spread) |
| (x − μ)² / σ² | **Squared distance** from x to μ, scaled by variance |
| exp(−½ · ...) | Turns distance into probability (closer = higher) |
| 1/√(2πσ²) | Normalizing constant (makes it integrate to 1) |

**The key insight:** The exponent contains **(x − μ)² / σ²** — a standardized squared distance. The multivariate version replaces this with a **quadratic form**.

### 3.2 The Multivariate Normal Density

A p-dimensional random vector **X** = [X₁, X₂, ..., Xₚ]' follows a **multivariate normal distribution** Nₚ(μ, Σ) if its density is:

**f(x) = (2π)^(−p/2) · |Σ|^(−1/2) · exp(−½ (x − μ)' Σ⁻¹ (x − μ))**

Let's map every piece:

| Univariate | Multivariate | Type |
|------------|-------------|------|
| μ (scalar) | **μ** (p×1 vector) | Mean vector |
| σ² (scalar) | **Σ** (p×p matrix) | Covariance matrix |
| (x−μ)²/σ² | **(x−μ)'Σ⁻¹(x−μ)** | Mahalanobis distance squared |
| 1/√(2πσ²) | (2π)^(−p/2) \|Σ\|^(−1/2) | Normalizing constant |

**⚠️ Requirements on Σ:**
- Σ must be **symmetric** (Σ = Σ') — always true for covariance matrices
- Σ must be **positive definite** — so Σ⁻¹ exists and the density is well-defined
- If Σ is only positive semi-definite (some eigenvalue = 0), the distribution is **degenerate** — it lives on a lower-dimensional subspace

### 3.3 The Mahalanobis Distance — The Heart of MVN

The expression **(x − μ)' Σ⁻¹ (x − μ)** is called the **Mahalanobis distance squared** (d²).

**Why not just use Euclidean distance?**

Euclidean distance treats all directions equally. But if your variables have different variances, or are correlated, Euclidean distance is misleading.

**Example:** Suppose height has σ = 10 cm and weight has σ = 15 kg. A person who is 5 cm above average height is "closer to the mean" in a standardized sense than someone 5 kg above average weight. And if height and weight are correlated, you need to account for that too.

The Mahalanobis distance does exactly this: it **standardizes** using the full covariance structure.

**Geometric interpretation:**
- Euclidean distance: circles (all directions equal)
- Mahalanobis distance: **ellipses** (shaped by Σ)
- The eigenvectors of Σ determine the **axes** of the ellipse
- The eigenvalues determine how **stretched** each axis is

**Rule:** The set of all points x where (x − μ)' Σ⁻¹ (x − μ) = c² is an **ellipsoid** centered at μ. This is a contour of constant probability density.

### 3.4 Worked Example — Bivariate Normal (p = 2)

Let's build a concrete example:

```
μ = [3]     Σ = [4  1]
    [1]         [1  2]
```

**Step 1: Verify Σ is positive definite.**

Eigenvalue test:
det(Σ − λI) = (4−λ)(2−λ) − 1 = λ² − 6λ + 7 = 0
λ = (6 ± √(36−28))/2 = (6 ± √8)/2
λ₁ = (6 + 2√2)/2 ≈ 4.414
λ₂ = (6 − 2√2)/2 ≈ 1.586

Both positive → **Σ is PD** ✅

**Step 2: Compute Σ⁻¹.**

For 2×2: if Σ = [a b; c d], then Σ⁻¹ = (1/det) · [d −b; −c a]

det(Σ) = 4·2 − 1·1 = 7

Σ⁻¹ = (1/7) · [2  −1] = [2/7   −1/7]
              [−1  4]   [−1/7   4/7]

**Step 3: Compute Mahalanobis distance for a specific point.**

Let x = [5, 2]'. How "far" is it from μ in Mahalanobis terms?

x − μ = [5−3, 2−1]' = [2, 1]'

d² = [2  1] · [2/7  −1/7] · [2]
              [−1/7  4/7]   [1]

   = [2  1] · [4/7 − 1/7]  = [2  1] · [3/7]
              [−2/7 + 4/7]           [2/7]

   = 6/7 + 2/7 = 8/7 ≈ 1.143

So d² ≈ 1.143 — this point is roughly 1.07 Mahalanobis "standard deviations" from the mean.

### 3.5 Key Properties of the Multivariate Normal

These are the properties you MUST know for the exam. Each one is a potential question.

**Property 1: Marginal distributions are normal.**

If X ~ Nₚ(μ, Σ), then any subset of components is also normal.

If X = [X₁, X₂, X₃]' ~ N₃(μ, Σ), then:
- X₁ ~ N(μ₁, σ₁₁)
- [X₁, X₃]' ~ N₂([μ₁, μ₃]', Σ₁₃)

where Σ₁₃ is the submatrix of Σ using rows/columns 1 and 3.

**⚠️ WARNING — The converse is FALSE:** Having all marginals be normal does NOT guarantee the joint is multivariate normal. This is a classic exam trap.

**Property 2: Linear combinations are normal.**

If X ~ Nₚ(μ, Σ) and **a** is any constant p×1 vector, then:

**a'X ~ N(a'μ, a'Σa)**

This is incredibly powerful. ANY linear combination of jointly normal variables is normal.

More generally, if A is a q×p matrix:

**AX ~ Nq(Aμ, AΣA')**

**Property 3: Independence ⟺ zero covariance (ONLY for MVN!).**

For multivariate normal variables: **uncorrelated ⟺ independent**.

This is SPECIAL to the normal distribution. In general, uncorrelated does NOT imply independent. But for MVN, it does.

Specifically: if the covariance matrix Σ is diagonal (all off-diagonal entries = 0), then the components X₁, ..., Xₚ are mutually independent.

**Property 4: Conditional distributions are normal.**

If X = [X₁', X₂']' is partitioned into two groups, the conditional distribution of X₁ given X₂ = x₂ is:

X₁ | X₂ = x₂ ~ N(μ₁ + Σ₁₂Σ₂₂⁻¹(x₂ − μ₂), Σ₁₁ − Σ₁₂Σ₂₂⁻¹Σ₂₁)

Key observations:
- The conditional **mean** is a linear function of x₂ (this is where regression comes from!)
- The conditional **covariance** does NOT depend on x₂ (homoscedasticity)

**Property 5: Chi-squared connection.**

If X ~ Nₚ(μ, Σ), then:

**(X − μ)' Σ⁻¹ (X − μ) ~ χ²(p)**

The Mahalanobis distance squared follows a chi-squared distribution with p degrees of freedom. This is how you build confidence ellipsoids and tests.

### 3.6 Summary Table — The MVN at a Glance

| Feature | Formula | Note |
|---------|---------|------|
| Notation | X ~ Nₚ(μ, Σ) | p dimensions |
| Parameters | μ (p×1), Σ (p×p, PD) | Mean vector + covariance matrix |
| Density | (2π)^(−p/2) \|Σ\|^(−1/2) exp(−½ d²) | d² = Mahalanobis distance² |
| Marginals | Always normal | Subvector → submatrix |
| Linear combos | a'X ~ N(a'μ, a'Σa) | ANY linear combo is normal |
| Uncorr ⟺ Indep | YES (MVN only!) | Diagonal Σ ⟹ independence |
| Conditional | Normal, linear mean | Where regression lives |
| d² distribution | χ²(p) | For tests and confidence regions |

---

## 4. INTERLEAVED PRACTICE (40 min)

*Mix of today's MVN topic + previous lessons. Attempt each one honestly. Answers in a separate file.*

### Problem 1 (Today — Mahalanobis distance)

Given μ = [0, 0]' and Σ = [[1, 0.5], [0.5, 1]], compute the Mahalanobis distance squared d² for the point x = [1, 1]'.

Steps: (a) find Σ⁻¹, (b) compute (x − μ), (c) compute d².

---

### Problem 2 (Today — Marginal distributions)

Suppose X = [X₁, X₂, X₃]' ~ N₃(μ, Σ) where:

```
μ = [2, -1, 5]'

Σ = [4   1  -1]
    [1   3   0]
    [-1  0   2]
```

a) What is the marginal distribution of X₂? Give its mean and variance.
b) What is the marginal distribution of [X₁, X₃]'? Give its mean vector and covariance matrix.

---

### Problem 3 (Today — Linear combinations)

X ~ N₂(μ, Σ) with μ = [10, 20]' and Σ = [[9, 3], [3, 16]].

Let Y = X₁ + X₂ (i.e., a = [1, 1]').

a) What is E[Y]?
b) What is Var(Y)?
c) What distribution does Y follow?

---

### Problem 4 (Today — Independence and correlation)

Two variables X₁ and X₂ are jointly MVN with covariance matrix:

```
Σ = [5  0]
    [0  8]
```

a) Are X₁ and X₂ independent? Why?
b) Now suppose X₁ and X₂ are NOT jointly normal, but have the same Σ. Can you still conclude independence? Why or why not?

---

### Problem 5 (Lesson 2 — Interleaving: Determinant and inverse)

Compute the determinant and inverse of:

```
A = [3  1]
    [2  5]
```

---

### Problem 6 (Lesson 1 — Interleaving: Matrix multiplication)

Compute:

```
[1  0  2] [4]
[3  1  0] [1]
          [2]
```

---

### Problem 7 (Python — Visualizing the bivariate normal)

Write Python code that:
1. Defines μ = [3, 1] and Σ = [[4, 1], [1, 2]]
2. Generates 1000 samples from this MVN using `np.random.multivariate_normal(mu, sigma, 1000)`
3. Creates a scatter plot of the samples
4. Draws the theoretical ellipse for the 95% confidence region

Hint: for the ellipse, you need eigenvalues/eigenvectors of Σ and the χ²(2) critical value at 95% (≈ 5.991). The semi-axes are √(λᵢ · χ²_crit).

Use `matplotlib.patches.Ellipse` — you'll need the angle = arctan(eigenvector slope) in degrees.

---

## 5. SUMMARY + ANKI SUGGESTIONS (5 min)

### Key Takeaways
1. **Nₚ(μ, Σ)** — fully described by mean vector μ and covariance matrix Σ
2. **Mahalanobis distance** replaces Euclidean distance — accounts for variance AND correlation
3. **Marginals are always normal** — just take the corresponding subvector/submatrix
4. **Linear combinations are normal** — a'X ~ N(a'μ, a'Σa)
5. **Uncorrelated ⟺ independent** is ONLY true for MVN — classic exam trap
6. **Conditional distributions are normal** with linear conditional mean — this is where regression comes from
7. **d² ~ χ²(p)** — the bridge to hypothesis testing

### Anki Card Ideas
- MVN density formula — what does each piece represent?
- Mahalanobis distance: formula and geometric interpretation (ellipsoids)
- True or false: "All marginals normal ⟹ joint is MVN" (FALSE!)
- True or false: "Uncorrelated ⟹ independent" — when is this true? (Only for MVN)
- What distribution does a'X follow if X ~ Nₚ(μ, Σ)? → N(a'μ, a'Σa)
- (X−μ)'Σ⁻¹(X−μ) ~ ??? → χ²(p)
- Conditional mean of X₁|X₂: linear function of x₂ — what's the formula?
- What happens to the conditional covariance — does it depend on x₂? (No)

### Monday's Preview
**Lesson 5: Sampling from the MVN — Sample Mean Vector, Sample Covariance, and Their Distributions.** We move from the theoretical distribution to what happens with real data. How do you estimate μ and Σ? What are the distributions of those estimators? This is where Wishart enters the picture.

---

*The matrix algebra trilogy built the tools. Today you learned the distribution that uses ALL of them. This is the beating heart of MVS — every test, every method in this course starts from the MVN.* ❤️

*Jun 24 nærmer seg. Du har verktøyene nå.* 🏔️
