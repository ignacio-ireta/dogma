# MVS Lesson 3: Quadratic Forms & Positive Definiteness
**Date:** February 25, 2026 | **Topic:** The Bridge to Multivariate Distributions
**Syllabus position:** Topic 1 (final) — Matrix Algebra Review → Transition to Probability

---

## 1. SPACED RETRIEVAL WARM-UP (10 min)

*Pulling from Lesson 1 (Feb 23, matrix foundations) and calculus (Feb 3-9). Try from memory first — the struggle is the learning.*

**Q1:** What does it mean geometrically when the dot product x'y = 0? And why does that matter in statistics?

**Q2:** You have A (4×3) and B (2×4). Can you compute AB? Can you compute BA? What are the dimensions of whichever one works?

**Q3:** Write the gradient ∇f as a column vector for f(x₁, x₂) = 5x₁² − 3x₁x₂ + x₂³.

**Q4:** The transpose reversal rule: (AB)' = ??? — fill in the blank. Why does the order flip?

*Take a solid minute on each. If you're blanking, that's normal — the retrieval attempt still strengthens the trace.*

---

## 2. PRE-TEST: Quadratic Forms & Positive Definiteness (5 min)

Before I teach anything, guess. Wrong answers prime the brain — they're GOOD.

**P1:** You know the scalar expression ax². Now imagine we replace the scalar x with a vector **x** and the scalar a with a matrix **A**. What do you think x'Ax looks like? Is the result a scalar, a vector, or a matrix?

**P2:** A covariance matrix measures how variables move together. Why do you think a covariance matrix can NEVER have a negative eigenvalue? (Hint: think about what variance means.)

**P3:** What do you think "positive definite" might mean for a matrix? Take your best guess based on the name alone.

---

## 3. EXPLICIT INSTRUCTION: Quadratic Forms & Positive Definiteness (30 min)

### 3.1 From Scalars to Matrices — The Quadratic Form

In single-variable calculus, you know the quadratic: **f(x) = ax²**

- If a > 0 → parabola opens upward → minimum exists
- If a < 0 → parabola opens downward → maximum exists
- If a = 0 → flat line → degenerate

**The multivariate version** replaces scalar x with vector **x** ∈ ℝᵖ, and scalar a with matrix **A** (p × p):

**Q(x) = x'Ax**

This is called a **quadratic form**. It maps a vector to a **scalar**.

#### Why it's a scalar:

If x ∈ ℝᵖ (p×1) and A is p×p:
- x' is 1×p
- A is p×p  
- x is p×1
- So: (1×p)(p×p)(p×1) = (1×1) = scalar ✅

### 3.2 Expanding the Quadratic Form

For p = 2, with x = [x₁, x₂]' and A = [a₁₁ a₁₂; a₂₁ a₂₂]:

```
x'Ax = [x₁  x₂] [a₁₁  a₁₂] [x₁]
                  [a₂₁  a₂₂] [x₂]

     = [x₁  x₂] [a₁₁x₁ + a₁₂x₂]
                  [a₂₁x₁ + a₂₂x₂]

     = a₁₁x₁² + a₁₂x₁x₂ + a₂₁x₁x₂ + a₂₂x₂²
     = a₁₁x₁² + (a₁₂ + a₂₁)x₁x₂ + a₂₂x₂²
```

**Key observation:** Only the **symmetric part** of A matters for the quadratic form. We can always replace A with (A + A')/2 without changing x'Ax. So we **always assume A is symmetric** in quadratic forms.

#### Worked Example:

```
A = [2  1]    x = [3]
    [1  4]        [1]

x'Ax = [3  1] [2  1] [3]
              [1  4] [1]

     = [3  1] [2·3+1·1] = [3  1] [7]
              [1·3+4·1]          [7]

     = 3·7 + 1·7 = 21 + 7 = 28
```

Or by expansion: 2(3²) + 2(1)(3)(1) + 4(1²) = 18 + 6 + 4 = 28 ✅

### 3.3 Positive Definiteness — The Classification System

Just like the sign of "a" in ax² tells you everything about the parabola, the **definiteness** of A tells you everything about x'Ax:

| Classification | Condition | What it means |
|---------------|-----------|---------------|
| **Positive definite (PD)** | x'Ax > 0 for ALL x ≠ 0 | Bowl shape — has a minimum |
| **Positive semi-definite (PSD)** | x'Ax ≥ 0 for all x | Flat-bottomed bowl — some directions are flat |
| **Negative definite (ND)** | x'Ax < 0 for ALL x ≠ 0 | Upside-down bowl |
| **Indefinite** | x'Ax > 0 for some x, < 0 for others | Saddle shape |

**⚠️ The critical fact for MVS:**

**Covariance matrices are ALWAYS positive semi-definite.** (And usually positive definite unless variables are perfectly linearly dependent.)

Why? Because variance can never be negative. The quadratic form x'Σx represents the variance of a linear combination of variables. Variance ≥ 0 always. So x'Σx ≥ 0 for all x.

### 3.4 How to Check Positive Definiteness — Three Equivalent Tests

For a symmetric matrix A, these are ALL equivalent:

| Test | Rule | When to use |
|------|------|------------|
| **Eigenvalue test** | ALL eigenvalues > 0 | Best for understanding |
| **Determinant test** | All leading principal minors > 0 | Best for hand calculation |
| **Cholesky test** | A = LL' exists with L lower triangular | Best for computation |

#### The Eigenvalue Test (most intuitive):

**A is positive definite ⟺ all eigenvalues of A are positive.**

Why this makes sense: eigenvectors are the "natural axes" of the quadratic form. The eigenvalue λᵢ tells you how steep the bowl is along eigenvector direction eᵢ. If all λᵢ > 0, the bowl goes up in EVERY direction → positive definite.

#### The Leading Principal Minor Test:

For A (p×p), compute these determinants in order:

- d₁ = a₁₁
- d₂ = det([a₁₁ a₁₂; a₂₁ a₂₂])  (top-left 2×2 block)
- d₃ = det(top-left 3×3 block)
- ... up to dₚ = det(A)

**A is positive definite ⟺ d₁ > 0, d₂ > 0, ..., dₚ > 0** (ALL positive)

#### Worked Example — Checking PD:

```
A = [2  1]
    [1  3]
```

**Eigenvalue test:**
det(A - λI) = (2-λ)(3-λ) - 1 = λ² - 5λ + 5 = 0
λ = (5 ± √5)/2 → λ₁ ≈ 3.618, λ₂ ≈ 1.382
Both positive → **A is positive definite** ✅

**Leading minor test:**
d₁ = 2 > 0 ✅
d₂ = det(A) = 2·3 - 1·1 = 5 > 0 ✅
Both positive → **A is positive definite** ✅

### 3.5 The Spectral Decomposition Revisited

From yesterday: a symmetric matrix decomposes as **A = PDP'**, where:
- P = matrix of eigenvectors (columns), orthogonal: P'P = I
- D = diagonal matrix of eigenvalues

Now here's the punchline: the quadratic form becomes

**x'Ax = x'PDP'x = y'Dy = λ₁y₁² + λ₂y₂² + ... + λₚyₚ²**

where y = P'x (rotation to eigenvector coordinates).

**This is just a weighted sum of squares!** Each eigenvalue λᵢ weights one dimension. If all λᵢ > 0, you get a sum of positive terms → positive definite.

**Geometric picture:** The quadratic form x'Ax = c (constant) traces out an **ellipsoid** in ℝᵖ. The eigenvectors are the axes of the ellipsoid. The eigenvalues determine how long each axis is (axis length ∝ 1/√λᵢ).

This is EXACTLY what you'll see in the multivariate normal distribution — the contours of equal probability density are ellipsoids whose shape is determined by the covariance matrix's eigenstructure.

### 3.6 Why This Matters — The Connection to MVN

The multivariate normal distribution's density is:

f(x) = (2π)^(-p/2) |Σ|^(-1/2) exp(−½ (x−μ)'Σ⁻¹(x−μ))

See that **(x−μ)'Σ⁻¹(x−μ)**? That's a quadratic form! Specifically:
- It's the **Mahalanobis distance** from x to μ
- Σ⁻¹ is positive definite (since Σ is)
- The level curves where this distance is constant form ellipsoids
- The principal axes of these ellipsoids are the eigenvectors of Σ

Everything connects. Matrix algebra → quadratic forms → positive definiteness → multivariate normal. Next lesson, we build the MVN distribution itself.

---

## 4. INTERLEAVED PRACTICE (40 min)

*Mix of today's topic + previous material. Answers in a separate file — just attempt each one honestly.*

### Problem 1 (Today — Compute a quadratic form)
Let A = [[3, 1], [1, 2]] and x = [2, -1]'.

Compute x'Ax by hand. Show your steps.

---

### Problem 2 (Today — Positive definiteness by eigenvalues)
Is the following matrix positive definite?

```
B = [4   2]
    [2   1]
```

Find the eigenvalues and use them to determine definiteness. What does it tell you about using B as a covariance matrix?

---

### Problem 3 (Today — Leading principal minor test)
Check positive definiteness of the following matrix using the leading principal minor test (compute d₁, d₂, d₃):

```
C = [3   1  0]
    [1   4  2]
    [0   2  5]
```

---

### Problem 4 (Today — Connecting quadratic forms to geometry)
The quadratic form x'Ax = 1, with A = [[2, 0], [0, 8]], traces an ellipse in 2D.

a) What are the eigenvalues of A? (Hint: A is already diagonal.)
b) Along which axis (x₁ or x₂) is the ellipse longer?
c) What are the semi-axis lengths? (Recall: semi-axis length = 1/√λ along each eigenvector direction)

---

### Problem 5 (Lesson 1 — Interleaving: Matrix operations)
Given:
```
F = [1  2  0]      g = [3]
    [0  1  3]          [1]
                       [2]
```

a) Compute Fg.
b) Compute g'g (the dot product of g with itself). What is this value called?

---

### Problem 6 (Lesson 2 — Interleaving: Eigenvalues)
Find the eigenvalues of:
```
G = [5  2]
    [2  5]
```
Is G positive definite? Without computing eigenvectors — what can you say about them since G is symmetric?

---

### Problem 7 (Python — Positive definiteness check)
Write Python code that:
1. Creates the matrix C = [[3,1,0],[1,4,2],[0,2,5]]
2. Finds its eigenvalues using `np.linalg.eigvalsh` (the `h` = assumes symmetric — more numerically stable)
3. Prints the eigenvalues
4. Prints whether C is positive definite (all eigenvalues > 0)
5. **Bonus:** Use `np.linalg.cholesky(C)` to verify — if it runs without error, C is PD

---

## 5. SUMMARY + ANKI SUGGESTIONS (5 min)

### Key Takeaways
1. **Quadratic form x'Ax** = the multivariate version of ax². Maps a vector to a scalar.
2. **Only the symmetric part matters** — always assume A is symmetric in quadratic forms.
3. **Positive definite = all eigenvalues > 0** = bowl shape = valid covariance matrix.
4. **Three equivalent tests:** eigenvalues all > 0 / leading minors all > 0 / Cholesky exists.
5. **x'Ax = λ₁y₁² + λ₂y₂² + ... + λₚyₚ²** after rotating to eigenvector coordinates.
6. **Ellipsoids:** quadratic form = constant traces an ellipsoid. Eigenvectors = axes, eigenvalues = axis scaling.
7. **This IS the MVN shape:** the multivariate normal's density contours are exactly these ellipsoids.

### Anki Card Ideas
- What is a quadratic form? (x'Ax — maps vector to scalar)
- Positive definite ↔ eigenvalues: what's the relationship?
- What does det(A) = 0 mean for positive definiteness? (PSD at best, not PD)
- Leading principal minor test: what must ALL be positive?
- Covariance matrices are always ___-definite. (positive semi-definite)
- What shape do the level curves x'Ax = c form? (ellipsoid)
- Mahalanobis distance formula: (x−μ)'Σ⁻¹(x−μ) — what kind of expression is this? (quadratic form)

### Tomorrow's Preview
**Lesson 4: The Multivariate Normal Distribution** — The foundation of classical multivariate statistics. You now have ALL the tools: matrices, eigenvalues, quadratic forms, positive definiteness. Tomorrow we assemble them into the MVN — its density, its properties, and why literally every test in MVS assumes it. The payoff lesson.

### Textbook References
- Anderson Ch. 2.5–2.6 (Quadratic forms, positive definiteness)
- Modern Multivariate Statistical Techniques Ch. 2.3–2.5
- 3Blue1Brown "Change of Basis" — helps visualize the spectral decomposition

---

*You just completed the matrix algebra trilogy. Three days, three building blocks: operations → eigenstructure → quadratic forms. Tomorrow the statistics actually begins.* 🧱🧱🧱

*Jun 24 er målet. Vi bygger stein for stein.* 🏔️
