# MVS Lesson 6: Generalized Variance & The Wishart Distribution
**Date:** March 10, 2026 | **Topic:** |S|, the Wishart Distribution, and Sampling Theory for x̄ and S
**Syllabus position:** Topic 3 (continued) — Generalized Variance, Random Sampling Distributions

---

## 1. SPACED RETRIEVAL WARM-UP (10 min)

*Pulling from Lessons 3–4 (Feb 25 + Feb 27 — 13 and 11 days ago). Not yesterday. Struggle first.*

**WU1 (L4 — Feb 27):** The multivariate normal density is:
```
f(x) = (2π)^(-p/2) · |Σ|^(-1/2) · exp(-½ (x − μ)ᵀ Σ⁻¹ (x − μ))
```
Three questions:
- (a) What is the name of the quantity inside the exp? What does it measure?
- (b) If p = 1 (one variable), what does this reduce to? (Write it in familiar terms.)
- (c) What shape do contours of constant density form in 2D? What matrix controls the shape and orientation?

**WU2 (L4 — Feb 27):** Suppose X = [X₁, X₂]' ~ N₂(μ, Σ). Which of the following is true?
- (a) X₁ and X₂ are always independent.
- (b) X₁ and X₂ are independent if and only if Cov(X₁, X₂) = 0.
- (c) Any linear combination aX₁ + bX₂ is also normally distributed.

Which are true? Explain briefly — no formulas needed, just reasoning.

**WU3 (L3 — Feb 25):** A 3×3 matrix A is **positive definite** if...
- (a) State the quadratic form definition.
- (b) Give two equivalent ways to test it (no need to derive — just state the tests).
- (c) Why does it matter for Σ (the covariance matrix of a multivariate normal) to be positive definite?

**WU4 (L5 — yesterday, but from the geometry section):** Fill in the blank with the correct word: "The correlation r_{jk} between two variables equals the _______ of the angle between their deviation vectors in sample space."

(One word. This is the geometric punchline from yesterday.)

---

## 2. PRE-TEST: Generalized Variance & Wishart (5 min)

*Before I teach — guess. Wrong answers are productive.*

**PT1:** You saw yesterday that |S| is called the "generalized variance." Intuitively, what do you think it measures? Can it ever be negative? Can it be zero?

**PT2:** The chi-squared distribution χ²(k) shows up when you square standard normal variables. In multivariate statistics, we have samples from a **multivariate** normal. What do you think the multivariate generalization of χ² might be? (Just guess a name or describe what you'd expect it to look like.)

**PT3:** If x̄ is the sample mean from n observations of a p-variate normal, and the true mean is μ... what distribution do you think x̄ follows? Does increasing n make it more or less spread out?

---

## 3. EXPLICIT INSTRUCTION (30 min)

### 3.1 Generalized Variance — A Deeper Look

Recall from yesterday: the **generalized variance** of a sample is **|S|**, the determinant of the sample covariance matrix.

But *why* is the determinant the right choice? Let's build the intuition properly.

#### 3.1.1 What the Determinant Measures

From Lesson 2: det(A) gives the **volume scaling factor** of the linear transformation defined by A.

For the covariance matrix S, consider the **deviation vectors** $\mathbf{d}_1, \ldots, \mathbf{d}_p$ (one per variable, each in ℝⁿ). These vectors define a **parallelotope** (the p-dimensional generalization of a parallelogram) in sample space.

**Key result:**
$$|\mathbf{S}| = \frac{1}{(n-1)^p} \cdot \left(\text{volume of parallelotope formed by } \mathbf{d}_1, \ldots, \mathbf{d}_p\right)^2$$

So |S| literally measures the **squared volume of the data cloud** (up to a scaling constant).

#### 3.1.2 The Three Cases

| Case | |S| | What it means |
|------|----|---------------|
| **Variables are uncorrelated** | = s₁² · s₂² · ... · sₚ² (product of variances) | Maximum possible volume for given individual spreads |
| **Variables are correlated** | < product of variances | Some volume "collapses" due to alignment |
| **Perfect linear dependency** | = 0 | Two (or more) variables are linearly dependent → parallelotope has zero volume |

**Worked example (2 variables):**

$$\mathbf{S} = \begin{pmatrix} 4 & r \cdot 2 \cdot 3 \\ r \cdot 2 \cdot 3 & 9 \end{pmatrix} = \begin{pmatrix} 4 & 6r \\ 6r & 9 \end{pmatrix}$$

where s₁ = 2, s₂ = 3, and r is the correlation. Then:

$$|\mathbf{S}| = 4 \cdot 9 - (6r)^2 = 36 - 36r^2 = 36(1 - r^2)$$

Observe:
- r = 0 → |S| = 36 = s₁² · s₂² ✓ (maximum)
- r = 0.8 → |S| = 36(1 − 0.64) = 12.96
- r = 1 → |S| = 0 (perfect correlation → singular)

**General formula** (generalizes from 2D):
$$|\mathbf{S}| = s_1^2 \cdot s_2^2 \cdots s_p^2 \cdot |\mathbf{R}|$$

where **R** is the correlation matrix. This decomposes generalized variance into **individual spreads × correlation structure**.

#### 3.1.3 Total Variance vs. Generalized Variance

We have two summaries of "total spread":

| Measure | Formula | What it captures | Used in |
|---------|---------|-----------------|---------|
| **Total variance** | tr(S) = Σ sₖₖ | Sum of individual spreads; ignores covariance | PCA (trace) |
| **Generalized variance** | \|S\| | Volume of data cloud; accounts for all covariances | Inference, testing |

Neither is "better" — they answer different questions. PCA (coming soon) focuses on tr(S); tests about Σ use |S|.

---

### 3.2 The Wishart Distribution

This is the multivariate generalization of the chi-squared distribution. Let's build up to it carefully.

#### 3.2.1 Warm-Up: Chi-Squared in One Dimension

If $Z \sim N(0,1)$, then $Z^2 \sim \chi^2(1)$.

If $Z_1, Z_2, \ldots, Z_k$ are i.i.d. N(0,1), then $\sum_{i=1}^k Z_i^2 \sim \chi^2(k)$.

More relevantly: if $X_1, \ldots, X_n$ are i.i.d. N(μ, σ²), then:
$$\frac{(n-1)s^2}{\sigma^2} \sim \chi^2(n-1)$$

where $s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2$ is the sample variance. This is the key link between sampling and chi-squared.

#### 3.2.2 The Wishart Distribution — Definition

Now go multivariate. Let $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n$ be a random sample from $N_p(\boldsymbol{0}, \boldsymbol{\Sigma})$ (mean-zero for simplicity).

Define the random matrix:
$$\mathbf{W} = \sum_{i=1}^n \mathbf{x}_i \mathbf{x}_i^T$$

This is a **p×p random matrix** — each term $\mathbf{x}_i \mathbf{x}_i^T$ is a rank-1 outer product. Adding n of them.

We say $\mathbf{W} \sim W_p(n, \boldsymbol{\Sigma})$: **Wishart distribution with p dimensions, n degrees of freedom, and scale matrix Σ**.

**Connection to chi-squared:**
- When p = 1: $W_1(n, \sigma^2)$ is just $\sigma^2 \cdot \chi^2(n)$ ✓ (it reduces to the scalar case)
- The Wishart is the matrix-valued generalization of the scaled chi-squared

#### 3.2.3 The Main Result for Sample Covariance

Here is the theorem we actually use:

**Theorem:** Let $\mathbf{x}_1, \ldots, \mathbf{x}_n$ be i.i.d. $N_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ (note: non-zero mean now). Then:

$$\boxed{(n-1)\mathbf{S} \sim W_p(n-1, \boldsymbol{\Sigma})}$$

**Reading this:**
- $(n-1)\mathbf{S}$ is a p×p random matrix following a Wishart distribution
- Degrees of freedom: $n - 1$ (we lose 1 df by estimating μ with x̄)
- Scale matrix: Σ (the true covariance)
- Compare: $(n-1)s^2/\sigma^2 \sim \chi^2(n-1)$ in one dimension — same structure!

#### 3.2.4 Key Properties of the Wishart

| Property | Statement |
|----------|-----------|
| **Mean** | E[W] = n · Σ |
| **Degrees of freedom add** | If W₁ ~ Wₚ(n₁, Σ) and W₂ ~ Wₚ(n₂, Σ) independent, then W₁ + W₂ ~ Wₚ(n₁+n₂, Σ) |
| **Positive definiteness** | W is positive definite (with probability 1) when n ≥ p |
| **Connection to det** | |W| / |Σ|ⁿ follows a product of chi-squared distributions |
| **Independence** | x̄ and S are **independent** under MVN — just like x̄ and s² in the univariate case! |

The last property is **crucial**: it means we can analyze the mean and the covariance separately.

---

### 3.3 Sampling Distribution of x̄

Under random sampling from $N_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:

$$\bar{\mathbf{x}} = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i \sim N_p\!\left(\boldsymbol{\mu},\ \frac{1}{n}\boldsymbol{\Sigma}\right)$$

**Why?** Each $\mathbf{x}_i$ is MVN. A linear combination of MVN is MVN (from L4). The mean of n i.i.d. copies has mean μ and covariance (1/n)Σ.

**Standardizing:**

$$\sqrt{n}(\bar{\mathbf{x}} - \boldsymbol{\mu}) \sim N_p(\mathbf{0},\ \boldsymbol{\Sigma})$$

This is the starting point for all multivariate hypothesis tests on the mean (Hotelling's T², coming soon).

**The Mahalanobis distance of x̄:**

$$n(\bar{\mathbf{x}} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\bar{\mathbf{x}} - \boldsymbol{\mu}) \sim \chi^2(p)$$

This is a p-dimensional chi-squared — because x̄ lives in p dimensions, and this is the squared Mahalanobis distance of a p-variate normal.

---

### 3.4 The Big Picture: A Roadmap to Testing

Here's where all these pieces connect:

```
Population: Nₚ(μ, Σ)
         ↓ random sample of size n
Sample:  x̄ (estimates μ)
         S (estimates Σ)

Distributions:
  x̄  ~  Nₚ(μ, Σ/n)
  (n−1)S  ~  Wₚ(n−1, Σ)
  x̄ and S are independent!

Questions we can answer:
  "Is μ = μ₀?" → Hotelling's T² test (uses x̄ and S together)
  "Is Σ = Σ₀?" → Likelihood ratio tests (uses |S|)
  "What are the main axes of variation?" → PCA (uses eigenvectors of S)
```

Everything downstream — Hotelling's T², MANOVA, PCA, discriminant analysis — is built on the Wishart result and the distribution of x̄.

---

## 4. INTERLEAVED PRACTICE (40 min)

*Questions only — answers saved separately. Mix of today's topic and prior material.*
*Answer in words, simple numbers, or Python code — no LaTeX required.*

---

**P1 (Generalized variance — computation):** Given the covariance matrix:
```
S = [[9, 6],
     [6, 9]]
```
(a) Compute |S| by hand (det of a 2×2 is ad − bc).
(b) What are s₁, s₂ (the standard deviations of each variable)?
(c) Use the formula |S| = s₁² · s₂² · |R| to find |R|. From this, what is the correlation r₁₂?
(d) Does the value of r₁₂ make sense given the numbers in S? Explain briefly.

---

**P2 (Generalized variance — Python):**
```python
import numpy as np

S = np.array([[9., 6.],
              [6., 9.]])
```
(a) Use `np.linalg.det(S)` to find |S|. Does it match your answer from P1?
(b) Use `np.linalg.eigvals(S)` to find the eigenvalues. What is the product of the eigenvalues? Does it equal |S|? (This is always true — why?)

---

**P3 (Wishart — conceptual):** Suppose we draw n = 20 observations from a 3-variate normal N₃(μ, Σ).
(a) What distribution does (n−1)S follow? Write it out with the specific numbers.
(b) What are the degrees of freedom? Why isn't it n = 20?
(c) We said x̄ and S are independent. In the univariate case, x̄ and s² are also independent (under normality). What famous theorem does the univariate version come from?

---

**P4 (Sampling distribution of x̄):** We sample n = 25 observations from N₂(μ, Σ) where:
```
μ = [3, 7]
Σ = [[4, 2],
     [2, 1]]
```
(a) What is the mean of x̄? (Easy — write it out.)
(b) What is the covariance matrix of x̄? (Use the formula — show your arithmetic.)
(c) Something unusual about Σ — check: is Σ positive definite? Use Python or by hand:
```python
np.linalg.eigvals(np.array([[4,2],[2,1]]))
```
What do you find? What does this imply?

---

**P5 (Connecting to earlier: eigenvalues + generalized variance):** A sample of n = 100 observations on p = 3 variables gives a covariance matrix S with eigenvalues:
- λ₁ = 5, λ₂ = 3, λ₃ = 0.5

(a) What is tr(S) (total variance)?
(b) What is |S| (generalized variance)? (Use the product-of-eigenvalues property.)
(c) What fraction of total variance does each eigenvalue account for? This is the **PCA decomposition** — preview of what's coming next week.
(d) If we kept only the first two eigenvalues (and threw away the third), what fraction of total variance would we retain?

---

**P6 (Mahalanobis + chi-squared):** Suppose x̄ is the sample mean from n = 50 observations of a bivariate normal N₂(μ, Σ), and we know μ and Σ exactly. We compute:
```
T = n · (x̄ − μ)ᵀ Σ⁻¹ (x̄ − μ)
```
(a) What distribution does T follow?
(b) If T = 8.5, is this unusual? (Hint: for χ²(2), the 95th percentile is about 5.99 and the 99th is about 9.21.)
(c) In words: what would a large value of T tell you about your sample mean x̄?

---

**P7 (Conceptual synthesis — three questions, short answers):**
(a) We said (n−1)S ~ Wₚ(n−1, Σ). What happens to the distribution of S as n → ∞? (Think about what S converges to.)
(b) The Wishart requires n ≥ p for S to be positive definite (with probability 1). What goes wrong if you have p = 50 variables but only n = 30 observations?
(c) In one sentence: what is the Wishart distribution to covariance matrices as chi-squared is to variances?

---

## 5. SUMMARY + ANKI SUGGESTIONS (5 min)

### Key Takeaways 🗝️

- **|S|** = generalized variance = volume² of data cloud; zero means variables are linearly dependent
- **Formula:** |S| = s₁² · s₂² · ... · sₚ² · |R| — separates individual spreads from correlation structure
- **Wishart distribution:** Wₚ(n, Σ) is the matrix-valued analogue of the scaled chi-squared
- **Key theorem:** (n−1)S ~ Wₚ(n−1, Σ) when sampling from Nₚ(μ, Σ)
- **x̄ ~ Nₚ(μ, Σ/n):** bigger n → tighter sampling distribution → better estimate of μ
- **x̄ and S are independent** under MVN — allows us to analyze mean and variance separately
- **Mahalanobis distance of x̄** ~ χ²(p): the p-dimensional test statistic for "is this mean plausible?"
- Everything downstream (Hotelling's T², PCA, discriminant analysis) is built on these results

### Anki Card Suggestions 🃏

1. **Q:** What distribution does (n−1)S follow when sampling from Nₚ(μ, Σ)?
   **A:** Wₚ(n−1, Σ) — Wishart with n−1 degrees of freedom and scale matrix Σ

2. **Q:** How does Wishart relate to chi-squared? (One sentence)
   **A:** Wishart is the matrix-valued generalization of chi-squared — Wₚ(k, Σ) reduces to σ²·χ²(k) when p=1

3. **Q:** |S| = 0 means... (complete this)
   **A:** At least two variables are perfectly linearly dependent (S is singular)

4. **Q:** Formula: |S| in terms of individual variances and the correlation matrix?
   **A:** |S| = s₁² · s₂² · ... · sₚ² · |R|

5. **Cloze:** Under MVN, x̄ and S are ___ of each other.
   **A:** independent

6. **Q:** The statistic n(x̄ − μ)ᵀ Σ⁻¹ (x̄ − μ) follows what distribution?
   **A:** χ²(p) — chi-squared with p degrees of freedom

### Tomorrow's Topic Preview 🔭

**Principal Component Analysis (PCA) — Part 1: The Idea**
- Finding the axes of maximum variance in multivariate data
- Eigendecomposition of S in action
- "What directions capture the most spread?"
- Preview: the fraction-of-variance decomposition you did in P5 today is exactly PCA

This is where all the matrix algebra (L1–L3) and sampling theory (L5–L6) pays off. PCA is the most used technique in the course and a guaranteed exam topic.

---

*This is showing up on Jun 24. The Wishart result is the engine under the hood of every multivariate test. Once you own this, the rest of the course is reading the manual.*

*Estás bien. 20 minutos, luego break. 💙*
