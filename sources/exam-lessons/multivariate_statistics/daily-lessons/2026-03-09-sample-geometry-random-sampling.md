# MVS Lesson 5: Sample Geometry & Random Sampling
**Date:** March 9, 2026 | **Topic:** From Population to Data — The Sample Mean, Covariance, and Geometry
**Syllabus position:** Topic 3 — Sample Geometry, Random Sampling, Generalized Variance

---

## 1. SPACED RETRIEVAL WARM-UP (10 min)

*It's been 10 days since our last session. All questions below are from L1–L4 (Feb 23–27). No peeking — the struggle is the learning.*

**WU1 (L4 — Feb 27):** The multivariate normal density contains the term **(x − μ)ᵀ Σ⁻¹ (x − μ)**. What is this quantity called, and what does it measure geometrically? What shape do the contours of constant value form in 2D?

**WU2 (L3 — Feb 25):** A matrix A is called **positive definite** if it passes a certain test involving quadratic forms. State that test. Then: if A has eigenvalues λ₁ = 4, λ₂ = 1, λ₃ = −0.5 — is A positive definite? (Answer without computing anything.)

**WU3 (L2 — Feb 24):** For the matrix
```
A = [[5, 1],
     [1, 5]]
```
Set up the equation to find the eigenvalues. You don't need to fully solve it — just write the characteristic equation and identify what the eigenvalues will be. (Hint: use det(A − λI) = 0.)

**WU4 (L4 — Feb 27):** True or false, and explain in one sentence each:
- (a) If X ~ Nₚ(μ, Σ), then any single component Xᵢ is also normally distributed.
- (b) If X₁ and X₂ are jointly multivariate normal and Cov(X₁, X₂) = 0, then X₁ and X₂ are independent.

*Take a genuine minute on each before moving on. Partial answers count.*

---

## 2. PRE-TEST: Sample Geometry (5 min)

*Before I teach — guess. Wrong answers are productive.*

**PT1:** Suppose you have 50 people, and for each person you measured 3 things: height, weight, and age. How would you organize this data into a matrix? What are the dimensions?

**PT2:** What do you think the "sample covariance matrix" looks like for 3 variables? Is it 3×3 or 50×3? What do you think sits on the diagonal vs. off-diagonal?

**PT3:** Here's a strange question: could you ever think of the *columns* of your data matrix as vectors in n-dimensional space (n = 50 people)? What would the angle between two such column vectors represent, geometrically?

---

## 3. EXPLICIT INSTRUCTION: Sample Geometry & Random Sampling (30 min)

### 3.1 The Setup: Your Data Matrix

We have **n observations** of **p variables**. Organize this as an **n×p matrix** (rows = observations, columns = variables):

$$\mathbf{X} = \begin{pmatrix} x_{11} & x_{12} & \cdots & x_{1p} \\ x_{21} & x_{22} & \cdots & x_{2p} \\ \vdots & & \ddots & \vdots \\ x_{n1} & x_{n2} & \cdots & x_{np} \end{pmatrix}$$

**Reading the indices:** $x_{ij}$ = observation $i$, variable $j$.

Think of X as:
- **n row vectors** (each person's measurement profile): $\mathbf{x}'_1, \mathbf{x}'_2, \ldots, \mathbf{x}'_n$, each of length p
- **p column vectors** (one variable's measurements across all people): $\mathbf{y}_1, \mathbf{y}_2, \ldots, \mathbf{y}_p$, each of length n

Both views are useful. You'll use the second one for geometry.

---

### 3.2 The Sample Mean Vector x̄

The **sample mean vector** is just the column-wise means:

$$\bar{\mathbf{x}} = \frac{1}{n} \sum_{j=1}^{n} \mathbf{x}_j = \begin{pmatrix} \bar{x}_1 \\ \bar{x}_2 \\ \vdots \\ \bar{x}_p \end{pmatrix}$$

where $\bar{x}_k = \frac{1}{n}\sum_{j=1}^{n} x_{jk}$ is the mean of the k-th variable.

**In matrix form:**

$$\bar{\mathbf{x}} = \frac{1}{n} \mathbf{X}^T \mathbf{1}_n$$

where **1**ₙ is an n×1 vector of ones. This says: "sum each column of X, then divide by n."

**Example:** 4 people, 2 variables (height in cm, age in years):
```
X = [[170, 25],
     [165, 30],
     [180, 22],
     [175, 28]]
```
x̄ = [mean(170,165,180,175), mean(25,30,22,28)]
   = [172.5, 26.25]

---

### 3.3 The Sample Covariance Matrix S

The **sample covariance matrix** S is a p×p matrix where:
- **Diagonal entries:** $s_{kk} = s_k^2$ = sample variance of variable k
- **Off-diagonal entries:** $s_{jk} = s_{kj}$ = sample covariance between variables j and k

**Definition:**

$$\mathbf{S} = \frac{1}{n-1} \sum_{i=1}^{n} (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T$$

Each term $(\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T$ is a **p×p outer product** (rank-1 matrix). We average n of them.

**In matrix form** (this is the compact, computation-friendly version):

$$\mathbf{S} = \frac{1}{n-1} \left( \mathbf{X}^T \mathbf{X} - n\bar{\mathbf{x}}\bar{\mathbf{x}}^T \right)$$

Or equivalently, defining the **centered data matrix** $\tilde{\mathbf{X}}$ where each row has x̄ subtracted:

$$\mathbf{S} = \frac{1}{n-1} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}}$$

**Why n−1?** Dividing by n−1 (not n) gives an **unbiased estimator** — because using the sample mean x̄ instead of the true μ "uses up" one degree of freedom.

**Key properties of S:**
1. S is **symmetric**: $s_{jk} = s_{kj}$
2. S is **positive semidefinite** (all eigenvalues ≥ 0) — and usually positive definite when n > p
3. S is the sample analogue of the population covariance Σ — as n → ∞, S → Σ

---

### 3.4 The Sample Correlation Matrix R

Correlation standardizes covariance. Define the standard deviations $s_1, s_2, \ldots, s_p$ (the square roots of S's diagonal). Then:

$$r_{jk} = \frac{s_{jk}}{\sqrt{s_{jj} \cdot s_{kk}}} = \frac{s_{jk}}{s_j \cdot s_k}$$

The full correlation matrix:

$$\mathbf{R} = \mathbf{D}^{-1/2} \mathbf{S} \mathbf{D}^{-1/2}$$

where **D** is a diagonal matrix with the variances $s_{11}, s_{22}, \ldots, s_{pp}$ on the diagonal.

**R always has:**
- 1's on the diagonal (each variable correlates perfectly with itself)
- Entries between −1 and 1 off-diagonal
- Same eigenvectors as S (when variables have equal variance)

---

### 3.5 The Geometry of the Sample — Deviation Vectors

Here's the beautiful geometric view. Think of each **variable** as a vector in **n-dimensional space** (one dimension per observation).

For variable k, its column in X is $\mathbf{y}_k = (x_{1k}, x_{2k}, \ldots, x_{nk})^T \in \mathbb{R}^n$.

Define the **mean vector**: $\bar{y}_k \cdot \mathbf{1}_n = (\bar{x}_k, \bar{x}_k, \ldots, \bar{x}_k)^T$

Define the **deviation vector** (centered): $\mathbf{d}_k = \mathbf{y}_k - \bar{x}_k \mathbf{1}_n$

Then:

$$s_{kk} = \frac{1}{n-1} \|\mathbf{d}_k\|^2 \quad \text{(variance = squared length, scaled)}$$

$$s_{jk} = \frac{1}{n-1} \mathbf{d}_j^T \mathbf{d}_k \quad \text{(covariance = dot product, scaled)}$$

$$r_{jk} = \cos\theta_{jk} \quad \text{(correlation = cosine of angle between deviation vectors!)}$$

**This is the punchline:** Correlation is literally the cosine of the angle between two centered variable-vectors in sample space.

- r = 1 → vectors point in the same direction (θ = 0°)
- r = 0 → vectors are orthogonal (θ = 90°)
- r = −1 → vectors point in opposite directions (θ = 180°)

---

### 3.6 Generalized Variance: |S|

For a single variable, variance $s^2$ measures spread. For p variables, we need a single number to summarize the "total spread" of the joint distribution.

**Generalized variance** = **|S|** (the determinant of S)

**Why the determinant?**
- Recall from L2: the determinant measures the "volume scaling factor" of a linear transformation.
- |S| measures the **volume of the data cloud** in p-dimensional space.
- Large |S| → variables spread out, little overlap in information
- |S| = 0 → at least two variables are **perfectly linearly related** (no new information)
- |S| > 0 → variables provide genuinely independent information

**Geometric picture in 2D:**
The deviation vectors $\mathbf{d}_1, \mathbf{d}_2$ form a parallelogram. The area of that parallelogram is proportional to $\sqrt{|S|}$. When the two variables are perfectly correlated (r = 1), the parallelogram collapses to a line — area = 0, so |S| = 0.

**Formula connection:**
$$|\mathbf{S}| = s_1^2 \cdot s_2^2 \cdots s_p^2 \cdot (1 - r_{12}^2)(1 - r_{13|2}^2) \cdots$$
(Partial correlations appear when you expand the determinant recursively — a preview of regression!)

**Total variance** (alternative summary): **tr(S)** = sum of diagonal = $s_1^2 + s_2^2 + \cdots + s_p^2$. This is simpler but ignores the covariance structure. PCA will use this.

---

### 3.7 Random Sampling: When Does This All Work?

The formulas above are **descriptive statistics** — they describe your sample. For **inference** (drawing conclusions about the population), we need assumptions.

**The standard setup:**
> We observe $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n$ as a **random sample** from some population with mean vector **μ** and covariance matrix **Σ**.

**What this means:**
- Each $\mathbf{x}_i$ is drawn independently from the same distribution (i.i.d.)
- x̄ is an **unbiased estimator** of μ: E[x̄] = μ
- S is an **unbiased estimator** of Σ: E[S] = Σ

**Under multivariate normality** (which the rest of the course assumes), we also know the exact distributions of x̄ and S:
- $\bar{\mathbf{x}} \sim N_p(\boldsymbol{\mu}, \frac{1}{n}\boldsymbol{\Sigma})$
- $(n-1)\mathbf{S} \sim W_p(n-1, \boldsymbol{\Sigma})$ — a **Wishart distribution** (the multivariate generalization of chi-squared)

The Wishart distribution is to the covariance matrix what χ² is to the variance. We'll use it in later topics (hypothesis testing on Σ).

---

## 4. INTERLEAVED PRACTICE (40 min)

*Questions only — answers saved separately. Mix of today's topic and earlier material.*
*Aim: words, numbers, or Python code — no LaTeX required in your answers.*

---

**P1 (Sample mean — computation):** You have the following data (3 observations, 2 variables):
```
X = [[2, 8],
     [4, 6],
     [6, 4]]
```
(a) Compute the sample mean vector x̄ by hand.
(b) Write a Python one-liner using NumPy to confirm: `X = np.array([[2,8],[4,6],[6,4]])` — what command gives x̄?

---

**P2 (Sample covariance — conceptual):** Using the same X from P1:
(a) Without computing anything — looking at the data, do you expect s₁₂ (covariance between variable 1 and variable 2) to be positive, negative, or zero? Explain in one sentence why.
(b) What do you expect the correlation r₁₂ to be approximately (close to +1, close to 0, or close to −1)?

---

**P3 (Geometry + old topic: eigenvalues):** Suppose a 2×2 sample covariance matrix S has eigenvalues λ₁ = 9, λ₂ = 1.
(a) What is the generalized variance |S|?
(b) What is the total variance tr(S)?
(c) What fraction of the "total variance" does the first eigenvalue account for? (This is the essence of PCA — a preview!)

---

**P4 (Sample covariance — Python):** Write Python code (NumPy) to compute the sample covariance matrix of this dataset:
```python
import numpy as np
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])
```
(a) Use `np.cov(X, rowvar=False)` — what does `rowvar=False` mean here?
(b) Visually examine the output — do you notice anything surprising about this covariance matrix? (Hint: look at the relationship between the three variables.)

---

**P5 (Connecting to MVN — old topic):** Suppose we have a random sample x₁, ..., x₅₀ from a bivariate normal N₂(μ, Σ), where:
```
μ = [0, 0]
Σ = [[4, 2],
     [2, 1]]
```
(a) Is Σ positive definite? Answer by checking the eigenvalues — but first tell me: what's a quick shortcut test using just det(Σ) and the diagonal entries?
(b) What does it mean that one of the eigenvalues of Σ might be zero (if it is)? What does that imply about the joint distribution of X₁ and X₂?

---

**P6 (Deviation vectors + correlation):** Here's a mini-dataset (n=4 observations, 2 variables):
```
y₁ = [1, 3, 5, 7]   (variable 1)
y₂ = [2, 4, 6, 8]   (variable 2)
```
(a) Compute the deviation vectors d₁ and d₂ (subtract each variable's mean).
(b) Compute d₁ · d₂ (dot product). Then compute ||d₁|| and ||d₂|| (lengths).
(c) Use the formula r = cos(θ) = (d₁ · d₂) / (||d₁|| · ||d₂||) to find the correlation. Does the value make sense given the raw data?

---

**P7 (Generalized variance — conceptual):** For each scenario, say whether |S| would be large, small, or zero — and explain briefly:
(a) X₁ = income, X₂ = income × 1.1 (10% more income, always)
(b) X₁ = height, X₂ = weight (measured in 500 unrelated people, moderate correlation r ≈ 0.5)
(c) X₁ = test score, X₂ = number of hours studied, X₃ = height (nearly uncorrelated with the others)

---

## 5. SUMMARY + ANKI SUGGESTIONS (5 min)

### Key Takeaways

- **Data matrix X** is n×p (rows = observations, columns = variables)
- **x̄** (p×1) = column means; estimates μ
- **S** (p×p) = sample covariance; estimates Σ; uses n−1 in denominator (unbiased)
- **S = (1/(n−1)) X̃ᵀX̃** where X̃ is the centered data matrix
- **r_{jk} = cos(θ)** between deviation vectors — correlation is geometry!
- **|S|** = generalized variance = volume of data cloud; 0 means redundant variables
- **tr(S)** = total variance = sum of individual variances
- **Under MVN:** x̄ ~ Nₚ(μ, Σ/n) and (n−1)S ~ Wishart distribution

### Anki Card Suggestions 🃏

1. **Q:** What are the dimensions of the sample covariance matrix S for p=4 variables?
   **A:** 4×4

2. **Q:** Why do we divide by n−1 instead of n in the sample covariance?
   **A:** Using x̄ instead of μ costs one degree of freedom; n−1 makes S unbiased

3. **Q:** What does it mean geometrically when the correlation r₁₂ = 0?
   **A:** The deviation vectors d₁ and d₂ are orthogonal (perpendicular) in n-dimensional space

4. **Q:** |S| = 0 means...?
   **A:** At least two variables are perfectly linearly dependent — no new information

5. **Q:** What distribution does (n−1)S follow when sampling from Nₚ(μ, Σ)?
   **A:** Wishart distribution Wₚ(n−1, Σ)

6. **Cloze:** Total variance = ___ (one operation on S) | Generalized variance = ___ (different operation on S)
   **A:** tr(S) | |S|

### Tomorrow's Topic Preview 🔭

**Generalized Variance (deeper) + The Wishart Distribution**
- When is |S| large vs. small vs. zero? (We scratched the surface today)
- The Wishart distribution: what it is, why it matters for inference
- Sampling distributions of x̄ and S — used in hypothesis testing

This connects everything: our matrix algebra (L1-L3) + MVN (L4) + today's sample stats → the machinery for real statistical tests.

---

*This could show up on Jun 24. The covariance matrix S is the heart of classical multivariate statistics — PCA, factor analysis, canonical correlation, and discriminant analysis all live inside it.*

*20 more minutes, then you move on. You've got this. 💙*
