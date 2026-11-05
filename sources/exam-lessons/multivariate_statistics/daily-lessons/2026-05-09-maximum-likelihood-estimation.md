# MVS Lesson 9 (Patch): Maximum Likelihood Estimation
**Date:** May 9, 2026 | **Topic:** MLE — The Engine Behind Hypothesis Tests in MVS
**Syllabus position:** Foundation patch — required before Factor Analysis and LR tests

---

## 1. SPACED RETRIEVAL WARM-UP (10 min)

*From Lessons 7-8 (SVD, projections — this week) and Lesson 4 (MVN, Feb 27).*

**WU1 (L7 — SVD):** What is the relationship between singular values of a centered data matrix X and the eigenvalues of the sample covariance S?

**WU2 (L8 — projections):** Write the projection formula. If Vₖ contains the first k right singular vectors, what is the projected data matrix X̂ in terms of the SVD?

**WU3 (L4 — MVN):** Write the MVN density f(x | μ, Σ). What are the two parameters? What role does |Σ| play?

**WU4 (L4 — MVN):** If X₁, ..., Xₙ are iid from Nₚ(μ, Σ), what is the distribution of the sample mean x̄? What happens to its variance as n → ∞?

---

## 2. PRE-TEST: Maximum Likelihood (5 min)

*Before I teach — guess.*

**PT1:** You flip a coin 10 times and get 7 heads. What's your best guess for the probability of heads? How did you decide? (Don't overthink — just give the intuitive answer.)

**PT2:** "Likelihood" and "probability" sound similar but aren't the same. If the MVN density f(x | μ, Σ) gives the probability of data x given parameters (μ, Σ), what do you think the "likelihood" of (μ, Σ) given data x₁,...,xₙ means?

**PT3:** If you have two hypotheses H₀ and H₁, and you compute the maximum likelihood under each, what do you think the ratio L(H₁)/L(H₀) measures? What would a large value mean?

---

## 3. EXPLICIT INSTRUCTION (30 min)

### 3.1 The Core Idea: Invert the Question

**Probability:** Given parameters θ, how likely is the observed data?
```
P(data | θ)
```

**Likelihood:** Given observed data, how "supported" is a particular θ?
```
L(θ | data) = P(data | θ)
```

Same formula. Different perspective. We treat the data as **fixed** (we already observed it) and vary θ to find the value that makes the data most probable.

**Maximum Likelihood Estimation (MLE):** Find the θ that maximizes L(θ | data).

### 3.2 The Coin Example (Concrete)

Flip a coin n times, get k heads. Parameter: θ = P(heads).

```
L(θ) = θᵏ (1-θ)ⁿ⁻ᵏ

log L(θ) = k·log(θ) + (n-k)·log(1-θ)

d/dθ log L = k/θ - (n-k)/(1-θ) = 0

→ θ̂_MLE = k/n
```

For 7 heads out of 10: θ̂ = 7/10 = 0.7. Your intuition was right — and now you know *why*.

### 3.3 MLE for the Multivariate Normal

This is what matters for MVS. Given x₁, ..., xₙ iid from Nₚ(μ, Σ):

```
# Joint density (product of individual densities)
L(μ, Σ) = Πᵢ₌₁ⁿ f(xᵢ | μ, Σ)

# Log-likelihood (easier to maximize)
ℓ(μ, Σ) = -np/2 · log(2π) - n/2 · log|Σ| - 1/2 · Σᵢ (xᵢ-μ)ᵀ Σ⁻¹(xᵢ-μ)
```

**Maximizing over μ:**
```
∂ℓ/∂μ = Σᵢ Σ⁻¹(xᵢ - μ) = 0
→ μ̂_MLE = x̄  (the sample mean)
```

**Maximizing over Σ:**
```
∂ℓ/∂Σ = 0
→ Σ̂_MLE = 1/n · Σᵢ (xᵢ - x̄)(xᵢ - x̄)ᵀ  (the sample covariance, divided by n not n-1)
```

**Important note:** The MLE of Σ divides by **n**, not (n-1). This makes Σ̂_MLE slightly biased (underestimates Σ). The unbiased version divides by (n-1) — that's the S you've been using. For large n, the difference vanishes. But for likelihood ratio tests, we use the MLE version.

### 3.4 Likelihood Ratio Tests — The MVS Testing Framework

Here's where MLE becomes the engine of MVS hypothesis testing.

**Setup:**
- Full model: parameters θ (unrestricted) → maximum likelihood L₁ = max L(θ)
- Restricted model: parameters under H₀ (constrained) → maximum likelihood L₀ = max L(θ | H₀)

**The Likelihood Ratio:**
```
Λ = L₀ / L₁
```

Since L₁ ≥ L₀ (full model always fits at least as well), we have 0 ≤ Λ ≤ 1.

**Wilks' Theorem (the key result):**
Under H₀, as n → ∞:
```
-2 log Λ ~ χ²(df)
```
where df = number of restrictions (difference in free parameters between full and restricted models).

**Small Λ → large -2 log Λ → reject H₀.** The restricted model doesn't fit well enough.

### 3.5 MVS Examples of LR Tests

| Test | H₀ | Full Model | df |
|------|-----|-----------|-----|
| Mean vector | μ = μ₀ | μ unrestricted | p |
| Equality of means | μ₁ = μ₂ | μ₁ ≠ μ₂ | p |
| Covariance structure | Σ = Σ₀ | Σ unrestricted | p(p+1)/2 |
| Sphericity | Σ = σ²I | Σ unrestricted | p(p+1)/2 - 1 |

**For Factor Analysis specifically:**
- H₀: Σ = ΛΛᵀ + Ψ (k-factor model fits)
- H₁: Σ unrestricted
- df = [p(p+1)/2] - [p(k+1) - k(k-1)/2]

This is why MLE matters — every hypothesis test in MVS is some version of "does the restricted model fit adequately?"

### 3.6 Practical Computation (What You Actually Do)

In practice, you won't derive MLEs by hand. But you need to understand:

1. **When software reports "Log-likelihood = -245.3"**, that's ℓ(μ̂, Σ̂) — the maximized log-likelihood.
2. **When it reports "Chi-square = 12.4, df = 5, p = 0.030"**, that's -2 log Λ tested against χ²(5).
3. **When you compare models (3 factors vs 2 factors)**, you're comparing -2(ℓ_restricted - ℓ_full) against χ² with appropriate df.

---

## 4. INTERLEAVED PRACTICE (40 min)

### Conceptual

**P1:** Explain in one sentence the difference between "probability of data given parameters" and "likelihood of parameters given data."

**P2:** Why do we maximize the log-likelihood instead of the likelihood? (Two reasons.)

**P3:** The MLE of the covariance divides by n, but the unbiased estimator divides by n-1. Which do you use for:
- (a) Estimating Σ for descriptive purposes?
- (b) Computing a likelihood ratio test?

### Computation

**P4:** You observe 3 data points from a univariate normal: x₁ = 2, x₂ = 5, x₃ = 6.
- (a) What is μ̂_MLE?
- (b) What is σ²_MLE? (Remember: divide by n.)
- (c) What is the unbiased sample variance s²? (Divide by n-1.)

**P5:** A univariate test: H₀: μ = 3 vs H₁: μ ≠ 3, with n = 50 observations.
- (a) How many restrictions does H₀ impose?
- (b) If -2 log Λ = 5.2, what χ² distribution do you compare to?
- (c) Is this significant at α = 0.05? (χ²(1, 0.95) = 3.84)

### Connection

**P6 (connects L4 + L9):** In Lesson 4 you learned the MVN density. Today you used it for MLE. The MVN log-likelihood has three terms:
```
ℓ = constant - n/2 · log|Σ| - 1/2 · Σᵢ (xᵢ-μ)ᵀΣ⁻¹(xᵢ-μ)
```
- (a) Which term contains the Mahalanobis distances you learned in L4?
- (b) When we maximize over μ, why does only the third term matter?
- (c) When we maximize over Σ, what does log|Σ| penalize? (Hint: what happens to |Σ| when variables are highly correlated?)

---

## 5. SUMMARY + ANKI SUGGESTIONS (5 min)

### Key Takeaways
1. **MLE inverts the question:** instead of P(data | θ), maximize L(θ | data). Same formula, different perspective.
2. **For MVN:** μ̂ = x̄, Σ̂ = (1/n)Σ(xᵢ-x̄)(xᵢ-x̄)ᵀ. MLE of Σ divides by n (biased), not n-1.
3. **Likelihood Ratio Tests:** compare L(restricted) / L(full). Small ratio → reject H₀.
4. **Wilks' Theorem:** -2 log Λ ~ χ²(df) under H₀. This is the backbone of every MVS hypothesis test.
5. **Every MVS test (mean, covariance, factor analysis, discriminant analysis)** is a likelihood ratio test in disguise.

### Anki Cards
- Q: "What is Maximum Likelihood Estimation?" → A: Find the parameter θ that maximizes the probability of observed data. L(θ) = P(data | θ), maximize over θ.
- Q: "What are the MLEs for μ and Σ in a multivariate normal?" → A: μ̂ = x̄ (sample mean), Σ̂ = (1/n)Σ(xᵢ-x̄)(xᵢ-x̄)ᵀ (divides by n, not n-1).
- Q: "What is a likelihood ratio test?" → A: Compare Λ = L₀/L₁ (restricted vs full model). Small Λ means the restriction doesn't fit. Under H₀: -2 log Λ ~ χ²(df).
- Q: "What does Wilks' theorem say?" → A: Under H₀, -2 log Λ converges in distribution to χ² with df = (free params in full model) - (free params in restricted model).
- Q: "Why does the MLE of Σ divide by n instead of n-1?" → A: It maximizes likelihood but is biased (underestimates Σ). The n-1 version is unbiased but isn't the likelihood maximizer.

---

*Likelihood turns observation into inference. The ratio turns inference into a decision.*
