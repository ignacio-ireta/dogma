# Probability & Statistics Refresher

*The stuff you need for Module 1. No more, no less.*

---

## ⚠️ Module 1 Critical Rules

1. **Answers must be rounded to EXACTLY 3 decimal places.** Not 2, not 4. Three. Always.
2. **The word problems are deliberately verbose.** The reading comprehension IS the difficulty. Extract the math variables, ignore the narrative.
3. **Two questions, target ≤10 minutes total.** These are free points if you don't overthink.
4. **Confirmed question types:** Binomial distribution, Bayes' theorem, expected value, conditional probability, card-drawing (dependent events).
5. **You cannot use a calculator app.** Do the arithmetic by hand or in your head.

---

## Binomial Distribution

**When to use:** Fixed number of independent trials, each with the same probability of success. "Out of n tries, what's the probability of exactly k successes?"

**Formula:**
```
P(X = k) = C(n, k) × p^k × (1-p)^(n-k)
```

Where:
- `n` = number of trials
- `k` = number of successes you want
- `p` = probability of success on each trial
- `C(n,k)` = n! / (k! × (n-k)!) = "n choose k"

**Python:**
```python
from scipy.stats import binom

# P(exactly 3 successes in 10 trials, p=0.5)
binom.pmf(k=3, n=10, p=0.5)

# P(at least 3 successes) = P(X >= 3) = 1 - P(X <= 2)
1 - binom.cdf(k=2, n=10, p=0.5)

# P(at most 3 successes) = P(X <= 3)
binom.cdf(k=3, n=10, p=0.5)
```

**Example (CodeSignal style):**
A class has 30 students, each with 70% chance of passing. P(at least 25 pass)?
```python
from scipy.stats import binom
p = 1 - binom.cdf(k=24, n=30, p=0.7)  # P(X >= 25) = 1 - P(X <= 24)
# ≈ 0.15
```

---

## Bayes' Theorem

**When to use:** You know P(B|A) and want P(A|B). Updating beliefs with new evidence.

**Formula:**
```
P(A|B) = P(B|A) × P(A) / P(B)
```

Where:
- `P(A|B)` = probability of A given B happened (posterior)
- `P(B|A)` = probability of B given A (likelihood)
- `P(A)` = prior probability of A
- `P(B)` = total probability of B

**The classic disease example:**
- 1% of population has disease: P(D) = 0.01
- Test is 99% accurate for sick people: P(+|D) = 0.99
- Test has 5% false positive rate: P(+|¬D) = 0.05
- You test positive. What's the chance you're sick?

```
P(D|+) = P(+|D) × P(D) / P(+)
P(+) = P(+|D)×P(D) + P(+|¬D)×P(¬D) = 0.99×0.01 + 0.05×0.99 = 0.0594
P(D|+) = 0.99 × 0.01 / 0.0594 ≈ 0.167 → only 16.7%!
```

**Python:**
```python
p_d = 0.01
p_pos_given_d = 0.99
p_pos_given_not_d = 0.05
p_not_d = 1 - p_d

p_pos = p_pos_given_d * p_d + p_pos_given_not_d * p_not_d
p_d_given_pos = (p_pos_given_d * p_d) / p_pos
# 0.167
```

**Interview insight:** Bayes shows why even accurate tests can have high false positive rates when the base rate is low. Sound familiar? It's the same problem as a churn model with a 6% churn rate means most "flagged" customers might be false alarms.

---

## Expected Value

**What it is:** The average outcome you'd expect over many repetitions.

**Formula:**
```
E[X] = Σ (x_i × P(x_i))
```

**Example:** Roll a fair die. E[X] = 1×(1/6) + 2×(1/6) + ... + 6×(1/6) = 3.5

**Weighted example:** A game costs $10 to play. You win $50 with probability 0.1, $20 with probability 0.2, and $0 otherwise.
```
E[profit] = 50×0.1 + 20×0.2 + 0×0.7 - 10 = 5 + 4 + 0 - 10 = -$1
```
Negative expected value → don't play.

---

## Key Distributions Cheatsheet

| Distribution | When to Use | Parameters | Python |
|-------------|-------------|------------|--------|
| **Binomial** | Fixed n trials, k successes | n, p | `binom.pmf(k, n, p)` |
| **Normal** | Continuous, bell-shaped | μ, σ | `norm.pdf(x, loc=μ, scale=σ)` |
| **Poisson** | Count of events in a time window | λ (rate) | `poisson.pmf(k, mu=λ)` |
| **Bernoulli** | Single trial, success/fail | p | `bernoulli.pmf(k, p)` |
| **Uniform** | All outcomes equally likely | a, b | `uniform.pdf(x, loc=a, scale=b-a)` |

```python
from scipy.stats import binom, norm, poisson

# Normal: P(X < 100) with μ=80, σ=15
norm.cdf(100, loc=80, scale=15)

# Poisson: P(exactly 3 events) when average is 5/hour
poisson.pmf(3, mu=5)
```

---

## Descriptive Statistics — Quick Formulas

```python
# All in pandas
df["col"].mean()       # average
df["col"].median()     # middle value (robust to outliers)
df["col"].mode()       # most frequent
df["col"].std()        # standard deviation
df["col"].var()        # variance
df["col"].quantile(0.75)  # 75th percentile

# Correlation
df[["a", "b"]].corr()  # Pearson correlation matrix
```

**When mean ≠ median:** The data is skewed. If mean > median → right-skewed (long tail on right, e.g., income). If mean < median → left-skewed.

---

## Hypothesis Testing (Conceptual)

**p-value:** The probability of seeing your result (or more extreme) IF the null hypothesis is true.
- p < 0.05 → reject null → result is "statistically significant"
- p ≥ 0.05 → fail to reject null → not enough evidence

**Type I error (false positive):** Rejecting the null when it's actually true. "Crying wolf."
**Type II error (false negative):** Failing to reject the null when it's actually false. "Missing the wolf."

**Confidence interval:** A range that contains the true parameter with X% probability.
- 95% CI means: if you repeated the experiment 100 times, ~95 of those CIs would contain the true value.

---

## Dependent Probability (Cards Without Replacement)

**This is a confirmed CodeSignal question type.** Drawing from a deck without replacement means the denominator changes.

**Example:** Draw 2 cards from a standard 52-card deck. P(Ace first, then King)?
```
P(Ace first) = 4/52
P(King second | Ace first) = 4/51  (deck now has 51 cards, still 4 Kings)
P(Ace then King) = (4/52) × (4/51) = 16/2652 ≈ 0.006
```

**Example:** P(both cards are Hearts)?
```
P(1st Heart) = 13/52
P(2nd Heart | 1st Heart) = 12/51
P(both Hearts) = (13/52) × (12/51) = 156/2652 ≈ 0.059
```

**Key insight:** After each draw without replacement:
- Numerator changes IF you drew from that category
- Denominator ALWAYS decreases by 1

---

## Z-Scores and Normal Distribution

**Z-score:** How many standard deviations a value is from the mean.
```
z = (x - μ) / σ
```

**Common z-values to memorize:**
- z = 1.0 → 84.1% below (15.9% above)
- z = 1.645 → 95% below (one-tail 5%)
- z = 1.96 → 97.5% below (two-tail 5% → 95% CI)
- z = 2.0 → 97.7% below
- z = 2.576 → 99.5% below (two-tail 1% → 99% CI)

**68-95-99.7 rule:** Within 1σ = 68%, 2σ = 95%, 3σ = 99.7%

```python
from scipy.stats import norm

# P(X < 100) given μ=80, σ=15
norm.cdf(100, loc=80, scale=15)  # = norm.cdf((100-80)/15) = norm.cdf(1.33)

# What value is at the 95th percentile?
norm.ppf(0.95, loc=80, scale=15)  # inverse CDF
```

---

## t-Test (Conceptual + Python)

**When to use:** Comparing means of two groups when population σ is unknown (which is almost always).

**Types:**
- **Independent t-test:** Are the means of two separate groups different? (e.g., treatment vs control)
- **Paired t-test:** Are the means of the same group different at two time points? (e.g., before vs after)

```python
from scipy.stats import ttest_ind, ttest_rel

# Independent: are group A and group B means different?
t_stat, p_value = ttest_ind(group_a, group_b)

# Paired: same subjects, two measurements
t_stat, p_value = ttest_rel(before, after)

# Interpret
if p_value < 0.05:
    print("Statistically significant difference")
```

**Key assumptions:** Data is approximately normal, observations are independent, equal variance (use `equal_var=False` for Welch's t-test if unsure).

---

## Selection Bias

**What it is:** When your sample doesn't represent the population. Your model learns patterns that don't generalize.

**Examples:**
- Survivorship bias: only analyzing companies that survived (ignoring failures)
- Self-selection: survey respondents differ from non-respondents
- Convenience sampling: using only easily available data

**Why it matters for CodeSignal:** A word problem might describe a sampling method and ask you to identify the bias. Or ask how a biased sample affects your conclusions.

---

## Common Traps in Scenario Questions

1. **"At least" problems:** P(X ≥ k) = 1 - P(X ≤ k-1). Don't forget the minus 1.
2. **Independent vs dependent events:** Drawing cards WITH replacement = independent. WITHOUT replacement = dependent (use conditional probability).
3. **Base rate neglect:** When the base rate is very low, even good tests have many false positives (Bayes!).
4. **Confusing P(A|B) with P(B|A):** "The probability of rain given clouds" ≠ "the probability of clouds given rain."
5. **Rounding too early:** Do all arithmetic first, round ONLY the final answer to 3 decimals.
6. **Complement rule:** If "at least one" is hard to calculate directly, use P(at least 1) = 1 - P(none).

---

## Practice Problems

**P1.** A factory produces items with 2% defect rate. You randomly select 50 items. What's the probability of finding exactly 3 defective items?

**P2.** A spam filter catches 95% of spam but also flags 3% of legitimate emails. If 20% of all emails are spam, what's the probability an email flagged as spam is actually spam?

**P3.** A die is rolled 12 times. What's the expected number of 6s? What's the probability of getting MORE than 3 sixes?

**P4.** Customer support gets an average of 8 calls per hour (Poisson). What's the probability of getting exactly 5 calls in a given hour? At least 10?

**P5.** Two machines produce widgets. Machine A (60% of output) has 1% defect rate. Machine B (40% of output) has 3% defect rate. A random widget is defective. What's the probability it came from Machine B?

---

### 🆕 Confirmed CodeSignal-Style Problems

**P6.** (Card drawing) You draw 2 cards from a standard 52-card deck without replacement. What is the probability that the first card is a Queen and the second card is a Jack? Round to 3 decimal places.

**P7.** (Binomial — "exactly k" style) A fair six-sided die is rolled 10 times. What is the probability of rolling a "6" exactly 3 times? Round to 3 decimal places.

**P8.** (Bayes — factory style) Two assembly lines produce widgets. Line A makes 60% of output with 2% defect rate. Line B makes 40% with 3% defect rate. A random widget is defective. What's the probability it came from Line A? Round to 3 decimal places.

**P9.** (Expected value) A game costs $5 to play. You flip a fair coin 3 times. You win $2 for each head. What is your expected profit? Round to 3 decimal places.

**P10.** (Binomial — "at least" style) A multiple choice test has 8 questions, each with 4 options. A student guesses randomly on all questions. What is the probability of getting at least 4 correct? Round to 3 decimal places.

*(Solutions at the bottom)*

---

## Solutions

**P1.** `binom.pmf(3, 50, 0.02)` ≈ 0.061

**P2.** Bayes: P(spam|flagged) = (0.95 × 0.20) / (0.95×0.20 + 0.03×0.80) = 0.19/0.214 ≈ 0.888

**P3.** E[6s] = 12 × (1/6) = 2. P(X > 3) = 1 - binom.cdf(3, 12, 1/6) ≈ 0.125

**P4.** P(X=5) = poisson.pmf(5, 8) ≈ 0.092. P(X≥10) = 1 - poisson.cdf(9, 8) ≈ 0.283

**P5.** P(B|def) = (0.03×0.40) / (0.01×0.60 + 0.03×0.40) = 0.012/0.018 = 0.667

**P6.** P(Q then J) = (4/52) × (4/51) = 16/2652 ≈ **0.006**

**P7.** C(10,3) × (1/6)³ × (5/6)⁷ = 120 × 0.00463 × 0.2791 = 120 × 0.001292 ≈ **0.155**

**P8.** P(A|def) = (0.02×0.60) / (0.02×0.60 + 0.03×0.40) = 0.012/0.024 = **0.500**
(Note: P7's Machine B version = 0.012/0.018 from P5 uses different rates. Read carefully!)

**P9.** E[heads in 3 flips] = 3 × 0.5 = 1.5. E[winnings] = 1.5 × $2 = $3. E[profit] = $3 - $5 = **-2.000**

**P10.** P(X ≥ 4) = 1 - P(X ≤ 3), where X ~ Binomial(n=8, p=0.25).
P(X ≤ 3) = binom.cdf(3, 8, 0.25) ≈ 0.8862. P(X ≥ 4) ≈ **0.114**
