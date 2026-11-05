---
title: Descriptive statistics
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

For a sample $x_1,\dots,x_n$, write $\bar x = \frac1n\sum_i x_i$. The statistics that describe it fall into
two groups: where the data sit, and how spread out they are. Quantiles use the sorted values
$x_{(1)}\le\dots\le x_{(n)}$.

| Statistic | Definition | Robust to outliers? | Use when |
|---|---|---|---|
| Mean | $\bar x$ | no (breakdown 0%) | symmetric data; totals are needed |
| Median | middle order statistic | yes (breakdown 50%) | skewed data; outliers present |
| Mode | most frequent value | not applicable | categories, or discrete counts |
| Sample variance | $s^2 = \frac{1}{n-1}\sum_i (x_i-\bar x)^2$ | no | unbiased estimate of $\sigma^2$ |
| Population variance | $\frac1n\sum_i (x_i-\bar x)^2$ | no | the sample is the whole population |
| IQR | $Q_3 - Q_1$ | yes | spread of the middle half |

**ddof** is the divisor offset: $n - \text{ddof}$. `ddof=1` gives the sample variance, `ddof=0` the population
variance. NumPy's `np.var` defaults to `ddof=0`; pandas' `Series.var` defaults to `ddof=1`. Check which one a
result uses before comparing numbers.

**Quantiles** use linear interpolation: the $q$-quantile sits at position $h = q(n-1)$, counted from zero,
between $x_{(\lfloor h\rfloor+1)}$ and the next value. This is NumPy's default method.

## Worked example

$\mathbf{x} = (1, 3, 4, 8, 9)$.

1. Mean: $25/5 = 5$. Median: the third value, $4$.
2. Deviations: $-4, -2, -1, 3, 4$. Squares: $16, 4, 1, 9, 16$, summing to $46$.
3. Population variance $46/5 = 9.2$. Sample variance $46/4 = 11.5$, and $s = 3.39$.
4. Quartiles: $h = 0.25\cdot 4 = 1$ gives $Q_1 = 3$, and $h = 0.75\cdot 4 = 3$ gives $Q_3 = 8$. So $\mathrm{IQR} = 5$.
   Tukey's fences $Q_1 - 1.5\,\mathrm{IQR} = -4.5$ and $Q_3 + 1.5\,\mathrm{IQR} = 15.5$ mark the outliers.

Now add one extreme value, $100$. The mean becomes $20.8$, and the median moves from $4$ to $6$. The fences become $(-5.0,\ 17.0)$, because the quartiles are $3.25$ and $8.75$. The value $100$ is flagged. The fences move by 1.5 at most, while the mean jumps from $5$ to $20.8$.

## Why it works

**Mean and median.** $\sum_i (x_i - c)^2$ is minimised at $c = \bar x$, because its derivative $-2\sum_i(x_i - c)$
vanishes there. The median minimises $\sum_i |x_i - c|$ instead. Squared error gives large deviations
a large pull, and absolute error does not.

**Why $n-1$.** Since $\sum_i (x_i - \bar x) = 0$, the deviations carry only $n-1$ free values. Averaging
$\sum_i (x_i-\bar x)^2$ over repeated samples gives $(n-1)\sigma^2$, so dividing by $n-1$ is unbiased. The
sample standard deviation is still slightly biased, which matters for small $n$.

**Numerical stability.** The textbook formula $s^2 = \frac{1}{n-1}\big(\sum x_i^2 - n\bar x^2\big)$ subtracts two large,
nearly equal numbers. For $n=1000$ values equal to $10^8$ plus standard normal noise (seed 5), it returned $-4.10$, a negative variance. The true value is $0.98870$. Welford's update avoids the subtraction by tracking the deviation from the running mean. On the same data it gave $0.98870$ with a relative error of $1\times10^{-10}$, in one pass and constant memory.

## Signal vs. noise

- **Signal:** the mean and median respond differently to outliers. Report both when they disagree.
- **Signal:** always state `ddof`, and use a one-pass variance when values are large relative to their spread.
- **Noise for now:** the nine quantile definitions in the literature (Hyndman & Fan). Use the linear one and
  name it.

## Where it breaks

- **Outliers.** A single bad value moves the mean and the standard deviation, and it can fool a rule built
  on them. Use the median and IQR as the first check.
- **Skew.** For right-skewed data the mean sits in the long tail, above most observations.
- **Cancellation.** The naive variance formula fails silently, and can return a negative number. Welford avoids
  it.
- **Mode on continuous data.** Every value is unique, so the mode is meaningless. Bin the data first.
- **Tiny samples.** A sample of one has no sample variance, and it is `nan` by definition.

## Interview angle

"When would you report the median rather than the mean?" Answer shape: when the data are skewed or contain
outliers, because the mean has no breakdown tolerance, while the median tolerates half the sample being
corrupted. Pair it with the IQR for spread. Follow-ups: why $n-1$, and how to compute a variance over a stream
without storing it. Full answers are in `interview.md`.

## Connections

- **Prerequisites:** none. This topic is the vocabulary for the statistics that follow.
- **Unlocks:** `cml/probability-basics` (expectations and variance), `cml/regression-metrics-and-robust-losses`
  (mean squared error versus absolute error), `cml/missing-values-and-outliers`, `cml/feature-scaling` (standardising
  with the mean and the standard deviation).
- **Lineage:** Gauss's least squares uses the mean as the best constant under squared error. Welford's 1962 note
  is the standard one-pass method for corrected sums of squares.
