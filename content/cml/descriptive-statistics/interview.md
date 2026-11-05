# Interview notes: Descriptive statistics

## Q1. When would you report the median rather than the mean?

**60-second answer.** When the data are skewed or contain outliers. The mean is the value that minimises
squared error, so a single extreme value can move it without bound. The median is the middle order
statistic. Its breakdown point is 50%: up to half the sample can be changed arbitrarily without moving
the median out of the original range. Latency percentiles and incomes are typical cases. I would report
the mean and the median together, because the gap between them is itself a signal about skew, and I
would pair the median with the IQR for spread, since the standard deviation is also sensitive to outliers.

**Follow-ups.**
- *What is the breakdown point of the mean?* Zero. One value can move it to any point.
- *Is the median always better?* No. It discards information about magnitudes and is less efficient than the
  mean for symmetric, light-tailed data.

## Q2. Why do we divide by n − 1 for the sample variance?

**60-second answer.** The sample mean is computed from the same data, so the deviations from it sum to zero and
only $n-1$ of them are free. Dividing the sum of squared deviations by $n-1$ gives an unbiased estimate of
the population variance. Dividing by $n$ underestimates it, by a factor of $(n-1)/n$. For large $n$ the difference
is negligible, but for small samples it matters. The two conventions differ between libraries, so it is worth
checking which one a result uses.

**Follow-ups.**
- *Is $n-1$ the best choice?* It is unbiased for the variance, not for the standard deviation. The standard
  deviation is still biased, slightly, for small $n$.
- *What does NumPy use by default?* `ddof=0`. pandas uses `ddof=1`.

## Q3. How do you compute a variance over a stream of numbers without storing them?

**60-second answer.** Use Welford's one-pass update. Keep the count, the running mean and $M_2$, the sum of
squared deviations. For each new value, update the mean, then add the product of the old and new deviations
to $M_2$. The variance is $M_2/(n-1)$. This needs constant memory and is numerically stable. The naive method,
accumulating $\sum x$ and $\sum x^2$, fails when the values are large relative to their spread, because the
subtraction $E[x^2] - E[x]^2$ cancels most of the significant digits.

**Follow-ups.**
- *How bad can it get?* With values near $10^8$ and a spread of about 1, the naive formula gave $-4.10$ for a true variance of $0.989$, while Welford kept about ten correct digits.
- *Can the streaming update be merged across machines?* Yes. Pairwise combination of counts, means and $M_2$
  works (the Chan et al. formula), which is how distributed systems compute variances.

## Common mistakes

- Mixing `ddof` between libraries, which silently changes the variance by a factor of $n/(n-1)$.
- Taking the mode of continuous data without binning it first.
- Using the mean and standard deviation to flag outliers in data that already has outliers.
- Computing the variance with $\sum x^2 - n\bar{x}^2$ on large values.
- Forgetting that a sample of size 1 has no sample variance.
