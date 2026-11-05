## Hint 1: concept

The explained-variance ratios say how much of the total variance each component carries, and they add up to 1.
Reaching a threshold means accumulating those ratios in order, from the largest, until the running sum is large
enough. The answer is a count of components, so the position where the running sum first reaches the threshold
matters.

## Hint 2: formula / approach

The variance of component $i$ is $s_i^2/(n-1)$, with $s_i$ the singular values of the centred data. Divide each by the
total variance of all columns, which is the sum of the column variances with divisor $n-1$. Accumulate the ratios
with a cumulative sum. The answer is the first index where the sum reaches the threshold, counted from one. Allow a
margin of $10^{-12}$ below the threshold, so that a tie in exact arithmetic is not lost to rounding.

## Hint 3: code shape

Validate first: 2-D, at least two rows, finite values, a threshold that is a real number in $(0, 1]$ and not a boolean,
and a non-zero total variance. Then:

```python
centred = x - x.mean(axis=0)
_, s, _ = np.linalg.svd(centred, full_matrices=False)
# ratios = (s**2 / (n - 1)) / total_variance
# cumulative = np.cumsum(ratios); find the first index where cumulative >= threshold - margin
# return that index, plus one
```
