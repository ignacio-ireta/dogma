## Hint 1 — concept

The autocorrelation at lag k measures how well a series predicts itself k steps later, after the mean is
removed. Every lag should be scaled by one common total: the sum of squared deviations of the whole series.

## Hint 2 — formula / approach

Centre the series once: d = x - mean(x). The denominator is the dot product of d with itself, over all n
values. The lag-k numerator is the dot product of d[0 : n - k] with d[k : n], which has n - k products. Divide
it by the same denominator, not by n - k. Entry 0 then comes out as exactly 1.

## Hint 3 — code shape

Check the arguments first and raise ValueError for the bad cases. A constant series is detected by comparing
every value with the first one, because the mean of a repeated decimal is not always exactly that decimal.
Allocate an array of length nlags + 1, then fill one entry per lag with np.dot(d[: n - k], d[k:]) / denominator.
