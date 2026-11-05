## Hint 1 — concept

A window moves by one value per step, so most of its statistics are still valid after a step. Keep the
count, the mean and the sum of squared deviations of the current window. Update them when a value enters
and when one leaves. Summing raw values or their squares over a long series loses digits.

## Hint 2 — formula / approach

Adding x with count n: delta = x - mean, mean += delta / n, M2 += delta * (x - mean_new). Removing the
value old from a window of n values: mean_new = mean + (mean - old) / (n - 1), and M2 -= (old - mean_new)
* (old - mean). The sample standard deviation is sqrt(M2 / (n - 1)). Centre the values first by
subtracting a reference value near the data, then add it back to the means. A large common offset
otherwise costs digits over a long series. Rescale deviations before squaring at extreme magnitudes; rescale the running mean and M2 consistently.

## Hint 3 — code shape

Validate the arguments before the loop. Loop once over the indices i: add x[i]; if the count is above
window, remove x[i - window]; if the count is at least min_periods, write the mean at i, and write the
standard deviation at i only when the count is above one. Return the two arrays.
