## Hint 1 — concept

PSI compares two binned distributions with one share per bin. The bins must be fixed by the reference sample.
If the drifted sample defined its own bins, it would fit them and the drift would disappear from the number.

## Hint 2 — formula / approach

Take the interior edges as the expected sample's quantiles at k / bins for k = 1 .. bins - 1. Assign each value,
in both samples, to a bin by counting how many interior edges lie strictly below it, which is one vectorised
search. Turn the counts into shares, floor each zero share at eps, then add up (q - p) * ln(q / p) over the bins.

## Hint 3 — code shape

Validate every argument first. Compute the edges, then the bin index of every value in both samples, then one
count vector per sample with exactly `bins` entries. Divide by the sample sizes, replace zeros, and return the
sum of the terms as a float.
