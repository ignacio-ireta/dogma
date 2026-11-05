## Hint 1 — concept

A median and an interquartile range are fitted from the training rows only, and they describe the
middle of the data. Outliers live in the tails, so they do not move these two statistics much, which
is the whole reason to use them.

## Hint 2 — formula / approach

For each column, sort a copy of the training values. The median is the 0.5 quantile. The IQR is the
0.75 quantile minus the 0.25 quantile, both by linear interpolation at position `q * (n - 1)`. If the
IQR is below ten machine epsilons, use 1.0 instead.

## Hint 3 — code shape

Validate the two arrays, then loop over the columns of the training array. Inside the loop, sort the
column, take the three quantiles with one helper function, and store the center and scale. Finally
broadcast `(values - center) / scale`.
