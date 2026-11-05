## Hint 1 — concept

Fit and transform are separate operations. The rows being transformed must not
influence the fitted coordinate system.

## Hint 2 — formula / approach

Compute one mean and population standard deviation per column of training data.
An exactly constant training column needs scale one; a small nonzero scale does not.

## Hint 3 — code shape

Validate both two-dimensional arrays, compute training statistics along axis zero,
and broadcast their one-dimensional arrays across values. Rescaling centered
training columns before squaring avoids overflow and underflow.
