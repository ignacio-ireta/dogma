## Hint 1 — concept

A seasonal naive forecast copies the value from the same position in the last complete season. MASE compares
your errors with the errors that the same naive rule made on the training data, so the scale must come from
the training series alone.

## Hint 2 — formula / approach

Counting train positions from 0, step h (starting at 1) reads position n - season + ((h - 1) mod season). For the
scale, take the training values that have a partner season steps earlier: |train[season:] - train[:-season]|
has n - season terms, so the mean divides by n - season. MASE is the mean absolute error of the forecast divided
by that scale.

## Hint 3 — code shape

Validate every argument first. Build the forecast with one index array, for example
positions = n - season + (np.arange(H) % season), then index train with it. Compute the scale from the two
slices, and raise ValueError when it is zero. Return the forecast and the MASE as a float.
