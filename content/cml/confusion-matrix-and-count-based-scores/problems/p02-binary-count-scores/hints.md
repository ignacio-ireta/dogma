## Hint 1 — concept

Every score here is a ratio of two counts from the same 2 x 2 table. The table has four cells, and each score picks a different row or column as its denominator. The hard parts are the zero denominators and the size of the MCC product.

## Hint 2 — formula / approach

Validate the four counts first. Compute the four marginal sums (predicted positive, actual positive, predicted negative, actual negative), then use a helper that returns 0.0 when the denominator is 0. For the MCC, compute the product of the four marginals as an exact integer before taking the square root, and return 0.0 when that product is 0.

## Hint 3 — code shape

Start with a validation helper that accepts integer types and raises `ValueError` otherwise. Then `total = tp + fp + fn + tn`, a check for `total == 0`, and the marginals. Build the result dictionary with one line per key, each calling `_ratio(numerator, denominator)`, and compute `mcc` on its own line.
