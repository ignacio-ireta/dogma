## Hint 1 — concept

Entropy is the average surprise of the outcomes: an outcome with probability p is surprising by −log p, and the entropy averages that over the distribution. Two details decide most of the work. An outcome with probability zero must add nothing, because log 0 is −∞, so those entries are skipped rather than multiplied through. The base only changes the unit. Decide what counts as a valid input before computing anything.

## Hint 2 — formula / approach

H(p) = −Σ pᵢ log_b pᵢ, with 0 · log 0 taken as 0. Compute the sum with the natural logarithm and convert once at the end, because log_b x = ln x / ln b. Restrict the sum to the entries with p > 0. Validation comes first: one-dimensional and non-empty; every entry finite and non-negative; the entries sum to 1 within 1e-9; the base finite, greater than 0 and not equal to 1.

## Hint 3 — code shape

Convert the input with `np.asarray(p, dtype=float)` so lists and arrays behave the same. Run the checks in order, one `raise ValueError` for each, each message naming the broken rule. Then keep the positive entries, take the natural log of them, sum the products with a minus sign, and divide by `math.log(base)`. Return a plain `float`. If a hidden test fails, find the rule its message names and check the line that implements it.
