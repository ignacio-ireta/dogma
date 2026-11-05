## Hint 1 — concept

Both the naive formula and a two-branch formula fail for the same reason: they compute an exponential
of a large positive number. Look for a quantity you can exponentiate that is never large and positive.

## Hint 2 — formula / approach

Let $e = \exp(-|z|)$, which lies in $[0, 1]$. For $z \ge 0$, $\sigma(z) = 1/(1+e)$. For $z < 0$,
$\sigma(z) = e/(1+e)$. Both branches use the same $e$. Ask yourself which of these two forms keeps the
relative precision of the small value when $z$ is large and negative.

## Hint 3 — code shape

Convert the input with `np.asarray(z, dtype=np.float64)`, reject NaN with a `ValueError`, compute `e`
once, and combine the two branches with `np.where(z >= 0, ..., ...)`. Then check your function on
$z = \pm 10^4$ with `warnings.simplefilter("error", RuntimeWarning)`. Hint: `np.where` evaluates both
branches for every element, so every branch has to be safe on its own.
