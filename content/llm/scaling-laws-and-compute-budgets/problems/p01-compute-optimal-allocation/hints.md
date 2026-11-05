## Hint 1 — concept

The cost is fixed by two numbers, the parameter count `N` and the token count `D`. Choosing a
tokens-per-parameter ratio `r` ties them together, so there is only one unknown left: the model
size. The answer is the largest model size that still fits the budget.

## Hint 2 — formula / approach

Substitute `D = r * N` into `6 * N * D`. You get `6 * r * N**2`. The largest `N` with
`6 * r * N**2 <= C` is `N**2 <= C // (6 * r)`. Because `N**2` is an integer, floor division on the
budget is exact. Take the integer square root with `math.isqrt`, not a float square root. Return
`(N, r * N)`. If `N` is zero, no model fits, so raise `ValueError`. Check the types of both
arguments before any arithmetic, because `bool` is a subclass of `int`.

## Hint 3 — code shape

Validate: type checks first, then value checks. Compute `max_square = flop_budget // (6 *
tokens_per_param)`. Compute `n_params = math.isqrt(max_square)`. If `n_params` is zero, raise
`ValueError`. Return `(n_params, tokens_per_param * n_params)`. There is no float anywhere in the
computation.
