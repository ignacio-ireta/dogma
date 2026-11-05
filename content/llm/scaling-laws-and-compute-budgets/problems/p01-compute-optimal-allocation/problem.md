# Compute-optimal model and token counts

Training a language model costs about `6 * N * D` floating-point operations, where `N` is the
number of parameters and `D` the number of training tokens. Given a fixed compute budget `C`, a
larger model sees fewer tokens and a smaller model sees more. This function fixes the ratio
`D = r * N` (the number of tokens per parameter, `r`) and returns the largest model that fits the
budget at that ratio. Everything is integer arithmetic, because the budgets of real runs are far
above the range where floats stay exact.

## Contract

- `compute_optimal_allocation(flop_budget, tokens_per_param)` with:
  - `flop_budget`: a Python `int` `C >= 1`, the total training FLOPs. A `float` (even `1e20`) raises
    `TypeError`, because a float cannot hold every digit of a budget above `2**53`. `bool` raises
    `TypeError`. An `int` below 1 raises `ValueError`.
  - `tokens_per_param`: a Python `int` `r >= 1`. A `float`, a `bool` or any other type raises
    `TypeError`. An `int` below 1 raises `ValueError`.
- Cost model: `C = 6 * N * D` and `D = r * N`, so `C = 6 * r * N**2`.
- Rounding rule: `N` is the largest integer with `6 * N * (r * N) <= C`. Then `D = r * N`. The
  rounding never exceeds the budget, so `6 * N * D <= C` always holds. Real-valued answers are
  rounded **down**.
- Returns a tuple `(N, D)` of two Python `int`s.
- Raises `ValueError` when no model fits, that is when `C < 6 * r` and `N` would be `0`.
- Pure integer arithmetic. The float square-root ban is deliberate: a float root of a budget above
  `2**53` can land on the wrong side of an exact boundary `C = 6 r N**2`, and the hidden tests
  check those boundaries. A float estimate that you correct afterwards is not what this problem
  asks for either, since the aim is exact arithmetic (the tests cannot tell the two apart). It
  blocks no correct approach, because `math.isqrt` is exact on Python ints and is allowed.
  Worked floor: for `C = 1000`, `r = 20`, `C // (6 * r) = 8`, and `math.isqrt(8) = 2`, so
  `N = 2` and `D = 40`.

## Examples

```python
>>> compute_optimal_allocation(120 * 10**18, 20)     # C = 6 * N * (20 N) gives N = 1e9
(1000000000, 20000000000)
```

```python
>>> compute_optimal_allocation(1000, 20)            # real N = sqrt(1000 / 120) = 2.89, so 2
(2, 40)                                              # 6 * 2 * 40 = 480 <= 1000
```

```python
>>> compute_optimal_allocation(100, 20)             # 6 * 20 = 120 > 100: no model fits
ValueError
```

## Constraints

- Allowed: Python integers, `math.isqrt`, and integer division `//`.
- Not allowed: `math.sqrt` and `numpy.sqrt`, and any other float square root. A float square
  root rounds at the exact boundaries that the hidden tests check. The `** 0.5` power on floats is
  the same mistake.

## Done looks like

All public tests pass. Submit then runs the hidden tests. They cover agreement with an exhaustive
integer search on small budgets, the exact boundary `C = 6 r N**2` above `2**53` (and one below
it), agreement with a binary-search oracle on large random budgets, the rule that the budget is
never exceeded and the next integer would exceed it, the type rules (float, bool), the range rules
for `r` and the budget, the no-fit error, the types of the returned values, and that `D` equals
`r * N` exactly.
