# Expected tokens and speedup per speculative round

Implement `speculative_speedup(alpha, gamma, c)`. It returns two numbers for one
speculative decoding setup: the expected number of tokens emitted per round, and the
walltime speedup over plain decoding. A round has a draft model propose `gamma` tokens
at a relative cost of `c` target passes each, then one target verification pass. Each
drafted token is accepted independently with probability `alpha`.

## Contract

- `alpha`: real float, finite, `0 <= alpha <= 1`. A bool is invalid.
- `gamma`: Python `int`, `gamma >= 0`. A bool, a float such as `4.0`, or a negative
  value is invalid.
- `c`: finite real float, `c >= 0`. A bool is invalid.
- Returns: a tuple `(tokens_per_round, speedup)` of two Python floats.
  - `tokens_per_round` is `E[N]`, the expected tokens emitted per round. It counts the
    target's own token (the correction or the bonus), so it is `1 + (expected accepted
    draft tokens)`. It equals `(1 - alpha**(gamma+1)) / (1 - alpha)` for `alpha < 1`,
    and `gamma + 1` for `alpha == 1`.
  - `speedup` is `tokens_per_round / (gamma * c + 1)`.
- Raises: `ValueError` for any invalid argument above, including NaN or infinite
  values.
- Every valid input must return finite floats. No `ZeroDivisionError` at `alpha = 1`
  or `gamma = 0`.
- Must not mutate inputs.

## Examples

```python
>>> speculative_speedup(0.8, 4, 0.1)   # E[N] = 3.3616, round cost = 1.4
(3.3616, 2.4011428571...)
```

```python
>>> speculative_speedup(0.5, 2, 0.0)   # E[N] = 1 + 0.5 + 0.25, free draft
(1.75, 1.75)
```

```python
>>> speculative_speedup(0.0, 5, 0.2)   # every draft rejected: one token per round
(1.0, 0.5)
```

```python
>>> speculative_speedup(1.0, 3, 0.25)  # every draft accepted: gamma + 1 tokens
(4.0, 2.2857142857...)
```

## Constraints

- Allowed: the Python standard library `math` module, and plain arithmetic.
- Not allowed: `fractions`, `decimal`, and `numpy`. Write the formula yourself.

## Done looks like

Passing means the four examples above match, and the closed form holds for every
valid `alpha` and `gamma`. Hidden tests cover:

- seeded random inputs compared with an exact rational computation;
- the `alpha = 1`, `alpha = 0` and `gamma = 0` edge cases;
- a zero-cost draft, where speedup equals tokens per round;
- an acceptance rate near 1 with a long draft, where a naive subtraction loses digits;
- a draft slower than the target at low acceptance;
- invalid `alpha`, `gamma` and `c`, bool and float types, and the return types.

Some hidden tests take their inputs from the real datasets `gutenberg_pride_prejudice` and
`gutenberg_walden` (see `docs/DATASETS.md`).
