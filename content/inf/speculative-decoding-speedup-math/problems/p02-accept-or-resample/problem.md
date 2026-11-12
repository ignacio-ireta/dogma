# Accept or resample one speculative token

Implement `accept_or_resample(p, q, draft_token, rng)`, the speculative sampling step for one
position. A draft model with distribution `q` over a vocabulary of size `V` proposed
`draft_token = x`, which was drawn from `q`. `p` is the target model's distribution at the same
position. The function returns the token this step emits. Use this rule:

1. Draw `u1 = rng.random()`. Accept `x` when `u1 < min(1, p[x] / q[x])`. The comparison is strict.
2. If `x` is accepted, return `x`. Otherwise draw `u2 = rng.random()` and return the first index `i`
   whose cumulative residual mass is greater than `u2`. The residual is `max(0, p - q)`, normalized to
   sum to 1. Its cumulative sums are normalized so the last one is exactly 1.
3. If the residual total is zero (`p` is elementwise at most `q`), use `p` in place of the residual.

Draw order is part of the contract. Every call draws exactly one uniform on acceptance and exactly two
on rejection, all with `rng.random()`. Make no other random calls.

## Contract

- `p`: real numeric array-like, 1-D, length `V >= 1`. Entries are finite and nonnegative, and they sum
  to 1 within `1e-9`. Other ndim, empty input, negative or NaN entries, and a wrong sum are invalid.
- `q`: same rules as `p`, with the same length `V`.
- `draft_token`: Python `int` or NumPy integer in `[0, V)`. A bool or a float is invalid. Raise
  `ValueError` when `q[draft_token] == 0`, because the draft could not have proposed that token.
- `rng`: an object with a `random()` method that returns a float in `[0, 1)`. Only `random()` may be
  called.
- Returns: a Python `int` in `[0, V)`.
- Raises: `ValueError` for any invalid argument above.
- Must not mutate `p` or `q`.

## Examples

With `rng` a stand-in whose `random()` returns the given values in order:

```python
accept_or_resample([0.6, 0.4, 0.0], [0.2, 0.3, 0.5], 1, rng=[0.99])
# ratio p[1]/q[1] = 4/3 >= 1, so u1 = 0.99 < 1 accepts. Returns 1.
```

```python
accept_or_resample([0.6, 0.4, 0.0], [0.2, 0.3, 0.5], 2, rng=[0.10, 0.70])
# ratio p[2]/q[2] = 0, so u1 = 0.10 is rejected. Residual = [0.4, 0.1, 0] normalized to
# [0.8, 0.2, 0], cumulative [0.8, 1.0, 1.0]. u2 = 0.70 < 0.8, so it returns 0 even though the draft was 2.
```

```python
accept_or_resample([0.25, 0.75], [0.5, 0.5], 0, rng=[0.5, 0.3])
# The comparison is strict: u1 = 0.5 is not below ratio 0.5, so it is rejected.
# Residual = [0, 0.25] normalized to [0, 1], so the resample returns 1.
```

## Constraints

- Allowed: NumPy and the Python standard library. Use `rng.random()` for every draw.
- Not allowed: `numpy.random.choice` and `numpy.random.multinomial`, including `rng.choice`.
  Those draw with a different order and cannot match the contract.

## Done looks like

Passing means the examples match and the draw counts follow the contract. Hidden tests cover:

- draw counts (one on acceptance, two on rejection) over a grid of uniforms that sweeps the acceptance
  threshold;
- the resampled token following the normalized residual, not `p` and not `q`;
- the strict acceptance boundary at an exact ratio;
- a zero residual falling back to `p`;
- invalid probability vectors, draft tokens and draft probabilities of zero;
- a single-token vocabulary and tiny probabilities that stay finite;
- an empirical output distribution over seeded draws that equals `p`;
- inputs left unchanged, and the Python `int` return type.

The hidden tests draw their bulk inputs from the real datasets `gutenberg_pride_prejudice` and
`gutenberg_walden` (see `docs/DATASETS.md`).
