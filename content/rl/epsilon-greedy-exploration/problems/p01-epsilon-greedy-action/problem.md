# Epsilon-greedy action with a fixed draw order

Implement `epsilon_greedy_action(q_values, epsilon, rng)`. Choose one action index from the
current action-value estimates. Use the random generator that the caller passes in, and follow the
draw order below exactly, so that the same seed always gives the same sequence of actions.

## Contract

- Draw order (exact). Every call does the following, in this order:

  1. Call `rng.random()` once. Call the result `u`. This draw happens on every call, whatever
     `epsilon` is and whether there are ties.
  2. If `u < epsilon` (strictly), explore. Call `rng.integers(n_actions)` once, and return the result as an
     `int`. This is the only other draw, and it happens only when exploring. Exploration is uniform
     over all actions, including the greedy one.
  3. Otherwise, exploit. Return the lowest index among the actions with the largest `q_values`.
     Ties are broken by lowest index, and no further draw is made.
- Do not use the global `numpy.random` state or the standard `random` module.
- `q_values`: one-dimensional real NumPy array of length `A >= 1`, every entry finite. Integer
  arrays are valid. Boolean, complex and object arrays are invalid.
- `epsilon`: a real number in `[0, 1]`. `bool` and NaN are invalid.
- `rng`: a `numpy.random.Generator`. Any other type raises `TypeError`.
- Returns: a Python `int` in `0 .. A-1`.
- Raises `ValueError` when `q_values` is empty, not one-dimensional, or contains non-finite values,
  or when `epsilon` is invalid. Must not mutate `q_values`.

## Examples

```python
rng = np.random.default_rng(0)
epsilon_greedy_action(np.array([0.1, 0.9, 0.3]), 0.0, rng)
# -> 1   (no exploration; the greedy action)
```

```python
rng = np.random.default_rng(0)
epsilon_greedy_action(np.array([2.0, 2.0, 1.0]), 0.0, rng)
# -> 0   (tie between indices 0 and 1; the lowest index wins)
```

With `epsilon = 0.2` and `q_values = [0.2, 0.9, 0.9, 0.4]`, the four selection probabilities are
`0.05`, `0.85`, `0.05` and `0.05`. Index 1 is the lowest greedy index, so it gets the exploit share.
Index 2 is tied but not lowest, so it is chosen only by exploration. The hidden tests check these
frequencies over 20000 draws with a fixed seed: each frequency must be within 0.02 (absolute) of
its probability.

## Constraints

Allowed: Python standard library and NumPy. The generator draws in the order above and nothing
else. Not allowed: `numpy.random` module-level functions, `random` and `scipy.stats`, because they
break the draw order or the reproducibility.

## Done looks like

The examples pass. Hidden tests replay the draw order with an independent implementation and
require identical action sequences for several epsilon values and tie patterns. They also check
that a call with epsilon zero consumes exactly one uniform draw, that the selection frequencies
match the analytic table within 0.02, that ties are broken by lowest index, and
the invalid-input cases.

The hidden tests draw their bulk inputs from the real datasets `digits` and `wine` (see
`docs/DATASETS.md`).
