# Welford mean and variance

Implement `welford(xs)`. It returns the count, the mean and the sample variance of the values in
`xs`, computed in a single pass with constant memory, using Welford's update. `xs` can be any
iterable of real numbers, including a generator that can be consumed only once. Keep the running
count, the mean and M2 (the sum of squared deviations from the mean), and update them one value at a
time.

## Contract

- `xs`: an iterable of real numbers: a list, a tuple, a NumPy array or a generator. Integers allowed.
  Read it once, from start to end, and never store the values.
- Returns: `(n, mean, variance)`. `n` is an int, the number of values. `mean` is a float. `variance` is
  the sample variance, M2 / (n − 1), a float, and it is `nan` when `n == 1`.
- Raises `ValueError` when `xs` is empty.
- Must use a single pass with O(1) memory, so the input is never collected into a list or array.

## Examples

```python
>>> welford([1.0, 3.0, 4.0, 8.0, 9.0])
(5, 5.0, 11.5)              # mean 5; sum of squared deviations 46; 46 / 4 = 11.5
>>> welford([4.0])
(1, 4.0, nan)               # one value has no sample variance
>>> welford(x for x in [2, 4, 4, 4, 5, 5, 7, 9])
(8, 5.0, 4.571428571428571) # a generator works as well: M2 = 32, and 32 / 7 is the variance
>>> welford([])
ValueError                  # there is no mean of zero values
```

## Constraints

- Allowed: the Python standard library (`math`, `collections.abc`, …).
- Not allowed: `numpy`, `statistics` and `scipy`. Compute the sums yourself, one value at a time.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with
`np.mean` and `np.var(ddof=1)` on random data of several sizes, and on random finite lists; a large
offset (values near 10⁹ with a spread of about 1), where the naive sum-of-squares formula fails;
generator input read once; a streamed input of 300,000 values that must not be stored; constant data
with exactly zero variance; the single-value case; integer input giving float results; scaling
behaviour of the variance; and the empty-input error.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer`,
`california_housing`, `nile` and `sunspots` (see `docs/DATASETS.md`).
