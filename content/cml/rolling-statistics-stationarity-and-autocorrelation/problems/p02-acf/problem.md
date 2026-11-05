# Sample autocorrelation (biased estimator)

Implement `acf(x, nlags)`. It returns the sample autocorrelation of `x` at lags 0 through `nlags`. Subtract
the mean first. Then every lag is divided by the same total sum of squares, the biased estimator that
time-series texts start from. Dividing each lag by the number of pairs it actually sums is a different
estimator, and it is the most common wrong answer.

## Contract

- `x`: 1-D sequence of finite real numbers (list, tuple or NumPy array), length n >= 2. Integers are allowed.
  Must not be mutated.
- `nlags`: integer with 0 <= nlags <= n - 1. A bool or a float is rejected.
- Returns: a float64 array of length `nlags + 1`. Entry k is the autocorrelation at lag k, and entry 0 is
  exactly 1.
- Raises `ValueError` when `x` is not 1-D, has fewer than two values, or contains NaN or infinity; when
  `nlags` is not an integer in 0 .. n - 1; or when `x` is constant (every value equal), because the
  autocorrelation is then undefined.

With `d_t = x_t - mean(x)`, the autocorrelation at lag k is

    r_k = (sum over t = 0 .. n-k-1 of d_t * d_(t+k)) / (sum over t = 0 .. n-1 of d_t^2)

## Examples

```python
>>> acf([1, 3, 2, 5, 4, 6], 3)
array([ 1.      ,  0.1     ,  0.342857, -0.442857])
```

```python
>>> acf([1, 2, 3, 4], 1)
array([1.  , 0.25])
```

In the first example the mean is 3.5 and the denominator is 17.5. At lag 1 the numerator sums five products,
and it is divided by 17.5, not by 5.

## Constraints

- Allowed: NumPy and the Python standard library.
- Not allowed: `statsmodels` and `pandas`. Their autocorrelation functions are the reference the tests compare
  against, so the formula has to be yours.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare the output with
`statsmodels`' biased autocorrelation (`adjusted=False`) on random series of several lengths and lag counts,
including series with a large non-zero mean. They check a closed form: an alternating series has a known
autocorrelation at every lag, and the adjusted (n - k) denominator fails that check. They check that lag 0
is exactly one, that shifting or scaling the series leaves the result unchanged, and that the boundary values
of `nlags` work. They also cover invalid `nlags`, a series too short to have lags, a constant series including
one whose mean is not exactly representable, non-finite input, integer input, and inputs that must not be
mutated.
