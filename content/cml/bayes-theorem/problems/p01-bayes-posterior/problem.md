# Posterior from observations

Implement `posterior(prior, likelihood, observations)`. A discrete hypothesis $H$ has $h$ values
with prior $P(H=i)$. Each observation is a category index $j \in \{0,\dots,o-1\}$, and given the
hypothesis the observations are independent, with $P(\text{obs}=j \mid H=i)$ stored in
`likelihood[i, j]`. Return the posterior $P(H=i \mid \text{observations})$. Multiplying many
likelihoods underflows float64, so compute the update in log space.

## Contract

- `prior`: 1-D, length $h \ge 1$. Entries are finite and non-negative and sum to 1 within $10^{-9}$.
- `likelihood`: 2-D, shape `(h, o)` with $o \ge 1$. Entries are finite and non-negative. Each row
  sums to 1 within $10^{-9}$. A row is a distribution over observations, not over hypotheses.
- `observations`: 1-D sequence of integer indices in `[0, o)`. Repeats are allowed. An empty
  sequence is valid and returns the prior.
- Returns: NumPy `float64` array of shape `(h,)` that sums to 1.
- A hypothesis with prior 0 has posterior exactly 0, whatever the data.
- A hypothesis that gives some observation likelihood 0 has posterior exactly 0.
- Raises `ValueError` when: the shapes disagree; a probability is negative, NaN or infinite; the
  prior or a likelihood row does not sum to 1; an observation is not an integer; an observation index
  is negative or at least `o`; or the evidence is zero, because every hypothesis gives the
  observations probability 0.
- Must not mutate its inputs.

## Examples

```python
>>> posterior([0.01, 0.99], [[0.99, 0.01], [0.05, 0.95]], [0])     # one positive test
array([0.16666667, 0.83333333])
>>> posterior([0.5, 0.5], [[0.9, 0.1], [0.2, 0.8]], [0, 0])        # two observations of 0
array([0.95294118, 0.04705882])
>>> posterior([0.5, 0.5], [[0.9, 0.1], [0.2, 0.8]], [])            # no data: the prior
array([0.5, 0.5])
>>> posterior([1.0, 0.0], [[1.0, 0.0], [0.5, 0.5]], [1])           # the only possible
ValueError: impossible observations                                # hypothesis says "no"
```

## Constraints

- Allowed: NumPy (`np.array`, `np.log`, `np.exp`, `np.errstate`, `np.isfinite`, …) and the
  standard library.
- Not allowed: `scipy`. `scipy.special.logsumexp` would perform the log-sum-exp step for you, and
  that step is what this problem asks you to write.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with an exact
rational Bayes update (`fractions.Fraction`) on short sequences, and a 2000-observation sequence
where a plain product underflows to zero, and cases with a very wide spread of finite log-weights,
where centring on the mean or the median overflows. They also check sequential updating equals one
batch update, invariance to the order of observations, and exact zeros for zero-prior and
zero-likelihood hypotheses. Error cases cover impossible evidence, a negative or out-of-range index,
a non-integer index, and shape, sum and sign violations. The families also include the empty
sequence, NaN probabilities, normalised output, and inputs left unchanged.
