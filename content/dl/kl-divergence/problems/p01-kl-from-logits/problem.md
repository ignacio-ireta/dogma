# KL divergence between batches of categorical distributions

Implement `kl_divergence_from_logits(p_logits, q_logits)`. Each row of the two inputs holds the
unnormalised log-scores of one categorical distribution over `k` classes. Return, for every row,
the Kullback-Leibler divergence $\mathrm{KL}(P_i \,\|\, Q_i)$ in nats, where $P_i=\mathrm{softmax}(p_i)$
and $Q_i=\mathrm{softmax}(q_i)$. Compute from the logits with log-sum-exp, not by first forming the
probabilities and then taking logs. A probability that underflows to zero must not produce `nan`.

## Contract

- `p_logits`, `q_logits`: array-likes of shape `(n, k)` with `n >= 1` and `k >= 1`, the same shape.
  Entries are real numbers. An entry may be `-inf`, which means that class has probability zero.
  Booleans, complex numbers and objects are invalid. `NaN` and `+inf` are invalid.
- Every row of each input must contain at least one finite entry, so that the distribution is defined.
- Returns: float64 array of shape `(n,)`. Element `i` is
  $$\mathrm{KL}(P_i\|Q_i)=\sum_{j} p_{ij}\,(\log p_{ij}-\log q_{ij}).$$
- Conventions: a class with $p_{ij}=0$ contributes $0$, even when $q_{ij}=0$ (that is $0\log 0=0$).
  A class with $p_{ij}>0$ and $q_{ij}=0$ makes that row's KL `+inf`. The output can therefore
  contain `inf`. It is never `-inf` and never `nan`.
- Raises `ValueError` when: an input is not 2-D, is empty, or has a dtype that is not real numeric;
  the shapes differ; an input contains `NaN` or `+inf`; or a row of either input has no finite entry.
- Must not mutate its inputs.

## Examples

```python
>>> kl_divergence_from_logits([[0.0, 0.0]], [[0.0, 0.0]])
array([0.])                           # identical distributions: KL = 0
>>> kl_divergence_from_logits([[0.0, 0.0]], [[0.0, np.log(3.0)]])
array([0.14384104])                   # P = (0.5, 0.5) and Q = (0.25, 0.75)
>>> kl_divergence_from_logits([[0.0, -np.inf]], [[-np.inf, 0.0]])
array([inf])                          # P = (1, 0) and Q = (0, 1): p_0 > 0 where q_0 = 0
>>> kl_divergence_from_logits([[0.0, -np.inf]], [[0.0, -np.inf]])
array([0.])                           # the masked class has p = 0, so it adds nothing
```

## Constraints

- Allowed: NumPy, including `np.exp`, `np.log`, `np.max`, `np.sum`, `np.where`, `np.isfinite`
  and `np.errstate`, plus the standard library.
- Not allowed: `scipy.special` and `scipy.stats`, including `scipy.special.softmax`,
  `scipy.special.rel_entr` and `scipy.stats.entropy`. They are the test oracles, not the tool.
  Write log-softmax and the sum yourself.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They check: agreement with a
`scipy.special.rel_entr` reference on random batches with several shapes, including masked classes;
the zero-probability rules in both directions; forward versus reverse KL (the two must not be
swapped); nats rather than bits; stability for logits near one thousand and for a row whose
probabilities underflow; the relation KL = cross-entropy minus entropy; a non-negative result for
every row; the shape `(n,)` and float64 dtype; each row computed independently of the others;
every `ValueError` rule; and inputs left unchanged.
