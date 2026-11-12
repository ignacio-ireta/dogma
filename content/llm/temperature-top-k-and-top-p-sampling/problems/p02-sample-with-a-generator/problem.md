# Sample a token with a passed-in generator

Implement `sample_token(probs, generator)`. It draws one token index from a probability vector,
using only the `torch.Generator` the caller passes in. This is the last step of decoding. The
probability vector is the output of the previous problem, after temperature, top-k and top-p.

## Contract

- `probs`: `torch.float64` tensor of shape `(V,)`, `V >= 1`. Every entry is finite and `>= 0`. The
  entries sum to `1` within `1e-9`.
- `generator`: a `torch.Generator` instance. `None` and other types are rejected.
- Returns: a Python `int` in `[0, V)`. Over many draws, index `i` appears with frequency
  `probs[i]`.
- Method: draw one uniform value `u` in `[0, 1)` from `generator`. Let `cdf = cumsum(probs)`.
  Return the first index `i` with `cdf[i] > u`, with the **strict** inequality. In PyTorch that is
  `torch.searchsorted(cdf, u, right=True)`. Do not use `right=False` (`cdf[i] >= u`): when `u` is
  exactly `0` and `probs[0] = 0`, it returns index `0`, which has probability `0`.
  Rounding can leave `cdf[-1]` a hair below `u`, and the search then returns `V`. In that case return
  the last index with `probs > 0`, never an index whose probability is `0`. The sum check allows a
  deficit of up to `1e-9`, so this case has probability at most about `1e-9` per draw. A seeded test
  cannot hit it reliably, so the rule is pinned here rather than tested by draws.
- Consequence: a token with probability `0` is never returned. Each call uses exactly one draw from
  the generator, so a generator seeded the same way gives the same sequence of tokens.
- Raises `ValueError` when: `probs` is not a non-empty 1-D `float64` tensor; it has a non-finite or
  negative entry; its sum differs from `1` by more than `1e-9`; or `generator` is not a
  `torch.Generator`.
- Must not mutate `probs`. The function must not change the global random state.

## Examples

```python
>>> g = torch.Generator().manual_seed(0)
>>> sample_token(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), g)
2
```
A one-hot distribution always returns its only non-zero index.

```python
>>> g = torch.Generator().manual_seed(0)
>>> probs = torch.tensor([0.5, 0.0, 0.5], dtype=torch.float64)
>>> [sample_token(probs, g) for _ in range(6)]     # index 1 never appears
[2, 2, 0, 2, 2, 2]
```
The exact list depends on how the generator's uniform values are consumed. The property that it
never contains `1` is what the contract guarantees.

```python
>>> sample_token(torch.tensor([1.0], dtype=torch.float64), torch.Generator().manual_seed(1))
0
```

## Constraints

- Allowed: `torch` tensor operations such as `cumsum`, `searchsorted`, `nonzero` and `rand` with
  an explicit `generator=` argument; plain Python `int`, `float` and `bool` checks.
- Not allowed: `torch.multinomial`, `torch.bucketize`, `torch.distributions` and `torch.nn` in any form.
  `torch.bucketize` does the whole inverse-CDF lookup in one call, so use `searchsorted` instead. Also no
  `torch.rand` or `torch.randn` without the `generator=` argument, because they change the global
  state. Write the inverse-CDF draw yourself.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They cover: empirical frequencies
over thousands of seeded draws within a five-standard-error band per category; tokens with zero
probability never returned, including at the start and at the end of the vector; a one-hot vector
returning its index; a one-entry vocabulary; the same seed reproducing the same sequence and a
different seed giving a different one; the global random state left unchanged; the result
independent of the global seed; the input left unchanged; and the `ValueError` rules for shape,
dtype, sign, sum, non-finite values and the generator type.

The hidden tests draw their bulk inputs from the real dataset `gutenberg_frankenstein` (see
`docs/DATASETS.md`).
