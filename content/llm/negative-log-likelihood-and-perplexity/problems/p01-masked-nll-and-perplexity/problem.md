# Masked mean NLL and perplexity from logits

Implement `masked_nll_and_perplexity(logits, targets, *, ignore_index=-100)`. It returns the mean
negative log-likelihood of the targets over the tokens that are not ignored, and the perplexity
`exp(mean NLL)`. Write the log-softmax yourself. It must stay finite for large logits.

## Contract

- `logits`: floating-point tensor of shape `(B, T, V)` with `B >= 1`, `T >= 1`, `V >= 1`. Must not be modified.
- `targets`: `torch.int64` tensor of shape `(B, T)`. Each entry is either `ignore_index` or an id in `[0, V-1]`.
- `ignore_index`: Python `int`. Default `-100`. A `bool` is invalid. It may be any integer, including a valid class id.
- For each non-ignored position `(n, t)`, with target `y` and logits `z = logits[n, t, :]`:
  - `nll[n, t] = logsumexp(z) - z[y]`, where `logsumexp(z) = log(sum_v exp(z_v))`.
  - Compute it stably: subtract the maximum of `z` before exponentiating.
- `mean_nll = sum of nll over non-ignored positions / (number of non-ignored positions)`. The denominator is
  the count of non-ignored tokens, not `B * T` and not `B`.
- `perplexity = exp(mean_nll)`.
- Returns: `(mean_nll, perplexity)`, two 0-dimensional tensors with the dtype of `logits`.
- Raises `ValueError` when: `logits` is not 3-D or not floating-point; `targets` is not 2-D, not `torch.int64`,
  or its shape is not `(B, T)`; a non-ignored target is outside `[0, V-1]`; `ignore_index` is not a Python
  `int`; or every target is ignored (the mean is undefined).
- Must not mutate `logits` or `targets`.

## Examples

```python
logits = [
    [
        [0.0, 0.0, 0.0, 0.0],  # uniform over V = 4
        [0.0, 0.0, 0.0, 0.0],
    ]
]  # B = 1, T = 2
targets = [[0, 3]]
# nll = log 4 at both positions, mean_nll = log 4 = 1.3863, perplexity = 4.0
```

```python
logits = [[[2.0, 0.0]], [[0.0, 0.0]]]  # B = 2, T = 1, V = 2
targets = [[0], [1]]
# row 0: nll = log(1 + e^-2) = 0.1269; row 1: nll = log 2 = 0.6931
# mean_nll = 0.4100, perplexity = 1.5069
```

```python
logits = [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]  # B=1, T=3
targets = [[1, -100, 2]]
# the ignored middle position is skipped; mean_nll = log 4 = 1.3863 (divide by 2, not 3), perplexity = 4.0
```

```python
logits = [[[1000.0, 999.0]]]  # naive exp(1000) overflows float64
targets = [[1]]
# nll = 1 + log(1 + e^-1) = 1.3133, perplexity = 1 + e = 3.7183
```

## Constraints

- Allowed: torch tensor arithmetic, `max`, `exp`, `log`, `sum`, boolean masks, indexing and `gather`.
- Not allowed: `torch.nn` (this covers `torch.nn.functional.cross_entropy` and `log_softmax`), `torch.log_softmax`,
  `torch.logsumexp`, `torch.special` and `torch.distributions`. Those compute the log-softmax or the loss as a
  whole. Write the log-sum-exp from its definition.

## Done looks like

The four examples pass. Hidden tests compare the mean NLL and the perplexity with
`torch.nn.functional.cross_entropy(..., ignore_index=...)` on seeded random batches with some ignored positions,
check that the denominator is the non-ignored count (a batch where one row is entirely ignored), that adding a
constant to each row of logits leaves the NLL unchanged, that large common offsets and large logits stay
finite and match the oracle, that a float32 input returns float32 with a declared bound, that a custom
`ignore_index` works, that a single-class vocabulary gives perplexity 1, and that invalid shapes, dtypes,
out-of-range targets and an all-ignored batch raise `ValueError`. Inputs are not mutated.
