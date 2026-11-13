# Symmetric InfoNCE loss for paired embeddings

Implement `symmetric_infonce_loss(a, b, *, temperature)`. `a` and `b` hold `N` paired embeddings of dimension
`D`: row `i` of `a` and row `i` of `b` form a positive pair, and every other row of `b` is a negative for row `i`
of `a` (and the reverse). Normalise each row to unit length, compute the `N x N` similarity matrix divided by the
temperature, and return the average of the cross-entropy over rows (`a` to `b`) and over columns (`b` to `a`),
with the diagonal as the correct label.

## Contract

- `a`, `b`: floating-point tensors of the same shape `(N, D)`, with `N >= 1` and `D >= 1`, and the same dtype.
  Must not be modified.
- `temperature`: a real number (`int` or `float`, not `bool`), finite and `> 0`. Keyword-only and required.
- Normalisation: `a_hat[i] = a[i] / ||a[i]||_2`, and likewise for `b`. A row with zero norm has no direction, so
  raise `ValueError`.
- Similarity: `S[i, j] = a_hat[i] . b_hat[j] / temperature`, shape `(N, N)`.
- Loss: `L_ab = mean over i of ( logsumexp_j S[i, j] - S[i, i] )`, `L_ba = mean over j of ( logsumexp_i S[i, j] - S[j, j] )`,
  and the result is `(L_ab + L_ba) / 2`. Compute the log-sum-exp stably: subtract the row (or column) maximum before
  exponentiating.
- Returns: a 0-dimensional tensor with the dtype of `a`. With `N = 1` the loss is exactly `0`.
- Raises `ValueError` when: `a` or `b` is not a 2-D floating-point tensor; the shapes differ; `N = 0` or `D = 0`;
  an embedding contains a non-finite value; a row has zero norm; `temperature` is a bool, not a real number,
  not finite, or not positive.
- Must not mutate `a` or `b`.
- Differentiable: the result stays on the autograd graph. When an input has
  `requires_grad=True`, backpropagating through the result gives the gradient of the formula
  above. Do not detach, and do not leave PyTorch (`.item()`, `.numpy()`, `torch.no_grad()`) on
  the way.

## Examples

```python
a = [[1.0, 0.0], [0.0, 1.0]]
b = [[1.0, 0.0], [0.0, 1.0]]
symmetric_infonce_loss(a, b, temperature=1.0)
# S = I, each row gives log(1 + e^-1) = 0.3133; both directions agree, so the loss is 0.3133
```

```python
symmetric_infonce_loss(a, b, temperature=0.1)
# S = 10 I, each row gives log(1 + e^-10) = 0.0000454
```

```python
a = [[2.0, 0.0], [0.0, 5.0]]  # same directions as the previous example, longer vectors
symmetric_infonce_loss(a, b, temperature=1.0)
# 0.3133: the normalisation removes the length, so the loss is the same as with unit vectors
```

```python
symmetric_infonce_loss([[3.0, 4.0]], [[1.0, 0.0]], temperature=0.5)
# N = 1 has one candidate, so the loss is exactly 0.0
```

## Constraints

- Allowed: torch tensor arithmetic and reductions, such as `matmul`, `norm` or `sqrt`, `max`, `exp`, `log`, `sum`, `mean`,
  `gather` and `arange`.
- Not allowed: `torch.nn` (this covers `torch.nn.functional.normalize`, `cross_entropy` and `log_softmax`),
  `torch.cosine_similarity`, `torch.logsumexp`, `torch.log_softmax`, `torch.special` and `torch.distributions`.
  Write the normalisation, the similarity matrix and the log-sum-exp from their definitions.

## Done looks like

The four examples pass. Hidden tests compare the loss with `torch.nn.functional.cross_entropy` on the similarity
matrix (rows, then its transpose), on seeded random inputs at several temperatures and batch sizes, with a float64
tolerance. They check the chance level (identical embeddings give `log N`), the symmetry (swapping `a` and `b`
changes nothing), the invariance to row scaling, a small temperature where a naive exponential overflows, `N = 1`,
float32 dtype within a declared bound, invalid shapes, zero rows and temperatures, non-finite embeddings, and
that inputs are not mutated. A property test checks that the loss is non-negative and unchanged when the pairs are
permuted together.

The hidden tests also backpropagate through the result and compare the gradients with PyTorch's
own.

The hidden tests draw their bulk inputs from the real dataset `digits` (see `docs/DATASETS.md`).
