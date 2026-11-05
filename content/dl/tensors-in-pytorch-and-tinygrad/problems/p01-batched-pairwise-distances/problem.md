# Batched pairwise squared distances

Implement `batched_sq_distances(X, Y)`. Each batch element holds two point clouds: `N` query
points and `M` reference points, each in `D` dimensions. Return the squared Euclidean distance
between every query and every reference point, for every batch element, with no Python loop over
batches or pairs. Use broadcasting.

## Contract

- `X`: floating tensor of shape `(B, N, D)`. `Y`: floating tensor of shape `(B, M, D)`.
  Float32 and float64 are both valid, and the two arguments may differ. `B`, `N`, `M` and `D` are
  each zero or more. Complex, integer and boolean tensors are invalid.
- Returns: a tensor of shape `(B, N, M)` where
  `out[b, i, j] = sum over d of (X[b, i, d] - Y[b, j, d])**2`.
  Its dtype is `torch.promote_types(X.dtype, Y.dtype)`, so float32 with float32 stays float32 and
  float32 with float64 becomes float64.
- Every returned entry is `>= 0`, including entries for identical points.
- Accuracy: each entry matches the exact value computed in float64 from the input values, with
  `atol=1e-6` and `rtol=1e-5`. This holds when the points share a large common offset.
- Raises: `ValueError` when an argument is not a floating tensor, is not 3-dimensional, or has a
  batch size or feature size that differs from the other argument.
- Must not mutate inputs.

## Examples

```python
X = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])  # B=1, N=2, D=2
Y = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])  # B=1, M=2, D=2
batched_sq_distances(X, Y)
# -> tensor([[[ 0.,  1.],
#             [25., 20.]]])
```

```python
X = torch.tensor([[[1.0], [2.0]], [[0.0], [0.0]]])  # B=2, N=2, D=1
Y = torch.tensor([[[0.0]], [[5.0]]])  # B=2, M=1, D=1
batched_sq_distances(X, Y)
# -> tensor([[[ 1.],
#             [ 4.]],
#            [[25.],
#             [25.]]])
```

## Constraints

- Allowed: PyTorch tensor operations, including broadcasting, arithmetic and reductions.
- Not allowed: `torch.cdist` and `torch.nn` (including `torch.nn.functional.pairwise_distance`).
  No Python loop over batches or pairs. Use the broadcast shape `(B, N, M, D)` or the expansion
  `||x||^2 - 2 x.y + ||y||^2`, and handle its rounding.

## Done looks like

Your function matches the examples and handles empty batch, query, reference and feature
dimensions. The hidden tests check dtype promotion and the dtype of the result, agreement with an
independent float64 computation on seeded random inputs, non-negative output for identical points,
accuracy when all points share a large offset, scale behaviour for very small coordinates, and
input validation and non-mutation.
