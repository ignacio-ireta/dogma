# Rotary position embedding on a sequence

Implement `apply_rope(x)`. It applies rotary position embedding (RoPE) to a tensor of queries or
keys, shaped as a sequence. Along the sequence axis, row `t` is rotated by an angle that grows with
`t`. Within each row, the features form adjacent pairs `(2i, 2i+1)`. Each pair is rotated in its own
plane, by angle `t * theta_i`, where `theta_i = 10000 ** (-2i / d)`.

Read a pair `(x_{2i}, x_{2i+1})` as one complex number `x_{2i} + i x_{2i+1}`. Then RoPE multiplies
it by `exp(i * t * theta_i)`. The output has the same shape as the input.

## Contract

- `x`: `torch.Tensor` with shape `(..., L, d)`, where `...` is any number of leading batch axes
  (zero or more), `L >= 0` is the sequence length and `d` is the feature width. `d` must be
  **even and at least 2**. Values are real. Float64 is the tested dtype.
- Position `t` is the index along the second-to-last axis, counted from 0.
- Output: `(t, pair i)` rotation, written out:
  - `out[..., t, 2i]   = x[..., t, 2i] * cos(t * theta_i) - x[..., t, 2i+1] * sin(t * theta_i)`
  - `out[..., t, 2i+1] = x[..., t, 2i] * sin(t * theta_i) + x[..., t, 2i+1] * cos(t * theta_i)`
- Returns: a tensor with the same shape and dtype as `x`.
- Consequences you can check: position 0 is the identity; every row keeps its Euclidean norm;
  for two rotated vectors `q` at position `m` and `k` at position `n`, the dot product `q·k` depends
  only on `m - n`.
- Raises `ValueError` when: `x` has fewer than 2 dimensions; `d` is odd; `d` is zero.
- Must not mutate `x`.

## Examples

```python
>>> apply_rope(torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64))
tensor([[1.0000, 0.0000],
        [0.5403, 0.8415]], dtype=torch.float64)
# Row 1 rotates (1, 0) by theta_0 = 1 radian: (cos 1, sin 1).
```

```python
>>> apply_rope(torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64))
tensor([[1., 2., 3., 4.]], dtype=torch.float64)
# A single row is position 0, so nothing rotates.
```

```python
>>> apply_rope(torch.zeros(2, 3, dtype=torch.float64))
ValueError: feature width must be an even number of at least 2
```

## Constraints

- Allowed: the `torch` package (for example `torch.arange`, `torch.cos`, `torch.sin`, `torch.stack`,
  `torch.empty_like`, `torch.exp`), and the `math` module.
- Not allowed: `torch.nn` in any form, including `torch.nn.functional`. Do not call a built-in RoPE
  or rotary helper from any library: the rotation itself is the exercise. Do not import `numpy`.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They cover: agreement with a complex
multiplication reference on seeded batches; the adjacent-pair convention (not the split-half
convention used by some libraries); the rotation angle at position 1 and the sign of the sine;
position 0 as the identity; per-row norm preservation; the relative-offset property of query-key
dot products, checked on repeated vectors at shifted positions; leading batch axes handled
independently; an empty sequence; large sequence positions and small input magnitudes; the dtype
and shape contract; the `ValueError` rules (too few dimensions, odd width, zero width); and the input
left unchanged.
