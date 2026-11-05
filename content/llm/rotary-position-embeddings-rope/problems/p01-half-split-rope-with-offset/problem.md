# Half-split RoPE on queries and keys with a position offset

Implement `apply_rope_qk(q, k, offset)`. It applies rotary position embedding (RoPE) to the query
and key tensors of one attention layer, in the **half-split** layout used by GPT-NeoX and LLaMA.
The value tensor is never rotated, so the function does not take it.

In the half-split layout, feature `i` is paired with feature `i + d/2`, not with feature `i + 1`.
Pair `i` rotates in its own plane by an angle that grows with the token position. Positions
start at `offset`, so a decoder that already holds `offset` cached tokens can rotate only the new
ones.

## Contract

- `q`, `k`: floating-point tensors of the same shape `(B, H, T, d)`, with `B >= 0`, `H >= 0`,
  `T >= 0`, and `d` even and at least 2. `B` is the batch, `H` the heads, `T` the tokens, `d` the
  head width. Both have the same dtype.
- `offset`: `int` (not `bool`) and at least 0. Token `t` (`0 <= t < T`) sits at absolute position
  `p_t = offset + t`.
- Frequencies: with `d/2` pairs, `theta_i = 10000 ** (-2 i / d)` for `i = 0, ..., d/2 - 1`.
- Rotation, with `a = x[..., :d/2]` and `b = x[..., d/2:]`, and the angle $\phi_{t,i} = p_t\,\theta_i$:
  - `out[..., :d/2] = a * cos(phi) - b * sin(phi)`
  - `out[..., d/2:] = a * sin(phi) + b * cos(phi)`
  - `cos` and `sin` have shape `(T, d/2)` and broadcast over `B` and `H`.
- Returns: a tuple `(q_rot, k_rot)`, each of shape `(B, H, T, d)` and the input dtype. The function
  never rotates a third tensor.
- Consequences you can check: position `p_t = 0` is the identity; each `(b, h, t)` row keeps its
  Euclidean norm; the query-key dot product after rotation depends only on the difference of the
  positions.
- Raises `ValueError` when: `q` or `k` is not a 4-D floating tensor; their shapes differ; their dtypes
  differ; `d` is odd or zero; `offset` is not an `int`, is a `bool`, or is negative.
- Must not mutate `q`, `k` or `offset`.

## Examples

```python
>>> q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)      # B=1, H=1, T=1, d=2
>>> k = torch.zeros(1, 1, 1, 2, dtype=torch.float64)
>>> q_rot, _ = apply_rope_qk(q, k, offset=1)
>>> q_rot
tensor([[[[0.5403, 0.8415]]]], dtype=torch.float64)
# d = 2 has one pair (x0, x1), theta_0 = 1, position 1: (cos 1, sin 1).
```

```python
>>> q = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]], dtype=torch.float64)   # d = 4, two pairs
>>> q_rot, _ = apply_rope_qk(q, torch.zeros_like(q), offset=1)
>>> q_rot
tensor([[[[0.5403, 0.0000, 0.8415, 0.0000]]]], dtype=torch.float64)
# Pair 0 is (x0, x2) = (1, 0), angle 1 * theta_0 = 1. The output is (cos 1, 0, sin 1, 0).
```

```python
>>> apply_rope_qk(torch.zeros(1, 1, 1, 3, dtype=torch.float64), torch.zeros(1, 1, 1, 3, dtype=torch.float64), 0)
ValueError: feature width must be even and at least 2
```

## Constraints

- Allowed: PyTorch tensor operations such as `reshape`, `cat`, `stack`, `cos`, `sin`, `arange`,
  `exp`, `torch.is_tensor`, plus the standard library (`math`).
- Not allowed: `torch.nn`, including `torch.nn.functional`. No library RoPE or rotary helper is
  allowed, since the rotation is the exercise. The test oracle works with complex numbers and
  `torch.polar`, and you may use complex arithmetic yourself if you convert the result back.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They check: agreement with a complex-number
reference on seeded tensors of several shapes, including zero-size axes; the half-split pairing, not
adjacent pairs, using a fixed permutation between the two layouts; the sign of the sine; position zero as
the identity; per-row norm preservation; the relative-offset property of query-key scores under a common
shift; the cached-decoding property, where a new token rotated at `offset = past_len` matches the full
sequence; the value-free contract (only two tensors returned); large offsets and small magnitudes; float32
inputs; inputs left unchanged; and every `ValueError` rule above. A wrong variant gets a named message, for
example the adjacent-pair layout, a shift applied to only one of q and k, or a block of positions that
starts at 1 instead of at `offset`.
