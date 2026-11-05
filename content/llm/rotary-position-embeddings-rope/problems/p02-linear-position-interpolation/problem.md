# RoPE with linear position interpolation

Implement `apply_rope_scaled(q, k, offset, scale)`. It is `apply_rope_qk` from the core problem,
with one change: every position is divided by `scale` before it is turned into an angle. This is
linear position interpolation. A model trained on context length `L` can read a window about
`scale` times longer, because the extra positions are squeezed into the angles it already saw.

The layout is the same half-split layout as the core problem. Pair `i` is `(x_i, x_{i+d/2})`. The
value tensor is never rotated.

## Contract

- `q`, `k`: floating-point tensors of the same shape `(B, H, T, d)`, with `B >= 0`, `H >= 0`,
  `T >= 0`, and `d` even and at least 2. They have the same dtype.
- `offset`: `int` (not `bool`) and at least 0.
- `scale`: a finite real number greater than 0, given as `int` or `float` (not `bool`). `scale = 1`
  means no interpolation.
- Token `t` (`0 <= t < T`) sits at the scaled position $s_t = (\mathrm{offset} + t)/\mathrm{scale}$.
- Frequencies: $\theta_i = 10000^{-2i/d}$ for $i = 0,\dots,d/2-1$. The base is **not** changed.
- Angle $\phi_{t,i} = s_t\,\theta_i$. Rotation, with $a = x[\dots,:d/2]$ and $b = x[\dots,d/2:]$:
  - `out[..., :d/2] = a * cos(phi) - b * sin(phi)`
  - `out[..., d/2:] = a * sin(phi) + b * cos(phi)`
- Returns: a tuple `(q_rot, k_rot)`, each of shape `(B, H, T, d)` and the input dtype.
- Consequences you can check: `scale = 1` gives the core problem's output; the lowest pair (`i = 0`,
  with `theta_0 = 1`) rotates token `t` by exactly `(offset + t) / scale` radians; each row keeps its
  norm; query-key scores depend only on the difference of positions, which is divided by `scale`.
- Raises `ValueError` when: `q` or `k` is not a 4-D floating tensor; their shapes or dtypes differ;
  `d` is odd or zero; `offset` is not a non-negative `int`; `scale` is a `bool`, not a number, not
  finite, or not greater than 0.
- Must not mutate `q`, `k`, `offset` or `scale`.

## Examples

```python
>>> q = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float64).reshape(1, 1, 2, 2)
>>> k = torch.zeros_like(q)
>>> q_rot, _ = apply_rope_scaled(q, k, offset=0, scale=2)
>>> q_rot[0, 0]
tensor([[1.0000, 0.0000],
        [0.8776, 0.4794]], dtype=torch.float64)
# Token 1 sits at position 1/2, so pair 0 rotates it by 0.5 radians: (cos 0.5, sin 0.5).
```

```python
>>> q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
>>> q_rot, _ = apply_rope_scaled(q, torch.zeros_like(q), offset=3, scale=2)
>>> q_rot
tensor([[[[0.0707, 0.9975]]]], dtype=torch.float64)
# The offset is scaled too: position (3 + 0) / 2 = 1.5, so the rotation is (cos 1.5, sin 1.5).
```

```python
>>> apply_rope_scaled(q, torch.zeros_like(q), offset=0, scale=0)
ValueError: scale must be a finite number greater than 0
```

## Constraints

- Allowed: PyTorch tensor operations such as `reshape`, `cat`, `stack`, `cos`, `sin`, `arange`,
  `exp`, `torch.is_tensor`, plus the standard library (`math`).
- Not allowed: `torch.nn`, including `torch.nn.functional`. No library RoPE or rotary helper is
  allowed. Do not change the base `10000`, since that is a different method from this one.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They check: agreement with a complex-number
reference for several scales, including a non-integer one; `scale = 1` as the unscaled rotation; the
lowest pair rotating by exactly `(offset + t) / scale`, which a changed base cannot produce; the offset
being scaled together with the position, not added after scaling; relative-offset invariance of scores
under a common shift; large offsets; extreme magnitudes; float32; inputs left unchanged; and every
`ValueError` rule above. A wrong variant gets a named message, for example a changed base instead of
scaled positions, or an offset that is not divided by the scale.
