# Residual block with identity or projection shortcut

Implement `residual_block(x, w1, b1, w2, b2, w_shortcut=None)` for a two-layer residual
branch. The block returns `shortcut(x) + F(x)`, where `F(x) = relu(x @ w1 + b1) @ w2 + b2`.
When `w_shortcut` is `None` the shortcut is the identity; otherwise it is the linear map
`x @ w_shortcut`. The output must stay differentiable with respect to `x` and every weight:
the identity term of the input gradient must reach `x`.

## Contract

- `x`: floating-point torch tensor of shape `(..., d_in)`, with `d_in >= 1`. Leading axes are
  arbitrary and may be zero-sized.
- `w1`: shape `(d_in, h)`, `h >= 1`. `b1`: shape `(h,)`.
- `w2`: shape `(h, d_out)`, `d_out >= 1`. `b2`: shape `(d_out,)`.
- `w_shortcut`: `None`, or shape `(d_in, d_out)`.
- Every tensor must be floating point and have the same dtype as `x`.
- Returns: torch tensor of shape `(..., d_out)`, same dtype as `x`.
- Raises: `ValueError` when `x` is not a tensor or not floating point, when `x` is 0-dimensional,
  when any weight has the wrong shape or dtype, when `w_shortcut` is `None` and `d_out != d_in`
  (an identity shortcut needs matching widths), or when d_in, h or d_out is zero. Zero-sized leading axes are allowed.
- Must not mutate inputs. Must not call `detach()` on any input.

## Examples

```python
>>> x = torch.tensor([[1.0, -2.0]])
>>> residual_block(x, torch.eye(2), torch.zeros(2), torch.eye(2), torch.zeros(2))
tensor([[ 2., -2.]])
# F(x) = relu([1, -2]) @ I = [1, 0]; identity shortcut adds x = [1, -2]; sum = [2, -2].
```

```python
>>> residual_block(x, torch.eye(2), torch.zeros(2), torch.tensor([[1.0], [0.0]]),
...                torch.zeros(1), w_shortcut=torch.tensor([[0.5], [0.5]]))
tensor([[0.5000]])
# F(x) = [1]; projection shortcut x @ [0.5, 0.5]^T = 1*0.5 + (-2)*0.5 = -0.5; sum = 0.5.
```

```python
>>> residual_block(x, torch.eye(2), torch.zeros(2), torch.eye(2)[:, :1], torch.zeros(1))
ValueError: identity shortcut needs d_out == d_in; pass w_shortcut for a projection
```

## Constraints

- Allowed: torch tensor operations, including `torch.relu`, `torch.clamp`, `@` and `+`.
- Not allowed: `torch.nn` (including `torch.nn.functional`). Write the dense layers and the
  shortcut from tensor operations so the gradient path is yours.

## Done looks like

The public examples and the identity-width rejection pass. Hidden tests compare seeded random
blocks with a `torch.nn.functional` oracle, cover the projection and identity paths, check the
full input Jacobian against the identity-plus-branch decomposition, and cover zero-size batches,
float32 dtype, mixed dtype and integer rejection, scale extremes, input non-mutation, and
projection use even when widths match.
