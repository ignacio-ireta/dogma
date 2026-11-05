# Forward and backward through a scalar graph

Implement `forward_backward(x, w, b, t)`. It evaluates a fixed graph of scalar operations and
then runs the backward pass, returning the loss and the gradient of the loss with respect to each
of the four inputs. The graph has eight steps, applied in this order:

1. `p = w * x`
2. `z = p + b`
3. `h = tanh(z)`
4. `q = h * w`
5. `e = exp(h)`
6. `y = q + e`
7. `r = y - t`
8. `L = r * r`

`L` is the loss. `w` feeds steps 1 and 4, `h` feeds steps 4 and 5, and `r` feeds step 8 twice.
Each use is a separate path back to the input, and the gradient of a value is the sum of the
contributions from its paths. Getting that sum right is the point of the problem.

## Contract

- `x`, `w`, `b`, `t`: real numbers, `int` or `float`. Each must be finite.
- Returns: a tuple `(loss, grads)`. `loss` is a `float` equal to `L`. `grads` is a `dict` with
  exactly the keys `"x"`, `"w"`, `"b"` and `"t"`, each a `float` equal to the partial derivative of
  `L` with respect to that input.
- Raises `ValueError` when any input is NaN or infinite.
- Must not mutate inputs. Each call is independent: no state may carry over from one call to the
  next.
- Plain Python floats only. Return analytic graph derivatives evaluated in floating point,
  rather than numerical derivative estimates. Finite inputs whose graph intermediates or returned
  values overflow are outside the supported representable-arithmetic scope.

## Examples

```python
>>> loss, g = forward_backward(1.0, 0.5, 0.0, 0.0)
>>> round(loss, 6), {k: round(v, 6) for k, v in g.items()}
(3.306905, {'x': 2.985334, 'w': 7.651379, 'b': 5.970668, 't': -3.63698})
```

```python
>>> forward_backward(0.0, 0.0, 0.0, 0.0)
(1.0, {'x': 0.0, 'w': 0.0, 'b': 2.0, 't': -2.0})
```

```python
>>> forward_backward(float("nan"), 0.5, 0.0, 0.0)
ValueError: input x must be a finite number
```

## Constraints

- Allowed: the `math` module and plain Python floats.
- Not allowed: `numpy`, `torch` and `jax`, or any other autodiff library. Write the backward
  pass by hand, one node at a time.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They compare the gradients with the
closed-form derivatives on seeded random inputs, and with central differences of the forward pass.
They check a case where the residual is zero, a case where `tanh` saturates so its local derivative
is exactly zero, and integer inputs. They check that repeated calls do not share state, and that
NaN and infinite inputs raise `ValueError` for every argument. They also check the return shape:
a tuple of a float and a dictionary with exactly four keys.
