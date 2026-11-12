# Numerical derivative

Implement `numerical_derivative(f, x, method="central", h=None)`, which estimates the derivative
$f'(x)$ of a scalar function $f:\mathbb{R}\to\mathbb{R}$ at a point $x$ with a finite difference.
Two schemes are supported: the one-sided **forward** difference $\dfrac{f(x+h)-f(x)}{h}$ and the
symmetric **central** difference $\dfrac{f(x+h)-f(x-h)}{2h}$. When `h` is not given, choose a step
that scales with `x`, because a step that is right near $x = 1$ is wrong near $x = 10^8$.

## Contract

- `f`: a callable that takes one float and returns a real number. Only evaluate it; never change
  anything it is given.
- `x`: a finite real number (an int is fine). A non-finite `x` raises `ValueError`.
- `method`: exactly `"forward"` or `"central"`. Any other value raises `ValueError`. Do not fall
  back to a default scheme.
- `h`: `None` for the default step, or a positive finite number. The default is
  $h = \varepsilon^{1/2}\max(1,|x|)$ for forward and $h = \varepsilon^{1/3}\max(1,|x|)$ for
  central, where $\varepsilon$ is the float64 machine epsilon (`np.finfo(np.float64).eps`).
  A zero, negative, NaN or infinite `h` raises `ValueError`.
- Returns: a Python `float`.
- Raises: `ValueError` for an unknown `method`, a non-finite `x`, or an invalid `h`.

## Examples

```python
>>> round(numerical_derivative(lambda t: t**3 - 2 * t + 1, 2.0), 6)   # 3x^2 - 2 at x = 2
10.0
>>> round(numerical_derivative(lambda t: t**3, 1.0, h=0.1), 6)         # central: 3x^2 + h^2
3.01
>>> round(numerical_derivative(lambda t: t**3, 1.0, method="forward", h=0.1), 6)  # 3x^2 + 3xh + h^2
3.31
>>> numerical_derivative(lambda t: t, 0.0, method="sideways")
ValueError: method must be "forward" or "central"
```

## Constraints

- Allowed: the `math` module and NumPy (`np.finfo`, `np.isfinite`, ...).
- Not allowed: `scipy.misc` and `scipy.differentiate` (library derivative routines), and
  `numpy.gradient`. Write the difference quotient yourself.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: the central method is
accurate to about 1e-8 relative error on smooth functions, and the forward method to about 1e-6;
the exact truncation error of a cubic, which is $h^2$ for central and $3xh + h^2$ for forward; the
default step at large and at tiny `x`, for both schemes and for negative as well as positive `x`;
agreement with a complex-step reference on a composite function; a random polynomial; integer `x`;
and the validation of `method` (exact spelling only), `x` and `h`.

The hidden tests draw their bulk inputs from the real dataset `wine` (see `docs/DATASETS.md`).
