# Huber loss and its gradient

Implement `huber_loss_and_grad(residuals, delta)`. It returns the mean Huber loss of a vector of
residuals and the gradient of that mean loss with respect to each residual. The Huber loss is
quadratic for small residuals and linear for large ones. Because its slope is capped at `delta`, a
large residual has a bounded loss derivative. This does not protect against arbitrarily large feature vectors in a regression fit.

## Contract

- `residuals`: 1-D array-like of finite real numbers, length n ≥ 1.
- `delta`: a finite real number strictly greater than 0.
- Per residual r, with threshold δ = `delta`:
  - ℓ(r) = ½ r² when |r| ≤ δ, and δ (|r| − δ/2) otherwise.
  - ψ(r) = r when |r| ≤ δ, and δ · sign(r) otherwise. ψ is the derivative of ℓ.
- Returns: `(loss, grad)`.
  - `loss` = (1/n) Σ ℓ(rᵢ), a Python float.
  - `grad` = float64 array of shape (n,) with gradᵢ = ψ(rᵢ) / n. It is the derivative of `loss` with respect to each rᵢ, so it carries the factor 1/n.
- Raises `ValueError` when `residuals` is not 1-D, is empty, or contains NaN or infinity, or when `delta` is not a finite number greater than 0.
- Residuals may be as large as `1e308`. Square a residual only on the quadratic branch, and do not let
  the sum overflow: three residuals of `1e308` with `delta = 1` have a `loss` near `1e308`.
- Must not mutate inputs. The returned gradient must be a new array.

## Examples

```python
>>> huber_loss_and_grad([0.5, -0.5, 3.0], 1.0)
(0.9166666666666666, array([ 0.16666667, -0.16666667,  0.33333333]))
>>> huber_loss_and_grad([-4.0], 2.0)
(6.0, array([-2.]))
>>> huber_loss_and_grad([1.0], 0.0)
ValueError   # delta must be greater than 0
```

## Constraints

- Allowed: NumPy.
- Not allowed: `scipy` (including `scipy.special.huber`, which is the oracle) and `sklearn`. Write the
  piecewise formulas yourself.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare the loss with SciPy's
Huber function on random residuals, and the gradient with central finite differences of the mean loss.
They check each branch on its own, the continuity at |r| = δ, and that the gradient is bounded by δ/n
for outliers. They also check the homogeneity rule (scaling the residuals and δ together), that the loss
is a mean and not a sum, the rejected inputs (empty, non-finite, non-positive or non-finite δ, wrong
shape), integer input, and that the caller's array is left unchanged.
