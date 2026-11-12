# Sigmoid MSE gradient

Implement `mse_sigmoid_grad(w, X, y)`, which returns the gradient with respect to the weight vector
`w` of the one-unit model with a squared-error loss on sigmoid outputs:

$$L(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n}\big(\sigma(z_i) - y_i\big)^2,
\qquad z_i = \mathbf{x}_i^\top\mathbf{w},
\qquad \sigma(z) = \frac{1}{1 + e^{-z}}.$$

Differentiate with the chain rule. The path from $\mathbf{w}$ to $L$ is $\mathbf{w} \to z_i \to
\sigma(z_i) \to (\sigma(z_i) - y_i)^2$. Multiply the local derivative of each link along that path, then
sum over the rows. The result is

$$\nabla_{\mathbf{w}} L = \frac{2}{n}\,X^\top\big[(\sigma(z) - y) \odot \sigma(z) \odot (1 - \sigma(z))\big].$$

## Contract

- `w`: 1-D array-like of `d` finite real numbers.
- `X`: 2-D array-like of shape `(n, d)`, with $n \geq 1$ and $d \geq 1$. Each row is one sample.
- `y`: 1-D array-like of shape `(n,)`, with one target per row of `X`.
- Returns: a NumPy array of shape `(d,)` and dtype `float64`, the gradient of $L$ with respect to `w`.
- The sigmoid must be numerically stable for large $|z|$. Never exponentiate a large positive
  number, because $e^{z}$ overflows to `inf` for $z$ above about 709.
- The gradient must keep relative precision deep in saturation. The product $s(1-s)$ becomes `0` once
  $s$ rounds to 1, so do not compute the sigmoid's derivative that way.
- Raises `ValueError` when:
  - `X` is not 2-D, or `w` is not 1-D.
  - `y` is not 1-D. A column vector of shape `(n, 1)` is rejected.
  - `y` does not have one entry per row of `X`, or `w` does not have one entry per column of `X`.
  - `X` has no rows or no columns.
  - Any entry of `w`, `X` or `y` is NaN or infinite.
- Must not mutate its inputs.

## Examples

```python
>>> np.round(mse_sigmoid_grad(np.array([0.0]), np.array([[1.0], [2.0]]), np.array([1.0, 0.0])), 6)
array([0.125])
>>> np.round(mse_sigmoid_grad(np.zeros(2), np.eye(2), np.full(2, 0.5)), 6)  # predictions already match
array([0., 0.])
>>> mse_sigmoid_grad(np.zeros(1), np.ones((2, 1)), np.ones((2, 1)))
ValueError: y must be 1-D with one target per row of X
```

## Constraints

- Allowed: NumPy and the `math` module. Write the sigmoid yourself.
- Not allowed: `scipy.special.expit` (the stable sigmoid is part of the exercise), and automatic
  differentiation libraries such as `torch`, `jax` and `autograd`. Derive the gradient by hand.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: agreement with central
finite differences of the loss on random problems; closed-form values for one sample and for three
samples, which pin the sigmoid-derivative factor and the $2/n$ mean; a deep-saturation case checked
against a high-precision reference; large positive and negative logits that must neither overflow
nor produce NaN; a set of shape mistakes, including a column vector `y`; NaN and infinite inputs;
integer inputs; and that the inputs are left unchanged.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).
