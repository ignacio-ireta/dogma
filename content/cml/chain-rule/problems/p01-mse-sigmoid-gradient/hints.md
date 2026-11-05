## Hint 1 — concept

Write the loss as a path of simple links: `w -> z = X w -> s = sigmoid(z) -> (s - y)**2`. The
chain rule says the derivative along a path is the product of the local derivatives of its links,
and the gradient sums the paths over all samples. Each link contributes one factor, and each factor
has a known shape: the square gives a residual, the sigmoid gives `s (1 - s)`, and the linear map
gives `X.T`.

## Hint 2 — formula / approach

With `z = X @ w`, `s = sigmoid(z)` and `n` rows, the gradient is
`(2/n) * X.T @ ((s - y) * s * (1 - s))`. Two details matter. First, compute the sigmoid from
`e = exp(-abs(z))`, so that no exponent is ever large and positive. Second, the local derivative
`s (1 - s)` equals `e / (1 + e)**2`, which keeps its digits when `s` is close to 1.

## Hint 3 — code shape

Convert the three inputs with `np.asarray(..., dtype=np.float64)` and validate them before any
arithmetic: `X` must be 2-D, `w` and `y` 1-D, the lengths must agree, `X` must be non-empty, and
every entry must be finite. Then compute `z`, `e`, `s` with `np.where(z >= 0, ...)`, the local
derivative, the residual scaled by `2/n`, and finish with `X.T @ (outer * local)`.
