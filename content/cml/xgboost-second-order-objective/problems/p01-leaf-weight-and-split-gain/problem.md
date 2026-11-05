# Leaf weight and split gain

A gradient-boosted tree grows one split at a time. Every training example carries a first-order gradient $g_i$ and a second-order Hessian $h_i$ of the loss, taken at the current prediction. For one candidate split, the samples fall into a left leaf and a right leaf. Compute the optimal weight of each leaf and the regularised gain of the split.

Implement `leaf_weight_and_split_gain(g_left, h_left, g_right, h_right, *, reg_lambda, gamma)`. Let $G_L = \sum_{i \in L} g_i$ and $H_L = \sum_{i \in L} h_i$, with $G_R$ and $H_R$ defined the same way on the right. With $\lambda$ = `reg_lambda` and $\gamma$ = `gamma`, return $(w_L, w_R, \text{gain})$, where

$$w_L = -\frac{G_L}{H_L + \lambda}, \qquad w_R = -\frac{G_R}{H_R + \lambda},$$

$$\text{gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma.$$

## Contract
- `g_left`, `g_right`: 1-D array-like of finite numbers. Either may be empty.
- `h_left`, `h_right`: 1-D array-like, the same length as the matching gradient array. Every entry is finite and zero or positive.
- `reg_lambda`: finite real number, zero or positive.
- `gamma`: finite real number, zero or positive.
- Each side must have $H + \lambda > 0$. An empty side therefore needs `reg_lambda > 0`, and so does a side whose hessians sum to zero.
- Returns: `(w_left, w_right, gain)`, three Python `float` values.
- Raises: `ValueError` for any violation of the rules above.
- Must not mutate inputs.

## Examples

```python
>>> leaf_weight_and_split_gain([-2.0, -1.0], [1.0, 1.0], [2.0], [1.0], reg_lambda=1.0, gamma=0.5)
(1.0, -1.0, 1.875)
>>> leaf_weight_and_split_gain([1.0], [1.0], [-1.0], [1.0], reg_lambda=1.0, gamma=0.0)
(-0.5, 0.5, 0.5)
>>> leaf_weight_and_split_gain([1.0], [1.0], [1.0], [1.0], reg_lambda=0.0, gamma=0.0)
(-1.0, -1.0, 0.0)
>>> leaf_weight_and_split_gain([1.0], [1.0], [1.0], [1.0], reg_lambda=-1.0, gamma=0.0)
ValueError: reg_lambda must be a finite number that is zero or positive
```

## Constraints
- Allowed: NumPy and the standard library (`math`). Write the leaf weight and gain formulas yourself.
- Not allowed: `xgboost`, `scipy`, `sklearn`, `statsmodels`, `torch`. Using a library tree learner would skip the formulas this problem is about.

## Done looks like
All public tests pass, then **Submit** runs the hidden tests. They check: the closed forms against an independent evaluation on random splits; that each leaf weight minimises the quadratic leaf objective (checked by finite differences); that the gain equals the drop in the regularised objective, computed numerically from the leaf minimums; that $\gamma$ lowers the gain one for one and a larger $\lambda$ shrinks the leaf weights; empty sides and zero-hessian sides with and without $\lambda$; rejection of negative regularisers, negative hessians, non-finite gradients and mismatched lengths; finite and accurate results at large and small magnitudes; invariance to sample order; and that the inputs are left unchanged.
