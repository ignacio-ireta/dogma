# Binary cross-entropy from logits

Implement `binary_cross_entropy_with_logits(z, y)`, the mean binary cross-entropy of logits `z` against
targets `y`. Each logit $z_i$ is a score for the positive class, and $y_i \in [0, 1]$ is the target. The
targets are usually 0 or 1, and soft targets such as 0.1 or 0.95 (label smoothing) are also valid. The
per-example loss is

$$\ell_i = -\big[y_i \log\sigma(z_i) + (1 - y_i)\log(1 - \sigma(z_i))\big] = \max(z_i, 0) - z_i\,y_i + \log\!\big(1 + e^{-|z_i|}\big),$$

and the function returns $\frac{1}{n}\sum_i \ell_i$. The second form is the one to implement, because
$\log\sigma(z)$ underflows to $\log 0$ for large negative $z$, and $1 - \sigma(z)$ rounds to $0$ for large positive $z$.

## Contract

- `z`: 1-D array-like of finite real numbers, length $n \ge 1$.
- `y`: 1-D array-like of finite values in $[0, 1]$, the same length as `z`.
- Returns: a Python `float`, the mean loss over the $n$ examples. Never negative.
- Raises `ValueError` when: `z` or `y` is not 1-D; their lengths differ; `z` is empty; any entry of `z` is NaN or infinite;
  any entry of `y` is NaN, infinite, below 0 or above 1.
- Each per-example loss must keep **relative** precision, including the tiny loss of a confident correct
  prediction (a large positive logit with target 1, or a large negative logit with target 0): avoid any form that
  computes it as the difference of two large, nearly equal numbers.
- The mean must not overflow when it is representable: three losses of `1e308` have the mean `1e308`.
- Must not mutate its inputs. Must not emit a NumPy overflow, invalid-value or divide warning for any finite input.

## Examples

```python
>>> round(binary_cross_entropy_with_logits(np.array([0.0, 0.0]), np.array([1.0, 0.0])), 6)
0.693147
>>> round(binary_cross_entropy_with_logits(np.array([2.0, -1.0]), np.array([1.0, 0.0])), 6)
0.220095
>>> binary_cross_entropy_with_logits(np.array([800.0]), np.array([0.0]))    # finite, equals 800
800.0
>>> binary_cross_entropy_with_logits(np.array([0.0]), np.array([1.5]))
ValueError: targets must lie in [0, 1]
```

## Constraints

- Allowed: NumPy and the standard library.
- Not allowed: `scipy`, `sklearn` and `torch`, including `scipy.special.log_expit` and `scipy.special.expit`.
  The stable form is the exercise.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: agreement with an independent stable
reference on moderate logits; agreement with a 60-digit reference at logits of magnitude up to $10^3$ (to relative precision, so tiny
confident-correct losses must not round to zero), where a
probability-based formula returns the wrong value or NaN; no overflow warning for any logit in the finite
range; soft targets; the mean, not the sum; integer input; every rejection family (shapes and lengths, empty
input, NaN and infinite logits, targets outside $[0, 1]$, NaN targets); and that the inputs are left unchanged.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).
