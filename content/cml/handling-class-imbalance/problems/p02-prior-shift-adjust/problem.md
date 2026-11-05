# Prior-shift correction of probabilities

Implement `prior_shift_adjust(probs, train_prior, deploy_prior)`. A model was trained where the positive class had base rate `train_prior`, and it will be used where the base rate is `deploy_prior`. Its probabilities are correct for the training base rate. The function returns them corrected for the deployment base rate, so the same evidence gives the probability it would have in deployment.

The correction assumes a prior shift: the distribution of the features given the class is the same in both settings, and only the mix of classes changes. Bayes' rule then gives the rule. With a likelihood ratio $L(x)=P(x\mid y{=}1)/P(x\mid y{=}0)$, the posterior odds are the prior odds times $L(x)$. Because $L(x)$ does not change, the odds move by the ratio of the two prior odds:

$$r=\frac{\pi_d/(1-\pi_d)}{\pi_t/(1-\pi_t)},\qquad p_d=\frac{r\,p}{r\,p+1-p},$$

where $p$ is the probability under the training prior $\pi_t$ and $p_d$ the probability under the deployment prior $\pi_d$.

## Contract

- `probs`: 1-D sequence or NumPy array of probabilities P(y = 1 | x) under the training prior. Each value is finite and in [0, 1]. The length may be 0.
- `train_prior`, `deploy_prior`: numbers strictly between 0 and 1. They may be as small as the smallest positive float64, and the result must stay finite and accurate there, so a direct odds ratio is not enough.
- Returns a NumPy float array of the same length, with the deployment probabilities.
- A probability of exactly 0 maps to 0, and exactly 1 maps to 1. Equal priors return the input unchanged.
- Raises `ValueError` when `probs` is not 1-D, when a value is NaN or outside [0, 1], or when a prior is not strictly between 0 and 1.
- Must not mutate `probs`.

## Examples

```python
>>> prior_shift_adjust([0.8], 0.5, 0.01)
array([0.03883495])     # r = (0.01 / 0.99) / (0.5 / 0.5) = 0.0101; odds 4 * r = 0.0404; p = 0.0404 / 1.0404
```

```python
>>> prior_shift_adjust([0.2], 0.1, 0.5)
array([0.69230769])     # r = (0.5 / 0.5) / (0.1 / 0.9) = 9; odds 0.25 * 9 = 2.25; p = 2.25 / 3.25
```

```python
>>> prior_shift_adjust([0.0, 0.5, 1.0], 0.2, 0.2)
array([0. , 0.5, 1. ])  # equal priors: r = 1, nothing changes
```

## Constraints

- Allowed: the Python standard library and NumPy.
- Not allowed: `sklearn`. The formula is the point of the exercise.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. They compare the output with an independent route: the likelihood ratio is recovered from each training probability, and the deployment posterior is computed by Bayes' rule with the deployment prior. They check a seeded simulation of two Gaussian classes, where the mean of the corrected probabilities matches the deployment base rate and the corrected values match the true deployment posterior. They check that equal priors are the identity, that 0 and 1 stay exactly 0 and 1, that outputs stay in [0, 1] and are non-decreasing in the input, that the correction is undone by swapping the priors, and that invalid probabilities and priors raise `ValueError`. They also check that the caller's array is unchanged.
