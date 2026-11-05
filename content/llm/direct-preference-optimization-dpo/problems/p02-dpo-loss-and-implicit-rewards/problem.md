# DPO loss and implicit rewards from sequence log-probabilities

Direct preference optimization trains a policy on preference pairs, with a frozen reference model
as an anchor. For each pair, the input is the sequence log-probability of the chosen and of the
rejected response under the policy and under the reference. Sequence log-probabilities are already
summed over response tokens, so this step does not touch tokens.

Implement `dpo_loss_and_implicit_rewards(policy_chosen, policy_rejected, ref_chosen, ref_rejected, *,
beta)`. It returns the DPO loss and the two implicit rewards. With
$\pi$ the policy, $\pi_\text{ref}$ the reference, and $\beta > 0$:

$$\hat r_c = \beta\,[\log\pi(y_c \mid x) - \log\pi_\text{ref}(y_c \mid x)], \qquad
\hat r_r = \beta\,[\log\pi(y_r \mid x) - \log\pi_\text{ref}(y_r \mid x)]$$

$$\mathcal{L} = \frac{1}{B}\sum_{i=1}^{B} -\log\sigma(\hat r_{c,i} - \hat r_{r,i})$$

## Contract

- `policy_chosen`, `policy_rejected`, `ref_chosen`, `ref_rejected`: floating-point tensors, shape
  `(B,)`, `B >= 1`, all with the same dtype. Element `i` is the sequence log-probability of that
  response for pair `i`. Every value must be finite and `<= 0`.
- `beta`: keyword-only Python float, finite and `> 0`.
- Returns a tuple `(loss, chosen_reward, rejected_reward)`:
  - `loss`: 0-dimensional tensor, mean over pairs of $-\log\sigma(\hat r_c - \hat r_r)$.
  - `chosen_reward`, `rejected_reward`: tensors of shape `(B,)` holding $\hat r_c$ and $\hat r_r$.
  - All three keep the input dtype.
- Gradients flow into the policy arguments. The caller is responsible for passing reference values
  that carry no gradient.
- Raises `ValueError` when: an argument is not a tensor; any tensor is not 1-D; shapes differ;
  `B == 0`; the dtype is not floating-point or the dtypes differ; any value is not finite or is
  greater than zero (a probability, not a log-probability); `beta` is not a finite positive number.
- Must not mutate inputs.

## Examples

Tensors are shown as nested lists for brevity. Pass `torch.tensor(...)` values.

```python
dpo_loss_and_implicit_rewards(
    policy_chosen=[-10.0],
    policy_rejected=[-12.0],
    ref_chosen=[-11.0],
    ref_rejected=[-11.0],
    beta=0.1,
)
# rewards: chosen = 0.1 * (-10 + 11) = 0.1,  rejected = 0.1 * (-12 + 11) = -0.1
# margin = 0.2, loss = -log(sigmoid(0.2)) ~= 0.5981389
```

```python
dpo_loss_and_implicit_rewards(
    policy_chosen=[-4.0],
    policy_rejected=[-2.0],
    ref_chosen=[-3.0],
    ref_rejected=[-3.0],
    beta=0.5,
)
# rewards: chosen = 0.5 * (-1) = -0.5,  rejected = 0.5 * (+1) = 0.5
# margin = -1.0, so the policy prefers the rejected response; loss ~= 1.3132617
```

```python
dpo_loss_and_implicit_rewards(
    policy_chosen=[-3.0],
    policy_rejected=[-5.0],
    ref_chosen=[-3.0],
    ref_rejected=[-5.0],
    beta=0.1,
)
# the policy equals the reference: both rewards are 0.0, loss = log 2 ~= 0.6931472
```

## Constraints

- Allowed: Python standard library and PyTorch tensor operations, such as `clamp`, `abs`, `exp`,
  `log1p`, `mean`, `isfinite`.
- Not allowed: `torch.nn` (this covers `torch.nn.functional`, so `F.logsigmoid` and `F.softplus`
  are banned too), and `torch.binary_cross_entropy_with_logits`, which computes this loss as a whole.
  Write the stable form of $-\log\sigma$ yourself.

## Done looks like

The three examples above pass. The hidden tests compare the loss and both rewards with an independent
log-sigmoid reference over several values of $\beta$, check that rewards scale linearly with $\beta$,
that a shared constant added to a policy log-probability and its reference cancels, that a policy
equal to its reference gives $\log 2$, that margins of plus and minus $10^4$ stay finite, and that
float32 keeps its dtype. They also check the gradient with respect to the policy arguments against the
analytic form and against a float64 finite-difference estimate, and the rules on $\beta$, probabilities
instead of log-probabilities, shapes, empty input, NaN, integer dtypes and mutation.
