# REINFORCE loss for a batch of episodes

Implement `reinforce_loss(logits, actions, rewards, gamma, baseline=False)`. Return the scalar
REINFORCE loss for a batch of episodes that all have the same length `T`. The function computes the
reward-to-go returns itself, so it takes the raw rewards. Use PyTorch tensor operations on the
logits, and let autograd differentiate the result with respect to the logits.

### The loss

For episode `b` and step `t`, let `G[b, t] = sum over k >= 0 of gamma**k * rewards[b, t + k]`, where
the sum stops at the last step `T - 1`. This is the discounted reward from step `t` onward.

If `baseline` is `False`, the advantage is `A[b, t] = G[b, t]`. If `baseline` is `True`, subtract the
mean over the batch of the returns at the same step: `A[b, t] = G[b, t] - mean over b' of G[b', t]`.
The mean includes episode `b` itself, so the baseline is not independent of the sample. In
expectation the centred gradient is `(1 - 1/B)` times the uncentred one. This O(1/B) shrink is the
bias of a batch-mean baseline, and it vanishes as `B` grows.

The loss is

`loss = -(1 / B) * sum over b and t of log_softmax(logits[b, t])[actions[b, t]] * A[b, t]`

where `B` is the number of episodes. Dividing by `B` makes the loss independent of the batch size.
Dividing by `T` is not done.

## Contract

- `logits`: floating-point `torch.Tensor` of shape `(B, T, A)`, with `B >= 1`, `T >= 1`, `A >= 1`.
  Values can be large; the loss must stay finite for logits up to about `1e3` in magnitude.
- `actions`: integer `torch.Tensor` of shape `(B, T)` with entries in `0 .. A-1`.
- `rewards`: floating-point `torch.Tensor` of shape `(B, T)`, every entry finite. `rewards[b, t]` is
  the reward received after the action at step `t`.
- `gamma`: real number in `[0, 1]`. `bool` and NaN are invalid.
- `baseline`: `bool`. Other types are invalid.
- Returns: a 0-dimensional tensor with the dtype of `logits`. When `logits` requires grad, the result
  must too, so that `backward()` gives gradients with respect to the logits.
- Raises `ValueError` for any shape, dtype, range, `gamma` or `baseline` violation, including empty
  batches or episodes.
- Must not mutate any input tensor.

## Examples

```python
logits = torch.zeros(1, 2, 2)  # one episode, two steps, two actions
actions = torch.tensor([[1, 0]])
rewards = torch.tensor([[0.0, 1.0]])
reinforce_loss(logits, actions, rewards, gamma=1.0)
# -> tensor(1.3863)   which is -2 * log(0.5): G = [1, 1] and each log-probability is log(0.5)
```

```python
# gamma = 0 keeps only the immediate reward. With one episode and a baseline, the advantage is zero
# because the baseline is the only return.
reinforce_loss(logits, actions, rewards, gamma=0.0, baseline=True)
# -> tensor(0.)
```

## Constraints

Allowed: Python standard library and PyTorch tensor operations, such as `log_softmax`, `gather`,
`sum`, `mean` and `flip`. Not allowed: `torch.nn` (this covers `torch.nn.functional`, so no
`cross_entropy`), `torch.distributions`, `torch.autograd`, `torch.func`, and `numpy`. Do not call
`backward()` inside the function.

## Done looks like

The examples pass. Hidden tests compare the loss with an independent cross-entropy reference on
random batches with and without the baseline, check the logit gradient against the closed-form score
function and against finite differences in float64, check that a constant-return batch gives zero
baseline loss and zero gradient, cover gamma 0 and 1, extreme logits, float32 inputs, a single
action, invalid arguments, and input mutation.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).
