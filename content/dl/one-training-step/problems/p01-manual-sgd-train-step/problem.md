# One manual SGD step

Implement `train_step(model, x, y, loss_fn, lr)`. It performs exactly one stochastic-gradient-descent
update on a batch and returns the loss from before the update. Do it by hand, in the five steps of a
training step: clear the old gradients, run the forward pass, compute the scalar loss, run backward,
and update the parameters. No optimizer object.

## Contract

- `model`: a `torch.nn.Module` whose parameters are floating tensors with `requires_grad=True`.
- `x`, `y`: the batch inputs and targets. `loss_fn(prediction, y)` returns a 0-dimensional tensor.
- `lr`: a finite `int` or `float` greater than zero. Booleans are invalid.
- Procedure, in this order:
  1. Clear every parameter's `.grad` (set it to `None`).
  2. Compute `prediction = model(x)` and `loss = loss_fn(prediction, y)`.
  3. Call `loss.backward()`, so each parameter's `.grad` holds the gradient for this batch.
  4. Under `torch.no_grad()`, update each parameter in place: `p <- p - lr * p.grad`. Parameters
     whose `.grad` is `None` stay unchanged.
  5. Return `float(loss)` computed before step 4.
- After the call, every parameter is still the same object, still a leaf, still has
  `requires_grad=True`, and its `.grad` still holds the gradient used for the update.
- Returns: a Python `float`.
- Raises: `ValueError` when `lr` is not a finite number greater than zero, or when `loss_fn`
  returns a tensor that is not 0-dimensional.
- Must not mutate `x` or `y`.

## Examples

```python
# A 1-to-1 linear model with weight 2 and bias 0, one input 1.0, target 0.0, MSE loss.
# prediction = 2, loss = (2 - 0)**2 = 4, dloss/dweight = 4, dloss/dbias = 4.
train_step(model, x=torch.tensor([[1.0]]), y=torch.tensor([[0.0]]), loss_fn=mse, lr=0.1)
# -> 4.0; weight becomes 1.6, bias becomes -0.4
```

```python
# Same model, after the call above: the next call starts fresh, it does not add to the old gradient.
train_step(model, x=torch.tensor([[1.0]]), y=torch.tensor([[0.0]]), loss_fn=mse, lr=0.1)
# -> the pre-update loss of the new parameters, (1.6 - 0.4)**2 = 1.44
```

## Constraints

- Allowed: any `torch` tensor operations, `torch.no_grad`, and the model's own `parameters()`.
- Not allowed: `torch.optim` (including `torch.optim.SGD`). Writing your own update is the point.
- Not allowed: `torch.Tensor.data`, which means `.data` (for example `p.data.sub_(...)`). It skips autograd's bookkeeping, so the
  update looks right here and breaks elsewhere. Update the existing tensors in place under `torch.no_grad()`.
- Do not build a new `Parameter` object: update the existing tensors in place.

## Done looks like

Your function matches a `torch.optim.SGD` run over several steps, in parameters and in returned
losses. The hidden tests check repeated calls on the same batch (stale gradients), that the update
is applied and not tracked by autograd, that the returned loss is the pre-update value, the sign and
size of the step, that parameters keep their identity, degenerate zero-gradient batches, input
validation, and that `x` and `y` are unchanged.
