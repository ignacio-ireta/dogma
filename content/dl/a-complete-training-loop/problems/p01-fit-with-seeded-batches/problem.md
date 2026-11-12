# Seeded training loop with best-checkpoint restore

Implement `fit(model, X, y, X_val, y_val, *, epochs, batch_size, lr, seed)`. It trains a
`torch.nn.Module` with mini-batch stochastic gradient descent, returns the mean training loss and the
validation loss for every epoch, and leaves the model holding the parameters from the best epoch on
validation. The run must be reproducible from `seed`, so the contract fixes the optimizer, the loss and
the shuffle order.

## Contract

- `model`: a `torch.nn.Module` whose parameters already have the dtype of `X` and `y`. `fit` does not cast.
- `X`: `(n, d)` tensor, `n >= 1`. `y`: `(n, k)` tensor with the same `n`.
- `X_val`: `(m, d)` tensor, `m >= 1`. `y_val`: `(m, k)` tensor with the same `m`.
- `epochs`: `int >= 1`. `batch_size`: `int >= 1`; booleans are invalid. A `batch_size` above `n` means one batch.
- `lr`: a finite number `> 0`. `seed`: `int >= 0`.
- Returns: `(train_losses, val_losses)`, two Python lists of floats, each of length `epochs`.
- Raises `ValueError` when `X` and `y` have different row counts, when `X_val` and `y_val` have
  different row counts, or when `epochs`, `batch_size`, `lr` or `seed` is outside its range.
- Must not mutate `X`, `y`, `X_val` or `y_val`.

### Procedure

Follow these steps in order. Each rule is part of the contract, and the hidden tests check each one.

1. **Validate** the arguments and raise `ValueError` as listed above.
2. **Optimizer:** `opt = torch.optim.SGD(model.parameters(), lr=lr)`. No momentum, no weight decay, no scheduler.
3. **Loss:** the mean of squared errors over every element of a batch, `((pred - target) ** 2).mean()`.
   Batch losses are averaged over elements, not summed.
4. **One generator:** `g = torch.Generator().manual_seed(seed)`, created once at the start of `fit`.
   It is the only source of randomness. Do not use the global random state.
5. **Set up** `train_losses = []`, `val_losses = []`, `best_val = math.inf`, `best_state = None`.
6. **For each epoch** (`epochs` times):
   a. Draw the shuffle with exactly one call: `perm = torch.randperm(n, generator=g)`.
      Draw nothing else from `g` during the epoch.
   b. Call `model.train()`.
   c. **For each batch:** `idx = perm[start : start + batch_size]` for
      `start = 0, batch_size, 2*batch_size, ...` while `start < n`. The last batch may be partial,
      and it is kept. Then:
      i. `opt.zero_grad()`.
      ii. Compute `loss = ((model(X[idx]) - y[idx]) ** 2).mean()` on the current parameters.
      iii. `loss.backward()`, then `opt.step()`.
      iv. Add `loss.item() * len(idx)` to a running total. Use the value from before the step.
   d. `train_loss = running_total / n`. This is the batch-size-weighted mean, so the partial batch
      counts for its own rows only.
   e. Call `model.eval()`. Under `torch.no_grad()`, compute `val_loss` as the mean squared error over
      all `m` validation rows in one pass, `((model(X_val) - y_val) ** 2).mean()`. There is no batching
      for validation.
   f. **Best-state rule:** if `val_loss < best_val` (strictly), set `best_val = val_loss` and
      `best_state = copy.deepcopy(model.state_dict())`. A tie keeps the earlier epoch.
   g. Append `train_loss` and `val_loss` to the two lists.
7. **After the last epoch:** `model.load_state_dict(best_state)` and `model.eval()`. Return the two lists.

### Worked example: the shuffle order

Take `n = 10`, `batch_size = 4`, `seed = 0`. The batches are the slices `[0:4]`, `[4:8]`, `[8:10]`,
so the sizes are 4, 4, 2. With a generator seeded by `0`, the calls below produce these exact orders:

| Epoch | `perm` from `torch.randperm(10, generator=g)` | Batches (indices) | Sizes |
|---|---|---|---|
| 1 | `[4, 1, 7, 5, 3, 9, 0, 8, 6, 2]` | `[4,1,7,5]`, `[3,9,0,8]`, `[6,2]` | 4, 4, 2 |
| 2 | `[3, 9, 4, 2, 7, 8, 6, 0, 5, 1]` | `[3,9,4,2]`, `[7,8,6,0]`, `[5,1]` | 4, 4, 2 |

Epoch 2 is the second draw from the same generator. Creating a new generator each epoch would repeat
epoch 1's order and is wrong. The training loss for epoch 1 uses the three batches above in that order,
and the validation loss uses no shuffle at all.

## Examples

```python
# n = 10, batch_size = 4: each epoch runs batches of sizes 4, 4, 2 (the last one is kept).
train_losses, val_losses = fit(model, X, y, X_val, y_val, epochs=3, batch_size=4, lr=0.05, seed=0)
# len(train_losses) == len(val_losses) == 3
# With batch_size = 10 (one batch), the epoch's train loss is the loss measured before that one step.
```

```python
# Best-state rule with val_losses = [0.9, 0.6, 0.6, 0.65]:
# the model ends with the parameters saved after epoch 2 (the first 0.6), not epoch 4.
```

## Constraints

- Allowed: `torch.nn`, `torch.optim` (the topic is the loop, not the optimizer), `copy.deepcopy`,
  `torch.Generator` and `torch.randperm`.
- Not allowed: `torch.utils.data.DataLoader`, `torch.utils.data.RandomSampler` and
  `torch.utils.data.BatchSampler`. They draw their own shuffle order, which breaks the contract.
- Not allowed: any random draw other than the one `torch.randperm` per epoch from your generator.

## Done looks like

Your losses and final parameters match an independent reference loop that uses the same shuffle and
the same SGD arithmetic written out by hand. The hidden tests check: the partial last batch is weighted
by its size; the run reproduces from a seed and ignores the global random state; validation runs in
eval mode; the best epoch is restored even when validation rises later; tied validation losses keep the
earlier epoch; a batch size larger than `n` works; invalid arguments raise `ValueError`; and the inputs
are unchanged.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer` and `diabetes` (see
`docs/DATASETS.md`).
