# Early-stopping point

Implement `early_stopping(val_losses, *, patience, min_delta=0.0)`. Training produces one validation loss per
epoch. Early stopping ends training once the validation loss has failed to improve for `patience` epochs. The
function returns the epoch of the best checkpoint and the epoch at which training stops. Epochs are 0-based.
The rule is Keras-style: an improvement must beat the best loss so far by more than `min_delta`, and training
stops when the wait counter reaches `patience`.

## Contract

- `val_losses`: 1-D array-like of finite real numbers, one per epoch, epoch 0 first. Non-empty.
- `patience`: int, at least 1.
- `min_delta`: float, finite and non-negative. Default `0.0`.
- **Improvement.** Let `best` be the loss at the last improvement, or `+infinity` before any. Epoch `e` improves
  when `val_losses[e] < best - min_delta`. A tie with the best is not an improvement, and a decrease of exactly
  `min_delta` is not an improvement either.
- **Bookkeeping.** An improving epoch sets `best` to its loss, records `best_epoch = e`, and resets `wait` to 0.
  Any other epoch adds 1 to `wait`.
- **Stopping.** Training stops at the first epoch where `wait` reaches `patience`. That epoch is `stop_epoch`.
  If it never does, `stop_epoch` is the last epoch.
- Returns: `(best_epoch, stop_epoch)` as Python ints. `best_epoch` is the last improvement at or before
  `stop_epoch`.
- Raises `ValueError` when `val_losses` is empty, not 1-D, or contains NaN or infinity; when `patience < 1`; or when
  `min_delta` is negative or not finite.
- Must not mutate inputs.

## Examples

```python
>>> early_stopping([1.0, 0.8, 0.7, 0.72, 0.71, 0.75, 0.74], patience=3)
(2, 5)          # best at epoch 2; epochs 3, 4 and 5 do not improve, so training stops at epoch 5
>>> early_stopping([1.0, 0.75, 0.6], patience=5, min_delta=0.25)
(2, 2)          # 0.75 is exactly 0.25 below 1.0, which is not an improvement; 0.6 is
>>> early_stopping([0.5, 0.5, 0.5, 0.5], patience=2)
(0, 2)          # ties are not improvements, so epoch 2 is the stop
```

## Constraints

- Allowed: the standard library and NumPy.
- Not allowed: `keras` and `tensorflow`. Their `EarlyStopping` callbacks would implement this rule for you.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. They compare the result with a brute-force reference
over seeded random loss sequences and several patience and `min_delta` settings. They check the exact-boundary rule
for `min_delta`, ties, a patience of 1, a single epoch, and a sequence that never triggers the stop. They check that
the wait counter resets after an improvement, that the best checkpoint is the earliest minimal epoch when
`min_delta` is zero, and that a triggered stop comes exactly `patience` epochs after the best checkpoint; without a trigger,
`stop_epoch` is the final available epoch. The hidden tests
also reject invalid arguments and check that inputs are left unchanged.
