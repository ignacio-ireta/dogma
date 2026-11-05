# Deterministic two-layer training

Implement `train_mlp(X, y, hidden, steps, lr, seed)`, a complete full-batch
regression training loop. Use tanh hidden units, one linear output, half-mean-
squared loss, and simultaneous SGD updates. Return a parameter checkpoint and
an initial-plus-post-update loss history.

## Contract

- `X`: real numeric NumPy array `(n,d)`; `y`: `(n,1)`. Dimensions n,d are 1..32;
  entries are finite with absolute value <=10. Integer or strided arrays are
  valid; complex, bool, or object arrays are invalid. Convert to float64.
- `hidden`: integer 1..32; `steps`: integer 0..200; `seed`: integer 0..2**32-1.
  Boolean integer controls are invalid. `lr`: finite Python int/float, `0<lr<=1`.
- Create `rng=np.random.default_rng(seed)`. In this exact order, draw
  `W1=rng.standard_normal((d,hidden))/sqrt(d)` and
  `W2=rng.standard_normal((hidden,1))/sqrt(hidden)`. Biases `b1:(hidden,)`
  and `b2:(1,)` start at zero; bias initialization consumes no random draws.
- Compute `H=tanh(X@W1+b1)`, `prediction=H@W2+b2`, and
  `loss=mean((prediction-y)**2)/2`. Compute all four gradients from the same
  pre-update checkpoint, then subtract `lr*gradient` from each parameter.
- Return `((W1,b1,W2,b2), losses)`: all float64 arrays. `losses` has shape
  `(steps+1,)`, starting with the initial loss and followed by each post-update
  loss. Zero steps returns initial parameters and one loss. No shuffling,
  regularization, early stopping, disk I/O, or normalization beyond the loss.
- Raise `ValueError` for invalid shapes, dtypes, bounds, or hyperparameters.
  Raise `ValueError` also if training reaches a nonfinite loss checkpoint.
  Inputs and the global NumPy random state must remain unchanged.

## Examples

```python
params, losses = train_mlp(np.array([[0.0]]), np.array([[2.0]]), 1, 0, 0.1, 0)
# parameter shapes -> ((1,1), (1,), (1,1), (1,))
# losses -> array([2.])
```

```python
params, losses = train_mlp(np.array([[0.0]]), np.array([[2.0]]), 1, 1, 0.1, 0)
# W1 -> [[0.12573022]], W2 -> [[-0.13210486]]
# b1 -> [-0.02642097], b2 -> [0.2]
# losses -> approximately [2.0, 1.61372494]
```

## Constraints

Allowed: Python stdlib and NumPy. Not allowed: `torch`, `jax`, `tensorflow`, or
`sklearn`; derive the gradients yourself. Small deterministic arrays suffice.
Test tolerance is atol=1e-8, rtol=1e-6, including external finite differences.
Loss need not decrease for arbitrary valid datasets and learning rates.

## Done looks like

Initial and updated checkpoints follow the exact convention. Hidden families
cover initialization order, every parameter's finite-difference derivative,
multi-step numeric-gradient oracles, fixed-seed reproducibility, singleton and
integer data, a simple fit, input/global-RNG preservation, and invalid inputs.
