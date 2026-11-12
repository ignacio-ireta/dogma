# Dense layer forward for a batch

Implement `dense_forward(X, W, b)`, the forward pass of one dense (fully connected) layer on a batch.
Each row of `X` is one example. The output is `Y = X @ W + b`, with the bias added to every row.
The weights use the lesson's layout: `W` has shape `(d_in, d_out)`, which is the transpose of
PyTorch's `nn.Linear` storage.

## Contract

- `X`: 2-D array-like, shape `(n, d_in)`, real numbers. One example per row.
- `W`: 2-D array-like, shape `(d_in, d_out)`. Column `j` holds the weights of output unit `j`.
  Rows index inputs and columns index outputs. PyTorch stores `nn.Linear.weight` as
  `(d_out, d_in)`; to use such a weight, pass its transpose `W_torch.T`.
- `b`: 1-D array-like, shape exactly `(d_out,)`. A `(1, d_out)` or `(d_out, 1)` bias is rejected,
  not broadcast.
- Any dimension may be 0. `n = 0` gives `(0, d_out)`. `d_in = 0` gives the bias repeated on every
  row, because an empty sum is 0.
- Returns: `float64` array of shape `(n, d_out)`. Integer inputs are converted to `float64`.
- Raises `ValueError` when: `X` is not 2-D; `W` is not 2-D; `b` is not 1-D; `X.shape[1] != W.shape[0]`
  (feature mismatch); or `b.shape[0] != W.shape[1]`.
- Inputs are numeric and real. Arithmetic cases are limited to finite inputs with representable
  intermediate products, partial sums and outputs; nonnumeric values and overflow behavior are
  outside the contract.
- Must not mutate inputs.

## Examples

```python
>>> dense_forward(np.array([[1.0, 2.0]]),
...               np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]]),
...               np.array([0.5, 0.0, -1.0]))
array([[1.5, 2. , 7. ]])
>>> dense_forward(np.array([[1.0, 2.0], [3.0, 4.0]]), np.eye(2), np.zeros(2))
array([[1., 2.],
       [3., 4.]])
>>> dense_forward(np.zeros((2, 3)), np.zeros((4, 2)), np.zeros(2))
ValueError: feature mismatch: X has 3 features but W has 4 input rows
```

## Constraints

- Allowed: NumPy, including `@`, `np.dot`, broadcasting and `np.asarray`.
- Not allowed: `scipy` and `sklearn`. They are the test oracles, not the tool.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They cover: agreement with an explicit
per-entry reference on random batches of several shapes (single row, square, wide, tall, many units);
the weight layout on square and non-square `W`, including a batch whose size equals `d_out`; the bias
added once per output unit to every row; every `ValueError` rule (non-2-D `X` or `W`, row- or
column-shaped bias, feature mismatch in the PyTorch layout); the empty batch and zero-width features;
integer input returning `float64`; large and small magnitudes with float64 precision; each row's
output depending only on that row; and inputs left unchanged.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer` and `digits` (see
`docs/DATASETS.md`).
