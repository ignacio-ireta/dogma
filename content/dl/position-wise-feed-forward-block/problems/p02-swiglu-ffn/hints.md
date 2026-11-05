## Hint 1 — concept

The block has two parallel expansions of the same input. Only one of them is activated, and the
activated path multiplies the other one. The result is then projected back. Each position is
computed on its own, and there are no biases.

## Hint 2 — formula / approach

Compute `gate = X @ W.T` and `up = X @ V.T`. Apply SiLU to the gate only: `gate * sigmoid(gate)`.
Multiply by `up` elementwise. Project back with `hidden @ W2.T`. Check first that `W` and `V` have
the same shape, and that `W2` is `(d_model, d_ff)`.

## Hint 3 — code shape

Check `X.ndim`, then `W.ndim` and `V.ndim`, then `W.shape == V.shape`, then `X.shape[-1]` against
`W.shape[1]`, then `W2.shape`. Use `torch.sigmoid` for the sigmoid. Do not call `torch.nn` helpers.
