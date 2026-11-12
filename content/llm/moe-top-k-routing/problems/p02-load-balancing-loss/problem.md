# Switch-style load-balancing loss from router logits

A router that always prefers one expert wastes the others: that expert is overloaded, the rest
sit idle, and the layer's capacity goes unused. The auxiliary load-balancing loss adds a penalty
that is small when tokens and router probability are spread evenly over the experts and large when
they pile onto a few. This function computes that penalty from the router logits of one batch.

## Contract

- `load_balance_loss(logits, alpha=1.0)` with:
  - `logits`: `torch.float64` tensor of shape `(T, E)`, `T >= 1` tokens and `E >= 1` experts. Every
    value is finite. Otherwise `ValueError` (wrong rank, wrong dtype, a NaN or infinite value,
    `T = 0`, or `E = 0`).
  - `alpha`: a non-negative finite `int` or `float`, the loss coefficient. `bool` and other types
    raise `TypeError`. A negative or non-finite value raises `ValueError`.
- Definitions. Let `p[t, i]` be the softmax over the experts `i` of the logits of token `t`, and let
  `a_t` be the top-1 expert of token `t`: the expert with the largest logit. Ties go to the **lower
  expert index**.
  - Router probability per expert: `P_i = (1 / T) * sum over t of p[t, i]`.
  - Dispatch fraction per expert: `f_i = (1 / T) * sum over t of 1[a_t = i]`, the fraction of
    tokens whose top-1 expert is `i`.
  - Loss: `loss = alpha * E * sum over i of f_i * P_i`.
- Returns a **0-dimensional** `torch.float64` tensor (shape `()`), not a Python float.
- Gradients flow through `P` only. `f` is a count, so it is a constant with respect to the logits.
  Do not detach `P`, and do not try to differentiate the top-1 choice.
- With a perfectly balanced router, `f_i = P_i = 1/E` for every `i`, so the loss is exactly `alpha`.
- Must not mutate `logits`.

## Examples

```python
>>> logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
>>> load_balance_loss(logits)            # token 0 -> expert 0, token 1 -> expert 1
tensor(1., dtype=torch.float64)          # f = [0.5, 0.5], P = [0.5, 0.5], so 2 * 0.5 = 1
```

```python
>>> logits = torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
>>> load_balance_loss(logits).item()
1.0846215728839737
# f = [2/3, 1/3]; P = [0.62693, 0.37307]; loss = 2 * (2/3 * 0.62693 + 1/3 * 0.37307)
```

```python
>>> logits = torch.tensor([[0.0, 0.0], [5.0, -5.0]], dtype=torch.float64)
>>> load_balance_loss(logits).item()    # the tie in row 0 goes to expert 0
1.4999546021312975                       # f = [1, 0]; a tie at expert 1 would give 1.0
```

## Constraints

- Allowed: the `torch` package for tensor operations, for example `torch.softmax`, `torch.sort`
  (with `stable=True`) and `torch.bincount`. Also the `math` module.
- Not allowed: `torch.nn` in any form, including `torch.nn.functional`. The loss is a few lines of
  tensor arithmetic and must be written out in full.

## Done looks like

All public tests pass. Submit then runs the hidden tests. They cover agreement with a pure-Python
reference on seeded random logits of several shapes and coefficients; a finite-difference check of
the gradient with respect to the logits; the balanced case, which must score `alpha` exactly; a
fully collapsed router, which must score `alpha * E`; the tie rule on a case where it changes the
value; a single expert; large magnitudes where a naive exp overflows; the scaling in `alpha`; the
0-dimensional float64 output; the error rules for `T`, `E`, `alpha` and the logits dtype and shape;
and inputs left unchanged.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).
