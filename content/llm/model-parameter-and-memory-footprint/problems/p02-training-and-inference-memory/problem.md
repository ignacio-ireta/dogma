# Training and inference memory in bytes

Implement `estimate_memory(num_params, *, mode, dtype, optimizer, master_weights, activation_bytes)`.
It returns the bytes needed to hold a model's state, split into terms, for two uses:

- **Inference:** only the weights are held.
- **Training:** the weights, one gradient per weight, the optimizer's state, and optionally an fp32
  master copy of the weights when the working dtype is lower precision.

Activations, the intermediate tensors of a forward and backward pass, depend on batch size and
sequence length. The function takes them as a separate number, `activation_bytes`, and adds it to the
total. It does not estimate them.

## Contract

- `num_params`: `int` (not `bool`) and at least 0. The number of trainable scalars.
- `mode`: `"inference"` or `"training"`.
- `dtype`: the working dtype of the weights and gradients. One of:

  | name | bytes per number |
  |---|---|
  | `"fp32"` | 4 |
  | `"fp16"`, `"bf16"` | 2 |
  | `"fp8"`, `"int8"` | 1 |

- `optimizer`: one of `"none"`, `"sgd"`, `"momentum"`, `"adam"`, `"adamw"`. Optimizer states are
  always stored in fp32, four bytes per number, regardless of `dtype`. The number of states per
  parameter is: `sgd` 0, `momentum` 1, `adam` 2, `adamw` 2.
- `master_weights`: `bool`. In training with a `dtype` other than `"fp32"`, `True` keeps an fp32
  copy of every weight, four bytes per parameter, held by the optimizer. With `"fp32"`, a master copy
  adds nothing to store, so `master_weights=True` with `"fp32"` raises `ValueError`.
- `activation_bytes`: `int` (not `bool`) and at least 0. Added as its own term.
- Rules that make the modes consistent:
  - `mode="inference"` requires `optimizer="none"` and `master_weights=False`.
  - `mode="training"` requires `optimizer` other than `"none"`.
- Returns a `dict[str, int]` with these keys, in this order:
  - `"weights"`: `num_params * bytes(dtype)`.
  - `"gradients"`: training only, `num_params * bytes(dtype)`. Zero for inference.
  - `"master_weights"`: training, `master_weights` true and `dtype` not `"fp32"`, `4 * num_params`. Else zero.
  - `"optimizer_states"`: training only, `4 * states * num_params`, where `states` comes from the table
    above. Zero for inference.
  - `"activations"`: `activation_bytes`.
  - `"total"`: the exact sum of the other five entries.
- Raises `ValueError` when: any argument has the wrong type or range; an unknown name is given for
  `mode`, `dtype` or `optimizer`; a rule above is violated.
- Integers are exact Python integers, so large parameter counts must not lose precision.

## Examples

```python
>>> estimate_memory(1_000_000, mode="inference", dtype="fp32", optimizer="none",
...                 master_weights=False, activation_bytes=0)
{'weights': 4000000, 'gradients': 0, 'master_weights': 0, 'optimizer_states': 0,
 'activations': 0, 'total': 4000000}
```

```python
>>> # Mixed-precision Adam: bf16 weights and gradients, fp32 master copy, two fp32 states.
>>> estimate_memory(1_000_000, mode="training", dtype="bf16", optimizer="adam",
...                 master_weights=True, activation_bytes=0)["total"]
16000000
```

```python
>>> estimate_memory(10, mode="inference", dtype="fp32", optimizer="adam",
...                 master_weights=False, activation_bytes=0)
ValueError: inference holds no optimizer state: use optimizer="none"
```

## Constraints

- Allowed: the Python standard library.
- Not allowed: `numpy`, `torch`, `jax`, `sklearn`. Write the byte table from the rules above. Do not
  allocate tensors or query a device.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They check: a hand-worked table of
bytes per parameter for every training and inference combination that the contract allows; the
16-byte rule for mixed-precision Adam and for fp32 Adam; a 7-billion-parameter example with exact
integer totals; the zero-state optimizers; the activation term added as given and not scaled; the
rule that master copies need a lower-precision dtype; each invalid argument; and exact integers at
large sizes. A wrong variant gets a named message, for example an optimizer state stored in the
working dtype, a missing gradient term, or a master copy counted for fp32 weights.
