# LoRA trainable parameter budget

Implement `lora_budget(*, n_layers, d_model, d_ff, rank, targets, base_params)`. It returns the number of
trainable LoRA parameters for a decoder-only transformer, and the fraction they are of the frozen base.

Each adapted matrix is a linear map with input width `d_in` and output width `d_out`. Its LoRA adapter has
`A` of shape `(rank, d_in)` and `B` of shape `(d_out, rank)`, so it holds

$$
\text{params}(\text{matrix}) = \text{rank} \cdot (d_\text{in} + d_\text{out}).
$$

The adapted matrices are named by `targets`. Every layer has the same set:

| target | role | `d_in` | `d_out` |
|---|---|---|---|
| `q`, `k`, `v`, `o` | attention projections | `d_model` | `d_model` |
| `gate`, `up` | MLP input projections | `d_model` | `d_ff` |
| `down` | MLP output projection | `d_ff` | `d_model` |

The attention projections are `d_model` by `d_model` because this topic counts multi-head attention with
full key and value width. Grouped-query attention would make `k` and `v` smaller and is not counted here.

## Contract

- `n_layers`: `int` (not `bool`), `>= 1`. Number of transformer blocks, each with the same targets.
- `d_model`: `int` (not `bool`), `>= 1`.
- `d_ff`: `int` (not `bool`), `>= 1`, or `None`. Required when `targets` contains `gate`, `up` or `down`.
  When no MLP target is present, `d_ff` is not read or checked: it may be `None` or any value, including an invalid one.
- `rank`: `int` (not `bool`), `>= 1`.
- `targets`: a `list` or `tuple` of one or more distinct names from `q`, `k`, `v`, `o`, `gate`, `up`, `down`.
  Order does not matter.
- `base_params`: `int` (not `bool`), `>= 1`. The total parameter count of the frozen base model, which the
  caller supplies.
- Returns a tuple `(trainable, fraction)`:
  - `trainable`: `int`, exact. `n_layers` times the sum of `rank * (d_in + d_out)` over the targets.
  - `fraction`: `float`, `trainable / base_params`.
- Raises `ValueError` when: any integer argument is not an `int`, is a `bool`, or is out of range; `targets`
  is empty, is not a list or tuple, has a name outside the list above, or repeats a name; an MLP target is
  given and `d_ff` is `None`.
- Integers are exact Python integers. Only `fraction` is a float.
- Must not mutate `targets`.

## Examples

```python
>>> lora_budget(n_layers=2, d_model=8, d_ff=None, rank=2, targets=("q", "v"), base_params=1000)
(128, 0.128)
# Each q or v matrix: 2 * (8 + 8) = 32. Two matrices per layer: 64. Two layers: 128. 128 / 1000 = 0.128.
```

```python
>>> lora_budget(n_layers=3, d_model=8, d_ff=32, rank=1,
...             targets=("q", "k", "v", "o", "gate", "up", "down"), base_params=10_000)
(552, 0.0552)
# Attention: 4 * 1 * (8 + 8) = 64. gate, up and down: 1 * (8 + 32) = 40 each, 120 in total.
# Per layer: 184. Three layers: 552.
```

```python
>>> lora_budget(n_layers=1, d_model=8, d_ff=None, rank=2, targets=("up",), base_params=100)
ValueError: an MLP target needs d_ff
```

## Constraints

- Allowed: Python integers, lists, tuples, sets and dictionaries, the `math` module, and plain division.
- Not allowed: `numpy`, `torch`, `jax` and `sklearn`. The count is arithmetic. Write it out.

## Done looks like

All public tests pass. **Submit** then runs the hidden tests. They check the count against a plain-Python
reference that sums per matrix, on seeded random configurations, and against hand-computed budgets for a
seven-target configuration. They cover: one target and all seven targets; a rank of one and a rank above
the width; the factor `n_layers`; the `d_in + d_out` sum (not the product); the MLP shapes with `d_ff` both
larger and smaller than `d_model`; a configuration of realistic size, where the count must stay exact;
`fraction` close to `trainable / base_params`; the order of `targets` not mattering; duplicate and unknown
target names; `bool` and non-integer arguments; a missing `d_ff`; `targets` left unchanged.

Some hidden tests take their inputs from the real datasets `breast_cancer`, `diabetes`, `digits`,
`iris` and `wine` (see `docs/DATASETS.md`).
