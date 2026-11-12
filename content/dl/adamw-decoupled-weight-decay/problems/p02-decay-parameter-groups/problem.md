# Decay and no-decay parameter groups

Implement `build_adamw_groups(named_params, weight_decay)`. It takes a list of `(name, tensor)`
pairs, such as `model.named_parameters()` gives, and returns AdamW parameter groups. Weight
decay should apply to weight matrices and convolution kernels. It should not apply to biases or
one-dimensional norm gains, which is common practice in transformer training. The groups are
the input to `torch.optim.AdamW`.

## Contract

- `named_params`: a non-empty list of `(name, tensor)` pairs. Names are unique strings. Values are
  `torch.Tensor` objects.
- A tensor is **decayed** when it has at least two dimensions and its final dotted name component
  is not `"bias"`. Every other tensor is **undecayed**. A two-dimensional tensor named `...bias` is
  undecayed. Classification uses only the tensor's rank and name, never the module type:
  - an embedding table such as `embed.weight` is two-dimensional, so it **is** decayed;
  - a one-dimensional norm gain such as `norm.weight` is **not** decayed, whatever its name;
  - only the final component counts: `attn.bias_scale.weight` and `head.lm_bias` are two-dimensional
    and not named `bias`, so they **are** decayed.
- `weight_decay`: finite and `>= 0`.
- Return a list of dicts:
  - the decayed group first, `{"params": [...], "weight_decay": weight_decay}`
  - then the undecayed group, `{"params": [...], "weight_decay": 0.0}`
  - omit a group that would be empty
- Within each group, keep the original order. Place the caller's tensor objects in the groups; do
  not copy them.
- Raises `ValueError` for an empty list, duplicate names, a value that is not a tensor, or a
  `weight_decay` that is negative or not finite.
- Must not mutate the input list or its tensors.

## Examples

```python
groups = build_adamw_groups(
    [("linear.weight", w), ("linear.bias", b), ("norm.weight", g)], weight_decay=0.1
)
# groups[0] -> {"params": [w], "weight_decay": 0.1}
# groups[1] -> {"params": [b, g], "weight_decay": 0.0}
```

A model whose parameters are all decayed gives one group:

```python
groups = build_adamw_groups([("embed.weight", e)], weight_decay=0.05)
# groups -> [{"params": [e], "weight_decay": 0.05}]
```

## Constraints

Allowed: Python, `torch` tensor objects and attributes, and `math`. Not allowed: `torch.optim`, which
contains the optimizer you are checking against. Do not call the optimizer; only build the groups.

## Done looks like

The public tests pass for the basic split, a single group and an empty input. Hidden tests check
membership by name and rank, including a two-dimensional bias. They check the exact weight-decay
values, identity of tensors, order within groups, a run of `torch.optim.AdamW` with your groups
against a run with reference groups, and invalid input. A failing test names the mistake, such as
a norm gain that received decay or a group built from copies.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).
