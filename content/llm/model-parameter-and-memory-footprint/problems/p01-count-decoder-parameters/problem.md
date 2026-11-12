# Count the parameters of a decoder-only transformer

Implement `count_parameters(config)`. It returns the exact number of trainable scalars in a
decoder-only transformer, computed from its configuration alone. Do not build any tensor. The
count has these parts, and each one is either present or absent according to the config:

- **Token embedding:** a `(V, D)` table.
- **Position table:** a learned `(C, D)` table, present only when `learned_positions` is true.
  A rotary model has no table.
- **Per block, attention:** four `(D, D)` projections for query, key, value and output. There is
  no grouped-query sharing: every projection is full width. With `bias` true, add four `(D,)`
  bias vectors.
- **Per block, MLP:** a `gelu` MLP has two matrices, `(D, F)` and `(F, D)`. A `swiglu` MLP has
  three matrices, `(D, F)` gate, `(D, F)` up and `(F, D)` down. With `bias` true, add one bias per
  output: `F` for each expansion matrix and `D` for the down projection.
- **Per block, norms:** two norms. A `layernorm` has a scale and a shift of `(D,)` each. An
  `rmsnorm` has a scale only, `(D,)`.
- **Final norm:** one more norm of the same kind, applied after the last block.
- **LM head:** a `(D, V)` matrix, present only when `tie_embeddings` is false. It has no bias.
  When the head is tied, it adds no parameters, because it reuses the token embedding.

Attention heads do not change the count, since they only split the `D` features. `n_heads` must
still divide `D`.

## Contract

- `config`: a `dict` with exactly these keys and no others:

  | key | type | meaning |
  |---|---|---|
  | `vocab_size` | `int` | `V`, at least 1 |
  | `d_model` | `int` | `D`, at least 1 |
  | `n_layers` | `int` | `N`, at least 0 |
  | `n_heads` | `int` | `H`, at least 1, and must divide `D` |
  | `d_ff` | `int` | `F`, the MLP expansion width, at least 1 |
  | `context_length` | `int` | `C`, at least 1, the length of the position table |
  | `learned_positions` | `bool` | whether the position table exists |
  | `bias` | `bool` | whether linear layers carry bias vectors |
  | `mlp` | `str` | `"gelu"` or `"swiglu"` |
  | `norm` | `str` | `"layernorm"` or `"rmsnorm"` |
  | `tie_embeddings` | `bool` | whether the LM head reuses the token embedding |

- Integers must be real `int` values, not `bool`, not floats, and not NumPy scalars.
- Returns: a `dict[str, int]` with these keys, in this order: `"token_embedding"`,
  `"position_embedding"`, `"attention"`, `"mlp"`, `"norms"`, `"final_norm"`, `"lm_head"`, `"total"`.
  Each component counts **all** layers, so `attention` is `N` times one block's attention. `total`
  is the exact sum of the other seven entries.
- Raises `ValueError` when: `config` is not a `dict`; a key is missing or unexpected; a value has the
  wrong type or range; `n_heads` does not divide `d_model`; `mlp` or `norm` is not one of the allowed
  strings.
- Must not mutate `config`. Python integers are unbounded, so large configs must give exact results.

## Examples

```python
>>> cfg = {"vocab_size": 10, "d_model": 4, "n_layers": 1, "n_heads": 2, "d_ff": 8,
...        "context_length": 6, "learned_positions": True, "bias": False, "mlp": "gelu",
...        "norm": "layernorm", "tie_embeddings": True}
>>> count_parameters(cfg)["total"]
# token 40, position 24, attention 64, mlp 64, norms 16, final norm 8, head 0 (tied): 216
216
```

```python
>>> # GPT-2 small: V=50257, D=768, N=12, H=12, F=3072, C=1024, learned positions, bias,
>>> # gelu, layernorm, tied head.
>>> count_parameters(gpt2_small)["total"]
124439808
```

```python
>>> # LLaMA-7B-style: V=32000, D=4096, N=32, H=32, F=11008, no bias, swiglu, rmsnorm, untied,
>>> # no learned positions (C is still required, but unused).
>>> count_parameters(llama_7b)["total"]
6738415616
```

```python
>>> count_parameters({**cfg, "mlp": "relu"})
ValueError: mlp must be "gelu" or "swiglu"
```

## Constraints

- Allowed: the Python standard library, for example `math`.
- Not allowed: `numpy`, `torch`, `jax`, `sklearn`. Write the count from the formulas. Do not allocate
  any tensor, and do not call a framework to count parameters.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They check: the GPT-2 small and
LLaMA-7B-style totals; an independent reference that enumerates every tensor shape and multiplies
it out, on seeded random configs; each component when one feature is switched on or off, for example
tying, bias, learned positions and the gated MLP; zero layers; large configs with exact integer
results; the type and range rules; the `ValueError` cases; and the config left unchanged. A wrong
variant gets a named message, for example counting the head when it is tied, or counting two
matrices for a gated MLP.

Some hidden tests take their inputs from the real dataset `gutenberg_frankenstein` (see
`docs/DATASETS.md`).
