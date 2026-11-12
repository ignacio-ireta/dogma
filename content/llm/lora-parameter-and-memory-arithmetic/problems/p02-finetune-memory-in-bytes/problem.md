# Fine-tuning memory in bytes: frozen base plus LoRA adapter

Implement `finetune_memory_bytes(*, base_params, base_bits, lora_params, adapter_bits)`. It returns the bytes
needed to hold the state of a LoRA fine-tuning run, split into terms, as an exact integer count of bytes.

The state has four parts:

- **Frozen base weights.** The base model is stored in `base_bits` bits per parameter (for example 16 for
  bf16, or 4 for a 4-bit quantized base). It receives no gradient and has no optimizer state.
- **Adapter weights.** The `lora_params` trainable numbers, stored in `adapter_bits` bits each (16 or 32).
- **Adapter gradients.** One gradient per adapter number, stored in the same `adapter_bits` as the weights.
- **Adam states.** Two fp32 optimizer tensors per adapter number, the first and second moments, four bytes
  each. Optimizer states stay in fp32 whatever the adapter dtype is.

Activations, the context cache, framework overhead and memory fragmentation are not counted. They depend on
batch size and sequence length, and this function does not estimate them.

Bit counts that are not a multiple of eight leave a fraction of a byte. Round each stored tensor **up** to whole
bytes, after multiplying its number count by its bit width: `ceil(count * bits / 8)`.

## Contract

- `base_params`: `int` (not `bool`), `>= 0`.
- `base_bits`: `int` (not `bool`), one of `4`, `8`, `16`, `32`.
- `lora_params`: `int` (not `bool`), `>= 0`.
- `adapter_bits`: `int` (not `bool`), one of `16`, `32`.
- Returns a `dict[str, int]` with these keys, in this order:
  - `"base_weights"`: `ceil(base_params * base_bits / 8)`.
  - `"adapter_weights"`: `ceil(lora_params * adapter_bits / 8)`.
  - `"adapter_gradients"`: the same value as `"adapter_weights"`.
  - `"adam_states"`: `8 * lora_params`, two fp32 tensors of four bytes each.
  - `"total"`: the exact sum of the four entries above.
- Raises `ValueError` when: an argument is not an `int`, is a `bool`, or is out of its allowed range or set.
- Integers are exact. Do not use floats, and do not round by division. Use integer arithmetic for `ceil`.

## Examples

```python
>>> finetune_memory_bytes(base_params=1_000_000, base_bits=16, lora_params=1_000, adapter_bits=16)
{'base_weights': 2000000, 'adapter_weights': 2000, 'adapter_gradients': 2000,
 'adam_states': 8000, 'total': 2012000}
```

```python
>>> finetune_memory_bytes(base_params=3, base_bits=4, lora_params=0, adapter_bits=16)
{'base_weights': 2, 'adapter_weights': 0, 'adapter_gradients': 0, 'adam_states': 0, 'total': 2}
# 3 numbers at 4 bits is 12 bits, which is 1.5 bytes, rounded up to 2.
```

```python
>>> finetune_memory_bytes(base_params=10, base_bits=8, lora_params=10, adapter_bits=8)
ValueError: adapter_bits must be 16 or 32
```

## Constraints

- Allowed: Python integers, dictionaries, and integer operators. Floats are not needed.
- Not allowed: `numpy`, `torch`, `jax` and `sklearn`. Do not estimate activations.

## Done looks like

All public tests pass. **Submit** then runs the hidden tests. They check the four terms against a plain-Python
reference on seeded configurations. They cover the 4-bit rounding, where the bit count is not a multiple of
eight; a zero adapter; a zero base; the adam term being fp32 whatever the adapter dtype; the gradient term
equal to the weight term; the total as the exact sum; a realistic configuration, where every term must stay an
exact integer; the key order; and every `ValueError` rule, including the allowed sets and `bool` arguments.

Some hidden tests take their inputs from the real datasets `breast_cancer`, `diabetes`, `digits`,
`iris` and `wine` (see `docs/DATASETS.md`).
