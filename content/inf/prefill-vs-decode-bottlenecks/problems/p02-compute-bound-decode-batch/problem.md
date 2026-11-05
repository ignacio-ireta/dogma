# Batch size at which decode becomes compute-bound

Implement `compute_bound_decode_batch(params, config, context_len, element_bytes, *, peak_flops, bandwidth)`. It returns
the smallest batch size at which one decode step of the model in `serving_step_times` (the previous problem) is
compute-bound, or `0` when no batch size makes it compute-bound.

**The model (the same as the previous problem).**

For a batch of `B` sequences, each with `context_len` cached tokens:

- Weight bytes `W = params * element_bytes`.
- KV bytes per token per sequence `kv = 2 * num_layers * num_kv_heads * head_dim * element_bytes`.
- FLOPs of one decode step `F = 2 * params * B`.
- Bytes of one decode step `Q = W + B * (context_len + 1) * kv`.
- The step is compute-bound when `F / Q >= peak_flops / bandwidth`, that is, when its intensity reaches the ridge.
  A tie counts as compute-bound. Decide it exactly with integers: `F * bandwidth >= peak_flops * Q`.

Return the smallest `B >= 1` that satisfies the rule. Because intensity rises with `B` only while the weight read is
shared, this is a threshold: if one batch size is compute-bound, every larger batch size is too. Return `0` if no
batch size is compute-bound. That happens when the KV traffic per extra sequence grows faster than the FLOPs.

## Contract

- `params`: Python `int`, at least 1.
- `config`: `dict` with the Python `int` keys `num_layers`, `num_kv_heads` and `head_dim`, each at least 1. Other keys
  are ignored, including `sliding_window`. Must not be mutated.
- `context_len`: Python `int`, at least 0.
- `element_bytes`: Python `int`, at least 1.
- `peak_flops`, `bandwidth`: keyword-only Python `int`, at least 1.
- Returns a Python `int`, never negative, and never a float.
- Raises `TypeError` when a value is not a Python `int` (`bool`, `float`, `str` and `None` are rejected) or `config`
  is not a `dict`. Raises `ValueError` when a value is below its minimum or a required key is missing.

## Examples

```python
>>> PEAK, BW = 312_000_000_000_000, 2_000_000_000_000        # ridge 156 FLOP/byte
>>> GQA = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128}
>>> compute_bound_decode_batch(7_000_000_000, GQA, 512, 2, peak_flops=PEAK, bandwidth=BW)
623     # 623 sequences of 513 tokens: the shared weight read is amortised enough to reach the ridge
```

```python
>>> MHA = {"num_layers": 32, "num_kv_heads": 32, "head_dim": 128}
>>> compute_bound_decode_batch(7_000_000_000, MHA, 512, 2, peak_flops=PEAK, bandwidth=BW)
0       # the cache traffic grows with the batch, and the intensity stays near 52 FLOP/byte
```

```python
>>> tiny = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}
>>> compute_bound_decode_batch(1_000_000, tiny, 0, 1, peak_flops=1, bandwidth=1)
1       # already compute-bound at batch 1
```

```python
>>> compute_bound_decode_batch(7_000_000_000, GQA, -1, 2, peak_flops=PEAK, bandwidth=BW)
ValueError      # context_len must be at least 0
```

## Constraints

- Allowed: built-in types and the standard library.
- Not allowed: `numpy`, `scipy`, `torch`, and any helper that searches for the answer or computes the roofline for you.
  Use Python integers so the threshold is exact.

**Why integers.** The threshold is a ratio of large byte and FLOP counts. A float division can round a value such as
`2**70 + 0.5` down to `2**70`, which is one batch short. Take the ceiling with integer arithmetic, for example
`-(-numerator // denominator)`, and never call `math.ceil` on a float quotient.

## Done looks like

The public tests check the examples above. **Submit** adds hidden tests. They compare your answer with an
independent search over batch sizes on seeded random models and devices. They also cover: the smallest batch and not
the one before it; a tie at the ridge counting as compute-bound; a model that is compute-bound from batch 1; a model
whose KV traffic keeps it memory-bound at every batch size; a threshold that a floating-point division would round
down, which must come out exact; the return type; configs that must not change; and each validation rule.
