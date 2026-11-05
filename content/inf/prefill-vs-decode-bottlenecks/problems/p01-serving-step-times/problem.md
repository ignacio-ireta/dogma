# Prefill and decode step times

Implement `serving_step_times(params, config, prompt_len, context_len, batch, element_bytes, *, peak_flops, bandwidth)`.
It estimates two serving phases for a dense decoder on one device, with the first-order model below. It returns
the time to prefill `batch` prompts, the time of one decode step for the batch, and the decode throughput.

**The model (use it exactly).**

Every time is the larger of a compute floor and a memory floor, as in the roofline topic.

- Weight bytes, read once per forward pass: `W = params * element_bytes`.
- KV bytes per token per sequence: `kv = 2 * num_layers * num_kv_heads * head_dim * element_bytes`. The factor 2 is K and V.
- Prefill of `batch` prompts of `prompt_len` tokens, processed in one parallel pass:
  - FLOPs `F = 2 * params * prompt_len * batch`
  - Bytes `Q = W + batch * prompt_len * kv` (weights once, and the K and V of every prompt token written)
  - `prefill_s = max(F / peak_flops, Q / bandwidth)`
- One decode step, which emits one token for each of `batch` sequences that each already hold `context_len` tokens:
  - FLOPs `F = 2 * params * batch`
  - Bytes `Q = W + batch * (context_len + 1) * kv` (weights once, and the K and V of the cached tokens plus the new one, read for every sequence). The write of the new token is not counted.
  - `decode_step_s = max(F / peak_flops, Q / bandwidth)`
- Decode throughput, in tokens per second over the whole batch: `decode_tokens_per_s = batch / decode_step_s`.

Ignored, on purpose: embedding lookups, attention score FLOPs, norms, and a `sliding_window` key. The attention
score term is a small correction at these context lengths; a window is not modelled, so treat attention as full.

## Contract

- `params`: Python `int`, at least 1. Parameters multiplied by a matrix in every token's pass.
- `config`: `dict` with the Python `int` keys `num_layers`, `num_kv_heads` and `head_dim`, each at least 1. Other keys,
  including `sliding_window`, are ignored. Must not be mutated.
- `prompt_len`: Python `int`, at least 1.
- `context_len`: Python `int`, at least 0. Tokens already cached per sequence before the decode step.
- `batch`: Python `int`, at least 1.
- `element_bytes`: Python `int`, at least 1. Weights and cache use the same width.
- `peak_flops`, `bandwidth`: keyword-only Python `int`, at least 1, in FLOP/s and bytes/s.
- Returns a `dict` with exactly the keys `"prefill_s"`, `"decode_step_s"` and `"decode_tokens_per_s"`, all `float`.
  Times are in seconds. No rounding.
- Raises `TypeError` when a value is not a Python `int` (`bool`, `float`, `str` and `None` are rejected) or `config`
  is not a `dict`. Raises `ValueError` when a value is below its minimum or a required key is missing.

## Examples

Device `peak_flops = 312_000_000_000_000` and `bandwidth = 2_000_000_000_000`. Model: 7e9 parameters, fp16 weights
and cache, and `config = {"num_layers": 32, "num_kv_heads": 32, "head_dim": 128}`, so one token of cache is 524,288 bytes.

```python
>>> cfg = {"num_layers": 32, "num_kv_heads": 32, "head_dim": 128}
>>> t = serving_step_times(7_000_000_000, cfg, 512, 512, 1, 2, peak_flops=PEAK, bandwidth=BW)
>>> round(t["prefill_s"], 5)          # 2 * 7e9 * 512 FLOPs over peak: compute-bound
0.02297
```

```python
>>> round(serving_step_times(7_000_000_000, cfg, 1, 0, 1, 2, peak_flops=PEAK, bandwidth=BW)["prefill_s"], 6)
0.007       # one prompt token: the weight read dominates, so the prefill is memory-bound
```

```python
>>> t = serving_step_times(7_000_000_000, cfg, 512, 512, 1, 2, peak_flops=PEAK, bandwidth=BW)
>>> round(t["decode_step_s"], 6), round(t["decode_tokens_per_s"], 2)
(0.007134, 140.16)  # batch 1: the weights stream from memory once per token
```

```python
>>> t = serving_step_times(7_000_000_000, cfg, 512, 512, 8, 2, peak_flops=PEAK, bandwidth=BW)
>>> round(t["decode_step_s"], 6), round(t["decode_tokens_per_s"], 1)
(0.008076, 990.6)   # batch 8: the weights are shared, so throughput rises about sevenfold
```

```python
>>> serving_step_times(7_000_000_000, cfg, 0, 512, 1, 2, peak_flops=PEAK, bandwidth=BW)
ValueError          # prompt_len must be at least 1
```

## Constraints

- Allowed: built-in types and the standard library.
- Not allowed: `numpy`, `scipy`, `torch` and any helper that computes the roofline or the step times for you. Use
  Python integers for the counts; divide to floats only at the end.

## Done looks like

The public tests check the four examples and the empty-prompt case. **Submit** adds hidden tests. They compare the
result with an exact rational computation over seeded random models, shapes and devices, to a relative tolerance of
1e-9. They also cover: decode bytes that do not scale the weights with the batch; the KV factor of 2; the context plus
the new token; prefill FLOPs that scale with both prompt length and batch; decode throughput that never falls as the
batch grows; a decode step that costs less than the batch times one step; a compute-bound and a memory-bound regime
for each phase; configs that must not change; the exact key set and float types; and each validation rule.
