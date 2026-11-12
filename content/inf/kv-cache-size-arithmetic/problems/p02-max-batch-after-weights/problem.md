# Largest batch that fits after the weights

Implement `max_batch_size(config, seq_len, dtype, weight_bytes, memory_bytes, *, reserve_bytes=0)`.
`reserve_bytes` is keyword-only. It returns
how many sequences of `seq_len` tokens can hold their KV cache on one device at the same time, after the model
weights and a reserve have been placed.

Per sequence, the cache costs what the KV-cache formula gives for one sequence. The config keys, the window
rule and the bytes per element are the same as in the KV-cache bytes problem:

- `config` is a `dict` with `num_layers`, `num_kv_heads` and `head_dim` (required), and an optional
  `sliding_window` (`None` or missing means full attention). Other keys are ignored.
- The per-sequence bytes are `2 * num_layers * num_kv_heads * head_dim * min(seq_len, window) * bytes_per_element`,
  with `window = sliding_window`, or `seq_len` when there is no window.
- `dtype` is `"fp32"` (4 bytes), `"fp16"` (2), `"bf16"` (2) or `"fp8"` (1).

The budget is `memory_bytes - weight_bytes - reserve_bytes`. A batch of `B` sequences fits when
`weight_bytes + reserve_bytes + B * per_sequence_bytes <= memory_bytes`. The answer is the largest such `B`.

## Contract

- `config`, `dtype`: as above. Validation is the same as the KV-cache bytes problem: TypeError for values
  that are not Python `int` (`bool`, `float`, `str` and `None` are rejected, except `sliding_window` may be
  `None`); ValueError for a missing required key, a value below its minimum, or an unknown dtype.
- `seq_len`: a Python `int` of at least 1. Zero is a ValueError, because an empty sequence has no bound.
- `weight_bytes`, `memory_bytes`, `reserve_bytes`: Python `int` values of at least 0. `reserve_bytes` defaults to 0.
- Returns: a Python `int` of at least 0. Never negative.
- Returns 0 when the weights and the reserve alone do not fit in `memory_bytes`, and also when the budget
  holds less than one sequence.
- The answer is a floor. A partial sequence does not count.
- Must not mutate `config`.

## Examples

```python
>>> cfg = {"num_layers": 32, "num_kv_heads": 32, "head_dim": 128}
>>> max_batch_size(cfg, 4096, "fp16", 14_000_000_000, 80 * 2**30)
33       # budget 71,899,345,920 bytes; one sequence is 2 GiB; 33 * 2 GiB fits, 34 does not
>>> max_batch_size(cfg, 4096, "fp16", 14_000_000_000, 16 * 2**30)
1        # budget 3,179,869,184 bytes: one sequence fits, two do not
>>> max_batch_size(cfg, 4096, "fp16", 14_000_000_000, 10 * 2**30)
0        # the weights alone are larger than the device memory
>>> gqa = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "sliding_window": 4096}
>>> max_batch_size(gqa, 32768, "fp16", 14_000_000_000, 80 * 2**30)
133      # the window stores 4096 tokens: one sequence is 512 MiB
>>> max_batch_size(cfg, 4096, "fp16", 14_000_000_000, 80 * 2**30, reserve_bytes=2 * 2**30)
32       # the reserve takes one sequence's worth of budget
>>> max_batch_size(cfg, 0, "fp16", 14_000_000_000, 80 * 2**30)
ValueError  # seq_len must be at least 1
```

## Constraints

- Allowed: built-in types and the standard library.
- Not allowed: `numpy`, `scipy`, `torch` and any other numeric library. Use integer arithmetic only.

## Done looks like

The public tests check the examples and the empty-budget case. **Submit** adds hidden tests that cover: a budget
that fits exactly, and one byte short of fitting; weights larger than memory; a reserve; sequences longer than
the window; a window longer than the sequence; zero-byte weights; very large budgets that stay exact beyond float
precision; the return type; each validation rule; and random configurations compared against an independent
search over batch sizes.

Some hidden tests take their inputs from the real datasets `gutenberg_frankenstein`,
`gutenberg_pride_prejudice` and `gutenberg_walden` (see `docs/DATASETS.md`).
