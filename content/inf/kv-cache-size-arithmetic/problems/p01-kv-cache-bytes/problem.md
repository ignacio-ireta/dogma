# KV-cache bytes from a config

Implement `kv_cache_bytes(config, seq_len, batch=1, dtype="fp16")`. It returns the exact number of bytes
a decoder's key-value cache occupies for `batch` sequences that each hold `seq_len` tokens of context.

The model config is a `dict` with these keys:

- `num_layers`: layers that keep a cache.
- `num_kv_heads`: key/value heads per layer. This is not the query-head count.
- `head_dim`: width of one head.
- `sliding_window` (optional): the window in tokens. `None`, or a missing key, means full attention.

Other keys, such as `hidden_size` or `num_attention_heads`, are ignored.

The cache holds one key and one value per token, per layer, per KV head:

$$
\text{bytes} = 2 \times L \times H_{kv} \times d_h \times \min(S, W) \times B \times b
$$

where `S = seq_len`, `W` is the sliding window (or `S` when there is none), `B = batch`, and `b` is the
bytes per element for the dtype. Use the integer arithmetic exactly as written.

## Contract

- `config`: a `dict` with the keys above. Values for the three required keys and the window must be Python
  `int` values. The window is `None` or an `int` of at least 1.
- `num_layers`, `num_kv_heads` and `head_dim` are each at least 1.
- `seq_len`: a Python `int`, at least 0. Zero means no cached tokens, so the result is 0.
- `batch`: a Python `int`, at least 1.
- `dtype`: a `str`, one of `"fp32"` (4 bytes), `"fp16"` (2), `"bf16"` (2) or `"fp8"` (1).
- Returns: a Python `int` number of bytes. Never a float.
- Raises `TypeError` when `config` is not a `dict`, or when `seq_len`, `batch`, or a config value is not a
  Python `int`. `bool`, `float`, `str` and `None` are rejected, except that `sliding_window` may be `None`.
- Raises `ValueError` when a required key is missing, when `num_layers`, `num_kv_heads`, `head_dim`,
  `batch` or the window is below its minimum, when `seq_len` is negative, or when `dtype` is unknown.
- Must not mutate `config`.

## Examples

```python
>>> kv_cache_bytes({"num_layers": 32, "num_kv_heads": 32, "head_dim": 128}, 4096)
2147483648        # 2 GiB: 2 * 32 * 32 * 128 * 4096 * 1 * 2
>>> kv_cache_bytes({"num_layers": 32, "num_kv_heads": 32, "head_dim": 128}, 0, batch=2)
0
>>> kv_cache_bytes({"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "sliding_window": 4096}, 32768)
536870912         # 512 MiB: the window stores 4096 tokens per layer, not 32768
>>> kv_cache_bytes({"num_layers": 32, "num_kv_heads": 32, "head_dim": 128}, 4096, dtype="fp8")
1073741824        # 1 GiB: one byte per element
>>> kv_cache_bytes({"num_layers": 32, "num_kv_heads": 32, "head_dim": 128}, 4096, dtype="int4")
ValueError        # dtype is not one of the four names
```

## Constraints

- Allowed: built-in types and the standard library.
- Not allowed: `numpy`, `scipy`, `torch` and any other numeric library. The arithmetic is plain integer
  multiplication, and a float intermediate would lose exactness above 2**53.

## Done looks like

The public tests check the examples and one empty-context case. **Submit** adds hidden tests. They
compare your count with an independent computation on seeded random configurations, and they cover:
grouped-query and multi-query head counts; a window smaller than, equal to and larger than the context;
zero context; batch scaling; every dtype name; extra config keys that must be ignored; exact results
beyond float precision; the return type; each validation rule; and a config that must not change.

Some hidden tests take their inputs from the real datasets `gutenberg_frankenstein`,
`gutenberg_pride_prejudice` and `gutenberg_walden` (see `docs/DATASETS.md`).
