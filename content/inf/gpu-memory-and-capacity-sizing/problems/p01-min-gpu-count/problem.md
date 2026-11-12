# Minimum tensor-parallel GPU count for a model and a target load

Implement `min_gpu_count(params, config, batch, context_len, element_bytes, gpu_memory_bytes, *, workspace_bytes, reserve_percent)`.
It returns the smallest tensor-parallel degree `T` (the number of GPUs that serve one model replica) at which one GPU
holds its share of the weights, its share of the KV cache for `batch` sequences of `context_len` tokens, and its
workspace, inside the usable part of its memory. It returns `0` if no allowed degree fits.

**The model (use it exactly).**

**Allowed degrees.** `T` is a power of two: 1, 2, 4, 8, and so on. It must also divide `num_kv_heads`, because each KV
head lives on exactly one GPU. A degree that does not divide the KV heads is not allowed, even if it would fit by bytes.
Candidates are taken in increasing order.

**Usable memory per GPU.** `usable = gpu_memory_bytes * (100 - reserve_percent) // 100`, using floor division. The
reserve covers fragmentation and runtime overhead.

**Per-GPU bytes at degree `T`.** Let `W = params * element_bytes` and `stored = context_len`, or `min(context_len, sliding_window)`
when the config has a window.

- Weights per GPU: `ceil(W / T)`. Round the shard up.
- KV per GPU: `2 * num_layers * (num_kv_heads // T) * head_dim * stored * batch * element_bytes`. The division by `T`
  is exact because `T` divides `num_kv_heads`.
- Total per GPU: weights + KV + `workspace_bytes`.

The degree fits when its total is at most `usable`. Equality fits. Return the smallest fitting `T`, or `0` when none fits.

**Config.** `config` is a `dict` with the Python `int` keys `num_layers`, `num_kv_heads` and `head_dim`, each at least 1,
and an optional `sliding_window` (`None`, or missing, means full attention; an `int` of at least 1 caps `stored`). A window that is neither `None`, missing nor a Python `int` raises `TypeError`;
a window below 1 raises `ValueError`.
Other keys are ignored. Must not be mutated.

## Contract

- `params`: Python `int`, at least 1. Total parameters, all of which are stored.
- `config`: as above.
- `batch`: Python `int`, at least 1. Concurrent sequences.
- `context_len`: Python `int`, at least 1. Tokens per sequence at the target, prompt plus generated.
- `element_bytes`: Python `int`, at least 1. Bytes per element for both weights and cache.
- `gpu_memory_bytes`: Python `int`, at least 1. Memory of one GPU in bytes.
- `workspace_bytes`: keyword-only Python `int`, at least 0. Activation and runtime workspace per GPU.
- `reserve_percent`: keyword-only Python `int`, from 0 to 99 inclusive. Percent of GPU memory held back.
- Returns a Python `int`, `0` or a power of two, never a float.
- Raises `TypeError` when a value is not a Python `int` (`bool`, `float`, `str` and `None` are rejected), or when `config`
  is not a `dict`. Raises `ValueError` when a value is out of range (including `reserve_percent` of 100 or more), a
  required key is missing, or the window is below 1.

## Examples

A 70B-class model: `params = 70_000_000_000`, `num_layers = 80`, `num_kv_heads = 8`, `head_dim = 128`, fp16 (`element_bytes = 2`),
80 GiB GPUs (`gpu_memory_bytes = 80 * 2**30`), 4 GiB of workspace, and a 10 percent reserve. The usable memory per GPU is
77,309,411,328 bytes. Weights are 140,000,000,000 bytes, and one token of cache is 327,680 bytes for all KV heads.

```python
>>> CFG = {"num_layers": 80, "num_kv_heads": 8, "head_dim": 128}
>>> GPU, WS = 80 * 2**30, 4 * 2**30
>>> min_gpu_count(70_000_000_000, CFG, 32, 4096, 2, GPU, workspace_bytes=WS, reserve_percent=10)
4        # T=1: weights alone exceed usable; T=2: 95.8e9 > 77.3e9; T=4: 50.0e9 fits
```

```python
>>> min_gpu_count(70_000_000_000, CFG, 1, 4096, 2, GPU, workspace_bytes=WS, reserve_percent=10)
2        # one sequence: T=2 needs 74,966,055,936 bytes, which fits in 77,309,411,328
```

```python
>>> min_gpu_count(70_000_000_000, CFG, 512, 32768, 2, GPU, workspace_bytes=WS, reserve_percent=10)
0        # no allowed degree (1, 2, 4 or 8) holds the cache of 512 sequences of 32,768 tokens
```

```python
>>> min_gpu_count(70_000_000_000, CFG, 1, 4096, 2, GPU, workspace_bytes=WS, reserve_percent=100)
ValueError       # reserve_percent must be at most 99
```

## Constraints

- Allowed: built-in types and the standard library.
- Not allowed: `numpy`, `scipy`, `torch`, and any helper that searches or sizes the deployment for you. Use Python
  integers for every byte count, so the fit test is exact.

## Done looks like

The public tests check the examples and the reserve validation. **Submit** adds hidden tests. They compare your answer
with an independent computation over seeded random models, shapes and memory sizes. They also cover: a total equal to
the usable memory fitting, and one byte more not; a reserve that is floored, not rounded; a weight shard rounded up
when it does not divide; a KV-head count that rules out a power of two that would otherwise fit; a sliding window that
caps the cache; a model whose weights alone never fit; large byte counts that stay exact; zero workspace and zero
reserve; the return type; configs that must not change; and each validation rule.

Some hidden tests take their inputs from the real datasets `gutenberg_frankenstein`,
`gutenberg_pride_prejudice` and `gutenberg_walden` (see `docs/DATASETS.md`).
