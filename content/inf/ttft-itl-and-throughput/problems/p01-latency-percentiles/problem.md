# Per-request latency and percentiles

Implement `latency_summary(requests)`. Each request records when it arrived and when each of its output
tokens appeared. From these timestamps compute three latencies per request, then the p50 and p95 of each
metric across requests.

Each request is a `dict` with two keys:

- `arrival`: the time the request arrived, in seconds.
- `token_times`: the times at which each output token appeared, in order. `token_times[0]` is the first token.

**Definitions.** For a request with `n` tokens (`n >= 1`) and `arrival = a`:

- **TTFT** = `token_times[0] - a`.
- **E2E** = `token_times[-1] - a`.
- **Mean ITL** = `(token_times[-1] - token_times[0]) / (n - 1)`. It exists only when `n >= 2`.

**Percentile.** Use linear interpolation between sorted values (NumPy's default). For `m` sorted values
`x[0] <= ... <= x[m-1]` and `p` in `0..100`, let `h = (m - 1) * p / 100`, `lo = floor(h)`, then

`value = x[lo] + (h - lo) * (x[lo + 1] - x[lo])`, and `value = x[m - 1]` when `lo + 1 == m`.

With one value, the percentile is that value.

**Which values feed each percentile.** TTFT and E2E use every request. Mean ITL uses only requests with at least
two tokens. A one-token request has no gaps, so it is not zero and not missing from the TTFT or E2E lists.

## Contract

- `requests`: a non-empty `list` of request `dict` values. An empty list raises `ValueError`.
- Each request must have `arrival` and `token_times`. A missing key raises `ValueError`.
- A request that is not a `dict`, or whose `token_times` is not a `list`, raises `TypeError`.
- `arrival` and each time value: `int` or `float`, finite, and not `bool`. Other types raise `TypeError`;
  non-finite values raise `ValueError`.
- `token_times` must be non-empty and strictly increasing. `arrival <= token_times[0]`. Violations raise `ValueError`.
- Returns: a `dict` with exactly these keys, each a Python `float`, or `None` where stated:
  `ttft_p50`, `ttft_p95`, `itl_p50`, `itl_p95`, `e2e_p50`, `e2e_p95`.
  `itl_p50` and `itl_p95` are `None` when no request has two or more tokens.
- Must not mutate `requests`.

## Examples

```python
>>> requests = [
...     {"arrival": 0.0, "token_times": [0.5, 0.6, 0.8]},
...     {"arrival": 0.2, "token_times": [1.2]},
...     {"arrival": 1.0, "token_times": [1.1, 1.3, 1.6, 2.0]},
... ]
>>> latency_summary(requests)
{'ttft_p50': 0.5, 'ttft_p95': 0.95, 'itl_p50': 0.225, 'itl_p95': 0.2925,
 'e2e_p50': 1.0, 'e2e_p95': 1.0}
>>> latency_summary([{"arrival": 0.0, "token_times": [0.25]}])
{'ttft_p50': 0.25, 'ttft_p95': 0.25, 'itl_p50': None, 'itl_p95': None,
 'e2e_p50': 0.25, 'e2e_p95': 0.25}
>>> latency_summary([{"arrival": 0.0, "token_times": [1.0, 1.0]}])
ValueError  # token_times must be strictly increasing
>>> latency_summary([])
ValueError  # at least one request is required
```

In the first example, request A has mean ITL `(0.8 - 0.5) / 2 = 0.15` and request C has `(2.0 - 1.1) / 3 = 0.3`.
Request B has no ITL. Its TTFT of 1.0 still enters the TTFT list.

## Constraints

- Allowed: built-in types and the standard library, except the modules below.
- Not allowed: `statistics` (its quantile functions compute the answer) and `numpy` (its percentile function does
  the same). Write the interpolation yourself.
- Not allowed: any approach whose cost grows quadratically with the number of requests, such as scanning the
  whole list once per rank. The hidden tests include a list of 50,000 requests.

## Done looks like

The public tests check the two worked examples and one validation rule. **Submit** adds hidden tests that compare
your percentiles with an exact-fraction computation on seeded random requests, and they cover: one request; all
one-token requests; mean ITL computed per request rather than over pooled gaps; E2E measured from arrival;
percentiles where nearest-rank and linear interpolation differ; ties; large timestamps and small gaps; ints and
floats mixed; the return types; each validation rule; input that must not change; and a 50,000-request list that
must finish within the time limit.

The hidden tests draw their bulk inputs from the real dataset `online_retail` (see
`docs/DATASETS.md`).
