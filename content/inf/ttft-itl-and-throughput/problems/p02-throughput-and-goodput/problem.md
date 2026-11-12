# Throughput and goodput under an SLO

Implement `serving_rates(requests, ttft_slo, itl_slo)`. It returns two rates over one measurement window: the
throughput in tokens per second, and the goodput in requests per second. Goodput counts only the requests that meet
the service-level objective (SLO).

Each request is a `dict` with `arrival` and `token_times`, exactly as in the latency-percentile problem. The
per-request definitions are the same:

- **TTFT** = `token_times[0] - arrival`.
- **Mean ITL** = `(token_times[-1] - token_times[0]) / (n - 1)`, when `n >= 2`. A one-token request has no ITL.

**Window.** `window = max(token_times[-1] over requests) - min(arrival over requests)`. It runs from the first
arrival to the last token of any request, so idle time at the start and the end counts.

**Throughput** `= (total tokens over all requests) / window`. Every token counts, including the first token of each
request. The total is the sum of `len(token_times)`.

**Goodput** `= (number of requests that meet the SLO) / window`.

A request **meets the SLO** when both bounds hold, and both bounds are inclusive:

- `TTFT <= ttft_slo`, and
- `mean ITL <= itl_slo`, or the request has one token, so it has no ITL to exceed.

## Contract

- `requests`: a non-empty `list` of request `dict` values, with the validation rules of the latency-percentile
  problem: `TypeError` for wrong types, `ValueError` for a missing key, an empty token list, non-increasing times,
  a first token before arrival, or a non-finite time.
- `ttft_slo`, `itl_slo`: `int` or `float`, finite, `>= 0`, and not `bool`. Wrong types raise `TypeError`; a negative
  or non-finite value raises `ValueError`.
- If `window <= 0`, raise `ValueError`: the measurement has no duration.
- Returns: a `tuple` of two Python `float` values, `(throughput_tokens_per_s, goodput_requests_per_s)`.
- Must not mutate `requests`.

## Examples

```python
>>> requests = [
...     {"arrival": 0.0, "token_times": [0.5, 0.6, 0.8]},
...     {"arrival": 0.2, "token_times": [1.2]},
...     {"arrival": 1.0, "token_times": [1.1, 1.3, 1.6, 2.0]},
... ]
>>> serving_rates(requests, 0.6, 0.2)
(4.0, 0.5)     # window 2.0 s; 8 tokens; only request A meets both bounds: 1 / 2.0
>>> serving_rates([{"arrival": 0.0, "token_times": [0.5, 1.0]},
...                {"arrival": 0.0, "token_times": [1.0]}], 0.5, 0.5)
(3.0, 1.0)     # window 1.0 s; 3 tokens; the first request sits exactly on both bounds
>>> serving_rates([{"arrival": 1.0, "token_times": [1.0]}], 0.5, 0.5)
ValueError     # window is 0.0 seconds
```

In the first example, request B fails on TTFT (1.0 > 0.6) and request C fails on ITL (0.3 > 0.2).

## Constraints

- Allowed: built-in types and the standard library, except the modules below.
- Not allowed: `statistics` and `numpy`. The rates need only sums, a maximum and a minimum.

## Done looks like

The public tests check the worked examples and the zero-window rule. **Submit** adds hidden tests that compare your
rates with an exact-fraction computation on seeded random requests, and they cover: inclusive bounds on both SLOs;
a window that starts at the earliest arrival rather than the first token; throughput counting every token; one-token
requests judged on TTFT alone; goodput counting requests rather than tokens; SLO values of zero; the return type;
each validation rule; and input that must not change.

The hidden tests draw their bulk inputs from the real dataset `online_retail` (see
`docs/DATASETS.md`).
