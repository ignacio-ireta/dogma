## Hint 1 — concept

Each request gives three numbers, and each number has its own population. TTFT and end-to-end latency come from every
request. Mean ITL comes only from requests that have a gap, so a one-token request does not supply one. A percentile is
a position in a sorted list, and the list is different for each metric.

## Hint 2 — formula / approach

For one request with arrival `a` and token times `t[0] < ... < t[n-1]`: TTFT is `t[0] - a`, end-to-end is
`t[n-1] - a`, and mean ITL is `(t[n-1] - t[0]) / (n - 1)` when `n >= 2`. Percentiles use linear interpolation: with
`m` sorted values, the position is `h = (m - 1) * p / 100`. Take the values on either side of `h` and weight them by
the fractional part of `h`. Do not round `h` to the nearest rank.

## Hint 3 — code shape

Validate each request in a loop, append its three numbers to three lists, then sort the lists once. Make a small
percentile helper that takes a sorted list and a percentile. The skeleton is:

```python
ttft.append(times[0] - arrival)
e2e.append(times[-1] - arrival)
if len(times) >= 2:
    itl.append((times[-1] - times[0]) / (len(times) - 1))
# after the loop: sort each list once, then call the helper with 50 and 95
```

Check the worked example on paper before you run the tests. Its TTFT list, sorted, is 0.1, 0.5, 1.0, so the p50 is 0.5.
