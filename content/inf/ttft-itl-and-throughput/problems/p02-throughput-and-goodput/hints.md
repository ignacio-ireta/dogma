## Hint 1 — concept

Two different numerators share one window. Throughput counts tokens and goodput counts requests, and both are
divided by the same duration. Ask first what the duration starts from. The earliest arrival is the answer that
matches how a user experiences the run, not the first token.

## Hint 2 — formula / approach

Compute `window = max(last token time) - min(arrival)`. Then `throughput = sum of len(token_times) / window`.
For goodput, a request passes when `TTFT <= ttft_slo` and, if it has two or more tokens, its mean ITL is at
most `itl_slo`. Both comparisons are inclusive, and a one-token request has no ITL to test.

## Hint 3 — code shape

Make one pass over the requests, collecting three running values: the earliest arrival, the latest token time
and the token total. Count passing requests in the same pass. Then divide:

```python
first_arrival = min(...)  # running minimum over arrivals
last_token = max(...)  # running maximum over the last token times
total_tokens += ...  # len of each request's token list
if ttft <= ttft_bound and itl_ok:
    passing += 1
window = last_token - first_arrival
return total_tokens / window, passing / window
```

Check the example from problem.md by hand before running the tests: the window is 2.0 s, there are 8 tokens, and one
request passes.
