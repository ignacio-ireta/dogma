## Hint 1 — concept

A forward pass has two costs: arithmetic, which grows with the number of tokens in the pass, and traffic, which
includes the weights (read once per pass whatever the batch) and the cache (read for every sequence). Each phase is
limited by whichever of the two floors is higher. Prefill puts many prompt tokens in one pass; decode puts one token
per sequence in each step.

## Hint 2 — formula / approach

Derive two byte counts and two FLOP counts. The weight bytes are the parameters times the element width. The cache
bytes per token are the factor 2 for keys and values, times the layers, the KV heads, the head width and the element
width. Prefill FLOPs are twice the parameters times the prompt tokens in the batch, and its bytes are the weights plus
the cache written for those tokens. A decode step's FLOPs are twice the parameters times the batch, and its bytes are
the weights plus the cache of the context plus one new token for every sequence. Each phase time is the larger of its
FLOPs over peak and bytes over bandwidth. Throughput is the batch over the decode step.

## Hint 3 — code shape

Validate every argument first, with a type check before the range check, and the config keys before any arithmetic.
Then compute, in this order:

- The weight bytes and the cache bytes per token, once each.
- For prefill, the FLOPs and the bytes, then the larger of the two divided by the device rates.
- For a decode step, the same pattern with the decode FLOPs and bytes.
- The throughput as a ratio of the batch to the step time.

Return a dict with the three float keys.
