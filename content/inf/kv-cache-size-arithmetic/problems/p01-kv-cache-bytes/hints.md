## Hint 1 — concept

The cache stores one key and one value per token, per layer, per KV head. Ask which head count is stored, not which head
count is computed. Then ask how many tokens each layer keeps when a sliding window is set.

## Hint 2 — formula / approach

Count the tokens per layer first: the context length, or the window when the window is smaller. Then multiply the
factor 2 for K and V, the layers, the KV heads, the head width, the stored tokens, the batch and the bytes per element.
Every factor is an integer, so the product is an exact integer. Check the units with the worked 7B example: a 4096-token
sequence at fp16 is 2 GiB.

## Hint 3 — code shape

Validate the arguments first, then compute in this order:

```python
stored_tokens = ...  # the context, capped by the window when one is set
return 2 * layers * kv_heads * head_dim * stored_tokens * batch * bytes_per_element
```

Look up bytes_per_element in a dictionary of the four dtype names, and raise ValueError for any other name.
