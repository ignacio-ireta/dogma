# Filtered next-token distribution

Implement `filtered_distribution(logits, temperature, top_k=None, top_p=None)`. It turns one
vector of next-token logits into the probability distribution a sampler draws from. The steps are,
in this order: scale the logits by temperature, take the softmax, keep only the top-k tokens, keep
only the nucleus (top-p) tokens, and renormalise. The function returns the distribution. It draws
nothing, so the result is deterministic.

## Contract

- `logits`: `torch.float64` tensor of shape `(V,)`, `V >= 1`. Every entry is finite.
- `temperature`: a finite real number with `temperature > 0`. A `bool` is rejected. Temperature
  `0` is rejected, because `logits / 0` is undefined. Greedy decoding (argmax) is a separate rule
  and is not this function.
- `top_k`: `None`, or an `int` with `top_k >= 1`. `None` disables top-k. A `bool` is rejected. If
  `top_k >= V`, nothing is removed.
- `top_p`: `None`, or a real number with `0 < top_p <= 1`. `None` and `1.0` both disable
  top-p. A `bool` is rejected.
- Steps:
  1. `z = logits / temperature`, then `p = softmax(z)`. Use a stable softmax.
  2. Top-k. Sort the tokens by `p`, descending. Ties go to the lower token index. Keep the first
     `top_k` tokens and set the rest to probability `0`. Renormalise the kept tokens to sum to `1`.
  3. Top-p. Use the probabilities **after step 2's renormalisation**, so the survivors of top-k sum
     to `1`. Sort the kept tokens by those probabilities, descending, with ties again going to the
     lower index. Let `c_j` be the cumulative sum of the sorted probabilities, in float64, and compare
     it to `top_p` directly, with no tolerance. Keep the smallest prefix whose cumulative sum satisfies
     `c_j >= top_p`, including the token that reaches it. The prefix always holds at least one token.
     Set the rest to `0` and renormalise. With `top_k` absent, step 3 runs on the full softmax.
- Returns: `torch.float64` tensor of shape `(V,)`. Every entry is `>= 0`, and the entries sum to `1`.
  Removed tokens are exactly `0`.
- Raises `ValueError` when: `logits` is not a non-empty 1-D `float64` tensor; it contains a
  non-finite value; `temperature` is not a finite positive number; `top_k` is neither `None` nor an
  `int >= 1`; or `top_p` is neither `None` nor a number in `(0, 1]`.
- Must not mutate `logits`.

## Examples

```python
>>> filtered_distribution(torch.zeros(4, dtype=torch.float64), temperature=1.0)
tensor([0.2500, 0.2500, 0.2500, 0.2500], dtype=torch.float64)
```

```python
>>> logits = torch.log(torch.tensor([0.5, 0.25, 0.125, 0.125], dtype=torch.float64))
>>> filtered_distribution(logits, temperature=1.0, top_p=0.7)   # 0.5 < 0.7 <= 0.75
tensor([0.6667, 0.3333, 0.0000, 0.0000], dtype=torch.float64)
```

```python
>>> filtered_distribution(torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float64),
...                       temperature=1.0, top_k=1)            # tie: the lower index wins
tensor([0., 1., 0., 0.], dtype=torch.float64)
```

```python
>>> # Order matters: top-k first, then top-p on the renormalised survivors.
>>> probs = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
>>> filtered_distribution(torch.log(probs), temperature=1.0, top_k=2, top_p=0.5)
tensor([1., 0., 0., 0.], dtype=torch.float64)
# Top-k keeps 0.4 and 0.3, renormalised to 0.571 and 0.429. The first one alone reaches 0.5.
# Using the unrenormalised 0.4 and 0.7 would keep two tokens instead.
```

```python
>>> filtered_distribution(torch.tensor([0.0, 1.0], dtype=torch.float64), temperature=0.5)
tensor([0.1192, 0.8808], dtype=torch.float64)                 # z = [0, 2]
```

## Constraints

- Allowed: `torch` tensor operations, such as `exp`, `sort` with `stable=True`, `cumsum`,
  `where`, `max`, `sum` and `nonzero`; plain Python `math` and `int`/`float`/`bool` checks.
- Not allowed: `torch.nn` in any form, `torch.softmax` and `torch.log_softmax` (and therefore
  `.softmax` and `.log_softmax` in any form, including `torch.special` and `torch.nn.functional`), and
  `torch.distributions`. Write the softmax yourself: subtract the maximum, exponentiate, divide by the sum.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They cover: agreement with an
independent pure-Python reference on seeded random logits across temperature, top-k and top-p
combinations; an exact top-p boundary (cumulative sum equal to `top_p`) that keeps the token that
reaches it and not the next one; ties at the boundary for top-p and top-k; `top_p` so small that
only the best token can survive; `top_p = 1` and `top_k >= V` as no-ops; `top_k = 1` as a one-hot
at the argmax; temperature `1` as a plain softmax; small temperatures and large logit magnitudes
that must stay finite; a one-entry vocabulary; the output sum, sign and dtype; the input left
unchanged; and the `ValueError` rules for every argument.
