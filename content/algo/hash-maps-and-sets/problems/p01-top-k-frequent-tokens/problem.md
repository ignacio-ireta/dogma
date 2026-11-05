# Top-k frequent tokens

Implement `top_k_frequent(tokens, k)`. Count how often each token appears, and return the `k` most
frequent tokens together with their counts. Tokens with equal counts are ordered alphabetically.

## Contract

- `tokens`: a list of `str`, possibly empty. Tokens are compared exactly: case-sensitive, with no
  stripping or normalisation. The empty string `""` is an ordinary token.
- `k`: an `int`. `k <= 0` returns `[]`. A `k` larger than the number of distinct tokens returns
  every distinct token.
- Returns: a `list` of `(token, count)` tuples, with `count` an `int` of at least 1. Sort by count,
  highest first. Break ties by token in ascending Python string order, so `"B"` comes before `"a"`
  because uppercase letters sort first.
- Length of the result: `min(max(k, 0), number of distinct tokens)`.
- Raises: nothing in this problem.
- Must not mutate `tokens`.

## Examples

```python
>>> top_k_frequent(["the", "cat", "the", "dog", "cat", "the"], 2)
[('the', 3), ('cat', 2)]
>>> top_k_frequent(["b", "a", "c", "a", "b"], 2)     # a and b tie at 2; a comes first
[('a', 2), ('b', 2)]
>>> top_k_frequent(["x", "y"], 0)
[]
>>> top_k_frequent(["x", "y"], 10)                   # k exceeds the vocabulary
[('x', 1), ('y', 1)]
>>> top_k_frequent([], 3)
[]
```

## Constraints

- Allowed: built-in types (`dict`, `set`, `list`), `sorted`, `typing`, and the standard library
  except `collections`.
- Not allowed: `collections`, as an import or as attribute access (for example
  `collections.Counter`). Write the counting yourself.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare your answer against an
independent `collections.Counter` reference on random token lists, and on a larger seeded stream. The
edge families are: k zero or negative, k larger than the vocabulary, empty input, case-sensitive
tokens, the empty string as a token, ties ordered alphabetically rather than by first appearance,
the element types of the result, and no mutation of the input.
