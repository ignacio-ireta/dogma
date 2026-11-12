# Bottom-right causal mask with key padding

Implement `causal_padding_mask(query_len, key_len, key_valid=None)`. It returns a boolean
attention mask in which `True` means the query may attend the key. The causal part is aligned to
the bottom right: the last query is the token at the last key position. This is the alignment a
key-value cache needs, where the `query_len` new tokens follow `key_len - query_len` cached tokens.
An optional key padding indicator removes padded keys from every query.

## Contract

- `query_len` (`L`) and `key_len` (`S`): Python `int`, `>= 0`. A `bool`, a `float` or a negative
  value is invalid. Either may be zero.
- Causal rule: query `i` (`0 <= i < L`) may attend key `j` (`0 <= j < S`) exactly when
  `j <= S - L + i`. The offset `S - L` may be negative (when `L > S`).
- `key_valid`: `None`, or a boolean tensor of shape `(N, S)` where `True` marks a real key and
  `False` marks padding. `N >= 0`.
- Returns, when `key_valid` is `None`: boolean tensor of shape `(L, S)`.
- Returns, when `key_valid` is given: boolean tensor of shape `(N, L, S)`, where the entry for
  batch `n`, query `i`, key `j` is the causal rule AND `key_valid[n, j]`.
- Fully masked rows: a query row with no `True` entry is returned as all `False`. The function
  never raises for such a row and never sets any entry to `True` to avoid it.
- Raises: `ValueError` for invalid `query_len` or `key_len`, a non-boolean `key_valid`, or a
  `key_valid` that is not 2-D with `S` columns.
- Must not mutate `key_valid`. The returned tensor must not share storage with `key_valid`.

## Examples

```python
>>> causal_padding_mask(2, 2)
tensor([[ True, False],
        [ True,  True]])
```

```python
>>> causal_padding_mask(2, 4)               # two new tokens after a cache of two
tensor([[ True,  True,  True, False],
        [ True,  True,  True,  True]])
```

```python
>>> causal_padding_mask(1, 3, torch.tensor([[True, False, True]]))
tensor([[[ True, False,  True]]])
```

```python
>>> causal_padding_mask(3, 2)               # L > S: query 0 has no allowed key
tensor([[False, False],
        [ True, False],
        [ True,  True]])
```

## Constraints

- Allowed: torch tensor operations and Python integer arithmetic, such as `torch.arange`,
  comparisons, broadcasting, `&` and `torch.ones(...).tril(...)`.
- Not allowed: `torch.nn` (including `torch.nn.Transformer.generate_square_subsequent_mask`,
  which only covers square masks). Build the mask from the rule above.

## Done looks like

The four examples pass. Hidden tests compare the causal part with a `tril(diagonal=S-L)` oracle
over many `(L, S)` pairs, including `L > S`, `L = S` and `L < S`. They check the padded mask against
an independent AND, confirm that the single-query decode step sees every key, show that the
bottom-right alignment differs from torch's top-left `is_causal` mask when `L < S`, check fully
masked rows (including all-padding batches), boolean output dtype, the shape rules, zero and
invalid sizes, and input non-mutation.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).
