# Packed causal mask

A training row can hold several short documents back to back. For one packed row of `T` tokens,
build two arrays. `packed_causal_mask(doc_ids)` returns the attention mask that lets each token see
only earlier tokens of its own document. `restart_positions(doc_ids)` returns position ids that
start again at 0 for each document.

The document ids label the tokens. A document is a maximal run of equal ids, so two adjacent
documents must carry different ids.

## Contract

- `doc_ids`: 1-D integer array-like of length $T \ge 1$. Each value names the document of its
  token. Each id must occupy one contiguous run in the row.
- `packed_causal_mask` returns: boolean NumPy array of shape `(T, T)`. `mask[i, j]` is `True`
  exactly when `doc_ids[i] == doc_ids[j]` and `j <= i`.
- `restart_positions` returns: `int64` NumPy array of shape `(T,)`. `pos[i]` is the offset of token `i` from the first token of its document, that is `i` minus the index
  of that first token, so each document starts at 0.
- Raises `ValueError` when `doc_ids` is not 1-D, is empty, is not an integer dtype (`bool` and
  float are rejected), or contains an id that reappears after a different id. Both functions
  apply the same rules.
- Must not mutate inputs.

## Examples

```python
>>> packed_causal_mask([0, 0, 1])
array([[ True, False, False],
       [ True,  True, False],
       [False, False,  True]])
```

```python
>>> restart_positions([0, 0, 1, 1, 1])
array([0, 1, 0, 1, 2])
```

```python
>>> packed_causal_mask([0, 1, 0])
ValueError: doc_ids must form contiguous runs
```

## Constraints

- Allowed: NumPy array operations (`np.asarray`, `np.arange`, comparisons, `&`, `np.flatnonzero`,
  `np.maximum.accumulate`).
- Not allowed: `numpy.tri` and `numpy.tril`. Build the causal rule from index comparisons, so the
  cross-document rule is written out in your code.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check a hand-built row of
eight tokens across three documents, equivalence with a plain causal mask for a single document,
agreement with a block-diagonal reference on seeded random rows, and property tests: no key from
another document is ever visible, each token attends to every earlier token of its own document,
and each token attends to itself. They also cover restart positions, the input contract (empty,
two-dimensional, non-integer and non-contiguous ids), outputs of the documented dtype and shape,
and inputs that are left unchanged.
