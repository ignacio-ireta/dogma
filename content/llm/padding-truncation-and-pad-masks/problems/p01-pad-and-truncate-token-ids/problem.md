# Pad and truncate token ids with an attention mask

Implement `pad_and_truncate(sequences, pad_id, max_length, *, truncation_side="right", padding_side="right")`.
It turns a list of token-id lists into a rectangular batch of shape `(B, max_length)` and an
attention mask of the same shape. Each row is truncated to `max_length` tokens on one side, padded
to `max_length` on one side, and marked with `1` for real tokens and `0` for padding.

## Contract

- `sequences`: a `list` or `tuple` of `B >= 0` rows. Each row is a `list` or `tuple` of Python `int`
  token ids. A row may be empty. Every id is `0 <= id <= 2**63 - 1`.
- `pad_id`: Python `int`, `>= 0`. A `bool` is invalid.
- `max_length`: Python `int`, `>= 1`. A `bool` is invalid.
- `truncation_side`: the string `"right"` or `"left"`.
- `padding_side`: the string `"right"` or `"left"`.
- For each row with `n` tokens, let `k = min(n, max_length)` be the number of kept tokens.
  - Truncation: if `n > max_length`, keep the first `max_length` tokens when `truncation_side` is
    `"right"`, or the last `max_length` tokens when it is `"left"`. Otherwise keep all `n` tokens.
  - Padding: place the `k` kept tokens in columns `0..k-1` when `padding_side` is `"right"`, or in
    columns `max_length-k..max_length-1` when it is `"left"`. Fill every other column with `pad_id`.
  - Mask: `attention_mask` is `1` exactly in the `k` columns that hold kept tokens, and `0` elsewhere.
    The mask depends on lengths only. A kept token whose id equals `pad_id` still gets `1`.
- Returns: `(input_ids, attention_mask)`, both `torch.int64` tensors of shape `(B, max_length)`.
  An empty `sequences` gives shape `(0, max_length)`.
- Raises `ValueError` when: `sequences` is not a list or tuple; a row is not a list or tuple; an id is
  not a Python `int`, is negative, or exceeds `2**63 - 1`; `pad_id` or `max_length` is not a valid
  Python `int`; `max_length < 1`; a side is not one of the two strings.
- Must not mutate `sequences` or any row.

## Examples

```python
pad_and_truncate([[5, 6, 7], [8]], pad_id=0, max_length=4)
# input_ids      = [[5, 6, 7, 0], [8, 0, 0, 0]]
# attention_mask = [[1, 1, 1, 0], [1, 0, 0, 0]]
```

```python
pad_and_truncate([[5, 6, 7], [8]], pad_id=0, max_length=4, padding_side="left")
# input_ids      = [[0, 5, 6, 7], [0, 0, 0, 8]]
# attention_mask = [[0, 1, 1, 1], [0, 0, 0, 1]]
```

```python
pad_and_truncate([[1, 2, 3, 4, 5]], pad_id=9, max_length=3, truncation_side="left")
# input_ids      = [[3, 4, 5]]
# attention_mask = [[1, 1, 1]]
```

```python
pad_and_truncate([[0, 4]], pad_id=0, max_length=3)
# input_ids      = [[0, 4, 0]]
# attention_mask = [[1, 1, 0]]     the first 0 is a real token, so it is marked 1
```

## Constraints

- Allowed: Python list and slice operations, and torch tensor creation and indexing, such as
  `torch.full`, `torch.zeros`, `torch.tensor` and slice assignment.
- Not allowed: `torch.nn` (this covers `torch.nn.utils.rnn.pad_sequence` and `torch.nn.functional.pad`,
  which pad a batch in one call), and `torch.nested` (its padded-tensor conversion does the whole job).
  Write the padding and the mask yourself.

## Done looks like

The four examples pass. Hidden tests compare right and left padding against a reference built from
`pad_sequence` (with a flip for left padding) on seeded random rows, and compare random ragged batches
with both truncation sides and both padding sides against a plain-Python reference. They also check
that a real token equal to the pad id stays unmasked, that an empty row is all padding with a zero mask,
that an empty batch gives shape `(0, max_length)`, that ids above $2^{24}$ survive exactly, that outputs
are `int64`, that invalid arguments raise `ValueError` (including `bool` sizes and out-of-range ids), and
that inputs are not mutated. A property test checks the mask row sums on random inputs.

The hidden tests draw their bulk inputs from the real datasets `gutenberg_frankenstein`,
`gutenberg_pride_prejudice` and `gutenberg_walden` (see `docs/DATASETS.md`).
