# Pack documents into next-token (inputs, targets) blocks

Implement `pack_next_token_blocks(documents, block_size, eos_id)`. It joins token-id documents into one
stream, with `eos_id` after every document, and cuts the stream into blocks for next-token training.
Each block has `block_size` inputs, and its targets are the same tokens shifted left by one.

## Contract

- `documents`: a `list` or `tuple` of documents. Each document is a `list` or `tuple` of Python `int`
  token ids. A document may be empty, and the list may be empty. Every id is `0 <= id <= 2**63 - 1`.
- `block_size` (`T`): Python `int`, `>= 1`. A `bool` is invalid.
- `eos_id`: Python `int`, `>= 0`. A `bool` is invalid. It may equal a token id that occurs in a document.
- Stream: `S` is the concatenation of `doc_1 + [eos_id] + doc_2 + [eos_id] + ...`, one EOS after every
  document, including the last. Its length is `N = sum(len(doc) + 1)`. An empty document contributes
  only its EOS.
- Blocks: there are `B = max(0, (N - 1) // T)` blocks. Block `b` (from 0) is
  - `inputs[b] = S[b*T : b*T + T]`
  - `targets[b] = S[b*T + 1 : b*T + T + 1]`
- Remainder rule: the last `N - 1 - B*T` tokens of the stream are dropped. Nothing is padded. A block
  that is not full is never returned.
- Returns: `(inputs, targets)`, both `torch.int64` tensors of shape `(B, T)`. When `B = 0`, both have
  shape `(0, T)`.
- Raises `ValueError` when: `documents` is not a list or tuple; a document is not a list or tuple; a
  token id is not a Python `int`, is negative, or exceeds `2**63 - 1`; `block_size` or `eos_id` is not a
  valid Python `int`; `block_size < 1`; `eos_id < 0`.
- Must not mutate `documents` or any document. In particular, do not append the EOS to the caller's lists.

## Examples

```python
pack_next_token_blocks([[1, 2], [3]], block_size=2, eos_id=0)
# stream S = [1, 2, 0, 3, 0], N = 5, B = (5 - 1) // 2 = 2
# inputs  = [[1, 2], [0, 3]]
# targets = [[2, 0], [3, 0]]
```

```python
pack_next_token_blocks([[7], [8, 9]], block_size=3, eos_id=0)
# stream S = [7, 0, 8, 9, 0], N = 5, B = 4 // 3 = 1
# inputs  = [[7, 0, 8]]
# targets = [[0, 8, 9]]           the final EOS is dropped, not padded
```

```python
pack_next_token_blocks([], block_size=4, eos_id=0)
# inputs and targets both have shape (0, 4)
```

## Constraints

- Allowed: Python list operations, integer arithmetic, and torch tensor creation and slicing, such as
  `torch.tensor`, `torch.arange`, `torch.stack` and indexing.
- Not allowed: `torch.nn`, `torch.utils.data` (its datasets and loaders do the windowing for you),
  `torch.as_strided`, and `torch.Tensor.unfold` (a sliding-window view). Build the windows from index
  arithmetic or slices.

## Done looks like

The examples pass. Hidden tests compare the blocks with a plain-Python reference that builds the stream
and slices it, on seeded random documents and block sizes. They check the shift (targets equal inputs
shifted by one, and each block's last target is the next input), that every document ends with one EOS in
the stream, that the remainder is dropped (a block count one below the padded count, for example), that
blocks larger than the stream give `(0, T)`, that empty documents and an empty list are handled, that ids
above $2^{24}$ survive exactly, that outputs are `int64`, that invalid arguments raise `ValueError`, and
that inputs are not mutated.
