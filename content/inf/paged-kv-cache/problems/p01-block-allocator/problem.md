# Paged KV-cache block allocator

Implement `BlockAllocator`, a fixed pool of KV-cache blocks shared by many sequences. A sequence
does not reserve its maximum length. It receives blocks one at a time, as its tokens arrive, and
it returns them when it finishes. The allocator records each sequence's **block table**: the
physical block ids that hold its tokens, in logical order.

## Contract

Pool: physical blocks have ids `0 .. num_blocks - 1`. Each block holds `block_size` token slots.

- `BlockAllocator(num_blocks, block_size)`: both are `int`, not `bool`, and at least 1, otherwise
  `ValueError`.
- `num_free_blocks` (property): `int`, the number of blocks not held by any sequence.
- `allocate(seq_id, n_tokens) -> None`: start sequence `seq_id` with `n_tokens` tokens (`int`,
  `>= 0`). It takes `ceil(n_tokens / block_size)` blocks.
  - `seq_id` already active: `ValueError`.
  - `n_tokens < 0` or not an `int`: `ValueError`.
  - Not enough free blocks: raise `OutOfBlocksError`, and change nothing.
- `append_token(seq_id) -> tuple[int, int]`: store one more token for an active sequence. Let `p`
  be its 0-based token position before the append. If `p % block_size == 0`, the sequence needs a
  new block, so one is taken from the pool. Returns `(physical_block, offset)` where the new token
  lives, with `offset = p % block_size`.
  - Unknown `seq_id`: `KeyError`.
  - A new block is needed and none is free: `OutOfBlocksError`, and change nothing.
- `num_tokens(seq_id) -> int`: tokens stored for the sequence. Unknown `seq_id`: `KeyError`.
- `block_table(seq_id) -> list[int]`: a **new** list of physical ids in logical order. Mutating
  it must not change the allocator. Unknown `seq_id`: `KeyError`.
- `free(seq_id) -> None`: return all of the sequence's blocks to the pool, and forget the
  sequence. Unknown `seq_id`: `KeyError`.
- Reuse is deterministic. Every new block is the **lowest-numbered free** physical id. Freed ids
  become free again and are reused by that rule.
- `class OutOfBlocksError(RuntimeError)` is defined in the starter. Raise it; do not replace it.

## Examples

With `BlockAllocator(num_blocks=8, block_size=4)`:

```python
>>> a = BlockAllocator(8, 4)
>>> a.allocate(0, 10)          # ceil(10 / 4) = 3 blocks: the lowest free ids 0, 1, 2
>>> a.block_table(0)
[0, 1, 2]
>>> a.allocate(1, 5)           # ceil(5 / 4) = 2 blocks: 3, 4
>>> a.block_table(1)
[3, 4]
>>> a.append_token(0)          # position 10 sits in block 2 at offset 2: no new block
(2, 2)
>>> a.append_token(0)          # position 11 sits at offset 3 of block 2
(2, 3)
>>> a.append_token(0)          # position 12 starts a new block: lowest free id is 5
(5, 0)
>>> a.free(1)                  # blocks 3 and 4 return to the pool
>>> a.allocate(2, 4)           # lowest free ids are now 3, then 4 (one block: 3)
>>> a.block_table(2)
[3]
```

Pool exhaustion, from a concrete state: `BlockAllocator(num_blocks=4, block_size=4)`, then
`allocate(0, 8)` takes `[0, 1]` and `allocate(1, 4)` takes `[2]`. Block 3 is the only free one.

```python
>>> a.allocate(2, 9)           # needs 3 blocks, only 1 free
OutOfBlocksError
>>> a.num_free_blocks          # unchanged
1
>>> a.block_table(2)           # sequence 2 was never created
KeyError
>>> a.append_token(1)          # position 4 starts a new block: block 3 is taken
(3, 0)
>>> a.append_token(1)          # position 5 is offset 1 of block 3: no new block
(3, 1)
>>> a.append_token(0)          # position 8 needs a block, and none is free
OutOfBlocksError
>>> a.num_tokens(0)            # unchanged
8
```

When a call breaks more than one rule, the `ValueError` for bad arguments is raised before
`OutOfBlocksError`.

## Constraints

- Allowed: the Python standard library (for example `heapq`, `dict`, `math`).
- Not allowed: `numpy` and `torch`. Their containers would hide the allocation rules this exercise
  is about. Do not use a real serving library.

## Done looks like

Public examples pass. Hidden tests replay seeded random sequences of allocate, append, free and
block-table reads, and compare every step with an independent model of the same rules (lowest-id
reuse, exact block counts, the atomic failure on exhaustion, and the copy returned by
`block_table`). They also check the error class for each misuse, the offset arithmetic for tokens
that start a new block, and that every sequence's last block holds fewer than `block_size` wasted
slots. Invalid constructor arguments raise `ValueError`.
