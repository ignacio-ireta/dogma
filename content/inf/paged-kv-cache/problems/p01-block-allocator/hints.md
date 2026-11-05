## Hint 1 — concept

The KV cache becomes a pool of fixed-size blocks, and each sequence keeps a list of block ids, the
block table. Blocks are taken one at a time, when a token needs a slot, and returned when the
sequence ends. Only the last block of a sequence can be partly empty.

## Hint 2 — formula / approach

A sequence with `n` tokens needs `ceil(n / block_size)` blocks, and token `p` (0-based) lives in
table index `p // block_size` at offset `p % block_size`. A token starts a new block exactly when
`p % block_size == 0`. For reuse, keep the free ids in a min-heap so that the lowest id is always
taken first. Check the free count before you take anything, so a failed request changes nothing.

## Hint 3 — code shape

Keep three structures: the free heap, a dict from sequence id to its block table, and a dict from
sequence id to its token count. In `allocate`, validate arguments, compute the block count, check
the free count, then pop that many ids. In `append_token`, branch on `position % block_size`, then
use `divmod` for the return value. `free` pushes the table back onto the heap and deletes both
entries.
