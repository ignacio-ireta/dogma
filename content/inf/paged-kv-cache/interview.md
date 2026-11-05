# Interview notes: Paged KV cache

## Q1. Why does paging the KV cache allow more concurrent requests?

**60-second answer.** Without paging, each request reserves a region for its maximum length, and most requests end well short of it. That reservation is memory nobody uses, and it caps the batch. A paged cache takes fixed blocks as tokens arrive and returns them at the end. Each sequence wastes at most one partly filled block, and every free block fits any request. The same memory then holds more sequences, which is the batch size.

**Follow-ups.** *What does a block table cost?* One indirection per attention read and a small per-sequence list. *How do you pick a block size?* A smaller block wastes less per sequence and makes the table longer; a larger block does the reverse. The right value depends on the length distribution.

## Q2. What must an allocator guarantee when it runs out of blocks?

**60-second answer.** It must fail without changing anything. A request that needs three blocks and gets two has leaked two blocks and left a sequence half-built, and the bug surfaces later as a wrong offset or a table that is too short. So the allocator checks the free count first, raises a named error, and leaves the pool as it was. Preemption, waiting and rejection belong to the scheduler, not to the allocator.

**Follow-ups.** *Why lowest-id reuse?* It makes runs reproducible, so a test can predict every block id. *Why return a copy of the table?* So a caller cannot corrupt the allocator by mutating what it was given.

## Common mistakes

- Growing the block table on every token instead of only at block boundaries.
- Using floor instead of ceiling for the block count, which loses the partial last block.
- Partly allocating a request and then raising, which leaks blocks.
- Returning the live block table, so callers can change allocator state.
