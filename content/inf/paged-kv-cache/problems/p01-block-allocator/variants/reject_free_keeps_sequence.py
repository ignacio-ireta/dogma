"""Wrong on purpose: free returns the blocks but keeps the sequence table."""

from __future__ import annotations

import heapq


class OutOfBlocksError(RuntimeError):
    """Raised when the pool has too few free blocks for an allocation or a new block."""


class BlockAllocator:
    """Fixed pool of KV-cache blocks with a block table per sequence (see problem.md)."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        for name, value in (("num_blocks", num_blocks), ("block_size", block_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int of at least 1")
        self._block_size = block_size
        # A min-heap of free physical ids. heappop always yields the lowest free id, which is the
        # deterministic reuse rule. range(n) is already a valid heap.
        self._free: list[int] = list(range(num_blocks))
        # Per sequence: the block table (logical order) and the token count. The table length is
        # always ceil(tokens / block_size), which is the paging invariant.
        self._tables: dict[int, list[int]] = {}
        self._tokens: dict[int, int] = {}

    @property
    def num_free_blocks(self) -> int:
        """Number of physical blocks not held by any sequence."""
        return len(self._free)

    def allocate(self, seq_id: int, n_tokens: int) -> None:
        """Start seq_id with n_tokens tokens, taking ceil(n_tokens / block_size) blocks."""
        if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or n_tokens < 0:
            raise ValueError("n_tokens must be a non-negative int")
        if seq_id in self._tables:
            raise ValueError(f"sequence {seq_id!r} is already allocated")
        # Ceiling division: a partly filled last block still costs a whole block.
        need = (n_tokens + self._block_size - 1) // self._block_size
        # Check before taking anything, so a failed request leaves the pool unchanged.
        if need > len(self._free):
            raise OutOfBlocksError(f"need {need} blocks, {len(self._free)} free")
        self._tables[seq_id] = [heapq.heappop(self._free) for _ in range(need)]
        self._tokens[seq_id] = n_tokens

    def append_token(self, seq_id: int) -> tuple[int, int]:
        """Store one more token and return (physical_block, offset) for it."""
        table = self._table(seq_id)
        position = self._tokens[seq_id]  # 0-based position of the new token
        if position % self._block_size == 0:
            # The token starts a new block: it is the only point where the table grows.
            if not self._free:
                raise OutOfBlocksError("no free block for a new token")
            table.append(heapq.heappop(self._free))
        self._tokens[seq_id] = position + 1
        block_index, offset = divmod(position, self._block_size)
        return table[block_index], offset

    def num_tokens(self, seq_id: int) -> int:
        """Tokens stored for seq_id."""
        self._table(seq_id)
        return self._tokens[seq_id]

    def block_table(self, seq_id: int) -> list[int]:
        """Return a copy, so callers cannot change the allocator through it."""
        return list(self._table(seq_id))

    def free(self, seq_id: int) -> None:
        """Return all blocks of seq_id to the pool and forget it."""
        for block in self._table(seq_id):
            heapq.heappush(self._free, block)
        del self._tokens[seq_id]

    def _table(self, seq_id: int) -> list[int]:
        """Return the live table of seq_id, or raise KeyError for an unknown sequence."""
        if seq_id not in self._tables:
            raise KeyError(f"unknown sequence {seq_id!r}")
        return self._tables[seq_id]
