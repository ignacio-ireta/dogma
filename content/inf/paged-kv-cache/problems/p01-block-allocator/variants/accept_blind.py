from __future__ import annotations

import heapq


class OutOfBlocksError(RuntimeError):
    """Raised when the pool has too few free blocks for an allocation or a new block."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class BlockAllocator:
    """Fixed pool of KV-cache blocks with a block table per sequence (see problem.md).

    Physical block ids are 0..num_blocks-1. A new block is always the lowest free id.
    Failed operations raise and leave the allocator unchanged.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        """Create a pool of num_blocks blocks, each with block_size token slots.

        Both must be ints (not bool) and at least 1, otherwise raise ValueError.
        """
        if not _is_int(num_blocks) or num_blocks < 1:
            raise ValueError("num_blocks must be an int of at least 1")
        if not _is_int(block_size) or block_size < 1:
            raise ValueError("block_size must be an int of at least 1")
        self._block_size = block_size
        self._free: list[int] = list(range(num_blocks))  # sorted, so a valid heap
        self._seqs: dict[int, dict[str, object]] = {}

    @property
    def num_free_blocks(self) -> int:
        """Number of physical blocks not held by any sequence."""
        return len(self._free)

    def _take(self, count: int) -> list[int]:
        return [heapq.heappop(self._free) for _ in range(count)]

    def allocate(self, seq_id: int, n_tokens: int) -> None:
        """Start seq_id with n_tokens tokens, taking ceil(n_tokens / block_size) blocks.

        Raise ValueError if seq_id is active or n_tokens is negative or not an int.
        Raise OutOfBlocksError, changing nothing, if there are too few free blocks.
        """
        if seq_id in self._seqs:
            raise ValueError(f"sequence {seq_id} is already active")
        if not _is_int(n_tokens) or n_tokens < 0:
            raise ValueError("n_tokens must be a non-negative int")
        needed = -(-n_tokens // self._block_size)
        if needed > len(self._free):
            raise OutOfBlocksError(f"need {needed} blocks, {len(self._free)} free")
        self._seqs[seq_id] = {"tokens": n_tokens, "blocks": self._take(needed)}

    def append_token(self, seq_id: int) -> tuple[int, int]:
        """Store one more token for seq_id and return (physical_block, offset) for it.

        A new block is taken from the pool when the token starts one. Raise KeyError for an
        unknown seq_id and OutOfBlocksError, changing nothing, if a new block is needed and
        none is free.
        """
        seq = self._seqs[seq_id]
        position = seq["tokens"]
        offset = position % self._block_size
        blocks: list[int] = seq["blocks"]  # type: ignore[assignment]
        if offset == 0:
            if not self._free:
                raise OutOfBlocksError("no free block for a new token")
            blocks.append(heapq.heappop(self._free))
        seq["tokens"] = position + 1
        return blocks[-1], offset

    def num_tokens(self, seq_id: int) -> int:
        """Tokens stored for seq_id. Raise KeyError for an unknown seq_id."""
        return self._seqs[seq_id]["tokens"]  # type: ignore[return-value]

    def block_table(self, seq_id: int) -> list[int]:
        """Return a new list of physical block ids in logical order. KeyError if unknown."""
        return list(self._seqs[seq_id]["blocks"])  # type: ignore[arg-type]

    def free(self, seq_id: int) -> None:
        """Return all blocks of seq_id to the pool and forget it. KeyError if unknown."""
        seq = self._seqs.pop(seq_id)
        for block in seq["blocks"]:  # type: ignore[union-attr]
            heapq.heappush(self._free, block)
