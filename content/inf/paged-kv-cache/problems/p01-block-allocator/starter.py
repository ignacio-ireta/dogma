from __future__ import annotations


class OutOfBlocksError(RuntimeError):
    """Raised when the pool has too few free blocks for an allocation or a new block."""


class BlockAllocator:
    """Fixed pool of KV-cache blocks with a block table per sequence (see problem.md).

    Physical block ids are 0..num_blocks-1. A new block is always the lowest free id.
    Failed operations raise and leave the allocator unchanged.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        """Create a pool of num_blocks blocks, each with block_size token slots.

        Both must be ints (not bool) and at least 1, otherwise raise ValueError.
        """
        raise NotImplementedError

    @property
    def num_free_blocks(self) -> int:
        """Number of physical blocks not held by any sequence."""
        raise NotImplementedError

    def allocate(self, seq_id: int, n_tokens: int) -> None:
        """Start seq_id with n_tokens tokens, taking ceil(n_tokens / block_size) blocks.

        Raise ValueError if seq_id is active or n_tokens is negative or not an int.
        Raise OutOfBlocksError, changing nothing, if there are too few free blocks.
        """
        raise NotImplementedError

    def append_token(self, seq_id: int) -> tuple[int, int]:
        """Store one more token for seq_id and return (physical_block, offset) for it.

        A new block is taken from the pool when the token starts one. Raise KeyError for an
        unknown seq_id and OutOfBlocksError, changing nothing, if a new block is needed and
        none is free.
        """
        raise NotImplementedError

    def num_tokens(self, seq_id: int) -> int:
        """Tokens stored for seq_id. Raise KeyError for an unknown seq_id."""
        raise NotImplementedError

    def block_table(self, seq_id: int) -> list[int]:
        """Return a new list of physical block ids in logical order. KeyError if unknown."""
        raise NotImplementedError

    def free(self, seq_id: int) -> None:
        """Return all blocks of seq_id to the pool and forget it. KeyError if unknown."""
        raise NotImplementedError
