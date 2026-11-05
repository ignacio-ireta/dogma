from __future__ import annotations

from submission import BlockAllocator


def test_allocation_takes_lowest_free_blocks_and_appends_in_place():
    a = BlockAllocator(8, 4)
    a.allocate(0, 10)
    assert a.block_table(0) == [0, 1, 2], "allocation mistake: take ceil(10 / 4) = 3 lowest ids"
    a.allocate(1, 5)
    assert a.block_table(1) == [3, 4], (
        "allocation mistake: a second sequence starts after the first"
    )
    assert a.append_token(0) == (2, 2), "offset mistake: position 10 is offset 2 of block 2"
    assert a.append_token(0) == (2, 3), "offset mistake: position 11 is offset 3 of block 2"
    assert a.append_token(0) == (5, 0), "growth mistake: position 12 starts a new block at offset 0"


def test_freed_blocks_return_to_the_pool_and_lowest_id_is_reused():
    a = BlockAllocator(8, 4)
    a.allocate(0, 10)
    a.allocate(1, 5)
    a.free(1)
    a.allocate(2, 4)
    assert a.block_table(2) == [3], "reuse mistake: freed ids must be reused lowest first"


def test_zero_token_sequence_takes_a_block_on_its_first_append():
    a = BlockAllocator(2, 4)
    a.allocate(7, 0)
    assert a.block_table(7) == [], "empty sequence must hold no blocks"
    assert a.append_token(7) == (0, 0), "first token must take the first free block at offset 0"
