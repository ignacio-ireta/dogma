from __future__ import annotations

import functools
import random
import re
from dataclasses import dataclass, field

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import BlockAllocator, OutOfBlocksError

from dogma_data import load_frame


def _raises(expected, call, mistake):
    try:
        call()
    except expected:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


def outcome(call, *args):
    """Run a learner call; return ('ok', value) or ('err', exception class name)."""
    try:
        return ("ok", call(*args))
    except Exception as exc:  # the class name is what the contract checks
        return ("err", type(exc).__name__)


@dataclass
class _Seq:
    tokens: int
    table: list[int] = field(default_factory=list)


class Model:
    """Independent model of the rules: a sorted list of free ids and plain dict state."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.free = list(range(num_blocks))
        self.bs = block_size
        self.seqs: dict[int, _Seq] = {}

    def allocate(self, sid: int, n: int) -> tuple[str, object]:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            return ("err", "ValueError")
        if sid in self.seqs:
            return ("err", "ValueError")
        need = -(-n // self.bs)
        if need > len(self.free):
            return ("err", "OutOfBlocksError")
        self.free.sort()
        table = self.free[:need]
        del self.free[:need]
        self.seqs[sid] = _Seq(tokens=n, table=table)
        return ("ok", None)

    def append(self, sid: int) -> tuple[str, object]:
        if sid not in self.seqs:
            return ("err", "KeyError")
        seq = self.seqs[sid]
        pos = seq.tokens
        if pos % self.bs == 0:
            if not self.free:
                return ("err", "OutOfBlocksError")
            self.free.sort()
            seq.table.append(self.free.pop(0))
        seq.tokens = pos + 1
        return ("ok", (seq.table[pos // self.bs], pos % self.bs))

    def free_seq(self, sid: int) -> tuple[str, object]:
        if sid not in self.seqs:
            return ("err", "KeyError")
        self.free.extend(self.seqs.pop(sid).table)
        return ("ok", None)


def same_outcome(learner, expected, where: str) -> None:
    if learner[0] != expected[0] or (learner[0] == "err" and learner[1] != expected[1]):
        raise AssertionError(
            f"{where}: expected {expected[1] if expected[0] == 'err' else 'success'}, "
            f"got {learner[1] if learner[0] == 'err' else 'success'}"
        )
    if learner[0] == "ok" and learner[1] != expected[1]:
        raise AssertionError(f"{where}: wrong return value (check the offset or the block id)")


@functools.cache
def _request_sizes() -> tuple[int, ...]:
    """Sizes of real requests, in the small units of this allocator.

    For each GSM8K problem: the number of sentences of its question and the number of
    calculations in its worked answer. Both stand for token counts. They run from 0 (an
    answer with no calculation) to 8, so a request takes from no block to several.
    """
    table = load_frame("gsm8k")
    sizes = [len(re.findall(r"[.?!](?:\s|$)", question)) for question in table["question"]]
    sizes += [answer.count("<<") for answer in table["answer"]]
    return tuple(sizes)


@pytest.mark.parametrize(
    "seed", list(range(30)), ids=[chr(97 + i // 26) + chr(97 + i % 26) for i in range(30)]
)
def test_random_operations_match_the_independent_model(seed):
    rng = random.Random(seed)  # noqa: S311 - seeded, reproducible model inputs
    num_blocks, block_size = rng.randint(1, 9), rng.randint(1, 4)
    learner = BlockAllocator(num_blocks, block_size)
    model = Model(num_blocks, block_size)
    for step in range(200):
        sid = rng.randint(0, 4)
        op = rng.choice(["allocate", "append", "append", "free", "read"])
        where = f"seed {seed}, step {step}, {op}"
        if op == "allocate":
            n = rng.choice(_request_sizes())
            same_outcome(outcome(learner.allocate, sid, n), model.allocate(sid, n), where)
        elif op == "append":
            same_outcome(outcome(learner.append_token, sid), model.append(sid), where)
        elif op == "free":
            same_outcome(outcome(learner.free, sid), model.free_seq(sid), where)
        else:
            got = outcome(learner.block_table, sid)
            want = ("ok", model.seqs[sid].table) if sid in model.seqs else ("err", "KeyError")
            same_outcome(got, want, where + ": block table")
        if learner.num_free_blocks != len(model.free):
            raise AssertionError(f"{where}: free block count drifted from the model")
        for other, seq in model.seqs.items():
            if learner.block_table(other) != seq.table:
                raise AssertionError(f"{where}: block table differs (check lowest-id reuse)")
            if learner.num_tokens(other) != seq.tokens:
                raise AssertionError(f"{where}: token count differs from the model")


def test_block_table_returns_a_copy_that_cannot_corrupt_the_allocator():
    a = BlockAllocator(4, 2)
    a.allocate(0, 3)
    table = a.block_table(0)
    table.append(99)
    table[0] = 77
    assert a.block_table(0) == [0, 1], "block_table must return a copy, not the live table"


def test_exhaustion_on_allocate_is_named_and_changes_nothing():
    a = BlockAllocator(3, 4)
    a.allocate(0, 8)
    _raises(
        OutOfBlocksError,
        lambda: a.allocate(1, 9),
        "allocate with too few free blocks must raise OutOfBlocksError",
    )
    assert a.num_free_blocks == 1, "failed allocation must not take any blocks"
    _raises(
        KeyError,
        lambda: a.block_table(1),
        "block_table of an unknown sequence must raise KeyError",
    )


def test_exhaustion_on_append_is_named_and_changes_nothing():
    a = BlockAllocator(2, 2)
    a.allocate(0, 2)
    a.allocate(1, 2)
    _raises(
        OutOfBlocksError,
        lambda: a.append_token(0),
        "append with no free block must raise OutOfBlocksError",
    )
    assert a.num_tokens(0) == 2, "failed append must not store the token"
    assert a.block_table(0) == [0], "failed append must not change the block table"


def test_offsets_wrap_exactly_at_block_boundaries():
    a = BlockAllocator(4, 3)
    a.allocate(5, 0)
    got = [a.append_token(5) for _ in range(7)]
    want = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)]
    if got != want:
        raise AssertionError(
            "offset mistake: offset is position % block_size, block is position // block_size"
        )


def test_misuse_raises_the_named_error_for_each_operation():
    a = BlockAllocator(4, 2)
    a.allocate(0, 1)
    checks = [
        (ValueError, lambda: a.allocate(0, 1), "duplicate allocate"),
        (ValueError, lambda: a.allocate(1, -1), "negative token count"),
        (ValueError, lambda: a.allocate(1, True), "bool token count"),
        (KeyError, lambda: a.append_token(9), "append to an unknown sequence"),
        (KeyError, lambda: a.free(9), "free of an unknown sequence"),
        (KeyError, lambda: a.num_tokens(9), "token count of an unknown sequence"),
    ]
    for expected, call, label in checks:
        try:
            call()
        except expected:
            continue
        except Exception as exc:
            got_name = type(exc).__name__
            raise AssertionError(f"{label}: expected {expected.__name__}, got {got_name}") from None
        raise AssertionError(f"{label}: no exception was raised")


@pytest.mark.parametrize(
    "args",
    [(0, 4), (4, 0), (-1, 4), (True, 4), (4, False)],
    ids=["zero-blocks", "zero-block-size", "negative-blocks", "bool-blocks", "bool-block-size"],
)
def test_invalid_pool_sizes_raise_value_error(args):
    _raises(
        ValueError,
        lambda: BlockAllocator(*args),
        "invalid pool size must raise ValueError",
    )


@settings(max_examples=60, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.integers(1, 6), st.integers(1, 200))
def test_last_block_is_the_only_one_with_wasted_slots(block_size, n_tokens):
    a = BlockAllocator(1000, block_size)
    a.allocate(0, n_tokens)
    table = a.block_table(0)
    waste = len(table) * block_size - n_tokens
    if not 0 <= waste < block_size:
        raise AssertionError(
            "fragmentation mistake: only the last block may be partly empty (ceil, not floor)"
        )
    if len(table) != len(set(table)):
        raise AssertionError("one physical block was handed to a sequence twice")
