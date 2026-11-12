"""Hidden tests: exact fits, weights and reserve, windows, validation, and an independent search.

Failure messages name the kind of mistake. They never print hidden configs or expected values.
"""

from __future__ import annotations

import copy
import functools
import random
from typing import Any

import pytest
from submission import max_batch_size

from dogma_data import load_text

BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}


def _same(got: Any, expected: int, mistake: str) -> None:
    if type(got) is not int:
        pytest.fail("return a Python int number of sequences, not a float or another type")
    if got != expected:
        pytest.fail(mistake)


def _per_sequence(config: dict[str, Any], seq_len: int, dtype: str) -> int:
    """An independent computation of one sequence's cache bytes."""
    window = config.get("sliding_window")
    tokens = seq_len if window is None else min(seq_len, window)
    kv_bytes = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"]
    return kv_bytes * tokens * BYTES[dtype]


def _search_oracle(per: int, weights: int, memory: int, reserve: int) -> int:
    """Binary search over batch sizes: the largest B whose total fits, with no floor division."""

    def fits(batch: int) -> bool:
        return weights + reserve + batch * per <= memory

    if not fits(0):
        return 0
    low, high = 0, 1
    while fits(high):
        high *= 2
    while high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low


def _expect_error(exc_type: type[Exception], call: Any, mistake: str) -> None:
    try:
        call()
    except exc_type:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; instead it raised {type(exc).__name__}")
    pytest.fail(mistake)


MHA = {"num_layers": 32, "num_kv_heads": 32, "head_dim": 128}
GQA = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "sliding_window": 4096}


@functools.cache
def _passage_lengths() -> tuple[int, ...]:
    """Lengths, in words, of real passages: one line of a novel, or its first k lines.

    A sequence length in the seeded test is the length of one such passage from Frankenstein,
    Pride and Prejudice or Walden, a word standing for a token. Lengths from 1 to 400 words
    are kept; the shortest real line is a single word.
    """
    lengths = set()
    for name in ("gutenberg_frankenstein", "gutenberg_pride_prejudice", "gutenberg_walden"):
        total = 0
        for line in load_text(name).splitlines():
            lengths.add(len(line.split()))
            total += len(line.split())
            lengths.add(total)
    return tuple(sorted(length for length in lengths if 1 <= length <= 400))


def test_budget_that_fits_exactly_counts_the_last_sequence() -> None:
    per = _per_sequence(MHA, 1000, "fp16")
    memory = 1_000 + 3 * per
    _same(
        max_batch_size(MHA, 1000, "fp16", 1_000, memory),
        3,
        "an exact fit must count: use <= when comparing the total with memory_bytes",
    )


def test_one_byte_short_of_the_last_sequence_loses_it() -> None:
    per = _per_sequence(MHA, 1000, "fp16")
    memory = 1_000 + 3 * per - 1
    _same(
        max_batch_size(MHA, 1000, "fp16", 1_000, memory),
        2,
        "one byte short of a whole sequence must not admit it: a floor division is needed",
    )


def test_partial_sequence_never_rounds_up() -> None:
    per = _per_sequence(MHA, 1000, "fp16")
    _same(
        max_batch_size(MHA, 1000, "fp16", 0, 5 * per - 1),
        4,
        "rounding up admits a sequence that does not fit: the answer is a floor",
    )


def test_reserve_comes_out_of_the_budget() -> None:
    per = _per_sequence(MHA, 512, "bf16")
    _same(
        max_batch_size(MHA, 512, "bf16", 100, 100 + 10 * per + 50, reserve_bytes=50),
        10,
        "the reserve must be subtracted with the weights, not ignored",
    )


def test_weights_exactly_equal_to_memory_leave_no_room() -> None:
    _same(
        max_batch_size(MHA, 64, "fp16", 5_000, 5_000),
        0,
        "with no budget left no sequence fits: the answer is 0, not 1",
    )


def test_negative_budget_returns_zero_not_a_negative_count() -> None:
    _same(
        max_batch_size(MHA, 64, "fp16", 9_000, 5_000),
        0,
        "weights larger than memory must give 0; a floor of a negative budget is negative",
    )


def test_one_sequence_too_big_for_the_budget_gives_zero() -> None:
    per = _per_sequence(MHA, 4096, "fp16")
    _same(
        max_batch_size(MHA, 4096, "fp16", 0, per - 1),
        0,
        "a single sequence that does not fit must give 0",
    )


def test_zero_weight_bytes_is_allowed() -> None:
    per = _per_sequence(MHA, 16, "fp8")
    _same(
        max_batch_size(MHA, 16, "fp8", 0, 7 * per),
        7,
        "zero weight bytes is a valid budget and must count the whole sequences",
    )


def test_context_beyond_the_window_does_not_shrink_the_batch_further() -> None:
    at_window = max_batch_size(GQA, 4096, "fp16", 14_000_000_000, 80 * 2**30)
    beyond = max_batch_size(GQA, 32768, "fp16", 14_000_000_000, 80 * 2**30)
    _same(
        beyond,
        at_window,
        "past the window a sequence costs the same as the window, so the batch must not change",
    )


def test_window_longer_than_the_context_uses_the_context() -> None:
    cfg = {"num_layers": 4, "num_kv_heads": 2, "head_dim": 16, "sliding_window": 10**6}
    plain = {"num_layers": 4, "num_kv_heads": 2, "head_dim": 16}
    _same(
        max_batch_size(cfg, 300, "fp16", 10, 10**7),
        max_batch_size(plain, 300, "fp16", 10, 10**7),
        "a window longer than the context must not change the per-sequence bytes",
    )


def test_seeded_random_budgets_match_an_independent_search() -> None:
    rng = random.Random(7)  # noqa: S311 -- seeded synthetic budgets, not security.
    dtypes = sorted(BYTES)
    for _ in range(250):
        config: dict[str, Any] = {
            "num_layers": rng.randint(1, 12),
            "num_kv_heads": rng.randint(1, 8),
            "head_dim": rng.choice([8, 16, 32]),
        }
        if rng.random() < 0.5:
            config["sliding_window"] = rng.randint(1, 300)
        seq_len = rng.choice(_passage_lengths())
        dtype = rng.choice(dtypes)
        weights = rng.randint(0, 5_000)
        reserve = rng.randint(0, 500)
        memory = rng.randint(0, 60_000)
        per = _per_sequence(config, seq_len, dtype)
        expected = _search_oracle(per, weights, memory, reserve)
        _same(
            max_batch_size(config, seq_len, dtype, weights, memory, reserve_bytes=reserve),
            expected,
            "the batch disagrees with the search: subtract weights and reserve before the "
            "division, use the window cap, and the per-element size of the dtype",
        )


def test_very_large_budgets_stay_exact_beyond_float_precision() -> None:
    per = _per_sequence(MHA, 2**20, "fp8")
    memory = 2**70 + 12345
    weights = 2**60
    expected = (memory - weights) // per
    assert expected > 2**53 // per, "test setup: the budget must span well past float precision"
    _same(
        max_batch_size(MHA, 2**20, "fp8", weights, memory),
        expected,
        "the batch lost precision on a huge budget: keep integer division, not float",
    )


def test_config_is_not_mutated() -> None:
    cfg = copy.deepcopy(GQA)
    before = copy.deepcopy(cfg)
    max_batch_size(cfg, 9000, "bf16", 1, 10**12)
    if cfg != before:
        pytest.fail("the config dict was changed: read the keys without writing to the input")


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((MHA, 4.0, "fp16", 0, 10), {}),
        ((MHA, True, "fp16", 0, 10), {}),
        ((MHA, 8, "fp16", 0.0, 10), {}),
        ((MHA, 8, "fp16", 0, True), {}),
        ((MHA, 8, "fp16", 0, 10), {"reserve_bytes": 1.5}),
        ((MHA, 8, "fp16", None, 10), {}),
    ],
    ids=[
        "float-seq-len",
        "bool-seq-len",
        "float-weights",
        "bool-memory",
        "float-reserve",
        "none-weights",
    ],
)
def test_non_int_budget_values_raise_type_error(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    _expect_error(
        TypeError,
        lambda: max_batch_size(*args, **kwargs),
        "a budget or sequence value that is not a Python int must raise TypeError",
    )


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((MHA, 0, "fp16", 0, 10), {}),
        ((MHA, -4, "fp16", 0, 10), {}),
        ((MHA, 8, "fp16", -1, 10), {}),
        ((MHA, 8, "fp16", 0, -10), {}),
        ((MHA, 8, "fp16", 0, 10), {"reserve_bytes": -1}),
        ((MHA, 8, "int4", 0, 10), {}),
        (({"num_layers": 1, "num_kv_heads": 1}, 8, "fp16", 0, 10), {}),
    ],
    ids=[
        "zero-seq-len",
        "negative-seq-len",
        "negative-weights",
        "negative-memory",
        "negative-reserve",
        "unknown-dtype",
        "missing-head-dim",
    ],
)
def test_out_of_range_or_unknown_values_raise_value_error(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    _expect_error(
        ValueError,
        lambda: max_batch_size(*args, **kwargs),
        "a value out of range, a missing key or an unknown dtype must raise ValueError",
    )


def test_config_that_is_not_a_dict_raises_type_error() -> None:
    # The validation is the same as the KV-cache problem, where a config that is not a dict is a
    # TypeError.
    _expect_error(
        TypeError,
        lambda: max_batch_size([], 8, "fp16", 0, 10),  # type: ignore[arg-type]
        "a config that is not a dict must raise TypeError",
    )


def test_zero_memory_bytes_is_valid_and_holds_no_sequence() -> None:
    # The contract allows memory_bytes of 0; an empty device holds no sequence.
    try:
        got = max_batch_size(MHA, 8, "fp16", 0, 0)
    except ValueError:
        pytest.fail("zero memory_bytes is valid and must return 0, not raise ValueError")
    _same(got, 0, "a device with zero memory must hold zero sequences")


def test_budget_one_byte_below_zero_returns_zero_not_a_negative_count() -> None:
    # Weights one byte above memory leave a budget of -1 bytes. The answer is 0, never negative.
    _same(
        max_batch_size(MHA, 8, "fp16", 11, 10),
        0,
        "weights larger than memory must give 0 sequences, not a negative count",
    )


def test_negative_memory_bytes_raises_value_error() -> None:
    _expect_error(
        ValueError,
        lambda: max_batch_size(MHA, 8, "fp16", 0, -1),
        "a negative memory_bytes is out of range and must raise ValueError",
    )


def test_zero_window_raises_value_error() -> None:
    # The contract says the window is None or at least 1, so a window of 0 must raise ValueError.
    cfg = dict(MHA, sliding_window=0)
    _expect_error(
        ValueError,
        lambda: max_batch_size(cfg, 8, "fp16", 0, 10**6),
        "a sliding window below 1 is out of range and must raise ValueError",
    )


def test_dtype_names_are_case_sensitive_and_upper_case_raises() -> None:
    # The dtype names are the lower-case strings fp32, fp16, bf16 and fp8. "FP16" is not one of
    # them, so it is an unknown dtype and must raise ValueError.
    cfg = {"num_layers": 32, "num_kv_heads": 32, "head_dim": 128}
    _expect_error(
        ValueError,
        lambda: max_batch_size(cfg, 8, "FP16", 0, 10**6),
        "an upper-case dtype name is not one of the four names and must raise ValueError",
    )
