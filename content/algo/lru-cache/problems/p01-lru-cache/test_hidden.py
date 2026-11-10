"""Hidden tests: an OrderedDict model as the oracle over seeded operation sequences, plus the
edge families and a 100,000-operation time box.

Every failure names the kind of mistake. No failure message carries a hidden key, value or answer.
"""

from __future__ import annotations

import random
import signal
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import LRUCache

LARGE_OPERATIONS = 100_000
LARGE_CAPACITY = 100_000
# The reference finishes the whole large sequence, model included, in about 0.2 s on a laptop.
# The budget is 25 times that, so a loaded runner does not turn a correct cache into a failure.
LARGE_BUDGET_S = 5.0


class _TooSlow(BaseException):
    """Raised inside the learner's code when the time box expires."""


@contextmanager
def _time_box(seconds: float) -> Iterator[None]:
    """Interrupt running code after ``seconds``. Only the main thread can use signals."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(
        signal, "setitimer"
    ):
        yield
        return

    def _expired(signum: int, frame: Any) -> None:
        raise _TooSlow

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _check(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


class _Model:
    """Independent oracle: an OrderedDict kept in recency order, least recent first."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key: Any) -> Any | None:
        if key not in self.data:
            return None
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key: Any, value: Any) -> None:
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = value
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)


def _get_mistake(got: Any, expected: Any) -> str | None:
    if got == expected:
        return None
    if expected is None:
        return (
            "a key that is not stored (or was evicted) still answered: remove the victim from "
            "the lookup map as well as from the list, and make a miss return None"
        )
    if got is None:
        return (
            "a stored key answered as a miss: the eviction removed the wrong entry, or a get "
            "lost an entry; check that a get hit refreshes recency"
        )
    return "a hit returned a stale value: a put of an existing key must overwrite the stored value"


def _learner(call: Any, what: str) -> Any:
    """Run one learner call. A crash becomes a named failure, not a raw traceback."""
    try:
        return call()
    except Exception as exc:
        pytest.fail(
            f"your cache raised {type(exc).__name__} during {what}: keep the lookup map and the "
            "recency list in step on every put, eviction and get"
        )


def _step(cache: LRUCache, model: _Model, operation: tuple[Any, ...]) -> str | None:
    """Apply one operation to the learner and the model. Return the first mistake, or None."""
    if operation[0] == "get":
        key = operation[1]
        got = _learner(lambda: cache.get(key), "get")
        mistake = _get_mistake(got, model.get(key))
    else:
        _, key, value = operation
        _learner(lambda: cache.put(key, value), "put")
        model.put(key, value)
        mistake = None
    if mistake is None and _learner(lambda: len(cache), "len") != len(model.data):
        mistake = (
            "len() disagrees with the number of stored entries: count every stored entry once, "
            "and never let the size pass the capacity"
        )
    return mistake


def _run_sequence(capacity: int, operations: list[tuple[Any, ...]]) -> None:
    """Apply the same operations to the learner and the model, and compare after each one."""
    cache = LRUCache(capacity)
    model = _Model(capacity)
    for operation in operations:
        mistake = _step(cache, model, operation)
        if mistake is not None:
            pytest.fail(mistake)


def _random_operations(rng: random.Random, count: int, keyspace: int) -> list[tuple[Any, ...]]:
    operations: list[tuple[Any, ...]] = []
    for step in range(count):
        key = rng.randrange(keyspace)
        if rng.random() < 0.5:
            operations.append(("get", key))
        else:
            operations.append(("put", key, step + 1))  # values are never None
    return operations


@settings(max_examples=80, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_random_sequences_match_the_ordered_dict_model(seed: int) -> None:
    rng = random.Random(seed)  # noqa: S311 -- seeded synthetic operations, not security.
    capacity = rng.randrange(1, 9)
    operations = _random_operations(rng, rng.randrange(1, 300), rng.randrange(2, 3 * capacity + 3))
    _run_sequence(capacity, operations)


@settings(max_examples=80, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_tiny_capacities_with_many_evictions_match_the_model(seed: int) -> None:
    rng = random.Random(seed)  # noqa: S311 -- seeded synthetic operations, not security.
    capacity = rng.randrange(1, 3)
    operations = _random_operations(rng, 200, rng.randrange(2, 6))
    _run_sequence(capacity, operations)


def test_get_miss_returns_none_and_does_not_change_the_eviction_order() -> None:
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    _check(cache.get("zzz") is None, "a miss must return None")
    cache.put("c", 3)
    _check(
        cache.get("a") is None,
        "a get miss changed the order: key a was the least recent entry and must be evicted",
    )
    _check(cache.get("b") == 2, "key b was used more recently than a and must survive")


def test_get_hit_protects_the_entry_from_the_next_eviction() -> None:
    cache = LRUCache(3)
    for key, value in [("p", 1), ("q", 2), ("r", 3)]:
        cache.put(key, value)
    _check(cache.get("p") == 1, "a get hit must return the stored value")
    cache.put("s", 4)
    _check(
        cache.get("q") is None,
        "the least recent entry should be evicted: a get hit must refresh the recency of its key",
    )
    _check(cache.get("p") == 1, "the entry read by get must survive the next eviction")


def test_put_of_existing_key_refreshes_recency_before_the_next_eviction() -> None:
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)
    cache.put("c", 3)
    _check(
        cache.get("b") is None,
        "a put of an existing key must make that key the most recent entry; "
        "key b should be evicted",
    )
    _check(cache.get("a") == 10, "the overwritten entry must keep its new value")


def test_put_of_existing_key_never_evicts_when_the_cache_is_full() -> None:
    cache = LRUCache(1)
    cache.put("a", 1)
    cache.put("a", 2)
    _check(
        len(cache) == 1,
        "overwriting the only key must keep the size at one, with no eviction",
    )
    _check(cache.get("a") == 2, "the overwritten value must be returned")


def test_removed_entry_does_not_come_back_with_its_old_value() -> None:
    cache = LRUCache(1)
    cache.put("a", 1)
    cache.put("b", 2)
    _check(cache.get("a") is None, "an evicted key must answer None")
    cache.put("a", 3)
    _check(
        cache.get("a") == 3,
        "a key stored again after eviction must hold the new value, not the old one",
    )
    _check(len(cache) == 1, "a re-stored key is one entry, not two")


def test_size_never_exceeds_capacity_after_many_distinct_inserts() -> None:
    cache = LRUCache(4)
    for key in range(40):
        cache.put(key, key * 10)
        _check(
            len(cache) <= 4,
            "the size passed the capacity: evict before inserting a new key when the cache is full",
        )
    _check(len(cache) == 4, "after many distinct keys the cache must be exactly full")
    _check(cache.get(39) == 390, "the most recent key must be stored")
    _check(cache.get(0) is None, "an old key must have been evicted")


def _expect_value_error(action: Any, mistake: str) -> None:
    try:
        action()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; instead it raised {type(exc).__name__}")
    pytest.fail(mistake)


@pytest.mark.parametrize(
    "capacity",
    [0, -3, 1.0, True, None, "2"],
    ids=["zero", "negative", "float", "bool", "none", "string"],
)
def test_invalid_capacity_raises_value_error(capacity: Any) -> None:
    _expect_value_error(
        lambda: LRUCache(capacity),
        "capacity must be a Python int of at least 1: reject zero, negatives, floats, bool "
        "(True is an int subclass), None and strings with ValueError",
    )


def test_none_value_is_rejected_with_value_error() -> None:
    cache = LRUCache(2)
    _expect_value_error(
        lambda: cache.put("k", None),
        "a None value must raise ValueError: None is the miss result of get",
    )


def test_large_sequence_finishes_within_the_time_limit() -> None:
    rng = random.Random(7)  # noqa: S311 -- seeded synthetic operations, not security.
    # Fill the cache first, so every later new key forces an eviction. A cache that scans its
    # entries on each operation cannot finish a cache this size inside the budget.
    operations: list[tuple[Any, ...]] = [("put", key, key + 1) for key in range(LARGE_CAPACITY)]
    operations += _random_operations(rng, LARGE_OPERATIONS, 2 * LARGE_CAPACITY)
    cache = LRUCache(LARGE_CAPACITY)
    model = _Model(LARGE_CAPACITY)
    try:
        with _time_box(LARGE_BUDGET_S):
            for operation in operations:
                mistake = _step(cache, model, operation)
                if mistake is not None:
                    pytest.fail(mistake)
    except _TooSlow:
        pass  # reported below, outside this except block, so no chained exception is shown
    else:
        return
    pytest.fail(
        "too slow on a long sequence: a scan over the stored entries on every operation is "
        "the usual cause; get and put must touch a fixed number of nodes"
    )


def test_stored_falsy_values_are_returned_as_stored_not_as_misses() -> None:
    cache = LRUCache(2)
    cache.put("zero", 0)
    cache.put("empty", "")
    _check(
        cache.get("zero") is not None,
        "a get hit must return the stored falsy value, not None: that is the miss result",
    )
    _check(cache.get("zero") == 0, "a get hit must return the stored value unchanged")
    _check(
        cache.get("empty") == "",
        "a get hit must return a stored empty string unchanged, not None",
    )


def test_list_value_is_stored_and_returned_unchanged() -> None:
    cache = LRUCache(2)
    _learner(lambda: cache.put("pair", [1, 2]), "a put of a list value")
    _check(
        cache.get("pair") == [1, 2],
        "a list is a valid value: put must store it and get must return it unchanged",
    )
