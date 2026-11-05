"""Public tests: the examples from problem.md, capacity one, the capacity rule and the None rule.

Each failure names the rule that was broken.
"""

from __future__ import annotations

from typing import Any

import pytest
from submission import LRUCache


def _check(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def _raises_value_error(action: Any, mistake: str) -> None:
    try:
        action()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; instead it raised {type(exc).__name__}")
    pytest.fail(mistake)


def test_get_protects_an_entry_so_the_untouched_one_is_evicted() -> None:
    cache = LRUCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    _check(cache.get(1) == "a", "a get hit must return the stored value")
    cache.put(3, "c")
    _check(
        cache.get(2) is None,
        "key 2 should have been evicted: a get on key 1 made it the most recent entry, so key 2 "
        "is the least recent; a get that does not refresh recency evicts key 1 instead",
    )
    _check(cache.get(3) == "c", "the newly stored key must be retrievable")
    _check(
        cache.get(1) == "a",
        "key 1 must survive the eviction: it was used after key 2 was used",
    )


def test_put_of_existing_key_overwrites_the_value_without_growing_the_cache() -> None:
    cache = LRUCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    cache.put(1, "A")
    _check(
        len(cache) == 2,
        "a put of an existing key must not change the size: do not evict or insert a second node",
    )
    _check(cache.get(1) == "A", "a put of an existing key must overwrite the stored value")
    _check(cache.get(2) == "b", "the other entry must keep its value")


def test_capacity_one_keeps_only_the_latest_key() -> None:
    cache = LRUCache(1)
    cache.put("x", 1)
    cache.put("y", 2)
    _check(cache.get("x") is None, "capacity one: the first key must be evicted by the second")
    _check(cache.get("y") == 2, "capacity one: the latest key must be stored")
    _check(len(cache) == 1, "capacity one: the size must stay at one")


def test_capacity_zero_is_rejected_with_value_error() -> None:
    _raises_value_error(
        lambda: LRUCache(0), "capacity 0 must raise ValueError: a cache must hold one"
    )


def test_none_value_is_rejected_with_value_error() -> None:
    cache = LRUCache(2)
    _raises_value_error(
        lambda: cache.put("k", None),
        "a None value must raise ValueError: None is the miss result of get",
    )
