"""Hidden tests: an oracle comparison with bisect.bisect_left, plus the edge cases.

Each failure message names the kind of mistake it points to. None of them carries an input value or
an expected answer.
"""

from __future__ import annotations

import bisect
import random
from collections.abc import Sequence
from typing import Any

from submission import lower_bound  # type: ignore[import-not-found]


class _ReadBudgetExceeded(BaseException):
    """Raised inside the learner's call when it reads more elements than the budget allows.

    It derives from BaseException, so a broad ``except Exception`` in the learner's code cannot
    swallow it.
    """


class _CountingSequence(Sequence[Any]):
    """A read-only sequence that counts the elements it hands out."""

    def __init__(self, values: list[Any], budget: int) -> None:
        self._values = values
        self._budget = budget
        self.reads = 0

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name: str) -> Any:
        # Only a learner who tries to change the input reaches this: the sequence is read-only.
        raise AssertionError(
            f"the input is read-only: do not call {name} on sorted_values, and do not change it"
        )

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            picked = self._values[index]
            self._charge(len(picked))
            return picked
        self._charge(1)
        return self._values[index]

    def _charge(self, count: int) -> None:
        self.reads += count
        if self.reads > self._budget:
            raise _ReadBudgetExceeded()


def _check(values: list[Any], target: Any) -> None:
    got = lower_bound(values, target)
    want = bisect.bisect_left(values, target)
    assert got == want, (
        "result differs from the reference lower bound: check the midpoint, which branch moves "
        "lo or hi, and the upper end of the starting range"
    )


def test_matches_bisect_bisect_left_on_random_integer_lists_with_duplicates() -> None:
    rng = random.Random(0)  # noqa: S311 - seeded, so the test is deterministic
    for _ in range(400):
        values = sorted(rng.randint(-6, 6) for _ in range(rng.randint(0, 40)))
        _check(values, rng.randint(-7, 7))


def test_matches_bisect_bisect_left_on_random_float_lists() -> None:
    rng = random.Random(1)  # noqa: S311 - seeded, so the test is deterministic
    for _ in range(300):
        # Rounding to one decimal creates duplicates, and targets from the same grid hit them.
        values = sorted(round(rng.uniform(-10.0, 10.0), 1) for _ in range(rng.randint(0, 30)))
        target = round(rng.uniform(-11.0, 11.0), 1)
        _check(values, target)


def test_duplicates_of_the_target_return_the_first_index_of_their_run() -> None:
    got = lower_bound([1, 2, 2, 2, 2, 3], 2)
    assert got == 1, (
        "returned past the run of duplicates (the upper bound): the lower bound is the first index "
        "whose value is at least the target"
    )
    assert lower_bound([4, 4, 4], 4) == 0, (
        "all elements equal the target: return the first index of the run, not the last one"
    )


def test_target_below_every_element_returns_zero() -> None:
    assert lower_bound([5, 6, 7], -100) == 0, (
        "target below every element: the answer is the first index, not a later one"
    )
    assert lower_bound([5, 6, 7], 5) == 0, (
        "target equal to the first element: the answer is the first index, not a later one"
    )


def test_target_above_every_element_returns_len_not_len_minus_one_or_minus_one() -> None:
    got = lower_bound([1, 2, 3], 4)
    assert got == 3, "target above every element: return len(sorted_values), not len - 1 and not -1"
    assert lower_bound([1, 2, 3], 3) == 2, (
        "target equal to the last element: the answer is that element's own index, not one past it"
    )


def test_empty_list_returns_zero_for_any_target() -> None:
    for target in (0, -1, 10, 0.5):
        assert lower_bound([], target) == 0, (
            "empty list: the answer is len(sorted_values) whatever the target is"
        )


def test_negative_and_float_values_are_compared_numerically() -> None:
    values = [-7.5, -3, -3, 0.0, 0.25, 2.5, 9]
    cases = [(-8, 0), (-3, 1), (-3.5, 1), (0.1, 4), (0.25, 4), (2.6, 6), (10, 7)]
    for target, want in cases:
        assert lower_bound(values, target) == want, (
            "negative or float values are compared wrongly: compare each element with the target "
            "as numbers, in sorted order"
        )


def test_float_target_is_not_truncated_to_an_integer() -> None:
    assert lower_bound([0.5, 1.2, 1.5, 2.7], 1.3) == 2, (
        "a float target was truncated or rounded: compare with the exact value of the target"
    )


def test_result_is_a_plain_int_between_zero_and_len() -> None:
    values = [1, 2, 3]
    for target in (0, 2, 10):
        result = lower_bound(values, target)
        assert type(result) is int, "return a plain int index, not a float or another type"
        assert 0 <= result <= len(values), (
            "the index must lie between 0 and len(sorted_values) inclusive"
        )


def test_input_list_is_not_mutated() -> None:
    values = [1, 3, 3, 8]
    snapshot = list(values)
    lower_bound(values, 3)
    lower_bound(values, 5)
    lower_bound(values, 0)
    assert values == snapshot, (
        "the input list was changed: do not append, sort, pop or otherwise modify sorted_values"
    )


def test_reads_a_logarithmic_number_of_elements_on_a_large_list() -> None:
    # 65,536 sorted values. A binary search reads about 17 of them; a linear scan reads far more
    # than the budget below, and is stopped as soon as it passes it.
    values = [2 * i for i in range(65_536)]
    budget = 4 * len(values).bit_length()
    for target in (0, 1, 65_536, 2 * len(values) - 1):
        seq = _CountingSequence(values, budget)
        try:
            got = lower_bound(seq, target)
        except _ReadBudgetExceeded:
            raise AssertionError(
                "lower_bound reads too many elements: a binary search reads about log2(n) of them, "
                "a linear scan reads most of them"
            ) from None
        assert got == bisect.bisect_left(values, target), (
            "result differs from the reference on a large list: check the midpoint and the range"
        )


def test_tuple_input_gives_the_same_answer_as_a_list() -> None:
    # The Contract says "a list (or any sequence)", so a tuple is a valid input.
    values = (1, 3, 5)
    # Target above every element: the answer is len(values) = 3, the same as for a list.
    assert lower_bound(values, 9) == 3, (
        "tuple input, target above every element: return len(sorted_values), as for a list"
    )
    # Target between 3 and 5: the answer is index 2, the same as for a list.
    assert lower_bound(values, 4) == 2, (
        "tuple input, target between two elements: return the index of the next larger one"
    )
    # Target below every element: the answer is index 0.
    assert lower_bound(values, 0) == 0, (
        "tuple input, target below every element: the answer is index 0"
    )
