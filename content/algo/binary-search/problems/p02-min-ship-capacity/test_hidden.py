"""Hidden tests: a brute-force oracle, minimality checks, invalid input, and a large input.

The oracle is a dynamic-programming count of runs, independent of the greedy count used in a
typical solution. Each failure message names the kind of mistake; none carries an input or an
answer.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from submission import min_ship_capacity  # type: ignore[import-not-found]

INFEASIBLE = 10**9


def _min_runs(weights: Sequence[int], capacity: int) -> int:
    """The fewest consecutive runs whose sums are all at most capacity (dynamic programming)."""
    n = len(weights)
    best = [0] + [INFEASIBLE] * n  # best[i]: fewest runs that cover the first i parcels
    for end in range(1, n + 1):
        total = 0
        for start in range(end, 0, -1):
            total += weights[start - 1]
            if total > capacity:
                break
            best[end] = min(best[end], best[start - 1] + 1)
    return best[n]


def _brute_force(weights: Sequence[int], days: int) -> int:
    """Try every capacity from the heaviest parcel up, and return the first one that fits."""
    for capacity in range(max(weights), sum(weights) + 1):
        if _min_runs(weights, capacity) <= days:
            return capacity
    raise AssertionError("the oracle found no capacity, which cannot happen for days >= 1")


class _ReadBudgetExceeded(BaseException):
    """Raised inside the learner's call when it reads more parcels than the budget allows.

    It derives from BaseException, so a broad ``except Exception`` in the learner's code cannot
    swallow it.
    """


class _CountingSequence(Sequence[Any]):
    """A read-only sequence that counts the parcels it hands out."""

    def __init__(self, values: list[Any], budget: int) -> None:
        self._values = values
        self._budget = budget
        self.reads = 0

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name: str) -> Any:
        # Only a learner who tries to change the input reaches this: the sequence is read-only.
        raise AssertionError(
            f"the input is read-only: do not call {name} on weights, and do not change it"
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


def _expect_value_error(call: Any, mistake: str) -> None:
    """Pass when ``call()`` raises ValueError; otherwise name what went wrong."""
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(
            f"{mistake}: raised {type(exc).__name__} instead of ValueError"
        ) from None
    raise AssertionError(f"{mistake}: no ValueError was raised")


def test_matches_brute_force_over_every_capacity_on_small_random_inputs() -> None:
    rng = random.Random(0)  # noqa: S311 - seeded, so the test is deterministic
    for _ in range(300):
        weights = [rng.randint(1, 9) for _ in range(rng.randint(1, 8))]
        days = rng.randint(1, 9)
        got = min_ship_capacity(list(weights), days)
        assert got == _brute_force(weights, days), (
            "the capacity differs from the brute-force answer: check the search range, the "
            "monotone update rules, and how the day count is built"
        )


def test_answer_is_the_smallest_capacity_that_fits_in_time() -> None:
    fixed = [
        ([3, 1, 4, 1, 5, 9, 2, 6], 3),
        ([7, 7, 7], 2),
        ([10, 1, 1, 1, 1, 1, 1, 1], 2),
        ([2, 2, 2, 2, 2, 2], 4),
    ]
    for weights, days in fixed:
        got = min_ship_capacity(list(weights), days)
        assert _min_runs(weights, got) <= days, (
            "the answer does not ship every parcel within the day limit: check the daily load "
            "and when a new day starts"
        )
        assert _min_runs(weights, got - 1) > days, (
            "the answer is feasible but not the smallest: a smaller capacity also ships in time, "
            "so the search must find the first capacity that works"
        )


def test_one_day_needs_the_total_weight_for_any_parcels() -> None:
    assert min_ship_capacity([3, 8, 2], 1) == 13, (
        "one day means one run: the capacity must hold the total weight, the upper end of the range"
    )
    assert min_ship_capacity([1], 1) == 1, (
        "a single parcel and a single day: the capacity is that parcel's weight"
    )


def test_days_at_least_the_number_of_parcels_needs_only_the_heaviest_parcel() -> None:
    assert min_ship_capacity([2, 9, 4], 5) == 9, (
        "with more days than parcels, the capacity only has to hold the heaviest parcel"
    )
    assert min_ship_capacity([5, 5, 5, 5], 2) == 10, (
        "equal parcels over a few days: check how the runs are formed and when a new day starts"
    )


def test_capacity_is_never_below_the_heaviest_parcel() -> None:
    # With spare days, a capacity below the heaviest parcel looks feasible to a sloppy check.
    assert min_ship_capacity([5, 5, 5], 4) == 5, (
        "the capacity is below the heaviest parcel: a truck must carry each parcel, so the lower "
        "end of the range is the heaviest parcel"
    )


def test_empty_weights_raise_value_error() -> None:
    _expect_value_error(
        lambda: min_ship_capacity([], 2),
        "empty weight list: validate the input and raise ValueError before searching",
    )


def test_zero_negative_or_non_integer_weight_raises_value_error() -> None:
    _expect_value_error(
        lambda: min_ship_capacity([3, 0, 4], 2),
        "a zero weight: weights must be positive, so raise ValueError",
    )
    _expect_value_error(
        lambda: min_ship_capacity([3, -1], 2),
        "a negative weight: weights must be positive, so raise ValueError",
    )
    _expect_value_error(
        lambda: min_ship_capacity([3, 2.5], 2),  # type: ignore[list-item]  # invalid on purpose
        "a non-integer weight: weights must be ints, so raise ValueError",
    )


def test_days_below_one_or_not_an_int_raises_value_error() -> None:
    _expect_value_error(
        lambda: min_ship_capacity([3, 4], 0),
        "zero days: days must be at least 1, so raise ValueError",
    )
    _expect_value_error(
        lambda: min_ship_capacity([3, 4], -2),
        "negative days: days must be at least 1, so raise ValueError",
    )
    _expect_value_error(
        lambda: min_ship_capacity([3, 4], 2.0),  # type: ignore[arg-type]  # invalid on purpose
        "days given as a float: days must be an int, so raise ValueError",
    )


def test_inputs_are_not_mutated() -> None:
    weights = [4, 2, 7, 3, 5]
    snapshot = list(weights)
    min_ship_capacity(weights, 2)
    assert weights == snapshot, (
        "the weights were changed: do not sort, reverse, pop or otherwise modify the input"
    )


def test_large_input_is_answered_by_a_search_not_a_scan() -> None:
    rng = random.Random(7)  # noqa: S311 - seeded, so the test is deterministic
    weights = [rng.randint(1, 100) for _ in range(5_000)]
    days = 20
    # A binary search makes about 20 passes over the parcels, plus three for validation and the
    # bounds. A scan over every capacity makes thousands of passes and is stopped after this budget.
    budget = 40 * len(weights)
    seq = _CountingSequence(weights, budget)
    try:
        got = min_ship_capacity(seq, days)
    except _ReadBudgetExceeded:
        raise AssertionError(
            "too many parcels read: search over capacities with about log(range) feasibility "
            "checks, not a scan over every capacity"
        ) from None
    assert _min_runs(weights, got) <= days, (
        "the answer does not ship every parcel within the day limit on the large input"
    )
    assert _min_runs(weights, got - 1) > days, (
        "the answer is feasible but not the smallest on the large input: find the first capacity "
        "that works"
    )
