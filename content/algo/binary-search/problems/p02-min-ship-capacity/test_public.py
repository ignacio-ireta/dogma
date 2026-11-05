"""Public tests: the problem.md examples, the one-day and one-parcel-per-day cases, bad input."""

from collections.abc import Callable

from submission import min_ship_capacity  # type: ignore[import-not-found]


def _expect_value_error(call: Callable[[], object], mistake: str) -> None:
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


def test_example_three_days_needs_capacity_twelve() -> None:
    assert min_ship_capacity([4, 2, 7, 3, 5, 1, 6], 3) == 12, (
        "wrong capacity: a capacity of 11 needs four days, and 12 is the smallest that fits in "
        "three"
    )


def test_one_day_needs_the_total_weight() -> None:
    assert min_ship_capacity([9, 4, 6], 1) == 19, (
        "one day means one run: the capacity must hold the total weight"
    )


def test_one_parcel_per_day_needs_only_the_heaviest_parcel() -> None:
    assert min_ship_capacity([9, 4, 6], 3) == 9, (
        "with as many days as parcels, the capacity only has to hold the heaviest parcel"
    )


def test_empty_weights_raise_value_error() -> None:
    _expect_value_error(
        lambda: min_ship_capacity([], 2),
        "empty weight list: validate the input and raise ValueError before searching",
    )
