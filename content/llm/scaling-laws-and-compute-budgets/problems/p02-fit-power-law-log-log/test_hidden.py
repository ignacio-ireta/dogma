# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of
# mistake and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import collections
import functools
import math
import re

import numpy as np
import pytest
from submission import fit_power_law as _learner

from dogma_data import load_text
from dogma_data import rng as data_rng

BOOKS = ("gutenberg_frankenstein", "gutenberg_pride_prejudice", "gutenberg_walden")


@functools.cache
def _word_counts(book):
    """Counts of the words of one vendored novel, most frequent first."""
    words = re.findall(r"[a-z]+", load_text(book).lower())
    return tuple(count for _, count in collections.Counter(words).most_common())


def _zipf(gen, n):
    """A real power law: word frequency against rank in a novel (Zipf's law).

    x holds n consecutive ranks starting at a randomly chosen rank from 1 to 20, and y the
    number of times the word of that rank occurs. The exponent is near -1 and the points
    scatter around the line, as measured data does; equal counts repeat far down the ranks.
    """
    counts = _word_counts(BOOKS[int(gen.integers(len(BOOKS)))])
    first = int(gen.integers(1, 21))
    x = np.arange(first, first + n, dtype=np.float64)
    return x, np.array(counts[first - 1 : first - 1 + n], dtype=np.float64)


def _call(x, y):
    try:
        return _learner(x, y)
    except ValueError:
        raise
    except Exception as exc:
        pytest.fail(f"valid arguments violated the contract: raised {type(exc).__name__}")


def _expect_value_error(x, y):
    try:
        _learner(x, y)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid input raised {type(exc).__name__}, expected ValueError")
    pytest.fail("invalid input was accepted instead of raising ValueError")


def _polyfit_oracle(x, y):
    """Independent fit: numpy.polyfit of degree 1 on the logs. Returns (a, b)."""
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return math.exp(intercept), slope


def test_seeded_noisy_fit_matches_polyfit_on_the_logs():
    rng = data_rng("fit-power-law-p02:polyfit")
    for n in (8, 20, 50):
        x, y = _zipf(rng, n)
        a, b = _call(x, y)
        a_want, b_want = _polyfit_oracle(x, y)
        assert abs(b - b_want) <= 1e-9 + 1e-9 * abs(b_want), (
            "slope disagrees with the least-squares line: check the centred cross-product"
        )
        assert abs(a - a_want) <= 1e-9 + 1e-9 * abs(a_want), (
            "intercept disagrees with the least-squares line: check exp of the intercept"
        )


def test_noise_free_power_laws_with_negative_and_positive_exponents_are_recovered():
    x = np.logspace(-2, 3, 25)
    for a_true, b_true in [(3.7e-4, -1.5), (2.5, 2.25)]:
        a, b = _call(x, a_true * x**b_true)
        assert abs(b - b_true) < 1e-9, "exponent recovered with the wrong sign or scale"
        assert abs(a / a_true - 1.0) < 1e-9, "coefficient a recovered wrong: check exp(intercept)"


def test_multiplying_y_scales_a_and_keeps_b():
    x, y = _zipf(data_rng("fit-power-law-p02:scale-y"), 15)
    a, b = _call(x, y)
    a2, b2 = _call(x, 1000.0 * y)
    assert abs(b2 - b) < 1e-9, "scaling y must not change the exponent"
    assert abs(a2 / (1000.0 * a) - 1.0) < 1e-9, "scaling y by c must scale a by c"


def test_multiplying_x_rescales_a_by_a_power_of_the_factor():
    x, y = _zipf(data_rng("fit-power-law-p02:scale-x"), 15)
    a, b = _call(x, y)
    s = 1000.0
    a2, b2 = _call(s * x, y)
    assert abs(b2 - b) < 1e-9, "scaling x must not change the exponent"
    assert abs(a2 / (a * s ** (-b)) - 1.0) < 1e-9, (
        "scaling x by s must rescale a by s**(-b): check the sign of the exponent"
    )


def test_huge_and_tiny_magnitudes_stay_finite_and_correct():
    x = np.array([1e150, 2e150, 4e150, 8e150])
    y = 1e-100 * x**-0.5
    a, b = _call(x, y)
    assert math.isfinite(a) and math.isfinite(b), "huge or tiny inputs produced non-finite output"
    assert abs(b + 0.5) < 1e-9, "exponent wrong for huge inputs: work in log space"
    assert abs(a / 1e-100 - 1.0) < 1e-9, "coefficient wrong for huge inputs"


def test_points_far_from_zero_in_log_space_keep_the_slope():
    # log(1e8 + k) differs from log(1e8) by about k * 1e-8, so the centred spread of u is tiny
    # next to its magnitude. The expanded formula cancels the spread away. The tolerance is
    # 1e-6 because each log carries about 1e-15 absolute rounding, which is about 1e-7 relative
    # to the spread. The oracle sees the same logs, so this tests the algorithm, not the logs.
    x = 1e8 + np.arange(5, dtype=np.float64)
    y = 2.0 * x**0.5
    _, b = _call(x, y)
    _, b_want = _polyfit_oracle(x, y)
    assert abs(b - b_want) < 1e-6, (
        "slope lost digits on offset inputs: centre u and v before the cross-product"
    )


def test_two_points_fit_the_line_through_them():
    a, b = _call(np.array([2.0, 8.0]), np.array([1.0, 4.0]))
    assert abs(b - 1.0) < 1e-12, "two points give slope log(4)/log(4) = 1"
    assert abs(a - 0.5) < 1e-12, "the line through (2, 1) and (8, 4) gives a = 1/2"


def test_single_point_raises_value_error():
    _expect_value_error(np.array([2.0]), np.array([3.0]))


def test_length_mismatch_raises_value_error():
    _expect_value_error(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_two_dimensional_input_raises_value_error():
    _expect_value_error(np.ones((2, 2)), np.ones((2, 2)))


@pytest.mark.parametrize(
    "x, y",
    [
        (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, -2.0, 3.0])),
    ],
    ids=["zero-x", "negative-y"],
)
def test_non_positive_values_raise_value_error(x, y):
    _expect_value_error(x, y)


@pytest.mark.parametrize("bad", [np.nan, np.inf], ids=["nan", "inf"])
def test_non_finite_value_raises_value_error(bad):
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, bad, 3.0])
    _expect_value_error(x, y)


def test_integer_arrays_are_accepted():
    x = np.array([1, 2, 4, 8], dtype=np.int64)
    y = np.array([3, 6, 12, 24], dtype=np.int64)
    a, b = _call(x, y)
    assert abs(b - 1.0) < 1e-9, "integer inputs must be cast to float before the logarithm"
    assert abs(a - 3.0) < 1e-9, "integer inputs give the same coefficients as floats"


def test_returns_a_pair_of_python_floats_with_positive_a():
    a, b = _call(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 5.0]))
    assert type(a) is float and type(b) is float, "the return values must be Python float"
    assert a > 0, "a must be positive: it is exp of the intercept"


def test_inputs_are_not_mutated():
    x = np.array([1.0, 2.0, 4.0])
    y = np.array([2.0, 3.0, 5.0])
    x0, y0 = x.copy(), y.copy()
    _call(x, y)
    assert np.array_equal(x, x0), "the x array must not be modified"
    assert np.array_equal(y, y0), "the y array must not be modified"


def test_two_dimensional_arrays_are_rejected_with_value_error():
    x = np.array([[1.0, 2.0], [4.0, 8.0]])
    y = np.array([[1.0, 3.0], [2.0, 5.0]])
    _expect_value_error(x, y)


def test_length_mismatch_against_a_single_value_raises_value_error():
    _expect_value_error(np.array([1.0, 2.0, 3.0]), np.array([4.0]))


def test_zero_y_value_raises_value_error():
    _expect_value_error(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 3.0]))


def test_y_value_between_minus_one_and_zero_raises_value_error():
    _expect_value_error(np.array([1.0, 2.0, 3.0]), np.array([1.0, -0.5, 3.0]))


@pytest.mark.parametrize("bad", [np.nan, np.inf], ids=["nan", "inf"])
def test_non_finite_x_value_raises_value_error(bad):
    x = np.array([1.0, bad, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    _expect_value_error(x, y)


def test_returns_a_tuple_of_two_values_not_a_list():
    result = _call(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 5.0]))
    assert isinstance(result, tuple), "the return value must be a tuple (a, b), not a list"
    assert len(result) == 2, "the tuple must hold exactly two values: a and b"
