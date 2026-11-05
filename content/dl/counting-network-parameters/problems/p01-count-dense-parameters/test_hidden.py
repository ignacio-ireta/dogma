from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import count_dense_parameters as _learner


def count_dense_parameters(*args):
    try:
        result = _learner(*args)
    except Exception as exc:
        pytest.fail(f"valid input violated the computation contract: {type(exc).__name__}")
    assert isinstance(result, tuple) and len(result) == 2, (
        "return per-layer counts and total as a tuple"
    )
    layers, total = result
    assert isinstance(layers, list), "per-layer counts must be a list"
    assert len(layers) == len(args[0]) - 1, "return exactly one count per dense layer"
    assert type(total) is int and all(type(value) is int for value in layers), (
        "all counts must be Python integers"
    )
    return result


def test_seeded_count_matches_explicit_small_parameter_slots():
    rng = np.random.default_rng(6)
    for _ in range(30):
        widths = rng.integers(1, 9, size=5).tolist()
        for bias in (False, True):
            expected = [
                len([(i, j) for i in range(a) for j in range(b)]) + len(range(b)) * bias
                for a, b in pairwise(widths)
            ]
            assert count_dense_parameters(widths, bias) == (expected, sum(expected)), (
                "count disagrees with explicit parameter slots"
            )


def test_large_counts_use_python_integers():
    value = 10**30
    layers, total = count_dense_parameters([value, value, 1])
    assert layers == [value * value + value, value + 1], (
        "large counts must not overflow fixed-width arithmetic"
    )
    assert total == value * value + 2 * value + 1, "total must retain integer precision"
    assert type(total) is int and all(type(x) is int for x in layers), (
        "counts must be Python integers"
    )


def test_invalid_widths_and_flag():
    for widths in ([], [2], [0, 3], [-1, 3], [True, 3], [2.0, 3], [np.int64(2), 3], "23"):
        _invalid(widths)
    for flag in (0, 1, None, "yes"):
        _invalid([2, 3], flag)


def _invalid(widths, flag=True):
    try:
        _learner(widths, flag)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid count arguments must raise ValueError, received {type(exc).__name__}")
    pytest.fail("invalid count arguments were accepted")


def test_widths_not_mutated():
    widths = [2, 7, 3, 1]
    before = widths.copy()
    count_dense_parameters(widths)
    assert widths == before, "width sequence must not be mutated"


def test_output_layer_and_order():
    assert count_dense_parameters([2, 3, 5, 1]) == ([9, 20, 6], 35), (
        "return every layer in forward order including output"
    )


@settings(max_examples=50, deadline=None, report_multiple_bugs=False)
@given(st.lists(st.integers(1, 100), min_size=2, max_size=8))
def test_bias_toggle_adds_exactly_one_slot_per_output(widths):
    with_bias, total = count_dense_parameters(widths)
    without_bias, base = count_dense_parameters(widths, False)
    assert total - base == sum(widths[1:]), "bias increment must equal the number of output units"
    assert all(
        a - b == width for a, b, width in zip(with_bias, without_bias, widths[1:], strict=True)
    ), "per-layer bias increment is incorrect"
