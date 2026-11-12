from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from submission import count_dense_parameters as _learner

from dogma_data import rng as data_rng
from dogma_data import sample_rows


def _real_normal(gen, size=None, loc=0.0, scale=1.0):
    """Real numbers where this test used to draw from a normal distribution.

    Standardised breast_cancer measurements (mean 0 and standard deviation 1 over the dataset,
    skewed, with a few values far out), drawn by ``gen``. A vector is one real measurement
    over that many tumours, a matrix is real tumours by real measurements, and any other
    shape is filled from whole rows. ``loc`` and ``scale`` shift and stretch them as before.
    """
    shape = () if size is None else tuple(int(side) for side in np.atleast_1d(size))
    total = int(np.prod(shape)) if shape else 1
    if len(shape) == 1 and 1 <= shape[0] <= 569:
        values = sample_rows("breast_cancer", gen, shape[0], 1, standardize=True)[0][:, 0]
    elif len(shape) == 2 and 1 <= shape[0] <= 569 and 1 <= shape[1] <= 30:
        values = sample_rows("breast_cancer", gen, shape[0], shape[1], standardize=True)[0]
    else:
        rows = max(1, min(569, -(-total // 30)))
        drawn = sample_rows("breast_cancer", gen, rows, standardize=True)[0]
        values = np.resize(drawn.ravel(), total).reshape(shape)
    values = values * scale + loc
    return values if shape else float(values)


def _real_unit(gen, size=None):
    """Real numbers between 0 and 1 where this test used to draw uniform ones.

    Each is the share of a standard normal that lies below a standardised breast_cancer
    measurement, so the values keep the order, the ties and the skew of real measurements.
    """
    return ndtr(_real_normal(gen, size))


def _real_ints(gen, low, high, size=None):
    """Real whole numbers from ``low`` up to ``high - 1``: the band a real measurement falls in.

    The range is cut into ``high - low`` bands of equal normal probability, and each value is
    the band of one standardised measurement. Real measurements are skewed, so the bands are
    not equally full, and neighbouring values tie as real categories and grades do.
    """
    bands = np.floor(_real_unit(gen, size) * (high - low))
    return (low + np.minimum(bands, high - low - 1)).astype(np.int64)


def _real_shares(gen, alpha, size=None):
    """Real probability vectors: positive real numbers (see `_real_unit`) divided by their sum."""
    k = int(np.size(alpha))
    shape = (k,) if size is None else (int(size), k)
    weights = _real_unit(gen, shape) + 1e-3
    return weights / weights.sum(axis=-1, keepdims=True)


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


def _real_ends():
    """(input width, output width) of a network for six real datasets.

    The input width is the number of features of the dataset and the output width its number
    of classes, or 1 for a regression target: iris 4 and 3, wine 13 and 3, breast_cancer 30 and
    2, digits 64 and 10, diabetes 10 and 1, california_housing 8 and 1.
    """
    ends = []
    for name, outputs in (
        ("iris", 3),
        ("wine", 3),
        ("breast_cancer", 2),
        ("digits", 10),
        ("diabetes", 1),
        ("california_housing", 1),
    ):
        features, _ = sample_rows(name, data_rng(f"p01-count-dense-parameters:{name}"), 1)
        ends.append((int(features.shape[1]), outputs))
    return ends


def test_seeded_count_matches_explicit_small_parameter_slots():
    # Each network is sized for a real dataset: its features come in and its classes come
    # out. The three hidden widths are drawn.
    rng = np.random.default_rng(6)
    ends = _real_ends()
    for trial in range(30):
        first, last = ends[trial % len(ends)]
        widths = [first, *_real_ints(rng, 1, 9, 3).tolist(), last]
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
