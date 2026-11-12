from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from scipy.stats import entropy as reference_entropy
from submission import entropy

from dogma_data import load_frame, rng, sample_rows


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


@functools.cache
def _adult():
    """The adult census table, read once: real categorical columns with very uneven shares."""
    return load_frame("adult")


CATEGORICAL = ("workclass", "education", "marital.status", "occupation", "relationship", "race")


def _real_distribution(gen):
    """Shares of one adult column's categories among a small draw of people.

    Every category of the column is listed, so the ones nobody in the draw holds have
    probability zero: real distributions are sparse.
    """
    table = _adult()
    column = table[CATEGORICAL[int(gen.integers(len(CATEGORICAL)))]]
    rows = gen.choice(len(table), size=int(gen.integers(5, 80)), replace=False)
    counts = column.iloc[rows].value_counts().reindex(sorted(column.unique()), fill_value=0)
    return counts.to_numpy(dtype=np.float64) / len(rows)


def _call(p, **kwargs):
    """Call entropy and check the value is a finite float. Messages never show inputs."""
    try:
        value = entropy(p, **kwargs)
    except Exception as exc:
        pytest.fail(
            f"a valid probability vector was rejected or crashed ({type(exc).__name__}): "
            "check the input rules in the contract"
        )
    if not isinstance(value, float):
        pytest.fail("the entropy must be returned as a Python float")
    if not math.isfinite(value):
        pytest.fail("the entropy is not finite: a zero probability must add 0, not nan")
    return value


def _expect_value_error(p, mistake, **kwargs):
    try:
        entropy(p, **kwargs)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; the function raised {type(exc).__name__} instead")
    pytest.fail(f"{mistake}; the function returned a value instead")


def test_matches_reference_entropy_on_random_distributions():
    gen = rng("entropy-p01:reference")
    for base in (2.0, math.e, 10.0):
        for _ in range(25):
            p = _real_distribution(gen)  # unseen categories exercise the zero-term rule
            np.testing.assert_allclose(
                _call(p, base=base),
                reference_entropy(p, base=base),
                rtol=1e-9,
                atol=1e-12,
                err_msg="entropy disagrees with the reference: check the minus sign, "
                "the zero-term rule and the change of base",
            )


def test_zero_probability_entries_add_nothing():
    value = _call([0.5, 0.0, 0.5])
    assert value == pytest.approx(1.0, abs=1e-12), (
        "zero-probability entries must add nothing to the entropy of the remaining outcomes"
    )


def test_certain_outcome_has_zero_entropy():
    value = _call([0.0, 0.0, 1.0, 0.0])
    assert abs(value) <= 1e-12, "a certain outcome must have zero entropy"


@pytest.mark.parametrize("size", [2, 3, 7, 64], ids=["few", "some", "many", "wide"])
def test_uniform_distribution_reaches_log_of_its_size(size):
    value = _call(np.full(size, 1.0 / size), base=2.0)
    assert value == pytest.approx(math.log2(size), rel=1e-12), (
        "a uniform distribution must reach log2 of its size: check the logarithm and the sign"
    )


def test_changing_the_base_rescales_the_value_by_a_constant():
    p = np.array([0.6, 0.3, 0.1])
    bits = _call(p, base=2.0)
    np.testing.assert_allclose(
        _call(p, base=math.e),
        bits * math.log(2),
        rtol=1e-12,
        err_msg="the base must only rescale the unit: convert with ln(x) / ln(base)",
    )
    np.testing.assert_allclose(
        _call(p, base=10.0),
        bits * math.log(2) / math.log(10),
        rtol=1e-12,
        err_msg="the base must only rescale the unit: convert with ln(x) / ln(base)",
    )


def test_base_below_one_gives_the_negated_value():
    p = np.array([0.6, 0.3, 0.1])
    np.testing.assert_allclose(
        _call(p, base=0.5),
        -_call(p, base=2.0),
        rtol=1e-12,
        err_msg="a base below 1 must give the negated entropy: the value is -sum p log_b p",
    )


def test_sum_within_tolerance_is_accepted():
    value = _call([0.5, 0.5 + 5e-10])
    assert value == pytest.approx(1.0, abs=1e-6), (
        "a vector summing to 1 within 1e-9 is a valid distribution and must be accepted"
    )


def test_sum_outside_tolerance_is_rejected():
    _expect_value_error(
        [0.5, 0.5 + 2e-9],
        (
            "a distribution outside the stated sum tolerance must raise ValueError "
            "rather than being repaired"
        ),
    )


INVALID_DISTRIBUTIONS = [
    ("negative-entry", [-0.2, 1.2], "a negative probability must raise ValueError"),
    ("sum-too-small", [0.2, 0.2], "a vector with sum below 1 must raise ValueError"),
    ("sum-too-large", [0.7, 0.7], "a vector with sum above 1 must raise ValueError"),
    ("nan-entry", [0.5, float("nan")], "a NaN entry must raise ValueError"),
    ("infinite-entry", [float("inf"), 0.0], "an infinite entry must raise ValueError"),
    ("empty", [], "an empty vector must raise ValueError"),
    ("two-dimensional", [[0.5, 0.5]], "a two-dimensional input must raise ValueError"),
]


@pytest.mark.parametrize(
    ("p", "mistake"),
    [(case[1], case[2]) for case in INVALID_DISTRIBUTIONS],
    ids=[case[0] for case in INVALID_DISTRIBUTIONS],
)
def test_invalid_distribution_raises_value_error(p, mistake):
    _expect_value_error(p, mistake)


INVALID_BASES = [
    ("zero-base", 0.0, "a base of 0 must raise ValueError: the base must be greater than 0"),
    ("negative-base", -2.0, "a negative base must raise ValueError"),
    ("unit-base", 1.0, "base 1 must raise ValueError: its logarithm is zero"),
    ("nan-base", float("nan"), "a NaN base must raise ValueError"),
    ("infinite-base", float("inf"), "an infinite base must raise ValueError"),
]


@pytest.mark.parametrize(
    ("base", "mistake"),
    [(case[1], case[2]) for case in INVALID_BASES],
    ids=[case[0] for case in INVALID_BASES],
)
def test_invalid_base_raises_value_error(base, mistake):
    _expect_value_error([0.5, 0.5], mistake, base=base)


def test_tiny_probabilities_keep_the_entropy_small_and_finite():
    p = np.array([1e-300, 1.0 - 1e-300])
    value = _call(p)
    assert 0.0 <= value <= 1e-290, (
        "tiny probabilities must give a small, finite entropy: take logarithms of the "
        "positive entries only"
    )


def test_input_array_is_not_mutated():
    p = np.array([0.5, 0.1, 0.4])  # unsorted, so an in-place sort would be visible
    before = p.copy()
    _call(p, base=2.0)
    np.testing.assert_array_equal(
        p, before, err_msg="the input array was modified: do not change the argument in place"
    )


def test_returns_a_python_float_for_a_list_input():
    value = entropy([0.25, 0.75])
    assert isinstance(value, float), "the entropy must be a Python float, not an array"


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=0, max_value=10_000), st.integers(min_value=1, max_value=12))
def test_entropy_lies_between_zero_and_log_of_the_number_of_outcomes(seed, k):
    rng = np.random.default_rng(seed)
    p = _real_shares(rng, np.ones(k))
    value = _call(p)
    assert -1e-12 <= value <= math.log2(k) + 1e-12, (
        "entropy must lie between 0 and log2 of the number of outcomes"
    )


def test_sum_slightly_outside_tolerance_is_rejected():
    _expect_value_error(
        [0.5, 0.5 + 1.5e-9],
        "a distribution whose sum is 1.5e-9 away from 1 is outside the 1e-9 tolerance and must "
        "raise ValueError rather than being repaired",
    )


def test_input_within_sum_tolerance_is_not_changed_in_place():
    # The sum is 1 + 4e-10: accepted as given, so it must not be renormalised in place.
    p = np.array([0.2, 0.3, 0.5 + 4e-10])
    before = p.copy()
    _call(p, base=2.0)
    np.testing.assert_array_equal(
        p, before, err_msg="the input array was modified: do not change the argument in place"
    )
