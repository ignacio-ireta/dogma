# Hidden tests (CONTENT_SPEC §6.4). Failure messages name the kind of mistake and never show
# hidden inputs or expected values. The independent oracle is a seeded Monte Carlo estimate.
from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from submission import diagonal_gaussian_kl

from dogma_data import load

ATOL = 1e-10  # closed-form values agree with the formula to round-off, about 1e-15 here
RTOL = 1e-10


def _closed(mp, sp, mq, sq):
    """Test-side formula used for the analytic and the scaling checks (not the learner's code)."""
    mp, sp, mq, sq = (np.asarray(a, dtype=float) for a in (mp, sp, mq, sq))
    return float(np.sum(np.log(sq / sp) + (sp**2 + (mp - mq) ** 2) / (2 * sq**2) - 0.5))


def _call(mp, sp, mq, sq):
    try:
        value = diagonal_gaussian_kl(mp, sp, mq, sq)
    except Exception as exc:
        pytest.fail(
            f"valid parameters were rejected or crashed ({type(exc).__name__}): check the contract"
        )
    if type(value) is not float:
        pytest.fail("return a Python float, not an array or a numpy scalar")
    if not math.isfinite(value):
        pytest.fail("non-finite KL for finite parameters: check the intermediate products")
    return value


def _assert_close(value, expected, mistake_variants=()):
    if math.isclose(value, expected, rel_tol=RTOL, abs_tol=ATOL):
        return
    for alt, message in mistake_variants:
        if math.isclose(value, alt, rel_tol=RTOL, abs_tol=ATOL):
            pytest.fail(message)
    pytest.fail("KL differs from the closed form: check the formula term by term")


def _variants(mp, sp, mq, sq):
    """Known wrong formulas, each named in the failure message when it matches the output."""
    mp, sp, mq, sq = (np.asarray(a, dtype=float) for a in (mp, sp, mq, sq))
    d = sp.size
    return [
        (_closed(mq, sq, mp, sp), "KL reversed: it computed KL(Q || P) instead of KL(P || Q)"),
        (
            float(np.sum(np.log(sq / sp) + (sp**2 + (mp - mq) ** 2) / (2 * sq**2))),
            "the constant -1/2 per coordinate is missing",
        ),
        (
            float(np.sum(np.log(sq / sp) + (sp + (mp - mq) ** 2) / (2 * sq**2) - 0.5)),
            "used standard deviations where variances belong in the spread term",
        ),
        (
            float(np.sum(-np.log(sq / sp) + (sp**2 + (mp - mq) ** 2) / (2 * sq**2) - 0.5)),
            "the log-scale term has the wrong sign: it is log(sigma_q / sigma_p)",
        ),
        (
            float(np.sum(np.log(sq / sp) + (sp**2 + (mp - mq) ** 2) / (2 * sp**2) - 0.5)),
            "the spread is divided by P's variance: it must be divided by Q's variance",
        ),
        (
            float(
                np.sum(np.log(sq / sp) + sp**2 / (2 * sq**2) + (mp - mq) ** 2 / (2 * sp**2) - 0.5)
            ),
            "the mean difference is divided by P's variance: divide it by Q's variance",
        ),
        (
            float(np.sum(np.log(sq / sp) + (sp**2 + (mp - mq) ** 2) / (2 * sq**2) - 0.5)) * d,
            "summed scalars instead of coordinates: check the axis of the reduction",
        ),
    ]


def _species(label, columns):
    """Mean and standard deviation of real iris measurements (cm) for one species.

    The diagonal Gaussian fitted to a species: label 0 is setosa, 1 versicolor, 2 virginica.
    """
    data = load("iris")
    rows = data.X[data.y == label][:, list(columns)]
    return rows.mean(axis=0), rows.std(axis=0, ddof=1)


def _mc_logpdf(x, mean, sd):
    return np.sum(-0.5 * np.log(2 * np.pi * sd * sd) - (x - mean) ** 2 / (2 * sd * sd), axis=-1)


@pytest.mark.parametrize(
    "mp, sp, mq, sq, expected",
    [
        ([0.0], [1.0], [0.0], [1.0], 0.0),
        ([0.0], [1.0], [1.0], [1.0], 0.5),
        ([0.0], [1.0], [0.0], [2.0], math.log(2.0) + 0.125 - 0.5),
        ([0.0], [2.0], [0.0], [1.0], math.log(0.5) + 2.0 - 0.5),
        ([3.0], [0.5], [3.0], [0.5], 0.0),
        ([0.0, 1.0], [1.0, 2.0], [0.0, 1.0], [1.0, 2.0], 0.0),
    ],
    ids=["identity", "mean-shift", "wider-q", "narrower-q", "same-shifted", "two-coordinates"],
)
def test_analytic_one_dimensional_and_coordinate_cases(mp, sp, mq, sq, expected):
    _assert_close(_call(mp, sp, mq, sq), expected, _variants(mp, sp, mq, sq))


def test_monte_carlo_estimate_agrees_within_five_standard_errors():
    # P is the Gaussian fitted to versicolor, Q the one fitted to virginica, over three
    # measurements. The samples are drawn from P, which is what the estimate needs.
    mp, sp = _species(1, (0, 1, 2))
    mq, sq = _species(2, (0, 1, 2))
    gen = np.random.default_rng(20260101)
    n = 400_000
    x = mp + sp * gen.standard_normal((n, 3))
    per_sample = _mc_logpdf(x, mp, sp) - _mc_logpdf(x, mq, sq)
    estimate = per_sample.mean()
    # 5 standard errors. With the seed fixed the test is deterministic, and a correct formula
    # lands about 0.7 standard errors away here. A tighter bound would test sampling noise.
    se = per_sample.std(ddof=1) / math.sqrt(n)
    value = _call(mp, sp, mq, sq)
    if abs(value - estimate) > 5 * se:
        for alt, message in _variants(mp, sp, mq, sq):
            if math.isclose(value, alt, rel_tol=RTOL, abs_tol=ATOL):
                pytest.fail(message)
        pytest.fail("KL disagrees with the Monte Carlo estimate beyond five standard errors")


def test_forward_and_reverse_divergence_are_distinct_on_asymmetric_parameters():
    # Setosa against virginica over petal length and petal width: narrow against wide.
    mp, sp = _species(0, (2, 3))
    mq, sq = _species(2, (2, 3))
    forward = _closed(mp, sp, mq, sq)
    reverse = _closed(mq, sq, mp, sp)
    if abs(forward - reverse) < 0.5:
        pytest.fail("test setup is degenerate: forward and reverse KL should differ clearly")
    _assert_close(_call(mp, sp, mq, sq), forward, _variants(mp, sp, mq, sq))


@pytest.mark.parametrize("scale", [1e-40, 1e40], ids=["tiny-scale", "huge-scale"])
def test_common_scale_factor_leaves_the_divergence_unchanged(scale):
    mp, sp = [0.3, -0.2], [0.8, 1.4]
    mq, sq = [0.1, 0.5], [1.1, 0.6]
    base = _closed(mp, sp, mq, sq)
    scaled = _call(
        np.array(mp) * scale, np.array(sp) * scale, np.array(mq) * scale, np.array(sq) * scale
    )
    _assert_close(scaled, base, _variants(mp, sp, mq, sq))


def test_identical_distributions_give_exactly_zero_at_many_scales():
    for sd in (1e-30, 0.5, 7.0, 1e30):
        value = _call([1.0, -2.0], [sd, 2 * sd], [1.0, -2.0], [sd, 2 * sd])
        if abs(value) > 1e-12:
            pytest.fail("identical distributions must have KL 0: check the -1/2 constant")


def test_integer_inputs_are_accepted():
    value = _call([0, 1], [1, 2], [1, 0], [1, 2])
    _assert_close(value, _closed([0, 1], [1, 2], [1, 0], [1, 2]))


def test_invalid_inputs_raise_value_error():
    cases = [
        ([[0.0]], [1.0], [0.0], [1.0], "two-dimensional mean was accepted"),
        ([], [], [], [], "empty vectors were accepted"),
        ([0.0, 0.0], [1.0], [0.0], [1.0], "mismatched lengths were accepted"),
        ([0.0], [0.0], [0.0], [1.0], "zero standard deviation was accepted"),
        ([0.0], [-1.0], [0.0], [1.0], "negative standard deviation was accepted"),
        ([0.0], [1.0], [0.0], [np.nan], "NaN standard deviation was accepted"),
        ([np.inf], [1.0], [0.0], [1.0], "infinite mean was accepted"),
        ([True], [1.0], [0.0], [1.0], "boolean mean was accepted"),
        ([0.0 + 1j], [1.0], [0.0], [1.0], "complex mean was accepted"),
    ]
    for mp, sp, mq, sq, mistake in cases:
        try:
            diagonal_gaussian_kl(mp, sp, mq, sq)
        except ValueError:
            continue
        except Exception as exc:
            pytest.fail(f"{mistake}; it raised {type(exc).__name__} instead of ValueError")
        pytest.fail(f"{mistake}; it returned a value instead of raising ValueError")


def test_inputs_are_not_mutated():
    arrays = [
        np.array([0.2, -0.5]),
        np.array([0.9, 1.3]),
        np.array([0.0, 0.1]),
        np.array([1.0, 2.0]),
    ]
    before = copy.deepcopy(arrays)
    diagonal_gaussian_kl(*arrays)
    for now, then in zip(arrays, before, strict=True):
        if not np.array_equal(now, then):
            pytest.fail("an input vector was modified: never write into the caller's arrays")


def _must_raise_value_error(thunk, mistake):
    try:
        thunk()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; it raised {type(exc).__name__} instead of ValueError")
    pytest.fail(f"{mistake}; it returned a value instead of raising ValueError")


def test_one_non_finite_mean_entry_raises_value_error_even_with_finite_entries():
    _must_raise_value_error(
        lambda: diagonal_gaussian_kl([0.0, float("nan")], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]),
        "a NaN in mu_p was accepted because the other entries were finite",
    )
    _must_raise_value_error(
        lambda: diagonal_gaussian_kl([0.0, 0.0], [1.0, 1.0], [0.0, float("inf")], [1.0, 1.0]),
        "an infinite entry in mu_q was accepted because the other entries were finite",
    )


def test_one_non_positive_standard_deviation_entry_raises_value_error():
    _must_raise_value_error(
        lambda: diagonal_gaussian_kl([0.0, 0.0], [1.0, -1.0], [0.0, 0.0], [1.0, 1.0]),
        "a negative sigma_p entry was accepted because the other entries were positive",
    )
    _must_raise_value_error(
        lambda: diagonal_gaussian_kl([0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]),
        "a zero sigma_q entry was accepted because the other entries were positive",
    )


def test_zero_or_negative_sigma_q_raises_value_error():
    for bad in ([0.0], [-1.0]):
        _must_raise_value_error(
            lambda bad=bad: diagonal_gaussian_kl([0.0], [1.0], [0.0], bad),
            "sigma_q must be strictly positive: zero or negative standard deviations were accepted",
        )
