import numpy as np
import pytest
from scipy import stats
from submission import mean_confidence_interval

from dogma_data import load, sample_rows
from dogma_data import rng as data_rng

# (seed, n, confidence) triples for the oracle comparison. The data are Student-t with 3 degrees of
# freedom scaled and shifted, so the samples are not normal and the quantile choice matters.
ORACLE_CASES = [
    (0, 2, 0.95),
    (1, 3, 0.90),
    (2, 5, 0.95),
    (3, 8, 0.99),
    (4, 13, 0.50),
    (5, 30, 0.95),
    (6, 50, 0.80),
    (7, 100, 0.999),
]


def _sample(seed, n):
    """Rooms per household in n real california_housing block groups: skewed and heavy-tailed."""
    gen = data_rng(f"mean-ci-p01:sample:{seed}:{n}")
    return sample_rows("california_housing", gen, n)[0][:, 2]


def _reference(x, confidence):
    """Independent reference: scipy's t interval with scale = s / sqrt(n), df = n - 1."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    sem = np.std(x, ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(confidence, df=n - 1, loc=x.mean(), scale=sem)
    return [float(x.mean()), float(lo), float(hi)]


def _must_raise_value_error(call, mistake):
    """Fail with the kind of mistake, instead of pytest's generic 'DID NOT RAISE'."""
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


@pytest.mark.parametrize(
    ("seed", "n", "confidence"),
    ORACLE_CASES,
    ids=[f"oracle{index}" for index in range(len(ORACLE_CASES))],
)
def test_matches_student_t_reference_on_seeded_samples(seed, n, confidence):
    x = _sample(seed, n)
    # Computed before the call, so a learner who writes into x is reported as an in-place change.
    expected = _reference(x, confidence)
    np.testing.assert_allclose(
        mean_confidence_interval(x, confidence=confidence),
        expected,
        rtol=1e-9,
        atol=1e-12,
        err_msg="endpoints differ from the Student-t reference: mean ± t * s / sqrt(n), df = n - 1",
    )


def test_half_width_uses_t_quantile_with_n_minus_one_degrees_of_freedom():
    _, lo, hi = mean_confidence_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    half_width = stats.t.ppf(0.975, df=4) * np.sqrt(2.5) / np.sqrt(5)
    np.testing.assert_allclose(
        (hi - lo) / 2,
        half_width,
        rtol=1e-9,
        err_msg="half-width must use the t quantile with n - 1 = 4 degrees of freedom",
    )


def test_standard_error_uses_sample_standard_deviation_with_ddof_one():
    _, lo, hi = mean_confidence_interval([0.0, 0.0, 0.0, 0.0, 10.0])
    # Squared deviations sum to 80; dividing by n - 1 = 4 gives a variance of 20.
    half_width = stats.t.ppf(0.975, df=4) * np.sqrt(20.0) / np.sqrt(5.0)
    np.testing.assert_allclose(
        (hi - lo) / 2,
        half_width,
        rtol=1e-9,
        err_msg="standard error must use the sample standard deviation with ddof = 1",
    )


def test_two_observations_use_one_degree_of_freedom():
    _, lo, hi = mean_confidence_interval([1.0, 3.0])
    # s = sqrt(2), so the standard error is 1, and the margin is the t quantile with df = 1.
    np.testing.assert_allclose(
        (hi - lo) / 2,
        stats.t.ppf(0.975, df=1),
        rtol=1e-9,
        err_msg="with n = 2 the t quantile has 1 degree of freedom: check df = n - 1",
    )


def test_higher_confidence_gives_a_wider_interval():
    x = _sample(11, 10)
    widths = []
    for level in (0.80, 0.95, 0.99):
        _, lo, hi = mean_confidence_interval(x, confidence=level)
        widths.append(hi - lo)
    assert widths[0] < widths[1] < widths[2], (
        "a higher confidence level must give a wider interval: check the tail probability"
    )


def test_long_run_coverage_of_the_true_mean_is_close_to_95_percent():
    # A real population with a mean that is known exactly: the worst smoothness of the 569
    # tumours of breast_cancer, close to normal. It is sorted, so the draws do not depend on
    # the order of the data, and samples of five are drawn from it with replacement.
    population = np.sort(load("breast_cancer").X[:, 24])
    rng = np.random.default_rng(1234)
    true_mean, n, reps = float(population.mean()), 5, 4000
    covered = 0
    for _ in range(reps):
        x = population[rng.integers(0, population.size, size=n)]
        _, lo, hi = mean_confidence_interval(x)
        if lo <= true_mean <= hi:
            covered += 1
    coverage = covered / reps
    assert abs(coverage - 0.95) <= 0.015, (
        "long-run coverage of the true mean is not near 95%: check the t quantile and ddof = 1"
    )


def test_large_common_offset_does_not_destroy_the_spread():
    x = 1e8 + np.array([0.5, -0.5, 1.0, -1.0, 0.0, 0.25])
    _, ref_lo, ref_hi = _reference(x, 0.95)
    _, lo, hi = mean_confidence_interval(x)
    np.testing.assert_allclose(
        hi - lo,
        ref_hi - ref_lo,
        rtol=1e-6,
        err_msg="the spread is wrong for data with a large common offset: subtract the mean first",
    )


def test_integer_list_input_returns_three_float_values():
    result = mean_confidence_interval([2, 4, 6, 8])
    assert len(result) == 3, "return a tuple of (mean, lo, hi): exactly three values"
    assert all(isinstance(value, float) for value in result), (
        "all three values should be floats, even for integer input"
    )
    np.testing.assert_allclose(
        result,
        _reference([2, 4, 6, 8], 0.95),
        rtol=1e-9,
        err_msg="integer input gives the wrong interval: convert to float64 before the arithmetic",
    )


INVALID_CASES = [
    (
        "two_dimensional",
        [[1.0, 2.0], [3.0, 4.0]],
        0.95,
        "a 2-D sample is not a 1-D vector: raise ValueError",
    ),
    ("empty", [], 0.95, "an empty sample has no mean: raise ValueError"),
    ("single_observation", [7.0], 0.95, "one observation has no standard error: raise ValueError"),
    ("nan_value", [1.0, float("nan"), 3.0], 0.95, "a NaN observation must raise ValueError"),
    (
        "infinite_value",
        [1.0, float("inf"), 3.0],
        0.95,
        "an infinite observation must raise ValueError",
    ),
    (
        "confidence_zero",
        [1.0, 2.0, 3.0],
        0.0,
        "confidence 0 is not strictly between 0 and 1: raise ValueError",
    ),
    (
        "confidence_one",
        [1.0, 2.0, 3.0],
        1.0,
        "confidence 1 is not strictly between 0 and 1: raise ValueError",
    ),
    ("confidence_above_one", [1.0, 2.0, 3.0], 1.5, "confidence above 1 must raise ValueError"),
    ("confidence_negative", [1.0, 2.0, 3.0], -0.1, "a negative confidence must raise ValueError"),
]


@pytest.mark.parametrize(
    ("case", "x", "confidence", "mistake"),
    INVALID_CASES,
    ids=[case[0] for case in INVALID_CASES],
)
def test_invalid_input_raises_value_error(case, x, confidence, mistake):
    _must_raise_value_error(lambda: mean_confidence_interval(x, confidence=confidence), mistake)


def test_input_array_is_not_mutated():
    x = np.array([3.0, 1.0, 2.0, 5.0])
    before = x.copy()
    mean_confidence_interval(x)
    np.testing.assert_array_equal(
        x,
        before,
        err_msg="the input array was modified in place: sort or centre a copy instead",
    )


def test_result_is_a_tuple_of_three_values():
    result = mean_confidence_interval([1.0, 2.0, 3.0, 4.0])
    assert isinstance(result, tuple), (
        "the return value must be a tuple (mean, lo, hi), not another sequence type"
    )
    assert len(result) == 3, "the tuple must hold exactly three values: mean, lo and hi"
