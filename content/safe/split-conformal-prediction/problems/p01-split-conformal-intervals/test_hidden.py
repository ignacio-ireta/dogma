# Hidden tests (CONTENT_SPEC §6.4). The oracle sorts the residuals and takes the rank from alpha's
# decimal text (repr) with Decimal arithmetic at 60 digits and a ceiling. That is independent of the
# learner's arithmetic, so it catches float rank rounding and binary-value rank errors. The coverage
# simulation uses a 4-sigma bound on a frequency over 4000 seeded trials.
import math
from decimal import ROUND_CEILING, Decimal, localcontext

import numpy as np
import pytest
from submission import split_conformal_interval

from dogma_data import rng as data_rng
from dogma_data import sample_rows


def house_predictions(gen, n, m):
    """Real predictions and real house values for calibration, and m more predictions.

    The model is least squares on median income, house age and latitude (standardised), fitted
    to sixty other california_housing block groups. Its residuals are real: mostly small, a
    few large, and several exactly equal where the value is capped.
    """
    X, value = sample_rows("california_housing", gen, n + m + 60, standardize=True)
    design = np.column_stack([np.ones(len(X)), X[:, [0, 1, 6]]])
    beta = np.linalg.lstsq(design[n + m :], value[n + m :], rcond=None)[0]
    pred = design @ beta
    return pred[:n], value[:n], pred[n : n + m]


def exact_rank(n, alpha):
    with localcontext() as context:
        context.prec = 60
        value = (n + 1) * (1 - Decimal(repr(float(alpha))))
        return int(value.to_integral_value(rounding=ROUND_CEILING))


def oracle(cal_pred, cal_true, test_pred, alpha):
    residuals = sorted(abs(float(t) - float(p)) for p, t in zip(cal_pred, cal_true, strict=True))
    n = len(residuals)
    k = exact_rank(n, alpha)
    threshold = math.inf if k > n else residuals[k - 1]
    lower = [float(p) - threshold for p in test_pred]
    upper = [float(p) + threshold for p in test_pred]
    return np.array(lower, dtype=float), np.array(upper, dtype=float)


def call(cal_pred, cal_true, test_pred, alpha):
    try:
        lower, upper = split_conformal_interval(
            np.asarray(cal_pred), np.asarray(cal_true), np.asarray(test_pred), alpha
        )
    except Exception as error:  # a crash on valid input is a failure, not a test error
        pytest.fail(
            f"valid input raised {type(error).__name__}: check the calibration scores and the rank",
            pytrace=False,
        )
    for name, value in (("lower", lower), ("upper", upper)):
        if not isinstance(value, np.ndarray) or value.dtype != np.float64:
            pytest.fail(f"return float64 NumPy arrays for {name}", pytrace=False)
        if value.shape != (len(test_pred),):
            pytest.fail("bounds must have one entry per test prediction", pytrace=False)
    return lower, upper


def same(actual, expected, mistake):
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=False):
        pytest.fail(mistake, pytrace=False)


def test_random_calibration_matches_sorted_index_oracle():
    rng = data_rng("conformal-intervals-p01:oracle")
    alphas = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.9]
    for _ in range(60):
        n = int(rng.integers(1, 31))
        alpha = alphas[int(rng.integers(len(alphas)))]
        cal_pred, cal_true, test_pred = house_predictions(rng, n, 5)
        lower, upper = call(cal_pred, cal_true, test_pred, alpha)
        want_lower, want_upper = oracle(cal_pred, cal_true, test_pred, alpha)
        same(
            lower,
            want_lower,
            "lower bound differs from the sorted-index oracle: check the rank k, "
            "the 1-indexed position, and the absolute residuals",
        )
        same(upper, want_upper, "upper bound differs: symmetric width is test_pred plus or minus q")


def test_rank_is_read_from_the_decimal_alpha_not_a_float_product():
    # n = 9 and alpha = 0.7: (n + 1)(1 - 0.7) = 3 exactly, so q is the 3rd smallest residual, 0.3.
    # The float product 10 * (1 - 0.7) is 3.0000000000000004, which would give the 4th, 0.4.
    cal_true = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9]
    lower, upper = call(np.zeros(9), cal_true, [0.0], 0.7)
    same(lower, [-0.3], "rank for n = 9 and alpha = 0.7 is 3: the threshold is the 3rd residual")
    same(upper, [0.3], "rank for n = 9 and alpha = 0.7 is 3: the threshold is the 3rd residual")


def test_alpha_three_tenths_is_read_as_written_not_as_its_binary_value():
    # alpha = 0.3 is 3/10 as written: (n + 1)(1 - 0.3) = 7 for n = 9, so q = 0.7. The float 0.3 is
    # slightly below 3/10, and its binary value would give rank 8 and q = 0.8.
    cal_true = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9]
    lower, upper = call(np.zeros(9), cal_true, [0.0], 0.3)
    same(lower, [-0.7], "alpha = 0.3 is read as written: rank 7, threshold 0.7")
    same(upper, [0.7], "alpha = 0.3 is read as written: rank 7, threshold 0.7")


def test_rank_above_calibration_size_gives_unbounded_interval():
    lower, upper = call(np.zeros(9), np.linspace(0.1, 0.9, 9), [1.0], 0.05)
    assert lower[0] == -math.inf and upper[0] == math.inf, (
        "k = 10 > n = 9 at alpha = 0.05: no finite interval exists, so return -inf and +inf"
    )


def test_single_calibration_point_at_alpha_above_one_half_is_finite():
    lower, upper = call([0.0], [2.0], [0.0], 0.6)
    # (1 + 1)(1 - 0.6) = 0.8, so k = 1 and the threshold is the only residual.
    same(lower, [-2.0], "one calibration point: the threshold is its residual")
    same(upper, [2.0], "one calibration point: the threshold is its residual")


def test_tied_residuals_return_the_common_value():
    cal_true = np.full(7, 0.3)
    lower, upper = call(np.zeros(7), cal_true, [1.0], 0.25)
    # k = ceil(8 * 0.75) = 6; every residual is 0.3.
    same(lower, [0.7], "tied residuals: the threshold is the common residual")
    same(upper, [1.3], "tied residuals: the threshold is the common residual")


def test_large_magnitudes_scale_the_bounds_without_overflow():
    cal_pred, cal_true, test_pred = house_predictions(
        data_rng("conformal-intervals-p01:large-magnitudes"), 20, 3
    )
    base_lower, base_upper = call(cal_pred, cal_true, test_pred, 0.2)
    scale = 1e200
    lower, upper = call(cal_pred * scale, cal_true * scale, test_pred * scale, 0.2)
    same(lower, base_lower * scale, "large magnitudes: squared residuals overflow, use |residual|")
    same(upper, base_upper * scale, "large magnitudes: squared residuals overflow, use |residual|")


def test_small_magnitudes_keep_nonzero_width():
    cal_pred, cal_true, test_pred = house_predictions(
        data_rng("conformal-intervals-p01:small-magnitudes"), 20, 3
    )
    base_lower, base_upper = call(cal_pred, cal_true, test_pred, 0.2)
    scale = 1e-200
    lower, upper = call(cal_pred * scale, cal_true * scale, test_pred * scale, 0.2)
    assert np.all(upper > lower), "tiny residuals must not underflow to a zero-width interval"
    same(
        upper - lower,
        (base_upper - base_lower) * scale,
        "tiny magnitudes: width must scale linearly",
    )


def test_permuting_calibration_points_does_not_change_the_bounds():
    rng = data_rng("conformal-intervals-p01:permutation")
    cal_pred, cal_true, test_pred = house_predictions(rng, 15, 4)
    order = rng.permutation(15)
    lower, upper = call(cal_pred, cal_true, test_pred, 0.3)
    lower2, upper2 = call(cal_pred[order], cal_true[order], test_pred, 0.3)
    same(lower2, lower, "the threshold depends on the sorted scores, not on calibration order")
    same(upper2, upper, "the threshold depends on the sorted scores, not on calibration order")


def test_empty_test_set_returns_empty_float64_arrays():
    lower, upper = call([0.0, 1.0], [0.5, 1.5], np.zeros(0), 0.2)
    assert lower.shape == (0,) and upper.shape == (0,), "no test points means empty bounds"


def test_integer_inputs_are_accepted_and_return_float64():
    lower, upper = call(np.zeros(4, dtype=int), np.array([1, 2, 3, 4]), np.array([0, 10]), 0.5)
    # k = ceil(5 * 0.5) = 3, so the threshold is the 3rd smallest residual, 3.
    same(lower, [-3.0, 7.0], "integer inputs: threshold is the 3rd smallest residual")
    same(upper, [3.0, 13.0], "integer inputs: threshold is the 3rd smallest residual")


@pytest.mark.parametrize(
    ("cal_pred", "cal_true", "test_pred", "alpha"),
    [
        ([], [], [0.0], 0.2),
        ([0.0, 1.0], [0.5], [0.0], 0.2),
        ([0.0, 1.0], [0.5, 1.0], [[0.0]], 0.2),
        ([0.0, 1.0], [0.5, np.nan], [0.0], 0.2),
        ([0.0, 1.0], [0.5, 1.0], [np.inf], 0.2),
        ([0.0, 1.0], [0.5, 1.0], [0.0], 0.0),
        ([0.0, 1.0], [0.5, 1.0], [0.0], 1.0),
        ([0.0, 1.0], [0.5, 1.0], [0.0], -0.1),
        ([0.0, 1.0], [0.5, 1.0], [0.0], 1.5),
        ([0.0, 1.0], [0.5, 1.0], [0.0], float("nan")),
        ([0.0, 1.0], [0.5, 1.0], [0.0], True),
        ([0.0, 1.0], [0.5, 1.0], [0.0], None),
        ([0.0 + 0j, 1.0], [0.5, 1.0], [0.0], 0.2),
    ],
    ids=[
        "empty-calibration",
        "length-mismatch",
        "test-not-1d",
        "nan-in-calibration",
        "inf-in-test",
        "alpha-zero",
        "alpha-one",
        "alpha-negative",
        "alpha-above-one",
        "alpha-nan",
        "alpha-bool",
        "alpha-none",
        "complex-calibration",
    ],
)
def test_invalid_calibration_or_alpha_raises_value_error(cal_pred, cal_true, test_pred, alpha):
    try:
        split_conformal_interval(
            np.asarray(cal_pred), np.asarray(cal_true), np.asarray(test_pred), alpha
        )
    except ValueError:
        return
    except Exception:
        pytest.fail("invalid input must raise ValueError, not another exception", pytrace=False)
    pytest.fail(
        "invalid input must raise ValueError: check the input and alpha guards", pytrace=False
    )


def test_inputs_are_not_mutated():
    cal_pred = np.array([0.0, 1.0, 2.0])
    cal_true = np.array([0.5, 3.0, 1.0])
    test_pred = np.array([4.0])
    before = cal_pred.copy(), cal_true.copy(), test_pred.copy()
    call(cal_pred, cal_true, test_pred, 0.5)
    assert all(
        np.array_equal(a, b) for a, b in zip((cal_pred, cal_true, test_pred), before, strict=True)
    ), "do not sort, shift or write into the caller's arrays"


def test_empirical_coverage_meets_one_minus_alpha_within_a_stated_bound():
    # Exchangeable draws: the model predicts 0 everywhere and the noise is heavy-tailed t(2), so
    # residuals are i.i.d. with no ties. n = 9 calibration points, one test point per trial.
    # With alpha = 0.15 the rank is k = 9, and expected coverage is exactly k / (n + 1) = 0.9
    # for any continuous exchangeable residuals. The 4-sigma tolerance over T = 4000 trials is
    # 4 * sqrt(alpha (1 - alpha) / T) ~ 0.023, so a correct implementation passes with
    # overwhelming probability, and the seed fixes the outcome.
    rng = np.random.default_rng(31)
    alpha, n, trials = 0.15, 9, 4000
    noise = rng.standard_t(2, size=(trials, n + 1))
    zeros = np.zeros(n)
    covered = 0
    for row in noise:
        lower, upper = call(zeros, row[:n], [0.0], alpha)
        covered += int(lower[0] <= row[n] <= upper[0])
    coverage = covered / trials
    tolerance = 4.0 * math.sqrt(alpha * (1 - alpha) / trials)
    assert coverage >= (1 - alpha) - tolerance, (
        "coverage below 1 - alpha: check the rank (use ceil, not floor) and that n + 1 is used"
    )
    assert coverage <= (1 - alpha) + 1.0 / (n + 1) + tolerance, (
        "coverage above the finite-sample ceiling: the rank is too large, "
        "or the interval is too wide"
    )


def test_read_only_input_arrays_are_accepted_and_left_alone():
    cal_pred = np.array([0.0, 1.0, 2.0])
    cal_true = np.array([0.5, -3.0, 1.0])
    test_pred = np.array([4.0])
    for array in (cal_pred, cal_true, test_pred):
        array.setflags(write=False)
    try:
        split_conformal_interval(cal_pred, cal_true, test_pred, 0.5)
    except Exception as exc:
        if "read-only" in str(exc):
            pytest.fail(
                "an input array was written to in place: inputs must not be mutated, so work "
                "on a copy",
                pytrace=False,
            )
        pytest.fail(
            "a valid call failed when its input arrays were read-only: a correct solution "
            "only reads its inputs",
            pytrace=False,
        )
