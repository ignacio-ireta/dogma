import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from submission import regression_report

from dogma_data import rng as data_rng
from dogma_data import sample_rows

KEYS = {"mae", "mse", "rmse", "r2"}


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


def _house_values_and_predictions(gen, n):
    """Real house values and the predictions of a real linear model for n block groups.

    The model is least squares on median income, house age and latitude (standardised), fitted
    to sixty other california_housing block groups. Its errors are what a real model makes:
    mostly small, a few large, and largest where the value is capped at 5.
    """
    X, value = sample_rows("california_housing", gen, n + 60, standardize=True)
    design = np.column_stack([np.ones(n + 60), X[:, [0, 1, 6]]])
    beta = np.linalg.lstsq(design[n:], value[n:], rcond=None)[0]
    return value[:n], design[:n] @ beta


def _data(seed, n, scale=1.0):
    """Real house values and a real model's predictions, both multiplied by ``scale``."""
    gen = data_rng(f"regression-report-p01:data:{seed}:{n}")
    y_true, y_pred = _house_values_and_predictions(gen, n)
    return scale * y_true, scale * y_pred


def _call(y_true, y_pred):
    """Call the learner's function on valid input, naming the mistake when the call fails."""
    try:
        return regression_report(y_true, y_pred)
    except ValueError:
        pytest.fail(
            "a valid input was rejected with ValueError: check the shape and finiteness rules"
        )
    except Exception as exc:
        pytest.fail(
            f"a valid input raised {type(exc).__name__}: return a dict of floats computed from 1-D "
            "arrays"
        )


def _report(y_true, y_pred):
    """Score a valid pair. The learner gets copies, so one test cannot corrupt another's inputs."""
    report = _call(np.array(y_true, copy=True), np.array(y_pred, copy=True))
    if not isinstance(report, dict) or set(report) != KEYS:
        pytest.fail("the report must have exactly the keys mae, mse, rmse and r2")
    for key in KEYS:
        if type(report[key]) is not float:
            pytest.fail(f"{key} must be a Python float: reduce, then convert the scalar explicitly")
    return report


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


@pytest.mark.parametrize(
    ("n", "seed", "scale"),
    [(10, 0, 1.0), (50, 1, 1.0), (200, 2, 1.0), (30, 3, 1e3), (30, 4, 1e-3)],
    ids=["small", "medium", "large", "big-scale", "small-scale"],
)
def test_matches_scikit_learn_on_random_predictions(n, seed, scale):
    y_true, y_pred = _data(seed, n, scale)
    # The oracle is computed before the learner runs, so a learner that edits its inputs cannot
    # corrupt the reference values.
    mae_ref = mean_absolute_error(y_true, y_pred)
    mse_ref = mean_squared_error(y_true, y_pred)
    r2_ref = r2_score(y_true, y_pred)
    report = _report(y_true, y_pred)
    np.testing.assert_allclose(
        report["mae"],
        mae_ref,
        rtol=1e-9,
        err_msg="MAE differs from the mean of |y_true - y_pred|: check the absolute value and n",
    )
    np.testing.assert_allclose(
        report["mse"],
        mse_ref,
        rtol=1e-9,
        err_msg="MSE differs from the mean squared residual: divide by n, not n - 1",
    )
    np.testing.assert_allclose(
        report["rmse"],
        math.sqrt(mse_ref),
        rtol=1e-9,
        err_msg="RMSE must be the square root of the MSE",
    )
    np.testing.assert_allclose(
        report["r2"],
        r2_ref,
        rtol=1e-9,
        atol=1e-12,
        err_msg="R^2 differs from scikit-learn: it is 1 - SS_res / SS_tot, with SS_tot centred on "
        "the mean of y_true, and it is not clipped",
    )


def test_r2_is_not_clipped_for_very_poor_predictions():
    y_true, _ = _data(11, 40)
    rng = np.random.default_rng(12)
    y_pred = -4.0 * y_true + 10.0 * _real_normal(rng, 40)
    r2_ref = r2_score(y_true, y_pred)
    report = _report(y_true, y_pred)
    assert r2_ref < -1.0, "the fixture must be a very poor fit"
    np.testing.assert_allclose(
        report["r2"],
        r2_ref,
        rtol=1e-9,
        err_msg="R^2 is clipped or mis-scaled: an arbitrarily poor fit gives a large negative R^2",
    )


def test_predicting_the_mean_gives_r2_of_zero():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    report = _report(y_true, np.full(5, y_true.mean()))
    np.testing.assert_allclose(
        report["r2"],
        0.0,
        atol=1e-12,
        err_msg="a constant prediction at the mean of y_true must give R^2 = 0: centre on y_true",
    )


def test_perfect_predictions_give_zero_errors_and_unit_r2():
    y_true = np.array([0.5, -1.0, 2.5, 4.0])
    report = _report(y_true, y_true.copy())
    assert report["mae"] == 0.0, "perfect predictions must give a zero MAE"
    assert report["mse"] == 0.0, "perfect predictions must give a zero MSE"
    assert report["rmse"] == 0.0, "perfect predictions must give a zero RMSE"
    np.testing.assert_allclose(
        report["r2"], 1.0, atol=1e-12, err_msg="perfect fit must give R^2 = 1"
    )


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(0, 10**6), c=st.floats(0.1, 50.0))
def test_scaling_truth_and_predictions_scales_errors_and_keeps_r2(seed, c):
    y_true, y_pred = _data(seed, 25)
    base = _report(y_true, y_pred)
    scaled = _report(c * y_true, c * y_pred)
    np.testing.assert_allclose(
        scaled["mae"],
        c * base["mae"],
        rtol=1e-9,
        err_msg="MAE is in the units of y: it must scale by c",
    )
    np.testing.assert_allclose(
        scaled["rmse"],
        c * base["rmse"],
        rtol=1e-9,
        err_msg="RMSE is in the units of y: it must scale by c",
    )
    np.testing.assert_allclose(
        scaled["mse"],
        c * c * base["mse"],
        rtol=1e-9,
        err_msg="MSE is in squared units: it must scale by c squared",
    )
    np.testing.assert_allclose(
        scaled["r2"],
        base["r2"],
        rtol=1e-9,
        atol=1e-12,
        err_msg="R^2 is unit-free: scaling both vectors must leave it unchanged",
    )


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(0, 10**6), shift=st.floats(-100.0, 100.0))
def test_shifting_both_vectors_leaves_every_score_unchanged(seed, shift):
    y_true, y_pred = _data(seed, 25)
    base = _report(y_true, y_pred)
    moved = _report(y_true + shift, y_pred + shift)
    for key in sorted(KEYS):
        np.testing.assert_allclose(
            moved[key],
            base[key],
            rtol=1e-7,
            atol=1e-9,
            err_msg=f"the {key} score changed under a common shift: it must depend only on "
            "the residuals and the spread of y_true",
        )


@pytest.mark.parametrize("value", [0.0, -2.5, 7.0], ids=["zero", "negative", "positive"])
def test_constant_y_true_raises_value_error(value):
    _expect_value_error(
        lambda: regression_report([value] * 4, [1.0, 2.0, 3.0, 4.0]),
        "every y_true value is equal, so R^2 is undefined: raise ValueError instead of dividing",
    )


def test_single_sample_raises_value_error():
    _expect_value_error(
        lambda: regression_report([5.0], [4.0]),
        "a single sample has zero variance in y_true, so R^2 is undefined: raise ValueError",
    )


def test_empty_input_raises_value_error():
    _expect_value_error(
        lambda: regression_report([], []),
        "empty input was accepted: there is nothing to average, so raise ValueError",
    )


def test_length_mismatch_raises_value_error():
    _expect_value_error(
        lambda: regression_report([1.0, 2.0, 3.0], [1.0, 2.0]),
        "y_true and y_pred of different lengths were accepted: check the lengths first",
    )


def test_two_dimensional_input_raises_value_error():
    y_true = np.arange(6.0).reshape(3, 2)
    _expect_value_error(
        lambda: regression_report(y_true, y_true + 1.0),
        "a 2-D input was accepted: y_true and y_pred must be 1-D",
    )


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([1.0, np.nan, 3.0], [1.0, 2.0, 3.0]),
        ([1.0, np.inf, 3.0], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [1.0, 2.0, np.nan]),
    ],
    ids=["nan-truth", "inf-truth", "nan-prediction"],
)
def test_non_finite_values_raise_value_error(y_true, y_pred):
    _expect_value_error(
        lambda: regression_report(y_true, y_pred),
        "a non-finite value was accepted: reject NaN and infinity before computing",
    )


def test_inputs_are_not_mutated():
    y_true, y_pred = _data(5, 20)
    truth_before, pred_before = y_true.copy(), y_pred.copy()
    _call(y_true, y_pred)
    np.testing.assert_array_equal(
        y_true, truth_before, err_msg="y_true was modified in place: work on a float64 copy"
    )
    np.testing.assert_array_equal(
        y_pred, pred_before, err_msg="y_pred was modified in place: work on a float64 copy"
    )


def test_integer_inputs_give_the_same_scores_as_floats():
    ints = _report([3, -1, 2, 7, 4], [2, 0, 2, 8, 4])
    floats = _report([3.0, -1.0, 2.0, 7.0, 4.0], [2.0, 0.0, 2.0, 8.0, 4.0])
    for key in sorted(KEYS):
        np.testing.assert_allclose(
            ints[key],
            floats[key],
            rtol=1e-12,
            err_msg="integer input gave different scores from the same values as floats: "
            "convert to float64 before dividing",
        )


def test_r2_remains_defined_when_nonconstant_target_squares_underflow():
    truth = np.array([-1e-200, 0.0, 1e-200])
    report = _report(truth, np.zeros(3))
    np.testing.assert_allclose(
        report["r2"],
        0.0,
        atol=1e-12,
        err_msg="tiny nonconstant targets still define R-squared: scale before squaring deviations",
    )


def test_prediction_of_a_different_length_is_rejected_even_when_it_broadcasts():
    _expect_value_error(
        lambda: regression_report([1.0, 2.0, 3.0], [2.0]),
        "y_true and y_pred of different lengths were accepted: check the lengths first",
    )


def test_nearly_constant_target_is_defined_and_a_perfect_fit_scores_one():
    # Two values one float apart: non-constant, so R^2 is defined. Their mean rounds onto the
    # larger value, so the largest deviation from the mean is exactly zero.
    truth = [-1.546255576046831, -1.5462555760468313]
    report = _report(truth, truth)
    np.testing.assert_allclose(
        report["r2"],
        1.0,
        atol=1e-12,
        err_msg="a non-constant y_true must give a defined R^2, even when its values differ only "
        "in the last float bit: a perfect fit scores 1",
    )


def test_repeated_target_values_all_count_toward_the_mean():
    # y_true = [1, 1, 2], y_pred = [1, 2, 2]: residuals are 0, -1, 0.
    # MAE = (0 + 1 + 0) / 3 = 1/3. Each row is one observation, so the repeated 1 counts too.
    report = _report([1.0, 1.0, 2.0], [1.0, 2.0, 2.0])
    np.testing.assert_allclose(
        report["mae"],
        1 / 3,
        rtol=1e-12,
        err_msg="rows whose target value repeats were dropped before scoring: every row counts",
    )
