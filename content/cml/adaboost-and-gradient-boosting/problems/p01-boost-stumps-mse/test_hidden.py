from __future__ import annotations

import math
import warnings
from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.ensemble import GradientBoostingRegressor
from submission import boost_stumps_mse

from dogma_data import rng, sample_rows

GAIN_TOLERANCE = 1e-9


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


def _fit(X, y, n_estimators, learning_rate):
    """Call the fit and check the answer's shape. Messages never show the inputs."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = boost_stumps_mse(X, y, n_estimators=n_estimators, learning_rate=learning_rate)
    except ValueError:
        pytest.fail("valid input was rejected: check the input rules in the contract")
    except RuntimeWarning:
        pytest.fail("the fit emitted a NumPy warning: check child counts and numerical stability")
    except Exception as exc:
        pytest.fail(f"valid input crashed the fit ({type(exc).__name__}): check the contract")
    if not isinstance(result, tuple) or len(result) != 2:
        pytest.fail("boost_stumps_mse must return a (predict, train_pred) pair")
    predict, train_pred = result
    train_pred = np.asarray(train_pred, dtype=np.float64)
    if train_pred.shape != (len(X),):
        pytest.fail("shape mismatch: train_pred needs one prediction per row of X")
    return predict, train_pred


def _stump_apply(Z, stump):
    feature, threshold, left, right = stump
    if feature is None:
        return np.full(Z.shape[0], left)
    return np.where(Z[:, feature] <= threshold, left, right)


def _oracle(X, y, n_estimators, learning_rate):
    """Brute force: enumerate every feature and midpoint, measure each gain by direct SSE."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    init = y.mean()
    fitted = np.full(len(y), init)
    stumps = []
    for _ in range(n_estimators):
        residual = y - fitted
        parent_sse = float(np.sum((residual - residual.mean()) ** 2))
        best = None
        for j in range(X.shape[1]):
            for a, b in pairwise(sorted(set(X[:, j].tolist()))):
                t = (a + b) / 2.0
                left = X[:, j] <= t
                right = ~left
                gain = parent_sse
                gain -= float(np.sum((residual[left] - residual[left].mean()) ** 2))
                gain -= float(np.sum((residual[right] - residual[right].mean()) ** 2))
                if best is None or gain > best[0] + GAIN_TOLERANCE:
                    best = (gain, (j, t, residual[left].mean(), residual[right].mean()))
        if best is None:
            stump = (None, None, residual.mean(), residual.mean())
        else:
            stump = best[1]
        stumps.append(stump)
        fitted = fitted + learning_rate * _stump_apply(X, stump)

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        out = np.full(Z.shape[0], init)
        for stump in stumps:
            out = out + learning_rate * _stump_apply(Z, stump)
        return out

    return predict, fitted


def _random_problem(seed, n=30, d=3):
    """n patients and d standardised features of diabetes, with the real disease progression.

    The target is centred and scaled by round numbers near its mean and spread, so the absolute
    tolerances below mean the same as they do for the hand-built cases. Feature values repeat
    (ages, the binary sex column), so equal values share a side of every split.
    """
    gen = rng(f"boost-stumps-p01:problem:{seed}")
    X, y = sample_rows("diabetes", gen, n, d, standardize=True)
    return X, (y - 150.0) / 75.0


def test_matches_a_brute_force_implementation_at_train_and_new_points():
    for seed in range(5):
        X_all, y_all = _random_problem(seed, n=42)  # 30 patients to fit, 12 held out
        X, y, X_new = X_all[:30], y_all[:30], X_all[30:]
        want_predict, want_train = _oracle(X, y, n_estimators=4, learning_rate=0.3)
        got_predict, got_train = _fit(X, y, n_estimators=4, learning_rate=0.3)
        np.testing.assert_allclose(
            got_train,
            want_train,
            rtol=0,
            atol=1e-9,
            err_msg="training predictions disagree with the procedure: check the residuals, the "
            "midpoints, the child means and the order of ties",
        )
        np.testing.assert_allclose(
            got_predict(X_new),
            want_predict(X_new),
            rtol=0,
            atol=1e-9,
            err_msg="predictions at new points disagree: the model must store every stump and "
            "replay it with x <= t going left",
        )


@pytest.mark.parametrize("learning_rate", [1.0, 0.5], ids=["unit-rate", "half-rate"])
def test_agrees_with_scikit_learn_depth_one_gradient_boosting_on_tie_free_data(learning_rate):
    rng = np.random.default_rng(41)
    X = _real_normal(rng, (40, 2))  # continuous values: no ties in any feature
    y = 2.0 * X[:, 0] + _real_normal(rng, 40, scale=0.5)
    reference = GradientBoostingRegressor(
        n_estimators=5,
        learning_rate=learning_rate,
        max_depth=1,
        subsample=1.0,
        max_features=None,
        random_state=0,
    ).fit(X, y)
    _, got = _fit(X, y, n_estimators=5, learning_rate=learning_rate)
    np.testing.assert_allclose(
        got,
        reference.predict(X),
        rtol=0,
        atol=1e-5,
        err_msg="disagrees with depth-one gradient boosting: the same initial mean, residuals, "
        "stumps and learning rate should give the same fit",
    )


def test_predictions_at_training_points_equal_the_returned_train_predictions():
    X, y = _random_problem(7)
    predict, train_pred = _fit(X, y, n_estimators=6, learning_rate=0.4)
    np.testing.assert_allclose(
        predict(X),
        train_pred,
        rtol=0,
        atol=1e-12,
        err_msg="predict on the training rows must reproduce the returned training predictions",
    )


def test_constant_features_leave_the_mean_unchanged_for_every_round():
    X = np.full((8, 2), 3.0)
    y = np.array([1.0, 4.0, 2.0, 2.0, 5.0, 0.0, 3.0, 3.0])
    for n_estimators in (1, 3, 7):
        _, train_pred = _fit(X, y, n_estimators=n_estimators, learning_rate=0.7)
        np.testing.assert_allclose(
            train_pred,
            np.full(8, y.mean()),
            rtol=0,
            atol=1e-12,
            err_msg="with no feature that varies, every stump is constant at the residual mean, "
            "so the model must stay at the mean of y",
        )


def test_exact_tie_across_features_goes_to_the_lowest_feature_index():
    # Feature 0 and feature 1 split the rows into the same two groups with equal gain. The lower
    # index must win. Its prediction at (0, 0) is 0.5 - 0.5 = 0; the other feature would give 1.
    X = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    predict, _ = _fit(X, y, n_estimators=1, learning_rate=1.0)
    got = float(predict(np.array([[0.0, 0.0]]))[0])
    assert abs(got - 0.0) <= 1e-9, "a tie between features must go to the lowest feature index"


def test_exact_tie_within_a_feature_goes_to_the_smallest_threshold():
    # Thresholds 0.5 and 2.5 give the same gain. The smallest must win: x = 0.2 falls left of 0.5
    # (prediction 0.0), while the threshold 2.5 would put it in a leaf worth 2/3.
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    predict, _ = _fit(X, y, n_estimators=1, learning_rate=1.0)
    left = float(predict(np.array([[0.2]]))[0])
    right = float(predict(np.array([[2.7]]))[0])
    assert abs(left - 0.0) <= 1e-9, (
        "a tie within a feature must go to the smallest threshold, not the largest"
    )
    assert abs(right - 2.0 / 3.0) <= 1e-9, (
        "the right leaf after the selected split must predict its own residual mean"
    )


def test_a_point_exactly_on_the_threshold_goes_left():
    # The midpoint of 2 and 3 is 2.5. The contract says x <= t goes left, so 2.5 is in the leaf
    # of the lower group (prediction 0.0), not the upper one (prediction 1.0).
    predict, _ = _fit(
        np.array([[1.0], [2.0], [3.0], [4.0]]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        n_estimators=1,
        learning_rate=1.0,
    )
    got = float(predict(np.array([[2.5]]))[0])
    assert abs(got - 0.0) <= 1e-9, (
        "a value equal to the threshold must go to the left child: the rule is x <= t"
    )


def test_the_learning_rate_scales_the_first_step_linearly():
    X, y = _random_problem(9)
    _, base = _fit(X, y, n_estimators=1, learning_rate=1.0)
    _, half = _fit(X, y, n_estimators=1, learning_rate=0.5)
    init = y.mean()
    np.testing.assert_allclose(
        half - init,
        0.5 * (base - init),
        rtol=0,
        atol=1e-12,
        err_msg="the learning rate must scale each step: the step at rate 0.5 is half the step at "
        "rate 1",
    )


def test_training_error_does_not_increase_with_more_rounds():
    X, y = _random_problem(11)
    previous = math.inf
    for n_estimators in range(1, 7):
        _, train_pred = _fit(X, y, n_estimators=n_estimators, learning_rate=0.5)
        error = float(np.mean((y - train_pred) ** 2))
        assert error <= previous + 1e-12, (
            "training squared error must not increase when a round is added: check that each stump "
            "is fitted to the current residuals"
        )
        previous = error


def test_single_row_predicts_its_own_target():
    predict, train_pred = _fit(
        np.array([[2.0]]), np.array([3.5]), n_estimators=4, learning_rate=0.5
    )
    np.testing.assert_allclose(
        train_pred,
        [3.5],
        atol=1e-12,
        err_msg="one row has no split, so the model stays at its target",
    )
    np.testing.assert_allclose(
        predict(np.array([[9.0]])),
        [3.5],
        atol=1e-12,
        err_msg="a model trained on one row must predict that target at every new point",
    )


def test_inputs_are_not_mutated():
    X, y = _random_problem(12)
    X0, y0 = X.copy(), y.copy()
    _fit(X, y, n_estimators=3, learning_rate=0.5)
    np.testing.assert_array_equal(X, X0, err_msg="X was modified in place")
    np.testing.assert_array_equal(y, y0, err_msg="y was modified in place")


@pytest.mark.parametrize(
    ("X", "y", "n_estimators", "learning_rate", "mistake"),
    [
        (
            np.zeros((3, 1)),
            np.array([0.0, 1.0]),
            2,
            0.5,
            "a length mismatch between X and y must raise ValueError",
        ),
        (
            np.array([[0.0], [np.nan]]),
            np.array([0.0, 1.0]),
            2,
            0.5,
            "a NaN in X must raise ValueError",
        ),
        (
            np.array([[0.0], [1.0]]),
            np.array([0.0, np.inf]),
            2,
            0.5,
            "an infinite target must raise ValueError",
        ),
        (
            np.array([[0.0], [1.0]]),
            np.array([0.0, 1.0]),
            0,
            0.5,
            "zero estimators must raise ValueError",
        ),
        (
            np.array([[0.0], [1.0]]),
            np.array([0.0, 1.0]),
            True,
            0.5,
            "a boolean number of estimators must raise ValueError",
        ),
        (
            np.array([[0.0], [1.0]]),
            np.array([0.0, 1.0]),
            2,
            0.0,
            "a zero learning rate must raise ValueError",
        ),
        (
            np.array([[0.0], [1.0]]),
            np.array([0.0, 1.0]),
            2,
            -0.1,
            "a negative learning rate must raise ValueError",
        ),
        (np.zeros((0, 1)), np.zeros(0), 2, 0.5, "an empty X must raise ValueError"),
    ],
    ids=[
        "length",
        "nan-x",
        "inf-y",
        "no-estimators",
        "bool-estimators",
        "zero-rate",
        "negative-rate",
        "empty",
    ],
)
def test_invalid_inputs_raise_value_error(X, y, n_estimators, learning_rate, mistake):
    try:
        boost_stumps_mse(X, y, n_estimators=n_estimators, learning_rate=learning_rate)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; the function raised {type(exc).__name__} instead")
    pytest.fail(f"{mistake}; the function returned a model instead")


def test_predict_rejects_the_wrong_number_of_columns():
    X, y = _random_problem(13, d=2)
    predict, _ = _fit(X, y, n_estimators=2, learning_rate=0.5)
    try:
        predict(np.zeros((3, 3)))
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"new rows with the wrong width must raise ValueError; raised {type(exc).__name__}"
        )
    pytest.fail("new rows with the wrong width must raise ValueError, not return predictions")


@settings(max_examples=30, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=0, max_value=10_000))
def test_training_error_is_non_increasing_for_small_random_problems(seed):
    X, y = _random_problem(seed, n=12, d=2)
    previous = math.inf
    for n_estimators in (1, 2, 3, 4):
        _, train_pred = _fit(X, y, n_estimators=n_estimators, learning_rate=0.6)
        error = float(np.mean((y - train_pred) ** 2))
        assert error <= previous + 1e-12, (
            "training squared error must not increase when a round is added"
        )
        previous = error


def test_large_finite_features_preserve_the_stump_partition():
    X = np.array([[1e308], [1.4e308]])
    predict, fitted = _fit(X, [0.0, 1.0], n_estimators=1, learning_rate=1.0)
    np.testing.assert_allclose(
        fitted,
        [0.0, 1.0],
        rtol=0,
        atol=1e-12,
        err_msg="midpoint overflow must not send both children into the same stump leaf",
    )
    np.testing.assert_allclose(
        predict(X),
        fitted,
        rtol=0,
        atol=1e-12,
        err_msg="stored stump predictions must agree with training at large finite feature scales",
    )


def test_predict_rejects_a_nan_in_an_otherwise_finite_batch():
    predict, _ = _fit(
        np.array([[0.0], [1.0], [2.0]]), [0.0, 1.0, 2.0], n_estimators=2, learning_rate=0.5
    )
    try:
        predict(np.array([[1.0], [np.nan]]))
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"new rows with a NaN raised {type(exc).__name__}, not ValueError")
    pytest.fail("predict must reject any non-finite value in X_new, not only a batch of all NaN")


def test_predict_rejects_a_batch_that_is_all_nan():
    predict, _ = _fit(
        np.array([[0.0], [1.0], [2.0]]), [0.0, 1.0, 2.0], n_estimators=2, learning_rate=0.5
    )
    try:
        predict(np.array([[np.nan]]))
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a NaN new row raised {type(exc).__name__}, not ValueError")
    pytest.fail("predict must reject a NaN in X_new with ValueError")


def test_target_given_as_a_column_raises_value_error():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([[0.0], [1.0], [2.0]])
    try:
        boost_stumps_mse(X, y, n_estimators=2, learning_rate=0.5)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a two-dimensional y raised {type(exc).__name__}, not ValueError")
    pytest.fail("y must be one-dimensional: a column of targets must raise ValueError, not fit")


def test_a_split_is_only_placed_between_distinct_feature_values():
    # Only one candidate threshold exists: the midpoint 0.5 between the distinct values 0 and 1.
    # Left rows (x = 0) have residual mean -0.25, right rows (x = 1) have residual mean +0.25.
    _, train_pred = _fit(
        np.array([[0.0], [0.0], [1.0], [1.0]]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        n_estimators=1,
        learning_rate=1.0,
    )
    np.testing.assert_allclose(
        train_pred,
        [0.0, 0.0, 0.5, 0.5],
        rtol=0,
        atol=1e-9,
        err_msg="a split was placed between two equal feature values: thresholds are midpoints "
        "between consecutive distinct values only",
    )


def test_infinite_learning_rate_raises_value_error():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    try:
        boost_stumps_mse(X, y, n_estimators=1, learning_rate=math.inf)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"an infinite learning rate must raise ValueError; raised {type(exc).__name__}")
    pytest.fail("an infinite learning rate must raise ValueError, not return a model")


def test_one_target_for_many_rows_raises_value_error():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0])  # one target for four rows
    try:
        boost_stumps_mse(X, y, n_estimators=1, learning_rate=0.5)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"y needs one target per row of X; raised {type(exc).__name__} instead")
    pytest.fail("y needs one target per row of X; the function returned a model instead")
