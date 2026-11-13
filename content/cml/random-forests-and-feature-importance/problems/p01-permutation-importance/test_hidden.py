from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import permutation_importance

from dogma_data import rng, sample_rows

WEIGHTS = np.array([2.0, -1.0, 0.0])  # the model reads columns 0 and 1; column 2 is ignored


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


def _predict(Z):
    return (np.asarray(Z) @ WEIGHTS > 0.0).astype(int)


def _accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _problem(seed, n=40, d=3):
    """Semi-synthetic: real standardised breast_cancer columns, labels from the model's rule.

    The columns are strongly correlated, as real features are. The labels follow the fixed
    model's own rule plus noise, so the model is informative and each drop has a known cause.
    """
    gen = rng(f"permutation-importance-p01:problem:{seed}")
    X, _ = sample_rows("breast_cancer", gen, n, d, standardize=True)
    y = (X @ WEIGHTS[:d] + _real_normal(gen, n, scale=0.5) > 0.0).astype(int)
    return X, y


def _call(X, y, predict=_predict, n_repeats=4, seed=0):
    """Call the function and check the answer's shape. Messages never show the inputs."""
    try:
        result = permutation_importance(predict, X, y, _accuracy, n_repeats=n_repeats, seed=seed)
    except ValueError:
        pytest.fail("valid input was rejected: check the input rules in the contract")
    except Exception as exc:
        pytest.fail(f"valid input crashed the call ({type(exc).__name__}): check the contract")
    if not isinstance(result, tuple) or len(result) != 2:
        pytest.fail("permutation_importance must return a (mean_drop, std_drop) pair")
    mean_drop, std_drop = (np.asarray(part, dtype=np.float64) for part in result)
    if mean_drop.shape != (X.shape[1],) or std_drop.shape != (X.shape[1],):
        pytest.fail("shape mismatch: each output needs one entry per feature")
    return mean_drop, std_drop


def _reference(predict, X, y, n_repeats, seed):
    """An independent loop that follows the protocol word for word, with a fresh copy per draw."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    baseline = _accuracy(y, predict(X))
    n, d = X.shape
    drops = np.zeros((d, n_repeats))
    for feature in range(d):
        for repeat in range(n_repeats):
            order = rng.permutation(n)
            shuffled = X.copy()
            shuffled[:, feature] = X[order, feature]
            drops[feature, repeat] = baseline - _accuracy(y, predict(shuffled))
    return drops.mean(axis=1), drops.std(axis=1)


def _expect_value_error(mistake, *args, **kwargs):
    try:
        permutation_importance(*args, **kwargs)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; the function raised {type(exc).__name__} instead")
    pytest.fail(f"{mistake}; the function returned values instead")


def test_matches_an_independent_loop_that_follows_the_protocol():
    for seed in range(3):
        X, y = _problem(seed)
        want_mean, want_std = _reference(_predict, X, y, n_repeats=4, seed=seed + 10)
        got_mean, got_std = _call(X, y, n_repeats=4, seed=seed + 10)
        np.testing.assert_allclose(
            got_mean,
            want_mean,
            rtol=0,
            atol=1e-12,
            err_msg="mean drop disagrees with the protocol: one generator for the whole run, "
            "draws in feature-then-repeat order, and drop = baseline minus permuted score",
        )
        np.testing.assert_allclose(
            got_std,
            want_std,
            rtol=0,
            atol=1e-12,
            err_msg="standard deviation disagrees: use the population standard deviation "
            "(ddof = 0) over the repeats",
        )


def test_feature_the_model_ignores_has_exactly_zero_mean_and_spread():
    X, y = _problem(4)
    mean_drop, std_drop = _call(X, y, n_repeats=5, seed=1)
    assert mean_drop[2] == 0.0 and std_drop[2] == 0.0, (
        "a feature the model never reads must have exactly zero drop and spread"
    )


def test_feature_the_model_uses_has_a_positive_mean_drop():
    X, y = _problem(5)
    mean_drop, _ = _call(X, y, n_repeats=5, seed=2)
    assert mean_drop[0] > 0.0, (
        "a feature the model relies on must lower the score when shuffled: check the drop sign"
    )
    assert mean_drop[0] > mean_drop[2], (
        "the used feature must show more importance than the ignored one"
    )


def test_harmful_feature_has_a_negative_drop_that_is_kept_negative():
    # The predictions are wrong everywhere, so the baseline accuracy is 0. Any shuffle can only
    # match the target more often, so the drop is negative. Taking absolute values hides this.
    X, _ = sample_rows(
        "breast_cancer", rng("permutation-importance-p01:13"), 40, 2, standardize=True
    )
    y = 1 - (X[:, 0] > 0.0).astype(int)

    def predict_first(Z):
        return (np.asarray(Z)[:, 0] > 0.0).astype(int)

    mean_drop, _ = _call(X, y, predict=predict_first, n_repeats=4, seed=3)
    assert mean_drop[0] < 0.0, (
        "a feature whose shuffling improves the score must give a negative drop: keep the sign"
    )


def test_single_repeat_has_zero_spread():
    X, y = _problem(6)
    _, std_drop = _call(X, y, n_repeats=1, seed=3)
    np.testing.assert_array_equal(
        std_drop,
        np.zeros(3),
        err_msg="a single repeat must have zero spread: divide by n_repeats (ddof = 0), "
        "not by n_repeats - 1",
    )


def test_same_seed_repeats_and_different_seed_changes_the_draws():
    X, y = _problem(7)
    first, _ = _call(X, y, n_repeats=4, seed=5)
    again, _ = _call(X, y, n_repeats=4, seed=5)
    np.testing.assert_array_equal(first, again, err_msg="the same seed must give the same result")
    # Accuracy moves in steps of 1/n, and real columns are correlated, so two seeds can give the
    # same mean drops by chance. Six other seeds all giving the same drops is not chance.
    others = [_call(X, y, n_repeats=4, seed=seed)[0] for seed in range(6, 12)]
    assert any(not np.array_equal(first, other) for other in others), (
        "a different seed must change the permutations, so the drops change"
    )


def test_inputs_are_not_mutated():
    X, y = _problem(8)
    X_before, y_before = X.copy(), y.copy()
    _call(X, y, n_repeats=3, seed=4)
    np.testing.assert_array_equal(X, X_before, err_msg="X was modified: permute a copy of X")
    np.testing.assert_array_equal(y, y_before, err_msg="y was modified in place")


def test_zero_repeats_raise_value_error():
    X, y = _problem(9)
    _expect_value_error(
        "zero repeats must raise ValueError: n_repeats must be at least 1",
        _predict,
        X,
        y,
        _accuracy,
        n_repeats=0,
        seed=0,
    )


def test_non_integer_repeats_raise_value_error():
    X, y = _problem(9)
    _expect_value_error(
        "a float number of repeats must raise ValueError: n_repeats must be an integer",
        _predict,
        X,
        y,
        _accuracy,
        n_repeats=2.0,
        seed=0,
    )


def test_negative_seed_raises_value_error():
    X, y = _problem(9)
    _expect_value_error(
        "a negative seed must raise ValueError: seed must be non-negative",
        _predict,
        X,
        y,
        _accuracy,
        n_repeats=2,
        seed=-3,
    )


def test_one_dimensional_x_raises_value_error():
    _expect_value_error(
        "a one-dimensional X must raise ValueError: X must be a two-dimensional array",
        _predict,
        np.array([0.0, 1.0, 2.0]),
        np.array([0, 1, 1]),
        _accuracy,
        n_repeats=2,
        seed=0,
    )


def test_non_finite_x_raises_value_error():
    X, y = _problem(10)
    X[3, 1] = np.nan
    _expect_value_error(
        "a NaN in X must raise ValueError: X must contain only finite numbers",
        _predict,
        X,
        y,
        _accuracy,
        n_repeats=2,
        seed=0,
    )


def test_mismatched_targets_raise_value_error():
    X, y = _problem(11)
    _expect_value_error(
        "y with a different length from X must raise ValueError",
        _predict,
        X,
        y[:-1],
        _accuracy,
        n_repeats=2,
        seed=0,
    )


def test_predict_with_wrong_output_length_raises_value_error():
    X, y = _problem(12)

    def short_predict(Z):
        return np.zeros(len(Z) - 1)

    def zip_accuracy(y_true, y_pred):
        # zip truncates silently, so only an explicit check catches the wrong length
        return sum(a == b for a, b in zip(y_true, y_pred, strict=False)) / len(y_true)

    try:
        permutation_importance(short_predict, X, y, zip_accuracy, n_repeats=2, seed=0)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            "a predict that returns the wrong number of predictions must raise ValueError; "
            f"the function raised {type(exc).__name__} instead"
        )
    pytest.fail(
        "a predict that returns the wrong number of predictions must raise ValueError, "
        "not return values"
    )


@settings(max_examples=30, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=0, max_value=10_000))
def test_ignored_feature_is_exactly_zero_for_any_seed(seed):
    X, y = _problem(seed, n=25, d=2)

    def only_first(Z):
        return (np.asarray(Z)[:, 0] > 0.0).astype(int)

    mean_drop, std_drop = _call(X, y, predict=only_first, n_repeats=3, seed=seed)
    assert mean_drop[1] == 0.0 and std_drop[1] == 0.0, (
        "a feature the predictions do not read must have exactly zero drop and spread"
    )


def test_single_row_table_is_accepted_and_has_no_drop():
    X = np.array([[0.5, -1.0, 2.0]])
    y = np.array([1])
    mean_drop, std_drop = _call(X, y, n_repeats=3, seed=2)
    np.testing.assert_array_equal(
        mean_drop,
        np.zeros(3),
        err_msg="one row is a valid table: with one row a permutation changes nothing, so the "
        "drop is zero",
    )
    np.testing.assert_array_equal(std_drop, np.zeros(3), err_msg="one row: zero spread expected")


def test_single_feature_table_is_accepted_and_gets_one_entry():
    X, _ = sample_rows(
        "breast_cancer", rng("permutation-importance-p01:21"), 30, 1, standardize=True
    )
    y = (X[:, 0] > 0.0).astype(int)

    def only_feature(Z):
        return (np.asarray(Z)[:, 0] > 0.0).astype(int)

    mean_drop, _ = _call(X, y, predict=only_feature, n_repeats=3, seed=4)
    assert mean_drop.shape == (1,), "a one-feature table must return one entry"
    assert mean_drop[0] > 0.0, (
        "a single feature the model relies on must lower the score when shuffled"
    )


def test_empty_rows_raise_value_error():
    _expect_value_error(
        "a table with no rows must raise ValueError: X must be non-empty",
        _predict,
        np.empty((0, 3)),
        np.empty(0),
        _accuracy,
        n_repeats=2,
        seed=0,
    )


def test_zero_feature_table_raises_value_error():
    _expect_value_error(
        "a table with no feature columns must raise ValueError: X must be non-empty",
        lambda Z: np.zeros(len(Z), dtype=int),
        np.empty((4, 0)),
        np.array([0, 1, 0, 1]),
        _accuracy,
        n_repeats=2,
        seed=0,
    )


def test_two_dimensional_y_raises_value_error():
    X, y = _problem(13)
    _expect_value_error(
        "a y that is a column, not a 1-D vector, must raise ValueError: y must be 1-D",
        _predict,
        X,
        y.reshape(-1, 1),
        _accuracy,
        n_repeats=2,
        seed=0,
    )


def test_target_length_mismatch_raises_even_when_metric_truncates():
    X, y = _problem(14)

    def zip_accuracy(y_true, y_pred):
        # zip truncates silently, so only an explicit length check raises
        return sum(a == b for a, b in zip(y_true, y_pred, strict=False)) / len(y_true)

    _expect_value_error(
        "y with a different length from X must raise ValueError, whatever the metric does",
        _predict,
        X,
        y[:-1],
        zip_accuracy,
        n_repeats=2,
        seed=0,
    )


def test_non_integer_seed_raises_value_error():
    X, y = _problem(15)
    _expect_value_error(
        "a fractional seed must raise ValueError: seed must be a non-negative integer",
        _predict,
        X,
        y,
        _accuracy,
        n_repeats=2,
        seed=2.5,
    )


def test_metric_is_called_with_true_targets_first():
    X = np.array([[0.0, 5.0], [1.0, 3.0], [2.0, 9.0], [3.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    calls = []

    def metric(y_true, y_pred):
        calls.append((np.asarray(y_true), np.asarray(y_pred)))
        return _accuracy(y_true, y_pred)

    permutation_importance(
        lambda Z: np.ones(np.asarray(Z).shape[0]), X, y, metric, n_repeats=2, seed=0
    )
    assert calls, "the metric was never called, so no score was computed"
    for y_true, _ in calls:
        assert np.array_equal(y_true, y), (
            "the metric got its arguments in the wrong order: the true targets must come first"
        )


def test_plain_list_predictions_are_accepted():
    X = np.array([[0.0, 5.0], [1.0, 3.0], [2.0, 9.0], [3.0, 1.0]])
    y = np.array([0, 0, 1, 1])

    def predict_list(Z):
        return [int(v >= 1.5) for v in np.asarray(Z)[:, 0]]

    mean_drop, _ = _call(X, y, predict=predict_list, n_repeats=2)
    assert mean_drop[1] == 0.0, (
        "a predict that returns a plain list is valid, and a feature the model never reads "
        "cannot change the predictions, so its drop is exactly 0"
    )


def test_huge_finite_entries_are_accepted_and_ignored_column_has_no_drop():
    # Only NaN and infinity are invalid in X, so a very large finite entry must be kept as it is.
    # The model ignores column 2 (weight 0), so permuting it cannot change any prediction: its
    # drop is exactly 0 for every correct solution, whatever the draws.
    X = np.array([[1e300, 0.5, 0.2], [-1e300, 0.1, -0.4], [1.0, -2.0, 3.0], [0.3, 0.7, -1.1]])
    y = np.array([1, 0, 1, 0])
    try:
        mean_drop, std_drop = permutation_importance(_predict, X, y, _accuracy, n_repeats=3, seed=0)
    except ValueError:
        pytest.fail(
            "a very large but finite entry of X was rejected: the input rule only excludes "
            "NaN and infinity, so values beyond the range of a smaller float type must be kept"
        )
    mean_drop = np.asarray(mean_drop, dtype=np.float64)
    std_drop = np.asarray(std_drop, dtype=np.float64)
    assert np.all(np.isfinite(mean_drop)) and np.all(np.isfinite(std_drop)), (
        "every drop must be a finite number when X is finite"
    )
    assert mean_drop[2] == 0.0 and std_drop[2] == 0.0, (
        "a feature the model ignores must have exactly zero mean drop and zero spread"
    )


def test_list_inputs_give_the_same_result_as_arrays():
    X, y = _problem(7, n=12, d=3)
    kw = {"n_repeats": 3, "seed": 0}
    try:
        from_lists = permutation_importance(_predict, X.tolist(), y.tolist(), _accuracy, **kw)
    except Exception:
        pytest.fail(
            "a valid call with Python lists failed: the inputs are array-like, so convert them "
            "with np.asarray before using array methods",
            pytrace=False,
        )
    from_arrays = permutation_importance(_predict, X, y, _accuracy, **kw)
    for got, want in zip(from_lists, from_arrays, strict=True):
        np.testing.assert_allclose(
            got,
            want,
            rtol=1e-9,
            atol=1e-12,
            err_msg="lists and arrays of the same numbers gave different results",
        )
