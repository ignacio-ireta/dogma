import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_score
from submission import cross_validate

from dogma_data import rng, sample_rows


def _least_squares(X_tr, y_tr, X_te):
    design = np.column_stack([np.ones(len(X_tr)), X_tr])
    beta = np.linalg.lstsq(design, y_tr, rcond=None)[0]
    return np.column_stack([np.ones(len(X_te)), X_te]) @ beta


def _median_model(X_tr, y_tr, X_te):
    return np.full(len(X_te), np.median(y_tr))


def _mean_model(X_tr, y_tr, X_te):
    return np.full(len(X_te), np.mean(y_tr))


def _mse(t, p):
    return float(np.mean((t - p) ** 2))


def _mae(t, p):
    return float(np.mean(np.abs(t - p)))


def _under(t, p):
    """Asymmetric: penalises only the part of the target above the prediction."""
    return float(np.mean(np.maximum(t - p, 0.0)))


MODELS = {"least_squares": _least_squares, "median": _median_model, "mean": _mean_model}
METRICS = {"mse": _mse, "mae": _mae, "under": _under}


def _data(seed, n=23, d=2):
    """n diabetes patients, d standardised features, and the real disease progression.

    The target is centred and scaled by round numbers near its mean and spread.
    """
    gen = rng(f"cross-validate-p02:data:{seed}")
    X, y = sample_rows("diabetes", gen, n, d, standardize=True)
    return X, (y - 150.0) / 75.0


def _folds_for(kind, n, seed):
    if kind == "kfold-plain":
        return list(KFold(n_splits=4).split(np.arange(n)))
    if kind == "kfold-shuffled":
        return list(KFold(n_splits=4, shuffle=True, random_state=seed).split(np.arange(n)))
    # Overlapping, non-partition folds: test sets may repeat rows and skip others.
    return [
        (np.arange(0, 12), np.arange(0, 6)),
        (np.arange(6, n), np.arange(4, 10)),
        (np.arange(0, n, 2), np.arange(n - 3, n)),
    ]


class _FitPredictEstimator(BaseEstimator):
    """Adapts fit_predict to the estimator interface, so cross_val_score scores the same folds."""

    def __init__(self, fit_predict=None):
        self.fit_predict = fit_predict

    def fit(self, X, y):
        self.X_fit_ = np.asarray(X)
        self.y_fit_ = np.asarray(y)
        return self

    def predict(self, X):
        return self.fit_predict(self.X_fit_, self.y_fit_, np.asarray(X))


def _sklearn_scores(fit_predict, X, y, folds, metric):
    estimator = _FitPredictEstimator(fit_predict)
    scorer = lambda est, X_te, y_te: metric(np.asarray(y_te), np.asarray(est.predict(X_te)))  # noqa: E731
    return cross_val_score(estimator, X, y, cv=folds, scoring=scorer)


def _run(fit_predict, X, y, folds, metric):
    """Call the learner's function and check the return shape, naming the mistake if it is off."""
    try:
        scores, mean, std = cross_validate(fit_predict, X, y, folds, metric)
    except ValueError:
        pytest.fail("a valid evaluation raised ValueError: check the fold and prediction handling")
    except Exception as exc:
        pytest.fail(
            f"a valid evaluation raised {type(exc).__name__}: check the indexing of X and y"
        )
    if not isinstance(scores, list) or len(scores) != len(folds):
        pytest.fail("scores must be a list with one float per fold, in fold order")
    if not all(isinstance(s, float) for s in scores):
        pytest.fail("each per-fold score must be a Python float")
    if not (isinstance(mean, float) and isinstance(std, float)):
        pytest.fail("mean and std must be Python floats")
    return scores, mean, std


def _require(condition, mistake):
    if not condition:
        pytest.fail(mistake)


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


@pytest.mark.parametrize(
    ("model", "metric", "fold_kind", "seed"),
    [
        ("least_squares", "mse", "kfold-plain", 0),
        ("least_squares", "under", "kfold-shuffled", 1),
        ("median", "mae", "kfold-shuffled", 2),
        ("mean", "under", "kfold-plain", 3),
        ("least_squares", "mae", "overlapping", 4),
    ],
    ids=[
        "squared-plain",
        "asymmetric-shuffled",
        "median-absolute",
        "mean-asymmetric",
        "overlapping-folds",
    ],
)
def test_per_fold_scores_mean_and_std_match_sklearn_cross_val_score(model, metric, fold_kind, seed):
    X, y = _data(seed)
    folds = _folds_for(fold_kind, len(y), seed)
    fit_predict, score_fn = MODELS[model], METRICS[metric]
    expected = _sklearn_scores(fit_predict, X, y, folds, score_fn)
    scores, mean, std = _run(fit_predict, X, y, folds, score_fn)
    np.testing.assert_allclose(
        scores,
        expected,
        rtol=1e-10,
        atol=1e-12,
        err_msg="per-fold scores differ from cross_val_score: score each fold on its own test rows",
    )
    _require(
        abs(mean - float(np.mean(expected))) <= 1e-10,
        "the mean must be the unweighted mean of the per-fold scores, not a pooled score",
    )
    _require(
        abs(std - float(np.std(expected, ddof=1))) <= 1e-10,
        "the standard deviation must use the sample formula with ddof=1",
    )


def test_standard_deviation_of_four_scores_uses_one_degree_of_freedom():
    counter = {"value": 0}
    folds = [
        (np.array([1, 2, 3]), np.array([0])),
        (np.array([0, 2, 3]), np.array([1])),
        (np.array([0, 1, 3]), np.array([2])),
        (np.array([0, 1, 2]), np.array([3])),
    ]

    def returned_in_order(t, p):
        counter["value"] += 1
        return float(counter["value"])

    _, mean, std = _run(_mean_model, [[0.0]] * 4, [1.0, 2.0, 3.0, 4.0], folds, returned_in_order)
    _require(mean == 2.5, "mean must be the unweighted average of all fold scores")
    _require(
        abs(std - float(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))) < 1e-12,
        "std of the four scores must use ddof=1: divide the squared deviations by k - 1",
    )


def test_metric_receives_the_true_test_targets_first_then_the_predictions():
    X, y = _data(5, n=12)
    folds = _folds_for("kfold-plain", len(y), 5)
    seen = []

    def recording(y_true, y_pred):
        seen.append((np.asarray(y_true).copy(), np.asarray(y_pred).copy()))
        return 0.0

    _run(lambda X_tr, y_tr, X_te: np.zeros(len(X_te)), X, y, folds, recording)
    _require(len(seen) == len(folds), "metric must be called once per fold")
    for (y_true, _), (_, test) in zip(seen, folds, strict=True):
        _require(
            np.array_equal(y_true, y[test]),
            "metric's first argument must be the true targets of the test rows, in order",
        )


def test_fit_predict_is_called_once_per_fold_and_never_sees_a_test_row():
    X = np.arange(20, dtype=float).reshape(20, 1)
    y = np.arange(20, dtype=float)
    folds = _folds_for("kfold-shuffled", 20, 6)
    calls = []

    def fit_predict(X_tr, y_tr, X_te):
        overlap = np.intersect1d(X_tr[:, 0], X_te[:, 0]).size
        calls.append((X_tr.shape[0], overlap))
        return np.zeros(len(X_te))

    _run(fit_predict, X, y, folds, _mse)
    _require(len(calls) == len(folds), "fit_predict must be called exactly once per fold")
    _require(
        all(overlap == 0 for _, overlap in calls),
        "fit_predict received test rows in its training data: fit on X[train] only",
    )
    _require(
        [size for size, _ in calls] == [len(train) for train, _ in folds],
        "fit_predict must receive the training rows of each fold, in fold order",
    )


@pytest.mark.parametrize(
    ("case", "build", "mistake"),
    [
        (
            "one_fold",
            lambda: ([(np.array([2, 3]), np.array([0, 1]))], [1.0, 1.0, 3.0, 3.0]),
            "one fold gives no standard deviation: raise ValueError for fewer than two folds",
        ),
        (
            "no_folds",
            lambda: ([], [1.0, 1.0, 3.0, 3.0]),
            "an empty list of folds must raise ValueError",
        ),
        (
            "length_mismatch",
            lambda: (_two_folds(), [1.0, 1.0, 3.0]),
            "X and y with different numbers of rows must raise ValueError",
        ),
        (
            "index_out_of_range",
            lambda: ([_two_folds()[0], (np.array([2, 3]), np.array([0, 9]))], [1.0] * 4),
            "a test index beyond the last row must raise ValueError",
        ),
        (
            "empty_index",
            lambda: ([(np.array([2, 3]), np.array([], dtype=int)), _two_folds()[0]], [1.0] * 4),
            "an empty index array must raise ValueError",
        ),
        (
            "float_index",
            lambda: ([(np.array([2.0, 3.0]), np.array([0, 1])), _two_folds()[0]], [1.0] * 4),
            "float index arrays must raise ValueError: indices must be integers",
        ),
    ],
    ids=[
        "one-fold",
        "no-folds",
        "length-mismatch",
        "index-out-of-range",
        "empty-index",
        "float-index",
    ],
)
def test_invalid_evaluation_raises_value_error(case, build, mistake):
    folds, y = build()
    X = np.zeros((len(y), 1)) if case != "length_mismatch" else np.zeros((4, 1))
    _expect_value_error(
        lambda: cross_validate(_mean_model, X, y, folds, _mae),
        mistake,
    )


def _two_folds():
    return [(np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3]))]


@pytest.mark.parametrize(
    ("predict_length", "case"),
    [(lambda n: n - 1, "one-short"), (lambda n: 1, "single-value")],
    ids=["one-short", "single-value"],
)
def test_wrong_number_of_predictions_raises_value_error(predict_length, case):
    X, y = _data(7, n=12)
    folds = _folds_for("kfold-plain", len(y), 7)

    def fit_predict(X_tr, y_tr, X_te):
        return np.zeros(predict_length(len(X_te)))

    _expect_value_error(
        lambda: cross_validate(fit_predict, X, y, folds, _mse),
        "fit_predict must return one prediction per test row: a shorter or single-value"
        " array is not broadcast",
    )


def test_two_dimensional_targets_raise_value_error():
    X, y = _data(8, n=12)
    folds = _folds_for("kfold-plain", len(y), 8)
    _expect_value_error(
        lambda: cross_validate(_mean_model, X, y.reshape(-1, 1), folds, _mse),
        "y must be one-dimensional, with one target per row",
    )


def test_inputs_are_not_mutated():
    X, y = _data(9, n=15)
    folds = _folds_for("kfold-shuffled", len(y), 9)
    X_before, y_before = X.copy(), y.copy()
    folds_before = [(tr.copy(), te.copy()) for tr, te in folds]
    cross_validate(_least_squares, X, y, folds, _mse)
    np.testing.assert_array_equal(
        X, X_before, err_msg="X was modified in place: index it, do not write to it"
    )
    np.testing.assert_array_equal(
        y, y_before, err_msg="y was modified in place: index it, do not write to it"
    )
    for (tr, te), (tr0, te0) in zip(folds, folds_before, strict=True):
        _require(
            np.array_equal(tr, tr0) and np.array_equal(te, te0),
            "the fold index arrays were modified in place",
        )


def test_same_inputs_give_the_same_result():
    X, y = _data(10)
    folds = _folds_for("kfold-shuffled", len(y), 10)
    first = cross_validate(_least_squares, X, y, folds, _mse)
    second = cross_validate(_least_squares, X, y, folds, _mse)
    _require(first == second, "identical inputs must give identical scores, mean and std")


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=2, max_value=6), st.integers(min_value=0, max_value=10_000))
def test_summary_matches_numpy_on_random_scores(k, seed):
    n = 3 * k + 5
    X, y = _data(seed, n=n)
    folds = (
        _folds_for("kfold-shuffled", n, seed)
        if k == 4
        else [
            (np.setdiff1d(np.arange(n), np.arange(i, n, k)), np.arange(i, n, k)) for i in range(k)
        ]
    )
    scores, mean, std = _run(_mean_model, X, y, folds, _mae)
    _require(
        abs(mean - float(np.mean(scores))) <= 1e-12,
        "the mean must equal the average of the returned per-fold scores",
    )
    _require(
        abs(std - float(np.std(scores, ddof=1))) <= 1e-12,
        "the standard deviation must equal the ddof=1 deviation of the returned per-fold scores",
    )


def test_row_count_of_X_must_match_y():
    # Every index is in range for y, so only the row-count check can raise here.
    X = np.zeros((5, 1))
    y = np.array([1.0, 1.0, 3.0, 3.0])
    _expect_value_error(
        lambda: cross_validate(_mean_model, X, y, _two_folds(), _mae),
        "X and y with different numbers of rows must raise ValueError",
    )


def test_test_index_equal_to_the_row_count_is_out_of_range():
    X = np.zeros((4, 1))
    y = np.array([1.0, 1.0, 3.0, 3.0])
    folds = [(np.array([2, 3]), np.array([0, 1])), (np.array([2, 3]), np.array([0, 4]))]
    _expect_value_error(
        lambda: cross_validate(_mean_model, X, y, folds, _mae),
        "an index equal to the number of rows is out of range: raise ValueError",
    )


def test_negative_index_is_out_of_range_and_does_not_wrap_around():
    X = np.zeros((4, 1))
    y = np.array([1.0, 1.0, 3.0, 3.0])
    folds = [(np.array([2, 3]), np.array([0, 1])), (np.array([2, 3]), np.array([-1, 0]))]
    _expect_value_error(
        lambda: cross_validate(_mean_model, X, y, folds, _mae),
        "a negative index is outside 0..n-1: raise ValueError instead of wrapping to the end",
    )
