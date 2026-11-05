import numpy as np
import pytest
from submission import cross_validate


def _mean_model(X_tr, y_tr, X_te):
    return [float(np.mean(y_tr))] * len(X_te)


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _evaluate(fit_predict, X, y, folds, metric):
    """Run a valid example, naming the mistake if the learner's function crashes on it."""
    try:
        return cross_validate(fit_predict, X, y, folds, metric)
    except Exception as exc:
        pytest.fail(f"a valid example raised {type(exc).__name__}: check the fold handling")


def test_two_folds_give_per_fold_scores_with_zero_spread():
    folds = [(np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3]))]
    scores, mean, std = _evaluate(
        _mean_model, [[0.0], [0.0], [0.0], [0.0]], [1.0, 1.0, 3.0, 3.0], folds, _mae
    )
    assert scores == [2.0, 2.0] and mean == 2.0 and std == 0.0, (
        "per-fold scores, their mean or their spread are wrong for two equal folds"
    )


def test_three_folds_standard_deviation_uses_one_degree_of_freedom():
    returned = iter([1.0, 2.0, 3.0])
    folds = [
        (np.array([1, 2]), np.array([0])),
        (np.array([0, 2]), np.array([1])),
        (np.array([0, 1]), np.array([2])),
    ]
    scores, mean, std = _evaluate(
        _mean_model,
        [[0.0], [0.0], [0.0]],
        [1.0, 2.0, 3.0],
        folds,
        lambda t, p: next(returned),
    )
    assert scores == [1.0, 2.0, 3.0] and mean == 2.0, "scores or mean in fold order are wrong"
    assert std == pytest.approx(1.0), (
        "the standard deviation must use ddof=1 (sample, not population)"
    )


def test_a_single_fold_raises_value_error():
    folds = [(np.array([2, 3]), np.array([0, 1]))]
    try:
        cross_validate(_mean_model, [[0.0], [0.0], [0.0], [0.0]], [1.0, 1.0, 3.0, 3.0], folds, _mae)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"one fold raised {type(exc).__name__} instead of ValueError")
    pytest.fail("one fold must raise ValueError: a standard deviation needs two scores")
