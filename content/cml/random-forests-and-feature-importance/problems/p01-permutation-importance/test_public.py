import numpy as np
import pytest
from submission import permutation_importance

X_EXAMPLE = np.array([[0.0, 5.0], [1.0, 3.0], [2.0, 9.0], [3.0, 1.0]])
Y_EXAMPLE = np.array([0, 0, 1, 1])


def _predict_first_column(Z):
    return (Z[:, 0] >= 1.5).astype(int)


def _accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _call(**kwargs):
    """Call the function; a crash is reported as the broken contract rule, not as a traceback."""
    try:
        return permutation_importance(
            _predict_first_column, X_EXAMPLE, Y_EXAMPLE, _accuracy, **kwargs
        )
    except Exception as exc:
        pytest.fail(
            f"a valid request must return two arrays; the call raised {type(exc).__name__}: {exc}"
        )


def test_feature_the_model_ignores_has_exactly_zero_drop():
    mean_drop, std_drop = _call(n_repeats=3, seed=0)
    assert mean_drop.shape == (2,) and std_drop.shape == (2,), (
        "shape mismatch: each output needs one entry per feature"
    )
    assert mean_drop[1] == 0.0 and std_drop[1] == 0.0, (
        "a feature the model never reads cannot change the predictions, so its drop is exactly 0"
    )


def test_one_repeat_has_zero_spread_with_population_std():
    _, std_drop = _call(n_repeats=1, seed=0)
    np.testing.assert_array_equal(
        std_drop,
        np.zeros(2),
        err_msg="one repeat has zero spread: the standard deviation must divide by n_repeats, "
        "not by n_repeats - 1",
    )


def test_negative_seed_raises_value_error():
    try:
        permutation_importance(
            _predict_first_column, X_EXAMPLE, Y_EXAMPLE, _accuracy, n_repeats=2, seed=-1
        )
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a negative seed must raise ValueError; raised {type(exc).__name__}")
    pytest.fail("a negative seed must raise ValueError, not return values")
