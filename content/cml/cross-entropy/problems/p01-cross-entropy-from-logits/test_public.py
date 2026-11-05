import numpy as np
import pytest
from submission import cross_entropy_from_logits


def _call(logits, targets, mistake):
    """Call the function; a crash is reported as the broken contract rule, not as a traceback."""
    try:
        return cross_entropy_from_logits(logits, targets)
    except ValueError as exc:
        pytest.fail(f"{mistake}; the function raised ValueError: {exc}")
    except Exception as exc:
        pytest.fail(f"{mistake}; the function raised {type(exc).__name__}: {exc}")


def test_uniform_logits_give_log_two_and_antisymmetric_gradient():
    loss, grad = _call(np.zeros((2, 2)), np.array([0, 1]), "uniform logits failed")
    assert abs(loss - np.log(2.0)) <= 1e-12, (
        "uniform logits give probability 1/2 to each true class, so the mean loss is log 2"
    )
    np.testing.assert_allclose(
        grad,
        [[-0.25, 0.25], [0.25, -0.25]],
        atol=1e-12,
        err_msg="gradient must be (softmax - onehot) / n, so each row is (p - onehot) / 2",
    )


def test_skewed_logits_match_the_log_sum_exp_value():
    loss, grad = _call(np.array([[2.0, 1.0, 0.1]]), np.array([0]), "skewed logits failed")
    assert abs(loss - 0.41703001627783376) <= 1e-9, (
        "loss of one row: logsumexp(z) - z[target], here with z = (2, 1, 0.1) and target 0"
    )
    assert abs(grad.sum()) <= 1e-12, "each row of the gradient must sum to zero"


def test_integer_targets_outside_the_classes_raise_value_error():
    try:
        cross_entropy_from_logits(np.array([[0.0, 1.0]]), np.array([0.0]))
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"a float target must raise ValueError; the function raised {type(exc).__name__}"
        )
    pytest.fail("a float target array must raise ValueError: class indices must be integers")
