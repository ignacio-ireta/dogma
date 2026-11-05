import warnings

import numpy as np
import pytest
from submission import log_softmax


def _call(logits, mistake, **kwargs):
    """Call log_softmax with NumPy warnings as errors; a crash is a named contract failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            return np.asarray(log_softmax(logits, **kwargs), dtype=np.float64)
    except RuntimeWarning as exc:
        pytest.fail(f"{mistake}; NumPy warned: {exc}")
    except Exception as exc:
        pytest.fail(f"{mistake}; the function raised {type(exc).__name__}: {exc}")


def test_equal_logits_give_minus_log_two():
    np.testing.assert_allclose(
        _call(np.array([0.0, 0.0]), "equal logits failed"),
        [-np.log(2.0), -np.log(2.0)],
        rtol=1e-12,
        err_msg="two equal logits give probability 1/2 each, so the log is -log 2",
    )


def test_exponentials_of_the_result_sum_to_one():
    got = _call(np.array([1.0, 2.0, 3.0]), "log_softmax of a vector failed")
    assert abs(np.exp(got).sum() - 1.0) <= 1e-12, "exp of log_softmax must sum to 1"


def test_very_negative_logit_keeps_a_finite_log_probability():
    np.testing.assert_allclose(
        _call(np.array([0.0, -800.0]), "a tail logit must keep a finite log-probability"),
        [0.0, -800.0],
        rtol=1e-12,
        atol=1e-12,
        err_msg="log_softmax of [0, -800] is [0, -800]: a log of a rounded probability gives -inf",
    )
