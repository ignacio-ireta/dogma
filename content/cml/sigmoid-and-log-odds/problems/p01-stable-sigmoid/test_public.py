import warnings

import numpy as np
import pytest
from submission import sigmoid


def _evaluate(z, mistake):
    """Call sigmoid(z) with NumPy overflow and invalid warnings turned into errors."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            return np.asarray(sigmoid(z), dtype=np.float64)
    except RuntimeWarning as exc:
        pytest.fail(f"{mistake}; NumPy warned: {exc}")
    except Exception as exc:
        pytest.fail(f"{mistake}; the function raised {type(exc).__name__}: {exc}")


def test_plain_vector_matches_logistic_values():
    np.testing.assert_allclose(
        _evaluate(np.array([-2.0, 0.0, 2.0]), "sigmoid of a plain vector failed"),
        [0.11920292202211755, 0.5, 0.8807970779778823],
        rtol=1e-12,
        err_msg="wrong values: sigma(z) = 1 / (1 + exp(-z)), so sigma(-2), sigma(0), sigma(2) "
        "are the logistic values; check the sign of the exponent",
    )


def test_zero_gives_exactly_one_half():
    got = _evaluate(0.0, "sigmoid of zero failed")
    assert got == 0.5, "sigma(0) must be exactly 0.5, not a value that merely rounds to it"


def test_two_dimensional_input_keeps_its_shape():
    got = _evaluate(np.array([[1.0, -1.0], [0.0, 3.0]]), "sigmoid of a matrix failed")
    assert got.shape == (2, 2), "shape mismatch: sigmoid must return the shape of its input"


def test_extreme_logits_saturate_without_overflow_warnings():
    np.testing.assert_array_equal(
        _evaluate(np.array([-1e4, 1e4]), "extreme logits must not overflow"),
        [0.0, 1.0],
        err_msg="a logit of magnitude 1e4 must saturate to exactly 0 or 1, with no warning: "
        "exp(-z) overflows for large negative z in the naive formula",
    )
