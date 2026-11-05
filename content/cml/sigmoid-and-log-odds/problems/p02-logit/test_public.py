import numpy as np
import pytest
from submission import logit


def _call(p, mistake):
    """Call logit(p); a crash is reported as the broken contract rule, not as a traceback."""
    try:
        return np.asarray(logit(p), dtype=np.float64)
    except Exception as exc:
        pytest.fail(f"{mistake}; the function raised {type(exc).__name__}: {exc}")


def test_half_and_quarter_values():
    np.testing.assert_allclose(
        _call(np.array([0.5, 0.25, 0.75]), "logit of quarter values failed"),
        [0.0, -np.log(3.0), np.log(3.0)],
        rtol=1e-12,
        atol=1e-15,
        err_msg="wrong log-odds: logit(p) = log(p / (1 - p)), so logit(1/4) = -log 3 and "
        "logit(1/2) = 0",
    )


def test_matrix_shape_is_preserved():
    got = _call(np.array([[0.2, 0.8], [0.5, 0.9]]), "logit of a matrix failed")
    assert got.shape == (2, 2), "shape mismatch: logit must return the shape of its input"


@pytest.mark.parametrize("p", [0.0, 1.0], ids=["lower-boundary", "upper-boundary"])
def test_boundary_probabilities_raise_value_error(p):
    try:
        logit(p)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"boundary probability must raise ValueError; raised {type(exc).__name__}")
    pytest.fail("a probability of exactly 0 or 1 must raise ValueError, not return infinity")
