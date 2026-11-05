import math

import numpy as np
import pytest
from submission import prior_shift_adjust


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_low_base_rate_pulls_a_confident_probability_down():
    got = prior_shift_adjust([0.8], 0.5, 0.01)
    assert math.isclose(float(got[0]), 0.03883495145631068, rel_tol=1e-12), (
        "the odds must move by the ratio of the prior odds, then convert back to a probability"
    )


def test_higher_base_rate_pulls_a_low_probability_up():
    got = prior_shift_adjust([0.2], 0.1, 0.5)
    assert math.isclose(float(got[0]), 2.25 / 3.25, rel_tol=1e-12), (
        "r = (0.5 / 0.5) / (0.1 / 0.9) = 9, so the odds 0.25 become 2.25"
    )


def test_equal_priors_leave_probabilities_unchanged():
    got = prior_shift_adjust([0.0, 0.5, 1.0], 0.2, 0.2)
    np.testing.assert_allclose(
        got, [0.0, 0.5, 1.0], rtol=0, atol=1e-15, err_msg="equal priors must be the identity"
    )


def test_invalid_prior_raises_value_error():
    _expect_value_error(
        lambda: prior_shift_adjust([0.5], 0.0, 0.1),
        "a prior of 0 was accepted: raise ValueError unless each prior is strictly between 0 and 1",
    )
