import numpy as np
import pytest
from submission import sample_exponential


def _raises_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_seed_zero_draws_match_the_inverse_cdf_of_the_uniforms():
    draws = sample_exponential(2.0, 3, np.random.default_rng(0))
    np.testing.assert_allclose(
        draws,
        [0.50662345, 0.15720931, 0.0209183],
        rtol=1e-7,
        err_msg="draws differ from -log1p(-u) / rate applied to rng.random(3) with seed 0",
    )


def test_zero_draws_give_an_empty_float_array():
    draws = sample_exponential(1.0, 0, np.random.default_rng(0))
    assert isinstance(draws, np.ndarray), "return a NumPy array, not a list"
    assert draws.shape == (0,), "zero draws should give an array of shape (0,)"
    assert draws.dtype == np.float64, "draws should be float64 even when there are none"


def test_zero_rate_raises_value_error():
    _raises_value_error(
        lambda: sample_exponential(0.0, 5, np.random.default_rng(0)),
        "a rate of zero has no exponential distribution: raise ValueError",
    )
