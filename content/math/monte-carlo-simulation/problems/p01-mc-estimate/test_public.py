import numpy as np
import pytest
from submission import mc_estimate


def _unpack(result):
    """Return (estimate, standard error, lo, hi) as floats, or fail naming the return shape."""
    try:
        estimate, standard_error, (lo, hi) = result
        return float(estimate), float(standard_error), float(lo), float(hi)
    except (TypeError, ValueError) as exc:
        pytest.fail(
            "return (estimate, standard_error, (lo, hi)): a 3-tuple whose last item is a "
            f"2-tuple ({type(exc).__name__})"
        )


def _mc(f, sample, n, *, seed, confidence=0.95):
    """Call the learner on valid inputs. A crash is reported as a mistake, not a traceback."""
    try:
        return mc_estimate(f, sample, n, seed=seed, confidence=confidence)
    except Exception as exc:
        pytest.fail(f"valid inputs must not raise: mc_estimate raised {type(exc).__name__}")


def _raises_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_constant_sample_gives_the_hand_computed_estimate_and_interval():
    estimate, se, lo, hi = _unpack(
        _mc(lambda x: x, lambda rng, n: np.arange(1.0, n + 1), 4, seed=0)
    )
    np.testing.assert_allclose(
        [estimate, se, lo, hi],
        [2.5, 0.6454972243679028, 1.2348486881183405, 3.7651513118816595],
        rtol=1e-9,
        err_msg="draws [1, 2, 3, 4]: the mean is 2.5 and the standard error is sd(n - 1) / 2",
    )


def test_seeded_interval_for_the_square_of_a_uniform_covers_one_third():
    _, se, lo, hi = _unpack(_mc(lambda x: x**2, lambda rng, n: rng.random(n), 1000, seed=0))
    assert lo <= 1 / 3 <= hi, "the 95% interval for E[U^2] should cover 1/3 at this seed"
    assert 0.0 < se < 0.05, "the standard error of 1000 draws of U^2 should be about 0.009"


def test_one_draw_raises_value_error():
    _raises_value_error(
        lambda: mc_estimate(lambda x: x, lambda rng, n: rng.random(n), 1, seed=0),
        "one draw has no sample standard deviation: raise ValueError when n < 2",
    )
