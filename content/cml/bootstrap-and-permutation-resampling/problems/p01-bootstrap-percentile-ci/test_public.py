import numpy as np
import pytest
from submission import bootstrap_percentile_ci


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_example_interval_follows_the_resampling_protocol():
    lo, hi = bootstrap_percentile_ci([1, 2, 3, 4, 5], np.mean, seed=0)
    np.testing.assert_allclose(
        [lo, hi],
        [1.8, 4.2],
        atol=1e-12,
        err_msg="interval for 1..5 with seed 0 differs: draw indices with integers(0, n, (B, n))",
    )


def test_constant_sample_gives_a_point_interval():
    lo, hi = bootstrap_percentile_ci([7, 7, 7], np.mean, seed=0)
    assert (lo, hi) == (7.0, 7.0), "every resample of a constant sample has the same mean"


def test_zero_resamples_raises_value_error():
    _raised_value_error(
        lambda: bootstrap_percentile_ci([1.0, 2.0, 3.0], np.mean, n_resamples=0, seed=0),
        "n_resamples = 0 leaves no replicates to take quantiles of: raise ValueError",
    )
