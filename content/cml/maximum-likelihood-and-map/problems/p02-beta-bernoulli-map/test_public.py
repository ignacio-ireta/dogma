import pytest
from submission import beta_bernoulli_map


def test_interior_mode_for_three_heads_in_ten_flips():
    mle, map_estimate = beta_bernoulli_map(3, 10, 2, 2)
    assert mle == pytest.approx(0.3, abs=1e-12), (
        "the MLE is the observed proportion of heads: 3 / 10"
    )
    assert map_estimate == pytest.approx(1 / 3, abs=1e-12), (
        "MAP should be the posterior mode (a - 1) / (a + b - 2), with a = alpha + k = 5 and "
        "b = beta + n - k = 9"
    )


def test_uniform_prior_gives_map_equal_to_mle():
    mle, map_estimate = beta_bernoulli_map(3, 10, 1, 1)
    assert map_estimate == pytest.approx(mle, abs=1e-12), (
        "with a uniform prior the posterior mode is the observed proportion, so MAP equals MLE"
    )


def test_zero_trials_raises_value_error():
    try:
        beta_bernoulli_map(0, 0, 1, 1)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"an empty sample has no estimate; raised {type(exc).__name__}: {exc}")
    pytest.fail("an empty sample must raise ValueError: trials must be at least 1")
