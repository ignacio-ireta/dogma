import pytest
from submission import bias_variance


def _call(predictions, f_true, noise_var):
    try:
        return bias_variance(predictions, f_true, noise_var)
    except ValueError:
        raise
    except Exception as exc:
        pytest.fail(f"a valid example raised {type(exc).__name__}: check the per-point arrays")


def test_spread_without_bias_gives_variance_and_the_noise_term():
    bias2, variance, expected = _call([[1.0, 2.0], [3.0, 2.0]], [2.0, 2.0], 0.5)
    assert (bias2, variance, expected) == pytest.approx((0.0, 0.5, 1.0)), (
        "models that average to the truth have zero bias; variance is over the models (divisor B);"
        " expected_mse also adds noise_var"
    )


def test_identical_models_give_all_error_as_bias():
    bias2, variance, expected = _call([[1.0], [1.0], [1.0]], [3.0], 0.0)
    assert (bias2, variance, expected) == pytest.approx((4.0, 0.0, 4.0)), (
        "identical models have no variance; their squared distance to the truth is all bias"
    )


def test_single_exact_model_has_no_error():
    bias2, variance, expected = _call([[1.0, 2.0]], [1.0, 2.0], 0.0)
    assert (bias2, variance, expected) == pytest.approx((0.0, 0.0, 0.0)), (
        "one model that is exactly right has zero bias, zero variance and zero expected error"
    )
