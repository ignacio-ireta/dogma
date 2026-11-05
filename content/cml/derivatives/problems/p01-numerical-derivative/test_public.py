import pytest
from submission import numerical_derivative


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_derivative_of_cubic_at_two_uses_the_power_rule():
    assert numerical_derivative(lambda t: t**3 - 2 * t + 1, 2.0) == pytest.approx(10.0, rel=1e-8), (
        "d/dx (x^3 - 2x + 1) = 3x^2 - 2, which is 10 at x = 2"
    )


def test_central_error_on_cubic_is_exactly_h_squared():
    assert numerical_derivative(lambda t: t**3, 1.0, h=0.1) == pytest.approx(3.01, abs=1e-9), (
        "the central difference of x^3 is 3x^2 + h^2; with x = 1 and h = 0.1 that is 3.01"
    )


def test_unknown_method_raises_value_error():
    _raised_value_error(
        lambda: numerical_derivative(lambda t: t, 0.0, method="sideways"),
        'method "sideways" is not supported: raise ValueError, do not fall back to central',
    )
