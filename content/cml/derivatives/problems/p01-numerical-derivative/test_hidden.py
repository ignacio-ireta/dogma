import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import numerical_derivative

from dogma_data import rng as data_rng
from dogma_data import sample_rows

# Smooth test functions with known derivatives: (f, f', sampling interval for x).
CASES = [
    (np.sin, np.cos, (-3.0, 3.0)),
    (np.exp, np.exp, (-2.0, 2.0)),
    (np.log, lambda t: 1.0 / t, (0.2, 5.0)),
    (lambda t: t**3 - 2 * t + 1, lambda t: 3 * t**2 - 2, (-3.0, 3.0)),
]
FINITE = st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False)


def _real_points(what, lo, hi, count):
    """Evaluation points that are real measurements: standardised wine values inside [lo, hi].

    A standardised measurement is a number of standard deviations from the mean of its
    column, so the values crowd around zero and thin out towards three, with repeats where
    the instrument rounded.
    """
    gen = data_rng(f"p01-numerical-derivative:points:{what}")
    values, _ = sample_rows("wine", gen, 178, standardize=True)
    inside = values.ravel()
    inside = inside[(inside >= lo) & (inside <= hi)]
    return inside[:count]


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def _estimate_or_nan(f, x, method="central"):
    """Return the estimate, or NaN if the call raised. One assertion site then reports both."""
    try:
        return numerical_derivative(f, x, method=method)
    except Exception:
        return float("nan")


def test_central_matches_analytic_derivative_on_random_points():
    for case, (f, df, (lo, hi)) in enumerate(CASES):
        for x in _real_points(f"central-{case}", lo, hi, 25):
            want = float(df(x))
            got = _estimate_or_nan(f, float(x))
            assert abs(got - want) <= 1e-8 * max(1.0, abs(want)), (
                "central method is not accurate to 1e-8 on smooth functions: it must use the "
                "symmetric formula (f(x+h) - f(x-h)) / (2h), and its step must balance "
                "truncation against round-off"
            )


def test_central_matches_complex_step_reference_on_composite_function():
    def f(t):
        return np.exp(np.sin(t)) / (1.0 + t * t)

    for x in _real_points("composite", -2.0, 2.0, 25):
        # Complex-step differentiation has no subtraction, so it is exact to machine precision.
        want = float(np.imag(f(float(x) + 1e-20j)) / 1e-20)
        got = numerical_derivative(f, float(x))
        assert abs(got - want) <= 1e-8 * max(1.0, abs(want)), (
            "central difference disagrees with the complex-step reference on a composite "
            "function: check the step size and the formula"
        )


def test_forward_method_meets_its_first_order_accuracy_class():
    for case, (f, df, (lo, hi)) in enumerate(CASES):
        for x in _real_points(f"forward-{case}", lo, hi, 25):
            want = float(df(x))
            got = numerical_derivative(f, float(x), method="forward")
            assert abs(got - want) <= 1e-6 * max(1.0, abs(want)), (
                "forward difference is not accurate to 1e-6 on smooth functions: check the "
                "forward formula (f(x+h) - f(x)) / h and the default step"
            )


def test_central_error_on_a_cubic_is_exactly_h_squared():
    got = numerical_derivative(lambda t: t**3, -2.0, h=0.25)
    assert got == pytest.approx(12.0625, abs=1e-12), (
        "the central difference of x^3 equals 3x^2 + h^2 exactly; a different value means "
        "the formula is wrong or the explicit step h is being ignored"
    )


def test_forward_error_on_a_cubic_is_3xh_plus_h_squared():
    got = numerical_derivative(lambda t: t**3, -2.0, method="forward", h=0.25)
    assert got == pytest.approx(10.5625, abs=1e-12), (
        "the forward difference of x^3 equals 3x^2 + 3xh + h^2 exactly; a different value means "
        "the formula is wrong or the explicit step h is being ignored"
    )


def test_default_step_scales_with_large_x():
    x = 1e8
    got = numerical_derivative(lambda t: t * t, x)
    assert abs(got - 2 * x) <= 1e-9 * 2 * x, (
        "the default step must scale with |x|: a fixed absolute step is too small at large x, "
        "so round-off dominates"
    )


def test_default_step_does_not_shrink_to_zero_near_the_origin():
    got = numerical_derivative(np.exp, 1e-10)
    assert abs(got - 1.0) <= 1e-8, (
        "the derivative of exp at a tiny x is about 1; a step proportional to |x| alone "
        "shrinks toward zero here, and round-off dominates. Use max(1, |x|)"
    )


def test_linear_function_has_its_slope_for_both_methods():
    for x in _real_points("linear", -10.0, 10.0, 20):
        for method in ("forward", "central"):
            got = numerical_derivative(lambda t: 3.0 * t + 2.0, float(x), method=method)
            assert abs(got - 3.0) <= 1e-6, (
                f"a linear function's derivative is its slope for every scheme (method "
                f"{method}); a different value means the difference quotient is wrong"
            )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.lists(FINITE, min_size=1, max_size=4), st.floats(-4.0, 4.0, allow_nan=False))
def test_random_polynomial_derivative_is_accurate(coeffs, x):
    def f(t):
        return sum(c * t**k for k, c in enumerate(coeffs))

    want = sum(k * c * x ** (k - 1) for k, c in enumerate(coeffs) if k > 0)
    scale = sum(abs(c) * abs(x) ** k for k, c in enumerate(coeffs)) + 1.0
    got = _estimate_or_nan(f, x)
    assert abs(got - want) <= 1e-8 * scale, (
        "central method is wrong or raises on a random polynomial: check the symmetric "
        "formula, and that the step is nonzero and finite for every x"
    )


def test_integer_x_is_accepted_as_a_float():
    assert numerical_derivative(lambda t: t * t, 3) == pytest.approx(6.0, rel=1e-8), (
        "an integer x is a valid real number; convert it to float before stepping"
    )


def test_unknown_method_raises_value_error():
    _raised_value_error(
        lambda: numerical_derivative(np.sin, 1.0, method="midpoint"),
        "an unknown method must raise ValueError, not silently use a default scheme",
    )


@pytest.mark.parametrize(
    "bad_x", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "neg-inf"]
)
def test_non_finite_x_raises_value_error(bad_x):
    _raised_value_error(
        lambda: numerical_derivative(np.sin, bad_x),
        "a non-finite x must raise ValueError, not return a NaN estimate",
    )


@pytest.mark.parametrize(
    "bad_h",
    [0.0, -1e-3, float("nan"), float("inf")],
    ids=["zero", "negative", "nan", "inf"],
)
def test_non_positive_or_non_finite_step_raises_value_error(bad_h):
    _raised_value_error(
        lambda: numerical_derivative(np.sin, 1.0, h=bad_h),
        "h must be a positive finite number: raise ValueError for zero, negative, NaN or "
        "infinite steps",
    )


@pytest.mark.parametrize("method", ["central", "forward"], ids=["central", "forward"])
@pytest.mark.parametrize("x", [1e8, -1e8], ids=["large-positive", "large-negative"])
def test_default_step_scales_with_large_x_for_both_schemes_and_signs(method, x):
    # A forward step that ignores |x| rounds x + h back to x, so the quotient is 0. A step
    # that uses x instead of |x| is far too small when x is negative.
    got = numerical_derivative(lambda t: t * t, x, method=method)
    assert abs(got - 2 * x) <= 1e-6 * abs(2 * x), (
        f"the default {method} step must scale with |x| for large x of either sign: a fixed "
        "or sign-blind step is too small, so round-off dominates"
    )


@pytest.mark.parametrize(
    "bad_method",
    ["Central", "FORWARD", " central", "centre"],
    ids=["uppercase", "uppercase-forward", "trailing-space", "other-spelling"],
)
def test_method_name_must_match_exactly(bad_method):
    _raised_value_error(
        lambda: numerical_derivative(np.sin, 1.0, method=bad_method),
        "a method name that is not exactly 'forward' or 'central' must raise ValueError: "
        "no case folding, trimming or aliases",
    )


def test_default_central_step_near_the_origin_is_the_cube_root_of_epsilon():
    # For x**3 the central difference at 0 is exactly h**2, so the default step shows in the result.
    h = float(np.finfo(np.float64).eps) ** (1.0 / 3.0)
    got = numerical_derivative(lambda t: t**3, 0.0)
    assert got == pytest.approx(h * h, abs=1e-12), (
        "the default central step near x = 0 has the wrong size: for x**3 the central difference "
        "at 0 is h squared, so the default step must be the cube root of machine epsilon times "
        "max(1, |x|)"
    )


def test_callable_argument_keeps_its_attributes_after_the_call():
    class Cube:
        def __call__(self, t):
            return t**3

    f = Cube()
    before = dict(vars(f))
    for method in ("forward", "central"):
        numerical_derivative(f, 1.5, method=method, h=0.1)
    assert vars(f) == before, (
        "f must only be evaluated: the call must not add, remove or change attributes of f"
    )


def test_result_is_a_float_and_not_an_array():
    value = numerical_derivative(lambda t: t**3 - 2 * t + 1, 2.0)
    assert isinstance(value, float), (
        "the derivative must be a float: an array that holds one number is another type"
    )
