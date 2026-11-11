import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from submission import numerical_gradient

COORDS = st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False)


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def _gradient_or_nan(f, x):
    """Return the numerical gradient, or an all-NaN vector of shape (n,) if the call failed.

    A crash or a wrong output shape then fails the same assertion as a wrong value, and that
    assertion's message describes the mistake. Shape errors also have their own test below.
    """
    n = np.size(x)
    try:
        got = np.asarray(numerical_gradient(f, x), dtype=np.float64)
    except Exception:
        return np.full(n, np.nan)
    return got if got.shape == (n,) else np.full(n, np.nan)


def _closed_form_rosenbrock_gradient(x):
    g = np.zeros_like(x)
    g[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return g


def _rosenbrock(v):
    return float(np.sum(100.0 * (v[1:] - v[:-1] ** 2) ** 2 + (1.0 - v[:-1]) ** 2))


def _softmax(v):
    e = np.exp(v - np.max(v))
    return e / e.sum()


def _log_sum_exp(v):
    m = np.max(v)
    return float(m + np.log(np.sum(np.exp(v - m))))


def _complex_step_gradient(f, x, h=1e-20):
    """Independent reference: each partial by complex-step differentiation (no subtraction)."""
    out = np.zeros(x.size)
    for i in range(x.size):
        xc = x.astype(complex)
        xc[i] += 1j * h
        out[i] = np.imag(f(xc)) / h
    return out


def test_quadratic_form_gradient_matches_a_plus_a_transpose_times_x():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n = int(rng.integers(1, 6))
        a = rng.normal(size=(n, n))
        x = rng.uniform(-3.0, 3.0, size=n)
        want = (a + a.T) @ x
        got = _gradient_or_nan(lambda v, a=a: float(v @ a @ v), x)
        assert np.all(np.abs(got - want) <= 1e-9 * max(1.0, np.max(np.abs(want)))), (
            "gradient of x^T A x should be (A + A^T) x: perturb one coordinate at a time, "
            "take the symmetric difference divided by 2h, and keep the contract's step. A "
            "one-sided difference is off by about h, far above the tolerance"
        )


def test_log_sum_exp_gradient_is_the_softmax():
    rng = np.random.default_rng(1)
    for _ in range(40):
        n = int(rng.integers(1, 7))
        x = rng.uniform(-5.0, 5.0, size=n)
        want = _softmax(x)
        got = _gradient_or_nan(_log_sum_exp, x)
        assert np.all(np.abs(got - want) <= 1e-7 * max(1.0, np.max(np.abs(want)))), (
            "gradient of log-sum-exp should be the softmax of x: check each partial's "
            "perturbation and the central formula"
        )


def test_rosenbrock_gradient_matches_closed_form():
    rng = np.random.default_rng(2)
    for _ in range(40):
        n = int(rng.integers(2, 5))
        x = rng.uniform(-1.5, 1.5, size=n)
        want = _closed_form_rosenbrock_gradient(x)
        got = _gradient_or_nan(_rosenbrock, x)
        assert np.all(np.abs(got - want) <= 1e-7 * max(1.0, np.max(np.abs(want)))), (
            "gradient of the Rosenbrock function disagrees with its closed form: each partial "
            "must differentiate only with respect to its own coordinate"
        )


def test_every_partial_matches_complex_step_reference_on_composite_function():
    def f(v):
        return np.sum(np.exp(v / 3.0) * np.sin(v)) + v[0] * v[-1]

    rng = np.random.default_rng(3)
    for _ in range(30):
        n = int(rng.integers(2, 6))
        x = rng.uniform(-2.0, 2.0, size=n)
        want = _complex_step_gradient(f, x)
        got = _gradient_or_nan(lambda v: float(np.real(f(v))), x)
        assert np.all(np.abs(got - want) <= 1e-8 * max(1.0, np.max(np.abs(want)))), (
            "a partial derivative disagrees with the complex-step reference: check the "
            "perturbed coordinate and the step size"
        )


@pytest.mark.parametrize("x0", [1.0, -2.5], ids=["positive-point", "negative-point"])
def test_one_dimensional_square_has_slope_two_x0(x0):
    got = _gradient_or_nan(lambda v: float(v[0] ** 2), np.array([x0]))
    assert abs(got[0] - 2.0 * x0) <= 1e-9 * max(1.0, abs(2.0 * x0)), (
        "gradient of x^2 must be 2x0: check the divisor 2h, and use the symmetric difference "
        "(a one-sided one is off by about h here)"
    )


def test_large_coordinates_use_a_step_scaled_to_their_size():
    x = np.array([1e8, -2e8])
    got = _gradient_or_nan(lambda v: float(v @ v), x)
    assert np.all(np.abs(got - 2.0 * x) <= 1e-9 * np.max(np.abs(2.0 * x))), (
        "the step must scale with |x_i| for negative as well as positive coordinates: a fixed "
        "or sign-blind step is lost in round-off at large coordinates"
    )


def test_gradient_of_constant_function_is_zero():
    got = numerical_gradient(lambda v: 4.2, [0.3, -7.0, 2.0])
    assert np.all(np.abs(got) <= 1e-12), (
        "a constant function has zero partials everywhere; a nonzero value means the "
        "difference quotient is not symmetric"
    )


def test_integer_input_returns_float64_vector_of_length_n():
    got = numerical_gradient(lambda v: float(np.sum(v**2)), [1, 2, 3])
    assert isinstance(got, np.ndarray), "return a NumPy array, not a list or a scalar"
    assert got.shape == (3,), (
        f"shape mismatch: expected one partial per coordinate, got {got.shape}"
    )
    assert got.dtype == np.float64, "dtype should be float64 even for integer input"
    np.testing.assert_allclose(got, [2.0, 4.0, 6.0], rtol=1e-9, err_msg="partials of sum x^2")


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "neg-inf"]
)
def test_non_finite_coordinate_raises_value_error(bad):
    _raised_value_error(
        lambda: numerical_gradient(lambda v: float(np.sum(v)), [1.0, bad]),
        "a non-finite coordinate must raise ValueError: check np.isfinite(x) first",
    )


def test_matrix_input_raises_value_error():
    _raised_value_error(
        lambda: numerical_gradient(lambda v: float(np.sum(v)), [[1.0, 2.0], [3.0, 4.0]]),
        "x must be 1-D: raise ValueError for a 2-D input instead of flattening it",
    )


def test_input_array_is_not_mutated():
    x = np.array([1.0, -2.0, 0.5])
    before = x.copy()
    numerical_gradient(lambda v: float(np.sum(v**2)), x)
    np.testing.assert_array_equal(
        x, before, err_msg="input x was modified in place: perturb copies, not x itself"
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(arrays(np.float64, st.integers(min_value=1, max_value=6), elements=COORDS))
def test_random_quadratic_gradient_is_twice_the_point(x):
    got = _gradient_or_nan(lambda v: float(v @ v), x)
    assert np.all(np.abs(got - 2.0 * x) <= 1e-9 * max(1.0, np.max(np.abs(x)))), (
        "gradient of |x|^2 is 2x for every point: check the step and the divisor 2h"
    )


def test_empty_input_raises_value_error_not_an_empty_gradient():
    _raised_value_error(
        lambda: numerical_gradient(lambda v: 0.0, []),
        "an empty x must raise ValueError, not return an empty gradient: check the size "
        "before looping over coordinates",
    )


def test_small_coordinate_gets_the_small_step_not_a_larger_floor():
    # d/dx sin(k x) = k cos(k x); k is large so the step's truncation error is visible
    k = 1e4
    x = np.array([0.3])
    got = _gradient_or_nan(lambda v: float(np.sin(k * v[0])), x)
    want = k * np.cos(k * x)
    assert abs(got[0] - want[0]) <= 1.2e-3 * abs(want[0]), (
        "a coordinate near zero was perturbed by too large a step: the step must follow the "
        "contract's rule for small coordinates, not a larger fixed floor, otherwise the "
        "truncation error of the difference is too big for this fast-varying function"
    )
