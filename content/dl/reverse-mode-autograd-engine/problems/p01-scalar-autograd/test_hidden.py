from __future__ import annotations

import numpy as np
import pytest
from submission import Value

from dogma_data import rng as data_rng
from dogma_data import sample_rows


def checked(call, *, allow_none=False):
    try:
        result = call()
    except Exception:
        pytest.fail("valid input failed: honor the documented interface", pytrace=False)
    if allow_none:
        if result is not None:
            pytest.fail("backward must return None", pytrace=False)
        return result
    for value in result if isinstance(result, tuple) else (result,):
        try:
            data, grad = value.data, value.grad
        except Exception:
            pytest.fail("return a scalar node exposing data and grad", pytrace=False)
        if type(data) is not float or type(grad) is not float:
            pytest.fail("node data and grad must be Python floats", pytrace=False)
    return result


def rejects(call):
    try:
        call()
    except ValueError:
        return
    except Exception:
        pytest.fail("invalid input must raise ValueError", pytrace=False)
    pytest.fail("invalid input must raise ValueError", pytrace=False)


def close(actual, expected):
    if type(actual) is not float:
        pytest.fail("node data and grad must be Python floats", pytrace=False)
    if np.shape(actual) != np.shape(expected):
        pytest.fail("output shape differs from contract", pytrace=False)
    if not np.allclose(actual, expected, atol=1e-8, rtol=1e-6):
        pytest.fail("numerical result differs from the independent oracle", pytrace=False)


def test_shared_node_and_duplicate_edge_accumulation():
    x = checked(lambda: Value(2.0))
    q = checked(lambda: x * x)
    y = checked(lambda: q * q + q)
    checked(y.backward, allow_none=True)
    close(x.grad, 36.0)
    close(q.grad, 9.0)
    close(y.grad, 1.0)


def test_finite_difference_random_composition():
    # Twenty pairs of real numbers: two standardised measurements of twenty wines.
    pairs, _ = sample_rows("wine", data_rng("p01-scalar-autograd:pairs"), 20, 2, standardize=True)
    for a, b in pairs:
        x, z = checked(lambda a=a, b=b: (Value(float(a)), Value(float(b))))
        y = checked(lambda x=x, z=z: (x * z + x * x).tanh() + x * z)
        checked(y.backward, allow_none=True)

        def objective(c, d):
            return np.tanh(c * d + c * c) + c * d

        eps = 1e-5
        close(x.grad, (objective(a + eps, b) - objective(a - eps, b)) / (2 * eps))
        close(z.grad, (objective(a, b + eps) - objective(a, b - eps)) / (2 * eps))


def test_repeat_backward_resets_reachable_and_preserves_unreachable():
    x, other = checked(lambda: (Value(3.0), Value(8.0)))
    checked(other.backward, allow_none=True)
    q = checked(lambda: x * x)
    root = checked(lambda: q + q)
    checked(root.backward, allow_none=True)
    checked(root.backward, allow_none=True)
    close(x.grad, 12.0)
    checked(q.backward, allow_none=True)
    close(x.grad, 6.0)
    close(q.grad, 1.0)
    close(root.grad, 1.0)
    close(other.grad, 1.0)


def test_new_grad_is_zero_and_data_is_float():
    x = checked(lambda: Value(2))
    assert isinstance(x.data, float), "scalar data must be float"
    assert isinstance(x.grad, float), "scalar gradient must be float"
    close(x.grad, 0.0)


def test_tanh_saturation_and_zero_derivative():
    for a in (-100.0, 0.0, 100.0):
        x = checked(lambda a=a: Value(a))
        y = checked(x.tanh)
        checked(y.backward, allow_none=True)
        close(y.data, np.tanh(a))
        close(x.grad, 1 - np.tanh(a) ** 2)


def test_degenerate_cancellation_and_tiny_values():
    x, z = checked(lambda: (Value(1e-8), Value(0.0)))
    y = checked(lambda: x * z + x)
    checked(y.backward, allow_none=True)
    close(x.grad, 1.0)
    close(z.grad, 1e-8)


def test_invalid_values_rejected():
    for v in (float("nan"), float("inf"), 1e7, 1 + 2j, 10**400, -(10**400)):
        rejects(lambda v=v: Value(v))


def test_intermediate_arithmetic_obeys_scalar_bounds():
    large = checked(lambda: Value(1e6))
    negative = checked(lambda: Value(-1e6))
    factor = checked(lambda: Value(2.0))
    rejects(lambda: large + large)
    rejects(lambda: negative + negative)
    rejects(lambda: large * factor)
    rejects(lambda: negative * factor)


def test_forward_data_is_read_only():
    x = checked(lambda: Value(2.0))
    squared = checked(lambda: x * x)
    try:
        x.data = 3.0  # type: ignore[misc]  # exercise the read-only data contract
    except (AttributeError, TypeError, ValueError):
        pass
    else:
        pytest.fail("forward data must reject reassignment", pytrace=False)
    close(x.data, 2.0)
    checked(squared.backward, allow_none=True)
    close(x.grad, 4.0)


def test_tanh_of_large_valid_input_stays_finite():
    for a in (1000.0, -1000.0):
        x = checked(lambda a=a: Value(a))
        y = checked(x.tanh)
        assert np.isfinite(y.data), "tanh of a large valid input must stay finite"
        close(y.data, np.tanh(a))


def test_building_operation_keeps_operand_grad():
    x = checked(lambda: Value(2.0))
    y = checked(lambda: x * x)
    checked(y.backward, allow_none=True)
    close(x.grad, 4.0)  # d(x*x)/dx = 2x = 4 at x = 2
    checked(lambda: Value(1.0) + x)
    assert x.grad == 4.0, "building an operation must not change an operand grad"
