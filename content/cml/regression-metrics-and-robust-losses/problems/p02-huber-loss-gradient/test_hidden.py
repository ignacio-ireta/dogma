import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import huber
from submission import huber_loss_and_grad

from dogma_data import rng as data_rng
from dogma_data import sample_rows


def _house_values_and_predictions(gen, n):
    """Real house values and the predictions of a real linear model for n block groups.

    The model is least squares on median income, house age and latitude (standardised), fitted
    to sixty other california_housing block groups. Its errors are what a real model makes:
    mostly small, a few large, and largest where the value is capped at 5.
    """
    X, value = sample_rows("california_housing", gen, n + 60, standardize=True)
    design = np.column_stack([np.ones(n + 60), X[:, [0, 1, 6]]])
    beta = np.linalg.lstsq(design[n:], value[n:], rcond=None)[0]
    return value[:n], design[:n] @ beta


def _residuals(seed, n, scale=1.0):
    """Residuals of a real model on real house values: mostly small, a few large.

    Both branches of the loss are exercised, as they are for any real fit.
    """
    gen = data_rng(f"huber-p02:residuals:{seed}:{n}")
    y_true, y_pred = _house_values_and_predictions(gen, n)
    return scale * (y_true - y_pred)


def _call(residuals, delta):
    """Call the learner's function on valid input, naming the mistake when the call fails."""
    try:
        result = huber_loss_and_grad(residuals, delta)
        if not isinstance(result, tuple) or len(result) != 2:
            pytest.fail("return a two-item tuple containing the loss and gradient")
        return result
    except ValueError:
        pytest.fail("a valid input was rejected with ValueError: check the shape and delta rules")
    except Exception as exc:
        pytest.fail(
            f"a valid input raised {type(exc).__name__}: return a float loss and a 1-D float64 "
            "gradient"
        )


def _checked(residuals, delta):
    """Score a valid input on copies, so one test cannot corrupt another's arrays."""
    loss, grad = _call(np.array(residuals, copy=True), delta)
    if not isinstance(loss, float):
        pytest.fail("the loss must be a Python float, not an array: return float(np.mean(...))")
    if not isinstance(grad, np.ndarray) or grad.dtype != np.float64:
        pytest.fail("the gradient must be a float64 NumPy array")
    if np.shape(grad) != (len(residuals),):
        pytest.fail("the gradient must have one entry per residual, shape (n,)")
    return loss, np.asarray(grad, dtype=np.float64)


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


@pytest.mark.parametrize(
    ("seed", "n", "delta"),
    [(0, 10, 1.0), (1, 40, 0.5), (2, 100, 2.0), (3, 25, 0.1), (4, 60, 5.0)],
    ids=["small", "medium", "large", "tight-threshold", "loose-threshold"],
)
def test_loss_matches_scipy_huber_on_random_residuals(seed, n, delta):
    r = _residuals(seed, n)
    expected = float(np.mean(huber(delta, r)))  # computed before the learner runs
    loss, _ = _checked(r, delta)
    np.testing.assert_allclose(
        loss,
        expected,
        rtol=1e-10,
        err_msg="the mean Huber loss differs from scipy.special.huber: check both branches and the "
        "linear offset delta/2",
    )


@pytest.mark.parametrize(
    ("seed", "n", "delta"),
    [(5, 12, 1.0), (6, 30, 0.3), (7, 50, 2.5)],
    ids=["unit-threshold", "small-threshold", "large-threshold"],
)
def test_gradient_matches_central_differences_of_the_mean_loss(seed, n, delta):
    r = _residuals(seed, n)
    h = 1e-6
    numeric = np.empty(n)
    for i in range(n):
        up, down = r.copy(), r.copy()
        up[i] += h
        down[i] -= h
        numeric[i] = (np.mean(huber(delta, up)) - np.mean(huber(delta, down))) / (2 * h)
    _, grad = _checked(r, delta)
    np.testing.assert_allclose(
        grad,
        numeric,
        atol=1e-7,
        err_msg="the gradient is not the derivative of the MEAN loss: each entry is psi(r_i) / n",
    )


def test_inside_the_threshold_the_loss_is_half_the_square():
    r = np.array([-0.9, -0.2, 0.0, 0.4, 0.75])
    loss, grad = _checked(r, 1.0)
    np.testing.assert_allclose(
        loss,
        np.mean(0.5 * r**2),
        rtol=1e-12,
        err_msg="inside |r| <= delta the loss must be r^2 / 2, the squared-error branch",
    )
    np.testing.assert_allclose(
        grad,
        r / r.size,
        rtol=1e-12,
        err_msg="inside |r| <= delta the gradient must be r / n",
    )


def test_outside_the_threshold_the_loss_is_linear_with_offset():
    r = np.array([-3.0, 1.5, 2.25, 7.0])
    delta = 1.25
    loss, grad = _checked(r, delta)
    np.testing.assert_allclose(
        loss,
        np.mean(delta * (np.abs(r) - 0.5 * delta)),
        rtol=1e-12,
        err_msg="outside |r| > delta the loss must be delta * (|r| - delta/2): check the offset",
    )
    np.testing.assert_allclose(
        grad,
        delta * np.sign(r) / r.size,
        rtol=1e-12,
        err_msg="outside |r| > delta the gradient must be delta * sign(r) / n, with the sign kept",
    )


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["positive", "negative"])
def test_residual_exactly_at_the_threshold_matches_both_branches(sign):
    delta = 2.0
    r = np.array([sign * delta])
    loss, grad = _checked(r, delta)
    np.testing.assert_allclose(
        loss,
        0.5 * delta**2,
        rtol=1e-12,
        err_msg="at |r| = delta the quadratic and linear pieces agree: loss must be delta^2 / 2",
    )
    np.testing.assert_allclose(
        grad,
        [sign * delta],
        rtol=1e-12,
        err_msg="at |r| = delta the gradient is continuous: it must equal delta * sign(r)",
    )


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(0, 10**6), delta=st.floats(0.05, 10.0))
def test_gradient_entries_are_bounded_by_delta_over_n(seed, delta):
    r = _residuals(seed, 30, scale=5.0)
    _, grad = _checked(r, delta)
    assert np.all(np.abs(grad) <= delta / r.size * (1 + 1e-12)), (
        "an entry of the gradient exceeds delta / n: outliers must have bounded influence, "
        "so the linear branch has to cap the derivative at delta"
    )


def test_a_huge_outlier_has_the_same_capped_gradient_as_a_moderate_one():
    delta = 1.0
    _, grad = _checked([1e6, 0.2, -0.3], delta)
    np.testing.assert_allclose(
        grad[0],
        delta / 3,
        rtol=1e-12,
        err_msg="a residual of 1e6 must contribute delta / n to the gradient, not a value that "
        "grows with the residual",
    )


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(0, 10**6), c=st.floats(0.1, 20.0))
def test_scaling_residuals_and_delta_together_scales_loss_by_c_squared(seed, c):
    r = _residuals(seed, 20)
    delta = 0.7
    base_loss, base_grad = _checked(r, delta)
    loss, grad = _checked(c * r, c * delta)
    np.testing.assert_allclose(
        loss,
        c * c * base_loss,
        rtol=1e-9,
        err_msg="Huber is homogeneous of degree 2 in (residuals, delta): the loss must scale by "
        "c^2",
    )
    np.testing.assert_allclose(
        grad,
        c * base_grad,
        rtol=1e-9,
        atol=1e-12,
        err_msg="the gradient must scale by c when the residuals and delta are both scaled by c",
    )


def test_loss_is_a_mean_not_a_sum():
    r = _residuals(8, 15)
    loss, grad = _checked(r, 1.0)
    loss_twice, grad_twice = _checked(np.concatenate([r, r]), 1.0)
    np.testing.assert_allclose(
        loss_twice,
        loss,
        rtol=1e-12,
        err_msg="duplicating every residual must leave the mean loss unchanged: the loss is a mean",
    )
    np.testing.assert_allclose(
        grad_twice[: r.size],
        grad / 2,
        rtol=1e-12,
        err_msg="with twice as many residuals each gradient entry must halve (the 1/n factor)",
    )


def test_integer_residuals_give_the_same_result_as_floats():
    ints = np.array([0, 1, -2, 5, -7])
    loss_i, grad_i = _checked(ints, 2)
    loss_f, grad_f = _checked(ints.astype(float), 2.0)
    assert isinstance(loss_i, float), "the loss must be a Python float for integer input"
    np.testing.assert_allclose(
        loss_i,
        loss_f,
        rtol=1e-12,
        err_msg="integer residuals gave a different loss from the same values as floats: "
        "convert to float64 before computing",
    )
    np.testing.assert_allclose(
        grad_i, grad_f, rtol=1e-12, err_msg="integer input gave a different gradient"
    )


def test_empty_residuals_raise_value_error():
    _expect_value_error(
        lambda: huber_loss_and_grad([], 1.0),
        "empty residuals were accepted: there is no mean to take, so raise ValueError",
    )


@pytest.mark.parametrize(
    "residuals",
    [[1.0, np.nan, 2.0], [np.inf, 0.5]],
    ids=["nan", "infinity"],
)
def test_non_finite_residuals_raise_value_error(residuals):
    _expect_value_error(
        lambda: huber_loss_and_grad(residuals, 1.0),
        "a non-finite residual was accepted: reject NaN and infinity before computing",
    )


@pytest.mark.parametrize(
    "delta", [0.0, -1.0, np.nan, np.inf], ids=["zero", "negative", "nan", "infinite"]
)
def test_invalid_delta_raises_value_error(delta):
    _expect_value_error(
        lambda: huber_loss_and_grad([0.5, 2.0], delta),
        "a delta that is not a finite number greater than 0 was accepted: raise ValueError",
    )


def test_two_dimensional_residuals_raise_value_error():
    _expect_value_error(
        lambda: huber_loss_and_grad(np.ones((3, 2)), 1.0),
        "a 2-D residual array was accepted: residuals must be 1-D",
    )


def test_inputs_are_not_mutated_and_gradient_is_a_new_array():
    r = _residuals(9, 20)
    before = r.copy()
    _, grad = _call(r, 0.8)
    np.testing.assert_array_equal(
        r,
        before,
        err_msg="the residuals array was modified: write the gradient into a new array",
    )
    assert not np.shares_memory(grad, r), "the gradient must not share memory with the input"


def test_large_linear_branch_avoids_unused_square_and_mean_overflow():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        loss, grad = _checked(np.full(3, 1e308), 1.0)
    np.testing.assert_allclose(
        loss,
        1e308,
        rtol=1e-12,
        err_msg="a linear-branch mean must avoid unused squares and overflowing sums",
    )
    np.testing.assert_allclose(
        grad,
        np.full(3, 1.0 / 3),
        atol=1e-12,
        err_msg="a huge residual must keep the capped mean-loss gradient",
    )
