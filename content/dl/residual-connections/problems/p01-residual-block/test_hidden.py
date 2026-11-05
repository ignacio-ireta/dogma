# Hidden tests: oracle comparisons with torch.nn.functional, autograd checks of the identity term,
# and degenerate, dtype and mutation cases. Messages name the kind of mistake, never the values.
import functools

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 - oracle only
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import residual_block

DT = torch.float64


def checked(call):
    try:
        out = call()
    except Exception:
        pytest.fail(
            "valid input raised an exception: honor the documented interface", pytrace=False
        )
    if not isinstance(out, torch.Tensor):
        pytest.fail("return one tensor of shape (..., d_out)", pytrace=False)
    return out


def rejects(call):
    try:
        call()
    except ValueError:
        return
    except Exception:
        pytest.fail(
            "invalid input must raise ValueError, not another exception type", pytrace=False
        )
    pytest.fail("invalid input must raise ValueError but returned a result", pytrace=False)


def oracle_branch(x, w1, b1, w2, b2):
    # Independent oracle: torch.nn.functional dense layers and ReLU.
    return F.linear(F.relu(F.linear(x, w1.T, b1)), w2.T, b2)


def oracle_block(x, w1, b1, w2, b2, w_shortcut=None):  # noqa: PLR0917 - mirrors the contract
    shortcut = x if w_shortcut is None else F.linear(x, w_shortcut.T)
    return shortcut + oracle_branch(x, w1, b1, w2, b2)


def random_weights(d_in, h, d_out, seed, dtype=DT):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(d_in, h, generator=g, dtype=dtype),
        torch.randn(h, generator=g, dtype=dtype),
        torch.randn(h, d_out, generator=g, dtype=dtype),
        torch.randn(d_out, generator=g, dtype=dtype),
    )


def close(actual, expected, message):
    """Compare with a message that names the mistake; the first line is what learners see."""
    if tuple(actual.shape) != tuple(expected.shape):
        pytest.fail(f"output shape differs from contract: {message}", pytrace=False)
    if not torch.allclose(actual, expected, atol=1e-9, rtol=1e-9):
        pytest.fail(message, pytrace=False)


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(st.integers(1, 5), st.integers(1, 6), st.integers(0, 10_000))
def test_identity_block_matches_functional_oracle(d, h, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 3, d, generator=g, dtype=DT)
    w1, b1, w2, b2 = random_weights(d, h, d, seed + 1)
    got = checked(lambda: residual_block(x, w1, b1, w2, b2))
    close(
        got,
        oracle_block(x, w1, b1, w2, b2),
        "output differs from the oracle: identity shortcut on batched input",
    )


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(st.integers(1, 4), st.integers(1, 4), st.integers(1, 5), st.integers(0, 10_000))
def test_projection_block_matches_functional_oracle(d_in, h, d_out, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(3, d_in, generator=g, dtype=DT)
    w1, b1, w2, b2 = random_weights(d_in, h, d_out, seed + 1)
    ws = torch.randn(d_in, d_out, generator=g, dtype=DT)
    got = checked(lambda: residual_block(x, w1, b1, w2, b2, ws))
    close(
        got,
        oracle_block(x, w1, b1, w2, b2, ws),
        "output differs from the oracle: projection shortcut with width change",
    )


def test_projection_is_used_even_when_widths_match():
    x = torch.tensor([[1.0, 2.0, -1.0]], dtype=DT)
    w1, b1, w2, b2 = (
        torch.zeros(3, 2, dtype=DT),
        torch.zeros(2, dtype=DT),
        torch.zeros(2, 3, dtype=DT),
        torch.zeros(3, dtype=DT),
    )
    ws = 2.0 * torch.eye(3, dtype=DT)
    got = checked(lambda: residual_block(x, w1, b1, w2, b2, ws))
    close(
        got, 2.0 * x, "projection ignored: a same-width block must use w_shortcut when it is given"
    )


def test_input_gradient_contains_identity_term():
    # With w1 = 0 the branch is constant, so dF/dx = 0 and the block Jacobian must be exactly I.
    d = 4
    x = torch.randn(d, generator=torch.Generator().manual_seed(5), dtype=DT).requires_grad_(True)
    w1 = torch.zeros(d, 3, dtype=DT)
    b1 = torch.ones(3, dtype=DT)
    w2, b2 = random_weights(3, 3, d, 6)[2:]
    jac = torch.autograd.functional.jacobian(lambda z: residual_block(z, w1, b1, w2, b2), x)
    close(
        jac,
        torch.eye(d, dtype=DT),
        "input gradient is missing the identity term: with a flat branch dy/dx must be I "
        "(shortcut detached or dropped)",
    )


def test_block_jacobian_equals_identity_plus_branch_jacobian():
    # dy/dx = I + dF/dx, checked with autograd on the oracle branch and the submission.
    d, h = 3, 4
    x = torch.randn(d, generator=torch.Generator().manual_seed(9), dtype=DT)
    w1, b1, w2, b2 = random_weights(d, h, d, 10)
    got = torch.autograd.functional.jacobian(lambda z: residual_block(z, w1, b1, w2, b2), x)
    branch = torch.autograd.functional.jacobian(lambda z: oracle_branch(z, w1, b1, w2, b2), x)
    close(
        got,
        torch.eye(d, dtype=DT) + branch,
        "Jacobian mismatch: dy/dx must equal I + dF/dx for the identity shortcut",
    )


def test_finite_difference_jacobian_matches_autograd():
    d, h = 2, 3
    x = torch.randn(d, generator=torch.Generator().manual_seed(12), dtype=DT)
    w1, b1, w2, b2 = random_weights(d, h, d, 13)
    got = torch.autograd.functional.jacobian(lambda z: residual_block(z, w1, b1, w2, b2), x)
    numeric = torch.zeros(d, d, dtype=DT)
    eps = 1e-6
    for j in range(d):
        step = torch.zeros(d, dtype=DT)
        step[j] = eps
        numeric[:, j] = (
            residual_block(x + step, w1, b1, w2, b2) - residual_block(x - step, w1, b1, w2, b2)
        ) / (2 * eps)
    # Central differences: truncation about eps^2 and round-off about 1e-16/eps, so 1e-7 is safe.
    if not torch.allclose(got, numeric, atol=1e-7, rtol=1e-7):
        pytest.fail(
            "autograd Jacobian disagrees with finite differences: check the backward path",
            pytrace=False,
        )


def test_zero_size_batch_returns_empty_output():
    w1, b1, w2, b2 = random_weights(3, 2, 4, 14)
    out = checked(
        lambda: residual_block(
            torch.zeros(0, 3, dtype=DT), w1, b1, w2, b2, torch.zeros(3, 4, dtype=DT)
        )
    )
    if tuple(out.shape) != (0, 4):
        pytest.fail("zero-size batch must return shape (0, d_out)", pytrace=False)


def test_float32_input_returns_float32():
    x = torch.randn(2, 3, dtype=torch.float32)
    w1, b1, w2, b2 = (t.float() for t in random_weights(3, 2, 3, 15))
    out = checked(lambda: residual_block(x, w1, b1, w2, b2))
    if out.dtype != torch.float32:
        pytest.fail("output dtype must match the input dtype", pytrace=False)


def test_mixed_dtype_and_integer_inputs_rejected():
    w1, b1, w2, b2 = random_weights(2, 2, 2, 16)
    rejects(lambda: residual_block(torch.ones(1, 2, dtype=torch.int64), w1, b1, w2, b2))
    rejects(lambda: residual_block(torch.ones(1, 2, dtype=DT), w1.float(), b1, w2, b2))


def test_identity_shortcut_width_change_rejected():
    # Widening (3 -> 4) and narrowing (3 -> 2) are both width changes: no silent crop or pad.
    w1, b1, w2, b2 = random_weights(3, 2, 4, 17)
    rejects(lambda: residual_block(torch.ones(2, 3, dtype=DT), w1, b1, w2, b2))
    w1, b1, w2, b2 = random_weights(3, 2, 2, 21)
    rejects(lambda: residual_block(torch.ones(2, 3, dtype=DT), w1, b1, w2, b2))


def test_wrong_weight_shape_rejected():
    x = torch.ones(2, 3, dtype=DT)
    w1, b1, w2, b2 = random_weights(3, 2, 3, 18)
    rejects(lambda: residual_block(x, w1.T, b1, w2, b2))  # transposed first layer
    rejects(lambda: residual_block(x, w1, b1, w2.T, b2))  # transposed second layer


def test_scale_extremes_stay_finite_and_match_oracle():
    for scale in (1e-8, 1e8):
        x = torch.full((2, 3), scale, dtype=DT) * torch.tensor([[1.0, -1.0, 0.5]], dtype=DT)
        w1, b1, w2, b2 = random_weights(3, 2, 3, 19)
        got = checked(functools.partial(residual_block, x, w1, b1, w2, b2))
        if not torch.isfinite(got).all():
            pytest.fail(
                "output must be finite for finite inputs at large or small scale", pytrace=False
            )
        close(
            got, oracle_block(x, w1, b1, w2, b2), "output differs from the oracle at scale extremes"
        )


def test_inputs_are_not_mutated():
    x = torch.randn(2, 3, dtype=DT)
    w1, b1, w2, b2 = random_weights(3, 2, 3, 20)
    originals = [t.clone() for t in (x, w1, b1, w2, b2)]
    checked(lambda: residual_block(x, w1, b1, w2, b2))
    for before, after in zip(originals, (x, w1, b1, w2, b2), strict=True):
        if not torch.equal(before, after):
            pytest.fail(
                "input tensors must not be mutated (do not add the branch into x in place)",
                pytrace=False,
            )


def _rejects_with(message, call):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{message}: raised {type(exc).__name__} instead of ValueError", pytrace=False)
    pytest.fail(f"{message}: no ValueError was raised", pytrace=False)


def test_non_tensor_input_is_rejected_with_value_error():
    w1, b1, w2, b2 = random_weights(3, 2, 3, 30)
    _rejects_with(
        "a non-tensor x must be rejected with ValueError",
        lambda: residual_block([[1.0, 2.0, 3.0]], w1, b1, w2, b2),
    )


def test_integer_tensors_are_rejected_with_value_error():
    x = torch.ones(2, 3, dtype=torch.int64)
    w1 = torch.ones(3, 2, dtype=torch.int64)
    b1 = torch.zeros(2, dtype=torch.int64)
    w2 = torch.ones(2, 3, dtype=torch.int64)
    b2 = torch.zeros(3, dtype=torch.int64)
    _rejects_with(
        "integer x must be rejected with ValueError: the block needs floating-point tensors",
        lambda: residual_block(x, w1, b1, w2, b2),
    )


def test_zero_dimensional_input_is_rejected_with_value_error():
    w1, b1, w2, b2 = random_weights(1, 2, 1, 31)
    x = torch.tensor(1.0, dtype=DT)
    _rejects_with(
        "a 0-dimensional x has no feature axis and must be rejected with ValueError",
        lambda: residual_block(x, w1, b1, w2, b2),
    )


def test_wrong_first_layer_input_width_is_rejected_with_value_error():
    x = torch.ones(2, 3, dtype=DT)
    w1 = torch.randn(4, 2, dtype=DT)
    b1 = torch.zeros(2, dtype=DT)
    w2 = torch.randn(2, 3, dtype=DT)
    b2 = torch.zeros(3, dtype=DT)
    _rejects_with(
        "w1 with the wrong input width must be rejected with ValueError",
        lambda: residual_block(x, w1, b1, w2, b2),
    )


def test_first_layer_bias_with_wrong_shape_is_rejected_with_value_error():
    x = torch.ones(2, 3, dtype=DT)
    w1 = torch.randn(3, 2, dtype=DT)
    b1 = torch.zeros(1, dtype=DT)
    w2 = torch.randn(2, 3, dtype=DT)
    b2 = torch.zeros(3, dtype=DT)
    _rejects_with(
        "b1 with the wrong shape must be rejected with ValueError",
        lambda: residual_block(x, w1, b1, w2, b2),
    )


def test_zero_hidden_width_is_rejected_with_value_error():
    x = torch.ones(3, 2, dtype=DT)
    w1 = torch.zeros(2, 0, dtype=DT)
    b1 = torch.zeros(0, dtype=DT)
    w2 = torch.zeros(0, 2, dtype=DT)
    b2 = torch.zeros(2, dtype=DT)
    _rejects_with(
        "a zero hidden width h must be rejected with ValueError",
        lambda: residual_block(x, w1, b1, w2, b2),
    )


def test_zero_input_width_is_rejected_with_value_error():
    x = torch.ones(2, 0, dtype=DT)
    w1 = torch.zeros(0, 2, dtype=DT)
    b1 = torch.zeros(2, dtype=DT)
    w2 = torch.zeros(2, 3, dtype=DT)
    b2 = torch.zeros(3, dtype=DT)
    ws = torch.zeros(0, 3, dtype=DT)
    _rejects_with(
        "a zero input width d_in must be rejected with ValueError",
        lambda: residual_block(x, w1, b1, w2, b2, ws),
    )


def test_zero_output_width_is_rejected_with_value_error():
    x = torch.ones(2, 3, dtype=DT)
    w1 = torch.randn(3, 2, dtype=DT)
    b1 = torch.zeros(2, dtype=DT)
    w2 = torch.zeros(2, 0, dtype=DT)
    b2 = torch.zeros(0, dtype=DT)
    ws = torch.zeros(3, 0, dtype=DT)
    _rejects_with(
        "a zero output width d_out must be rejected with ValueError",
        lambda: residual_block(x, w1, b1, w2, b2, ws),
    )


def test_second_layer_with_wrong_hidden_height_is_rejected_with_value_error():
    x = torch.ones(2, 3, dtype=DT)
    w1 = torch.randn(3, 2, dtype=DT)
    b1 = torch.zeros(2, dtype=DT)
    w2 = torch.randn(3, 3, dtype=DT)
    b2 = torch.zeros(3, dtype=DT)
    _rejects_with(
        "w2 with the wrong leading dimension must be rejected with ValueError",
        lambda: residual_block(x, w1, b1, w2, b2),
    )


def test_second_layer_bias_with_wrong_shape_is_rejected_with_value_error():
    x = torch.ones(2, 3, dtype=DT)
    w1 = torch.randn(3, 2, dtype=DT)
    b1 = torch.zeros(2, dtype=DT)
    w2 = torch.randn(2, 3, dtype=DT)
    b2 = torch.zeros(1, dtype=DT)
    _rejects_with(
        "b2 with the wrong shape must be rejected with ValueError",
        lambda: residual_block(x, w1, b1, w2, b2),
    )


def test_projection_shortcut_with_wrong_shape_is_rejected_with_value_error():
    x = torch.ones(2, 3, dtype=DT)
    w1 = torch.randn(3, 2, dtype=DT)
    b1 = torch.zeros(2, dtype=DT)
    w2 = torch.randn(2, 2, dtype=DT)
    b2 = torch.zeros(2, dtype=DT)
    ws = torch.randn(3, 1, dtype=DT)
    _rejects_with(
        "w_shortcut with the wrong shape must be rejected with ValueError",
        lambda: residual_block(x, w1, b1, w2, b2, ws),
    )
