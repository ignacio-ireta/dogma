from __future__ import annotations

import pytest
import torch
from submission import grad_at

ATOL = 1e-6  # matches meta.yaml tolerance
RTOL = 1e-6  # matches meta.yaml tolerance


def checked(call) -> torch.Tensor:
    try:
        out = call()
    except RuntimeError as exc:
        if "does not have been used" in str(exc) or "appears to not have been used" in str(exc):
            raise AssertionError(
                "an input f ignores needs zero gradient: pass allow_unused=True"
            ) from None
        raise AssertionError(
            "valid input raised RuntimeError: honor the documented contract"
        ) from None
    except Exception as exc:
        raise AssertionError(
            f"valid input raised {type(exc).__name__}: honor the documented contract"
        ) from None
    if not isinstance(out, torch.Tensor):
        raise AssertionError("return a torch.Tensor of the gradient")
    return out


def rejects(call) -> None:
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(
            f"invalid input must raise ValueError, not {type(exc).__name__}"
        ) from None
    raise AssertionError("invalid input was accepted: check rank, dtype and output rank")


def central_difference(f, x: torch.Tensor, h: float = 1e-6) -> torch.Tensor:
    """Float64 central differences: an independent check that does not use autograd."""
    grad = torch.zeros_like(x)
    with torch.no_grad():
        for i in range(x.numel()):
            up = x.clone()
            down = x.clone()
            up[i] += h
            down[i] -= h
            grad[i] = (f(up) - f(down)) / (2 * h)
    return grad


SMOOTH_CASES = [
    ("cube-sum", lambda v: (v**3).sum(), lambda v: 3 * v**2),
    (
        "exp-weighted-square",
        lambda v: (torch.exp(v) * v**2).sum(),
        lambda v: torch.exp(v) * (v**2 + 2 * v),
    ),
    (
        "log-sum-exp",
        lambda v: torch.logsumexp(v, dim=0),
        lambda v: torch.softmax(v, dim=0),
    ),
    (
        "product-and-sine",
        lambda v: v[0] * v[1] + torch.sin(v[1]),
        lambda v: torch.stack([v[1], v[0] + torch.cos(v[1])]),
    ),
]


@pytest.mark.parametrize("name,f,analytic", SMOOTH_CASES, ids=[c[0] for c in SMOOTH_CASES])
def test_matches_analytic_gradient_on_seeded_random_points(name, f, analytic):
    gen = torch.Generator().manual_seed(0)
    for _ in range(3):
        x = torch.randn(2, generator=gen, dtype=torch.float64)
        expected = analytic(x)
        got = checked(lambda x=x: grad_at(f, x))
        assert got.dtype == torch.float64, "gradient dtype must match x"
        assert torch.allclose(got, expected, atol=ATOL, rtol=RTOL), (
            f"gradient of {name} disagrees with the analytic derivative: check each term's rule"
        )


def test_matches_central_finite_differences_in_float64():
    gen = torch.Generator().manual_seed(1)
    f = lambda v: (torch.exp(v) * v**2).sum() + torch.sin(v).prod()  # noqa: E731
    x = torch.randn(4, generator=gen, dtype=torch.float64)
    got = checked(lambda: grad_at(f, x))
    numeric = central_difference(f, x)
    assert torch.allclose(got, numeric, atol=ATOL, rtol=RTOL), (
        "autograd gradient disagrees with central finite differences"
    )


def test_preexisting_grad_field_is_left_unchanged_not_accumulated():
    x = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    x.grad = torch.full_like(x, 7.0)
    g = checked(lambda: grad_at(lambda v: (v**2).sum(), x))
    assert torch.allclose(g, torch.tensor([2.0, 4.0], dtype=torch.float64), atol=ATOL), (
        "the returned gradient is wrong when x already has a .grad"
    )
    assert torch.equal(x.grad, torch.full_like(x, 7.0)), (
        "x.grad changed: the call must not accumulate into the caller's .grad"
    )


def test_caller_values_and_requires_grad_are_unchanged_when_input_is_plain():
    x = torch.tensor([0.25, -0.75], dtype=torch.float64)
    before = x.clone()
    checked(lambda: grad_at(lambda v: (v**2).sum(), x))
    assert torch.equal(x, before), "the caller's values were modified"
    assert not x.requires_grad, "a plain input was turned into a tracked tensor"
    assert x.grad is None, "a plain input received a .grad field"


def test_works_inside_no_grad_context():
    x = torch.tensor([1.0, -2.0], dtype=torch.float64)
    with torch.no_grad():
        g = checked(lambda: grad_at(lambda v: (v**3).sum(), x))
    assert torch.allclose(g, torch.tensor([3.0, 12.0], dtype=torch.float64), atol=ATOL), (
        "gradient is wrong or missing inside torch.no_grad(): re-enable grad for the computation"
    )


def test_result_is_detached_with_no_graph_even_when_input_tracks_gradients():
    x = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    g = checked(lambda: grad_at(lambda v: (v**2).sum(), x))
    assert not g.requires_grad and g.grad_fn is None, (
        "the gradient must be detached: do not keep a graph for the derivative"
    )


def test_constant_and_unused_input_functions_give_zeros():
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    constant = checked(lambda: grad_at(lambda v: torch.tensor(2.0, dtype=torch.float64), x))
    assert torch.equal(constant, torch.zeros_like(x)), "a constant must have zero gradient"
    tracked = checked(lambda: grad_at(lambda v: (x * 0).sum() + 5.0, x))
    assert torch.equal(tracked, torch.zeros_like(x)), (
        "a function that does not use the input must have zero gradient, not an error"
    )


def test_float32_input_returns_float32_gradient():
    x = torch.tensor([0.5, -0.25, 0.75])
    g = checked(lambda: grad_at(lambda v: (v**2).sum(), x))
    assert g.dtype == torch.float32, "gradient dtype must match the input dtype"
    assert torch.allclose(g, 2 * x, atol=ATOL), "float32 gradient of sum(x**2) is 2x"


def test_large_and_small_magnitudes_keep_the_scale():
    big = torch.tensor([1e150, -1e150], dtype=torch.float64)
    g_big = checked(lambda: grad_at(lambda v: (v**2).sum(), big))
    assert torch.allclose(g_big, 2 * big, atol=0.0, rtol=RTOL), (
        "very large values must keep their relative gradient"
    )
    small = torch.tensor([1e-150, -3e-150], dtype=torch.float64)
    g_small = checked(lambda: grad_at(lambda v: (v**2).sum(), small))
    assert torch.allclose(g_small, 2 * small, atol=0.0, rtol=RTOL), (
        "very small values must keep their relative gradient"
    )


def test_single_element_input_is_supported():
    x = torch.tensor([1.5], dtype=torch.float64)
    g = checked(lambda: grad_at(lambda v: torch.exp(v).sum(), x))
    assert g.shape == (1,) and torch.allclose(g, torch.exp(x), atol=ATOL), (
        "a single-element input must give a gradient of shape (1,)"
    )


def test_graph_that_never_reads_x_gives_zero_gradient_not_an_error():
    other = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([1.0, -1.0], dtype=torch.float64)
    g = checked(lambda: grad_at(lambda v: (other * other).sum(), x))
    assert torch.equal(g, torch.zeros_like(x)), (
        "a graph that never reads x must give zero gradient, not None or an error"
    )


def test_two_dimensional_input_is_rejected_as_value_error():
    rejects(lambda: grad_at(lambda v: (v**2).sum(), torch.ones(2, 2)))


def test_integer_input_is_rejected_as_value_error():
    rejects(lambda: grad_at(lambda v: (v**2).sum(), torch.tensor([1, 2])))


def test_non_scalar_output_is_rejected_as_value_error():
    rejects(lambda: grad_at(lambda v: v**2, torch.tensor([1.0, 2.0])))


def test_function_writing_into_its_argument_leaves_the_caller_tensor_unchanged():
    x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    before = x.clone()

    def f(v):
        with torch.no_grad():
            v.add_(1.0)
        return (v**2).sum()

    checked(lambda: grad_at(f, x))
    if not torch.equal(x, before):
        pytest.fail(
            "the caller's tensor changed: grad_at must work on a private copy, not on x itself",
            pytrace=False,
        )
