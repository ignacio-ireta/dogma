from __future__ import annotations

import copy

import pytest
import torch
from submission import train_step

ATOL = 1e-9  # matches meta.yaml tolerance
RTOL = 1e-9  # matches meta.yaml tolerance


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((prediction - target) ** 2).mean()


def make_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)
    ).double()


def checked_step(model, x, y, lr) -> float:
    try:
        value = train_step(model, x, y, mse, lr)
    except ValueError:
        raise AssertionError("valid training step raised ValueError: check the lr rule") from None
    except Exception as exc:
        if "leaf" in str(exc) or "in-place" in str(exc):
            raise AssertionError(
                "the update must run under torch.no_grad(): an in-place write to a leaf that "
                "requires grad fails, and a tracked update would keep the graph"
            ) from None
        raise AssertionError(
            f"valid training step raised {type(exc).__name__}: honor the documented contract"
        ) from None
    if not isinstance(value, float):
        raise AssertionError("return the loss as a Python float")
    return value


def reference_run(model, x, y, lr, steps):
    """torch.optim.SGD on a deep copy: the library oracle the learner must match."""
    oracle = copy.deepcopy(model)
    optimizer = torch.optim.SGD(oracle.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = mse(oracle(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return oracle, losses


def batch(seed: int):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(5, 3, generator=gen, dtype=torch.float64)
    y = torch.randn(5, 1, generator=gen, dtype=torch.float64)
    return x, y


def test_five_steps_match_torch_optim_sgd_in_parameters_and_losses():
    x, y = batch(0)
    model = make_model(1)
    oracle, expected_losses = reference_run(model, x, y, lr=0.05, steps=5)
    losses = [checked_step(model, x, y, 0.05) for _ in range(5)]
    for got, want in zip(model.parameters(), oracle.parameters(), strict=True):
        assert torch.allclose(got, want, atol=ATOL, rtol=RTOL), (
            "parameters after several steps differ from the SGD reference: check the update sign "
            "and whether each step uses a fresh gradient"
        )
    assert torch.allclose(
        torch.tensor(losses, dtype=torch.float64),
        torch.tensor(expected_losses, dtype=torch.float64),
        atol=ATOL,
        rtol=RTOL,
    ), "returned losses differ from the pre-update losses"


def test_repeated_calls_on_the_same_batch_do_not_accumulate_old_gradients():
    # A missing zero_grad makes step two use gradient1 + gradient2, which moves the parameters
    # about twice as far. Calling twice on the same batch is the cleanest way to see it.
    x, y = batch(2)
    model = make_model(3)
    oracle, expected = reference_run(model, x, y, lr=0.1, steps=2)
    first = checked_step(model, x, y, 0.1)
    second = checked_step(model, x, y, 0.1)
    assert abs(first - expected[0]) <= ATOL + RTOL * abs(expected[0]), (
        "first pre-update loss is wrong"
    )
    assert abs(second - expected[1]) <= ATOL + RTOL * abs(expected[1]), (
        "second call's loss is wrong: the stale gradient from the first step was reused"
    )
    for got, want in zip(model.parameters(), oracle.parameters(), strict=True):
        assert torch.allclose(got, want, atol=ATOL, rtol=RTOL), (
            "second step moved parameters by the accumulated gradient: clear .grad before backward"
        )


def test_gradient_left_on_parameters_is_the_one_used_for_the_update():
    x, y = batch(4)
    model = make_model(5)
    before = [p.detach().clone() for p in model.parameters()]
    checked_step(model, x, y, 0.2)
    for p, old in zip(model.parameters(), before, strict=True):
        assert p.grad is not None, "every parameter with a gradient must keep its .grad"
        assert torch.allclose(old - 0.2 * p.grad, p.detach(), atol=ATOL, rtol=RTOL), (
            "the update must be exactly minus lr times the .grad left on the parameter"
        )


def test_parameters_keep_identity_leaf_status_and_requires_grad():
    x, y = batch(6)
    model = make_model(7)
    handles = list(model.parameters())
    checked_step(model, x, y, 0.1)
    assert all(a is b for a, b in zip(handles, model.parameters(), strict=True)), (
        "parameters were replaced: update the existing tensors in place"
    )
    for p in model.parameters():
        assert p.requires_grad, "the update must not turn off requires_grad"
        assert p.is_leaf and p.grad_fn is None, (
            "the update left a graph on a parameter: keep the update under torch.no_grad()"
        )


def test_returned_loss_is_the_pre_update_loss_not_the_post_update_loss():
    x, y = batch(8)
    model = make_model(9)
    with torch.no_grad():
        pre = mse(model(x), y).item()
    got = checked_step(model, x, y, 0.5)
    assert abs(got - pre) <= ATOL + RTOL * abs(pre), (
        "the returned value must be the loss before the update"
    )


def test_sign_and_scale_of_the_step_follow_the_learning_rate():
    x, y = batch(10)
    small = make_model(11)
    large = copy.deepcopy(small)
    checked_step(small, x, y, 0.01)
    checked_step(large, x, y, 0.02)
    for p0, p_small, p_large in zip(
        make_model(11).parameters(), small.parameters(), large.parameters(), strict=True
    ):
        moved_small = p_small.detach() - p0.detach()
        moved_large = p_large.detach() - p0.detach()
        assert torch.allclose(moved_large, 2 * moved_small, atol=ATOL, rtol=RTOL), (
            "doubling lr must double the step: check the scale and the sign of the update"
        )


def test_zero_gradient_batch_leaves_parameters_unchanged():
    model = torch.nn.Linear(2, 1).double()
    x = torch.zeros(3, 2, dtype=torch.float64)
    with torch.no_grad():
        model.bias.zero_()
    y = model(x).detach()  # prediction equals the target, so the loss and its gradient are zero
    before = [p.detach().clone() for p in model.parameters()]
    loss = checked_step(model, x, y, 0.3)
    assert loss == 0.0, "a perfect prediction must give zero loss"
    for p, old in zip(model.parameters(), before, strict=True):
        assert torch.equal(p.detach(), old), "zero gradient must leave parameters unchanged"


def test_parameters_without_gradient_are_not_updated():
    model = torch.nn.Linear(2, 1).double()
    unused = torch.nn.Parameter(torch.tensor([7.0], dtype=torch.float64))
    model.register_parameter("unused", unused)
    x, y = batch(12)
    x = x[:, :2]
    y = y[:, :1]
    checked_step(model, x, y, 0.1)
    assert torch.equal(unused.detach(), torch.tensor([7.0], dtype=torch.float64)), (
        "a parameter that receives no gradient must not change"
    )


def test_scalar_lr_bounds_and_bad_values_raise_value_error():
    x, y = batch(13)
    for bad in (0.0, -0.1, float("nan"), float("inf"), True):
        model = make_model(14)
        try:
            train_step(model, x, y, mse, bad)
        except ValueError:
            continue
        except Exception as exc:
            raise AssertionError(
                f"invalid lr must raise ValueError, not {type(exc).__name__}"
            ) from None
        raise AssertionError("invalid learning rate was accepted: check lr > 0 and finite")


def test_non_scalar_loss_is_rejected_as_value_error():
    x, y = batch(15)
    model = make_model(16)
    try:
        train_step(model, x, y, lambda p, t: (p - t) ** 2, 0.1)
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(
            f"non-scalar loss must raise ValueError, not {type(exc).__name__}"
        ) from None
    raise AssertionError("a non-scalar loss was accepted: the step needs one scalar to descend")


def test_inputs_and_targets_are_not_mutated():
    x, y = batch(17)
    x_before, y_before = x.clone(), y.clone()
    checked_step(make_model(18), x, y, 0.1)
    assert torch.equal(x, x_before), "the input batch was modified"
    assert torch.equal(y, y_before), "the targets were modified"


@pytest.mark.parametrize("steps", [1, 3], ids=["single-step", "several-steps"])
def test_short_runs_match_reference_for_different_step_counts(steps):
    x, y = batch(19)
    model = make_model(20)
    oracle, _ = reference_run(model, x, y, lr=0.03, steps=steps)
    for _ in range(steps):
        checked_step(model, x, y, 0.03)
    for got, want in zip(model.parameters(), oracle.parameters(), strict=True):
        assert torch.allclose(got, want, atol=ATOL, rtol=RTOL), (
            "parameters differ from the reference after a short run"
        )
