# Hidden tests: oracle comparisons against PyTorch's log-sigmoid, gradient checks and edge cases.
# The oracle is used only here; the submission may not use torch.nn or its functional module.
# Messages name the kind of mistake and never show hidden inputs or expected values.
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as ref_f  # oracle only
from submission import bradley_terry_loss_and_accuracy

ATOL = 1e-9  # float64 agreement; far below any real mistake in this problem
RTOL = 1e-9
FLOAT32_ATOL = (
    1e-6  # float32 has a 24-bit mantissa (about 1e-7 relative); 1e-6 is the tightest safe bound
)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _randn(shape, seed, scale=1.0, offset=0.0, dtype=torch.float64):
    return (torch.randn(shape, generator=_gen(seed), dtype=dtype) * scale + offset).clone()


def _call(chosen, rejected):
    """Call the learner, check it did not mutate inputs, and check the return shape and dtype."""
    snap_c, snap_r = chosen.clone(), rejected.clone()
    try:
        out = bradley_terry_loss_and_accuracy(chosen, rejected)
    except ValueError as exc:
        pytest.fail(
            f"valid input raised ValueError ({exc}): check the contract's shape and dtype rules",
            pytrace=False,
        )
    except Exception as exc:  # a valid call must not fail with a generic error
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check shapes, dtypes and the contract",
            pytrace=False,
        )
    if not isinstance(out, tuple) or len(out) != 2 or not all(torch.is_tensor(t) for t in out):
        pytest.fail("return a tuple of two tensors: (loss, accuracy)", pytrace=False)
    loss, acc = out
    if loss.ndim != 0 or acc.ndim != 0:
        pytest.fail("loss and accuracy must be 0-dimensional: reduce over the pairs", pytrace=False)
    if loss.dtype != chosen.dtype or acc.dtype != chosen.dtype:
        pytest.fail(
            "loss and accuracy must keep the input dtype; do not downcast or return bool",
            pytrace=False,
        )
    if not torch.equal(snap_c, chosen) or not torch.equal(snap_r, rejected):
        pytest.fail(
            "an input reward tensor was modified in place; compute new tensors", pytrace=False
        )
    return loss, acc


def _reference(chosen, rejected):
    """Independent oracle: PyTorch's stable log-sigmoid and a strict comparison."""
    loss = -ref_f.logsigmoid(chosen - rejected).mean()
    acc = (chosen > rejected).to(chosen.dtype).mean()
    return loss, acc


def _close(actual, expected, message, atol=ATOL, rtol=RTOL):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        pytest.fail(message, pytrace=False)


def _expect_value_error(thunk, message):
    try:
        thunk()
    except ValueError:
        return
    except Exception:
        pytest.fail(
            "invalid input must raise ValueError, not another exception type", pytrace=False
        )
    pytest.fail(message, pytrace=False)


def test_loss_and_accuracy_match_reference_on_random_pairs():
    for seed, batch in enumerate([1, 3, 8, 64, 257]):
        chosen = _randn((batch,), seed, scale=2.0)
        rejected = _randn((batch,), seed + 100, scale=2.0)
        loss, acc = _call(chosen, rejected)
        want_loss, want_acc = _reference(chosen, rejected)
        _close(
            loss,
            want_loss,
            "loss disagrees with the reference: check the margin sign (chosen minus rejected), "
            "the -log(sigmoid) term, and that the loss is averaged over pairs, not summed",
        )
        _close(
            acc,
            want_acc,
            "accuracy disagrees with the reference: count strictly chosen > rejected pairs "
            "and divide by the number of pairs",
        )


def test_adding_one_constant_to_both_rewards_changes_nothing():
    for seed in range(4):
        chosen = _randn((16,), seed, scale=1.5)
        rejected = _randn((16,), seed + 50, scale=1.5)
        base_loss, base_acc = _call(chosen, rejected)
        shifted_loss, shifted_acc = _call(chosen + 37.5, rejected + 37.5)
        _close(
            shifted_loss,
            base_loss,
            "loss depends on the absolute reward level: only the difference r_c - r_r may enter",
        )
        _close(
            shifted_acc,
            base_acc,
            "accuracy depends on the absolute reward level: compare the two rewards of each pair",
        )


def test_large_positive_margin_gives_a_finite_loss_near_zero():
    loss, acc = _call(
        torch.tensor([1000.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64)
    )
    assert torch.isfinite(loss), (
        "loss is not finite for a large positive margin: use the softplus form"
    )
    _close(
        loss,
        torch.tensor(0.0, dtype=torch.float64),
        "a confidently correct pair must give a loss near zero",
        atol=1e-12,
    )
    _close(
        acc,
        torch.tensor(1.0, dtype=torch.float64),
        "a confidently correct pair must count as correct",
    )


def test_large_negative_margin_loss_equals_minus_margin():
    chosen = torch.tensor([0.0], dtype=torch.float64)
    rejected = torch.tensor([1000.0], dtype=torch.float64)
    loss, acc = _call(chosen, rejected)
    assert torch.isfinite(loss), (
        "loss overflowed or underflowed for a large negative margin: compute -log(sigmoid(d)) as "
        "softplus(-d) = max(-d, 0) + log1p(exp(-|d|)), not as log(sigmoid(d))"
    )
    _close(
        loss,
        torch.tensor(1000.0, dtype=torch.float64),
        "a confidently wrong pair must give a loss equal to the size of its margin",
        atol=1e-9,
        rtol=1e-12,
    )
    _close(
        acc,
        torch.tensor(0.0, dtype=torch.float64),
        "a confidently wrong pair must count as incorrect",
    )


def test_extreme_margins_match_the_stable_reference_in_float64():
    margins = torch.tensor(
        [-1e4, -750.0, -40.0, -1.0, 0.0, 1.0, 40.0, 750.0, 1e4], dtype=torch.float64
    )
    chosen = margins
    rejected = torch.zeros_like(margins)
    loss, _ = _call(chosen, rejected)
    want_loss, _ = _reference(chosen, rejected)
    assert torch.isfinite(loss), "loss is not finite across extreme margins: check the stable form"
    _close(
        loss,
        want_loss,
        "loss over extreme margins disagrees with the reference: the stable form must hold for "
        "both very large positive and very large negative margins",
        rtol=1e-9,
    )


def test_float32_inputs_keep_dtype_and_match_reference():
    chosen = _randn((32,), 7, scale=2.0, dtype=torch.float32)
    rejected = _randn((32,), 8, scale=2.0, dtype=torch.float32)
    loss, acc = _call(chosen, rejected)
    assert loss.dtype == torch.float32, "loss must stay float32; do not upcast or downcast"
    want_loss, want_acc = _reference(chosen, rejected)
    _close(
        loss,
        want_loss,
        "float32 loss disagrees with the reference beyond float32 rounding",
        atol=FLOAT32_ATOL,
        rtol=FLOAT32_ATOL,
    )
    _close(
        acc,
        want_acc,
        "float32 accuracy disagrees with the reference: count strictly chosen > rejected",
    )


def test_gradient_matches_the_analytic_form_in_float64():
    chosen = _randn((12,), 21, scale=1.0).requires_grad_(True)
    rejected = _randn((12,), 22, scale=1.0).requires_grad_(True)
    loss, _ = bradley_terry_loss_and_accuracy(chosen, rejected)
    assert loss.requires_grad, "loss is detached from autograd: do not call .detach() or .item()"
    loss.backward()
    margin = (chosen - rejected).detach()
    # d(-log sigmoid(d))/dd = -(1 - sigmoid(d)); the mean divides every pair's term by B.
    coeff = -(1.0 - torch.sigmoid(margin)) / margin.numel()
    _close(
        chosen.grad,
        coeff,
        "gradient wrt the chosen reward disagrees with -(1 - sigmoid(d)) / B: "
        "check the sign of the margin and the 1/B factor from the mean",
    )
    _close(
        rejected.grad,
        -coeff,
        "gradient wrt the rejected reward disagrees with +(1 - sigmoid(d)) / B: "
        "check the sign of the margin and the 1/B factor from the mean",
    )


def test_gradient_matches_central_finite_differences_in_float64():
    chosen = _randn((5,), 31, scale=1.0)
    rejected = _randn((5,), 32, scale=1.0)

    def fn(c):
        value, _ = bradley_terry_loss_and_accuracy(c, rejected)
        return value

    x = chosen.clone().requires_grad_(True)
    value, _ = bradley_terry_loss_and_accuracy(x, rejected)
    if not value.requires_grad:
        pytest.fail(
            "loss is detached from autograd: do not call .detach() or .item() on the loss",
            pytrace=False,
        )
    value.backward()
    h = 1e-6
    numeric = torch.zeros_like(chosen)
    for i in range(chosen.numel()):
        up, down = chosen.clone(), chosen.clone()
        up[i] += h
        down[i] -= h
        numeric[i] = (fn(up) - fn(down)) / (2 * h)
    # Central differences in float64 with h = 1e-6 agree to about 1e-9; 1e-7 is still tight.
    _close(
        x.grad,
        numeric,
        "autograd gradient disagrees with finite differences: the loss may be detached",
        atol=1e-7,
        rtol=1e-7,
    )


def test_tied_pairs_are_counted_as_incorrect():
    chosen = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float64)
    rejected = torch.tensor([0.0, 2.0, 3.0], dtype=torch.float64)
    _, acc = _call(chosen, rejected)
    _close(
        acc,
        torch.tensor(1.0 / 3.0, dtype=torch.float64),
        "a tied pair was counted as correct: use a strict comparison r_c > r_r",
        atol=1e-12,
        rtol=1e-12,
    )


def test_shape_mismatch_raises_value_error():
    _expect_value_error(
        lambda: bradley_terry_loss_and_accuracy(_randn((3,), 1), _randn((4,), 2)),
        "mismatched chosen and rejected shapes were accepted: they must match pair by pair",
    )


def test_empty_batch_raises_value_error():
    _expect_value_error(
        lambda: bradley_terry_loss_and_accuracy(
            torch.zeros((0,), dtype=torch.float64), torch.zeros((0,), dtype=torch.float64)
        ),
        "an empty batch was accepted: a batch with no pairs must raise ValueError",
    )


def test_two_dimensional_rewards_raise_value_error():
    _expect_value_error(
        lambda: bradley_terry_loss_and_accuracy(_randn((2, 3), 3), _randn((2, 3), 4)),
        "2-D rewards were accepted: pass one reward per pair as a 1-D tensor",
    )


def test_nan_or_inf_rewards_raise_value_error():
    bad = torch.tensor([1.0, float("nan")], dtype=torch.float64)
    good = torch.tensor([0.0, 0.0], dtype=torch.float64)
    _expect_value_error(
        lambda: bradley_terry_loss_and_accuracy(bad, good),
        "NaN rewards were accepted: non-finite values must raise ValueError",
    )
    inf = torch.tensor([float("inf"), 1.0], dtype=torch.float64)
    _expect_value_error(
        lambda: bradley_terry_loss_and_accuracy(inf, good),
        "infinite rewards were accepted: non-finite values must raise ValueError",
    )


def test_integer_rewards_raise_value_error():
    ints = torch.tensor([1, 2], dtype=torch.int64)
    _expect_value_error(
        lambda: bradley_terry_loss_and_accuracy(ints, ints),
        "integer rewards were accepted: the loss needs floating-point rewards",
    )


def test_mixed_dtypes_raise_value_error():
    _expect_value_error(
        lambda: bradley_terry_loss_and_accuracy(
            torch.tensor([1.0, 2.0], dtype=torch.float32),
            torch.tensor([0.0, 1.0], dtype=torch.float64),
        ),
        "chosen and rejected rewards with different dtypes were accepted: require one shared dtype",
    )


def test_inputs_are_not_mutated_by_the_call():
    chosen = _randn((6,), 41)
    rejected = _randn((6,), 42)
    before_c, before_r = chosen.clone(), rejected.clone()
    _call(chosen, rejected)
    assert torch.equal(chosen, before_c), "the chosen reward tensor was modified in place"
    assert torch.equal(rejected, before_r), "the rejected reward tensor was modified in place"


def test_chosen_list_instead_of_tensor_raises_value_error():
    _expect_value_error(
        lambda: bradley_terry_loss_and_accuracy([1.0], torch.tensor([0.0], dtype=torch.float64)),
        "a chosen reward that is not a torch.Tensor must raise ValueError, not another error",
    )


def test_rejected_list_instead_of_tensor_raises_value_error():
    _expect_value_error(
        lambda: bradley_terry_loss_and_accuracy(torch.tensor([1.0], dtype=torch.float64), [0.0]),
        "a rejected reward that is not a torch.Tensor must raise ValueError, not another error",
    )


def test_nan_in_rejected_rewards_raises_value_error_even_with_finite_entries():
    chosen = torch.tensor([0.0, 0.0], dtype=torch.float64)
    rejected = torch.tensor([1.0, float("nan")], dtype=torch.float64)
    _expect_value_error(
        lambda: bradley_terry_loss_and_accuracy(chosen, rejected),
        "a NaN among the rejected rewards must raise ValueError, even beside finite entries",
    )


def test_inf_in_rejected_rewards_raises_value_error_even_with_finite_entries():
    chosen = torch.tensor([0.0, 0.0], dtype=torch.float64)
    rejected = torch.tensor([1.0, float("inf")], dtype=torch.float64)
    _expect_value_error(
        lambda: bradley_terry_loss_and_accuracy(chosen, rejected),
        "an infinite rejected reward must raise ValueError, even when other entries are finite",
    )
