# Hidden tests (CONTENT_SPEC §6.4). Failure messages name the kind of mistake and never show
# hidden inputs, shapes or expected values. The oracle is torch.nn.functional.conv2d; the
# learner may not import it.
from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias for the oracle
from submission import conv2d_forward

from dogma_data import rng, sample_rows
from dogma_data import rng as data_rng

ATOL = 1e-10  # float64 sums of at most a few hundred terms of order one
RTOL = 1e-10


def _real_values(gen, shape, dtype=torch.float64):
    """Real numbers in the shape asked for: standardised breast_cancer measurements.

    The tensors of this problem are weights, gradients, scores or hidden states. No dataset
    holds those, so real measurements stand in for them: skewed, correlated along a row, with
    a few entries far from the rest. The torch generator picks the draw.
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
        shape = shape[0]
    shape = tuple(int(side) for side in shape)
    total = math.prod(shape)
    if total == 0:
        return torch.zeros(shape, dtype=dtype)
    draw = int(torch.randint(0, 2**31 - 1, (1,), generator=gen))
    key = f"p01-conv2d-forward:values:{draw}"
    rows = max(1, min(569, -(-total // 30)))
    drawn, _ = sample_rows("breast_cancer", data_rng(key), rows, standardize=True)
    flat = torch.tensor(drawn.ravel(), dtype=torch.float64)
    flat = flat.repeat(-(-total // flat.numel()))[:total]
    return flat.reshape(shape).to(dtype)


def _digit_images(key, n, c, h, w):
    """Real images: n examples with c channels of h by w pixels, cut from the digits set.

    Every channel is a different handwritten digit. A digit is 8 by 8 pixels; larger sizes
    tile several digits, and the result is cropped around its centre, where the ink is. Pixels
    run from 0 to 16 and are normalised with the mean (4.9) and the spread (6.0) of the whole
    set, as image pipelines do, so the background is a negative constant.
    """
    tiles_h, tiles_w = -(-h // 8), -(-w // 8)
    pixels, _ = sample_rows("digits", rng(key), n * c * tiles_h * tiles_w)
    canvas = pixels.reshape(n, c, tiles_h, tiles_w, 8, 8).transpose(0, 1, 2, 4, 3, 5)
    canvas = canvas.reshape(n, c, tiles_h * 8, tiles_w * 8)
    top, left = (canvas.shape[2] - h) // 2, (canvas.shape[3] - w) // 2
    crop = canvas[:, :, top : top + h, left : left + w]
    return torch.tensor((crop - 4.9) / 6.0, dtype=torch.float64)


def _same(a, b, atol=ATOL, rtol=RTOL):
    return a.shape == b.shape and torch.allclose(a, b, atol=atol, rtol=rtol)


def _call(x, w, b, s, p):
    """Run the learner on clones, so an in-place write cannot corrupt the oracle's inputs."""
    call_b = None if b is None else b.clone()
    try:
        out = conv2d_forward(x.clone(), w.clone(), call_b, stride=s, padding=p)
    except RuntimeError:
        pytest.fail(
            "the computation failed on a valid input: the usual cause is an output size that is "
            "not floor((H + 2p - k) / s) + 1, so a window slice and the output disagree; "
            "also check that weight's second axis is C_in"
        )
    except Exception as exc:
        pytest.fail(
            f"valid input was rejected or crashed ({type(exc).__name__}): check the contract"
        )
    if not torch.is_tensor(out):
        pytest.fail("return a tensor, not a Python object")
    if out.dtype != x.dtype:
        pytest.fail("output dtype must match x")
    return out


def _assert_conv(out, x, w, b, s, p, atol=ATOL, rtol=RTOL):  # noqa: PLR0917
    expected = F.conv2d(x, w, b, stride=s, padding=p)
    if out.shape != expected.shape:
        h_ceil = math.ceil((x.shape[2] + 2 * p - w.shape[2]) / s) + 1
        w_ceil = math.ceil((x.shape[3] + 2 * p - w.shape[3]) / s) + 1
        if tuple(out.shape[2:]) == (h_ceil, w_ceil):
            pytest.fail("output size rounded up: use floor((H + 2p - k) / s) + 1, not ceil")
        pytest.fail("output shape differs: use floor((H + 2p - k) / s) + 1 per spatial axis")
    if torch.allclose(out, expected, atol=atol, rtol=rtol):
        return
    variants = [
        (
            F.conv2d(x, w.flip(-2, -1), b, stride=s, padding=p),
            "kernel was flipped: deep learning convolution is cross-correlation, do not flip",
        ),
        (
            F.conv2d(x, w, None, stride=s, padding=p),
            "the bias is missing: add bias[o] to every position of output channel o",
        ),
        (
            F.conv2d(x, w, None, stride=s, padding=p)
            + w.shape[2] * w.shape[3] * b.reshape(1, -1, 1, 1),
            "the bias is added once per kernel offset: "
            "add it once per output channel, after the loops",
        ),
        (
            F.conv2d(x[:, :1], w[:, :1], b, stride=s, padding=p),
            "only the first input channel was used: sum over all C_in channels",
        ),
        (
            F.conv2d(F.pad(x, (0, 2 * p, 0, 2 * p)), w, b, stride=s, padding=0),
            "padding applied to one side only: pad each side of H and W by p",
        ),
    ]
    for alt, message in variants:
        if _same(out, alt, atol, rtol):
            pytest.fail(message)
    pytest.fail(
        "convolution values differ from the reference: check the window positions and stride"
    )


@pytest.mark.parametrize(
    "n, c, h, w_, o, kh, kw, s, p",
    [
        (1, 1, 4, 4, 1, 2, 2, 1, 0),
        (2, 3, 7, 6, 4, 3, 2, 2, 1),
        (1, 2, 5, 5, 3, 3, 3, 1, 1),
        (2, 1, 6, 6, 2, 3, 3, 2, 0),
        (3, 4, 9, 8, 5, 2, 3, 3, 2),
        (2, 2, 3, 7, 2, 3, 1, 1, 2),
    ],
    ids=["square", "stride-two-pad-one", "same-padding", "stride-two-no-pad", "stride-three-rect",
         "tall-kernel-big-pad"],
)  # fmt: skip
def test_matches_torch_conv2d_on_seeded_configurations(  # noqa: PLR0917
    n, c, h, w_, o, kh, kw, s, p
):
    gen = torch.Generator().manual_seed(100 * n + h)
    x = _digit_images(f"conv2d-p01:configurations:{n}:{h}", n, c, h, w_)
    w = _real_values(gen, [o, c, kh, kw], torch.float64)
    b = _real_values(gen, [o], torch.float64)
    _assert_conv(_call(x, w, b, s, p), x, w, b, s, p)


@pytest.mark.parametrize(
    "h, k, s, p",
    [(6, 3, 2, 0), (8, 3, 2, 0), (5, 2, 2, 0), (7, 3, 3, 1)],
    ids=["six-wide", "eight-wide", "five-wide", "padded-stride-three"],
)
def test_output_size_rounds_down_when_the_last_window_is_partial(h, k, s, p):
    gen = torch.Generator().manual_seed(h)
    x = _digit_images(f"conv2d-p01:floor:{h}", 1, 1, h, h)
    w = _real_values(gen, [1, 1, k, k], torch.float64)
    out = _call(x, w, None, s, p)
    expected_side = math.floor((h + 2 * p - k) / s) + 1
    if out.shape[2:] != (expected_side, expected_side):
        pytest.fail("output size must be floor((H + 2p - k) / s) + 1, not ceil or round")
    _assert_conv(out, x, w, None, s, p)


def test_kernel_is_not_flipped_on_an_asymmetric_kernel():
    gen = torch.Generator().manual_seed(21)
    x = _digit_images("conv2d-p01:asymmetric-kernel", 2, 1, 4, 5)
    w = _real_values(gen, [1, 1, 3, 2], torch.float64)
    _assert_conv(_call(x, w, None, 1, 0), x, w, None, 1, 0)


def test_each_output_channel_sums_all_input_channels():
    x = torch.stack([torch.full((3, 3), float(c + 1)) for c in range(3)])[None].double()
    w = torch.zeros(2, 3, 1, 1, dtype=torch.float64)
    w[0] = 1.0  # output 0 sums the three channels with weight 1
    w[1] = 2.0  # output 1 sums them with weight 2
    out = _call(x, w, None, 1, 0)
    if not torch.allclose(out[0, 0], torch.full((3, 3), 6.0, dtype=torch.float64)):
        pytest.fail("each output channel must sum over all input channels, not just one")
    if not torch.allclose(out[0, 1], torch.full((3, 3), 12.0, dtype=torch.float64)):
        pytest.fail("each output channel must use its own filter over the input channels")


def test_bias_is_added_once_per_output_channel():
    x = torch.zeros(1, 2, 3, 4, dtype=torch.float64)
    w = torch.zeros(3, 2, 2, 2, dtype=torch.float64)
    b = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    out = _call(x, w, b, 1, 0)
    for channel, value in enumerate(b.tolist()):
        if not torch.allclose(out[0, channel], torch.full((2, 3), value, dtype=torch.float64)):
            pytest.fail("the bias must be added once per output channel, at every position")


def test_batch_examples_are_computed_independently():
    gen = torch.Generator().manual_seed(7)
    x = _digit_images("conv2d-p01:batch-independence", 3, 2, 5, 5)
    w = _real_values(gen, [2, 2, 3, 3], torch.float64)
    together = _call(x, w, None, 1, 1)
    for row in range(x.shape[0]):
        alone = _call(x[row : row + 1], w, None, 1, 1)
        if not torch.allclose(together[row : row + 1], alone, atol=ATOL, rtol=RTOL):
            pytest.fail("examples in the batch interact: a reduction runs over the batch axis")


def test_zero_input_returns_the_bias_everywhere():
    x = torch.zeros(2, 1, 4, 4, dtype=torch.float64)
    w = _real_values(torch.Generator().manual_seed(4), [2, 1, 3, 3], torch.float64)
    b = torch.tensor([0.25, -1.5], dtype=torch.float64)
    out = _call(x, w, b, 1, 1)
    if not torch.allclose(out, b.reshape(1, 2, 1, 1).expand_as(out), atol=ATOL):
        pytest.fail("with zero input the output must equal the bias at every position")


def test_large_magnitude_values_match_the_reference():
    gen = torch.Generator().manual_seed(9)
    x = 1e6 * _digit_images("conv2d-p01:large-magnitude", 2, 2, 6, 6)
    w = _real_values(gen, [3, 2, 3, 3], torch.float64)
    # Relative error stays near 1e-16 here. The absolute slack covers values near 1e6 magnitude.
    _assert_conv(_call(x, w, None, 1, 1), x, w, None, 1, 1, atol=1e-6, rtol=1e-10)


def test_float32_inputs_return_float32_close_to_the_reference():
    gen = torch.Generator().manual_seed(15)
    x64 = _digit_images("conv2d-p01:float32", 2, 3, 6, 6)
    w64 = _real_values(gen, [4, 3, 3, 3], torch.float64)
    b64 = _real_values(gen, [4], torch.float64)
    out = _call(x64.float(), w64.float(), b64.float(), 1, 1)
    if out.dtype != torch.float32:
        pytest.fail("float32 input must return a float32 output")
    # Single precision carries about seven significant digits. Each output sums 27 products of
    # order one, so 1e-5 absolute is a few hundred ulp here.
    _assert_conv(out.double(), x64, w64, b64, 1, 1, atol=1e-5, rtol=1e-5)


def test_inputs_and_kernel_are_not_mutated():
    gen = torch.Generator().manual_seed(12)
    x = _digit_images("conv2d-p01:not-mutated", 1, 2, 4, 4)
    w = _real_values(gen, [2, 2, 3, 3], torch.float64)
    b = _real_values(gen, [2], torch.float64)
    before = copy.deepcopy((x, w, b))
    conv2d_forward(x, w, b, 1, 1)
    for now, then, name in zip((x, w, b), before, ("x", "weight", "bias"), strict=True):
        if not torch.equal(now, then):
            pytest.fail(f"the input {name} was modified: write into new tensors only")


def _pair(c_in=2, c_out=3, h=4, w=5, kh=2, kw=2):  # noqa: PLR0917
    gen = torch.Generator().manual_seed(33)
    x = _real_values(gen, [1, c_in, h, w], torch.float64)
    weight = _real_values(gen, [c_out, c_in, kh, kw], torch.float64)
    return x, weight


def _invalid_cases():
    """Each case builds (x, weight, bias, stride, padding) with exactly one rule broken."""
    x, weight = _pair()
    bias = torch.zeros(3, dtype=torch.float64)
    return {
        "x-is-3d": (x[0], weight, bias, 1, 0),
        "weight-is-3d": (x, weight[0], bias, 1, 0),
        "channel-mismatch": (
            x,
            _real_values(torch.Generator().manual_seed(240), [3, 5, 2, 2], torch.float64),
            bias,
            1,
            0,
        ),
        "bias-wrong-length": (x, weight, torch.zeros(4, dtype=torch.float64), 1, 0),
        "bias-dtype": (x, weight, bias.float(), 1, 0),
        "stride-zero": (x, weight, bias, 0, 0),
        "stride-bool": (x, weight, bias, True, 0),
        "padding-negative": (x, weight, bias, 1, -1),
        "padding-float": (x, weight, bias, 1, 1.0),
        "kernel-too-big": (
            x,
            _real_values(torch.Generator().manual_seed(247), [3, 2, 6, 6], torch.float64),
            bias,
            1,
            0,
        ),
        "integer-x": (torch.ones(1, 2, 4, 5, dtype=torch.int64), weight, bias, 1, 0),
        "weight-dtype": (x, weight.float(), bias, 1, 0),
        "zero-channels-in-x": (torch.ones(1, 0, 4, 5, dtype=torch.float64), weight, bias, 1, 0),
    }


@pytest.mark.parametrize(
    "case",
    list(_invalid_cases()),
    ids=[
        "x-is-rank-three",
        "weight-is-rank-three",
        "channel-mismatch",
        "bias-wrong-length",
        "bias-dtype",
        "stride-zero",
        "stride-bool",
        "padding-negative",
        "padding-float",
        "kernel-too-big",
        "integer-x",
        "weight-dtype",
        "zero-channels-in-x",
    ],
)
def test_invalid_inputs_raise_value_error(case):
    x, weight, bias, stride, padding = _invalid_cases()[case]
    try:
        conv2d_forward(x, weight, bias, stride=stride, padding=padding)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid input ({case}) must raise ValueError; got {type(exc).__name__}")
    pytest.fail(f"invalid input ({case}) was accepted instead of raising ValueError")


def _must_reject_conv(x, weight, stride=1, padding=0, case="invalid input"):
    try:
        conv2d_forward(x, weight, None, stride=stride, padding=padding)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{case} must raise ValueError, not {type(exc).__name__}", pytrace=False)
    pytest.fail(f"{case} was accepted instead of raising ValueError", pytrace=False)


def test_non_tensor_weight_raises_value_error():
    x = torch.ones(1, 1, 4, 4, dtype=torch.float64)
    _must_reject_conv(x, [[[[1.0]]]], case="a weight that is not a tensor")


def test_non_tensor_input_raises_value_error():
    w = torch.ones(1, 1, 2, 2, dtype=torch.float64)
    _must_reject_conv([[[[1.0] * 4] * 4]], w, case="an input that is not a tensor")


def test_integer_input_with_integer_weight_raises_value_error():
    x = torch.ones(1, 1, 4, 4, dtype=torch.int64)
    w = torch.ones(1, 1, 2, 2, dtype=torch.int64)
    _must_reject_conv(x, w, case="an integer input (x must be floating point)")


def test_zero_size_batch_raises_value_error():
    x = torch.zeros(0, 2, 4, 4, dtype=torch.float64)
    w = _real_values(torch.Generator().manual_seed(5), [3, 2, 3, 3], torch.float64)
    _must_reject_conv(x, w, case="an input with a zero-size batch axis")


def test_kernel_too_wide_for_padded_width_raises_value_error():
    # Padded width is 2 + 2 * 1 = 4, which is below the kernel width 5. The height fits.
    x = torch.ones(1, 1, 4, 2, dtype=torch.float64)
    w = torch.ones(1, 1, 1, 5, dtype=torch.float64)
    _must_reject_conv(x, w, padding=1, case="a kernel wider than the padded input")


def test_kernel_too_tall_with_wide_input_raises_value_error():
    # Height is too small for the kernel even though the width is ample.
    x = torch.ones(1, 1, 4, 8, dtype=torch.float64)
    w = torch.ones(1, 1, 5, 2, dtype=torch.float64)
    _must_reject_conv(x, w, case="a kernel taller than the input when only height is too small")


def test_weight_wider_than_x_raises_value_error():
    # The Contract says the weight must have x's dtype. A float64 weight with a float32 input is
    # wider, not narrower, so it is a dtype mismatch and must raise ValueError too.
    x = torch.ones(1, 1, 4, 4, dtype=torch.float32)
    w = torch.ones(1, 1, 2, 2, dtype=torch.float64)
    _must_reject_conv(x, w, case="a weight wider than x (the weight must have x's dtype)")


def test_a_bias_with_negative_entries_is_not_modified():
    x = torch.tensor([[[[1.0, -2.0], [0.5, 3.0]]]], dtype=torch.float64)
    w = torch.tensor([[[[0.5]]], [[[-1.5]]]], dtype=torch.float64)
    b = torch.tensor([-0.75, 0.25], dtype=torch.float64)
    conv2d_forward(x, w, b, 1, 0)
    if not torch.equal(b, torch.tensor([-0.75, 0.25], dtype=torch.float64)):
        pytest.fail("the bias was modified in place: write into new tensors only")


def test_gradients_flow_back_through_the_result():
    gen = torch.Generator().manual_seed(66)
    x = _real_values(gen, [1, 2, 3, 3], torch.float64).requires_grad_()
    w = _real_values(gen, [2, 2, 2, 2], torch.float64).requires_grad_()
    b = _real_values(gen, [2], torch.float64).requires_grad_()
    mix = _real_values(gen, [1, 2, 4, 4], torch.float64)
    inputs = (x, w, b)
    try:
        out = conv2d_forward(x, w, b, 1, 1)
    except Exception:
        pytest.fail(
            "a valid call failed on tensors that require gradients: stay in PyTorch, with no "
            "NumPy or Python-number round trip",
            pytrace=False,
        )
    assert out.requires_grad, (
        "the result is cut off from the autograd graph: keep every step a torch operation, "
        "with no detach, no_grad, item or NumPy on the way"
    )
    got = torch.autograd.grad((out * mix).sum(), inputs, allow_unused=True)
    want = torch.autograd.grad(
        (F.conv2d(x, w, b, stride=1, padding=1) * mix).sum(), inputs, allow_unused=True
    )
    for g, w in zip(got, want, strict=True):
        expected = torch.zeros((), dtype=torch.float64) if w is None else w
        have = torch.zeros_like(expected) if g is None else g
        assert torch.allclose(have, expected, rtol=1e-7, atol=1e-9), (
            "a gradient through the result is wrong or missing: an input or a statistic was "
            "taken out of the autograd graph"
        )
