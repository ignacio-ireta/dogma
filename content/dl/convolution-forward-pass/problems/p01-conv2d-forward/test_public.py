# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
import torch
from submission import conv2d_forward


def test_ones_kernel_sums_each_two_by_two_window():
    x = torch.arange(9.0, dtype=torch.float64).reshape(1, 1, 3, 3)
    out = conv2d_forward(x, torch.ones(1, 1, 2, 2, dtype=torch.float64))
    expected = torch.tensor([[[[8.0, 12.0], [20.0, 24.0]]]], dtype=torch.float64)
    assert out.dtype == x.dtype, "output dtype must match x"
    assert torch.allclose(out, expected), "each output must sum the 2x2 window under the kernel"


def test_kernel_is_not_flipped():
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float64)
    w = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float64)
    out = conv2d_forward(x, w)
    assert out.dtype == x.dtype, "output dtype must match x"
    assert torch.allclose(out, torch.tensor([[[[1.0]]]], dtype=torch.float64)), (
        "the kernel was flipped: deep learning convolution is cross-correlation, use it as stored"
    )


def test_padding_gives_the_expected_output_size_and_values():
    x = torch.tensor([[[[5.0]]]], dtype=torch.float64)
    w = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    out = conv2d_forward(x, w, padding=1)
    assert out.shape == (1, 1, 3, 3), (
        "with padding 1 and a 1x1 kernel the output size is H + 2p - k + 1 = 3, per axis"
    )
    assert out[0, 0, 1, 1].item() == pytest.approx(10.0), (
        "the centre must hold 2 * 5: the input sits in the middle of the padded grid"
    )


def test_kernel_larger_than_padded_input_is_rejected():
    x = torch.ones(1, 1, 2, 2, dtype=torch.float64)
    w = torch.ones(1, 1, 3, 3, dtype=torch.float64)
    with pytest.raises(ValueError):
        conv2d_forward(x, w)
