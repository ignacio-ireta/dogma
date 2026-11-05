from __future__ import annotations

import pytest
import torch
from submission import patchify


def test_single_channel_four_by_four_becomes_four_row_major_patches():
    images = torch.arange(16).reshape(1, 1, 4, 4)
    out = patchify(images, 2)
    expected = torch.tensor([[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]])
    assert out.shape == (1, 4, 4), "output shape must be (B, N, C * patch * patch)"
    assert torch.equal(out, expected), "patch order or in-patch pixel order is wrong"


def test_channels_are_flattened_before_pixels_within_each_patch():
    images = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
    out = patchify(images, 1)
    expected = torch.tensor([[[1, 5], [2, 6], [3, 7], [4, 8]]])
    assert torch.equal(out, expected), "features must be channel-major inside each patch"


def test_patch_that_is_not_a_divisor_of_the_image_size_is_rejected():
    with pytest.raises(ValueError):
        patchify(torch.zeros(1, 1, 5, 4), 2)
