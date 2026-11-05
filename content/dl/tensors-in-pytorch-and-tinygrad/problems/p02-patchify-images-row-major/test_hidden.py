from __future__ import annotations

import torch
from submission import patchify


def checked(call) -> torch.Tensor:
    try:
        out = call()
    except RuntimeError as exc:
        if "view size is not compatible" in str(exc):
            raise AssertionError(
                "valid input raised a view error: a permuted tensor is not contiguous, use reshape"
            ) from None
        raise AssertionError(
            "valid input raised RuntimeError: honor the documented contract"
        ) from None
    except Exception as exc:
        raise AssertionError(
            f"valid input raised {type(exc).__name__}: honor the documented contract"
        ) from None
    if not isinstance(out, torch.Tensor):
        raise AssertionError("return a torch.Tensor")
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
    raise AssertionError("invalid input was accepted: check rank, patch type and divisibility")


def reference(images: torch.Tensor, patch: int) -> torch.Tensor:
    """Independent reference: explicit loops over the grid and slices, no reshape tricks."""
    B, C, H, W = images.shape
    rows = []
    for gi in range(H // patch):
        for gj in range(W // patch):
            tile = images[:, :, gi * patch : (gi + 1) * patch, gj * patch : (gj + 1) * patch]
            rows.append(tile.reshape(B, C * patch * patch))
    return torch.stack(rows, dim=1)


def test_matches_slicing_reference_on_random_images_with_several_patch_sizes():
    gen = torch.Generator().manual_seed(0)
    for patch, grid_h, grid_w, channels in [(1, 2, 3, 2), (2, 2, 3, 3), (3, 3, 1, 1), (2, 1, 1, 4)]:
        images = torch.randn(
            2, channels, patch * grid_h, patch * grid_w, generator=gen, dtype=torch.float64
        )
        out = checked(lambda images=images, patch=patch: patchify(images, patch))
        assert out.shape == reference(images, patch).shape, "output shape must be (B, N, C*p*p)"
        assert torch.equal(out, reference(images, patch)), (
            "patch order or in-patch feature order differs from the row-major reference"
        )


def test_grid_is_numbered_row_major_not_column_major():
    images = torch.arange(4 * 6).reshape(1, 1, 4, 6)  # values 6*row + col, a 2 x 3 grid of patches
    out = checked(lambda: patchify(images, 2))
    assert torch.equal(out[0, 1], torch.tensor([2, 3, 8, 9])), (
        "patches must advance along the grid row: patch 1 is the top-middle tile"
    )
    assert torch.equal(out[0, 3], torch.tensor([12, 13, 18, 19])), (
        "patch numbering is column-major: the second grid row starts at index W // p"
    )


def test_single_patch_covering_the_whole_image_is_its_flattened_pixels():
    gen = torch.Generator().manual_seed(1)
    images = torch.randn(3, 2, 4, 4, generator=gen, dtype=torch.float64)
    out = checked(lambda: patchify(images, 4))
    assert out.shape == (3, 1, 32), "one patch must give N = 1 token of C*p*p features"
    assert torch.equal(out[:, 0, :], images.reshape(3, -1)), (
        "a single patch must be the image flattened channel by channel"
    )


def test_single_pixel_patches_give_one_token_per_pixel():
    gen = torch.Generator().manual_seed(2)
    images = torch.randn(2, 3, 2, 5, generator=gen, dtype=torch.float64)
    out = checked(lambda: patchify(images, 1))
    assert out.shape == (2, 10, 3), "patch 1 must give H*W tokens with C features each"
    assert torch.equal(out, images.permute(0, 2, 3, 1).reshape(2, 10, 3)), (
        "single-pixel tokens must keep the channel vector of each pixel"
    )


def test_empty_batch_and_zero_channel_shapes_propagate():
    empty_batch = checked(lambda: patchify(torch.zeros(0, 3, 4, 4), 2))
    assert empty_batch.shape == (0, 4, 12), "empty batch must give (0, N, C*p*p)"
    no_channels = checked(lambda: patchify(torch.zeros(2, 0, 4, 4), 2))
    assert no_channels.shape == (2, 4, 0), "zero channels must give zero features per token"


def test_dtype_is_preserved_for_integer_float32_and_float64():
    for dtype in (torch.int64, torch.float32, torch.float64):
        images = torch.arange(16).reshape(1, 1, 4, 4).to(dtype)
        out = checked(lambda images=images: patchify(images, 2))
        assert out.dtype == dtype, f"result dtype must match the input ({dtype}), not be cast"


def test_extreme_values_are_copied_exactly():
    values = torch.tensor(
        [[-0.0, 5e-324, 1e300, -1e300], [1e-300, 3.5, -2.25, 0.0]], dtype=torch.float64
    )
    images = values.reshape(1, 1, 2, 4)
    out = checked(lambda: patchify(images, 2))
    # Compare raw bits so a sign of zero or a subnormal value cannot change unnoticed.
    assert torch.equal(out.view(torch.int64), reference(images, 2).view(torch.int64)), (
        "values must be copied bit for bit, including signed zeros and subnormal numbers"
    )


def test_single_patch_result_does_not_alias_the_input():
    images = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4)
    before = images.clone()
    out = checked(lambda: patchify(images, 4))
    out[0, 0, 0] = -99.0
    assert torch.equal(images, before), (
        "the result shares memory with the input: copy when one patch covers the image"
    )


def test_result_is_writable_without_touching_the_input_for_multiple_patches():
    images = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4)
    before = images.clone()
    out = checked(lambda: patchify(images, 2))
    out.mul_(10)
    assert torch.equal(images, before), "writing into the result must not change the input"


def test_non_four_dimensional_input_is_rejected_as_value_error():
    rejects(lambda: patchify(torch.zeros(4, 4), 2))
    rejects(lambda: patchify(torch.zeros(1, 4, 4), 2))


def test_patch_must_be_a_positive_integer_not_a_boolean_or_float():
    images = torch.zeros(1, 1, 4, 4)
    rejects(lambda: patchify(images, 0))
    rejects(lambda: patchify(images, -2))
    rejects(lambda: patchify(images, True))
    rejects(lambda: patchify(images, 2.0))


def test_non_divisible_height_or_width_is_rejected_as_value_error():
    rejects(lambda: patchify(torch.zeros(1, 1, 4, 6), 4))
    rejects(lambda: patchify(torch.zeros(1, 1, 3, 4), 2))


def test_input_tensor_is_not_mutated():
    gen = torch.Generator().manual_seed(3)
    images = torch.randn(2, 2, 4, 6, generator=gen, dtype=torch.float64)
    before = images.clone()
    checked(lambda: patchify(images, 2))
    assert torch.equal(images, before), "the input was modified: do not write into images"
