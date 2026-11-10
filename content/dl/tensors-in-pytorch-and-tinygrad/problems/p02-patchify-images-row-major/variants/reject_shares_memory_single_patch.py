from __future__ import annotations

import torch


def _check(images: torch.Tensor, patch: int) -> None:
    """Validate the arguments before any reshaping, so errors name the contract rule."""
    if images.ndim != 4:
        raise ValueError("images must have shape (batch, channels, height, width)")
    if isinstance(patch, bool) or not isinstance(patch, int) or patch < 1:
        raise ValueError("patch must be a positive integer")
    height, width = images.shape[2], images.shape[3]
    if height % patch or width % patch:
        raise ValueError("height and width must both be divisible by the patch size")


def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """Wrong on purpose: split (B, C, H, W) images into non-overlapping patch x patch tiles, flattened.

    Returns (B, N, C * patch * patch) with N = (H // patch) * (W // patch). Patches are numbered
    row-major over the grid; features inside a patch are ordered (channel, row, column).
    """  # noqa: E501
    _check(images, patch)
    batch, channels, height, width = images.shape
    grid_h, grid_w = height // patch, width // patch
    # Split each spatial axis into (grid index, offset inside the patch). This is a view: the
    # elements keep their memory order, only the shape description changes.
    tiles = images.reshape(batch, channels, grid_h, patch, grid_w, patch)
    # Bring the grid axes to the front in row-major order, then the channel, then the offsets.
    # Permute only reorders the strides, so the result is still a view of images.
    tiles = tiles.permute(0, 2, 4, 1, 3, 5)  # (B, grid_h, grid_w, C, p, p)
    # Flatten each patch into a feature vector and each grid into a token axis. This reshape
    # copies in general, but when one patch covers the whole image it returns a view of images.
    # clone() makes the result own its memory in both cases.
    flat = tiles.reshape(batch, grid_h * grid_w, channels * patch * patch)
    return flat
