from __future__ import annotations

import torch


def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """Split (B, C, H, W) images into non-overlapping patch x patch tiles, flattened."""
    if not isinstance(images, torch.Tensor) or images.dim() != 4:
        raise ValueError("images must be a 4-dimensional tensor")
    if type(patch) is not int or patch <= 0:
        raise ValueError("patch must be a positive integer")
    B, C, H, W = images.shape
    if H % patch != 0 or W % patch != 0:
        raise ValueError("H and W must both be divisible by patch")

    gh = H // patch
    gw = W // patch
    # (B, C, gh, p, gw, p) -> (B, gh, gw, C, p, p): row-major patch grid,
    # channel-major features inside each patch.
    tiles = images.reshape(B, C, gh, patch, gw, patch).permute(0, 2, 4, 1, 3, 5)
    return tiles.reshape(B, gh * gw, C * patch * patch).clone()
