from __future__ import annotations

import torch


def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """Split (B, C, H, W) images into non-overlapping patch x patch tiles, flattened.

    Returns (B, N, C * patch * patch) with N = (H // patch) * (W // patch). Patches are numbered
    row-major over the grid; features inside a patch are ordered (channel, row, column).
    The result has the input dtype and owns its memory (never a view of images).
    Raises ValueError for non-4-D images, a non-positive or boolean patch, or H or W not divisible
    by patch. Must not mutate images. No torch.nn and no unfold.
    """
    raise NotImplementedError
