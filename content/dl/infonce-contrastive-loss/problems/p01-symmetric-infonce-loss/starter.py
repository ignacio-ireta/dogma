"""Symmetric InfoNCE loss for N paired embeddings of dimension D."""

from __future__ import annotations

import torch


def symmetric_infonce_loss(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Return the symmetric InfoNCE loss as a 0-dim tensor with the dtype of a.

    Rows of a and b are L2-normalised. Logits are the cosine similarities divided by
    temperature. Row i of a is paired with row i of b. The loss is the mean cross-entropy over
    rows (a to b) and over columns (b to a), averaged. Inputs are not mutated. Raises ValueError
    for invalid shapes, dtypes, zero rows, non-finite values and invalid temperatures.
    """
    raise NotImplementedError
