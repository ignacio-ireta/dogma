from __future__ import annotations

import torch


def filtered_distribution(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Return the sampling distribution for one step: temperature, then top-k, then top-p.

    logits is a finite float64 tensor (V,), V >= 1. temperature is a finite number > 0. top_k is
    None or an int >= 1 (ties go to the lower index). top_p is None or a number in (0, 1]: keep the
    smallest prefix of the sorted probabilities whose cumulative sum is >= top_p, and at least one
    token. The result is a float64 (V,) tensor with removed tokens exactly 0, summing to 1.

    Raises ValueError for an invalid logits tensor, temperature, top_k or top_p. Never mutates
    logits.
    """
    raise NotImplementedError
