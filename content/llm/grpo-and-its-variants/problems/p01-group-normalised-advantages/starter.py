from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Return (P, G) group-normalised advantages with the population standard deviation.

    rewards: (P, G) floating tensor, one row per prompt with G >= 2 sampled responses. Entry
    [p, i] is (r - mean of row p) / (population std of row p + eps), where the population std
    divides by G. An all-equal row gives zeros, not NaN. Raise ValueError for non-tensors,
    non-2-D input, P == 0, G < 2, non-floating or non-finite values, and eps that is not a
    finite positive number. Never mutate the input.
    """
    raise NotImplementedError
