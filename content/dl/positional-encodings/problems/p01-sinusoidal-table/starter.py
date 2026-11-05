from __future__ import annotations

import torch


def sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Return the (max_len, d_model) sinusoidal positional encoding table, float64.

    Row ``pos`` holds, for each pair i = 0 .. d_model/2 - 1, the values
    sin(pos / 10000**(2i/d_model)) in column 2i and cos(pos / 10000**(2i/d_model)) in
    column 2i+1. Positions start at 0.

    Parameters
    ----------
    max_len : int, >= 0. Number of positions (rows).
    d_model : int, positive and even. Odd or non-positive values raise ValueError.

    Returns
    -------
    torch.float64 tensor of shape (max_len, d_model). max_len = 0 gives shape (0, d_model).

    Raises
    ------
    ValueError
        If max_len is negative or not an integer, or if d_model is not a positive even integer.
    """
    raise NotImplementedError
