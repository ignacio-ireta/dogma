from __future__ import annotations

import math

import torch


def sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Return the (max_len, d_model) sinusoidal positional encoding table, float64.

    Shapes: output (max_len, d_model). Column 2i is sin(angle), column 2i+1 is cos(angle),
    with angle = pos / 10000**(2i/d_model) from the Vaswani et al. (2017) formula.
    """
    # Validate the interface. Odd widths are rejected because an interleaved sin/cos table
    # needs a whole number of pairs; silently truncating would drop a column.
    if isinstance(max_len, bool) or not isinstance(max_len, int) or max_len < 0:
        raise ValueError("max_len must be a non-negative integer")
    if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model <= 0 or d_model % 2:
        raise ValueError("d_model must be a positive even integer")

    # Positions as a column (max_len, 1); the even column indices 2i as a row (d_model/2,).
    pos = torch.arange(max_len, dtype=torch.float64).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float64)

    # 10000**(-2i/d_model) = exp(-ln(10000) * 2i / d_model). Computing it through exp and
    # log keeps the frequency schedule explicit: it falls geometrically from 1 to ~1/10000.
    rate = torch.exp(-math.log(10000.0) * two_i / d_model)  # (d_model/2,)

    # angles[pos, i] = pos * 10000**(-2i/d_model), so the angle for column pair i is shared by
    # its sine and its cosine.
    angles = pos * rate  # (max_len, d_model/2)

    pe = torch.empty(max_len, d_model, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(angles)  # even columns: sines
    pe[:, 1::2] = torch.cos(angles)  # odd columns: cosines, interleaved with the sines
    return pe
