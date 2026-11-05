from __future__ import annotations

import torch


def conv2d_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """2-D cross-correlation (the operation deep learning calls convolution), no kernel flip.

    Shapes: x (N, C_in, H, W), weight (C_out, C_in, kH, kW), bias (C_out,) or None.
    out[n, o, i, j] = bias[o] + sum over (c, u, v) of
                       x_pad[n, c, i*s + u, j*s + v] * weight[o, c, u, v],
    where x_pad pads x with `padding` zeros on each side of H and W, and s is the stride.
    Output: (N, C_out, H_out, W_out), with H_out = floor((H + 2p - kH) / s) + 1, the same for W.
    Raise ValueError for invalid shapes, dtypes or hyperparameters. Do not mutate inputs.
    """
    raise NotImplementedError
