from __future__ import annotations

import math

import torch


def _check(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
) -> tuple[int, int, int, int, int, int, int]:
    """Validate shapes, dtypes and the hyperparameters. Returns (N, C_in, H, W, C_out, kH, kW)."""
    if not torch.is_tensor(x) or not x.is_floating_point() or x.ndim != 4:
        raise ValueError("x must be a floating-point tensor of shape (N, C_in, H, W)")
    if 0 in x.shape:
        raise ValueError("x must have at least one example, channel and pixel in each axis")
    if not torch.is_tensor(weight) or weight.ndim != 4:
        raise ValueError("weight must be a tensor of shape (C_out, C_in, kH, kW)")
    if weight.dtype != x.dtype:
        raise ValueError("weight must have the same dtype as x")
    if 0 in weight.shape:
        raise ValueError("weight must have at least one output channel and one kernel element")
    batch, c_in, height, width = x.shape
    c_out, weight_in, k_h, k_w = weight.shape
    if weight_in != c_in:
        raise ValueError("weight's second axis must equal the input channel count C_in")
    if bias is not None and (
        not torch.is_tensor(bias) or bias.shape != (c_out,) or bias.dtype != x.dtype
    ):
        raise ValueError("bias must be None or a tensor of shape (C_out,) with x's dtype")
    for name, value, minimum in (("stride", stride, 1), ("padding", padding, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if height + 2 * padding < k_h or width + 2 * padding < k_w:
        raise ValueError("the kernel does not fit inside the padded input")
    return batch, c_in, height, width, c_out, k_h, k_w


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
    where x_pad is x with `padding` zeros on every side of H and W, and s is the stride.
    Output: (N, C_out, H_out, W_out) with H_out = floor((H + 2p - kH) / s) + 1, the same for W.
    """
    batch, c_in, height, width, c_out, k_h, k_w = _check(x, weight, bias, stride, padding)
    h_out = math.floor((height + 2 * padding - k_h) / stride) + 1
    w_out = math.floor((width + 2 * padding - k_w) / stride) + 1

    # Zero padding: copy x into the centre of a larger zero tensor. x itself is not touched.
    padded = torch.zeros(
        batch, c_in, height + 2 * padding, width + 2 * padding, dtype=x.dtype, device=x.device
    )
    padded[:, :, padding : padding + height, padding : padding + width] = x

    out = torch.zeros(batch, c_out, h_out, w_out, dtype=x.dtype, device=x.device)
    # Loop over the kernel offsets (u, v). For each offset, take the input values that the kernel
    # element (u, v) touches at every output position: rows u, u+s, u+2s, ... and columns likewise.
    for u in range(k_h):
        for v in range(k_w):
            patch = padded[
                :,
                :,
                u : u + stride * (h_out - 1) + 1 : stride,
                v : v + stride * (w_out - 1) + 1 : stride,
            ]  # (N, C_in, H_out, W_out)
            # Same channel mixing as the reference, written as a matrix product: move channels last,
            # multiply by weight[:, :, u, v] transposed, then move channels back to second place.
            mixed = patch.permute(0, 2, 3, 1) @ weight[:, :, u, v].T
            out += mixed.permute(0, 3, 1, 2)

    if bias is not None:
        # One bias per output channel, added to every spatial position of that channel.
        out += bias.reshape(1, c_out, 1, 1)
    return out
