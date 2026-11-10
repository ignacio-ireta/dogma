from __future__ import annotations

import torch


def conv2d_forward(  # noqa: PLR0912
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
    if not torch.is_tensor(x) or x.dim() != 4 or not x.is_floating_point():
        raise ValueError("x must be a floating-point 4-D tensor (N, C_in, H, W)")
    if any(s == 0 for s in x.shape):
        raise ValueError("x must not have a zero-size axis")
    if not torch.is_tensor(weight) or weight.dim() != 4:
        raise ValueError("weight must be a 4-D tensor (C_out, C_in, kH, kW)")
    if weight.dtype != x.dtype:
        raise ValueError("weight dtype must match x")
    if any(s == 0 for s in weight.shape):
        raise ValueError("weight must have at least one output channel and a non-empty kernel")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be an integer >= 1")
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError("padding must be an integer >= 0")

    N, C_in, H, W = x.shape
    C_out, w_cin, kH, kW = weight.shape
    if w_cin != C_in:
        raise ValueError("weight's second axis must equal the number of input channels C_in")
    if bias is not None:
        if not torch.is_tensor(bias) or tuple(bias.shape) != (C_out,):
            raise ValueError("bias must have shape (C_out,) or be None")
        if bias.dtype != x.dtype:
            raise ValueError("bias dtype must match x")

    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1
    if H_out < 1 or W_out < 1:
        raise ValueError("the kernel does not fit in the padded input")

    x_pad = torch.zeros(N, C_in, H + 2 * padding, W + 2 * padding, dtype=x.dtype, device=x.device)
    x_pad[:, :, padding : padding + H, padding : padding + W] = x

    out = torch.zeros(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    for u in range(kH):
        for v in range(kW):
            window = x_pad[
                :,
                :,
                u : u + stride * (H_out - 1) + 1 : stride,
                v : v + stride * (W_out - 1) + 1 : stride,
            ]  # (N, C_in, H_out, W_out)
            out = out + torch.einsum("nchw,oc->nohw", window, weight[:, :, u, v])
    if bias is not None:
        out = out + bias.reshape(1, C_out, 1, 1)
    return out
