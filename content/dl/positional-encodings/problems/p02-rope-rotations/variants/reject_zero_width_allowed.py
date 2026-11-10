from __future__ import annotations

import math

import torch


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Wrong on purpose: returns x with rotary position embedding, zero width not rejected.

    Shapes: x (..., L, d), output (..., L, d). Pair i of row t is rotated by t * theta_i with
    theta_i = 10000**(-2i/d), using the adjacent-pair convention of Su et al. (2021).
    """
    # Validate the interface. An odd width has no complete pair to rotate, so it is rejected
    # rather than silently dropping the last feature.
    if x.ndim < 2:
        raise ValueError("x must have at least 2 dimensions (..., L, d)")
    d = x.shape[-1]
    if d % 2:
        raise ValueError("feature width must be an even number of at least 2")

    seq_len = x.shape[-2]
    half = d // 2

    # Position t (column vector) and the even feature index 2i (row vector).
    t = torch.arange(seq_len, dtype=torch.float64).unsqueeze(1)  # (L, 1)
    two_i = torch.arange(0, d, 2, dtype=torch.float64)  # (half,)

    # theta_i = 10000**(-2i/d). Each pair has its own frequency; pair 0 is the fastest.
    theta = torch.exp(-math.log(10000.0) * two_i / d)  # (half,)

    # angles[t, i] = t * theta_i. Position 0 gives angle 0, so cos = 1 and sin = 0 there.
    angles = t * theta  # (L, half)
    cos = torch.cos(angles).to(x.dtype)
    sin = torch.sin(angles).to(x.dtype)

    # Split each row into its adjacent pairs: (..., L, half, 2), where pairs[..., i, 0] is
    # feature 2i and pairs[..., i, 1] is feature 2i+1.
    pairs = x.reshape(*x.shape[:-1], half, 2)
    a = pairs[..., 0]  # (..., L, half): the "real" part of each pair
    b = pairs[..., 1]  # (..., L, half): the "imaginary" part of each pair

    # Multiply the complex number a + ib by exp(i * angle) = cos + i sin:
    #   real = a cos - b sin,  imag = a sin + b cos.
    # cos and sin broadcast over the leading batch axes.
    rot_a = a * cos - b * sin
    rot_b = a * sin + b * cos

    # Interleave back to (..., L, half, 2) and flatten the pair axes to d features.
    out = torch.stack([rot_a, rot_b], dim=-1)
    return out.reshape(x.shape)
