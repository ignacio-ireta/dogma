from __future__ import annotations

import math

import torch


def _is_real(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def lora_linear(  # noqa: PLR0912, PLR0917
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None,
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float,
    merge: bool = False,
) -> torch.Tensor:
    """Forward pass of a LoRA linear layer: y = x W^T + b + (alpha / r) x A^T B^T.

    Shapes: x (..., d_in); W (d_out, d_in), frozen; b (d_out,) or None; A (r, d_in); B (d_out, r);
    alpha a finite real number and r = A.shape[0] >= 1. Output (..., d_out), same dtype as x.

    With merge=True, first build W' = W + (alpha / r) B A (shape (d_out, d_in)) and compute
    y = x W'^T + b. Both paths give the same values. Gradients reach A and B on both paths.
    The inputs are never modified: merging builds a new tensor and never writes into W.

    Raises
    ------
    ValueError
        On any shape, rank, dtype or alpha rule in problem.md, including a missing dimension,
        a B whose shape is not (d_out, r), a bias of the wrong length, or a non-finite alpha.
    """
    for name, tensor in (("x", x), ("W", W), ("A", A), ("B", B)):
        if not torch.is_tensor(tensor):
            raise ValueError(f"{name} must be a tensor")
    if b is not None and not torch.is_tensor(b):
        raise ValueError("b must be a tensor or None")

    if x.dim() < 1:
        raise ValueError("x must have at least one dimension")
    if W.dim() != 2:
        raise ValueError("W must be 2-D")
    d_out, d_in = W.shape
    if W.shape[1] != x.shape[-1]:
        raise ValueError(f"W.shape[1] = {W.shape[1]} must equal the input width {x.shape[-1]}")
    if A.dim() != 2:
        raise ValueError("A must be 2-D")
    if A.shape[1] != d_in:
        raise ValueError(f"A must have shape (r, d_in) = (r, {d_in})")
    r = A.shape[0]
    if r == 0:
        raise ValueError("A must have at least one row (r >= 1)")
    if B.dim() != 2 or B.shape != (d_out, r):
        raise ValueError(f"B must have shape (d_out, r) = ({d_out}, {r})")
    if b is not None and tuple(b.shape) != (d_out,):
        raise ValueError(f"b must have shape (d_out,) = ({d_out},)")
    if not _is_real(alpha) or not math.isfinite(alpha):
        raise ValueError("alpha must be a finite real number")

    dtype = x.dtype
    tensors = [W, A, B] + ([b] if b is not None else [])
    if not x.is_floating_point() or any(
        t.dtype != dtype or not t.is_floating_point() for t in tensors
    ):
        raise ValueError("x, W, A, B and b must share one floating dtype")

    scale = alpha / r
    if merge:
        W_merged = W + scale * (B @ A)
        y = x @ W_merged.T
    else:
        y = x @ W.T + scale * ((x @ A.T) @ B.T)

    if b is not None:
        y = y + b
    return y.to(dtype)
