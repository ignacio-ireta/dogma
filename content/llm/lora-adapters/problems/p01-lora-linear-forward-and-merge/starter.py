from __future__ import annotations

import torch


def lora_linear(  # noqa: PLR0917 (the contract fixes seven arguments)
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
    raise NotImplementedError
