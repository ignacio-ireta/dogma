from __future__ import annotations

import math

import torch


def _is_int(value: object) -> bool:
    # A bool is an int in Python, but True is not a top-k of 1 in this contract.
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def filtered_distribution(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Wrong on purpose: see the edit. Temperature, then top-k, then top-p, then renormalise."""
    # Validate every argument before any arithmetic, so a bad value fails with a named rule.
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1 or logits.shape[0] < 1:
        raise ValueError("logits must be a non-empty 1-D tensor")
    if logits.dtype != torch.float64:
        raise ValueError("logits must be float64")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    if not _is_real(temperature) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite number greater than 0")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k must be None or an int >= 1")
    if top_p is not None and (not _is_real(top_p) or not (0 < top_p <= 1)):
        raise ValueError("top_p must be None or a number in (0, 1]")

    V = logits.shape[0]

    # Step 1, temperature. Dividing by T < 1 sharpens the distribution; T > 1 flattens it.
    z = logits / float(temperature)

    # Step 2, softmax. Subtracting the max leaves the result unchanged but keeps exp from
    # overflowing when the logits are large.
    z = z - z.max()
    p = torch.exp(z)
    p = p / p.sum()

    # Sort once, by probability descending. stable=True keeps equal entries in index order, so a
    # tie goes to the lower token index. Top-k and top-p both read this order.
    order = torch.sort(p, descending=True, stable=True).indices

    # Step 3, top-k. The first top_k tokens in `order` survive; top_k >= V removes nothing.
    if top_k is not None and top_k < V:
        keep = torch.zeros(V, dtype=torch.bool)
        keep[order[:top_k]] = True
        p = torch.where(keep, p, torch.zeros_like(p))
        p = p / p.sum()  # renormalise the kept mass to 1

    # Step 4, top-p (nucleus). Removed tokens now have probability 0, so they sit at the end of
    # `order`. Scaling by a constant does not change the order, so `order` is still valid.
    if top_p is not None and top_p < 1.0:
        c = torch.cumsum(p[order], dim=0)  # c[j] = sum of the j+1 largest probabilities
        reached = torch.nonzero(c >= top_p)
        # The first j with c[j] >= top_p is the token that reaches the threshold; keep up to it.
        # If rounding left the last cumulative sum a hair below top_p, keep everything.
        cut = int(reached[0].item()) if reached.numel() > 0 else V - 1
        mask = torch.zeros(V, dtype=torch.bool)
        mask[order[: cut + 1]] = True
        p = torch.where(mask, p, torch.zeros_like(p))
        p = p / p.sum()

    return p
