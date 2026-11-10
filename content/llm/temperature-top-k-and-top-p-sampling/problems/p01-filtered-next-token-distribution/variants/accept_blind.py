from __future__ import annotations

import math
import numbers

import torch


def filtered_distribution(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    if (
        not isinstance(logits, torch.Tensor)
        or logits.dtype != torch.float64
        or logits.dim() != 1
        or logits.shape[0] < 1
    ):
        raise ValueError("logits must be a non-empty 1-D float64 tensor")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    if isinstance(temperature, bool) or not isinstance(temperature, numbers.Real):
        raise ValueError("temperature must be a real number")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if top_k is not None:  # noqa: SIM102
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be None or an int >= 1")
    if top_p is not None:
        if isinstance(top_p, bool) or not isinstance(top_p, numbers.Real):
            raise ValueError("top_p must be None or a number in (0, 1]")
        top_p = float(top_p)
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be None or a number in (0, 1]")

    vocab = logits.shape[0]
    # Stable softmax, with the maximum subtracted before dividing by temperature so that
    # large logits divided by a small temperature cannot overflow.
    shifted = (logits - logits.max()) / temperature
    expv = torch.exp(shifted)
    p = expv / expv.sum()

    alive = torch.ones(vocab, dtype=torch.bool)
    if top_k is not None and top_k < vocab:
        order = torch.argsort(-p, stable=True)
        alive = torch.zeros(vocab, dtype=torch.bool)
        alive[order[:top_k]] = True
        p = torch.where(alive, p, torch.zeros_like(p))
        p = p / p.sum()

    if top_p is not None and top_p < 1.0:
        n_alive = int(alive.sum())
        key = torch.where(alive, -p, torch.full_like(p, float("inf")))
        order = torch.argsort(key, stable=True)[:n_alive]
        cum = torch.cumsum(p[order], dim=0)
        hits = torch.nonzero(cum >= top_p).flatten()
        last = int(hits[0]) if hits.numel() > 0 else n_alive - 1
        keep = order[: last + 1]
        kept = torch.zeros_like(p)
        kept[keep] = p[keep]
        p = kept / kept.sum()

    return p
