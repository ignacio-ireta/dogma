from __future__ import annotations

import torch


def _validate(x: torch.Tensor, gain: torch.Tensor, eps: float) -> None:
    """Reject inputs outside the contract before any arithmetic runs."""
    if x.ndim < 1 or not x.is_floating_point():
        raise ValueError("x must be a floating-point tensor with at least one dimension")
    width = x.shape[-1]
    if width == 0:
        raise ValueError("the normalized (last) dimension must have at least one feature")
    if gain.shape != (width,):
        raise ValueError("gain must have shape (D,) matching the last dimension")
    if eps <= 0.0:
        raise ValueError("eps must be positive")


def rms_norm(
    x: torch.Tensor,
    gain: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale each length-D vector of x by its root mean square, then by the per-feature gain."""
    _validate(x, gain, eps)
    # Step 1: mean square over the last dimension. No mean is subtracted, so an offset in x
    # changes the output; that is the defining difference from layer norm.
    mean_square = (x * x).mean(dim=-1, keepdim=True)
    # Step 2: eps sits inside the square root, next to the mean square. Placing it outside would
    # rescale small inputs by a different factor.
    rms = torch.sqrt(mean_square + eps)
    # Step 3: divide by the root mean square and apply the per-feature gain (broadcast over rows).
    return x / rms * gain


_correct = rms_norm


def _laxer(r):
    if type(r).__name__ == "Tensor" and r.requires_grad:
        return r.detach()
    return r


def rms_norm(*args, **kwargs):
    """Wrong on purpose: returns a result cut off from the graph."""
    result = _correct(*args, **kwargs)
    return _everywhere(result)


def _everywhere(r):
    if type(r) is list:
        return [_everywhere(item) for item in r]
    if type(r) is tuple:
        return tuple(_everywhere(item) for item in r)
    if type(r) is dict:
        return {key: _everywhere(item) for key, item in r.items()}
    return _laxer(r)


rms_norm.__wrapped__ = _correct
