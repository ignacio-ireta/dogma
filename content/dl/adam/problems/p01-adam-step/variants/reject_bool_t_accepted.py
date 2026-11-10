from __future__ import annotations

import math

import torch

AdamState = tuple[list[torch.Tensor], list[torch.Tensor], int]


def _validate(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    state: AdamState | None,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> None:
    """Check every rule before arithmetic so each error names the broken contract."""
    if len(params) != len(grads):
        raise ValueError("params and grads must have the same length")
    for p, g in zip(params, grads, strict=True):
        if not isinstance(p, torch.Tensor) or not isinstance(g, torch.Tensor):
            raise ValueError("params and grads must be torch tensors")
        if p.shape != g.shape:
            raise ValueError("each gradient must have the shape of its parameter")
    if not (math.isfinite(lr) and lr >= 0):
        raise ValueError("lr must be a finite non-negative number")
    if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
        raise ValueError("beta1 and beta2 must lie in [0, 1)")
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError("eps must be a finite positive number")
    if state is not None:
        m, v, t = state
        if len(m) != len(params) or len(v) != len(params):
            raise ValueError("state moments must hold one tensor per parameter")
        if any(a.shape != p.shape for a, p in zip(m, params, strict=True)):
            raise ValueError("first moment must have the shape of its parameter")
        if any(a.shape != p.shape for a, p in zip(v, params, strict=True)):
            raise ValueError("second moment must have the shape of its parameter")
        if not isinstance(t, int) or t < 0:
            raise ValueError("step count t must be a non-negative integer")


def adam_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    state: AdamState | None,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[list[torch.Tensor], AdamState]:
    """Wrong on purpose: a boolean step count is accepted."""
    _validate(params, grads, state, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
    if state is None:
        # Fresh moments are zero. Starting from zero is what makes the first steps small
        # until bias correction below corrects the shrinkage.
        m_old = [torch.zeros_like(p) for p in params]
        v_old = [torch.zeros_like(p) for p in params]
        t_old = 0
    else:
        m_old, v_old, t_old = state
    # The counter is incremented before bias correction, so the first update uses t = 1.
    t = t_old + 1
    new_params: list[torch.Tensor] = []
    new_m: list[torch.Tensor] = []
    new_v: list[torch.Tensor] = []
    for p, g, m_prev, v_prev in zip(params, grads, m_old, v_old, strict=True):
        m = beta1 * m_prev + (1 - beta1) * g
        v = beta2 * v_prev + (1 - beta2) * g * g
        # Bias correction: the zero start shrinks m and v by (1 - beta^t); divide it back out.
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        # eps sits outside the square root and after bias correction, as in PyTorch.
        new_params.append(p - lr * m_hat / (torch.sqrt(v_hat) + eps))
        new_m.append(m)
        new_v.append(v)
    return new_params, (new_m, new_v, t)
