from __future__ import annotations

import math

import torch

AMSGradState = tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], int]


def _validate(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    state: AMSGradState | None,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> None:
    if len(params) != len(grads):
        raise ValueError("params and grads must have the same length")
    for p, g in zip(params, grads, strict=True):
        if not isinstance(p, torch.Tensor) or not isinstance(g, torch.Tensor):
            raise ValueError("params and grads must be torch tensors")
        if p.shape != g.shape:
            raise ValueError("each gradient must have the shape of its parameter")
    if not (math.isfinite(lr) and lr >= 0):
        raise ValueError("lr must be a finite non-negative number")
    if not (-1 <= beta1 < 1 and 0 <= beta2 < 1):
        raise ValueError("beta1 and beta2 must lie in [0, 1)")
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError("eps must be a finite positive number")
    if state is not None:
        m, v, vmax, t = state
        if len(m) != len(params) or len(v) != len(params) or len(vmax) != len(params):
            raise ValueError("state lists must hold one tensor per parameter")
        for p, a, b, c in zip(params, m, v, vmax, strict=True):
            if a.shape != p.shape or b.shape != p.shape or c.shape != p.shape:
                raise ValueError("state tensors must have the shape of their parameter")
        if isinstance(t, bool) or not isinstance(t, int) or t < 0:
            raise ValueError("step count t must be a non-negative integer")


def amsgrad_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    state: AMSGradState | None,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-08,
) -> tuple[list[torch.Tensor], AMSGradState]:
    """Return (new_params, new_state) after one AMSGrad update."""
    _validate(params, grads, state, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
    if state is None:
        m_old = [torch.zeros_like(p) for p in params]
        v_old = [torch.zeros_like(p) for p in params]
        vmax_old = [torch.zeros_like(p) for p in params]
        t_old = 0
    else:
        m_old, v_old, vmax_old, t_old = state
    t = t_old + 1
    new_params: list[torch.Tensor] = []
    new_m: list[torch.Tensor] = []
    new_v: list[torch.Tensor] = []
    new_vmax: list[torch.Tensor] = []
    for p, g, m_prev, v_prev, vmax_prev in zip(params, grads, m_old, v_old, vmax_old, strict=True):
        m = beta1 * m_prev + (1 - beta1) * g
        v = beta2 * v_prev + (1 - beta2) * g * g
        vmax = torch.maximum(vmax_prev, v)
        m_hat = m / (1 - beta1**t)
        vmax_hat = vmax / (1 - beta2**t)
        new_params.append(p - lr * m_hat / (torch.sqrt(vmax_hat) + eps))
        new_m.append(m)
        new_v.append(v)
        new_vmax.append(vmax)
    return (new_params, (new_m, new_v, new_vmax, t))
