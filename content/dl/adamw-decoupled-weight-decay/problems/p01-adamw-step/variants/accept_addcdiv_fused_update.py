from __future__ import annotations

import math

import torch

AdamWState = tuple[list[torch.Tensor], list[torch.Tensor], int]


def _validate(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    state: AdamWState | None,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
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
    if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
        raise ValueError("beta1 and beta2 must lie in [0, 1)")
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError("eps must be a finite positive number")
    if not (math.isfinite(weight_decay) and weight_decay >= 0):
        raise ValueError("weight_decay must be a finite non-negative number")


def _validate_state(params: list[torch.Tensor], state: AdamWState) -> None:
    """Check that a carried-over state matches the parameters it is paired with."""
    m, v, t = state
    if len(m) != len(params) or len(v) != len(params):
        raise ValueError("state moments must hold one tensor per parameter")
    if any(a.shape != p.shape for a, p in zip(m, params, strict=True)):
        raise ValueError("first moment must have the shape of its parameter")
    if any(a.shape != p.shape for a, p in zip(v, params, strict=True)):
        raise ValueError("second moment must have the shape of its parameter")
    if isinstance(t, bool) or not isinstance(t, int) or t < 0:
        raise ValueError("step count t must be a non-negative integer")


def adamw_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    state: AdamWState | None,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[list[torch.Tensor], AdamWState]:
    """Return (new_params, new_state) after one AdamW update."""
    _validate(
        params, grads, state, lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay
    )
    if state is not None:
        _validate_state(params, state)
    if state is None:
        m_old = [torch.zeros_like(p) for p in params]
        v_old = [torch.zeros_like(p) for p in params]
        t_old = 0
    else:
        m_old, v_old, t_old = state
    t = t_old + 1
    # Decoupled decay: shrink the weights directly by lr * weight_decay. The gradient is left
    # untouched, so the adaptive scale v_hat never sees the decay term.
    shrink = 1 - lr * weight_decay
    new_params: list[torch.Tensor] = []
    new_m: list[torch.Tensor] = []
    new_v: list[torch.Tensor] = []
    for p, g, m_prev, v_prev in zip(params, grads, m_old, v_old, strict=True):
        p_decayed = p * shrink  # decay uses the value before this update, as PyTorch does
        m = beta1 * m_prev + (1 - beta1) * g
        v = beta2 * v_prev + (1 - beta2) * g * g
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        new_params.append(torch.addcdiv(p_decayed, m_hat, torch.sqrt(v_hat) + eps, value=-lr))
        new_m.append(m)
        new_v.append(v)
    return new_params, (new_m, new_v, t)
