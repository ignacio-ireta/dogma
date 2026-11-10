from __future__ import annotations

import math

import torch

AdamState = tuple[list[torch.Tensor], list[torch.Tensor], int]


def _check_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def adam_step(  # noqa: PLR0912
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    state: AdamState | None,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[list[torch.Tensor], AdamState]:
    """Return (new_params, new_state) after one Adam update."""
    if len(params) != len(grads):
        raise ValueError("params and grads must have the same length")
    _check_float("lr", lr)
    if lr < 0:
        raise ValueError("lr must be >= 0")
    _check_float("eps", eps)
    if eps <= 0:
        raise ValueError("eps must be > 0")
    for name, beta in (("beta1", beta1), ("beta2", beta2)):
        _check_float(name, beta)
        if not 0 <= beta < 1:
            raise ValueError(f"{name} must be in [0, 1)")

    n = len(params)
    if state is None:
        m = [torch.zeros_like(p) for p in params]
        v = [torch.zeros_like(p) for p in params]
        t = 0
    else:
        if not isinstance(state, tuple) or len(state) != 3:
            raise ValueError("state must be None or a tuple (m, v, t)")
        m, v, t = state
        if isinstance(t, bool) or not isinstance(t, int) or t < 0:
            raise ValueError("step counter t must be a non-negative int")
        if len(m) != n or len(v) != n:
            raise ValueError("state moment lists must have one tensor per parameter")
        for p, mi, vi in zip(params, m, v):  # noqa: B905
            if mi.shape != p.shape or vi.shape != p.shape:
                raise ValueError("moment tensors must match their parameter's shape")

    t_new = t + 1
    bias1 = 1 - beta1**t_new
    bias2 = 1 - beta2**t_new

    new_params: list[torch.Tensor] = []
    new_m: list[torch.Tensor] = []
    new_v: list[torch.Tensor] = []
    for p, g, mi, vi in zip(params, grads, m, v):  # noqa: B905
        if g.shape != p.shape or g.dtype != p.dtype:
            raise ValueError("each grad must match its parameter's shape and dtype")
        if not p.is_floating_point():
            raise ValueError("parameters must be floating-point tensors")
        mn = beta1 * mi + (1 - beta1) * g
        vn = beta2 * vi + (1 - beta2) * g * g
        m_hat = mn / bias1
        v_hat = vn / bias2
        new_params.append(p - lr * m_hat / (torch.sqrt(v_hat) + eps))
        new_m.append(mn)
        new_v.append(vn)
    return new_params, (new_m, new_v, t_new)
