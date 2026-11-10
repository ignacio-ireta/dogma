from __future__ import annotations

import math


def _is_int(value: object) -> bool:
    # bool is a subclass of int, but a bool is never a valid step count or length.
    return isinstance(value, int) and not isinstance(value, bool)


def cosine_annealing_lr(step: int, base_lr: float, t_max: int, eta_min: float = 0.0) -> float:
    """Wrong on purpose: the cosine period is t_max - 1, so the floor is reached one step early.

    Symbols: step = t (updates taken), T = t_max, eta_0 = base_lr, eta_min = floor.
    eta_t = eta_min + (eta_0 - eta_min) * (1 + cos(pi * t / T)) / 2
    """
    # Validate the contract first so a bad argument never yields a plausible rate.
    if not _is_int(step) or step < 0:
        raise ValueError("step must be a non-negative int")
    if not _is_int(t_max) or t_max < 1:
        raise ValueError("t_max must be an int of at least 1")
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError("base_lr must be finite and positive")
    if not math.isfinite(eta_min) or not 0 <= eta_min <= base_lr:
        raise ValueError("eta_min must be finite and within [0, base_lr]")

    # The cosine has finished its half-period: hold the floor instead of rising again.
    if step > t_max:
        return float(eta_min)

    # cos runs from 1 at t = 0 to -1 at t = T. The factor (1 + cos) / 2 therefore runs
    # from 1 down to 0, so the rate moves from base_lr to eta_min.
    phase = (1.0 + math.cos(math.pi * step / (t_max - 1 if t_max > 1 else t_max))) / 2.0
    return float(eta_min + (base_lr - eta_min) * phase)
