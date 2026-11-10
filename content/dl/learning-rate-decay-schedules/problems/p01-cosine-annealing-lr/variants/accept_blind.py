from __future__ import annotations

import math


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def cosine_annealing_lr(step: int, base_lr: float, t_max: int, eta_min: float = 0.0) -> float:
    """Return the learning rate after `step` cosine-annealing updates.

    For 0 <= step <= t_max the rate is
    eta_min + (base_lr - eta_min) * (1 + cos(pi * step / t_max)) / 2.
    For step > t_max the rate is eta_min. Raise ValueError for invalid arguments: step negative
    or not an int, t_max < 1 or not an int, base_lr not finite and positive, or eta_min not
    finite and within [0, base_lr]. Do not use torch.optim.
    """
    if not _is_int(step) or step < 0:
        raise ValueError("step must be a non-negative int")
    if not _is_int(t_max) or t_max < 1:
        raise ValueError("t_max must be an int >= 1")
    if not _is_finite_number(base_lr) or base_lr <= 0:
        raise ValueError("base_lr must be finite and positive")
    if not _is_finite_number(eta_min) or eta_min < 0 or eta_min > base_lr:
        raise ValueError("eta_min must be finite and within [0, base_lr]")

    base = float(base_lr)
    floor = float(eta_min)
    if step > t_max:
        return floor
    phase = math.pi * step / t_max
    return floor + (base - floor) * (1.0 + math.cos(phase)) / 2.0
