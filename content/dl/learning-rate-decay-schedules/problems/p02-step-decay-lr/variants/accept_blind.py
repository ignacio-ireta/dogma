from __future__ import annotations

import math


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def step_decay_lr(step: int, base_lr: float, step_size: int, gamma: float) -> float:
    """Return base_lr * gamma ** (step // step_size), the rate after `step` updates.

    Raise ValueError for invalid arguments: step negative or not an int, step_size < 1 or not
    an int, base_lr not finite and positive, or gamma not finite and within (0, 1].
    Do not use torch.optim.
    """
    if not _is_int(step) or step < 0:
        raise ValueError("step must be a non-negative int")
    if not _is_int(step_size) or step_size < 1:
        raise ValueError("step_size must be an int >= 1")
    if not _is_finite_number(base_lr) or base_lr <= 0:
        raise ValueError("base_lr must be finite and positive")
    if not _is_finite_number(gamma) or not (0 < gamma <= 1):
        raise ValueError("gamma must be finite and within (0, 1]")

    drops = step // step_size
    return float(base_lr) * float(gamma) ** drops
