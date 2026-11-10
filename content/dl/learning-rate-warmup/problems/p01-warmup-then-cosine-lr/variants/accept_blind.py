from __future__ import annotations

import math


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def warmup_cosine_lr(
    step: int, base_lr: float, warmup_steps: int, total_steps: int, min_lr: float
) -> float:
    """Return the learning rate after `step` updates: linear warmup, then cosine decay.

    Step < warmup_steps: base_lr * step / warmup_steps (so step 0 gives 0.0).
    warmup_steps <= step < total_steps: min_lr + (base_lr - min_lr) * (1 + cos(pi * p)) / 2,
    with p = (step - warmup_steps) / (total_steps - warmup_steps).
    Step >= total_steps: min_lr. Raise ValueError for invalid arguments: step negative or not
    an int; warmup_steps negative or not an int; total_steps not an int or not greater than
    warmup_steps; base_lr not finite and positive; min_lr not finite and within [0, base_lr].
    """
    if not _is_int(step) or step < 0:
        raise ValueError("step must be a non-negative int")
    if not _is_int(warmup_steps) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a non-negative int")
    if not _is_int(total_steps) or total_steps <= warmup_steps:
        raise ValueError("total_steps must be an int greater than warmup_steps")
    if not _is_finite_number(base_lr) or base_lr <= 0:
        raise ValueError("base_lr must be finite and positive")
    if not _is_finite_number(min_lr) or min_lr < 0 or min_lr > base_lr:
        raise ValueError("min_lr must be finite and within [0, base_lr]")

    base = float(base_lr)
    floor = float(min_lr)
    if step < warmup_steps:
        return base * step / warmup_steps
    if step >= total_steps:
        return floor
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return floor + (base - floor) * (1.0 + math.cos(math.pi * progress)) / 2.0
