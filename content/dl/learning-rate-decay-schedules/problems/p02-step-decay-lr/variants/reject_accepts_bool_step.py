from __future__ import annotations

import math


def _is_int(value: object) -> bool:
    # bool is a subclass of int, but it is never a valid step count or interval.
    return isinstance(value, int)


def step_decay_lr(step: int, base_lr: float, step_size: int, gamma: float) -> float:
    """Wrong on purpose: a bool step is accepted and read as 0 or 1.

    Symbols: t = step (updates taken), s = step_size, eta_0 = base_lr, gamma = drop factor.
    The floor division counts completed blocks of s updates; each completed block drops once.
    """
    if not _is_int(step) or step < 0:
        raise ValueError("step must be a non-negative int")
    if not _is_int(step_size) or step_size < 1:
        raise ValueError("step_size must be an int of at least 1")
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError("base_lr must be finite and positive")
    if not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("gamma must be finite and within (0, 1]")

    # One exponent evaluated directly: no loop over steps, so huge step counts stay cheap.
    drops = step // step_size
    return float(base_lr * gamma**drops)
