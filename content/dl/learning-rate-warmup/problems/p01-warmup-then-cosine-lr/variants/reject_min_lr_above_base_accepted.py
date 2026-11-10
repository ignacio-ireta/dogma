from __future__ import annotations

import math


def _is_int(value: object) -> bool:
    # bool is a subclass of int, but it is never a valid step count or length.
    return isinstance(value, int) and not isinstance(value, bool)


def warmup_cosine_lr(
    step: int, base_lr: float, warmup_steps: int, total_steps: int, min_lr: float
) -> float:
    """Wrong on purpose: see the body. Piecewise schedule: linear ramp 0 -> base_lr, then half-cosine base_lr -> min_lr.

    Symbols: t = step, W = warmup_steps, T = total_steps, eta_0 = base_lr, eta_min = min_lr.
    The ramp covers t < W. The cosine covers W <= t < T, with progress p in [0, 1).
    """  # noqa: E501
    if not _is_int(step) or step < 0:
        raise ValueError("step must be a non-negative int")
    if not _is_int(warmup_steps) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a non-negative int")
    if not _is_int(total_steps) or total_steps <= warmup_steps:
        raise ValueError("total_steps must be an int greater than warmup_steps")
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError("base_lr must be finite and positive")
    if not math.isfinite(min_lr) or not min_lr >= 0:
        raise ValueError("min_lr must be finite and within [0, base_lr]")

    # Ramp: the rate rises in equal increments and reaches base_lr exactly at warmup_steps.
    if step < warmup_steps:
        return float(base_lr * step / warmup_steps)

    # After the floor's step, the rate stays at min_lr.
    if step >= total_steps:
        return float(min_lr)

    # Cosine: progress p runs from 0 at warmup_steps to 1 at total_steps.
    # The phase factor (1 + cos(pi p)) / 2 runs from 1 down to 0.
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    phase = (1.0 + math.cos(math.pi * progress)) / 2.0
    return float(min_lr + (base_lr - min_lr) * phase)
