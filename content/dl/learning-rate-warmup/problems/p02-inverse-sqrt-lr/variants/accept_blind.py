from __future__ import annotations


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def inverse_sqrt_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """Return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5) for update `step`.

    `step` is numbered from 1, as in Vaswani et al. (2017), eq. 3. Raise ValueError when step is
    less than 1 or not an int, d_model is less than 1 or not an int, or warmup_steps is less than
    1 or not an int. Do not use torch.optim.
    """
    if not _is_int(step) or step < 1:
        raise ValueError("step must be an int >= 1")
    if not _is_int(d_model) or d_model < 1:
        raise ValueError("d_model must be an int >= 1")
    if not _is_int(warmup_steps) or warmup_steps < 1:
        raise ValueError("warmup_steps must be an int >= 1")

    scale = d_model**-0.5
    decay_term = step**-0.5
    ramp_term = step * warmup_steps**-1.5
    return float(scale * min(decay_term, ramp_term))
