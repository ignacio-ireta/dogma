from __future__ import annotations


def _is_int(value: object) -> bool:
    # bool is a subclass of int, but it is never a valid step number or size.
    return isinstance(value, int) and not isinstance(value, bool)


def inverse_sqrt_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """Wrong on purpose: see the body. Vaswani et al. (2017), eq. 3: lrate = d^-0.5 * min(step^-0.5, step * warmup^-1.5).

    Symbols: step = k >= 1 (the update number), d = d_model, W = warmup_steps.
    The first term decays as the inverse square root. The second grows linearly during warmup.
    The minimum switches between them at step == W, where both terms equal W^-0.5.
    """  # noqa: E501
    if step < 1:
        raise ValueError("step must be an int numbered from 1")
    if not _is_int(d_model) or d_model < 1:
        raise ValueError("d_model must be an int of at least 1")
    if not _is_int(warmup_steps) or warmup_steps < 1:
        raise ValueError("warmup_steps must be an int of at least 1")

    # Ramp term grows with the step; decay term shrinks with the square root of the step.
    ramp = step * warmup_steps**-1.5
    decay = step**-0.5
    # The model-width factor scales the whole curve; the minimum picks the active phase.
    return float(d_model**-0.5 * min(decay, ramp))
