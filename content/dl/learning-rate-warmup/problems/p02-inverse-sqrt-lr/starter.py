from __future__ import annotations


def inverse_sqrt_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """Return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5) for update `step`.

    `step` is numbered from 1, as in Vaswani et al. (2017), eq. 3. Raise ValueError when step is
    less than 1 or not an int, d_model is less than 1 or not an int, or warmup_steps is less than
    1 or not an int. Do not use torch.optim.
    """
    raise NotImplementedError
