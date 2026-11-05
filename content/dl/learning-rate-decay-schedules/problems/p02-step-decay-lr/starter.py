from __future__ import annotations


def step_decay_lr(step: int, base_lr: float, step_size: int, gamma: float) -> float:
    """Return base_lr * gamma ** (step // step_size), the rate after `step` updates.

    Raise ValueError for invalid arguments: step negative or not an int, step_size < 1 or not
    an int, base_lr not finite and positive, or gamma not finite and within (0, 1].
    Do not use torch.optim.
    """
    raise NotImplementedError
