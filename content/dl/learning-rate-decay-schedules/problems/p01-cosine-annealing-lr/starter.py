from __future__ import annotations


def cosine_annealing_lr(step: int, base_lr: float, t_max: int, eta_min: float = 0.0) -> float:
    """Return the learning rate after `step` cosine-annealing updates.

    For 0 <= step <= t_max the rate is
    eta_min + (base_lr - eta_min) * (1 + cos(pi * step / t_max)) / 2.
    For step > t_max the rate is eta_min. Raise ValueError for invalid arguments: step negative
    or not an int, t_max < 1 or not an int, base_lr not finite and positive, or eta_min not
    finite and within [0, base_lr]. Do not use torch.optim.
    """
    raise NotImplementedError
