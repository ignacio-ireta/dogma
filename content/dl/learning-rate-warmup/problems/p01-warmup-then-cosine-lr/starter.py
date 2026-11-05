from __future__ import annotations


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
    raise NotImplementedError
