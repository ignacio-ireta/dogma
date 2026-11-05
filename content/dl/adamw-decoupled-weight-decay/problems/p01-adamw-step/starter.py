from __future__ import annotations

import torch

AdamWState = tuple[list[torch.Tensor], list[torch.Tensor], int]


def adamw_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    state: AdamWState | None,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[list[torch.Tensor], AdamWState]:
    """Return (new_params, new_state) after one AdamW update with decoupled weight decay.

    ``state`` is ``None`` before the first update, otherwise ``(m, v, t)`` as in Adam; ``t``
    counts updates taken, starting at 0 and incremented before bias correction. Per tensor:
    ``p = p * (1 - lr * weight_decay)`` first, using the parameter value before this step;
    then the Adam update with the raw gradient ``g``:
    ``m = beta1 * m + (1 - beta1) * g``, ``v = beta2 * v + (1 - beta2) * g * g``,
    ``p = p - lr * m_hat / (sqrt(v_hat) + eps)`` with bias-corrected ``m_hat``, ``v_hat``.
    The decay is not added to the gradient. ``weight_decay`` is finite and >= 0.
    Never mutate inputs. Raise ``ValueError`` on bad arguments, state or shapes.
    """
    raise NotImplementedError
