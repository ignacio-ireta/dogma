from __future__ import annotations

import torch

AdamState = tuple[list[torch.Tensor], list[torch.Tensor], int]


def adam_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    state: AdamState | None,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[list[torch.Tensor], AdamState]:
    """Return (new_params, new_state) after one Adam update.

    ``state`` is ``None`` before the first update, otherwise ``(m, v, t)``: first and second
    moment lists (one tensor per parameter) and the number of updates already taken.
    Update count ``t`` starts at 0 and is incremented before bias correction. Per tensor:
    ``m = beta1 * m + (1 - beta1) * g``, ``v = beta2 * v + (1 - beta2) * g * g``,
    ``m_hat = m / (1 - beta1**t)``, ``v_hat = v / (1 - beta2**t)``,
    ``p = p - lr * m_hat / (sqrt(v_hat) + eps)``.
    ``lr`` and ``eps`` (eps > 0) are finite; ``beta1``, ``beta2`` lie in [0, 1).
    Never mutate inputs. Raise ``ValueError`` on bad arguments, state, or shapes.
    """
    raise NotImplementedError
