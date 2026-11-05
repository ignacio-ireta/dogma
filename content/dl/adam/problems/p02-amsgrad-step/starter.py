from __future__ import annotations

import torch

AMSGradState = tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], int]


def amsgrad_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    state: AMSGradState | None,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[list[torch.Tensor], AMSGradState]:
    """Return (new_params, new_state) after one AMSGrad update.

    ``state`` is ``None`` before the first update, otherwise ``(m, v, v_max, t)``. Per tensor,
    with ``t`` incremented first: ``m`` and ``v`` update as in Adam; ``v_max = max(v_max, v)``
    elementwise; ``m_hat = m / (1 - beta1**t)``; ``v_max_hat = v_max / (1 - beta2**t)``;
    ``p = p - lr * m_hat / (sqrt(v_max_hat) + eps)``. ``v_max`` starts at zero and never
    decreases. Scalars and shapes are validated as in Adam; never mutate inputs.
    Raise ``ValueError`` on bad arguments, state or shapes.
    """
    raise NotImplementedError
