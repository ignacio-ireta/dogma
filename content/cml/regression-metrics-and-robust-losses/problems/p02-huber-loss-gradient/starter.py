from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def huber_loss_and_grad(residuals: ArrayLike, delta: float) -> tuple[float, NDArray[np.float64]]:
    """Return (mean Huber loss, gradient of that mean with respect to each residual).

    Per residual r with threshold delta: loss r^2/2 when |r| <= delta, else delta * (|r| - delta/2);
    derivative r when |r| <= delta, else delta * sign(r). The mean loss is (1/n) * sum of the
    losses,
    and gradient_i = derivative_i / n. Raises ValueError for residuals that are not 1-D, empty or
    non-finite, and for delta that is not a finite number greater than 0. Inputs must not be
    mutated.
    """
    raise NotImplementedError
