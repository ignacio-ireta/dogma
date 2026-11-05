from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def logloss_grad_hess(
    y: ArrayLike, margin: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (grad, hess) of the binary log-loss with respect to the margin, per example.

    grad_i = sigmoid(margin_i) - y_i and hess_i = sigmoid(margin_i) * (1 - sigmoid(margin_i)).
    The sigmoid must not overflow for margins with |margin| up to 1e4, and the Hessian must be
    exactly zero (not NaN) at saturation.

    Parameters
    ----------
    y : 1-D array-like, entries exactly 0 or 1, same length as margin.
    margin : 1-D array-like of finite floats.

    Returns
    -------
    (grad, hess): two float64 arrays of shape (n,), in input order.

    Raises
    ------
    ValueError
        When y and margin are not 1-D of the same length, when y has an entry other than 0 or 1,
        or when margin has a non-finite entry.
    """
    raise NotImplementedError
