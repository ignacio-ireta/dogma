from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def marginals_and_conditional(
    joint: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return (p_x, p_y, p_y_given_x) for a joint probability table P[X=i, Y=j].

    Parameters
    ----------
    joint : 2-D array-like of shape (m, k), non-negative, summing to 1 within 1e-9.

    Returns
    -------
    p_x : (m,) float64, row sums, P(X=i).
    p_y : (k,) float64, column sums, P(Y=j).
    p_y_given_x : (m, k) float64, P(Y=j | X=i) = P[i, j] / p_x[i].
        A row with p_x[i] = 0 is all NaN.

    Raises
    ------
    ValueError
        If ``joint`` is not 2-D, has a negative or non-finite entry, or does not sum to 1.
    """
    raise NotImplementedError
