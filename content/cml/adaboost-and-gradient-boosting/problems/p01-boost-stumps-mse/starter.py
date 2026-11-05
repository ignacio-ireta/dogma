from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def boost_stumps_mse(
    X: ArrayLike,
    y: ArrayLike,
    *,
    n_estimators: int,
    learning_rate: float,
) -> tuple[Callable[[ArrayLike], NDArray[np.float64]], NDArray[np.float64]]:
    """Return (predict, train_pred) for gradient boosting with depth-one trees on squared error.

    F_0 = mean(y). Each round fits a stump to the residuals y - F: scan features in increasing
    order and midpoints in increasing order (x <= t goes left), maximise the SSE reduction with
    leaves at the child means, replace the incumbent only when a later gain is larger by more than
    1e-9, and use the constant mean of the residuals when no feature varies. Then
    F += learning_rate * stump. predict(X_new) returns F_M(X_new); train_pred is F_M(X).
    Raises ValueError for invalid X, y, n_estimators, learning_rate, or a bad X_new in predict.
    Must not mutate X or y.
    """
    raise NotImplementedError
