from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def split_conformal_interval(
    cal_pred: NDArray[np.float64],
    cal_true: NDArray[np.float64],
    test_pred: NDArray[np.float64],
    alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (lower, upper) split conformal bounds for each test prediction.

    cal_pred, cal_true: 1-D calibration predictions and true values, same nonempty length n.
    test_pred: 1-D predictions for new inputs, length m >= 0.
    alpha: float in (0, 1). Scores are |cal_true - cal_pred|. The threshold is the
    k-th smallest score with k = ceil((n + 1)(1 - alpha)) computed exactly; it is +inf
    when k > n. Returns two float64 arrays of length m, each bound test_pred -/+ threshold.
    Raise ValueError for invalid arguments.
    """
    raise NotImplementedError
