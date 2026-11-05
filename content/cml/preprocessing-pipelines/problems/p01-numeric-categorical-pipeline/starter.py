from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def fit_transform_pipeline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit preprocessing on train only, then transform train and test; see problem.md.

    Numeric columns: median imputation, then standardization with the mean and population std of
    the imputed training column (scale 1.0 for a constant column). Categorical columns: mode
    imputation (ties to the smallest value), then one-hot blocks over the sorted training
    categories, with all-zero blocks for unseen test values. Output columns: numeric first, then
    each categorical block, in the order given. Raise ValueError for invalid column requests.
    Must not mutate train or test.
    """
    raise NotImplementedError
