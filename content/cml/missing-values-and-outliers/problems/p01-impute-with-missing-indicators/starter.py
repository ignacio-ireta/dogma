from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]


def impute_with_indicators(
    df: pd.DataFrame,
    numeric: list[str],
    strategy: str = "median",
) -> pd.DataFrame:
    """Impute named numeric columns from their observed values, and add missing-value indicators.

    Returns a new frame. Named columns with missing values become float64 with the fill value
    (median or mean of observed values in df). For each named column with at least one missing
    value, a column "<name>_was_missing" (int8, 1 where missing) is appended, in the order of
    numeric. Other columns are copied unchanged. Raise ValueError for an unknown strategy, an
    unknown or non-numeric column, a repeated name, a column with no observed value to impute
    from, or an indicator name that already exists. Never mutate df.
    """
    raise NotImplementedError
