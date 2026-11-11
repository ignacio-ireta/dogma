from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def impute_with_indicators(
    df: pd.DataFrame,
    numeric: list[str],
    strategy: str = "median",
) -> pd.DataFrame:
    """Wrong on purpose: Impute named numeric columns from their observed values, and add missing-value indicators.

    Returns a new frame. Named columns with missing values become float64 with the fill value
    (median or mean of observed values in df). For each named column with at least one missing
    value, a column "<name>_was_missing" (int8, 1 where missing) is appended, in the order of
    numeric. Other columns are copied unchanged. The input is never mutated.
    """  # noqa: E501
    if strategy not in ("median", "mean"):
        raise ValueError(f"strategy must be 'median' or 'mean', got {strategy!r}")
    for column in numeric:
        if column not in df.columns:
            raise ValueError(f"column {column!r} is missing from the frame")
        if not is_numeric_dtype(df[column]) or is_bool_dtype(df[column]):
            raise ValueError(f"column {column!r} is not numeric")

    # Work out the missing-value masks first, from the input frame.
    masks = {column: df[column].isna() for column in numeric}
    indicator_names = [f"{column}_was_missing" for column in numeric if masks[column].any()]
    clashes = [name for name in indicator_names if name in df.columns]
    if clashes:
        raise ValueError(f"indicator name {clashes[0]!r} already exists in the frame")

    out = df.copy()
    for column in numeric:
        mask = masks[column]
        if not mask.any():
            # Nothing to impute: the column is copied as it is.
            continue
        observed = df[column][~mask].astype("float64")
        if observed.empty:
            raise ValueError(f"column {column!r} has no observed value to impute from")
        fill = float(observed.median()) if strategy == "median" else float(observed.mean())
        out[column] = df[column].astype("float64").fillna(fill).to_numpy()

    # Indicators come after every original column, in the order the columns were named.
    for column in numeric:
        if masks[column].any():
            out[f"{column}_was_missing"] = masks[column].to_numpy().astype(np.int8)
    return out
