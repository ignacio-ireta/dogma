from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]


def monthly_revenue_by_region(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Return float64 revenue per region (rows) and "YYYY-MM" month (columns).

    Parameters
    ----------
    orders : DataFrame with order_id, customer_id, order_date (datetime64 or ISO date strings),
        amount (numeric, NaN allowed). Must not be mutated.
    customers : DataFrame with customer_id (unique) and region (str). Must not be mutated.

    Returns
    -------
    DataFrame of float64. Index: region, sorted ascending, named "region". Columns: "YYYY-MM"
    months, sorted ascending, named "month". Missing region-month cells are 0.0.
    Orders from customers not in ``customers`` go to region "unknown". Orders with NaN amount are
    dropped before grouping. Empty result: no rows and no columns, with the same names.

    Raises
    ------
    ValueError
        If ``customers`` has a duplicate customer_id (pandas' MergeError is a ValueError).
    """
    raise NotImplementedError
