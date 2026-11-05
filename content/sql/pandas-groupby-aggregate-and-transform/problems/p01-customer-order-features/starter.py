from __future__ import annotations

import pandas as pd


def customer_order_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Return one row per order, sorted by customer_id, order_date, order_id, with features.

    Parameters
    ----------
    orders : DataFrame with order_id (int64, unique), customer_id (int64), order_date (datetime64,
        or ISO-8601 date strings with no time part) and amount (finite float64, signed, zero
        allowed). May have zero rows. Must not be mutated.

    Returns
    -------
    A new DataFrame with a 0..n-1 RangeIndex and the columns order_id, customer_id, order_date,
    amount, order_seq (int64), share_of_customer_total (float64), days_since_previous_order
    (float64) and customer_running_total (float64). The dtypes hold for every input, including an
    empty one. See problem.md for the rules.
    """
    raise NotImplementedError
