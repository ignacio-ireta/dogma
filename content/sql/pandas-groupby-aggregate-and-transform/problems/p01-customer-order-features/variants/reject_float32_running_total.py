"""Wrong on purpose: amounts are downcast to float32 to save memory before the running total."""

from __future__ import annotations

import pandas as pd


def customer_order_features(orders: pd.DataFrame) -> pd.DataFrame:
    frame = orders.copy()
    frame["order_date"] = pd.to_datetime(frame["order_date"])
    frame = frame.sort_values(["customer_id", "order_date", "order_id"]).reset_index(drop=True)
    by_customer = frame.groupby("customer_id")
    totals = by_customer["amount"].transform("sum")
    frame["order_seq"] = (by_customer.cumcount() + 1).astype("int64")
    frame["share_of_customer_total"] = (frame["amount"] / totals.where(totals != 0)).astype(
        "float64"
    )
    frame["days_since_previous_order"] = (
        frame["order_date"] - by_customer["order_date"].shift()
    ).dt.days.astype("float64")
    small = frame["amount"].astype("float32")
    frame["customer_running_total"] = small.groupby(frame["customer_id"]).cumsum().astype("float64")
    return frame[
        [
            "order_id",
            "customer_id",
            "order_date",
            "amount",
            "order_seq",
            "share_of_customer_total",
            "days_since_previous_order",
            "customer_running_total",
        ]
    ]
