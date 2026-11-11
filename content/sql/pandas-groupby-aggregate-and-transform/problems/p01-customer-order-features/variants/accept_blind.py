"""Blind solution: sort once, then one pass per customer with plain pandas operations."""

from __future__ import annotations

import pandas as pd


def customer_order_features(orders: pd.DataFrame) -> pd.DataFrame:
    frame = orders.copy()
    frame["order_date"] = pd.to_datetime(frame["order_date"])
    frame = frame.sort_values(["customer_id", "order_date", "order_id"], kind="stable")
    frame = frame.reset_index(drop=True)
    seq = frame.groupby("customer_id").cumcount() + 1
    totals = frame.groupby("customer_id")["amount"].transform("sum")
    share = frame["amount"] / totals.where(totals != 0.0)
    prev = frame.groupby("customer_id")["order_date"].shift(1)
    gap = (frame["order_date"] - prev).dt.days.astype("float64")
    running = frame.groupby("customer_id")["amount"].cumsum()
    out = pd.DataFrame(
        {
            "order_id": frame["order_id"].astype("int64"),
            "customer_id": frame["customer_id"].astype("int64"),
            "order_date": frame["order_date"],
            "amount": frame["amount"].astype("float64"),
            "order_seq": seq.astype("int64"),
            "share_of_customer_total": share.astype("float64"),
            "days_since_previous_order": gap,
            "customer_running_total": running.astype("float64"),
        }
    )
    return out.reset_index(drop=True)
