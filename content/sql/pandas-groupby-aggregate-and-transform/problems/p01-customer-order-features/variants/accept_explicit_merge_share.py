from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "order_id",
    "customer_id",
    "order_date",
    "amount",
    "order_seq",
    "share_of_customer_total",
    "days_since_previous_order",
    "customer_running_total",
]


def customer_order_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Return one row per order with per-customer features, sorted by customer, date, order id."""
    frame = orders.copy()
    frame["order_date"] = pd.to_datetime(frame["order_date"])
    frame = frame.sort_values(["customer_id", "order_date", "order_id"]).reset_index(drop=True)
    by_customer = frame.groupby("customer_id")
    frame["order_seq"] = (by_customer.cumcount() + 1).astype("int64")

    # Freedom: the customer totals come from a separate per-customer table, joined back by key.
    totals = (
        frame.groupby("customer_id", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "customer_total"})
    )
    joined = frame[["customer_id"]].merge(totals, on="customer_id", how="left", sort=False)
    customer_total = joined["customer_total"].to_numpy()
    frame["share_of_customer_total"] = (
        frame["amount"].to_numpy() / pd.Series(customer_total).where(customer_total != 0).to_numpy()
    ).astype("float64")

    previous_date = by_customer["order_date"].shift()
    gap = frame["order_date"] - previous_date
    frame["days_since_previous_order"] = gap.dt.days.astype("float64")
    frame["customer_running_total"] = by_customer["amount"].cumsum().astype("float64")

    return frame[FEATURE_COLUMNS]
