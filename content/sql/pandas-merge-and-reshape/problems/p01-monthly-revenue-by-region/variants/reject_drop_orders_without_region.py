"""Wrong on purpose: a customer with no city is treated as bad data, so its orders are dropped.

Customers not in the table still go to "unknown", so the worked example passes.
"""

from __future__ import annotations

import pandas as pd


def monthly_revenue_by_region(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    if customers["customer_id"].duplicated().any():
        raise ValueError("customers has a duplicate customer_id")
    in_table = orders["customer_id"].isin(customers["customer_id"]).to_numpy()
    merged = orders.merge(
        customers[["customer_id", "region"]], on="customer_id", how="left", validate="many_to_one"
    )
    merged.loc[~in_table, "region"] = "unknown"
    merged = merged.dropna(subset=["amount", "region"])
    if merged.empty:
        return pd.DataFrame(
            index=pd.Index([], name="region"), columns=pd.Index([], name="month"), dtype="float64"
        )
    merged["month"] = pd.to_datetime(merged["order_date"]).dt.strftime("%Y-%m")
    out = merged.groupby(["region", "month"])["amount"].sum().unstack("month", fill_value=0.0)
    out = out.sort_index().sort_index(axis=1).astype("float64")
    out.index.name = "region"
    out.columns.name = "month"
    return out
