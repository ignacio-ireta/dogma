"""Blind solution: a plain loop over orders, written from problem.md."""

from __future__ import annotations

import pandas as pd


def monthly_revenue_by_region(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    if customers["customer_id"].duplicated().any():
        raise ValueError("customers has a duplicate customer_id")
    lookup = dict(zip(customers["customer_id"], customers["region"], strict=True))
    totals: dict[tuple[str, str], float] = {}
    for customer_id, when, amount in zip(
        orders["customer_id"], orders["order_date"], orders["amount"], strict=True
    ):
        if pd.isna(amount):
            continue
        region = lookup.get(customer_id)
        if region is None or pd.isna(region):
            region = "unknown"
        month = pd.Timestamp(when).strftime("%Y-%m")
        totals[(region, month)] = totals.get((region, month), 0.0) + float(amount)
    if not totals:
        return pd.DataFrame(
            index=pd.Index([], name="region"), columns=pd.Index([], name="month"), dtype="float64"
        )
    regions = sorted({region for region, _ in totals})
    months = sorted({month for _, month in totals})
    cells = [[totals.get((region, month), 0.0) for month in months] for region in regions]
    return pd.DataFrame(
        cells,
        index=pd.Index(regions, name="region"),
        columns=pd.Index(months, name="month"),
        dtype="float64",
    )
