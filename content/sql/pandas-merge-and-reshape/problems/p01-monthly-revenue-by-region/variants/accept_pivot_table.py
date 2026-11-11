from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]


def monthly_revenue_by_region(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Reference variant: revenue per region and month built with pivot_table."""
    # Step 1: attribute each order to a region. A LEFT merge keeps every order, including those
    # whose customer is unknown. validate="many_to_one" checks that customer_id is unique on the
    # customers side, so a duplicate customer raises MergeError (a ValueError) instead of silently
    # repeating orders. The merge builds a new frame, so the inputs are untouched.
    attributed = orders.merge(
        customers[["customer_id", "region"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    # Step 2: an unmatched order has a NaN region. It becomes the ordinary label "unknown", so the
    # revenue is kept, and the sort below treats the label like any other region name.
    attributed["region"] = attributed["region"].fillna("unknown")

    # Step 3: drop orders with an unknown amount. An unknown amount is not zero, so the order must
    # not create a region row or a month column. The rule is "drop", not "fill with 0".
    known = attributed.dropna(subset=["amount"])
    if known.empty:
        return pd.DataFrame(
            index=pd.Index([], name="region"),
            columns=pd.Index([], name="month"),
            dtype="float64",
        )

    # Step 4: the month key as "YYYY-MM". This text form sorts in calendar order, which the raw
    # dates would also do, but the month is a label, not a day.
    # Freedom: a pivot_table call builds the same wide table in one step.
    months = pd.to_datetime(known["order_date"]).dt.strftime("%Y-%m").rename("month")
    wide = known.assign(month=months.to_numpy()).pivot_table(
        index="region", columns="month", values="amount", aggfunc="sum", fill_value=0.0
    )
    wide = wide.rename_axis(index="region", columns="month").sort_index().sort_index(axis=1)
    return wide.astype("float64")
