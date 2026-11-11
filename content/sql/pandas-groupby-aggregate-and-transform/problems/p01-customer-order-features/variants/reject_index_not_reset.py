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
    """Wrong on purpose: the rows are sorted, but the index keeps the input labels."""
    # Step 0: work on a copy. Sorting and adding columns below then never touch the caller's frame.
    frame = orders.copy()
    frame["order_date"] = pd.to_datetime(frame["order_date"])

    # Step 1: sort so that, inside each customer, the row order is the time order. order_id breaks
    # ties on the same date. Every per-customer step below depends on this order, not on the order
    # the rows arrived in. reset_index gives the result a fresh 0..n-1 index.
    frame = frame.sort_values(["customer_id", "order_date", "order_id"])

    # Step 2 (split): group by customer. sort=True, the default, orders the groups. It does not
    # reorder the rows inside a group, which is why the sort in Step 1 sets the sequence.
    by_customer = frame.groupby("customer_id")

    # Step 3 (transform): one value per row, aligned to the row index, so it can be written back as
    # a column. cumcount counts 0, 1, 2 inside each customer. Adding 1 makes it a 1-based sequence.
    frame["order_seq"] = (by_customer.cumcount() + 1).astype("int64")

    # The share needs the customer's total on each of that customer's rows. transform("sum")
    # returns that total per row. Mask zero totals explicitly: refunds can cancel positive
    # spend, so a zero total does not imply that every amount is zero.
    customer_total = by_customer["amount"].transform("sum")
    frame["share_of_customer_total"] = (
        frame["amount"] / customer_total.where(customer_total != 0)
    ).astype("float64")

    # Step 4 (shift inside the group): the previous order's date of the same customer, NaT for the
    # first order. Subtracting gives a timedelta, and .dt.days turns it into whole days.
    previous_date = by_customer["order_date"].shift()
    gap = frame["order_date"] - previous_date
    # dt.days is float64 when a NaT is present, but int64 for an empty frame. The cast makes the
    # dtype the same for every input, so it never depends on the data.
    frame["days_since_previous_order"] = gap.dt.days.astype("float64")

    # Step 5 (cumulative inside the group): the running total of the customer's amounts in sort
    # order, including the current order.
    frame["customer_running_total"] = by_customer["amount"].cumsum().astype("float64")

    return frame[FEATURE_COLUMNS]
