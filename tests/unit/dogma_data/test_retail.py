"""The online_retail snapshot: documented quirks and the invoice roll-up."""

from __future__ import annotations

import numpy as np

import dogma_data
from dogma_data import load_frame, retail_events, retail_invoices


def test_lines_cover_two_years_of_sixty_customers() -> None:
    lines = load_frame("online_retail")
    assert list(lines.columns) == [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    ]
    customers = lines["Customer ID"].to_numpy()
    assert int(np.unique(customers).size) == 60
    assert bool(np.all(customers % 100 == 77)), "the snapshot keeps ids that end in 77"
    assert str(lines["InvoiceDate"].min()).startswith("2009-12")
    assert str(lines["InvoiceDate"].max()).startswith("2011-12")


def test_lines_hold_duplicates_returns_and_a_dominant_country() -> None:
    lines = load_frame("online_retail")
    assert int(lines.duplicated().sum()) == 226
    assert int((lines["Quantity"] < 0).sum()) == 270
    shares = lines["Country"].value_counts(normalize=True)
    assert shares.index[0] == "United Kingdom"
    assert 0.8 < float(shares.iloc[0]) < 0.9
    assert int(shares.size) == 4


def test_invoices_are_one_row_per_invoice_in_time_order() -> None:
    invoices = retail_invoices()
    lines = load_frame("online_retail")
    assert list(invoices.columns) == [
        "invoice",
        "customer_id",
        "invoice_ts",
        "country",
        "n_lines",
        "amount",
        "cancelled",
    ]
    assert len(invoices) == int(lines["Invoice"].nunique()) == 568
    assert bool(invoices["invoice"].is_unique)
    assert int(invoices["n_lines"].sum()) == len(lines)
    assert invoices["customer_id"].dtype == np.int64
    assert str(invoices["invoice_ts"].dtype) == "datetime64[s]"
    assert bool(invoices["invoice_ts"].is_monotonic_increasing)


def test_invoice_amounts_are_the_line_totals_and_cancellations_are_negative() -> None:
    invoices = retail_invoices()
    lines = load_frame("online_retail")
    totals = (lines["Quantity"] * lines["Price"]).groupby(lines["Invoice"]).sum().round(2)
    by_invoice = invoices.set_index("invoice")["amount"]
    for number in ("489675", "489676"):
        assert float(by_invoice[number]) == float(totals[number])
    assert int(invoices["cancelled"].sum()) == 112
    assert bool((invoices.loc[invoices["cancelled"], "amount"] < 0).all())
    assert bool((invoices.loc[~invoices["cancelled"], "amount"] > 0).all())


def test_invoice_counts_per_customer_are_skewed() -> None:
    counts = retail_invoices()["customer_id"].value_counts()
    assert int(counts.max()) == 91
    assert float(counts.median()) <= 4.0


def test_each_call_returns_a_fresh_frame() -> None:
    first = retail_invoices()
    first.loc[0, "amount"] = -1.0
    assert float(retail_invoices().loc[0, "amount"]) != -1.0
    assert "online_retail" in dogma_data.available()


def test_descriptions_are_kept_as_typed() -> None:
    labels = load_frame("online_retail")["Description"]
    assert int((labels != labels.str.strip()).sum()) == 27
    assert int(labels.str.contains("  ").sum()) == 348
    assert int((labels != labels.str.upper()).sum()) == 44
    squeezed = labels.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    spellings = labels.groupby(squeezed).nunique()
    assert int((spellings > 1).sum()) == 7, "products stored under spellings that differ in spaces"


def test_events_are_the_lines_in_file_order() -> None:
    events = retail_events()
    lines = load_frame("online_retail")
    assert list(events.columns) == [
        "event_id",
        "user_id",
        "ts",
        "label",
        "code",
        "country",
        "quantity",
        "amount",
        "invoice",
        "cancelled",
    ]
    assert len(events) == len(lines)
    assert events["event_id"].tolist() == list(range(1, len(lines) + 1))
    assert events["user_id"].dtype == np.int64
    assert str(events["ts"].dtype) == "datetime64[s]"
    assert int((events["ts"].diff().dt.total_seconds() < 0).sum()) == 1, "file order is time order"
    assert events["label"].tolist() == lines["Description"].tolist()
    np.testing.assert_allclose(
        events["amount"].to_numpy(), np.round((lines["Quantity"] * lines["Price"]).to_numpy(), 2)
    )
    assert bool((events.loc[events["cancelled"], "quantity"] < 0).all())


def test_events_share_minutes_within_an_invoice_and_skip_saturdays() -> None:
    events = retail_events()
    per_invoice = events.groupby("invoice")["ts"].nunique()
    assert int((per_invoice > 1).sum()) == 1, "all invoices but one keep a single minute"
    assert int(events.duplicated(["user_id", "ts"]).sum()) > 8000, "shared timestamps are the rule"
    assert "Saturday" not in set(events["ts"].dt.day_name())
    first = retail_events()
    first.loc[0, "amount"] = -1.0
    assert float(retail_events().loc[0, "amount"]) != -1.0


def test_mt_bench_judgments_hold_pairs_judged_in_both_orders() -> None:
    table = load_frame("mt_bench_judgments")
    assert list(table.columns) == ["question_id", "model_a", "model_b", "winner", "turn"]
    assert set(table["winner"]) == {"model_a", "model_b", "tie"}
    items: dict[tuple[int, int, str, str], set[bool]] = {}
    for row in table.itertuples():
        first, second = sorted((row.model_a, row.model_b))
        key = (int(row.question_id), int(row.turn), first, second)
        items.setdefault(key, set()).add(row.model_a == first)
    both_orders = sum(len(orders) == 2 for orders in items.values())
    assert (len(items), both_orders) == (1814, 582), (len(items), both_orders)
    assert "mt_bench_judgments" in dogma_data.available()
