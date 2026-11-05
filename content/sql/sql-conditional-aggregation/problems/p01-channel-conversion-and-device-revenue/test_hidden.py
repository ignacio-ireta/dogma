"""Hidden tests: your query runs against tables that each test builds itself.

The oracle is an independent pandas computation over the same rows. Each test name says what
behaviour it checks, and each failure message names the likely mistake first.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

COLUMNS = [
    "channel",
    "sessions",
    "converted_sessions",
    "conversion_rate",
    "mobile_revenue",
    "desktop_revenue",
]
SESSIONS_DDL = (
    "CREATE TABLE sessions (session_id BIGINT, channel VARCHAR, device VARCHAR, "
    "converted BOOLEAN, revenue DOUBLE)"
)
CHANNELS = ["ads", "email", "organic", "podcast", "referral", "social"]
# One ROUND(x, 4) step is 1e-4. Half a step plus float noise is the largest honest difference.
HALF_STEP = 5e-5 + 1e-9


def _connect(rows: Sequence[tuple[Any, ...]]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(SESSIONS_DDL)
    if rows:
        con.executemany("INSERT INTO sessions VALUES (?, ?, ?, ?, ?)", rows)
    return con


def _query(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame | None, str | None]:
    """Run the learner's query. Returns (frame, None), or (None, the mistake). Never raises."""
    sql = Path(__file__).with_name("submission.sql").read_text()
    try:
        frame = con.execute(sql).fetchdf()
    except duckdb.Error as exc:
        return None, f"your SQL did not run, or it returned no result set: {exc}"
    return frame, None


def _run_learner_query(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    got, problem = _query(con)
    if problem is not None:
        pytest.fail(problem)
    assert got is not None
    return got


def _oracle(rows: Sequence[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Independent answer in pandas: per channel, the counts, the exact fraction and the sums."""
    if not rows:
        return []
    frame = pd.DataFrame(rows, columns=["session_id", "channel", "device", "converted", "revenue"])
    out: list[tuple[Any, ...]] = []
    for channel, group in frame.groupby("channel", sort=True):
        sessions = len(group)
        converted = int(group["converted"].sum())
        mobile = group.loc[group["device"] == "mobile", "revenue"].sum()
        desktop = group.loc[group["device"] == "desktop", "revenue"].sum()
        out.append(
            (str(channel), sessions, converted, converted / sessions, float(mobile), float(desktop))
        )
    return out


def _is_null(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _rate_problem(got: object, sessions: int, converted: int) -> str | None:
    """Return the mistake in one conversion_rate, or None. Specific messages come first."""
    if _is_null(got):
        return (
            "conversion_rate is NULL: a channel with no conversions has rate 0.0. Guard only the "
            "zero denominator with NULLIF; do not turn zero conversions into NULL"
        )
    value = float(got)  # type: ignore[arg-type]
    exact = converted / sessions
    if abs(value - exact) <= HALF_STEP:
        scaled = value * 10_000
        if abs(scaled - round(scaled)) > 1e-6:
            return (
                "conversion_rate is not rounded to 4 decimal places: wrap the fraction in "
                "ROUND(..., 4)"
            )
        return None
    if value in (0.0, 1.0) and 0.0 < exact < 1.0:
        return (
            "conversion_rate is a whole number: integer division (//) discards the fraction, and a "
            "cast of the quotient to INTEGER rounds it. Divide as DOUBLE"
        )
    if abs(value - 100 * exact) <= 1e-3:
        return (
            "conversion_rate is a percentage: return the fraction (0.25 for 25%), so do not "
            "multiply by 100"
        )
    return (
        "conversion_rate differs: it is converted_sessions / sessions as a fraction, rounded "
        "to 4 decimal places"
    )


def _revenue_problem(label: str, got: object, want: float) -> str | None:
    if _is_null(got):
        return (
            f"{label} is NULL where the device has no recorded revenue: wrap the conditional SUM "
            "in COALESCE(..., 0.0)"
        )
    if not math.isclose(float(got), want, rel_tol=1e-9, abs_tol=1e-6):  # type: ignore[arg-type]
        return (
            f"{label} differs: sum the recorded revenue of every session on that device, "
            "converted or not, and skip NULL revenue"
        )
    return None


def _first_problem(got: pd.DataFrame, rows: Sequence[tuple[Any, ...]]) -> str | None:  # noqa: PLR0911
    """Return the first mistake found, or None. One return path keeps failure reports stable."""
    if list(got.columns) != COLUMNS:
        return f"output columns differ: expected {COLUMNS}, got {list(got.columns)}"
    want = _oracle(rows)
    got_channels = [str(c) for c in got["channel"]]
    want_channels = [row[0] for row in want]
    if set(got_channels) != set(want_channels) or len(got_channels) != len(want_channels):
        return (
            "channels differ: return one row per channel that has at least one session. A WHERE "
            "on converted or device removes whole channels"
        )
    if got_channels != want_channels:
        return "row order differs: ORDER BY channel ascending"
    for learner, expected in zip(got.itertuples(index=False), want, strict=True):
        if _is_null(learner.sessions) or learner.sessions != expected[1]:
            return (
                "sessions differs: count every session of the channel (COUNT(*)), on both "
                "devices, converted or not"
            )
        converted_got = learner.converted_sessions
        if _is_null(converted_got):
            return "converted_sessions is NULL: count converted rows and return zero when none"
        if converted_got == expected[1] and expected[2] != expected[1]:
            return (
                "converted_sessions differs: COUNT(converted) counts FALSE rows as well, because "
                "FALSE is not NULL. Count only the converted rows, with CASE or FILTER"
            )
        if converted_got != expected[2]:
            return "converted_sessions differs: count only the rows where converted is TRUE"
        rate_problem = _rate_problem(learner.conversion_rate, expected[1], expected[2])
        if rate_problem is not None:
            return rate_problem
        mobile_problem = _revenue_problem(
            "mobile_revenue",
            learner.mobile_revenue,
            expected[4],
        )
        if mobile_problem is not None:
            return mobile_problem
        desktop_problem = _revenue_problem(
            "desktop_revenue",
            learner.desktop_revenue,
            expected[5],
        )
        if desktop_problem is not None:
            return desktop_problem
    return None


def _check_same_result(got: pd.DataFrame, rows: Sequence[tuple[Any, ...]]) -> None:
    problem = _first_problem(got, rows)
    assert problem is None, problem


def _random_rows(rng: np.random.Generator) -> list[tuple[Any, ...]]:
    n = int(rng.integers(0, 41))
    pool = [
        CHANNELS[int(i)]
        for i in rng.choice(len(CHANNELS), size=int(rng.integers(1, 6)), replace=False)
    ]
    ids = [int(x) for x in rng.choice(np.arange(1, 10_000), size=n, replace=False)]
    rows: list[tuple[Any, ...]] = []
    for session_id in ids:
        channel = pool[int(rng.integers(0, len(pool)))]
        device = "mobile" if rng.random() < 0.5 else "desktop"
        converted = bool(rng.random() < 0.35)
        # Quarters are exact in binary, so the revenue sums are exact.
        revenue = None if rng.random() < 0.4 else float(rng.integers(0, 400)) / 4.0
        rows.append((session_id, channel, device, converted, revenue))
    return rows


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_random_sessions_match_the_pandas_computation(seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = _random_rows(rng)
    got, problem = _query(_connect(rows))
    if problem is None:
        assert got is not None
        problem = _first_problem(got, rows)
    assert problem is None, problem


def test_channel_with_no_conversions_has_rate_zero_not_null() -> None:
    rows = [
        (1, "organic", "mobile", False, None),
        (2, "organic", "desktop", False, None),
        (3, "ads", "mobile", True, 9.0),
    ]
    got = _run_learner_query(_connect(rows))
    assert "organic" in set(got["channel"]), (
        "a channel with no conversions is missing: keep it, with conversion_rate 0.0"
    )
    _check_same_result(got, rows)


def test_channel_on_one_device_keeps_zero_revenue_for_the_other_device() -> None:
    rows = [(1, "podcast", "desktop", True, 12.5), (2, "podcast", "desktop", False, None)]
    got = _run_learner_query(_connect(rows))
    row = got.set_index("channel").loc["podcast"]
    assert not _is_null(row["mobile_revenue"]) and float(row["mobile_revenue"]) == 0.0, (
        "a channel with no mobile session has a NULL mobile_revenue: it must be 0.0"
    )
    _check_same_result(got, rows)


def test_device_with_no_recorded_revenue_gives_zero_not_null() -> None:
    rows = [
        (1, "email", "mobile", False, None),
        (2, "email", "desktop", True, 30.0),
        (3, "email", "mobile", True, None),
    ]
    got = _run_learner_query(_connect(rows))
    _check_same_result(got, rows)


def test_converted_session_without_revenue_counts_but_adds_nothing() -> None:
    rows = [
        (1, "ads", "desktop", True, None),
        (2, "ads", "mobile", True, 20.0),
        (3, "ads", "mobile", False, None),
    ]
    got = _run_learner_query(_connect(rows))
    _check_same_result(got, rows)


def test_rate_is_a_fraction_not_an_integer_quotient() -> None:
    rows = [
        (1, "ads", "mobile", True, None),
        (2, "ads", "mobile", False, None),
        (3, "ads", "desktop", False, None),
    ]
    got = _run_learner_query(_connect(rows))
    _check_same_result(got, rows)


def test_counts_equal_exact_row_counts_without_truncation() -> None:
    rows = [(1, "ads", "mobile", True, 4.0), (2, "ads", "desktop", False, None)]
    got = _run_learner_query(_connect(rows))
    _check_same_result(got, rows)


def test_rate_is_a_fraction_not_a_percentage() -> None:
    rows = [
        (1, "ads", "mobile", True, 5.0),
        (2, "ads", "mobile", False, None),
        (3, "ads", "desktop", False, None),
        (4, "ads", "desktop", False, None),
    ]
    got = _run_learner_query(_connect(rows))
    _check_same_result(got, rows)


def test_revenue_counts_sessions_that_did_not_convert() -> None:
    rows = [
        (1, "social", "mobile", False, 8.0),
        (2, "social", "mobile", True, 4.0),
        (3, "social", "desktop", False, 2.0),
    ]
    got = _run_learner_query(_connect(rows))
    row = got.set_index("channel").loc["social"]
    assert math.isclose(float(row["mobile_revenue"]), 12.0), (
        "mobile_revenue differs: sum the revenue of every mobile session, converted or not"
    )
    _check_same_result(got, rows)


def test_empty_table_returns_zero_rows_with_the_six_columns() -> None:
    got = _run_learner_query(_connect([]))
    assert list(got.columns) == COLUMNS, (
        f"columns differ on an empty result: expected {COLUMNS}, got {list(got.columns)}"
    )
    assert len(got) == 0, f"no sessions must give no rows, got {len(got)}"


def test_rows_are_ordered_by_channel_ascending_when_inserted_in_reverse() -> None:
    rows = [
        (i + 1, channel, "mobile" if i % 2 else "desktop", i % 3 == 0, float(i))
        for i, channel in enumerate(reversed(CHANNELS))
    ]
    got = _run_learner_query(_connect(rows))
    assert list(got["channel"]) == sorted(CHANNELS), (
        "row order differs: ORDER BY channel ascending; a GROUP BY alone has no defined order"
    )
    _check_same_result(got, rows)
