"""Public tests: the fixture answer, the column order, a channel with no conversions and a channel
with one device type. The hidden tests use tables that each test builds itself."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
COLUMNS = [
    "channel",
    "sessions",
    "converted_sessions",
    "conversion_rate",
    "mobile_revenue",
    "desktop_revenue",
]
RATE_TOLERANCE = 5e-5 + 1e-9  # half of one ROUND(x, 4) step, plus float noise


def _load_fixture() -> list[tuple[Any, ...]]:
    frame = pd.read_csv(FIXTURES / "sessions.csv")
    rows: list[tuple[Any, ...]] = []
    for rec in frame.itertuples(index=False):
        revenue = None if pd.isna(rec.revenue) else float(rec.revenue)
        rows.append(
            (int(rec.session_id), str(rec.channel), str(rec.device), bool(rec.converted), revenue)
        )
    return rows


def _connect(rows: list[tuple[Any, ...]]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE sessions (session_id BIGINT, channel VARCHAR, device VARCHAR, "
        "converted BOOLEAN, revenue DOUBLE)"
    )
    con.executemany("INSERT INTO sessions VALUES (?, ?, ?, ?, ?)", rows)
    return con


def _run_learner_query(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = Path(__file__).with_name("submission.sql").read_text()
    try:
        result = con.execute(sql)
    except duckdb.Error as exc:
        pytest.fail(f"your SQL did not run: {exc}")
    if result is None:
        pytest.fail("submission.sql returned no result set: finish it with a SELECT")
    return result.fetchdf()


def _oracle(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Independent answer in pandas: per channel, counts, the exact fraction and revenue sums."""
    frame = pd.DataFrame(rows, columns=["session_id", "channel", "device", "converted", "revenue"])
    out: list[tuple[Any, ...]] = []
    for channel, group in frame.groupby("channel", sort=True):
        sessions = len(group)
        converted = int(group["converted"].sum())
        mobile = group.loc[group["device"] == "mobile", "revenue"].sum()
        desktop = group.loc[group["device"] == "desktop", "revenue"].sum()
        out.append(
            (channel, sessions, converted, converted / sessions, float(mobile), float(desktop))
        )
    return out


def _check_fixture(got: pd.DataFrame) -> None:
    assert list(got.columns) == COLUMNS, (
        f"output columns differ: expected {COLUMNS}, got {list(got.columns)}"
    )
    want = _oracle(_load_fixture())
    assert list(got["channel"]) == [row[0] for row in want], (
        "channels or their order differ: one row per channel, ordered by channel ascending"
    )
    for learner, expected in zip(got.itertuples(index=False), want, strict=True):
        assert learner.sessions == expected[1], (
            f"sessions differs for {expected[0]}: count every session of the channel, "
            "on both devices, converted or not"
        )
        assert learner.converted_sessions == expected[2], (
            f"converted_sessions differs for {expected[0]}: count only the converted rows"
        )
        assert learner.conversion_rate is not None and not pd.isna(learner.conversion_rate), (
            f"conversion_rate is NULL for {expected[0]}: a channel with no conversions has 0.0"
        )
        assert abs(float(learner.conversion_rate) - expected[3]) <= RATE_TOLERANCE, (
            f"conversion_rate differs for {expected[0]}: it is converted_sessions / sessions "
            "as a fraction, rounded to 4 decimal places"
        )
        for label, got_value, want_value in (
            ("mobile_revenue", learner.mobile_revenue, expected[4]),
            ("desktop_revenue", learner.desktop_revenue, expected[5]),
        ):
            assert got_value is not None and not pd.isna(got_value), (
                f"{label} is NULL for {expected[0]}: wrap the conditional SUM in COALESCE(..., 0.0)"
            )
            assert math.isclose(float(got_value), want_value, rel_tol=1e-9, abs_tol=1e-6), (
                f"{label} differs for {expected[0]}: sum the revenue of every session on the device"
            )


def test_fixture_answer_matches_the_pandas_computation() -> None:
    _check_fixture(_run_learner_query(_connect(_load_fixture())))


def test_output_has_the_six_columns_in_order() -> None:
    got = _run_learner_query(_connect(_load_fixture()))
    assert list(got.columns) == COLUMNS, (
        f"output columns differ: expected {COLUMNS}, got {list(got.columns)}"
    )


def test_channel_with_no_conversions_and_one_device_keeps_zero_values() -> None:
    got = _run_learner_query(_connect(_load_fixture())).set_index("channel")
    assert "organic" in got.index and "podcast" in got.index, (
        "a channel is missing: keep every channel that has a session, even with no conversions"
    )
    organic = got.loc["organic"]
    assert float(organic["conversion_rate"]) == 0.0, (
        "organic has no conversions, so its conversion_rate is 0.0, not NULL or another value: "
        "check converted_sessions (count only the converted rows) and the division"
    )
    podcast = got.loc["podcast"]
    assert float(podcast["mobile_revenue"]) == 0.0, (
        "podcast has no mobile session: its mobile_revenue is 0.0, not NULL"
    )
    assert np.isclose(float(podcast["desktop_revenue"]), 12.5), (
        "podcast desktop revenue is wrong: sum the desktop sessions' revenue"
    )
