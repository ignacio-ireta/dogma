from pathlib import Path

import duckdb


def _run(sql: str, rows: list[tuple[int, float | None]]) -> list[tuple[int, float | None]]:
    con = duckdb.connect()
    con.execute("CREATE TABLE orders (user_id INTEGER, amount DOUBLE)")
    con.executemany("INSERT INTO orders VALUES (?, ?)", rows)
    return sorted(con.execute(sql).fetchall(), key=lambda row: (row[0] is None, row[0] or 0))


def _sql() -> str:
    return Path(__file__).with_name("submission.sql").read_text()


def test_matches_reference_on_a_larger_fixture() -> None:
    rows = [(user % 5, float(user)) for user in range(100)]
    expected = sorted(
        (key, float(sum(amount for k, amount in rows if k == key))) for key in range(5)
    )
    assert _run(_sql(), rows) == expected, "totals differ from the reference"


def test_null_amounts_are_ignored_by_sum() -> None:
    assert _run(_sql(), [(1, None), (1, 2.0)]) == [(1, 2.0)], "NULL amount not ignored"


def test_all_null_group_sums_to_null() -> None:
    assert _run(_sql(), [(3, None)]) == [(3, None)], "all-NULL group is not NULL"


def test_result_has_one_row_per_user() -> None:
    rows = [(1, 1.0), (1, 2.0), (2, 3.0)]
    result = _run(_sql(), rows)
    assert len({user for user, _ in result}) == len(result), "user repeated in result"


def test_input_table_is_not_changed() -> None:
    con = duckdb.connect()
    con.execute("CREATE TABLE orders (user_id INTEGER, amount DOUBLE)")
    con.execute("INSERT INTO orders VALUES (1, 2.0)")
    con.execute(_sql())
    assert con.execute("SELECT COUNT(*) FROM orders").fetchone() == (1,), "input table changed"
