from pathlib import Path

import duckdb


def _run(sql: str, rows: list[tuple[int, float]]) -> list[tuple[int, float]]:
    con = duckdb.connect()
    con.execute("CREATE TABLE orders (user_id INTEGER, amount DOUBLE)")
    con.executemany("INSERT INTO orders VALUES (?, ?)", rows)
    return sorted(con.execute(sql).fetchall())


def test_totals_each_user_from_the_example() -> None:
    sql = Path(__file__).with_name("submission.sql").read_text()
    assert _run(sql, [(1, 2.5), (1, 1.5), (2, 4.0)]) == [(1, 4.0), (2, 4.0)]


def test_empty_table_gives_no_rows() -> None:
    sql = Path(__file__).with_name("submission.sql").read_text()
    assert _run(sql, []) == []


def test_single_order_user_keeps_its_amount() -> None:
    sql = Path(__file__).with_name("submission.sql").read_text()
    assert _run(sql, [(7, 3.25)]) == [(7, 3.25)]
