"""SQL problems through ``runner.run``: the submission is a query run against DuckDB fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from dogma.grading.models import RunStatus
from dogma.runner import run

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_ORDERS = REPO_ROOT / "tests" / "fixtures" / "problems" / "sql-orders"
CORRECT = "SELECT country, SUM(amount) AS total FROM orders WHERE amount >= 10 GROUP BY country;\n"


def test_correct_query_passes_public_and_hidden_tests(settings, submit):
    report = run(
        SQL_ORDERS, submit(CORRECT, "submission.sql"), include_hidden=True, settings=settings
    )
    assert report.status is RunStatus.PASSED
    assert report.counts("public") == (2, 2)
    assert report.counts("hidden") == (3, 3)
    assert report.runtime == "sql-duckdb"


def test_off_by_one_threshold_fails_and_names_the_missing_country(settings, submit):
    wrong = CORRECT.replace(">=", ">")
    report = run(
        SQL_ORDERS, submit(wrong, "submission.sql"), include_hidden=True, settings=settings
    )
    assert report.status is RunStatus.FAILED
    by_name = {t.name: t for t in report.tests}
    threshold = by_name["test_amount_exactly_ten_is_included"]
    assert threshold.file == "hidden"
    assert threshold.outcome == "fail"
    assert "exactly 10" in threshold.message


def test_query_with_wrong_columns_fails_the_column_test(settings, submit):
    wrong = "SELECT country FROM orders WHERE amount >= 10 GROUP BY country;\n"
    report = run(
        SQL_ORDERS, submit(wrong, "submission.sql"), include_hidden=False, settings=settings
    )
    assert report.status is RunStatus.FAILED
    by_name = {t.name: t for t in report.tests}
    assert by_name["test_output_has_exactly_country_and_total_columns"].outcome == "fail"


def test_sql_syntax_error_fails_with_the_database_message(settings, submit):
    report = run(
        SQL_ORDERS,
        submit("SELEC country FROM orders;\n", "submission.sql"),
        include_hidden=False,
        settings=settings,
    )
    assert report.status is RunStatus.FAILED
    failures = [t for t in report.tests if t.outcome == "fail"]
    assert failures
    assert "ParserException" in failures[0].message or "syntax" in failures[0].message.lower()


def test_empty_sql_file_is_rejected_without_running(settings, submit):
    report = run(
        SQL_ORDERS, submit("   \n", "submission.sql"), include_hidden=True, settings=settings
    )
    assert report.status is RunStatus.REJECTED
    assert [f.rule for f in report.static] == ["empty"]
    assert not settings.runs_dir.exists()
