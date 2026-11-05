"""Harness output helpers: truncation, trimming and the atomic JSON write."""

from __future__ import annotations

import json
from pathlib import Path

from dogma_harness.report import (
    MAX_MESSAGE_CHARS,
    MAX_STDOUT_BYTES,
    HarnessResult,
    trim_message,
    trim_tail,
    truncate_stdout,
    write_report,
)


def test_short_stdout_is_returned_unchanged():
    assert truncate_stdout("hello\n") == "hello\n"


def test_one_megabyte_of_stdout_is_cut_to_the_limit_with_a_marker():
    out = truncate_stdout("y" * 1_000_000)
    assert len(out.encode("utf-8")) <= MAX_STDOUT_BYTES
    assert out.startswith("yyyy")
    assert out.endswith("]\n")


def test_truncation_never_splits_a_multibyte_character():
    out = truncate_stdout("é" * 100_000)  # two bytes per character
    assert len(out.encode("utf-8")) <= MAX_STDOUT_BYTES
    out.encode("utf-8")  # would raise if a character had been split


def test_trim_message_keeps_the_head_within_the_limit():
    text = "a" * (MAX_MESSAGE_CHARS + 50)
    trimmed = trim_message(text)
    assert len(trimmed) == MAX_MESSAGE_CHARS
    assert trimmed.endswith("...")


def test_trim_tail_keeps_the_end_where_the_exception_line_is():
    text = "x" * 5000 + "E   SyntaxError: boom"
    trimmed = trim_tail(text)
    assert len(trimmed) == MAX_MESSAGE_CHARS
    assert trimmed.endswith("E   SyntaxError: boom")


def test_short_messages_are_not_altered():
    assert trim_message("ok") == "ok"
    assert trim_tail("ok") == "ok"


def test_errored_result_carries_the_message_only():
    result = HarnessResult.errored("harness failure: boom")
    assert result.status == "error"
    assert result.tests == ()
    assert result.collection_error == "harness failure: boom"


def test_write_report_is_valid_json_and_leaves_no_temp_file(tmp_path: Path):
    target = tmp_path / "report.json"
    write_report(target, HarnessResult.errored("nope"))
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == {
        "status": "error",
        "tests": [],
        "stdout": "",
        "collection_error": "nope",
    }
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]
