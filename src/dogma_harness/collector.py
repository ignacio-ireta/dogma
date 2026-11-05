"""Pytest plugin that records each test's outcome, message, duration and captured stdout.

Outcome per test: the worst of its phases. A failing call is ``fail``; a failing setup or
teardown is ``error``; a skip is ``skip``; a test whose call passed is ``pass``.
"""

from __future__ import annotations

import io
import sys
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest

from dogma_harness.report import (
    MAX_STDOUT_BYTES,
    HarnessResult,
    TestRecord,
    trim_message,
    trim_tail,
    truncate_stdout,
)

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer

_Outcome = Literal["pass", "fail", "error", "skip"]
_RANK: dict[str, int] = {"pass": 0, "skip": 1, "fail": 2, "error": 3}
_HIDDEN_FILE = "test_hidden.py"
_DROPPED_MARKER = "\n[... output beyond the 64 KB limit was discarded ...]\n"


class _BoundedBuffer(io.BytesIO):
    """A byte sink that keeps its first ``limit`` bytes and discards the rest.

    A print loop therefore costs CPU until the time limit, never memory.
    """

    def __init__(self, limit: int) -> None:
        super().__init__()
        self._limit = limit
        self._kept = 0
        self.dropped = 0

    def write(self, data: ReadableBuffer, /) -> int:
        view = memoryview(data).cast("B")
        keep = min(view.nbytes, self._limit - self._kept)
        if keep > 0:
            super().write(view[:keep])
            self._kept += keep
        self.dropped += view.nbytes - max(keep, 0)
        return view.nbytes


@dataclass
class _Entry:
    nodeid: str
    outcome: str | None = None
    message: str = ""
    duration_s: float = 0.0


def describe_exception(exc: BaseException) -> str:
    """Learner-facing text for a failure: the assertion or exception message, plus notes.

    Hypothesis puts the falsifying example in the notes, so they are kept.
    """
    head = type(exc).__name__
    text = str(exc).strip()
    # AssertionError and pytest.fail() already read as a sentence; other exceptions get a prefix.
    plain = isinstance(exc, AssertionError) or type(exc).__module__.startswith("_pytest")
    message = (text or head) if plain else (f"{head}: {text}" if text else head)
    notes: list[str] = list(getattr(exc, "__notes__", ()))
    return trim_message("\n".join([message, *notes]))


def _phase_outcome(report: pytest.TestReport) -> str | None:
    if report.skipped:
        return "skip"
    if report.failed:
        return "fail" if report.when == "call" else "error"
    if report.when == "call":
        return "pass"
    return None  # a passing setup or teardown says nothing about the verdict


def _test_name(nodeid: str) -> str:
    return nodeid.split("::", 1)[1] if "::" in nodeid else nodeid


def _test_file(nodeid: str) -> Literal["public", "hidden"]:
    return "hidden" if Path(nodeid.split("::", 1)[0]).name == _HIDDEN_FILE else "public"


class Collector:
    """Pytest plugin instance passed to ``pytest.main(plugins=[...])``."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._messages: dict[str, str] = {}
        self._stdout: list[str] = []
        self._stdout_bytes = 0
        self._collection_errors: list[str] = []

    @contextmanager
    def _captured_stdout(self) -> Iterator[None]:
        """Capture ``sys.stdout`` for one phase, in memory and bounded (see ``_BoundedBuffer``)."""
        buffer = _BoundedBuffer(MAX_STDOUT_BYTES)
        stream = io.TextIOWrapper(
            buffer, encoding="utf-8", errors="replace", newline="", write_through=True
        )
        previous = sys.stdout
        sys.stdout = stream
        try:
            yield
        finally:
            sys.stdout = previous
            stream.flush()
            text = buffer.getvalue().decode("utf-8", errors="ignore")
            if buffer.dropped:
                text += _DROPPED_MARKER
            self._add_stdout(text)

    def _add_stdout(self, text: str) -> None:
        if not text or self._stdout_bytes >= MAX_STDOUT_BYTES:
            return
        self._stdout.append(text)
        self._stdout_bytes += len(text.encode("utf-8", errors="replace"))

    @pytest.hookimpl(wrapper=True)
    def pytest_make_collect_report(
        self, collector: pytest.Collector
    ) -> Generator[None, pytest.CollectReport | None, pytest.CollectReport | None]:
        with self._captured_stdout():
            report = yield
        return report

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_setup(self, item: pytest.Item) -> Generator[None, None, None]:
        with self._captured_stdout():
            yield

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_call(self, item: pytest.Item) -> Generator[None, None, None]:
        with self._captured_stdout():
            yield

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_teardown(
        self, item: pytest.Item, nextitem: pytest.Item | None
    ) -> Generator[None, None, None]:
        with self._captured_stdout():
            yield

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(
        self, item: pytest.Item, call: pytest.CallInfo[None]
    ) -> Generator[None, pytest.TestReport, pytest.TestReport]:
        report = yield
        if report.failed and call.excinfo is not None:
            # Keep the first failing phase: a setup error is the root cause, not a teardown echo.
            self._messages.setdefault(item.nodeid, describe_exception(call.excinfo.value))
        return report

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self._collection_errors.append(str(report.longrepr))

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        entry = self._entries.setdefault(report.nodeid, _Entry(report.nodeid))
        entry.duration_s += report.duration
        phase = _phase_outcome(report)
        if phase is None:
            return
        if entry.outcome is None or _RANK[phase] > _RANK[entry.outcome]:
            entry.outcome = phase
        if phase in ("fail", "error") and not entry.message:
            entry.message = self._messages.get(report.nodeid, "")

    def result(self, exit_code: int) -> HarnessResult:
        """Turn the recorded state into the report. ``exit_code`` is pytest's own."""
        stdout = truncate_stdout("".join(self._stdout))
        tests = tuple(self._test_records())
        if self._collection_errors:
            text = "\n\n".join(self._collection_errors)
            return HarnessResult("error", tests, stdout, trim_tail(text))
        if not tests:
            return HarnessResult("error", (), stdout, "no tests were collected")
        if exit_code not in (0, 1):
            return HarnessResult(
                "error", tests, stdout, f"pytest stopped early (exit code {exit_code})"
            )
        status: Literal["passed", "failed"] = (
            "passed" if all(t.outcome == "pass" for t in tests) else "failed"
        )
        return HarnessResult(status, tests, stdout, None)

    def _test_records(self) -> list[TestRecord]:
        records: list[TestRecord] = []
        for entry in self._entries.values():
            outcome: _Outcome = _as_outcome(entry.outcome)
            records.append(
                TestRecord(
                    name=_test_name(entry.nodeid),
                    file=_test_file(entry.nodeid),
                    outcome=outcome,
                    message=entry.message,
                    duration_ms=round(entry.duration_s * 1000),
                )
            )
        return records


_OUTCOMES: dict[str, _Outcome] = {"pass": "pass", "fail": "fail", "error": "error", "skip": "skip"}


def _as_outcome(outcome: str | None) -> _Outcome:
    """A test that never finished its call (e.g. the run was aborted) counts as an error."""
    return _OUTCOMES.get(outcome or "", "error")
