"""``python -m dogma_harness --run-dir DIR --tests FILE... --report PATH``.

Runs pytest in-process inside the run directory and always writes the report, even when
the tests cannot be collected (ARCHITECTURE §4.3). Exit code is 0 whenever a report exists.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from dogma_harness.collector import Collector
from dogma_harness.report import HarnessResult, write_report

# ``--capture=no``: the collector captures stdout itself, per phase and bounded in memory.
PYTEST_ARGS = (
    "-q",
    "-p",
    "no:cacheprovider",
    "--hypothesis-seed=0",
    "--capture=no",
    "--tb=short",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns 0 once a report has been written."""
    args = _parser().parse_args(argv)
    report_path = Path(args.report).resolve()
    run_dir = Path(args.run_dir).resolve()
    tests_dir = Path(args.tests_dir).resolve() if args.tests_dir is not None else None
    try:
        result = _run(run_dir, tuple(args.tests), tests_dir)
    except BaseException as exc:  # the harness owes the runner a report, whatever broke
        result = HarnessResult.errored(f"harness failure: {type(exc).__name__}: {exc}")
    write_report(report_path, result)
    return 0


def _run(run_dir: Path, tests: tuple[str, ...], tests_dir: Path | None = None) -> HarnessResult:
    """Collect and run ``tests`` from ``tests_dir`` (default: ``run_dir``), with ``run_dir`` as cwd.

    Test paths are absolute, so the test files may sit outside the writable run directory. The
    pytest configuration is named explicitly for the same reason.
    """
    source = tests_dir if tests_dir is not None else run_dir
    missing = [name for name in tests if not (source / name).is_file()]
    if missing:
        return HarnessResult.errored(f"test file not found: {', '.join(missing)}")
    os.chdir(run_dir)
    # A pytest.ini of our own stops pytest from walking up to the repo's pyproject.toml.
    config = run_dir / "pytest.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    collector = Collector()
    targets = [str(source / name) for name in tests]
    # Node ids are relative to the rootdir, and the collector reads visibility from their file
    # part. Without this, tests outside the run dir would lose their file and count as public.
    rootdir = [f"--rootdir={source}"]
    exit_code = pytest.main(
        [*PYTEST_ARGS, "-c", str(config), *rootdir, *targets], plugins=[collector]
    )
    return collector.result(int(exit_code))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dogma_harness")
    parser.add_argument("--run-dir", required=True, help="directory holding the submission")
    parser.add_argument(
        "--tests-dir",
        default=None,
        help="directory holding the test files (default: --run-dir)",
    )
    parser.add_argument("--tests", nargs="+", required=True, help="test files, public first")
    parser.add_argument("--report", required=True, help="where to write report.json")
    return parser
