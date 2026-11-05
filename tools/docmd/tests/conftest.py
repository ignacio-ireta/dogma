"""Shared test infrastructure: golden-file comparison with --update-golden."""

from __future__ import annotations

import difflib
from collections.abc import Callable
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
EXPECTED_DIR = TESTS_DIR / "expected"

GoldenCheck = Callable[[str, Path], None]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite golden files in tests/expected/ from actual output.",
    )


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Build binary fixtures once per session; rebuild when the generator changes."""
    from tests.fixtures.generate_fixtures import generate_all

    build_dir = TESTS_DIR / "fixtures" / "_build"
    stamp = build_dir / ".stamp"
    generator = TESTS_DIR / "fixtures" / "generate_fixtures.py"
    if not stamp.exists() or stamp.stat().st_mtime < generator.stat().st_mtime:
        generate_all(build_dir)
        stamp.touch()
    return build_dir


@pytest.fixture
def assert_matches_golden(pytestconfig: pytest.Config) -> GoldenCheck:
    update: bool = pytestconfig.getoption("--update-golden")

    def check(actual: str, expected_path: Path) -> None:
        if update:
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            expected_path.write_text(actual, encoding="utf-8")
            return
        if not expected_path.exists():
            pytest.fail(
                f"golden file missing: {expected_path}\n"
                "run: uv run pytest -m golden --update-golden"
            )
        expected = expected_path.read_text(encoding="utf-8")
        if actual != expected:
            diff = "\n".join(
                difflib.unified_diff(
                    expected.splitlines(),
                    actual.splitlines(),
                    fromfile=str(expected_path),
                    tofile="actual",
                    lineterm="",
                )
            )
            pytest.fail(f"golden mismatch:\n{diff}")

    return check
