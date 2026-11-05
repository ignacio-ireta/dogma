"""Shared fixtures for the content tests: a mutable copy of the fixture packages, and real files."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from dogma.content.loader import load_core_path, load_syllabus
from dogma.content.schemas import CorePath, Syllabus

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "content"
REAL_SYLLABUS = REPO_ROOT / "curriculum" / "syllabus.yaml"
REAL_CORE_PATH = REPO_ROOT / "curriculum" / "paths" / "core.yaml"
REAL_EXAMPLE = REPO_ROOT / "content" / "cml" / "vector-operations"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def real_syllabus() -> Syllabus:
    return load_syllabus(REAL_SYLLABUS)


@pytest.fixture
def real_core() -> CorePath:
    return load_core_path(REAL_CORE_PATH)


@pytest.fixture
def real_example() -> Path:
    return REAL_EXAMPLE


@pytest.fixture
def syllabus() -> Syllabus:
    """The mini syllabus: tst/alpha (A), tst/beta (B), tst/gamma (A, no package)."""
    return load_syllabus(FIXTURES / "syllabus.yaml")


@pytest.fixture
def core() -> CorePath:
    return load_core_path(FIXTURES / "core.yaml")


@pytest.fixture
def packages(tmp_path: Path) -> Path:
    """A writable copy of the fixture ``packages/`` tree. Tests mutate this copy only."""
    target = tmp_path / "packages"
    shutil.copytree(FIXTURES / "packages", target)
    return target


@pytest.fixture
def alpha(packages: Path) -> Path:
    return packages / "tst" / "alpha"


@pytest.fixture
def beta(packages: Path) -> Path:
    return packages / "tst" / "beta"


@pytest.fixture
def alpha_problem(alpha: Path) -> Path:
    return alpha / "problems" / "p01-double-all"


@pytest.fixture
def edit() -> Callable[[Path, str, str], None]:
    """Replace one exact occurrence of ``old`` with ``new`` in ``path``."""

    def _edit(path: Path, old: str, new: str) -> None:
        text = path.read_text(encoding="utf-8")
        assert text.count(old) == 1, f"expected exactly one {old!r} in {path}"
        path.write_text(text.replace(old, new), encoding="utf-8")

    return _edit


@pytest.fixture
def write_syllabus(tmp_path: Path) -> Callable[[str], Syllabus]:
    """Write alternative syllabus YAML to a temp file and load it."""

    def _write(text: str) -> Syllabus:
        path = tmp_path / "syllabus-variant.yaml"
        path.write_text(text, encoding="utf-8")
        return load_syllabus(path)

    return _write


@pytest.fixture
def write_core(tmp_path: Path) -> Callable[[str], CorePath]:
    """Write alternative learning-path YAML to a temp file and load it."""

    def _write(text: str) -> CorePath:
        path = tmp_path / "core-variant.yaml"
        path.write_text(text, encoding="utf-8")
        return load_core_path(path)

    return _write
