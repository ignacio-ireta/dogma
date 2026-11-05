from __future__ import annotations

from pathlib import Path

import pytest

from dogma.core.errors import ErrorCode, RunnerError
from dogma.runner.meta import load_meta

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = REPO_ROOT / "content" / "cml" / "vector-operations" / "problems" / "p01-vector-projection"

VALID = """\
id: cml/x/p01-demo
version: 3
language: python
runtime: py-numpy
entry: demo
banned_imports: [numpy.linalg]
timeout_s: 12
"""


def _problem(tmp_path: Path, meta: str) -> Path:
    (tmp_path / "meta.yaml").write_text(meta, encoding="utf-8")
    return tmp_path


def test_loads_the_fields_the_runner_needs(tmp_path):
    meta = load_meta(_problem(tmp_path, VALID))
    assert (meta.id, meta.version, meta.language, meta.runtime, meta.entry) == (
        "cml/x/p01-demo",
        3,
        "python",
        "py-numpy",
        "demo",
    )
    assert meta.banned_imports == ("numpy.linalg",)
    assert meta.timeout_s == 12


def test_ignores_keys_the_runner_does_not_use(tmp_path):
    meta = load_meta(_problem(tmp_path, VALID + "difficulty: hard\ntolerance: {atol: 1}\n"))
    assert meta.id == "cml/x/p01-demo"


def test_optional_fields_default_to_none_and_empty(tmp_path):
    minimal = "id: a\nversion: 1\nlanguage: sql\nruntime: sql-duckdb\nentry: query\n"
    meta = load_meta(_problem(tmp_path, minimal))
    assert meta.banned_imports == ()
    assert meta.timeout_s is None


def test_missing_meta_file_is_a_content_error(tmp_path):
    with pytest.raises(RunnerError) as info:
        load_meta(tmp_path)
    assert info.value.code == ErrorCode.CONTENT_NOT_FOUND


@pytest.mark.parametrize(
    "meta",
    [
        VALID.replace("language: python", "language: cpp"),
        VALID.replace("version: 3", "version: 0"),
        VALID.replace("timeout_s: 12", "timeout_s: -1"),
        VALID.replace("entry: demo\n", ""),
        "- just\n- a list\n",
        "id: [unclosed\n",
    ],
    ids=[
        "bad-language",
        "bad-version",
        "bad-timeout",
        "missing-entry",
        "not-a-mapping",
        "bad-yaml",
    ],
)
def test_invalid_meta_is_a_schema_error(tmp_path, meta):
    with pytest.raises(RunnerError) as info:
        load_meta(_problem(tmp_path, meta))
    assert info.value.code == ErrorCode.CONTENT_SCHEMA_INVALID


def test_schema_error_names_the_offending_field(tmp_path):
    with pytest.raises(RunnerError) as info:
        load_meta(_problem(tmp_path, VALID.replace("language: python", "language: cpp")))
    assert "language" in info.value.message


def test_loads_the_real_example_problem():
    meta = load_meta(EXAMPLE)
    assert meta.id == "cml/vector-operations/p01-vector-projection"
    assert meta.language == "python"
    assert meta.entry == "vector_projection"
    assert "numpy.linalg" in meta.banned_imports
