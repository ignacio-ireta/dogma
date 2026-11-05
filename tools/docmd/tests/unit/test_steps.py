"""Step plumbing: output path mapping and step-name validation."""

from pathlib import Path

import pytest

from docmd.errors import ConfigError
from docmd.pipeline.steps import output_path_for, steps_from_names


def test_output_path_mirrors_tree() -> None:
    out = Path("/out")
    assert output_path_for("a/b.docx", out) == Path("/out/a/b.docx.md")
    assert output_path_for("report.pdf", out) == Path("/out/report.pdf.md")


def test_output_paths_collision_proof() -> None:
    out = Path("/out")
    assert output_path_for("report.docx", out) != output_path_for("report.pdf", out)


def test_steps_default_order() -> None:
    assert [s.name for s in steps_from_names()] == ["extract", "normalize", "write"]


def test_steps_subset_keeps_canonical_order() -> None:
    assert [s.name for s in steps_from_names(["write", "extract"])] == ["extract", "write"]


def test_steps_unknown_name_rejected() -> None:
    with pytest.raises(ConfigError):
        steps_from_names(["extract", "frobnicate"])
