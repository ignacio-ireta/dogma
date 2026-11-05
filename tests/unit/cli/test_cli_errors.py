"""Friendly errors, usage errors, and the --help examples."""

from __future__ import annotations

import pytest
from tests.unit.cli.support import PROBLEM

from dogma.cli import app


def test_a_broken_problem_file_is_one_line_with_its_code(invoke, repo):
    meta = repo / "content" / "cml" / "vector-operations" / "problems" / "p01-vector-projection"
    (meta / "meta.yaml").write_text("id: x\nunknown_key: true\n")

    outcome = invoke("open", PROBLEM)

    assert outcome.exit_code == 1
    assert "error [CONTENT_SCHEMA_INVALID]" in outcome.output
    assert "Traceback" not in outcome.output


def test_a_missing_workspace_file_is_a_usage_error(invoke):
    outcome = invoke("run", PROBLEM)

    assert outcome.exit_code == 2
    assert "run `dogma open" in outcome.output


@pytest.mark.parametrize(
    "command",
    [
        ("open",),
        ("run",),
        ("submit",),
        ("path",),
        ("content", "verify"),
        ("content", "coverage"),
        ("content", "new"),
        ("catalog", "build"),
    ],
)
def test_every_command_help_has_one_example(cli, settings, command):
    outcome = cli.invoke(app, [*command, "--help"], obj=settings)

    assert outcome.exit_code == 0, outcome.output
    assert "Example:" in outcome.output, command


def test_root_help_lists_the_commands(cli, settings):
    outcome = cli.invoke(app, ["--help"], obj=settings)

    assert outcome.exit_code == 0
    for word in ("open", "run", "submit", "path", "content", "catalog"):
        assert word in outcome.output
