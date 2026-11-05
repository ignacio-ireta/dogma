from __future__ import annotations

import pytest

from dogma.content.markdown import parse_outline, split_front_matter
from dogma.content.pyfacts import parse_python_facts


def test_front_matter_is_split_from_the_body() -> None:
    front, body = split_front_matter("---\ntitle: T\n---\n\n## One\n")
    assert front == "title: T\n"
    assert body == "\n## One\n"


def test_text_without_front_matter_is_all_body() -> None:
    assert split_front_matter("## One\n") == (None, "## One\n")


def test_unclosed_front_matter_is_an_error() -> None:
    with pytest.raises(ValueError, match="never closed"):
        split_front_matter("---\ntitle: T\n## One\n")


def test_outline_takes_first_h1_and_h2_sections_in_order() -> None:
    outline = parse_outline("# Title\n\nintro\n\n## Alpha\n\ntext\n\n## Beta\n")
    assert outline.title == "Title"
    assert outline.headings == ("Alpha", "Beta")


def test_headings_inside_fences_are_ignored() -> None:
    body = "## Real\n\n```md\n## fake\n```\n\n## Next\n"
    outline = parse_outline(body)
    assert outline.headings == ("Real", "Next")


def test_three_hash_headings_are_not_h2() -> None:
    assert parse_outline("## One\n### Sub\n").headings == ("One",)


def test_examples_counts_doctest_lines_and_fences() -> None:
    body = "## Examples\n```python\n>>> f(1)\n1\n>>> f(2)\n2\n```\n"
    (section,) = parse_outline(body).sections
    assert section.fences == 1
    assert section.doctests == 2
    assert section.examples == 2


def test_python_facts_read_top_level_names_tests_and_imports() -> None:
    source = (
        "from submission import double_all\n"
        "import numpy as np\n"
        "CONST = 1\n"
        "class Helper: ...\n"
        "def test_one() -> None: ...\n"
        "def helper() -> None: ...\n"
    )
    facts = parse_python_facts(source)
    assert facts.syntax_error is None
    assert facts.defined == frozenset({"CONST", "Helper", "test_one", "helper"})
    assert facts.test_functions == ("test_one",)
    assert facts.submission_names == frozenset({"double_all"})
    assert {"numpy", "submission"} <= facts.imported_modules


def test_python_syntax_error_is_reported_not_raised() -> None:
    facts = parse_python_facts("def broken(:\n")
    assert facts.syntax_error is not None
    assert facts.defined == frozenset()


def test_section_text_keeps_fenced_lines_but_not_the_heading() -> None:
    body = "## Constraints\n\n```text\nnot allowed: numpy.linalg\n```\n\n## Done\n"
    constraints, _done = parse_outline(body).sections
    assert "numpy.linalg" in constraints.text
    assert "## Done" not in constraints.text
