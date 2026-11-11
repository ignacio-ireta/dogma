"""The hidden-test lint runs inside validation, one mutated fixture copy per rule."""

from __future__ import annotations

from pathlib import Path

import pytest

from dogma.content.loader import load_topic
from dogma.content.schemas import Syllabus
from dogma.content.testlint import RULES
from dogma.content.validate import validate_topic

MUTATIONS = {
    "HIDDEN_PARAM_IDS": (
        "\n\n"
        "@pytest.mark.parametrize('value', [1.0, 2.0])\n"
        "def test_extra_case(value: float) -> None:\n"
        "    assert double_all([value]) == [2 * value], 'doubling is wrong'\n"
    ),
    "HIDDEN_BARE_RAISES": (
        "\n\ndef test_extra_raises() -> None:\n"
        "    with pytest.raises(TypeError):\n"
        "        double_all(None)  # type: ignore[arg-type]\n"
    ),
    "HIDDEN_GIVEN_SETTINGS": (
        "\n\n"
        "@given(st.floats(-1.0, 1.0))\n"
        "def test_extra_property(value: float) -> None:\n"
        "    assert double_all([value]) == [2 * value], 'doubling is wrong'\n"
    ),
    "HIDDEN_ASSERT_MESSAGE": (
        "\n\ndef test_extra_assert() -> None:\n    assert double_all([1.0]) == [2.0]\n"
    ),
}

HEADER = "import pytest\nfrom hypothesis import given\nfrom hypothesis import strategies as st\n"


def _codes(alpha: Path, syllabus: Syllabus) -> set[str]:
    return {issue.code for issue in validate_topic(load_topic(alpha), syllabus)}


def test_every_rule_has_a_mutation() -> None:
    assert set(MUTATIONS) == set(RULES)


def test_unmodified_fixture_gives_no_lint_issue(alpha: Path, syllabus: Syllabus) -> None:
    assert _codes(alpha, syllabus).isdisjoint(RULES)


@pytest.mark.parametrize("rule", RULES)
def test_each_lint_rule_is_reported_by_validation(
    rule: str, alpha: Path, alpha_problem: Path, syllabus: Syllabus
) -> None:
    hidden = alpha_problem / "test_hidden.py"
    hidden.write_text(HEADER + hidden.read_text(encoding="utf-8") + MUTATIONS[rule])
    assert rule in _codes(alpha, syllabus)
