"""Reading a problem's reviewed hints.md: both heading styles, labels, and every broken shape."""

from __future__ import annotations

from pathlib import Path

import pytest

from dogma.tutor.hints import load_reviewed_hints, parse_reviewed_hints

REPO_ROOT = Path(__file__).resolve().parents[3]

COLON_STYLE = """## Hint 1: concept

First level text.

## Hint 2: formula / approach

Second level text with `c = a / b`.

## Hint 3: code shape

Third level text.
"""

DASH_STYLE = (
    COLON_STYLE.replace(": concept", " — concept")
    .replace(": formula / approach", " — formula / approach")
    .replace(": code shape", " — code shape")
)


def test_the_colon_and_the_em_dash_styles_both_parse() -> None:
    for text in (COLON_STYLE, DASH_STYLE):
        reviewed = parse_reviewed_hints(text)

        assert reviewed.note is None
        assert [hint.level for hint in reviewed.hints] == [1, 2, 3]
        assert [hint.label for hint in reviewed.hints] == [
            "concept",
            "formula / approach",
            "code shape",
        ]
        second = reviewed.get(2)
        assert second is not None
        assert second.text == "Second level text with `c = a / b`."


def test_up_to_gives_the_ceiling_and_get_is_none_for_other_levels() -> None:
    reviewed = parse_reviewed_hints(COLON_STYLE)

    assert [hint.level for hint in reviewed.up_to(2)] == [1, 2]
    assert reviewed.get(4) is None
    assert parse_reviewed_hints("").up_to(3) == ()


def test_a_missing_file_is_a_note_not_an_error(tmp_path: Path) -> None:
    reviewed = load_reviewed_hints(tmp_path)

    assert reviewed.hints == ()
    assert reviewed.note is not None
    assert "missing" in reviewed.note
    assert "model alone" in reviewed.note


def test_a_file_that_is_not_utf8_is_a_note(tmp_path: Path) -> None:
    (tmp_path / "hints.md").write_bytes(b"\xff\xfe## Hint 1: x\n")

    reviewed = load_reviewed_hints(tmp_path)

    assert reviewed.hints == ()
    assert reviewed.note is not None
    assert "cannot be read" in reviewed.note


def test_the_file_is_read_from_the_problem_directory(tmp_path: Path) -> None:
    (tmp_path / "hints.md").write_text(COLON_STYLE, encoding="utf-8")

    assert len(load_reviewed_hints(tmp_path).hints) == 3


@pytest.mark.parametrize(
    "text",
    [
        "## Hint 1: concept\n\nOnly one level.\n",
        "## Hint 1: concept\n\nA.\n\n## Hint 2: approach\n\nB.\n",
        "## Hint 1: concept\n\nA.\n\n## Hint 3: code shape\n\nC.\n\n## Hint 2: approach\n\nB.\n",
        "## Hint 1: concept\n\nA.\n\n## Hint 2: approach\n\nB.\n\n## Hint 4: extra\n\nD.\n",
        "## Hint 1: concept\n\nA.\n\n## Hint 2: approach\n\n\n## Hint 3: code shape\n\nC.\n",
        "## Hint 1: concept\n\nA.\n\n## Hint 2: approach\n\nB.\n\n## Step 3\n\nC.\n",
        "## Hint 1: concept\n\nA.\n\n## Hint 2: approach\n\nB.\n\n## Hint 10: x\n\nC.\n",
        "# Title only\n",
    ],
)
def test_a_wrong_shape_gives_no_hints_and_a_malformed_note(text: str) -> None:
    reviewed = parse_reviewed_hints(text)

    assert reviewed.hints == ()
    assert reviewed.note is not None
    assert "malformed" in reviewed.note


def test_a_missing_label_gets_the_default_label() -> None:
    text = "## Hint 1\n\nA.\n\n## Hint 2\n\nB.\n\n## Hint 3\n\nC.\n"

    reviewed = parse_reviewed_hints(text)

    labels = [hint.label for hint in reviewed.hints]
    assert labels == ["concept", "formula / approach", "code shape"]


def test_fenced_code_with_a_hash_heading_inside_is_not_a_section() -> None:
    text = (
        "## Hint 1: concept\n\nA.\n\n## Hint 2: approach\n\n```\n## Hint 9: not a heading\n```\n\n"
        "## Hint 3: code shape\n\nC.\n"
    )

    reviewed = parse_reviewed_hints(text)

    assert reviewed.note is None
    assert "Hint 9" in reviewed.hints[1].text


def test_every_authored_hints_file_in_the_repository_is_usable() -> None:
    files = sorted((REPO_ROOT / "content").rglob("hints.md"))
    assert files, "the repository has authored hints"

    unusable = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in files
        if (loaded := load_reviewed_hints(path.parent)).note is not None or len(loaded.hints) != 3
    ]

    assert unusable == []
