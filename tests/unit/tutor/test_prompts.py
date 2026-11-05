"""Versioned prompts: front matter, level sections, rendering, and golden snapshots per mode.

Set DOGMA_UPDATE_GOLDEN=1 to rewrite the golden files. Read the diff before committing it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from tests.unit.tutor.support import sample_evidence

from dogma.core.errors import DogmaError, ErrorCode
from dogma.tutor.context import assemble_prompt
from dogma.tutor.models import CallMode
from dogma.tutor.prompts import PROMPT_DIR, golden_text, load_template, render_prompt

GOLDEN = Path(__file__).parent / "golden"
CASES = [
    ("hint-L1", CallMode.HINT, 1),
    ("hint-L2", CallMode.HINT, 2),
    ("hint-L3", CallMode.HINT, 3),
    ("hint-L4", CallMode.HINT, 4),
    ("debug", CallMode.DEBUG, None),
]


@pytest.mark.parametrize("mode", list(CallMode))
def test_every_mode_has_a_versioned_template(mode: CallMode) -> None:
    template = load_template(mode)

    assert template.mode is mode
    assert template.version >= 1


def test_hint_has_the_four_ladder_levels_and_debug_has_none() -> None:
    assert sorted(load_template(CallMode.HINT).sections) == [1, 2, 3, 4]
    assert load_template(CallMode.DEBUG).sections == {}


def test_the_rules_every_prompt_must_carry_are_in_the_system_text() -> None:
    for mode in CallMode:
        system = render_prompt(mode, 1 if mode is CallMode.HINT else None, "x").messages[0].content
        assert "correct or incorrect" in system
        assert "unsourced" in system
        assert "hidden" in system.lower()


@pytest.mark.parametrize(
    ("front", "reason"),
    [
        ("mode: hint\n", "no version"),
        ("version: one\nmode: hint\n", "version not an integer"),
        ("version: 1\nmode: debug\n", "mode does not match the file name"),
    ],
)
def test_a_bad_front_matter_is_a_prompt_error(tmp_path: Path, front: str, reason: str) -> None:
    (tmp_path / "hint.md").write_text(f"---\n{front}---\n## L1 · x\nbody\n", encoding="utf-8")

    with pytest.raises(DogmaError) as invalid:
        load_template(CallMode.HINT, tmp_path)

    assert invalid.value.code is ErrorCode.TUTOR_PROMPT_INVALID, reason


def test_unclosed_front_matter_is_a_prompt_error(tmp_path: Path) -> None:
    (tmp_path / "debug.md").write_text("---\nversion: 1\nno closing fence\n", encoding="utf-8")

    with pytest.raises(DogmaError) as invalid:
        load_template(CallMode.DEBUG, tmp_path)

    assert invalid.value.code is ErrorCode.TUTOR_PROMPT_INVALID


def test_a_hint_needs_a_level_between_one_and_four() -> None:
    with pytest.raises(DogmaError) as invalid:
        render_prompt(CallMode.HINT, None, "x")

    assert invalid.value.code is ErrorCode.TUTOR_PROMPT_INVALID


def test_the_rendered_hash_covers_the_user_text_and_is_stable() -> None:
    first = render_prompt(CallMode.HINT, 2, "problem A")
    again = render_prompt(CallMode.HINT, 2, "problem A")
    other = render_prompt(CallMode.HINT, 2, "problem B")

    assert first.sha256 == again.sha256
    assert first.sha256 != other.sha256
    assert len(first.sha256) == 64
    assert first.version == load_template(CallMode.HINT).version


def test_prompt_files_live_in_the_package() -> None:
    assert (PROMPT_DIR / "hint.md").is_file()
    assert (PROMPT_DIR / "debug.md").is_file()


@pytest.mark.parametrize(("name", "mode", "level"), CASES)
def test_rendered_prompt_matches_its_golden_snapshot(
    name: str, mode: CallMode, level: int | None
) -> None:
    assembled = assemble_prompt(sample_evidence(), mode=mode, level=level)
    rendered = golden_text(assembled.prompt)
    path = GOLDEN / f"{name}.txt"

    if os.environ.get("DOGMA_UPDATE_GOLDEN") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    assert path.is_file(), f"missing golden {path}; run with DOGMA_UPDATE_GOLDEN=1"
    assert rendered == path.read_text(encoding="utf-8")
