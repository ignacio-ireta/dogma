"""Golden snapshots for the score and interviewer prompts (ENGINEERING_STANDARDS §6).

Set DOGMA_UPDATE_GOLDEN=1 to rewrite the golden files. Read the diff before committing it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dogma.content.rubrics import load_rubric_version
from dogma.tutor.interviewer import build_user_text as interviewer_text
from dogma.tutor.models import CallMode
from dogma.tutor.prompts import golden_text, load_template, render_prompt
from dogma.tutor.score import build_user_text as score_text

GOLDEN = Path(__file__).parent / "golden"
CURRICULUM = Path(__file__).resolve().parents[3] / "curriculum"
SAMPLE_ANSWER = (
    "Start with a churn label defined as a cancelled line. Segment by plan first, because "
    "prepaid behaves differently. The trade-off is a holdout that costs some revenue."
)
SAMPLE_NOTES = "coding.narration: I said the loop out loud before writing it."
BRIEF = "A mid-size mobile operator has 12 million subscribers. Churn rose this year."
WHEN_ASKED = (
    "What does the current retention process do?",
    "How do the segments differ?",
)


def _prompts() -> dict[str, str]:
    topic = load_rubric_version(CURRICULUM, "topic-answer", 1)
    mock = load_rubric_version(CURRICULUM, "mock", 1)
    return {
        "score-topic": golden_text(
            render_prompt(CallMode.SCORE, None, score_text(topic, SAMPLE_ANSWER))
        ),
        "score-mock": golden_text(
            render_prompt(CallMode.SCORE, None, score_text(mock, SAMPLE_NOTES))
        ),
        "interviewer": golden_text(
            render_prompt(
                CallMode.INTERVIEWER,
                None,
                interviewer_text(BRIEF, WHEN_ASKED, "How do the segments differ?"),
            )
        ),
    }


def test_the_new_modes_have_versioned_templates() -> None:
    for mode in (CallMode.SCORE, CallMode.INTERVIEWER):
        template = load_template(mode)
        assert template.version == 1
        assert template.mode is mode


@pytest.mark.parametrize("name", ["score-topic", "score-mock", "interviewer"])
def test_rendered_prompt_matches_its_golden_snapshot(name: str) -> None:
    rendered = _prompts()[name]
    path = GOLDEN / f"{name}.txt"

    if os.environ.get("DOGMA_UPDATE_GOLDEN") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    assert path.is_file(), f"missing golden {path}; run with DOGMA_UPDATE_GOLDEN=1"
    assert rendered == path.read_text(encoding="utf-8")
