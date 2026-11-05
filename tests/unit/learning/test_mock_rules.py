"""Mock rules are pure: part totals, bands, weakest and strongest parts, and reveal matching."""

from __future__ import annotations

from pathlib import Path

import pytest

from dogma.content.rubrics import Rubric, load_rubric_version
from dogma.content.schemas import Reveal
from dogma.learning.mock_rules import (
    band_for,
    part_scores,
    reveal_for,
    strongest_part,
    weakest_over,
    weakest_part,
)

CURRICULUM = Path(__file__).resolve().parents[3] / "curriculum"


@pytest.fixture
def rubric() -> Rubric:
    return load_rubric_version(CURRICULUM, "mock", 1)


def _all(rubric: Rubric, score: int) -> dict[str, int]:
    return dict.fromkeys(rubric.criteria_ids, score)


def test_a_uniform_score_gives_the_same_percentage_in_every_part(rubric: Rubric) -> None:
    parts = part_scores(rubric, _all(rubric, 3))

    assert [part.part_id for part in parts] == [part.id for part in rubric.parts]
    assert {round(part.percent, 4) for part in parts} == {0.75}
    assert sum(part.score for part in parts) == 90


def test_the_weakest_part_is_the_lowest_percentage(rubric: Rubric) -> None:
    scores = _all(rubric, 4)
    for item in rubric.parts[3].items:  # structure: all 1s
        scores[item.id] = 1
    for item in rubric.parts[1].items:  # reframe: all 2s, weak but not the weakest
        scores[item.id] = 2

    weakest = weakest_part(part_scores(rubric, scores))

    assert weakest.part_id == rubric.parts[3].id


def test_ties_go_to_the_part_earlier_in_the_rubric(rubric: Rubric) -> None:
    parts = part_scores(rubric, _all(rubric, 2))

    assert weakest_part(parts).part_id == rubric.parts[0].id
    assert strongest_part(parts).part_id == rubric.parts[0].id


def test_the_weakest_part_over_several_mocks_uses_the_summed_part_totals(rubric: Rubric) -> None:
    strong = _all(rubric, 4)
    weak_ask = {**_all(rubric, 4), **{item.id: 1 for item in rubric.parts[2].items}}

    result = weakest_over(rubric, [strong, weak_ask, weak_ask])

    assert result is not None
    assert result.part_id == rubric.parts[2].id
    assert weakest_over(rubric, []) is None


def test_the_band_is_the_one_whose_range_holds_the_total(rubric: Rubric) -> None:
    for band in rubric.bands:
        assert band_for(rubric, band.low).label == band.label
        assert band_for(rubric, band.high).label == band.label
    assert band_for(rubric, rubric.maximum).label == rubric.bands[-1].label


def test_a_total_outside_the_bands_is_rejected(rubric: Rubric) -> None:
    with pytest.raises(ValueError):
        band_for(rubric, rubric.maximum + 1)


REVEALS = (
    Reveal(when_asked="What does the current retention process do?", answer="Calls get 20%."),
    Reveal(when_asked="How do the segments differ?", answer="Prepaid is 60%."),
    Reveal(when_asked="What are the competitors doing?", answer="Two rivals cut prices."),
)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Tell me about the retention process", REVEALS[0].answer),
        ("how do the segments differ by plan?", REVEALS[1].answer),
        ("What are competitors doing about price", REVEALS[2].answer),
    ],
)
def test_a_question_releases_the_reveal_whose_keywords_it_shares(
    question: str, expected: str
) -> None:
    reveal = reveal_for(question, REVEALS)

    assert reveal is not None
    assert reveal.answer == expected


@pytest.mark.parametrize("question", ["", "What is the weather?", "segments"])
def test_an_unrelated_or_too_thin_question_releases_nothing(question: str) -> None:
    assert reveal_for(question, REVEALS) is None
